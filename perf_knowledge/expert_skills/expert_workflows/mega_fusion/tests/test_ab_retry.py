"""CPU-only tests for the portable retrying interleaved A/B runner.

The suite checks retry/drop visibility, path markers, arm rotation, fused-kernel
attribution, strict post-result teardown handling, timeout decoding, and
environment-supplied runtime roots. The real benchmark is replaced by
``--fake-cmd``; no torch or GPU is required.
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

import pytest

TOOL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "tools", "ab_retry.py")


def _load():
    spec = importlib.util.spec_from_file_location("ab_retry", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rec(arm, guard, e2e, s1, s2):
    return {"arm": arm, "guard": guard, "e2e_max_ms": e2e, "stage1_max_ms": s1,
            "stage2_combine_max_ms": s2, "residual_max_ms": round(e2e - s1 - s2, 4)}


# --------------------------------------------------------------------------- kernel_time

def test_e2e_win_on_slower_kernels_is_visible():
    """End-to-end can improve while separately timed kernel sums get worse."""
    m = _load()
    g = [_rec("base", "8192_uniform", 4.5, 3.0, 1.0),
         _rec("cand", "8192_uniform", 4.3, 3.1, 1.0),
         _rec("base", "8192_uniform", 4.5, 3.0, 1.0),
         _rec("cand", "8192_uniform", 4.3, 3.1, 1.0)]
    kt = m.kernel_time_block(g, "base", "cand")

    assert m.pct_pairs(g, "base", "cand")[0] > 0, "e2e says the candidate won"
    assert kt["kernel_sum_pct"] < 0, "and the kernel sum says it did not -- both must be reported"
    assert kt["base"]["kernel_sum_ms"] == 4.0
    assert kt["cand"]["kernel_sum_ms"] == 4.1
    # the win is in the gaps, and the residual is where that becomes checkable
    assert kt["base"]["residual_ms"] == 0.5
    assert kt["cand"]["residual_ms"] == 0.2


def test_sign_convention_is_not_inverted():
    """Positive kernel_sum_pct must mean the candidate's kernels are FASTER."""
    m = _load()
    g = [_rec("base", "g", 4.5, 3.0, 1.0), _rec("cand", "g", 4.5, 2.5, 1.0)]
    assert m.kernel_time_block(g, "base", "cand")["kernel_sum_pct"] > 0


def test_half_filled_record_set_does_not_crash():
    """The doc is rewritten after EVERY leg, so this runs before the candidate has one.

    Caught by the self-test on the merge: the first version divided into a None and
    took the whole run down after leg 1, destroying the incremental-write guarantee
    that is the reason the aggregate is rewritten each time.
    """
    m = _load()
    kt = m.kernel_time_block([_rec("base", "g", 4.5, 3.0, 1.0)], "base", "cand")
    assert kt["kernel_sum_pct"] is None
    assert kt["cand"]["n"] == 0


# --------------------------------------------------------------------------- end to end

FAKE = r"""#!/bin/bash
if [[ -n "$FAKE_TIMEOUT" ]]; then sleep 2; fi
if [[ "$FAKE_TAG" == *_cand ]]; then S1=3.1; S2=1.0; E=4.3; else S1=3.0; S2=1.0; E=4.5; fi
if [[ -n "$FAKE_FAIL_ARM" && "$FAKE_TAG" == *_$FAKE_FAIL_ARM ]]; then
  echo "hip failed with out of memory" >&2; exit 255
fi
for r in 0 1; do echo "[default$r]:[megamoe] path=MEGA"; done
echo "[RESULT] tokens=8192 mega_e2e=${E}/${E}ms stage1=${S1}/${S1}ms stage2_combine=${S2}/${S2}ms"
if [[ -n "$FAKE_MORI_TEARDOWN" ]]; then
  if [[ -n "$FAKE_OTHER_FATAL" ]]; then echo "hip failed with out of memory" >&2; fi
  if [[ -n "$FAKE_OTHER_EXCEPTION" ]]; then echo "ImportError: broken candidate import" >&2; fi
  echo "python3: mori/include/mori/shmem/internal.hpp:121: void mori::shmem::ShmemStates::CheckStatusValid(): Assertion \`false' failed." >&2
  exit 1
fi
if [[ -n "$FAKE_GENERIC_POST_RESULT_FAIL" ]]; then exit 1; fi
"""

PLAN = {"guards": ["8192_uniform"], "blocks": 2, "iters": 3, "base_arm": "base",
        "arms": [{"name": "base", "tree": "/tmp"}, {"name": "cand", "tree": "/tmp"}]}


def _run(tmp, env_extra=None, attempts=1, timeout=1200):
    fake = os.path.join(tmp, "fake.sh")
    open(fake, "w").write(FAKE)
    plan = os.path.join(tmp, "plan.json")
    json.dump(PLAN, open(plan, "w"))
    out = os.path.join(tmp, "out.json")
    env = dict(os.environ)
    env.update(env_extra or {})
    p = subprocess.run([sys.executable, TOOL, "--plan", plan, "--out", out,
                        "--fake-cmd", f"bash {fake}", "--expect-markers", "2",
                        "--expect-path", "MEGA",
                        "--attempts", str(attempts), "--retry-sleep", "0",
                        "--timeout", str(timeout)],
                       capture_output=True, text=True, env=env, timeout=180)
    return p, json.load(open(out))


def test_full_run_completes_and_reports_both_metrics():
    with tempfile.TemporaryDirectory() as tmp:
        p, doc = _run(tmp)
        assert p.returncode == 0, p.stdout + p.stderr
        assert doc["claim_complete"] is True
        assert doc["n_dropped"] == 0
        cmp_ = doc["by_guard"]["8192_uniform"]["base_vs_cand"]
        assert cmp_["n"] == 2
        assert min(cmp_["e2e_pairs"]) > 0
        assert cmp_["kernel_time"]["kernel_sum_pct"] < 0


def test_a_leg_that_never_succeeds_is_dropped_loudly_not_omitted():
    with tempfile.TemporaryDirectory() as tmp:
        p, doc = _run(tmp, {"FAKE_FAIL_ARM": "cand"}, attempts=2)
        assert doc["n_dropped"] == 2, "both candidate legs failed and both must be recorded"
        assert all(d["arm"] == "cand" for d in doc["dropped_legs"])
        assert doc["dropped_legs"][0]["reason"].startswith("rc=")
        assert len(doc["dropped_legs"][0]["attempts"]) == 2, "it was retried before being dropped"
        assert "DROPPED" in p.stdout, "and the drop is printed, not only filed"
        # the surviving arm must not silently become a one-sided claim
        assert doc["by_guard"]["8192_uniform"]["base_vs_cand"]["e2e_pairs"] == []


def test_marker_count_violation_voids_the_leg():
    """An incomplete marker set is not a measurement."""
    with tempfile.TemporaryDirectory() as tmp:
        fake = os.path.join(tmp, "fake.sh")
        open(fake, "w").write(FAKE)
        plan = os.path.join(tmp, "plan.json")
        json.dump(PLAN, open(plan, "w"))
        out = os.path.join(tmp, "out.json")
        subprocess.run([sys.executable, TOOL, "--plan", plan, "--out", out,
                        "--fake-cmd", f"bash {fake}", "--expect-markers", "8",
                        "--attempts", "1", "--retry-sleep", "0"],
                       capture_output=True, text=True, timeout=180)
        doc = json.load(open(out))
        assert doc["n_dropped"] == 4
        assert "marker count 2 != 8" in doc["dropped_legs"][0]["reason"]


def test_marker_value_violation_voids_the_leg():
    """Eight plausible markers are still void when the intended arm fell back."""
    with tempfile.TemporaryDirectory() as tmp:
        fake = os.path.join(tmp, "fake.sh")
        open(fake, "w").write(FAKE)
        plan_data = json.loads(json.dumps(PLAN))
        for arm in plan_data["arms"]:
            arm["expect_paths"] = ["SCATTERED"]
        plan = os.path.join(tmp, "plan.json")
        json.dump(plan_data, open(plan, "w"))
        out = os.path.join(tmp, "out.json")
        subprocess.run([sys.executable, TOOL, "--plan", plan, "--out", out,
                        "--fake-cmd", f"bash {fake}", "--expect-markers", "2",
                        "--attempts", "1", "--retry-sleep", "0"],
                       capture_output=True, text=True, timeout=180)
        doc = json.load(open(out))
        assert doc["n_dropped"] == 4
        assert "path markers ['MEGA'] != expected ['SCATTERED']" in \
            doc["dropped_legs"][0]["reason"]


def test_exact_mori_teardown_after_complete_result_is_accepted_and_recorded():
    with tempfile.TemporaryDirectory() as tmp:
        p, doc = _run(tmp, {"FAKE_MORI_TEARDOWN": "1"})
        assert p.returncode == 0, p.stdout + p.stderr
        assert doc["n_dropped"] == 0
        assert doc["records"]
        assert all(r["accepted_post_result_teardown"] for r in doc["records"])
        assert all(r["rc"] == 1 for r in doc["records"])


def test_generic_nonzero_after_result_remains_void():
    with tempfile.TemporaryDirectory() as tmp:
        _p, doc = _run(tmp, {"FAKE_GENERIC_POST_RESULT_FAIL": "1"})
        assert doc["n_dropped"] == 4
        assert all(d["reason"] == "rc=1" for d in doc["dropped_legs"])


def test_mori_teardown_does_not_mask_an_unrelated_fatal_error():
    with tempfile.TemporaryDirectory() as tmp:
        _p, doc = _run(tmp, {
            "FAKE_MORI_TEARDOWN": "1",
            "FAKE_OTHER_FATAL": "1",
        })
        assert doc["n_dropped"] == 4
        assert all(d["reason"] == "rc=1" for d in doc["dropped_legs"])


def test_mori_teardown_does_not_mask_an_unlisted_python_exception():
    with tempfile.TemporaryDirectory() as tmp:
        _p, doc = _run(tmp, {
            "FAKE_MORI_TEARDOWN": "1",
            "FAKE_OTHER_EXCEPTION": "1",
        })
        assert doc["n_dropped"] == 4
        assert all(d["reason"] == "rc=1" for d in doc["dropped_legs"])


def test_timeout_bytes_are_decoded_and_retried_as_void():
    with tempfile.TemporaryDirectory() as tmp:
        p, doc = _run(tmp, {"FAKE_TIMEOUT": "1"}, timeout=1)
        assert p.returncode == 0, p.stdout + p.stderr
        assert doc["n_dropped"] == 4
        assert all(d["reason"] == "rc=124" for d in doc["dropped_legs"])


def test_rotation_changes_arm_position_between_blocks():
    """A fixed arm-within-block position manufactured ~1pp of separation on a null."""
    m = _load()
    seq = m.rotate_sequence(["g"], ["base", "cand"], 2)
    assert [a for _, a in seq] == ["base", "cand", "cand", "base"]


def test_runtime_roots_come_from_environment_not_repository_paths():
    src = open(TOOL).read()
    assert 'os.environ.get("AB_RETRY_MORI_ROOT")' in src
    assert 'os.environ.get("MORI_ROOT"' in src
    assert 'os.environ.get("AB_RETRY_JIT_DIR")' in src
    assert 'os.environ.get("AITER_JIT_DIR")' in src
    assert 'AB_RETRY_MORI_ROOT", "/"' not in src
    assert 'AB_RETRY_JIT_DIR", "/"' not in src


def test_real_run_requires_external_jit_cache(monkeypatch, tmp_path):
    m = _load()
    monkeypatch.delenv("AB_RETRY_JIT_DIR", raising=False)
    monkeypatch.delenv("AITER_JIT_DIR", raising=False)
    rc, out, _wall = m._attempt(
        str(tmp_path), {}, "8192_uniform", 1, "", "x", 1, "", 1,
    )
    assert rc == 2
    assert "set AB_RETRY_JIT_DIR or AITER_JIT_DIR" in out


def test_real_run_rejects_jit_cache_inside_candidate(monkeypatch, tmp_path):
    m = _load()
    jit = tmp_path / "jit"
    jit.mkdir()
    monkeypatch.setenv("AB_RETRY_JIT_DIR", str(jit))
    rc, out, _wall = m._attempt(
        str(tmp_path), {}, "8192_uniform", 1, "", "x", 1, "", 1,
    )
    assert rc == 2
    assert "outside the candidate tree" in out


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# --------------------------------------------------------------------------- guard ids

def test_guard_ids_carry_an_optional_mtpr():
    m = _load()
    assert m.parse_guard("8192_uniform") == ("8192", "uniform", None)
    assert m.parse_guard("128_uniform") == ("128", "uniform", None)
    assert m.parse_guard("128_uniform_mtpr128") == ("128", "uniform", "128")
    assert m.parse_guard("512_rank-mixed-skew_mtpr512") == ("512", "rank-mixed-skew", "512")
    with pytest.raises(ValueError):
        m.parse_guard("uniform")


def test_mtpr_guard_leg_is_void_unless_the_bench_ran_that_mtpr(tmp_path):
    script = tmp_path / "bench.sh"
    script.write_text(
        'echo "[RESULT] route=$FAKE_ROUTE tokens=$FAKE_TOKENS mtpr=$REPORTED '
        'mega_e2e=1.0/1.1ms stage1=0.1/0.2ms stage2_combine=0.3/0.4ms"\n')
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"arms": [{"name": "a", "tree": str(tmp_path)}],
                                "guards": ["128_uniform_mtpr128"], "blocks": 1}))
    docs = {}
    for reported in ("128", "8192"):
        out = tmp_path / f"out_{reported}.json"
        subprocess.run([sys.executable, TOOL, "--plan", str(plan), "--out", str(out),
                        "--fake-cmd", f"bash {script}", "--attempts", "1",
                        "--retry-sleep", "0"],
                       env={**os.environ, "REPORTED": reported}, check=True,
                       capture_output=True)
        docs[reported] = json.loads(out.read_text())
    assert len(docs["128"]["records"]) == 1
    assert not docs["8192"]["records"]
    assert "mtpr=128" in docs["8192"]["dropped_legs"][0]["attempts"][0]["void"]


def test_fallback_guard_records_markers_without_requiring_them(tmp_path):
    script = tmp_path / "bench.sh"
    script.write_text(
        'echo "[RESULT] route=$FAKE_ROUTE tokens=$FAKE_TOKENS mtpr=$FAKE_MTPR '
        'mega_e2e=1.0/1.1ms stage1=0.1/0.2ms stage2_combine=0.3/0.4ms"\n')
    arm = {"name": "cand", "tree": str(tmp_path), "expect_markers": 8,
           "expect_paths": ["MEGA"], "fallback_guards": ["128_uniform_mtpr128"]}
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"arms": [arm], "blocks": 1,
                                "guards": ["128_uniform_mtpr128", "512_uniform_mtpr512"]}))
    out = tmp_path / "out.json"
    subprocess.run([sys.executable, TOOL, "--plan", str(plan), "--out", str(out),
                    "--fake-cmd", f"bash {script}", "--attempts", "1", "--retry-sleep", "0"],
                   check=True, capture_output=True)
    doc = json.loads(out.read_text())
    assert [r["guard"] for r in doc["records"]] == ["128_uniform_mtpr128"]
    assert [d["guard"] for d in doc["dropped_legs"]] == ["512_uniform_mtpr512"]
