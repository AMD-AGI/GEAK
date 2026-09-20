#!/usr/bin/env python3
"""The e2e KB CLI's read/write surface, beyond the retraction round-trips in test_kb_retract.

Retraction is covered next door; this file drives the paths that one does not touch — the reference
renderer and artifact materializer a `resolve` emits, the kernel/head merge and artifact plumbing a
`write` performs, the two commands' failure modes, and the `identity` echo. Everything runs
`e2e_store.main([...])` in-process (not over a subprocess) so the assertions AND the coverage both
land on the module under test.
"""

import json
import os
import sys

import pytest

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPTS)))              # repo root, for `kb`
sys.path.insert(0, _SCRIPTS)
import e2e_store                                                            # noqa: E402

IDENTITY = ["--model", "M", "--gfx", "gfx950", "--framework", "vllm",
            "--framework-version", "0.26.0", "--precision", "fp8",
            "--tp", "8", "--isl", "1024", "--osl", "1024", "--conc", "64"]


def _run(command, *args, identity=IDENTITY):
    import io
    import contextlib
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        e2e_store.main([command] + identity + list(args))
    return json.loads(buffer.getvalue())


def _result(path, tput=1000.0, baseline=800.0, **extra):
    doc = {"final_throughput_tok_s": tput, "baseline_throughput_tok_s": baseline,
           "validation_status": "validated_win", "output_parity": "pass",
           "accepted_config": {"env": "A=1"}, "accepted_kernels": []}
    doc.update(extra)
    path.write_text(json.dumps(doc))
    return str(path)


def _write(tmp_path, name, direction, *args, tput=1000.0, **extra):
    result = _result(tmp_path / (name + ".json"), tput=tput, **extra)
    return _run("write", "--store", str(tmp_path / "store"), "--result", result,
                "--direction", direction, "--apply", *args)


# -- identity -----------------------------------------------------------------------------------


def test_identity_prints_the_full_ladder():
    out = _run("identity")
    assert out["identity"]["model"] == "m"                      # identity segments are normalized
    # Every rung the deployment reads/writes, most specific first, each with the metric it ranks on.
    assert out["ladder"][0]["tier"] == "exact"
    assert out["ladder"][0]["ranked_by"] == "throughput_tok_s"
    assert any(r["ranked_by"] == "speedup" for r in out["ladder"])


# -- resolve: reference prose + artifact bundles ------------------------------------------------


def test_resolve_writes_a_prose_reference(tmp_path):
    _write(tmp_path, "a", "tuned")
    out = _run("resolve", "--store", str(tmp_path / "store"),
               "--refs-dir", str(tmp_path / "refs"))
    assert out["read_reason"] == "read"
    refs = list((tmp_path / "refs").glob("e2e_reference_*.md"))
    assert len(refs) == 1
    text = refs[0].read_text()
    assert "# e2e warm start" in text and "throughput" in text and "config:" in text


def test_reference_spells_out_each_accepted_kernel(tmp_path):
    """The prose kernel line is what the Director reads before opening a patch — it must name the
    language, the isolated speedup and the stored path."""
    (tmp_path / "moe.patch").write_text("--- a\n+++ b\n")
    _write(tmp_path, "a", "kernelized",
           accepted_kernels=[{"name": "moe_stage1", "language": "triton",
                              "isolated_speedup": 1.84, "patch": str(tmp_path / "moe.patch")}])
    _run("resolve", "--store", str(tmp_path / "store"), "--refs-dir", str(tmp_path / "refs"))
    text = list((tmp_path / "refs").glob("*.md"))[0].read_text()
    assert "moe_stage1" in text and "triton" in text and "1.84x" in text
    assert "kernels/moe_stage1.patch" in text


def test_resolve_materializes_artifact_bundles(tmp_path):
    (tmp_path / "final.patch").write_text("--- a\n+++ b\n")
    _write(tmp_path, "a", "tuned", final_patch=str(tmp_path / "final.patch"))
    out = _run("resolve", "--store", str(tmp_path / "store"),
               "--cache-dir", str(tmp_path / "cache"))
    bundle = out["candidates"][0]["bundle"]
    assert "final.patch" in bundle["files"]
    assert os.path.isdir(bundle["path"])


def test_a_coarser_rung_match_is_flagged_non_comparable(tmp_path):
    """A record written at conc=64 also lands on the workload-agnostic rung; a resolve at conc=128
    misses the exact rung and matches there instead, and the prose must warn the numbers are not
    this deployment's."""
    _write(tmp_path, "a", "tuned")
    other_conc = [x if x != "64" else "128" for x in IDENTITY]
    out = _run("resolve", "--store", str(tmp_path / "store"),
               "--refs-dir", str(tmp_path / "refs"), identity=other_conc)
    if out["read_reason"] == "read" and out["match_tier"] != "exact":
        text = list((tmp_path / "refs").glob("*.md"))[0].read_text()
        assert "DIFFERENT workload" in text


def test_min_speedup_floors_the_offer(tmp_path):
    # baseline 800, tput 1000 => speedup 1.25; a 1.5 floor curates it away.
    _write(tmp_path, "a", "tuned", tput=1000.0)
    out = _run("resolve", "--store", str(tmp_path / "store"), "--min-speedup", "1.5")
    assert out["candidates"] == [] and out["read_reason"] == "e2e_page_not_found"


def test_resolve_on_a_missing_store_is_a_miss_not_a_crash(tmp_path):
    out = _run("resolve", "--store", str(tmp_path / "nope"))
    assert out["candidates"] == [] and out["read_reason"].startswith("no_such_store")


def test_a_read_that_raises_is_reported_not_propagated(tmp_path, monkeypatch):
    _write(tmp_path, "a", "tuned")

    def boom(self, *a, **k):
        raise RuntimeError("disk gone")

    monkeypatch.setattr("kb.store_local.LocalKBStore.candidates", boom)
    out = _run("resolve", "--store", str(tmp_path / "store"))
    assert out["read_reason"].startswith("read_failed")


# -- write: kernel/head merge, artifacts, state, failure modes ----------------------------------


def test_write_merges_accepted_kernels_and_heads(tmp_path):
    """Both tracks are read and merged on a name collision — the head entry fills fields the
    milestone entry left blank, and neither record is dropped."""
    (tmp_path / "k.patch").write_text("x")
    out = _write(
        tmp_path, "a", "both-tracks",
        accepted_kernels=[{"name": "op1", "language": "triton", "session_id": "kern-sess",
                           "patch": str(tmp_path / "k.patch"), "isolated_speedup": 1.5},
                          {"language": "triton"}],                # no name: dropped, not recorded
        accepted_heads=[{"name": "op1", "target_callable": "fwd", "language": ""},  # "" won't clobber
                        {"name": "op2", "language": "triton"}])
    assert out["applied"] is True and out["rungs"][0]["written"] is True
    view = _run("resolve", "--store", str(tmp_path / "store"))["candidates"][0]
    kernels = {k["name"]: k for k in view["accepted_kernels"]}
    assert set(kernels) == {"op1", "op2"}
    assert kernels["op1"]["target_callable"] == "fwd"          # merged in from the head track
    assert kernels["op1"]["patch"] == "kernels/op1.patch"      # stored name, from the milestone
    assert kernels["op1"].get("kernel_canonical_id")           # addressed: language known, real op
    assert kernels["op1"]["kernel_session_id"] == "kern-sess"


def test_an_env_win_is_not_addressed_as_a_kernel(tmp_path):
    """A routed env/flag win is not a rewrite — minting a kernel id for it fabricates a dead
    reference on a store with no delete."""
    _write(tmp_path, "a", "routed",
           accepted_kernels=[{"name": "moe", "language": "aiter", "winner_kind": "env"}])
    view = _run("resolve", "--store", str(tmp_path / "store"))["candidates"][0]
    assert view["accepted_kernels"][0].get("kernel_canonical_id") in (None, "")


def test_extra_files_are_attached(tmp_path):
    (tmp_path / "notes.txt").write_text("hi")
    out = _write(tmp_path, "a", "tuned", "--file", str(tmp_path / "notes.txt"))
    assert "notes.txt" in out["files"]


def test_a_validated_override_is_honored(tmp_path):
    # status/parity would say unvalidated; --validated true overrides to active.
    _write(tmp_path, "a", "tuned", "--validated", "true",
           validation_status="", output_parity="fail")
    view = _run("resolve", "--store", str(tmp_path / "store"))["candidates"][0]
    assert view["validated"] is True and view["lifecycle"] == "active"


def test_a_dry_run_records_nothing(tmp_path):
    result = _result(tmp_path / "r.json")
    out = _run("write", "--store", str(tmp_path / "store"), "--result", result,
               "--direction", "d")                              # no --apply
    assert out["applied"] is False
    assert all(not r["written"] for r in out["rungs"])
    assert not (tmp_path / "store").exists()


def test_write_refusing_a_run_with_no_measurement(tmp_path):
    (tmp_path / "r.json").write_text(json.dumps({"baseline_throughput_tok_s": 800.0}))
    with pytest.raises(SystemExit):
        _run("write", "--store", str(tmp_path / "store"), "--result",
             str(tmp_path / "r.json"), "--direction", "d", "--apply")


def test_write_with_an_unreadable_result(tmp_path):
    with pytest.raises(SystemExit):
        _run("write", "--store", str(tmp_path / "store"),
             "--result", str(tmp_path / "missing.json"), "--direction", "d", "--apply")


def test_write_rejects_a_non_object_result(tmp_path):
    (tmp_path / "r.json").write_text("[]")
    with pytest.raises(SystemExit):
        _run("write", "--store", str(tmp_path / "store"),
             "--result", str(tmp_path / "r.json"), "--direction", "d", "--apply")


def test_a_store_that_cannot_be_opened_stops_the_ladder(tmp_path):
    """--store points at a regular file: makedirs fails, the rung cannot open, and the write stops
    before publishing a partial ladder."""
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file, not a dir")
    result = _result(tmp_path / "r.json")
    out = _run("write", "--store", str(blocker), "--result", result,
               "--direction", "d", "--apply")
    assert out["ok"] is False
    assert out["rungs"][0]["written"] is False and out["rungs"][0]["error"]
    assert len(out["rungs"]) == 1                               # broke, did not try coarser rungs


def test_a_mid_ladder_write_failure_stops_and_mirror_is_best_effort(tmp_path, monkeypatch):
    """publish erroring on the exact rung stops the ladder; when a mirror is present its own failure
    is reported, never raised."""
    real = e2e_store.publish
    calls = {"n": 0}

    def flaky(store, recs, files, score_of, promote=True):
        calls["n"] += 1
        return [], [], "boom"                                   # every publish fails

    # Force a two-plane open so the mirror branch is exercised, then fail the write.
    local_dir = str(tmp_path / "store")
    os.makedirs(local_dir, exist_ok=True)
    from kb.store_local import LocalKBStore
    prim = LocalKBStore(local_dir, metric="throughput_tok_s", promote_floor=0.0)
    mirr = LocalKBStore(str(tmp_path / "mirror"), metric="throughput_tok_s", promote_floor=0.0)
    monkeypatch.setattr(e2e_store, "open_plane", lambda a, m, f, create=False: (prim, mirr, ""))
    monkeypatch.setattr(e2e_store, "publish", flaky)
    result = _result(tmp_path / "r.json")
    out = _run("write", "--store", local_dir, "--result", result,
               "--direction", "d", "--apply", "--plane", "both")
    assert out["ok"] is False and out["rungs"][0]["error"] == "boom"
    assert len(out["rungs"]) == 1                               # stopped after the exact rung
    assert calls["n"] >= 2                                      # primary AND mirror both attempted
    assert e2e_store.publish is flaky and real is not flaky     # sanity on the monkeypatch


def test_a_materialize_failure_degrades_the_offer(tmp_path, monkeypatch):
    """A download problem reports an error on the bundle rather than dropping the candidate — the
    config lives in the knowledge doc, so the offer is still usable."""
    (tmp_path / "final.patch").write_text("x")
    _write(tmp_path, "a", "tuned", final_patch=str(tmp_path / "final.patch"))
    from kb.store_local import KBStoreError

    def boom(self, *a, **k):
        raise KBStoreError("cache full")

    monkeypatch.setattr("kb.store_local.LocalKBStore.materialize", boom)
    out = _run("resolve", "--store", str(tmp_path / "store"),
               "--cache-dir", str(tmp_path / "cache"))
    assert out["candidates"][0]["bundle"]["error"]


def test_an_unwritable_refs_dir_lets_the_read_stand(tmp_path):
    """The prose page is a mirror of the offer, not the offer itself: if refs-dir cannot be made,
    resolve still returns the candidate."""
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file")
    _write(tmp_path, "a", "tuned")
    out = _run("resolve", "--store", str(tmp_path / "store"),
               "--refs-dir", str(blocker / "under-a-file"))
    assert out["read_reason"] == "read" and out["candidates"]


def test_a_mirror_only_failure_is_reported_without_gating_the_primary(tmp_path, monkeypatch):
    """The mirror never gates the primary: the exact rung writes locally and its mirror failure is
    surfaced on that rung, but the ladder keeps going."""
    from kb.store_local import LocalKBStore
    prim = LocalKBStore(str(tmp_path / "store"), metric="throughput_tok_s", promote_floor=0.0)

    class FailingMirror:
        def write(self, *a, **k):
            raise RuntimeError("mirror down")

    seen = {"n": 0}

    def one_bad_mirror(a, metric, floor, create=False):
        seen["n"] += 1
        # A mirror only on the first (exact) rung, so the ladder still advances past it.
        return (prim, FailingMirror(), "") if seen["n"] == 1 else (prim, None, "")

    monkeypatch.setattr(e2e_store, "open_plane", one_bad_mirror)
    result = _result(tmp_path / "r.json")
    out = _run("write", "--store", str(tmp_path / "store"), "--result", result,
               "--direction", "d", "--apply", "--plane", "both")
    assert out["rungs"][0]["written"] is True                   # primary landed
    assert "mirror down" in out["rungs"][0]["error"]            # mirror failure surfaced
    assert len(out["rungs"]) > 1                                # ladder was NOT stopped by it


# -- retract: recompute-from-result and missing-store paths -------------------------------------


def test_retract_recomputes_the_session_from_the_result(tmp_path):
    """Given the same --result and --direction, retract recomputes the exact session id the write
    minted, without the write's output having been kept."""
    result = _result(tmp_path / "r.json")
    written = _run("write", "--store", str(tmp_path / "store"), "--result", result,
                   "--direction", "d", "--apply")["session_id"]
    out = _run("retract", "--store", str(tmp_path / "store"), "--result", result,
               "--direction", "d", "--reason", "wrong", "--apply")
    assert out["session_id"] == written and out["ok"] is True


def test_retract_needs_a_session_or_a_result(tmp_path):
    with pytest.raises(SystemExit):
        _run("retract", "--store", str(tmp_path / "store"), "--reason", "wrong")


def test_retract_with_an_unreadable_result(tmp_path):
    with pytest.raises(SystemExit):
        _run("retract", "--store", str(tmp_path / "store"), "--result",
             str(tmp_path / "missing.json"), "--direction", "d", "--reason", "wrong")


def test_retract_on_a_missing_store_reports_not_found(tmp_path):
    out = _run("retract", "--store", str(tmp_path / "nope"),
               "--session-id", "whatever", "--reason", "wrong", "--apply")
    assert out["ok"] is False
    assert all(r["found"] is False for r in out["rungs"])


# -- resolve: the offer is ordered by absolute throughput on EVERY rung -------------------------


def _coarse_id(out):
    """The tp_any rung's canonical id — the coarse page a reader on another workload lands on."""
    return [r["canonical_id"] for r in out["ladder"] if r["tier"] == "tp_any"][0]


def test_a_coarse_rung_is_offered_by_throughput_not_by_its_champion_metric(tmp_path):
    """The behaviour the whole read path was changed for. A coarse rung still CROWNS on speedup —
    that is what makes its champion comparable across workload points — but the OFFER is ordered by
    absolute throughput, because a reader asking "what has run fast on this deployment" is not
    asking "what improved most over whatever baseline it happened to have"."""
    store = str(tmp_path / "store")
    # 1.9x off a slow baseline vs 1.05x off a fast one: opposite orders under the two metrics.
    _write(tmp_path, "slowbase", "big-ratio", tput=900.0, baseline=474.0)
    _write(tmp_path, "fastbase", "small-ratio", tput=1600.0, baseline=1524.0)
    out = _run("resolve", "--store", store, "--tp", "16")        # a workload only the coarse rungs hold
    assert out["match_tier"] != "exact"
    assert out["sorted_by"] == "throughput_tok_s"
    assert out["champion_metric"] == "speedup"                   # the rung's own metric, unchanged
    assert [c["direction"] for c in out["candidates"]] == ["small-ratio", "big-ratio"]


def test_sort_by_speedup_restores_the_old_order(tmp_path):
    _write(tmp_path, "slowbase", "big-ratio", tput=900.0, baseline=474.0)
    _write(tmp_path, "fastbase", "small-ratio", tput=1600.0, baseline=1524.0)
    out = _run("resolve", "--store", str(tmp_path / "store"), "--tp", "16",
               "--sort-by", "speedup")
    assert out["sorted_by"] == "speedup"
    assert [c["direction"] for c in out["candidates"]] == ["big-ratio", "small-ratio"]


def test_the_prose_reference_names_the_metric_it_ordered_by(tmp_path):
    _write(tmp_path, "a", "tuned")
    _run("resolve", "--store", str(tmp_path / "store"), "--refs-dir", str(tmp_path / "refs"))
    text = list((tmp_path / "refs").glob("e2e_reference_*.md"))[0].read_text()
    assert "ordered by `throughput_tok_s` (highest first)" in text


# -- attest: counting what happened when a record was actually run ------------------------------


def test_attest_counts_on_every_rung_without_moving_anything(tmp_path):
    """All three rungs share one session id, so a count that lands only on the exact rung leaves
    the two coarse pages — the ones a reader on another workload reads — quoting a stale ledger."""
    store = str(tmp_path / "store")
    sid = _write(tmp_path, "a", "tuned")["session_id"]
    before = _run("resolve", "--store", store)["candidates"][0]
    out = _run("attest", "--store", store, "--session-id", sid, "--outcome", "validated",
               "--measured-tok-s", "1100", "--baseline-tok-s", "1000", "--parity", "pass",
               "--measured-by", "boxA", "--apply")
    assert out["ok"] is True
    assert len(out["rungs"]) == 3 and all(r["rewritten"] for r in out["rungs"])
    assert all(r["attestations"]["validations"] == 1 for r in out["rungs"])
    after = _run("resolve", "--store", store)["candidates"][0]
    assert after["validations"] == 1 and after["recalls"] == 1
    assert after["throughput_tok_s"] == before["throughput_tok_s"]     # no scalar moved
    assert after["speedup"] == before["speedup"]
    assert after["is_champion"] == before["is_champion"]                # no champion re-pointed
    # and the evidence rode along, including the delta this command derives rather than trusting
    assert out["rungs"][0]["attestations"]["history"][-1]["delta_pct"] == 10.0


def test_repeated_failures_raise_a_retire_hint_but_retire_nothing(tmp_path):
    store = str(tmp_path / "store")
    sid = _write(tmp_path, "a", "tuned")["session_id"]
    for _ in range(2):
        _run("attest", "--store", store, "--session-id", sid,
             "--outcome", "not_reproduced", "--apply")
    view = _run("resolve", "--store", store)["candidates"][0]
    assert view["not_reproduced"] == 2 and "came back negative" in view["retire_hint"]
    assert view["lifecycle"] == "active"                  # advisory only; still offered, still ranked
    assert "**" in list_reference_text(tmp_path, store)   # and the prose says so in bold


def list_reference_text(tmp_path, store):
    _run("resolve", "--store", store, "--refs-dir", str(tmp_path / "refs2"))
    return list((tmp_path / "refs2").glob("e2e_reference_*.md"))[0].read_text()


def test_re_writing_the_same_config_keeps_its_attestations(tmp_path):
    """The bug this carry-forward exists for: _content_digest excludes the measurement, so a
    re-bench of one config replaces the SAME session — and a naive write would silently reset the
    record's whole validation history while looking perfectly well-formed."""
    store = str(tmp_path / "store")
    sid = _write(tmp_path, "a", "tuned", tput=1000.0)["session_id"]
    _run("attest", "--store", store, "--session-id", sid, "--outcome", "validated", "--apply")
    again = _write(tmp_path, "a", "tuned", tput=1200.0)          # same config, new measurement
    assert again["session_id"] == sid
    view = _run("resolve", "--store", store)["candidates"][0]
    assert view["throughput_tok_s"] == 1200.0                    # the number DID update
    assert view["validations"] == 1                              # the ledger did not reset


def test_attest_needs_a_session_id(tmp_path):
    with pytest.raises(SystemExit):
        e2e_store.main(["attest"] + IDENTITY + ["--store", str(tmp_path / "store"),
                                                "--outcome", "validated", "--session-id", ""])


def test_attest_on_a_missing_store_reports_not_found(tmp_path):
    out = _run("attest", "--store", str(tmp_path / "nope"), "--session-id", "whatever",
               "--outcome", "failed", "--apply")
    assert out["ok"] is False and all(r["found"] is False for r in out["rungs"])


# -- repro: every record must carry something you can actually run ------------------------------


def test_a_launch_script_is_synthesized_when_the_run_captured_none(tmp_path):
    """The common case: the workflow banked flags and env but no script. Rather than storing a
    record nobody can act on, the write builds one against bench_e2e.sh's env contract and says
    plainly that it was synthesized."""
    out = _write(tmp_path, "a", "tuned",
                 accepted_config={"flags": "--max-num-seqs 256", "env": "VLLM_USE_AITER=1"})
    assert "launch.sh" in out["files"]
    view = _run("resolve", "--store", str(tmp_path / "store"),
                "--cache-dir", str(tmp_path / "cache"))["candidates"][0]
    assert view["repro"]["launch"] == "launch.sh"
    assert view["repro"]["launch_origin"] == "synthesized"
    assert view["repro"]["env_pairs"] == {"VLLM_USE_AITER": "1"}
    assert view["repro"]["server_args"] == "--max-num-seqs 256"
    text = (tmp_path / "cache" / view["session_id"] / "files" / "launch.sh").read_text()
    assert "bench_e2e.sh" in text and "SYNTHESIZED" in text
    assert "EXTRA_SERVER_ARGS='--max-num-seqs 256'" in text
    assert "export VLLM_USE_AITER='1'" in text
    assert "TP='8'" in text and "ISL='1024'" in text        # the workload it was measured at


def test_a_captured_launch_script_is_stored_verbatim_and_says_so(tmp_path):
    (tmp_path / "run.sh").write_text("#!/bin/sh\necho the real thing\n")
    _write(tmp_path, "a", "tuned", final_launch_script=str(tmp_path / "run.sh"))
    view = _run("resolve", "--store", str(tmp_path / "store"),
                "--cache-dir", str(tmp_path / "cache"))["candidates"][0]
    assert view["repro"]["launch_origin"] == "captured"
    assert (tmp_path / "cache" / view["session_id"] / "files" / "launch.sh").read_text() \
        == "#!/bin/sh\necho the real thing\n"


def test_a_kernel_without_its_patch_is_counted_not_hidden(tmp_path):
    """A reader seeing three kernels and two patches cannot otherwise tell whether the third was a
    no-op or whether its bytes were simply lost, and those read in opposite directions."""
    (tmp_path / "k.patch").write_text("--- a\n+++ b\n")
    _write(tmp_path, "a", "tuned",
           accepted_kernels=[{"name": "op1", "language": "triton", "patch": str(tmp_path / "k.patch")},
                             {"name": "op2", "language": "triton"}])
    view = _run("resolve", "--store", str(tmp_path / "store"))["candidates"][0]
    assert view["repro"]["kernels_without_patch"] == 1
    assert view["repro"]["complete"] is False
    assert [k["patch"] for k in view["repro"]["kernels"]] == ["kernels/op1.patch", ""]


def test_a_kernel_patch_can_be_fetched_from_the_kernel_lanes_own_store(tmp_path):
    """--kernel-store closes the usual gap: by the time an e2e run finalizes, the kernel
    workflow's scratch is gone, so the bytes only survive where the kernel lane filed them."""
    from kb.store_local import LocalKBStore
    kroot = str(tmp_path / "kernelkb")
    kstore = LocalKBStore(kroot, metric="speedup")
    (tmp_path / "k.patch").write_text("--- a\n+++ b\n")
    kid = "geak:kernel:gfx950:op1:triton:rocm:7.2"
    kstore.write(kid, "ksess", {"schema": "geak.kernel.v1", "speedup": 1.4, "value": {}},
                 {"patch.diff": str(tmp_path / "k.patch")})
    kstore.promote(kid, "ksess", 1.4)
    _write(tmp_path, "a", "tuned", "--kernel-store", kroot, "--rocm-version", "7.2",
           accepted_kernels=[{"name": "op1", "language": "triton"}])
    view = _run("resolve", "--store", str(tmp_path / "store"),
                "--cache-dir", str(tmp_path / "cache"))["candidates"][0]
    assert view["repro"]["kernels_without_patch"] == 0 and view["repro"]["complete"] is True
    assert (tmp_path / "cache" / view["session_id"] / "files" / "kernels" / "op1.patch") \
        .read_text() == "--- a\n+++ b\n"


def test_write_refusing_a_record_nobody_could_reproduce(tmp_path):
    """A number with no config, no script, no patch and no overlay is not knowledge — and this
    store has no delete to take it back with."""
    (tmp_path / "bare.json").write_text(json.dumps(
        {"final_throughput_tok_s": 1000.0, "baseline_throughput_tok_s": 800.0}))
    with pytest.raises(SystemExit):
        _run("write", "--store", str(tmp_path / "store"), "--result", str(tmp_path / "bare.json"),
             "--direction", "d", "--apply")


def test_a_kernel_patch_alone_is_enough_to_be_reproducible(tmp_path):
    """The gate is "is there anything to act on", not "is there a config" — a run whose whole win
    was a kernel rewrite carries the patch and nothing else, and it is perfectly actionable."""
    (tmp_path / "k.patch").write_text("--- a\n+++ b\n")
    out = _write(tmp_path, "a", "kernels", accepted_config={},
                 accepted_kernels=[{"name": "op1", "language": "triton",
                                    "patch": str(tmp_path / "k.patch")}])
    assert out["rungs"][0]["written"] is True


def test_the_prose_reference_points_at_the_launch_script_and_the_patches(tmp_path):
    """The Director reads this and then opens a file, so the paths are spelled out — not "see the
    bundle", which sends it back to the store to ask a question the page already answered."""
    (tmp_path / "k.patch").write_text("--- a\n+++ b\n")
    _write(tmp_path, "a", "tuned",
           accepted_kernels=[{"name": "op1", "language": "triton",
                              "patch": str(tmp_path / "k.patch")}])
    _run("resolve", "--store", str(tmp_path / "store"), "--refs-dir", str(tmp_path / "refs"),
         "--cache-dir", str(tmp_path / "cache"))
    text = list((tmp_path / "refs").glob("e2e_reference_*.md"))[0].read_text()
    assert "- reproduce: " in text and "files/launch.sh" in text
    assert "files/kernels/op1.patch" in text
    assert "- track record: never benched by anyone since it was recorded" in text


# -- the identity handoff -----------------------------------------------------------------------
# The read leaves its ADDRESS on disk, not just its answer. The writer at the end of a run is a
# different process, and when the workflow dies it is a different program entirely (run_e2e.py
# salvaging from artifacts) — one that has the measurement but not the dimensions the Director
# established at preflight. Written from the same argv that formed the read, because two places
# formatting these dims independently is how a reader and a writer drift onto different pages.


def _identity_file(tmp_path, *args):
    out = tmp_path / "kb_identity.json"
    _run("resolve", "--store", str(tmp_path / "store"), "--identity-out", str(out), *args)
    return json.loads(out.read_text())


def test_the_read_leaves_its_address_behind_for_the_writer(tmp_path):
    doc = _identity_file(tmp_path)
    # `dims` is the raw argv, one key per flag, because its reader hands them straight back
    assert doc["dims"]["model"] == "M" and doc["dims"]["gfx"] == "gfx950"
    assert doc["dims"]["framework-version"] == "0.26.0"
    assert doc["dims"]["tp"] == "8", "the argv value travels verbatim, not reformatted"
    # `identity` is the derived form, kept so a human — or anyone matching a canonical id — can
    # read the file without reimplementing e2e_identity() to get there
    assert doc["identity"]["model"] == "m"
    assert doc["canonical_id"].startswith("geak:e2e:")


def test_the_address_is_written_even_when_there_is_nothing_to_read(tmp_path):
    """A cold start is exactly when the salvage writer matters most: nothing was read, but the
    run still produced a measurement that belongs on this page."""
    doc = _identity_file(tmp_path)
    assert doc["dims"]["model"] == "M"


def test_the_recorded_plane_is_the_runs_plane_not_the_reads(tmp_path):
    """A `both` run reads remote-first, so the read's own plane would talk the salvage writer
    out of the local mirror the run was configured to keep."""
    assert _identity_file(tmp_path, "--plane", "remote", "--identity-plane", "both")["plane"] \
        == "both", "the run's plane wins over the read's"
    assert _identity_file(tmp_path, "--plane", "remote")["plane"] == "remote"
    assert _identity_file(tmp_path)["plane"] == "local", "the read's plane is the fallback"


def test_an_unwritable_identity_path_does_not_fail_the_read(tmp_path):
    """Best-effort by construction: the caller asked for candidates, and it gets them."""
    _write(tmp_path, "a", "tuned")
    out = _run("resolve", "--store", str(tmp_path / "store"),
               "--identity-out", str(tmp_path / "nope" / "deep" / "id.json"))
    assert out["read_reason"] == "read" and out["candidates"]


# -- packing an overlay -------------------------------------------------------------------------
# A pure-overlay win is a directory, not a patch. It travels as a tarball so the reader gets one
# predictable directory to point PYTHONPATH at.


def _overlay_dir(tmp_path, *, manifest=True, patched=True):
    d = tmp_path / "overlay"
    (d / "_patched" / "vllm").mkdir(parents=True)
    if manifest:
        (d / e2e_store.OVERLAY_MANIFEST).write_text(json.dumps({"files": ["vllm/layer.py"]}))
    (d / "sitecustomize.py").write_text("# hook\n")
    if patched:
        (d / "_patched" / "vllm" / "layer.py").write_text("BLOCK = 256\n")
        (d / "_patched" / "vllm" / "layer.pyc").write_text("junk")
        (d / "_patched" / "vllm" / "__pycache__").mkdir()
        (d / "_patched" / "vllm" / "__pycache__" / "layer.cpython-311.pyc").write_text("junk")
    return str(d)


def test_a_directory_without_a_manifest_is_not_an_overlay(tmp_path):
    """The gate refuses the record rather than promise a tarball of something nobody can install."""
    assert e2e_store._pack_overlay(_overlay_dir(tmp_path, manifest=False)) == ""


def test_an_overlay_packs_its_manifest_hook_and_patched_tree(tmp_path):
    import tarfile
    out = e2e_store._pack_overlay(_overlay_dir(tmp_path))
    assert out.endswith(e2e_store.OVERLAY_TARBALL) and os.path.isfile(out)
    with tarfile.open(out, "r:gz") as tar:
        names = tar.getnames()
    assert "overlay/" + e2e_store.OVERLAY_MANIFEST in names
    assert "overlay/sitecustomize.py" in names
    assert "overlay/_patched/vllm/layer.py" in names


def test_compiled_bytecode_is_left_out_of_the_bundle(tmp_path):
    """`.pyc` files are built for one interpreter and are stale everywhere else; shipping them
    is how an overlay installs cleanly and then runs the code it was supposed to replace."""
    import tarfile
    with tarfile.open(e2e_store._pack_overlay(_overlay_dir(tmp_path)), "r:gz") as tar:
        names = tar.getnames()
    assert not any(n.endswith(".pyc") for n in names)
    assert not any("__pycache__" in n for n in names)


def _rebind_overlay(tmp_path, impl="engage"):
    """A `rebinds` overlay: the implementation is a sibling at the root, not under `_patched/`."""
    d = tmp_path / "overlay"
    d.mkdir()
    (d / e2e_store.OVERLAY_MANIFEST).write_text(json.dumps(
        {"modules": [], "rebinds": [{"target": "vllm.attn:fwd", "impl_module": impl,
                                     "impl_attr": "fwd"}]}))
    (d / "sitecustomize.py").write_text("# hook\n")
    return d


def test_a_rebind_ships_the_module_it_binds_to(tmp_path):
    """Packing only `_patched/` gave a manifest that rebinds to an import the tarball lacks."""
    import tarfile
    d = _rebind_overlay(tmp_path)
    (d / "engage.py").write_text("def fwd(): pass\n")
    with tarfile.open(e2e_store._pack_overlay(str(d)), "r:gz") as tar:
        names = tar.getnames()
    assert "overlay/engage.py" in names


def test_a_rebind_through_a_shim_ships_the_kernel_behind_it(tmp_path):
    """The manifest's module is regularly 1 KB re-exporting the real kernel from a sibling."""
    import tarfile
    d = _rebind_overlay(tmp_path)
    (d / "engage.py").write_text("from authored import fwd\n")
    (d / "authored.py").write_text("import triton\n\ndef fwd(): pass\n")
    with tarfile.open(e2e_store._pack_overlay(str(d)), "r:gz") as tar:
        names = tar.getnames()
    assert "overlay/authored.py" in names, "the shim shipped without the kernel behind it"
    assert not any("triton" in n for n in names), "a third-party import is not an overlay sibling"


def test_a_submodule_rebind_ships_its_package(tmp_path):
    """`geak_authored.gemm_flydsl` is addressed by the package that sits at the overlay root."""
    import tarfile
    d = _rebind_overlay(tmp_path, impl="geak_authored.gemm")
    (d / "geak_authored").mkdir()
    (d / "geak_authored" / "__init__.py").write_text("")
    (d / "geak_authored" / "gemm.py").write_text("def fwd(): pass\n")
    with tarfile.open(e2e_store._pack_overlay(str(d)), "r:gz") as tar:
        names = tar.getnames()
    assert "overlay/geak_authored/__init__.py" in names
    assert "overlay/geak_authored/gemm.py" in names


def test_sitecustomize_siblings_travel_with_it(tmp_path):
    """A capture or trace overlay names its helpers nowhere but in sitecustomize.py's imports."""
    import tarfile
    d = _rebind_overlay(tmp_path)
    (d / "engage.py").write_text("def fwd(): pass\n")
    (d / "sitecustomize.py").write_text("import os\ntry:\n    import capture_shapes\n"
                                        "except Exception:\n    pass\n")
    (d / "capture_shapes.py").write_text("SHAPES = []\n")
    with tarfile.open(e2e_store._pack_overlay(str(d)), "r:gz") as tar:
        names = tar.getnames()
    assert "overlay/capture_shapes.py" in names


def test_a_module_name_that_could_escape_the_root_is_refused(tmp_path):
    """`impl_module` is a module name; anything path-shaped is not resolved into the filesystem."""
    import tarfile
    d = tmp_path / "overlay"
    d.mkdir()
    (d / e2e_store.OVERLAY_MANIFEST).write_text(json.dumps(
        {"rebinds": [{"impl_module": "/etc/passwd"}, {"impl_module": "../escape"},
                     {"impl_module": "a/b"}]}))
    (d / "sitecustomize.py").write_text("# hook\n")
    with tarfile.open(e2e_store._pack_overlay(str(d)), "r:gz") as tar:
        names = tar.getnames()
    assert names == ["overlay/" + e2e_store.OVERLAY_MANIFEST, "overlay/sitecustomize.py"]


# -- the tuning track's lever, carried into the record ---------------------------------------------
#
# A tuned table is bound by an env var and deployed INTO the installed package, so it is structurally
# absent from final.patch. The DeepSeek-V4-Pro 20260823 run banked a 56-row a8w8 table measured at
# 3.29x isolated and wrote files [final.patch, launch.sh, report.md]: the lever stayed on the box and
# died with it. These cover the path that carries it instead — and, since the tuning track now READS
# this store too, the pointers a later run has to follow back to the bytes.


def _tuned(tmp_path, *names, gate="accepted", live=()):
    paths = []
    for name in names:
        (tmp_path / name).write_text("M,N,K,kernel\n1024,8192,7168,ck_cshuffle_v3\n")
        paths.append(str(tmp_path / name))
    return {"gate": gate, "artifacts": paths, "live_tree_files": [str(tmp_path / n) for n in live]}


def test_a_tuned_table_rides_along_with_the_record(tmp_path):
    out = _write(tmp_path, "a", "tuned", tuning_skillset=_tuned(tmp_path, "gemm.csv"))
    assert out["files"] == sorted(set(out["files"]))            # deterministic, no duplicate names
    assert "tuning/00_gemm.csv" in out["files"]
    view = _run("resolve", "--store", str(tmp_path / "store"),
                "--cache-dir", str(tmp_path / "cache"))["candidates"][0]
    landed = tmp_path / "cache" / view["session_id"] / "files" / "tuning" / "00_gemm.csv"
    assert landed.is_file() and "ck_cshuffle_v3" in landed.read_text()


def test_only_an_accepted_tuning_contributes_its_artifacts(tmp_path):
    """A rejected search leaves files behind too. Carrying them would let a later reader install a
    table that was measured and turned down, which is worse than having nothing to install."""
    out = _write(tmp_path, "a", "tuned", tuning_skillset=_tuned(tmp_path, "gemm.csv", gate="no_win"))
    assert not any(f.startswith("tuning/") for f in out["files"])


def test_a_live_tree_table_is_carried_even_though_no_diff_could_see_it(tmp_path):
    (tmp_path / "installed.csv").write_text("x")
    out = _write(tmp_path, "a", "tuned",
                 tuning_skillset=_tuned(tmp_path, "bundled.csv", live=("installed.csv",)))
    assert "tuning/00_bundled.csv" in out["files"] and "tuning/01_installed.csv" in out["files"]


def test_a_shape_named_table_cannot_take_the_whole_record_down_with_it(tmp_path):
    """`safe_rel_path` REJECTS `:` rather than sanitizing, and both stores build their entire
    {stored_name: source} map before uploading a byte — so one `gemm_m:1024.csv`, a perfectly
    ordinary thing for a tuner to emit, would abort the write and lose final.patch and launch.sh
    along with it. The name is mangled toward the validator; the record survives."""
    from kb.store_local import safe_rel_path
    out = _write(tmp_path, "a", "tuned", tuning_skillset=_tuned(tmp_path, "gemm_m:1024_n:8192.csv"))
    assert "tuning/00_gemm_m_1024_n_8192.csv" in out["files"]
    assert "launch.sh" in out["files"]                          # the rest of the record is intact
    for name in out["files"]:
        assert safe_rel_path(name) == name                      # every stored name is uploadable


def test_a_recalled_table_can_be_found_from_the_page_that_advertises_it(tmp_path):
    """The whole point of banking the lever: a later run reads this prose and has to be able to act
    on it. The workflow banks the absolute path the tuner wrote, which means nothing on another box,
    so the record must point at the bundle instead — while still saying where it came from."""
    out = _write(tmp_path, "a", "tuned",
                 tuning_skillset=_tuned(tmp_path, "gemm.csv"),
                 accepted_kernels=[{"name": "gemm_a8w8", "language": "ck", "winner_kind": "env",
                                    "isolated_speedup": 3.29, "from_tuning_skillset": True,
                                    "apply_env": "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=/x/gemm.csv",
                                    "tuning_artifact": str(tmp_path / "gemm.csv")}])
    assert "tuning/00_gemm.csv" in out["files"]
    resolved = _run("resolve", "--store", str(tmp_path / "store"),
                    "--cache-dir", str(tmp_path / "cache"), "--refs-dir", str(tmp_path / "refs"))
    kernel = resolved["candidates"][0]["accepted_kernels"][0]
    assert kernel["tuning_artifact"] == "tuning/00_gemm.csv"    # resolves inside the bundle
    assert kernel["tuning_artifact_source"] == str(tmp_path / "gemm.csv")
    page = "".join(p.read_text() for p in (tmp_path / "refs").glob("e2e_reference_*.md"))
    assert "tuning/00_gemm.csv" in page and "from tuning skillset" in page
    assert "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE" in page          # and how to bind it


def test_a_patched_kernel_reads_exactly_as_it_did_before(tmp_path):
    """The tuning branch is additive: a record with no tuning in it must render byte-identically."""
    assert e2e_store._kernel_line([{"name": "op1", "language": "triton", "isolated_speedup": 1.84,
                                    "patch": "kernels/op1.patch"}]) \
        == "op1 (triton, 1.84x, kernels/op1.patch)"


# -- the win gate --------------------------------------------------------------------------------
#
# ONE implementation, two callers: e2e_workflow.js at the end of a live run and run_e2e.py when it
# salvages a run whose workflow died first. The gate keys on the Director's VERDICT, not on the raw
# ratio, because a KB write is permanent — the service exposes no DELETE, so a wrong record cannot
# be cleaned up, only outranked.


def test_win_gate_lets_a_declared_win_through():
    assert e2e_store.win_gate({"throughput_speedup": 1.07, "final_throughput_tok_s": 787.0,
                               "validation_status": "validated_win"}) == ""


@pytest.mark.parametrize("status", ["validated_no_win", "recovered_no_gain",
                                    "flagged_parity", "flagged_"])
def test_win_gate_refuses_a_declared_no_win_however_good_the_ratio(status):
    """The 20260822 gemma-4-26B shape: 1.0215x same-session while measuring 0.9453x against its
    provided baseline. Keying on the ratio alone minted that below-baseline number as champion."""
    why = e2e_store.win_gate({"throughput_speedup": 1.0215, "final_throughput_tok_s": 900.0,
                              "validation_status": status})
    assert why
    assert "box-drift" in why and status in why


@pytest.mark.parametrize("status", ["validated_win", "recovered_intermediate", "", None])
def test_win_gate_does_not_overmatch_verdict_prefixes(status):
    assert e2e_store.win_gate({"throughput_speedup": 1.2, "final_throughput_tok_s": 900.0,
                               "validation_status": status}) == ""


@pytest.mark.parametrize("speedup", [1.0, 0.9453, 0.0, -1.0, None, "n/a", float("inf"),
                                     float("nan")])
def test_win_gate_refuses_anything_that_is_not_above_1x(speedup):
    why = e2e_store.win_gate({"throughput_speedup": speedup, "final_throughput_tok_s": 900.0,
                              "validation_status": "validated_win"})
    assert why.startswith("no win to record")


@pytest.mark.parametrize("final", [0.0, None, "", "n/a", float("nan")])
def test_win_gate_refuses_a_result_with_no_final_number(final):
    """A ratio with nothing measured under it is a claim, not a measurement — and it is checked
    FIRST, so the message names the real problem instead of blaming the ratio."""
    assert e2e_store.win_gate({"throughput_speedup": 1.5, "final_throughput_tok_s": final,
                               "validation_status": "validated_win"}) \
        == "no final throughput measured"


def test_require_win_declines_the_write_without_failing_the_caller(tmp_path):
    """A refused write is `ok: True, applied: False` — a no-win is a normal outcome, not an error,
    and a caller that treated it as one would fail every honest run."""
    result = _result(tmp_path / "nowin.json", tput=900.0, throughput_speedup=1.0215,
                     validation_status="recovered_no_gain")
    out = _run("write", "--store", str(tmp_path / "store"), "--result", result,
               "--direction", "none", "--apply", "--require-win")
    assert out["ok"] is True and out["applied"] is False and out["skipped"] is True
    assert "Director declared no win" in out["why"]
    assert out["rungs"] == [] and out["files"] == [] and out["session_id"] == ""
    assert not (tmp_path / "store").exists()           # nothing was published


def test_without_require_win_a_human_can_still_backfill(tmp_path):
    """The gate is opt-in: `write` is also the backfill path, where a human has evidence the result
    JSON cannot carry (a framework-layer win the sub-run self-judged no-win)."""
    result = _result(tmp_path / "backfill.json", tput=900.0, throughput_speedup=1.0215,
                     validation_status="recovered_no_gain")
    out = _run("write", "--store", str(tmp_path / "store"), "--result", result,
               "--direction", "none", "--apply")
    assert out["applied"] is True and out["rungs"]


def test_require_win_passes_a_real_win_straight_through(tmp_path):
    out = _write(tmp_path, "win", "attn-split", "--require-win", throughput_speedup=1.25)
    assert out["applied"] is True and out.get("skipped") is not True
    assert out["rungs"]


# -- the version cut's legacy rungs ---------------------------------------------------------------
# `framework_version` is now the RELEASE (`0.5.15`), not the build string. Every record filed under
# the old spelling sits on a page the new address cannot name, and no rung here drops the version, so
# there is no coarse page to catch them. The read gets those addresses back; the write must not.
DEV_VERSION = "0.5.15.post1.dev20260723+g6c9fd0adc5"
DEV_IDENTITY = ["--model", "M", "--gfx", "gfx950", "--framework", "sglang",
                "--framework-version", DEV_VERSION, "--precision", "mxfp4",
                "--tp", "8", "--isl", "1024", "--osl", "1024", "--conc", "64"]


def _as_the_old_lane_did(monkeypatch):
    """Undo the cut for the duration of a write: the pre-#438 lane segmented the build string whole,
    so this is what "a record that is already in the store" is spelled like."""
    monkeypatch.setattr(e2e_store.kbid, "_release_version",
                        lambda raw: e2e_store.kbid.segment(raw, e2e_store.kbid.UNKNOWN_VERSION))


def test_a_record_written_before_the_cut_is_still_reachable(tmp_path, monkeypatch):
    """The whole point. Without the legacy rung this read is a cold start on a store that has the
    answer — and a cold start is not an error, so nothing would ever report it."""
    result = _result(tmp_path / "old.json")
    _as_the_old_lane_did(monkeypatch)
    written = _run("write", "--store", str(tmp_path / "store"), "--result", result,
                   "--direction", "tuned", "--apply", identity=DEV_IDENTITY)
    assert written["applied"] is True
    monkeypatch.undo()
    out = _run("resolve", "--store", str(tmp_path / "store"), identity=DEV_IDENTITY)
    assert out["read_reason"] == "read" and out["candidates"]
    assert out["match_tier"] == "legacy_version"
    assert DEV_VERSION in out["canonical_id"]      # the build string verbatim, as it was filed


def test_the_legacy_rungs_are_read_only(tmp_path, monkeypatch):
    """A rescue rung that also accepts writes would keep the old page alive forever, splitting one
    deployment's history across two addresses — the state the cut exists to end."""
    out = _run("identity", identity=DEV_IDENTITY)
    assert [r["tier"] for r in out["ladder"]] == ["exact", "workload_any", "tp_any"]
    assert all("0.5.15:" in r["canonical_id"] for r in out["ladder"])
    assert [r["tier"] for r in out["legacy_read_only"]] == \
        ["legacy_version", "legacy_version_workload_any", "legacy_version_tp_any"]
    # and a write files at the new address only
    written = _run("write", "--store", str(tmp_path / "store"),
                   "--result", _result(tmp_path / "new.json"), "--direction", "tuned", "--apply",
                   identity=DEV_IDENTITY)
    assert all("0.5.15:" in r["canonical_id"] for r in written["rungs"])
    assert not any("dev20260723" in r["canonical_id"] for r in written["rungs"])


def test_a_release_shaped_version_grows_no_legacy_rungs(tmp_path):
    """The cut is a no-op for a version already spelled as its release, and the ladder must be too:
    duplicate rungs would re-read the same page and count one record twice in `tried`."""
    out = _run("identity")                       # IDENTITY's version is a bare "0.26.0"
    assert out["legacy_read_only"] == []


def test_the_canonical_ladder_is_still_tried_first(tmp_path, monkeypatch):
    """Rescue, never shadow. A current record and a legacy one both exist; the current one answers."""
    _as_the_old_lane_did(monkeypatch)
    _run("write", "--store", str(tmp_path / "store"), "--result", _result(tmp_path / "o.json",
         tput=2000.0), "--direction", "old", "--apply", identity=DEV_IDENTITY)
    monkeypatch.undo()
    _run("write", "--store", str(tmp_path / "store"), "--result", _result(tmp_path / "n.json",
         tput=1000.0), "--direction", "new", "--apply", identity=DEV_IDENTITY)
    out = _run("resolve", "--store", str(tmp_path / "store"), identity=DEV_IDENTITY)
    # the legacy page ranks HIGHER on throughput, and still must not be the one that answers
    assert out["match_tier"] == "exact" and [c["direction"] for c in out["candidates"]] == ["new"]


# -- ranking: a record nobody can reproduce must not evict the ones they can ---------------------
#
# The direction collapse keeps exactly ONE entry per direction, the first in rank order. So a record
# that outranks its group on the raw scalar does not merely lead the group, it deletes the rest of
# it. A page holding an unreproducible 1.25x and a validated 1.05x, both `direction: kernels`,
# offered the first and hid the second — and the ranking never looked at which had ever worked.


def _fail_twice(tmp_path, sid):
    for who in ("boxA", "boxB"):
        _run("attest", "--store", str(tmp_path / "store"), "--session-id", sid,
             "--outcome", "failed", "--measured-by", who, "--apply")


def _offered(tmp_path):
    out = _run("resolve", "--store", str(tmp_path / "store"))
    return [c["session_id"] for c in out["candidates"]], out["curation"]


def test_a_hinted_record_sorts_behind_an_unhinted_one_whatever_the_numbers(tmp_path):
    loud = _write(tmp_path, "loud", "kernels", tput=1250.0)["session_id"]
    quiet = _write(tmp_path, "quiet", "kernels", tput=1050.0, accepted_config={"env": "B=2"})
    quiet = quiet["session_id"]
    assert _offered(tmp_path)[0] == [loud]                 # the higher number leads, as it should
    _fail_twice(tmp_path, loud)
    offered, curation = _offered(tmp_path)
    assert offered == [quiet]                              # ...until nobody can reproduce it
    assert curation["demoted_by_hint"] == 1


def test_the_demoted_record_is_still_on_the_page(tmp_path):
    """Demotion is not filtering. A record nobody has managed to reproduce is exactly the one worth
    keeping until something better replaces it; only a retraction removes it from a read. Two
    directions, so the collapse keeps both and the ORDER is the only thing under test."""
    loud = _write(tmp_path, "loud", "kernels", tput=1250.0)["session_id"]
    quiet = _write(tmp_path, "quiet", "config", tput=1050.0,
                   accepted_config={"env": "B=2"})["session_id"]
    _fail_twice(tmp_path, loud)
    offered, curation = _offered(tmp_path)
    assert offered == [quiet, loud]                        # behind, not gone
    assert curation["demoted_by_hint"] == 1
    demoted = _run("resolve", "--store", str(tmp_path / "store"))["candidates"][1]
    assert demoted["throughput_tok_s"] == 1250.0           # with its real number, not zeroed
    assert "came back negative" in demoted["retire_hint"]


def test_the_floor_is_applied_before_the_collapse_not_after(tmp_path):
    """A record that cannot be offered must not be able to hold a direction slot hostage.

    The collapse keeps ONE entry per direction, so filtering after it meant the group's best record
    took the slot and was then dropped by the floor - taking the whole direction with it, including
    a runner-up in the same group that cleared the floor comfortably. Both records are `kernels`,
    and the read is ranked on throughput, so the one that leads is the one the floor rejects.
    """
    _write(tmp_path, "leader", "kernels", tput=1200.0, baseline_throughput_tok_s=1190.0)
    solid = _write(tmp_path, "solid", "kernels", tput=1100.0, baseline_throughput_tok_s=800.0,
                   accepted_config={"env": "B=2"})["session_id"]
    out = _run("resolve", "--store", str(tmp_path / "store"), "--min-speedup", "1.05")
    assert [c["session_id"] for c in out["candidates"]] == [solid]
    assert out["curation"]["below_min_speedup"] == 1
    # The leader was gone before the collapse ran, so it collapsed nothing on its way out.
    assert out["curation"]["same_direction_collapsed"] == 0


def test_same_direction_runners_up_ride_along_with_the_record_that_evicted_them(tmp_path):
    """The collapse offers one entry per IDEA. The settings it dropped are still worth naming.

    Three records under `kernels` and one under `tuned`: the read offers two directions, and the two
    `kernels` records that lost the slot come back attached to the one that won it - not to the read
    as a whole, because which record an alternate is an alternate TO is the whole of its meaning.
    """
    lead = _write(tmp_path, "lead", "kernels", tput=1300.0)["session_id"]
    second = _write(tmp_path, "second", "kernels", tput=1200.0,
                    accepted_config={"env": "B=2"})["session_id"]
    third = _write(tmp_path, "third", "kernels", tput=1100.0,
                   accepted_config={"env": "B=3"})["session_id"]
    other = _write(tmp_path, "other", "tuned", tput=1050.0)["session_id"]
    out = _run("resolve", "--store", str(tmp_path / "store"))
    assert [c["session_id"] for c in out["candidates"]] == [lead, other]
    assert [alt["session_id"] for alt in out["candidates"][0]["alternates"]] == [second, third]
    assert out["candidates"][0]["alternates"][0]["throughput_tok_s"] == 1200.0
    assert out["candidates"][1]["alternates"] == []          # nothing else was filed under `tuned`
    assert out["candidates"][0]["alternates_omitted"] == 0
    # And the Director reads prose, not JSON, so the runners-up have to reach that file too.
    text = list_reference_text(tmp_path, str(tmp_path / "store"))
    assert second in text and third in text
    assert "nothing else recorded" in text                   # said for `tuned`, not left blank


def test_the_alternate_list_is_bounded_and_says_how_much_it_dropped(tmp_path):
    """A page can hold dozens of re-runs of one direction; a prompt cannot hold dozens of them."""
    from e2e_store import ALTERNATES_LIMIT
    for i in range(ALTERNATES_LIMIT + 3):
        _write(tmp_path, "r%d" % i, "kernels", tput=1300.0 - i,
               accepted_config={"env": "B=%d" % i})
    top = _run("resolve", "--store", str(tmp_path / "store"))["candidates"][0]
    assert len(top["alternates"]) == ALTERNATES_LIMIT
    # Reported, not silently truncated: a short list that does not say it is short reads as the
    # whole group, and "this direction was only tried three ways" is then simply false.
    assert top["alternates_omitted"] == 2


def test_an_alternate_carries_the_verdicts_that_would_make_you_skip_it(tmp_path):
    """A runner-up is most useful when the leader disappoints, which is exactly when you need to
    know that the runner-up has already been tried twice and never reproduced."""
    _write(tmp_path, "lead", "kernels", tput=1300.0)
    dud = _write(tmp_path, "dud", "kernels", tput=1200.0,
                 accepted_config={"env": "B=2"})["session_id"]
    for _ in range(2):
        _run("attest", "--store", str(tmp_path / "store"), "--session-id", dud,
             "--outcome", "not_reproduced", "--apply")
    alt = _run("resolve", "--store", str(tmp_path / "store"))["candidates"][0]["alternates"][0]
    assert alt["session_id"] == dud and "came back negative" in alt["retire_hint"]


# -- curate: the judgement between attesting and retracting --------------------------------------


def _curate(tmp_path, *args):
    return _run("curate", "--store", str(tmp_path / "store"), *args)


def test_curate_is_a_dry_run_by_default(tmp_path):
    """It acts on everything a scan turns up rather than on one id a human typed, so a mistaken
    --apply costs a page and the store has no delete to undo it with."""
    sid = _write(tmp_path, "bad", "kernels")["session_id"]
    _fail_twice(tmp_path, sid)
    out = _curate(tmp_path)
    assert out["applied"] is False
    assert [c["session_id"] for c in out["candidates"]] == [sid]
    assert not any(r["rewritten"] for r in out["rungs"])
    assert _offered(tmp_path)[0] == [sid]                  # still there, untouched


def test_curate_reports_the_evidence_and_not_just_the_verdict(tmp_path):
    sid = _write(tmp_path, "bad", "kernels")["session_id"]
    _fail_twice(tmp_path, sid)
    candidate = _curate(tmp_path)["candidates"][0]
    assert candidate["attestations"]["failures"] == 2
    assert "policy threshold is 2" in candidate["reason"]
    assert candidate["retire_hint"]                        # why it drew attention, separately


def test_curate_leaves_records_that_have_not_earned_it(tmp_path):
    good = _write(tmp_path, "good", "kernels")["session_id"]
    _run("attest", "--store", str(tmp_path / "store"), "--session-id", good,
         "--outcome", "failed", "--measured-by", "boxA", "--apply")
    out = _curate(tmp_path)
    assert out["candidates"] == [] and out["kept"] == 1 and out["ok"] is True


def test_curate_applies_to_every_rung(tmp_path):
    """One session id addresses all three. A record removed from the exact page while it still
    ranks on the two coarse ones is still being offered to most readers."""
    sid = _write(tmp_path, "bad", "kernels")["session_id"]
    _fail_twice(tmp_path, sid)
    out = _curate(tmp_path, "--apply")
    assert out["ok"] and len(out["rungs"]) == 3 and all(r["rewritten"] for r in out["rungs"])
    offered, curation = _offered(tmp_path)
    assert offered == [] and curation["retired"] == 1


def test_curate_skips_what_is_already_retracted(tmp_path):
    sid = _write(tmp_path, "bad", "kernels")["session_id"]
    _fail_twice(tmp_path, sid)
    _curate(tmp_path, "--apply")
    again = _curate(tmp_path)
    assert again["already_retired"] == 1 and again["candidates"] == []


def test_curate_honours_the_threshold(tmp_path):
    sid = _write(tmp_path, "bad", "kernels")["session_id"]
    _fail_twice(tmp_path, sid)
    assert _curate(tmp_path, "--threshold", "3")["candidates"] == []
    assert _curate(tmp_path)["candidates"]


def test_curate_on_an_unreadable_store_reports_rather_than_raises(tmp_path):
    """And does not report success. This is the command most likely to run unattended, so a
    mistyped --store must not be indistinguishable from a page with nothing to retire."""
    out = _curate(tmp_path)
    assert out["candidates"] == [] and out["ok"] is False and out["error"]


def test_curate_does_not_judge_a_page_from_another_workload(tmp_path):
    """It scans the FINEST rung only. The coarse rungs hold records from other workloads, and
    retiring one of those on this deployment's evidence judges a shape this run never ran."""
    sid = _write(tmp_path, "bad", "kernels")["session_id"]
    _fail_twice(tmp_path, sid)
    other = ["--model", "M", "--gfx", "gfx950", "--framework", "vllm",
             "--framework-version", "0.26.0", "--precision", "fp8",
             "--tp", "8", "--isl", "4096", "--osl", "1024", "--conc", "64"]
    out = _run("curate", "--store", str(tmp_path / "store"), identity=other)
    assert out["candidates"] == []


def test_the_reference_prose_names_ran_and_lost_separately(tmp_path):
    """`failures` is the bucket production actually fills now. Folded into "none reproduced a win"
    it reads identically to a record nobody has ever managed to launch, which is the opposite
    diagnosis and points at the opposite fix."""
    sid = _write(tmp_path, "bad", "kernels")["session_id"]
    _fail_twice(tmp_path, sid)
    _run("resolve", "--store", str(tmp_path / "store"), "--refs-dir", str(tmp_path / "refs"))
    prose = "\n".join(p.read_text() for p in (tmp_path / "refs").glob("e2e_reference_*.md"))
    assert "2 ran and did not win" in prose
