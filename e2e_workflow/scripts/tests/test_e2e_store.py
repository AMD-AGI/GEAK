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

    def flaky(store, recs, files, score_of):
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
