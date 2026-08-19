#!/usr/bin/env python3
"""Retraction: taking back a record on a store that has no delete.

Every test here is about a way retraction can LOOK done and not be. Marking the document while
leaving the ranking scalar, zeroing the scalar while leaving the champion pointing at it, filtering
on read while the writer never set the flag — each of those passes a casual inspection and leaves
the retracted record still steering the next run.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import e2e_store                                                            # noqa: E402
from kb_retract import is_retired, retracted_document, retraction_ok        # noqa: E402
from kb_store_local import KBStoreError, LocalKBStore                       # noqa: E402

CID = "geak:e2e:m:gfx950:vllm:0.26.0:fp8:tp_8:isl_1024:osl_1024:conc_64"
IDENTITY = ["--model", "M", "--gfx", "gfx950", "--framework", "vllm",
            "--framework-version", "0.26.0", "--precision", "fp8",
            "--tp", "8", "--isl", "1024", "--osl", "1024", "--conc", "64"]


def _result(path, tput, parity="pass", env="A=1", status="validated_win"):
    path.write_text(json.dumps({
        "final_throughput_tok_s": tput, "baseline_throughput_tok_s": 800.0,
        "validation_status": status, "output_parity": parity,
        "accepted_config": {"env": env}, "accepted_kernels": []}))
    return str(path)


def _write(tmp_path, name, tput, direction, **kw):
    result = _result(tmp_path / (name + ".json"), tput, **kw)
    out = _run("write", "--store", str(tmp_path / "store"), "--result", result,
               "--direction", direction, "--apply")
    return out["session_id"]


def _run(command, *args):
    import io
    import contextlib
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        e2e_store.main([command] + IDENTITY + ["--plane", "local"] + list(args))
    return json.loads(buffer.getvalue())


# -- the write-time state fields -------------------------------------------------------------


def test_parity_n_a_is_recorded_as_unvalidated(tmp_path):
    """"We did not check" must not read as "we checked and it was fine"."""
    _write(tmp_path, "b", 900.0, "unchecked", parity="n/a")
    view = _run("resolve", "--store", str(tmp_path / "store"))["candidates"][0]
    assert view["validated"] is False
    assert view["lifecycle"] == "candidate"
    assert view["parity"] == "n/a"


def test_a_win_with_parity_is_validated_and_says_which_gate(tmp_path):
    _write(tmp_path, "a", 1000.0, "tuned")
    view = _run("resolve", "--store", str(tmp_path / "store"))["candidates"][0]
    assert view["validated"] is True
    assert view["validation_basis"] == "hot_ab"


def test_state_does_not_enter_the_dedup_digest(tmp_path):
    """Re-recording one config with a corrected verdict must REPLACE it, not mint a second entry.

    The digest keys on the configuration, so the same config written twice is the same candidate.
    If `validated` leaked into it, every correction would leave the old verdict live alongside the
    new one — on a store with no delete, permanently.
    """
    first = _write(tmp_path, "a", 1000.0, "tuned", parity="pass")
    second = _write(tmp_path, "a2", 1000.0, "tuned", parity="fail")
    assert first == second
    assert len(_run("resolve", "--store", str(tmp_path / "store"))["candidates"]) == 1


# -- the rewrite ------------------------------------------------------------------------------


def test_a_retraction_needs_a_reason():
    try:
        retracted_document({"speedup": 2.0, "value": {}}, "  ", ["speedup"])
    except KBStoreError as e:
        assert "reason" in str(e)
    else:
        raise AssertionError("an unexplained retraction was accepted")


def test_the_ranking_scalar_is_zeroed_not_just_flagged():
    """A flag alone is inert: `sessions/top?metric=` reads the scalar, not the flag."""
    document = retracted_document({"speedup": 2.0, "throughput_tok_s": 900.0, "value": {}},
                                  "wrong bench key", ["speedup", "throughput_tok_s"])
    assert document["speedup"] == 0.0 and document["throughput_tok_s"] == 0.0
    assert is_retired(document["value"])
    # ...and the withdrawn claim survives, or the tombstone cannot be reviewed later.
    assert document["value"]["withdrawn_scores"] == {"speedup": 2.0, "throughput_tok_s": 900.0}


def test_retracting_the_champion_re_points_it_at_the_survivor(tmp_path):
    """Zeroing a document does not un-promote it. The champion is a separate object, and one left
    pointing at a retracted record keeps its old score as the bar every future candidate must
    clear — a single false record quietly closes the page."""
    champion = _write(tmp_path, "a", 1000.0, "tuned")
    survivor = _write(tmp_path, "b", 900.0, "other")
    out = _run("retract", "--store", str(tmp_path / "store"), "--session-id", champion,
               "--reason", "could not be reproduced", "--apply")
    assert out["ok"] is True
    store = LocalKBStore(str(tmp_path / "store"), metric="throughput_tok_s", promote_floor=0.0)
    assert store.champion(CID)["session_id"] == survivor


def test_the_last_record_on_a_page_self_zeroes_its_champion(tmp_path):
    session = _write(tmp_path, "a", 1000.0, "tuned")
    _run("retract", "--store", str(tmp_path / "store"), "--session-id", session,
         "--reason", "sole record, and wrong", "--apply")
    store = LocalKBStore(str(tmp_path / "store"), metric="throughput_tok_s", promote_floor=0.0)
    champion = store.champion(CID)
    # Still pointing somewhere — there is nothing else to point at — but no longer advertising a
    # win, and no longer acting as a floor a future candidate has to beat.
    assert champion["session_id"] == session and champion["value"] == 0.0


def test_artifacts_survive_the_rewrite(tmp_path):
    """The bytes are the evidence. A record has just been called false; that is exactly when
    someone wants to read the patch it shipped."""
    (tmp_path / "final.patch").write_text("--- a\n+++ b\n")
    (tmp_path / "r.json").write_text(json.dumps({
        "final_throughput_tok_s": 1000.0, "baseline_throughput_tok_s": 800.0,
        "validation_status": "validated_win", "output_parity": "pass",
        "final_patch": str(tmp_path / "final.patch"),
        "accepted_config": {"env": "A=1"}, "accepted_kernels": []}))
    session = _run("write", "--store", str(tmp_path / "store"), "--result",
                   str(tmp_path / "r.json"), "--direction", "d", "--apply")["session_id"]
    before = sum(1 for _r, _d, f in os.walk(str(tmp_path / "store")) for _ in f)
    _run("retract", "--store", str(tmp_path / "store"), "--session-id", session,
         "--reason", "wrong", "--apply")
    assert sum(1 for _r, _d, f in os.walk(str(tmp_path / "store")) for _ in f) == before


# -- the read ----------------------------------------------------------------------------------


def test_a_retracted_record_is_not_offered(tmp_path):
    champion = _write(tmp_path, "a", 1000.0, "tuned")
    _write(tmp_path, "b", 900.0, "other")
    _run("retract", "--store", str(tmp_path / "store"), "--session-id", champion,
         "--reason", "wrong", "--apply")
    out = _run("resolve", "--store", str(tmp_path / "store"))
    assert [c["direction"] for c in out["candidates"]] == ["other"]
    assert out["curation"]["retired"] == 1


def test_retracted_is_dropped_before_the_direction_collapse(tmp_path):
    """Order matters. The collapse keeps the best record PER DIRECTION; if a retracted entry is
    still in the list when it runs, it wins its direction and evicts the good alternative behind
    it — so one false record hides a real one instead of merely removing itself."""
    bad = _write(tmp_path, "a", 1000.0, "tuned")
    _write(tmp_path, "b", 950.0, "tuned", env="B=1")
    _run("retract", "--store", str(tmp_path / "store"), "--session-id", bad,
         "--reason", "wrong", "--apply")
    out = _run("resolve", "--store", str(tmp_path / "store"))
    assert [c["throughput_tok_s"] for c in out["candidates"]] == [950.0]


def test_an_all_retracted_page_reports_why_it_looks_empty(tmp_path):
    """"Nobody recorded this" and "everything recorded here was taken back" produce the same
    read_reason and mean opposite things about whether to try again."""
    session = _write(tmp_path, "a", 1000.0, "tuned")
    _run("retract", "--store", str(tmp_path / "store"), "--session-id", session,
         "--reason", "wrong", "--apply")
    out = _run("resolve", "--store", str(tmp_path / "store"))
    assert out["candidates"] == [] and out["read_reason"] == "e2e_page_not_found"
    assert out["curation"]["retired"] == 1


# -- reporting ---------------------------------------------------------------------------------


def test_a_page_that_never_held_the_record_is_not_a_failed_retraction():
    assert retraction_ok([{"found": True, "rewritten": True},
                          {"found": False, "rewritten": False}], True) is True


def test_finding_the_record_nowhere_is_a_failure():
    """Otherwise a typo'd session id reports success while the real record stays live."""
    assert retraction_ok([{"found": False, "rewritten": False}], True) is False


def test_a_dry_run_writes_nothing_and_shows_the_document(tmp_path):
    session = _write(tmp_path, "a", 1000.0, "tuned")
    out = _run("retract", "--store", str(tmp_path / "store"), "--session-id", session,
               "--reason", "wrong")
    assert out["applied"] is False
    assert all(not r["rewritten"] for r in out["rungs"])
    assert out["rungs"][0]["would_write"]["value"]["retired_reason"] == "wrong"
    assert _run("resolve", "--store", str(tmp_path / "store"))["candidates"] != []
