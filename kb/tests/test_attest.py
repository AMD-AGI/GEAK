#!/usr/bin/env python3
"""The attestation ledger: counting what happened when a record was actually tried on a box.

The arithmetic is small; what these tests pin down is the two things that make the counters worth
reading. First, an attestation must NOT move a record — no ranking scalar, no champion — because
one box's failure is evidence and a retraction is a verdict, and collapsing the two would make the
command too dangerous to run automatically. Second, the ledger has to survive a rewrite: session
ids are content-addressed off the config, so re-recording the same configuration replaces the same
document, and a carry-forward that silently drops the history leaves a well-formed record claiming
nobody ever tried it.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from kb.attest import (HISTORY_LIMIT, attest_session, attestation_ok,           # noqa: E402
                       attestations_of, attested_document, carry_attestations,
                       empty_attestations, record_attestation, retire_hint)
from kb.store_local import KBStoreError, LocalKBStore                           # noqa: E402


def _store(tmp_path):
    return LocalKBStore(str(tmp_path / "kb"), metric="speedup")


def _doc(speedup=1.5, **value):
    body = {"direction": "tuned", "retained": True}
    body.update(value)
    return {"schema": "geak.e2e.v1", "speedup": speedup, "value": body}


def _seed(store, cid="geak:e2e:m", sid="s-1", speedup=1.5):
    store.write(cid, sid, _doc(speedup), None)
    store.promote(cid, sid, speedup)
    return cid, sid


# -- the arithmetic -------------------------------------------------------------------------------


def test_an_unattested_value_reads_as_an_empty_ledger():
    """Every field is safe to read off a record written before this module existed."""
    assert attestations_of({}) == empty_attestations()
    assert attestations_of(None)["recalls"] == 0
    assert attestations_of({"attestations": "not a dict"})["validations"] == 0


def test_a_counter_that_is_not_a_number_reads_as_zero():
    """A hand-edited document must degrade to 0, not crash the read path that hydrates a page."""
    ledger = attestations_of({"attestations": {"recalls": "seven", "validations": True,
                                               "failures": -3, "not_reproduced": 2.9}})
    assert (ledger["recalls"], ledger["validations"]) == (0, 0)   # True is a bool, not a count
    assert (ledger["failures"], ledger["not_reproduced"]) == (0, 2)


@pytest.mark.parametrize("outcome,field", [("validated", "validations"), ("failed", "failures"),
                                           ("not_reproduced", "not_reproduced"),
                                           ("inapplicable", "inapplicable")])
def test_each_outcome_increments_recalls_and_its_own_counter(outcome, field):
    value = record_attestation({}, outcome, actor="boxA")
    ledger = value["attestations"]
    assert ledger["recalls"] == 1 and ledger[field] == 1
    assert ledger["last_outcome"] == outcome and ledger["last_by"] == "boxA"
    assert ledger["history"][-1]["outcome"] == outcome


def test_an_unknown_outcome_is_refused_rather_than_counted_as_something():
    with pytest.raises(KBStoreError):
        record_attestation({}, "worked_i_think")


def test_evidence_is_whitelisted_onto_the_history_entry():
    """Both lanes' measurement spellings ride; anything unrecognized is dropped rather than grown
    into the document, which is fetched once per candidate to rank a page."""
    value = record_attestation({}, "validated", evidence={
        "measured_tok_s": 1200, "measured_speedup": 1.8, "parity": "pass",
        "gpu_serial": "leak me"})
    entry = value["attestations"]["history"][-1]
    assert entry["measured_tok_s"] == 1200 and entry["measured_speedup"] == 1.8
    assert entry["parity"] == "pass" and "gpu_serial" not in entry


def test_history_is_bounded_but_the_counters_are_not():
    value = {}
    for _ in range(HISTORY_LIMIT + 5):
        value = record_attestation(value, "failed")
    assert len(value["attestations"]["history"]) == HISTORY_LIMIT
    assert value["attestations"]["recalls"] == HISTORY_LIMIT + 5


def test_record_attestation_does_not_mutate_its_input():
    original = {"direction": "tuned"}
    record_attestation(original, "validated")
    assert original == {"direction": "tuned"}


# -- the retire hint ------------------------------------------------------------------------------


def test_retire_hint_stays_silent_until_there_is_a_pattern():
    assert retire_hint({}) == ""
    assert retire_hint(record_attestation({}, "failed")) == ""             # one loss is not a case
    twice = record_attestation(record_attestation({}, "failed"), "failed")
    assert retire_hint(twice) == ""                                        # losing is not failing


def test_two_non_reproductions_with_no_win_is_a_hint():
    value = record_attestation(record_attestation({}, "not_reproduced"), "not_reproduced")
    assert "could not reproduce" in retire_hint(value)


def test_a_config_that_never_fit_this_box_is_no_evidence_against_the_record():
    """The reason `inapplicable` exists at all.

    A stored e2e config is a whole launch configuration, replayed on top of a baseline the reading
    run did not choose. When the two pin the same knob to different values it could not be applied
    HERE, which says nothing about whether it is right anywhere. Spelled `not_reproduced`, two such
    reads put a retire hint on a record nobody had found anything wrong with; spelled `inapplicable`
    they stay out of the arithmetic however many times they happen.
    """
    value = {}
    for _ in range(5):
        value = record_attestation(value, "inapplicable")
    assert value["attestations"]["recalls"] == 5        # the attempts are still counted...
    assert retire_hint(value) == ""                     # ...they just do not judge the record


def test_an_inapplicable_read_does_not_dilute_a_real_pattern():
    """Removing them from the denominator must not also remove the signal that is there."""
    value = record_attestation(record_attestation({}, "inapplicable"), "not_reproduced")
    assert retire_hint(value) == ""                     # one real non-reproduction is not a case
    value = record_attestation(value, "not_reproduced")
    assert "could not reproduce" in retire_hint(value)  # two of them still is


def test_a_ledger_of_only_inapplicable_reads_survives_a_rewrite():
    """`carry_attestations` decides "is there anything here" from the bucket list; a bucket missing
    from it drops the whole ledger on the next re-measurement, silently and well-formed."""
    previous = record_attestation({}, "inapplicable")
    fresh = carry_attestations(previous, {"direction": "tuned"})
    assert fresh["attestations"]["inapplicable"] == 1


def test_a_record_written_before_inapplicable_existed_reads_as_zero():
    old = {"attestations": {"recalls": 3, "validations": 0, "failures": 3}}
    ledger = attestations_of(old)
    assert ledger["inapplicable"] == 0
    assert "tried 3 times" in retire_hint(old)          # and its hint is unchanged


def test_a_single_validation_clears_the_hint_however_many_failures():
    value = {}
    for outcome in ("not_reproduced", "not_reproduced", "failed", "validated"):
        value = record_attestation(value, outcome)
    assert retire_hint(value) == ""


# -- carry-forward across a rewrite ---------------------------------------------------------------


def test_a_rewrite_carries_the_ledger_onto_the_replacing_document():
    previous = record_attestation({}, "validated")
    fresh = carry_attestations(previous, {"direction": "tuned"})
    assert fresh["attestations"]["validations"] == 1


def test_carrying_from_a_record_nobody_tried_adds_no_ledger():
    """An empty ledger and no ledger mean the same thing, and the shorter one does not imply
    somebody looked."""
    assert "attestations" not in carry_attestations({"attestations": empty_attestations()}, {})
    assert "attestations" not in carry_attestations(None, {})


# -- against a real store -------------------------------------------------------------------------


def test_attesting_moves_no_score_and_no_champion(tmp_path):
    """The whole reason this is not retraction. A record that failed once on one box keeps its
    place in the ordering — the ledger is input to a later human decision, not the decision."""
    store = _store(tmp_path)
    cid, sid = _seed(store, speedup=1.9)
    report = attest_session(store, cid, sid, "not_reproduced", actor="boxA")
    assert report["found"] and report["rewritten"]
    after = store.get_session(cid, sid)
    assert after["speedup"] == 1.9                                    # ranking scalar untouched
    assert after["value"]["retained"] is True                         # not a tombstone
    assert store.champion(cid)["session_id"] == sid                   # still crowned
    assert after["value"]["attestations"]["not_reproduced"] == 1


def test_attestations_accumulate_across_calls(tmp_path):
    store = _store(tmp_path)
    cid, sid = _seed(store)
    for outcome in ("validated", "failed", "not_reproduced"):
        attest_session(store, cid, sid, outcome)
    ledger = store.get_session(cid, sid)["value"]["attestations"]
    assert (ledger["recalls"], ledger["validations"]) == (3, 1)
    assert (ledger["failures"], ledger["not_reproduced"]) == (1, 1)


def test_a_local_rewrite_keeps_the_artifacts_it_was_not_given(tmp_path):
    """LocalKBStore.write() stages a fresh directory and swaps, so a rewrite that omitted the files
    would DELETE them. This is the rule kb/retract.py:existing_files owns and this module reuses."""
    store = _store(tmp_path)
    patch = tmp_path / "p.diff"
    patch.write_text("--- a\n+++ b\n")
    store.write("geak:e2e:m", "s-1", _doc(), {"final.patch": str(patch)})
    attest_session(store, "geak:e2e:m", "s-1", "validated")
    assert store.session_files("geak:e2e:m", "s-1") == ["final.patch"]


def test_a_dry_run_shows_the_document_and_writes_nothing(tmp_path):
    store = _store(tmp_path)
    cid, sid = _seed(store)
    report = attest_session(store, cid, sid, "validated", apply=False)
    assert report["would_write"]["value"]["attestations"]["validations"] == 1
    assert "attestations" not in store.get_session(cid, sid)["value"]


def test_a_session_that_is_not_there_reports_it_rather_than_raising(tmp_path):
    report = attest_session(_store(tmp_path), "geak:e2e:m", "nope", "validated")
    assert report["found"] is False and "no such session" in report["error"]
    assert report["rewritten"] is False


def test_an_unknown_outcome_against_a_store_reports_rather_than_raises(tmp_path):
    store = _store(tmp_path)
    cid, sid = _seed(store)
    report = attest_session(store, cid, sid, "probably_fine")
    assert report["found"] and not report["rewritten"] and "unknown attestation" in report["error"]


def test_attestation_ok_needs_the_record_found_somewhere(tmp_path):
    """A rung the write never reached has nothing to count; finding it NOWHERE means the caller's
    verdict was recorded against nothing at all."""
    assert attestation_ok([{"found": False}], True) is False
    assert attestation_ok([{"found": False}, {"found": True, "rewritten": True}], True) is True
    assert attestation_ok([{"found": True, "rewritten": False}], True) is False
    assert attestation_ok([{"found": False}], False) is True          # a dry run cannot fail


def test_attested_document_refuses_a_non_object():
    with pytest.raises(KBStoreError):
        attested_document("not a document", "validated")
