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
from kb.attest import (HISTORY_LIMIT, RECENT_WINDOW, attest_session,            # noqa: E402
                       attestation_ok, attestations_of, attested_document, carry_attestations,
                       empty_attestations, record_attestation, recent_verdicts, retire_hint,
                       should_retire)
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
    assert "came back negative" in retire_hint(twice)                      # two is the pattern


def test_two_non_reproductions_with_no_win_is_a_hint():
    value = record_attestation(record_attestation({}, "not_reproduced"), "not_reproduced")
    assert "not_reproduced, not_reproduced" in retire_hint(value)


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
    assert "came back negative" in retire_hint(value)   # two of them still is


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
    # No history to read, so the hint falls back to the lifetime counters - the only thing a
    # document like this can support. See recent_verdicts().
    assert "3 attempts ran it and none won" in retire_hint(old)


def test_a_validation_clears_every_loss_that_came_before_it():
    value = {}
    for outcome in ("not_reproduced", "not_reproduced", "failed", "validated"):
        value = record_attestation(value, outcome)
    assert retire_hint(value) == ""


def test_a_win_that_has_aged_out_of_the_window_stops_protecting_the_record():
    """The reason the verdicts are a window and not a lifetime total.

    Under a lifetime `validations > 0` veto a record that won once and has lost every time since
    was immune forever: it kept its direction slot against every alternative and no amount of
    accumulated evidence could ever curate it out. RECENT_WINDOW losses in a row now say what a
    reader actually wants to know - it does not work here any more.
    """
    value = record_attestation({}, "validated")
    for _ in range(RECENT_WINDOW - 1):                  # the win is still in view
        value = record_attestation(value, "failed")
        assert retire_hint(value) == ""
        assert should_retire(value) == ""
    value = record_attestation(value, "failed")         # ...and now it is not
    assert "came back negative" in retire_hint(value)
    assert should_retire(value) != ""
    assert value["attestations"]["validations"] == 1    # the lifetime counter is untouched


def test_a_fresh_win_re_arms_the_whole_window():
    """A record does not have to earn its reprieve twice. One validation inside the window clears
    it outright, and the losses behind it have to fall out again before it can be hinted."""
    value = {}
    for _ in range(RECENT_WINDOW + 2):
        value = record_attestation(value, "failed")
    assert retire_hint(value) != ""
    value = record_attestation(value, "validated")
    assert retire_hint(value) == ""


def test_inapplicable_reads_never_push_a_verdict_out_of_the_window():
    """A window slot spent on a read that never got the record onto the box would reprieve it for
    having been read on the wrong machine - the same mistake the bucket exists to prevent."""
    value = record_attestation(record_attestation({}, "failed"), "failed")
    for _ in range(RECENT_WINDOW + 3):
        value = record_attestation(value, "inapplicable")
    assert recent_verdicts(value["attestations"]) == ["failed", "failed"]
    assert "came back negative" in retire_hint(value)


def test_a_threshold_above_the_window_widens_it_instead_of_being_unreachable():
    value = {}
    for _ in range(RECENT_WINDOW + 1):
        value = record_attestation(value, "failed")
    assert should_retire(value, threshold=RECENT_WINDOW + 1) != ""
    assert should_retire(value, threshold=RECENT_WINDOW + 2) == ""   # not enough evidence yet


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


# -- the retire POLICY ----------------------------------------------------------------------------
#
# `retire_hint` asks whether a record is worth a look; `should_retire` asks whether the ledger meets
# the bar the project agreed to act on. The pair only works if the hint is the WIDER of the two — a
# record that can be retracted without ever having been demoted in the read path spends its entire
# accumulating life evicting the alternatives in its direction group and then vanishes.


def _negatives(n, outcome="failed", actor=""):
    value = {}
    for i in range(n):
        value = record_attestation(value, outcome, actor=actor or "box%d" % i)
    return value


def test_one_negative_is_not_enough_and_two_are():
    assert should_retire(_negatives(1)) == ""
    assert "policy threshold is 2" in should_retire(_negatives(2))


def test_not_reproduced_and_failed_are_added_together():
    """Two different ways of not working are still two attempts that did not work."""
    value = record_attestation(record_attestation({}, "failed"), "not_reproduced")
    assert should_retire(value)


def test_one_validation_anywhere_vetoes_the_whole_thing():
    """A ratio would retire this on the sixth loss. A store this small cannot afford that: one
    reproduction means the record is right about something, and the losses after it are a statement
    about the boxes it was replayed on."""
    value = record_attestation(_negatives(5), "validated")
    assert should_retire(value) == ""


def test_inapplicable_reads_never_retire_anything():
    value = {}
    for _ in range(6):
        value = record_attestation(value, "inapplicable")
    assert should_retire(value) == ""


def test_the_threshold_is_overridable_per_sweep():
    assert should_retire(_negatives(2), threshold=3) == ""
    assert should_retire(_negatives(3), threshold=3)
    assert should_retire(_negatives(1), threshold=0)          # clamped to 1, not to "always"
    assert should_retire({}, threshold=0) == ""


def test_the_hint_is_never_the_later_signal():
    """Whatever fires `should_retire` must already have fired `retire_hint`, at every count."""
    for n in range(0, 5):
        for outcome in ("failed", "not_reproduced"):
            value = _negatives(n, outcome)
            if should_retire(value):
                assert retire_hint(value), "retired at %d %s with no hint" % (n, outcome)


def test_a_reprieve_recorded_as_a_validation_stops_the_answered_negatives_counting_again():
    """Retract, re-bench, win, un-retire, retract, ... The negatives that got a record retracted
    have been answered by the win that lifted it. `e2e_store.py:_carrying_ledger` records that win
    as a validation, and the veto every other caller already relies on does the rest — no second
    notion of "which attempts still count" to keep in step with this one."""
    value = _negatives(2)
    assert should_retire(value)
    value = record_attestation(value, "validated", actor="the box that re-benched it")
    assert should_retire(value) == ""
    assert retire_hint(value) == ""          # and it stops being demoted in the read path too
