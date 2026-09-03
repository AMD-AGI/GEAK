#!/usr/bin/env python3
"""Attestation: counting what happened when a stored record was actually TRIED on a box.

A record's `validated` flag is a judgement its own writer made about its own measurement, once, at
write time. It answers "did the run that produced this number clear a gate", and it can never
answer the question a reader is really asking three months later: **has anyone else since pulled
this record out of the store, run it, and had it work.** That second question is the one that
decides whether a record is worth keeping, and nothing in the schema recorded it — so a config that
has been recalled six times and failed six times looked exactly like one nobody had ever tried.

This module adds that ledger, under `value.attestations`, in one vocabulary both lanes use:

    recalls          how many times this record was pulled and actually PUT ON A BOX
    validations      of those, how many reproduced a win   <- the retire signal
    failures         of those, how many ran but did not win
    not_reproduced   of those, how many could not be made to run at all
    inapplicable     of those, how many could not be applied to THIS box's baseline at all — a
                     verdict on the pairing, not on the record, and so excluded from the retire
                     arithmetic below
    last_outcome / last_at / last_by     the most recent attempt, for a reader in a hurry
    history          the last HISTORY_LIMIT attempts with their evidence

`recalls` counts ATTEMPTS ON HARDWARE, not reads. A record that a resolve listed in its top-N and
nobody benched has learned nothing about itself, and counting that as a recall would make the ratio
`validations / recalls` — the only number a retire pass can act on — decay for records that were
never actually doubted.

WHY THIS IS NOT RETRACTION, even though both rewrite a session document in place. Retraction
declares a record FALSE and therefore has to move it: it zeroes the ranking scalars and re-points
the champion, because a flag alone is inert against a reader that ranks on `knowledge.<metric>`
(see kb/retract.py). An attestation declares nothing. One failure on one box is evidence, not a
verdict — the box may have a different ROCm, the flag may have been renamed upstream, the workload
may be off the point the record was tuned for. So this module touches NO ranking scalar and NEVER
re-points the champion. Deciding that an accumulated ledger has become damning, and calling
`retract_session` on the strength of it, is a separate act by a separate caller, which is exactly
the separation that makes the counters trustworthy as input to it.

Rewrites go through the same `mode="replace"` path both planes' `write()` already use, and reuse
`kb/retract.py:existing_files` so a local rewrite re-supplies the artifacts it would otherwise
delete.
"""

from __future__ import annotations

import time

from kb.retract import existing_files
from kb.store_local import KBStoreError

# The four things that can happen when a record is taken off the shelf and run. They are counted
# separately because they mean opposite things to a retire pass: `failed` says the claim did not
# hold HERE (the record may still be right elsewhere), while `not_reproduced` says the record could
# not even be applied — a much stronger signal that it is missing something it promised.
#
# `inapplicable` splits a case that used to be spelled `not_reproduced` and does not belong there.
# A stored e2e config is a WHOLE launch configuration, and it is replayed on top of whatever
# baseline configuration the reading run was handed — under Hyperloom, a full flag string this run
# does not own. When the two pin the same knob to different values, the record could not be applied
# HERE for a reason that says nothing at all about the record: it may be perfectly right on the box
# it was written for and on the next box that reads it. Counting that as `not_reproduced` made the
# environment, not the record, the thing being judged — and two such reads were enough to put a
# retire hint on a record nobody had ever found anything wrong with.
VALIDATED = "validated"
FAILED = "failed"
NOT_REPRODUCED = "not_reproduced"
INAPPLICABLE = "inapplicable"
OUTCOMES = (VALIDATED, FAILED, NOT_REPRODUCED, INAPPLICABLE)

# `history` is bounded because it rides inside every knowledge document, and the documents are
# fetched one-per-candidate to rank a page. An unbounded audit log would make every read of a
# popular record slower for a benefit nobody has asked for; the counters are the durable part.
HISTORY_LIMIT = 20

# What a history entry may carry, whitelisted so an over-eager caller cannot grow the documents
# without meaning to. The two lanes measure different things — e2e in tokens per second against a
# baseline, kernels in an isolated speedup ratio — and both spellings are here rather than one
# generic `measurement`, because a reader that cannot tell 1.8 tok/s from 1.8x has learned nothing.
_EVIDENCE_KEYS = ("measured_tok_s", "baseline_tok_s", "delta_pct", "measured_speedup", "parity",
                  "note", "canonical_id", "workload")


# Every counter that is a BUCKET of `recalls`, in one place, because four call sites have to agree
# on the list and three of them fail silently when they disagree: a key missing from
# `attestations_of` reads as 0 forever, and one missing from `carry_attestations`'s emptiness test
# drops a whole ledger on the next rewrite.
BUCKETS = ("validations", "failures", "not_reproduced", "inapplicable")


def empty_attestations() -> dict:
    return {"recalls": 0, "validations": 0, "failures": 0, "not_reproduced": 0, "inapplicable": 0,
            "last_outcome": "", "last_at": "", "last_by": "", "history": []}


def _counter(value) -> int:
    """A count from a document we did not write. Anything unusable reads as 0, never as a crash."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return 0


def attestations_of(value) -> dict:
    """The ledger inside a record's `value`, normalized. Always safe to read fields off."""
    raw = value.get("attestations") if isinstance(value, dict) else None
    if not isinstance(raw, dict):
        return empty_attestations()
    ledger = empty_attestations()
    for key in ("recalls",) + BUCKETS:
        ledger[key] = _counter(raw.get(key))
    for key in ("last_outcome", "last_at", "last_by"):
        ledger[key] = str(raw.get(key) or "")
    history = raw.get("history")
    ledger["history"] = [h for h in history if isinstance(h, dict)][-HISTORY_LIMIT:] \
        if isinstance(history, list) else []
    return ledger


def carry_attestations(previous_value, fresh_value: dict) -> dict:
    """Move an existing record's ledger onto the document that is about to REPLACE it.

    Both lanes content-address their session ids off the thing being recorded and not off its
    measurement, so re-benching one configuration deliberately lands on the SAME session id and
    rewrites it. Without this, every re-measurement would silently reset the record's whole
    validation history to zero — and the reset would be invisible, because the new document looks
    perfectly well-formed. Returns `fresh_value` unchanged when there is nothing to carry.
    """
    if not isinstance(previous_value, dict) or "attestations" not in previous_value:
        return fresh_value
    ledger = attestations_of(previous_value)
    if not any(ledger[k] for k in ("recalls",) + BUCKETS):
        return fresh_value
    fresh_value["attestations"] = ledger
    return fresh_value


def record_attestation(value: dict, outcome: str, *, actor: str = "", evidence=None,
                       when: str = "") -> dict:
    """`value` with one attempt counted onto its ledger. Pure — returns a new dict.

    Split out from the store call so a dry run can show the exact ledger that would land, and so
    the kernel lane can apply the identical arithmetic to its on-disk meta.yaml without going
    through a KB plane at all.
    """
    outcome = str(outcome or "").strip().lower()
    if outcome not in OUTCOMES:
        raise KBStoreError("unknown attestation outcome %r; expected one of %s"
                           % (outcome, ", ".join(OUTCOMES)))
    ledger = attestations_of(value)
    stamp = when or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    # `inapplicable` increments `recalls` like the rest: the invariant that `recalls` is the sum of
    # its buckets is what lets a reader check a ledger for consistency, and hiding an attempt would
    # make a record that keeps failing to fit anywhere look untouched. It is subtracted where it
    # actually matters instead — see `retire_hint`.
    ledger["recalls"] += 1
    ledger[{VALIDATED: "validations", FAILED: "failures",
            NOT_REPRODUCED: "not_reproduced", INAPPLICABLE: "inapplicable"}[outcome]] += 1
    ledger.update({"last_outcome": outcome, "last_at": stamp, "last_by": str(actor or "")})
    entry = {"at": stamp, "outcome": outcome}
    if actor:
        entry["by"] = str(actor)
    for key in _EVIDENCE_KEYS:
        item = (evidence or {}).get(key) if isinstance(evidence, dict) else None
        if item not in (None, "", [], {}):
            entry[key] = item
    ledger["history"] = (ledger["history"] + [entry])[-HISTORY_LIMIT:]
    updated = dict(value if isinstance(value, dict) else {})
    updated["attestations"] = ledger
    return updated


def retire_hint(value) -> str:
    """Why a curation pass might want to look at this record, or "". Advisory, never enforced.

    Deliberately conservative and deliberately not a boolean: this is read by an agent prompt and
    by a human running a curation sweep, and both need to know WHICH pattern fired. Nothing in the
    read path filters on it — a record with a hint is still offered, still ranked, still adoptable.
    """
    ledger = attestations_of(value)
    # Attempts that TESTED THE RECORD. An `inapplicable` read never got as far as putting the
    # stored configuration on the box — the box's own baseline pinned a knob the record also pins,
    # and the pair is what failed. Leaving those in the denominator meant a record could be retired
    # for being read on the wrong machines, which is the opposite of what the counter is for.
    tried = ledger["recalls"] - ledger["inapplicable"]
    if tried <= 0:
        return ""
    if ledger["not_reproduced"] >= 2 and not ledger["validations"]:
        return ("%d attempts could not reproduce it at all and none ever succeeded — the record is "
                "probably missing something it promised" % ledger["not_reproduced"])
    if tried >= 3 and not ledger["validations"]:
        return ("tried %d times, never reproduced a win" % tried)
    return ""


def attested_document(knowledge: dict, outcome: str, *, actor: str = "", evidence=None) -> dict:
    """A copy of `knowledge` with the attempt counted. Pure — writes nothing.

    Every ranking scalar is left exactly as it was; see the module docstring for why an attestation
    must not move a record in the ordering the way a retraction must.
    """
    if not isinstance(knowledge, dict):
        raise KBStoreError("knowledge is not an object")
    document = dict(knowledge)
    value = document.get("value") if isinstance(document.get("value"), dict) else {}
    document["value"] = record_attestation(dict(value), outcome, actor=actor, evidence=evidence)
    return document


def attest_session(store, canonical_id: str, session_id: str, outcome: str, *, actor: str = "",
                   evidence=None, apply: bool = True):
    """Count one attempt against one session on one plane. Never raises for a missing record.

    Mirrors `retract_session`'s report shape so a caller that already handles one can handle the
    other: a rung the write never reached reports `found: false` rather than failing the run.
    """
    report = {"canonical_id": canonical_id, "session_id": session_id, "outcome": outcome,
              "found": False, "rewritten": False, "attestations": {}, "error": ""}
    knowledge = store.get_session(canonical_id, session_id)
    if not isinstance(knowledge, dict):
        report["error"] = "no such session on this plane (nothing to attest)"
        return report
    report["found"] = True
    try:
        document = attested_document(knowledge, outcome, actor=actor, evidence=evidence)
    except KBStoreError as e:
        report["error"] = str(e)
        return report
    report["attestations"] = document["value"]["attestations"]
    report["retire_hint"] = retire_hint(document["value"])
    if not apply:
        report["would_write"] = document
        return report
    try:
        store.write(canonical_id, session_id, document,
                    existing_files(store, canonical_id, session_id))
        report["rewritten"] = True
    except (KBStoreError, OSError) as e:
        report["error"] = "%s: %s" % (type(e).__name__, str(e)[:160])
    return report


def attestation_ok(reports, applied: bool) -> bool:
    """Did the attestation land, judged over every (page, plane) it visited.

    Same rule as `retraction_ok`: a rung that does not hold the record has nothing to count and is
    not a failure, but finding it nowhere at all means the session id is wrong and the caller's
    verdict was recorded against nothing.
    """
    if not applied:
        return True
    found = [r for r in reports if r.get("found")]
    return bool(found) and all(r.get("rewritten") for r in found)


__all__ = ["FAILED", "HISTORY_LIMIT", "NOT_REPRODUCED", "OUTCOMES", "VALIDATED",
           "attest_session", "attestation_ok", "attestations_of", "attested_document",
           "carry_attestations", "empty_attestations", "record_attestation", "retire_hint"]
