#!/usr/bin/env python3
"""Retraction for the knowledge base: how a record that turned out to be false is taken back.

The KB Store service exposes no DELETE. Nothing written to it can be removed, and both lanes'
writers say so in their prompts, deliberately. That is not the same as "a wrong record is forever
authoritative", and the difference is this module.

`put_knowledge(..., mode="replace")` rewrites a whole session document in place, and both planes'
`write()` already use that mode. A session id is a deterministic digest of the record's content, so
any record we wrote can be addressed again later without having kept a note of it. Retraction is
therefore a REWRITE to a tombstone, not a deletion: the page still holds a session with that id,
but the session now says "this was wrong, here is why".

**A flag alone would be inert.** Ranking on both planes reads a single top-level scalar
(`knowledge.<metric>`), and `sessions/top?metric=` is what a reader pages through. A document whose
`retained` is false but whose `speedup` still says 1.9 keeps its place at the head of the list for
every consumer that has not been taught the flag — including the service's own rollup. So a
retraction has to do three things at once, and doing two of them is worse than doing none:

  1. mark the document (`retained: false` + a `retired_reason` a human can act on),
  2. drop the ranking scalars below every promote floor, so it sinks in the ordering itself,
  3. re-point the identity's champion, because the champion pointer is a separate object that does
     not re-derive itself from the sessions it points at.

Step 3 is the one that surprises. Zeroing a document does not un-promote it; the champion record
keeps the score it was promoted with, and `champion_speedup()` keeps returning that number as the
bar every future candidate must clear. A retracted champion left in place quietly blocks the page.

What retraction does NOT do: it does not touch the artifacts. The bytes are what let someone audit
the claim afterwards, which is precisely what you want when a record has just been called false.
"""

from __future__ import annotations

import os
import time

from kb_store_local import Candidate, KBStoreError, finite_speedup


# Written into `value.lifecycle`. The other two values in circulation are "active" (reproduced) and
# "candidate" (recorded but unreproduced); this is a third state, not a degree of the first two.
RETRACTED = "retracted"


def is_retired(value) -> bool:
    """Whether a record's `value` says it has been taken back.

    Two spellings because two producers exist. `retained is False` is the kernel lane's local
    curation flag (note `is False`, not falsy: a record that never set the key is not retired, and
    `None` is "unstated", which is the common case). `retired_reason` is what retraction below
    writes, and a non-empty reason is itself the claim.
    """
    if not isinstance(value, dict):
        return False
    return value.get("retained") is False or bool(value.get("retired_reason"))


def retracted_document(knowledge: dict, reason: str, metrics, actor: str = "") -> dict:
    """A copy of `knowledge` rewritten into a tombstone. Pure — writes nothing.

    Kept separate from the store call so a dry run can show the exact document that would land.
    """
    if not isinstance(knowledge, dict):
        raise KBStoreError("knowledge is not an object")
    reason = str(reason or "").strip()
    if not reason:
        raise KBStoreError("a retraction needs a reason: it is the only thing a future reader has "
                           "to judge whether the retraction itself was right")
    document = dict(knowledge)
    value = dict(document.get("value") if isinstance(document.get("value"), dict) else {})
    # Preserve what the record CLAIMED before we zero the ranking copies. A tombstone that has
    # forgotten the number it was retracted for cannot be reviewed, only trusted.
    withdrawn = {m: document.get(m) for m in metrics if document.get(m) is not None}
    if withdrawn:
        value.setdefault("withdrawn_scores", withdrawn)
    value.update({"retained": False, "retired_reason": reason, "validated": False,
                  "lifecycle": RETRACTED,
                  "retracted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    if actor:
        value["retracted_by"] = str(actor)
    document["value"] = value
    for metric in metrics:
        # 0.0, not deleted. A missing scalar makes the record unrankable, and an unrankable record
        # on some routes sorts as null-last and on others is dropped from the page entirely — which
        # looks like the delete this store does not have, right up until a route that keeps it
        # surfaces the record with no score and no explanation.
        document[metric] = 0.0
    return document


def retraction_ok(reports, applied: bool) -> bool:
    """Did the retraction succeed, judged over every (page, plane) it visited.

    A page that does not hold the record is NOT a failure — a ladder rung the write never reached,
    or a plane that only ever saw half the ladder, has nothing to take back and reporting it as an
    incomplete retraction would train the caller to ignore the field. What IS a failure: finding the
    record and not rewriting it, or finding it nowhere at all, which means the session id is wrong
    and the record the caller wants to retract is still live somewhere they have not looked.
    """
    if not applied:
        return True
    found = [r for r in reports if r.get("found")]
    return bool(found) and all(r.get("rewritten") for r in found)


def _existing_files(store, canonical_id: str, session_id: str):
    """{relative path: absolute source} for a LOCAL session, or None for a remote one.

    The two planes lose artifacts differently on a rewrite and this is the whole reason the caller
    cannot just pass `files=None`. `LocalKBStore.write()` stages a fresh directory and swaps it in,
    so omitting the files DELETES them. `RemoteKBStore.write()` only calls `put_files` when it is
    given some, and the manifest it does not touch survives. So: re-supply on local, stay silent on
    remote (re-uploading identical bytes would be the only alternative, and it can fail).
    """
    lister = getattr(store, "session_files", None)
    if not callable(lister):
        return None
    root = os.path.join(store.session_dir(canonical_id, session_id), "files")
    return {rel: os.path.join(root, *rel.split("/")) for rel in lister(canonical_id, session_id)}


def _replacement_champion(store, canonical_id: str, session_id: str, metric: str, scan: int):
    """(session_id, score) of the best surviving candidate, or (None, None).

    Surviving means: not the record being retracted, not already retired, and carrying a real score
    under THIS rung's metric. The last clause matters because a coarse rung ranks on `speedup` and
    the exact rung on `throughput_tok_s`; promoting a session whose score came from the other
    metric would write a champion the store then compares future candidates against incomparably.
    """
    try:
        found = store.candidates(canonical_id, limit=max(1, int(scan)))
    except Exception:
        return None, None
    for candidate in found:
        if candidate.session_id == session_id:
            continue
        value = candidate.value if isinstance(candidate, Candidate) else {}
        if is_retired(value):
            continue
        score = finite_speedup((candidate.knowledge or {}).get(metric))
        if score is None:
            continue
        return candidate.session_id, score
    return None, None


def retract_session(store, canonical_id: str, session_id: str, reason: str, metric: str,
                    *, extra_metrics=(), actor: str = "", scan: int = 8, apply: bool = True):
    """Retract one session on one plane. Returns a report dict; never raises for a missing record.

    `metric` is the rung's ranking metric — the one the champion is compared under. `extra_metrics`
    are the other scalars the same document carries (the e2e document holds both `throughput_tok_s`
    and `speedup`, and a rewrite that zeroed only one of them would leave the record ranked exactly
    where it was on half the ladder).
    """
    report = {"canonical_id": canonical_id, "session_id": session_id, "found": False,
              "rewritten": False, "champion_was": "", "champion_now": "", "error": ""}
    metrics = [metric] + [m for m in extra_metrics if m and m != metric]
    knowledge = store.get_session(canonical_id, session_id)
    if not isinstance(knowledge, dict):
        report["error"] = "no such session on this plane (nothing to retract)"
        return report
    report["found"] = True
    report["was_retired"] = is_retired(knowledge.get("value"))
    try:
        document = retracted_document(knowledge, reason, metrics, actor=actor)
    except KBStoreError as e:
        report["error"] = str(e)
        return report
    report["champion_was"] = str(store.champion(canonical_id).get("session_id") or "")
    if not apply:
        report["would_write"] = document
        return report
    try:
        store.write(canonical_id, session_id, document, _existing_files(store, canonical_id,
                                                                        session_id))
        report["rewritten"] = True
    except (KBStoreError, OSError) as e:
        report["error"] = "%s: %s" % (type(e).__name__, str(e)[:160])
        return report
    if report["champion_was"] and report["champion_was"] != session_id:
        report["champion_now"] = report["champion_was"]      # someone else already holds the slot
        return report
    successor, score = _replacement_champion(store, canonical_id, session_id, metric, scan)
    if successor is None:
        # Nothing left to crown. Re-point at the tombstone with a zero so the pointer stops
        # advertising a win and stops acting as a floor — leaving the old score there would make
        # every future candidate have to beat a number we just declared false.
        successor, score = session_id, 0.0
    try:
        store.promote(canonical_id, successor, float(score))
        report["champion_now"] = successor
    except Exception as e:
        report["error"] = "champion not re-pointed: %s: %s" % (type(e).__name__, str(e)[:120])
    return report
