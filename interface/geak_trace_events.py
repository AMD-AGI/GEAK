#!/usr/bin/env python3
"""Explicit linkage contracts for the GEAK execution tracker.

Two relationships cannot be recovered from a transcript after the fact, because
nothing in the recorded data states them:

  * **result_supplied_to_dispatch** -- that a producer agent's returned result
    was supplied as input when a later agent was dispatched;
  * **spawn / return** -- that one agent invocation spawned another, and which
    return belongs to that spawn.

Byte-identical text in a producer's result and a consumer's prompt does NOT
establish the first: it shows only that both records contain those bytes. The
same text is returned by more than one producer in real runs, a path can come
independently from workflow arguments, and ordering does not disambiguate any of
that. A tool named ``Task``/``Agent``, or a ``spawnDepth`` level, does not
establish the second either: neither identifies WHICH parent, nor joins a child's
return to the event that started it.

So this module defines what a producer of those events must record for the edge
to exist, validates such events, and builds edges ONLY from validated ones.
Anything unknown or contradictory stays incomplete and is reported as such --
never completed by inference.

SCOPE CLAIM, stated precisely: this is an ADAPTER AND SCHEMA plus its fixtures.
Fixture support establishes that the tracker consumes these events correctly.
It does NOT establish that the native runtime emits them. Historical runs get
only the relationships their own records already support, and no legacy link is
ever synthesised here.
"""

import json
import os

SCHEMA = "geak.trace.events/1"

#: Event kinds this contract accepts.
RESULT_SUPPLIED = "result_supplied_to_dispatch"
SPAWN = "agent_spawn"
SPAWN_RETURN = "agent_spawn_return"

#: How a value reached the consumer. ``literal`` means the exact recorded bytes
#: were forwarded. ``transformed`` means the producer of the event states a
#: transformation occurred; without a recorded description that stays unknown --
#: it is never reconstructed by comparing texts.
FORWARD_LITERAL = "literal"
FORWARD_TRANSFORMED = "transformed"
FORWARD_UNKNOWN = "unknown"

_REQUIRED = {
    # event_id is required: without a stable identity a replay cannot be told
    # from a second, genuinely distinct transfer.
    RESULT_SUPPLIED: ("event_id", "producer_invocation_id", "producer_result_ref",
                      "consumer_invocation_id", "consumer_input_ref"),
    # spawn_tool_call_id is required: the agreed contract is parent + the ACTUAL
    # spawning tool event + child. A spawn with no tool reference is not joined.
    SPAWN: ("parent_invocation_id", "spawn_event_id", "spawn_tool_call_id",
            "child_invocation_id"),
    # attempt_id is required: repetition is not evidence of a retry.
    SPAWN_RETURN: ("spawn_event_id", "child_invocation_id", "attempt_id"),
}


class EventError(ValueError):
    """An event that does not meet the contract; never silently downgraded."""


def validate(event):
    """Return ``(kind, normalized)`` for a well-formed event, else raise.

    Validation is deliberately strict: a partially-specified linkage is an
    incomplete record, not a weaker edge. Callers collect the errors and report
    them as coverage gaps.
    """
    if not isinstance(event, dict):
        raise EventError("event is not an object")
    kind = event.get("type")
    if kind not in _REQUIRED:
        raise EventError("unknown event type %r" % (kind,))
    missing = [f for f in _REQUIRED[kind]
               if not isinstance(event.get(f), str) or not event[f].strip()]
    if missing:
        raise EventError("%s missing required field(s): %s"
                         % (kind, ", ".join(missing)))

    out = {k: v for k, v in event.items() if v is not None}
    if kind == RESULT_SUPPLIED:
        forwarding = out.get("forwarding") or FORWARD_UNKNOWN
        if forwarding not in (FORWARD_LITERAL, FORWARD_TRANSFORMED, FORWARD_UNKNOWN):
            raise EventError("invalid forwarding %r" % (forwarding,))
        # A transformation without a recorded description is UNKNOWN provenance,
        # not a described transformation. We do not infer what changed.
        if forwarding == FORWARD_TRANSFORMED and not out.get("transformation"):
            out["forwarding"] = FORWARD_TRANSFORMED
            out["transformation_known"] = False
        else:
            out["transformation_known"] = bool(out.get("transformation"))
        out["forwarding"] = forwarding
    if kind == SPAWN_RETURN:
        status = out.get("status") or "unknown"
        if status not in ("returned", "error", "cancelled", "unknown"):
            raise EventError("invalid spawn return status %r" % (status,))
        out["status"] = status
    return kind, out


def read_events(path):
    """Read an events JSONL file, returning ``(validated, problems)``.

    A malformed or partial line is a reported problem, never a silent skip: the
    whole point is that coverage of these edges is auditable.
    """
    validated, problems = [], []
    if not path or not os.path.exists(path):
        return validated, problems
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except ValueError as exc:
                    problems.append({"line": lineno, "error": "unparsable: %s" % exc})
                    continue
                try:
                    kind, norm = validate(raw)
                except EventError as exc:
                    problems.append({"line": lineno, "error": str(exc)})
                    continue
                norm["_kind"] = kind
                validated.append(norm)
    except OSError as exc:
        problems.append({"line": None, "error": "unreadable: %s" % exc})
    return validated, problems


def _same(a, b, fields):
    return all(a.get(f) == b.get(f) for f in fields)


def build_edges(events, known_invocations=None):
    """Turn validated events into edges, reporting what could not be joined.

    Joins are by stable identity and every identity-defining field is checked.
    A contradiction INVALIDATES the affected join rather than selecting the
    first or last record seen: a conflicting parent, child, destination or
    outcome means we do not know the relationship, which is not the same as
    knowing one of the two candidates. Replays of an identical record are
    idempotent.

    ``known_invocations`` is the set of invocation ids the trace observed. An
    EMPTY set means nothing is joinable -- it must not disable the check and let
    edges be built to invocations that were never seen.
    """
    known = set(known_invocations or ())
    edges, unresolved = [], []
    conflicted = set()

    def unjoinable(ev, reason):
        unresolved.append({"event": ev, "reason": reason})

    def refs_present(ev, *ids):
        absent = [r for r in ids if r not in known]
        return absent

    # ---- spawns: identity is spawn_event_id; every field must agree ----------
    spawns = {}
    for ev in events:
        if ev.get("_kind") != SPAWN:
            continue
        key = ev["spawn_event_id"]
        prior = spawns.get(key)
        if prior is None:
            spawns[key] = ev
            continue
        if _same(prior, ev, ("parent_invocation_id", "child_invocation_id",
                             "spawn_tool_call_id")):
            continue  # exact replay: idempotent
        conflicted.add(key)
        unjoinable(ev, "conflicting spawn record for spawn_event_id %s "
                       "(parent/child/tool disagree); join invalidated" % key)

    # ---- returns: identity is (spawn_event_id, attempt_id) ------------------
    returns = {}
    for ev in events:
        if ev.get("_kind") != SPAWN_RETURN:
            continue
        key = (ev["spawn_event_id"], ev["attempt_id"])
        prior = returns.get(key)
        if prior is None:
            returns[key] = ev
            continue
        if _same(prior, ev, ("child_invocation_id", "status", "result_ref", "error")):
            continue  # exact replay: idempotent
        conflicted.add(ev["spawn_event_id"])
        unjoinable(ev, "contradictory return for attempt %s of spawn %s; "
                       "join invalidated" % (ev["attempt_id"], ev["spawn_event_id"]))

    # ---- transfers: identity is event_id ------------------------------------
    transfers = {}
    for ev in events:
        if ev.get("_kind") != RESULT_SUPPLIED:
            continue
        key = ev["event_id"]
        prior = transfers.get(key)
        if prior is None:
            transfers[key] = ev
            continue
        if _same(prior, ev, ("producer_invocation_id", "producer_result_ref",
                             "consumer_invocation_id", "consumer_input_ref",
                             "forwarding", "transformation")):
            continue  # exact replay: idempotent
        conflicted.add(key)
        unjoinable(ev, "conflicting transfer record for event_id %s; "
                       "join invalidated" % key)

    for key, ev in transfers.items():
        if key in conflicted:
            continue
        absent = refs_present(ev, ev["producer_invocation_id"],
                              ev["consumer_invocation_id"])
        if absent:
            unjoinable(ev, "invocation(s) not present in this trace: %s"
                           % ", ".join(absent))
            continue
        edges.append({
            "type": "result_supplied_to_dispatch",
            "from": "agent:%s" % ev["producer_invocation_id"],
            "to": "agent:%s" % ev["consumer_invocation_id"],
            "event_id": key,
            "provenance": "recorded_event", "proven": True,
            "producer_result_ref": ev["producer_result_ref"],
            "consumer_input_ref": ev["consumer_input_ref"],
            "forwarding": ev.get("forwarding", FORWARD_UNKNOWN),
            "transformation": ev.get("transformation"),
            "transformation_known": ev.get("transformation_known", False),
        })

    for key, ev in spawns.items():
        if key in conflicted:
            continue
        parent, child = ev["parent_invocation_id"], ev["child_invocation_id"]
        absent = refs_present(ev, parent, child)
        if absent:
            unjoinable(ev, "invocation(s) not present in this trace: %s"
                           % ", ".join(absent))
            continue
        attempts, mismatched = [], False
        for (skey, attempt_id), ret in sorted(returns.items()):
            if skey != key:
                continue
            # A return must name the SAME child as its spawn.
            if ret["child_invocation_id"] != child:
                mismatched = True
                unjoinable(ret, "return names child %s but spawn %s declares child "
                                "%s; join invalidated"
                                % (ret["child_invocation_id"], key, child))
                continue
            attempts.append({"attempt_id": attempt_id, "status": ret["status"],
                             "result_ref": ret.get("result_ref"),
                             "error": ret.get("error")})
        if mismatched:
            continue
        edges.append({
            "type": "agent_spawn",
            "from": "agent:%s" % parent, "to": "agent:%s" % child,
            "spawn_event_id": key,
            "spawn_tool_call_id": ev["spawn_tool_call_id"],
            "provenance": "recorded_event", "proven": True,
            "attempts": attempts,
            "return_status": ("unmatched" if not attempts
                              else attempts[-1]["status"]),
        })
        if not attempts:
            unjoinable(ev, "spawn has no matching return event (child may still be "
                           "running, or the return was never recorded)")

    orphans = [r for (skey, _), r in returns.items() if skey not in spawns]
    for ret in orphans:
        unjoinable(ret, "return references a spawn_event_id with no spawn event")

    # Identity keys of every relationship a contradiction invalidated. Callers
    # must keep these unusable: a claim does not become true again because a
    # later pass no longer sees the contradicting record.
    invalidated_keys = []
    for key in conflicted:
        for ev in list(transfers.values()) + list(spawns.values()):
            if ev.get("event_id") == key:
                invalidated_keys.append(
                    ["result_supplied_to_dispatch",
                     "agent:%s" % ev.get("producer_invocation_id"),
                     "agent:%s" % ev.get("consumer_invocation_id"), key])
            elif ev.get("spawn_event_id") == key:
                invalidated_keys.append(
                    ["agent_spawn", "agent:%s" % ev.get("parent_invocation_id"),
                     "agent:%s" % ev.get("child_invocation_id"), key])

    stats = {
        "result_supplied_edges": sum(1 for e in edges
                                     if e["type"] == "result_supplied_to_dispatch"),
        "spawn_edges": sum(1 for e in edges if e["type"] == "agent_spawn"),
        "unresolved": len(unresolved),
        "orphan_returns": len(orphans),
        "conflicts": len(conflicted),
        "spawns_without_return": sum(1 for e in edges if e["type"] == "agent_spawn"
                                     and e["return_status"] == "unmatched"),
    }
    return edges, unresolved, stats, invalidated_keys


def attach(trace, events_path):
    """Attach recorded-linkage edges to a trace, with explicit coverage.

    Absence of an events file is the normal case for every run recorded so far,
    and is reported as such -- NOT as "no relationships exist".
    """
    known = {a.get("agent_id") for a in (trace.get("agents") or [])}
    events, problems = read_events(events_path)
    if not events and not problems:
        trace.setdefault("run", {})["linkage"] = {
            "source": events_path, "present": False,
            "note": ("No recorded linkage events for this run. Result-transfer and "
                     "agent-spawn edges are therefore NOT SHOWN, which means they "
                     "were not recorded -- not that they did not occur."),
        }
        return trace

    edges, unresolved, stats, invalidated = build_edges(events, known)
    trace.setdefault("edges", []).extend(edges)
    trace.setdefault("run", {})["linkage"] = {
        "source": events_path, "present": True, "events_read": len(events),
        "malformed": len(problems), "stats": stats,
        "invalidated": invalidated,
        "complete": not problems and not unresolved,
    }
    trace["run"]["linkage"]["unresolved"] = unresolved[:50]
    trace["run"]["linkage"]["problems"] = problems[:50]
    if problems or unresolved:
        trace.setdefault("warnings", []).append(
            "Recorded linkage is INCOMPLETE: %d malformed event(s), %d unresolved "
            "reference(s). Missing edges mean unrecorded, not absent."
            % (len(problems), len(unresolved)))
    return trace
