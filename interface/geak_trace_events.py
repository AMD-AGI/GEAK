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
    RESULT_SUPPLIED: ("producer_invocation_id", "producer_result_ref",
                      "consumer_invocation_id", "consumer_input_ref"),
    SPAWN: ("parent_invocation_id", "spawn_event_id", "child_invocation_id"),
    SPAWN_RETURN: ("spawn_event_id", "child_invocation_id"),
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


def build_edges(events, known_invocations=None):
    """Turn validated events into edges, reporting what could not be joined.

    ``known_invocations`` is the set of invocation ids the trace actually
    observed. An event referring to an invocation that is not in the trace is
    left UNRESOLVED rather than creating a node for it -- a dangling reference is
    a coverage gap, not evidence of a hidden agent.

    Returns ``(edges, unresolved, stats)``.
    """
    known = set(known_invocations or ())
    spawns, returns = {}, {}
    edges, unresolved = [], []

    for ev in events:
        kind = ev.get("_kind")
        if kind == SPAWN:
            key = ev["spawn_event_id"]
            if key in spawns:
                # Duplicate/replayed record: same id must describe the same spawn.
                if spawns[key]["child_invocation_id"] != ev["child_invocation_id"]:
                    unresolved.append({"event": ev, "reason":
                                       "conflicting child for spawn_event_id"})
                continue
            spawns[key] = ev
        elif kind == SPAWN_RETURN:
            returns.setdefault(ev["spawn_event_id"], []).append(ev)

    for ev in events:
        if ev.get("_kind") != RESULT_SUPPLIED:
            continue
        missing = [r for r in (ev["producer_invocation_id"], ev["consumer_invocation_id"])
                   if known and r not in known]
        if missing:
            unresolved.append({"event": ev,
                               "reason": "invocation(s) not present in this trace: %s"
                                         % ", ".join(missing)})
            continue
        edges.append({
            "type": "result_supplied_to_dispatch",
            "from": "agent:%s" % ev["producer_invocation_id"],
            "to": "agent:%s" % ev["consumer_invocation_id"],
            "provenance": "recorded_event",
            "proven": True,
            "producer_result_ref": ev["producer_result_ref"],
            "consumer_input_ref": ev["consumer_input_ref"],
            "forwarding": ev.get("forwarding", FORWARD_UNKNOWN),
            "transformation": ev.get("transformation"),
            "transformation_known": ev.get("transformation_known", False),
            "event_id": ev.get("event_id"),
        })

    for key, ev in spawns.items():
        parent, child = ev["parent_invocation_id"], ev["child_invocation_id"]
        absent = [r for r in (parent, child) if known and r not in known]
        if absent:
            unresolved.append({"event": ev,
                               "reason": "invocation(s) not present in this trace: %s"
                                         % ", ".join(absent)})
            continue
        rets = returns.get(key) or []
        # Several returns for one spawn are retries/attempts; each keeps its own
        # attempt identity rather than being collapsed into "the" return.
        attempts = [{"attempt_id": r.get("attempt_id"), "status": r["status"],
                     "result_ref": r.get("result_ref"), "error": r.get("error")}
                    for r in rets]
        edges.append({
            "type": "agent_spawn",
            "from": "agent:%s" % parent,
            "to": "agent:%s" % child,
            "provenance": "recorded_event",
            "proven": True,
            "spawn_event_id": key,
            "spawn_tool_call_id": ev.get("spawn_tool_call_id"),
            "attempts": attempts,
            "return_status": ("unmatched" if not attempts else
                              attempts[-1]["status"]),
        })
        if not attempts:
            unresolved.append({"event": ev, "reason":
                               "spawn has no matching return event (child may still "
                               "be running, or the return was never recorded)"})

    orphan_returns = [r for key, rs in returns.items() if key not in spawns for r in rs]
    for ret in orphan_returns:
        unresolved.append({"event": ret, "reason":
                           "return references a spawn_event_id with no spawn event"})

    stats = {
        "result_supplied_edges": sum(1 for e in edges
                                     if e["type"] == "result_supplied_to_dispatch"),
        "spawn_edges": sum(1 for e in edges if e["type"] == "agent_spawn"),
        "unresolved": len(unresolved),
        "orphan_returns": len(orphan_returns),
        "spawns_without_return": sum(1 for e in edges
                                     if e["type"] == "agent_spawn"
                                     and e["return_status"] == "unmatched"),
    }
    return edges, unresolved, stats


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

    edges, unresolved, stats = build_edges(events, known)
    trace.setdefault("edges", []).extend(edges)
    trace.setdefault("run", {})["linkage"] = {
        "source": events_path, "present": True, "events_read": len(events),
        "malformed": len(problems), "stats": stats,
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
