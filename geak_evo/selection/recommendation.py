"""The planner's standing recommendation of a final program.

The planner may end a turn with a recommendation next to its proposal:
``keep`` (no change), ``set`` (pick one offered choice by its handle, with a
reason) or ``clear`` (withdraw the current pick, with a reason). This module
turns that reply into a deterministic record, commits records into the
campaign state as a chain of events, and reads them back. It never ranks
programs, never falls back to an earlier or "best" pick, and makes no model
call.

``extract_structured_response`` finds the one JSON object in a free-text model
reply. The whole reply may be a strict JSON object or literal ``null``.
Otherwise exactly one object must start a physical line, either bare or inside
a triple-backtick ``json`` fence, and nothing but whitespace may follow it.
Inline JSON inside prose, other backtick code spans, standalone arrays or
strings, malformed containers, unclosed code spans and trailing text are not
results. The caller's strict decoder still applies to the selected bytes.
"""

import copy
import json
from pathlib import Path
import re

from .._util import canonical_json, require, sha256_hex


SCHEMA = "geak-member-recommendation-v1"
NS = 1_000_000_000
WHITESPACE = " \t\r\n"
_REJECTED = "invalid recommendation: use keep, set with an offered choice_handle and reason, or clear with reason"


def encoded(value):
    return canonical_json(value)


def digest(value):
    return sha256_hex(canonical_json(value))


def need(condition, message):
    require(condition, message, ValueError)


# ---------------------------------------------------------------- structured reply

def _decode_or_too_deep(decode, raw):
    try:
        return decode(raw)
    except RecursionError as exc:
        raise ValueError("structured response nesting is too deep") from exc


def _fenced_object(text, body_start, decode):
    closing = re.search(r"^[ \t]*```[ \t]*\r?(?:\n|$)", text[body_start:], re.MULTILINE)
    if closing is None or text[body_start + closing.end():].strip(WHITESPACE):
        raise ValueError("ambiguous JSON fence or non-whitespace tail")
    selected = text[body_start:body_start + closing.start()].encode("utf-8")
    if type(_decode_or_too_deep(decode, selected)) is not dict:
        raise ValueError("fenced structured response must be one object")
    return selected


def _code_span_end(text, position, end, marker):
    close = text.find(marker, end)
    while close >= 0 and ((close > 0 and text[close - 1] == "`")
                          or (close + len(marker) < len(text) and text[close + len(marker)] == "`")):
        close = text.find(marker, close + len(marker))
    if close < 0:
        raise ValueError("unclosed code span before structured response")
    return close + len(marker)


def extract_structured_response(raw, *, decode):
    """Return the selected bytes unchanged, using the caller's strict JSON decoder."""
    if type(raw) is not bytes:
        raise TypeError("structured response must be UTF-8 bytes")
    try:
        value = decode(raw)
    except RecursionError as exc:
        raise ValueError("structured response nesting is too deep") from exc
    except ValueError:
        pass
    else:
        if value is None or type(value) is dict:
            return raw
        raise ValueError("structured response must be an object or whole-response null")
    text = raw.decode("utf-8")
    decoder = json.JSONDecoder()
    position, line_blank = 0, True
    while position < len(text):
        char = text[position]
        first = line_blank
        if char == "`":
            end = position
            while end < len(text) and text[end] == "`":
                end += 1
            marker = text[position:end]
            opening = re.match(r"json[ \t]*\r?\n", text[end:]) if first and marker == "```" else None
            if opening:
                return _fenced_object(text, end + opening.end(), decode)
            position = _code_span_end(text, position, end, marker)
            line_blank = False
            continue
        if first and char == "~" and text.startswith("~~~", position):
            raise ValueError("tilde-fenced response is not a standalone JSON result")
        if first and char in '["':
            raise ValueError("standalone non-object JSON before structured response")
        if char in "{[":
            # Never salvage a nested or later object from a malformed outer one.
            try:
                value, end = decoder.raw_decode(text, position)
            except (ValueError, RecursionError) as exc:
                raise ValueError("malformed JSON container in response") from exc
            selected = text[position:end].encode("utf-8")
            _decode_or_too_deep(decode, selected)  # Duplicate keys and non-finite numbers stay forbidden.
            if first:
                if type(value) is not dict or text[end:].strip(WHITESPACE):
                    raise ValueError("ambiguous structured response or non-whitespace tail")
                return selected
            position = end
            line_blank = False
            continue
        if char == "\n":
            line_blank = True
        elif char not in WHITESPACE:
            line_blank = False
        position += 1
    raise ValueError("no standalone JSON result object")


# ---------------------------------------------------------------- recommendation records

def split_response(value):
    """Separate the optional recommendation from the proposal; the proposal is not checked here."""
    if type(value) is dict and ({"proposal", "recommendation"} & set(value)):
        need(set(value) == {"proposal", "recommendation"},
             "recommendation envelope requires exactly proposal and recommendation")
        return value["proposal"], value["recommendation"]
    return value, {"op": "keep"}


def decision(directive, choices):
    """Interpret the planner's directive. Only a malformed directive is rejected;
    errors in the host's own choice list propagate."""
    handles = [row["choice_handle"] for row in choices]
    need(len(handles) == len(set(handles)), "duplicate host recommendation choice")
    rejected = {"disposition": "rejected", "choice": None, "reason": None, "rejection": _REJECTED}
    if type(directive) is not dict:
        return rejected
    operation = directive.get("op")
    if operation == "keep" and set(directive) == {"op"}:
        return {"disposition": "keep", "choice": None, "reason": None, "rejection": None}
    fields = {"op", "reason", "choice_handle"} if operation == "set" else {"op", "reason"}
    if (operation not in ("set", "clear") or set(directive) != fields
            or type(directive.get("reason")) is not str or not directive["reason"].strip()
            or len(directive["reason"]) > 2000):
        return rejected
    choice = None
    if operation == "set":
        if type(directive["choice_handle"]) is not str or directive["choice_handle"] not in handles:
            return dict(rejected, rejection="recommendation choice_handle is not an exact eligible frozen handle")
        choice = choices[handles.index(directive["choice_handle"])]
    return {"disposition": operation, "choice": copy.deepcopy(choice),
            "reason": directive["reason"], "rejection": None}


def make_record(*, context, intent, clock_ref, clock, receipt_ref, response_raw,
                proposal_raw, directive, choices, accepted_ns):
    """Build the deterministic record; validation rebuilds it to check a saved one."""
    need(type(accepted_ns) is int, "integer recommendation acceptance time required")
    need(context["campaign_id"] == intent["campaign_id"]
         and context["state"]["binding"] == intent["binding"]
         and clock_ref == intent["admission"]["clock_ref"], "recommendation domain differs")
    start = clock["started_monotonic_ns"]
    end = min(start + int(clock["wall_seconds"] * NS),
              start + int((context["state"]["deadline"] - clock["started_unix"]) * NS))
    result = decision(directive, choices)
    if not start <= intent["started_monotonic_ns"] <= accepted_ns < intent["deadline_monotonic_ns"] <= end:
        result = {"disposition": "rejected", "choice": None, "reason": None,
                  "rejection": "recommendation acceptance is outside the original campaign or planner cutoff"}
    return {"schema": SCHEMA, "campaign_id": context["campaign_id"],
            "binding": copy.deepcopy(context["state"]["binding"]),
            "context_revision": context["revision"], "context_sha256": digest(context),
            "operation_id": intent["operation_id"],
            "previous_head": copy.deepcopy(context["state"].get("recommendation_head")),
            "clock_ref": copy.deepcopy(clock_ref), "receipt_ref": copy.deepcopy(receipt_ref),
            "accepted_monotonic_ns": accepted_ns,
            "planner_started_monotonic_ns": intent["started_monotonic_ns"],
            "planner_deadline_monotonic_ns": intent["deadline_monotonic_ns"],
            "campaign_deadline_monotonic_ns": end,
            "response_sha256": sha256_hex(response_raw),
            "proposal_sha256": sha256_hex(proposal_raw),
            "choices_sha256": digest(choices), "directive": copy.deepcopy(directive), **result}


def commit_record(state, record, *, planner_ref, context, commit_revision, in_budget):
    """Append one recommendation event to ``state``. A record moves the head only
    when it is in budget and is a set or clear; it never changes the proposal."""
    need(record["schema"] == SCHEMA and record["operation_id"] == state["pending"]["operation_id"]
         and record["campaign_id"] == context["campaign_id"]
         and record["binding"] == state["binding"] == context["state"]["binding"]
         and record["context_revision"] == context["revision"]
         and record["context_sha256"] == digest(context)
         and record["previous_head"] == state.get("recommendation_head")
         and record["previous_head"] == context["state"].get("recommendation_head"),
         "recommendation does not match pending frozen context or head")
    ref = {"evidence_id": planner_ref["evidence_id"], "artifact": "recommendation-binding.json"}
    events = state.setdefault("recommendation_events", [])
    need(not any(row["record_ref"] == ref for row in events), "recommendation was already committed")
    applied = in_budget and record["disposition"] in ("set", "clear")
    if applied:
        need(record["planner_started_monotonic_ns"] <= record["accepted_monotonic_ns"]
             < min(record["planner_deadline_monotonic_ns"], record["campaign_deadline_monotonic_ns"]),
             "recommendation update lacks pre-cutoff acceptance")
        state["recommendation_head"] = ref
    events.append({"record_ref": ref, "commit_revision": commit_revision,
                   "applied": applied, "in_budget": bool(in_budget)})


def _object_bytes(store, identity):
    need(type(identity) is str and len(identity) == 64
         and all(letter in "0123456789abcdef" for letter in identity), "exact immutable object identity required")
    path = Path(store.root).resolve(strict=True) / "objects" / identity
    need(path.resolve(strict=True) == path and path.is_file(), "canonical immutable object required")
    raw = path.read_bytes()
    need(sha256_hex(raw) == identity, "recommendation immutable object changed")
    return raw


def read_record(store, ref):
    """Read one saved recommendation record by its evidence reference."""
    need(type(ref) is dict and set(ref) == {"evidence_id", "artifact"}
         and ref["artifact"] == "recommendation-binding.json", "exact recommendation artifact reference required")
    # The record shares its evidence envelope with raw planner events: read only
    # the named artifact, never the whole evidence group.
    envelope = json.loads(_object_bytes(store, ref["evidence_id"]))
    need(envelope.get("kind") == "evidence", "recommendation evidence envelope required")
    record = json.loads(_object_bytes(store, envelope["artifacts"][ref["artifact"]]))
    candidate = store.get_candidate(envelope["candidate_id"])
    need(record["schema"] == SCHEMA
         and all(candidate[key] == value for key, value in record["binding"].items())
         and envelope["binding"] == dict(record["binding"], candidate_source_id=candidate["source_id"]),
         "recommendation artifact and evidence domain differ")
    return record


def recommendation_view(store, snapshot):
    """Current head record and latest event record of a campaign snapshot (read-only)."""
    state = snapshot["state"]
    head = state.get("recommendation_head")
    last = state.get("recommendation_events", [])
    current = read_record(store, head) if head else None
    return {"status": "selected" if current and current["disposition"] == "set" else "missing",
            "head": head, "current": current,
            "latest_event": read_record(store, last[-1]["record_ref"]) if last else None}
