#!/usr/bin/env python3
"""Atomic run-state and obligation transitions for canonical finalization.

The state file is the authority for a run generation.  Opening a critic or resweep
obligation advances the generation, supersedes every previous completed obligation,
and recreates those obligations for explicit revalidation.  A final report is valid
only for the generation that the state machine finalizes.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from typing import Any, Callable


STATE_SCHEMA = "kernel_opt.run_state/1"
STATES = (
    "open", "paused", "crashed", "orphaned", "finalizing", "closed", "blocked",
    "deferred_needs_user", "partial_time_limit",
)
OBLIGATION_STATES = ("open", "done", "superseded")
CHILD_STATES = ("running", "draining", "completed", "cancelled", "orphaned", "adopted")
OBLIGATION_KINDS = (
    "canonical_entry", "comparator", "profile", "arm_gate", "critic", "resweep", "sweep", "report",
)
CRITICAL_KINDS = ("critic", "resweep", "sweep")


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_time_limit(value: str) -> int:
    text = str(value).strip().lower()
    multiplier = 1
    if text.endswith("h"):
        multiplier, text = 3600, text[:-1]
    elif text.endswith("m"):
        multiplier, text = 60, text[:-1]
    elif text.endswith("s"):
        text = text[:-1]
    try:
        seconds = float(text) * multiplier
    except ValueError as exc:
        raise ValueError("time limit must be a positive duration like 2h, 90m, or 3600s") from exc
    if seconds <= 0:
        raise ValueError("time limit must be positive")
    return int(seconds)


def _parse_timestamp(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(dt.timezone.utc)


def _finding(out: list[dict], field: str, code: str, detail: str) -> None:
    out.append({"field": field, "code": code, "detail": detail, "severity": "error"})


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_state(doc: Any, prefix: str = "run_state") -> list[dict]:
    """Validate state structure and its non-reusable obligation lifecycle."""
    out: list[dict] = []
    if not isinstance(doc, dict):
        _finding(out, prefix, "type", "expected an object")
        return out
    if doc.get("schema") != STATE_SCHEMA:
        _finding(out, f"{prefix}.schema", "schema", f"expected {STATE_SCHEMA!r}")
    if not _nonempty(doc.get("run_id")):
        _finding(out, f"{prefix}.run_id", "missing", "must be a non-empty id")
    generation = doc.get("generation")
    if not isinstance(generation, int) or isinstance(generation, bool) or generation < 1:
        _finding(out, f"{prefix}.generation", "generation", "must be a positive integer")
        generation = 0
    state = doc.get("state")
    if state not in STATES:
        _finding(out, f"{prefix}.state", "enum", f"expected one of {list(STATES)}")
    stage = doc.get("stage")
    if stage is not None and not _nonempty(stage):
        _finding(out, f"{prefix}.stage", "type", "must be a non-empty string or null")
    history = doc.get("stage_history")
    if history is not None:
        if not isinstance(history, list):
            _finding(out, f"{prefix}.stage_history", "type", "must be a list or null")
        else:
            for index, transition in enumerate(history):
                item = f"{prefix}.stage_history[{index}]"
                if not isinstance(transition, dict):
                    _finding(out, item, "type", "must be an object")
                    continue
                for field in ("from", "to", "owner", "at", "receipt_ref"):
                    if not _nonempty(transition.get(field)):
                        _finding(out, f"{item}.{field}", "missing", "must be non-empty")
                item_generation = transition.get("generation")
                if (not isinstance(item_generation, int) or isinstance(item_generation, bool)
                        or item_generation < 1 or item_generation > generation):
                    _finding(out, f"{item}.generation", "generation",
                             "must be a positive generation not later than the current generation")
    deadline = doc.get("deadline")
    if deadline is not None:
        if not isinstance(deadline, dict):
            _finding(out, f"{prefix}.deadline", "type", "must be an object or null")
        else:
            for field in ("started_at", "deadline_at", "owner"):
                if not _nonempty(deadline.get(field)):
                    _finding(out, f"{prefix}.deadline.{field}", "missing", "must be non-empty")
            if not isinstance(deadline.get("time_limit_s"), int) or deadline["time_limit_s"] <= 0:
                _finding(out, f"{prefix}.deadline.time_limit_s", "type", "must be a positive integer")
            if not isinstance(deadline.get("close_reserve_s"), int) or deadline["close_reserve_s"] < 0:
                _finding(out, f"{prefix}.deadline.close_reserve_s", "type",
                         "must be a non-negative integer")
    children = doc.get("children")
    if children is not None:
        if not isinstance(children, list):
            _finding(out, f"{prefix}.children", "type", "must be a list or null")
        else:
            attempts = set()
            for index, child in enumerate(children):
                item = f"{prefix}.children[{index}]"
                if not isinstance(child, dict):
                    _finding(out, item, "type", "must be an object")
                    continue
                for field in ("attempt_id", "role", "input_ref", "checkpoint_ref",
                              "deadline_at", "state"):
                    if not _nonempty(child.get(field)):
                        _finding(out, f"{item}.{field}", "missing", "must be non-empty")
                attempt = child.get("attempt_id")
                if attempt in attempts:
                    _finding(out, f"{item}.attempt_id", "unique", "must be unique")
                attempts.add(attempt)
                if child.get("state") not in CHILD_STATES:
                    _finding(out, f"{item}.state", "enum",
                             f"expected one of {list(CHILD_STATES)}")
                child_generation = child.get("parent_generation")
                if (not isinstance(child_generation, int) or isinstance(child_generation, bool)
                        or child_generation < 1 or child_generation > generation):
                    _finding(out, f"{item}.parent_generation", "generation",
                             "must be a positive generation not later than the current parent generation")
    heartbeat = doc.get("heartbeat")
    if heartbeat is not None:
        if not isinstance(heartbeat, dict):
            _finding(out, f"{prefix}.heartbeat", "type", "must be an object or null")
        else:
            for field in ("at", "phase", "owner", "next_step"):
                if not _nonempty(heartbeat.get(field)):
                    _finding(out, f"{prefix}.heartbeat.{field}", "missing", "must be non-empty")
    obligations = doc.get("obligations")
    if not isinstance(obligations, list):
        _finding(out, f"{prefix}.obligations", "type", "expected a list")
        return out
    ids = set()
    for i, obligation in enumerate(obligations):
        path = f"{prefix}.obligations[{i}]"
        if not isinstance(obligation, dict):
            _finding(out, path, "type", "expected an object")
            continue
        oid = obligation.get("obligation_id")
        if not _nonempty(oid) or oid in ids:
            _finding(out, f"{path}.obligation_id", "unique", "must be a unique non-empty id")
        ids.add(oid)
        if obligation.get("kind") not in OBLIGATION_KINDS:
            _finding(out, f"{path}.kind", "enum",
                     f"expected one of {list(OBLIGATION_KINDS)}")
        if obligation.get("status") not in OBLIGATION_STATES:
            _finding(out, f"{path}.status", "enum",
                     f"expected one of {list(OBLIGATION_STATES)}")
        og = obligation.get("generation")
        if not isinstance(og, int) or isinstance(og, bool) or og < 1 or og > generation:
            _finding(out, f"{path}.generation", "generation",
                     "must be a positive generation not later than state generation")
        if obligation.get("status") == "done" and not _nonempty(obligation.get("completed_at")):
            _finding(out, f"{path}.completed_at", "missing",
                     "done obligation must record completion time")
        if obligation.get("status") == "open" and og != generation:
            _finding(out, f"{path}.generation", "stale_open",
                     "open obligations must belong to the current generation")
        if obligation.get("status") == "superseded":
            if not isinstance(obligation.get("superseded_by_generation"), int):
                _finding(out, f"{path}.superseded_by_generation", "missing",
                         "superseded obligation must name the replacing generation")
            elif obligation["superseded_by_generation"] <= og:
                _finding(out, f"{path}.superseded_by_generation", "generation",
                         "must be later than the obligation generation")
    return out


def _atomic_write(path: str, doc: dict) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".run_state.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(doc, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        try:
            dir_fd = os.open(directory, os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


@contextmanager
def _locked(path: str):
    """Hold an adjacent advisory lock across load, transition, and atomic replace."""
    lock_path = f"{path}.lock"
    directory = os.path.dirname(os.path.abspath(lock_path))
    os.makedirs(directory, exist_ok=True)
    with open(lock_path, "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _load(path: str) -> dict:
    with open(path) as fh:
        doc = json.load(fh)
    errors = validate_state(doc)
    if errors:
        raise ValueError(f"invalid state: {errors[0]['field']}: {errors[0]['detail']}")
    return doc


def _invalidate_context_lease(state_path: str) -> None:
    """Settle a fixed-path lease after a generation or stage boundary."""
    lease_path = os.path.join(os.path.dirname(os.path.abspath(state_path)), ".kod",
                              "context_lease.json")
    if not os.path.isfile(lease_path):
        return
    try:
        with open(lease_path, encoding="utf-8") as fh:
            lease = json.load(fh)
        if isinstance(lease, dict) and lease.get("status") != "settled":
            lease["status"] = "settled"
            _atomic_write(lease_path, lease)
    except (OSError, ValueError, json.JSONDecodeError):
        # A malformed lease already cannot validate; state authority must still advance.
        return


def init_state(path: str, run_id: str) -> dict:
    if not _nonempty(run_id):
        raise ValueError("run_id must be non-empty")
    with _locked(path):
        if os.path.exists(path):
            raise ValueError(f"state already exists: {path}")
        state = {
            "schema": STATE_SCHEMA,
            "run_id": run_id,
            "generation": 1,
            "state": "open",
            "stage": "entry",
            "stage_history": [],
            "deadline": None,
            "children": [],
            "obligations": [],
            "heartbeat": None,
            "updated_at": _now(),
        }
        _atomic_write(path, state)
        return state


def _update(path: str, transform: Callable[[dict], dict]) -> dict:
    with _locked(path):
        current = _load(path)
        next_state = transform(copy.deepcopy(current))
        errors = validate_state(next_state)
        if errors:
            raise ValueError(f"invalid transition: {errors[0]['field']}: {errors[0]['detail']}")
        next_state["updated_at"] = _now()
        _atomic_write(path, next_state)
        if (
            next_state.get("generation") != current.get("generation")
            or next_state.get("stage") != current.get("stage")
        ):
            _invalidate_context_lease(path)
        return next_state


def _new_id(kind: str, generation: int, existing: set[str]) -> str:
    stem = f"{kind}@g{generation}"
    if stem not in existing:
        return stem
    suffix = 2
    while f"{stem}.{suffix}" in existing:
        suffix += 1
    return f"{stem}.{suffix}"


def open_obligation(path: str, kind: str, reason: str, obligation_id: str | None = None) -> dict:
    """Open one obligation in a fresh generation.

    Critic/resweep are invalidating events.  Their transition supersedes all earlier
    completed obligations and opens a generation-local replacement for each one,
    making a stale ``done`` impossible to count toward the new finalization.
    """
    if kind not in OBLIGATION_KINDS:
        raise ValueError(f"kind must be one of {OBLIGATION_KINDS}")
    if not _nonempty(reason):
        raise ValueError("reason must be non-empty")

    def transform(state: dict) -> dict:
        active = [x["obligation_id"] for x in state["obligations"] if x["status"] == "open"]
        if active:
            raise ValueError(f"complete or supersede active obligations before opening another: {active}")
        generation = state["generation"] + 1
        state["generation"] = generation
        state["state"] = "open"
        existing = {x["obligation_id"] for x in state["obligations"]}
        replacements: list[dict] = []
        if kind in CRITICAL_KINDS:
            for old in state["obligations"]:
                if old["status"] != "done":
                    continue
                old["status"] = "superseded"
                old["superseded_at"] = _now()
                old["superseded_by_generation"] = generation
                replacement_id = _new_id(old["kind"], generation, existing)
                existing.add(replacement_id)
                replacements.append({
                    "obligation_id": replacement_id,
                    "kind": old["kind"],
                    "generation": generation,
                    "status": "open",
                    "reason": f"revalidate after {kind}: {reason}",
                    "supersedes": old["obligation_id"],
                    "opened_at": _now(),
                })
        oid = obligation_id or _new_id(kind, generation, existing)
        if oid in existing or not _nonempty(oid):
            raise ValueError(f"obligation_id already exists or is invalid: {oid!r}")
        state["obligations"].extend(replacements)
        state["obligations"].append({
            "obligation_id": oid,
            "kind": kind,
            "generation": generation,
            "status": "open",
            "reason": reason,
            "opened_at": _now(),
        })
        return state

    return _update(path, transform)


def complete_obligation(path: str, obligation_id: str, expected_generation: int,
                        evidence_ref: str) -> dict:
    """Complete only an open obligation from the state file's current generation."""
    if not _nonempty(evidence_ref):
        raise ValueError("evidence_ref must be non-empty")

    def transform(state: dict) -> dict:
        if state["generation"] != expected_generation:
            raise ValueError(
                f"generation changed: expected {expected_generation}, current {state['generation']}")
        obligation = next((x for x in state["obligations"]
                           if x["obligation_id"] == obligation_id), None)
        if obligation is None:
            raise ValueError(f"unknown obligation: {obligation_id}")
        if obligation["status"] != "open":
            raise ValueError(f"obligation {obligation_id} is {obligation['status']}, not open")
        if obligation["generation"] != state["generation"]:
            raise ValueError(f"obligation {obligation_id} belongs to stale generation "
                             f"{obligation['generation']}")
        obligation["status"] = "done"
        obligation["completed_at"] = _now()
        obligation["evidence_ref"] = evidence_ref
        return state

    return _update(path, transform)


def heartbeat(path: str, phase: str, owner: str, next_step: str, *,
              lease: str | None = None, last_artifact: str | None = None) -> dict:
    """Publish a non-invalidating progress marker for fleet supervision."""
    if not all(_nonempty(value) for value in (phase, owner, next_step)):
        raise ValueError("heartbeat phase, owner and next_step must be non-empty")

    def transform(state: dict) -> dict:
        state["heartbeat"] = {
            "at": _now(),
            "phase": phase,
            "owner": owner,
            "lease": lease,
            "last_artifact": last_artifact,
            "next_step": next_step,
        }
        return state

    return _update(path, transform)


def transition(path: str, expected_generation: int, expected_stage: str, next_stage: str,
               owner: str, receipt_ref: str, artifact_hashes: dict[str, str]) -> dict:
    """Advance the authoritative workflow stage after its receipt and inputs are verified.

    The caller owns stage-card ordering; this primitive makes the accepted transition durable,
    generation-bound and auditable.  A registry is only a convenience index and may not advance
    this state on its own.
    """
    if not all(_nonempty(value) for value in
               (expected_stage, next_stage, owner, receipt_ref)):
        raise ValueError("expected stage, next stage, owner, and receipt ref must be non-empty")
    if not isinstance(artifact_hashes, dict) or any(
            not _nonempty(name) or not _nonempty(value)
            for name, value in artifact_hashes.items()):
        raise ValueError("artifact_hashes must map non-empty artifact names to SHA256 values")

    def transform(state: dict) -> dict:
        if state["generation"] != expected_generation:
            raise ValueError(
                f"generation changed: expected {expected_generation}, current {state['generation']}")
        if state["state"] not in ("open", "paused"):
            raise ValueError(f"cannot transition workflow from state={state['state']!r}")
        current = state.get("stage") or "entry"
        if current != expected_stage:
            raise ValueError(f"stage changed: expected {expected_stage!r}, current {current!r}")
        state["stage"] = next_stage
        state["state"] = "open"
        state.setdefault("stage_history", []).append({
            "from": expected_stage,
            "to": next_stage,
            "owner": owner,
            "receipt_ref": receipt_ref,
            "artifact_hashes": dict(sorted(artifact_hashes.items())),
            "generation": state["generation"],
            "at": _now(),
        })
        return state

    return _update(path, transform)


def set_terminal_state(path: str, expected_generation: int, state_name: str, reason: str) -> dict:
    """Record a recoverable non-success stop without turning it into a search conclusion."""
    if state_name not in ("paused", "crashed", "orphaned", "partial_time_limit",
                          "blocked", "deferred_needs_user"):
        raise ValueError(f"unsupported terminal state: {state_name!r}")
    if not _nonempty(reason):
        raise ValueError("terminal-state reason must be non-empty")

    def transform(state: dict) -> dict:
        if state["generation"] != expected_generation:
            raise ValueError("generation changed before terminal-state transition")
        if state["state"] in ("closed", "finalizing"):
            raise ValueError(f"cannot stop a finalized run: {state['state']}")
        state["state"] = state_name
        state["stop_reason"] = reason
        state["stopped_at"] = _now()
        return state

    return _update(path, transform)


def _default_close_reserve(seconds: int) -> int:
    """Reserve finalization time for the public 2h/4h/8h invocation presets."""
    if seconds <= 2 * 3600:
        return 25 * 60
    if seconds <= 4 * 3600:
        return 40 * 60
    return 60 * 60


def set_deadline(path: str, expected_generation: int, time_limit: str, owner: str,
                 close_reserve_s: int | None = None) -> dict:
    """Set the immutable task-wide deadline before work starts."""
    seconds = _parse_time_limit(time_limit)
    if not _nonempty(owner):
        raise ValueError("deadline owner must be non-empty")
    if close_reserve_s is None:
        close_reserve_s = _default_close_reserve(seconds)
    if not isinstance(close_reserve_s, int) or close_reserve_s < 0 or close_reserve_s >= seconds:
        raise ValueError("close reserve must be non-negative and smaller than the task time limit")

    def transform(state: dict) -> dict:
        if state["generation"] != expected_generation:
            raise ValueError("generation changed before deadline was set")
        if state["deadline"] is not None:
            raise ValueError("task deadline is immutable once set")
        if state["state"] != "open" or state.get("stage") != "entry":
            raise ValueError("task deadline must be set while the run is open at entry")
        now = dt.datetime.now(dt.timezone.utc)
        state["deadline"] = {
            "started_at": now.isoformat().replace("+00:00", "Z"),
            "deadline_at": (now + dt.timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z"),
            "time_limit_s": seconds,
            "close_reserve_s": close_reserve_s,
            "owner": owner,
        }
        return state

    return _update(path, transform)


def remaining_seconds(state: dict, *, now: dt.datetime | None = None) -> float | None:
    """Return remaining end-to-end wall clock for a state, or None when no limit is declared."""
    deadline = state.get("deadline")
    if not isinstance(deadline, dict) or not _nonempty(deadline.get("deadline_at")):
        return None
    try:
        limit = _parse_timestamp(deadline["deadline_at"])
    except (TypeError, ValueError):
        return None
    current = now or dt.datetime.now(dt.timezone.utc)
    return (limit - current).total_seconds()


TERMINAL_STATES = ("closed", "blocked", "deferred_needs_user", "partial_time_limit")


def liveness(state: dict, *, now: dt.datetime | None = None) -> dict:
    """Is this owner progressing, waiting on someone, finished, or out of time -- from DECLARATIONS.

    WHY THIS READER EXISTS. Every field it reads was already being WRITTEN; nothing could read it.
    So a supervisor with no reader invented one from file modification times plus a fixed
    silent-for-N rule, and that rule fired on healthy owners: between a child publishing its result
    and the parent finishing its own close, a correct owner legitimately writes nothing for a long
    time, and its length is a property of the work rather than a constant anyone can pick. Twice
    that inferred rule would have interrupted a captain mid-close -- the most expensive moment to
    interrupt, because the close is what makes the preceding work quotable.

    So there is NO absolute duration here, and there must not be one. The verdicts are:

      terminal        the run declared an end state. Silence after this is the correct behaviour.
      finalizing      the close is in progress; interrupting it discards the close, not the wait.
      awaiting_child  a child is registered `running`/`draining`. The parent is SUPPOSED to be
                      quiet, and the child's own deadline is the bound on that quiet.
      overdue         the owner's OWN declared deadline has passed and no end state was written.
                      This is the only "act now", and it is relative to what the owner declared.
      in_close_reserve inside the reserve the owner itself set aside for closing.
      progressing     open, inside its declared time, with a heartbeat to name the phase.
      unheard         open and inside its time, but nothing has ever named a phase. Not a stall:
                      an owner that never wrote a heartbeat is an INSTRUMENTATION gap, and the
                      first move is to ask for one rather than to resume over the top of it.
    """
    current = now or dt.datetime.now(dt.timezone.utc)
    run_state = state.get("state")
    children = [c for c in (state.get("children") or [])
                if isinstance(c, dict) and c.get("state") in ("running", "draining")]
    heartbeat = state.get("heartbeat") if isinstance(state.get("heartbeat"), dict) else None
    remaining = remaining_seconds(state, now=current)
    reserve = (state.get("deadline") or {}).get("close_reserve_s")
    out = {
        "run_id": state.get("run_id"),
        "declared_state": run_state,
        "stage": state.get("stage"),
        "generation": state.get("generation"),
        "running_children": [{"attempt_id": c.get("attempt_id"), "role": c.get("role"),
                              "deadline_at": c.get("deadline_at"), "state": c.get("state")}
                             for c in children],
        "heartbeat": heartbeat,
        "remaining_seconds": remaining,
        "close_reserve_s": reserve,
        "open_obligations": [o.get("obligation_id") for o in (state.get("obligations") or [])
                             if isinstance(o, dict) and o.get("status") == "open"],
    }
    if run_state in TERMINAL_STATES:
        verdict, why = "terminal", f"the run declared {run_state!r}; silence after this is correct"
    elif run_state in ("paused", "crashed", "orphaned"):
        verdict, why = "needs_operator", f"the run declared {run_state!r} rather than an end state"
    elif remaining is not None and remaining <= 0:
        verdict, why = "overdue", ("the owner's OWN declared deadline has passed with no end state "
                                   "written; act on the declaration, not on elapsed silence")
    elif run_state == "finalizing":
        verdict, why = "finalizing", ("the close is in progress. Interrupting here discards the "
                                      "close itself, which is what makes the work quotable")
    elif children:
        verdict, why = "awaiting_child", (
            f"{len(children)} child(ren) registered running/draining, so quiet is expected here; "
            f"the bound on it is each child's own declared deadline, not a fixed interval")
    elif remaining is not None and reserve and remaining <= float(reserve):
        verdict, why = "in_close_reserve", ("inside the reserve the owner set aside for closing; "
                                            "new optimization work is not expected to start")
    elif heartbeat:
        verdict, why = "progressing", (f"open at stage {state.get('stage')!r}, phase "
                                       f"{heartbeat.get('phase')!r}, next {heartbeat.get('next_step')!r}")
    else:
        verdict, why = "unheard", ("open and inside its declared time, but no heartbeat has ever "
                                   "named a phase. That is an instrumentation gap: ask for a "
                                   "heartbeat before resuming over the top of live work")
    out["verdict"], out["why"] = verdict, why
    out["act_now"] = verdict in ("overdue", "needs_operator")
    return out


def dispatch_child(path: str, expected_generation: int, attempt_id: str, role: str,
                   input_ref: str, checkpoint_ref: str, deadline_at: str) -> dict:
    """Durably register a child before it receives work."""
    if not all(_nonempty(value) for value in
               (attempt_id, role, input_ref, checkpoint_ref, deadline_at)):
        raise ValueError("child attempt_id, role, input_ref, checkpoint_ref, and deadline_at are required")
    try:
        _parse_timestamp(deadline_at)
    except (TypeError, ValueError) as exc:
        raise ValueError("child deadline_at must be an ISO-8601 timestamp") from exc

    def transform(state: dict) -> dict:
        if state["generation"] != expected_generation:
            raise ValueError("generation changed before child dispatch")
        if state["state"] != "open":
            raise ValueError(f"cannot dispatch child from state={state['state']!r}")
        parent_deadline = remaining_seconds(state)
        if parent_deadline is not None and _parse_timestamp(deadline_at) > _parse_timestamp(
                state["deadline"]["deadline_at"]):
            raise ValueError("child deadline may not exceed the parent deadline")
        if any(child["attempt_id"] == attempt_id for child in state.get("children", [])):
            raise ValueError(f"child attempt already exists: {attempt_id}")
        state.setdefault("children", []).append({
            "attempt_id": attempt_id,
            "role": role,
            "parent_generation": state["generation"],
            "input_ref": input_ref,
            "checkpoint_ref": checkpoint_ref,
            "deadline_at": deadline_at,
            "state": "running",
            "dispatched_at": _now(),
        })
        return state

    return _update(path, transform)


def settle_child(path: str, expected_generation: int, attempt_id: str, state_name: str,
                 receipt_ref: str, *, adopter_role: str | None = None) -> dict:
    """Cancel, drain, orphan, complete, or adopt a durable child attempt."""
    if state_name not in CHILD_STATES or state_name == "running":
        raise ValueError(f"unsupported child settlement state: {state_name!r}")
    if not _nonempty(receipt_ref):
        raise ValueError("child settlement requires a receipt/result reference")
    if state_name == "adopted" and not _nonempty(adopter_role):
        raise ValueError("orphan adoption requires the receiving role")

    def transform(state: dict) -> dict:
        if state["generation"] != expected_generation:
            raise ValueError("generation changed before child settlement")
        child = next((item for item in state.get("children", [])
                      if item.get("attempt_id") == attempt_id), None)
        if child is None:
            raise ValueError(f"unknown child attempt: {attempt_id}")
        if child["state"] in ("cancelled", "adopted"):
            raise ValueError(f"child attempt already terminal: {child['state']}")
        if state_name == "adopted":
            if child["state"] != "orphaned":
                raise ValueError("only an orphaned child result may be adopted")
            if adopter_role != child["role"]:
                raise ValueError("an orphan result must be adopted by the same accountable role")
            child["adopted_by"] = adopter_role
        child["state"] = state_name
        child["settled_at"] = _now()
        child["receipt_ref"] = receipt_ref
        return state

    return _update(path, transform)


def begin_finalization(path: str, expected_generation: int) -> dict:
    def transform(state: dict) -> dict:
        if state["generation"] != expected_generation:
            raise ValueError("generation changed before finalization")
        if state["state"] not in ("open", "finalizing"):
            raise ValueError(f"cannot finalize from state={state['state']}")
        open_ids = [x["obligation_id"] for x in state["obligations"] if x["status"] == "open"]
        if open_ids:
            raise ValueError(f"open obligations prevent finalization: {open_ids}")
        active_children = [child["attempt_id"] for child in state.get("children", [])
                           if child.get("state") in ("running", "draining", "orphaned")]
        if active_children:
            raise ValueError(f"active/orphaned child attempts prevent finalization: {active_children}")
        state["state"] = "finalizing"
        state["stage"] = "close"
        return state
    return _update(path, transform)


def finalize_state(path: str, expected_generation: int) -> dict:
    """Atomically close a generation after its artifact reconciliation succeeds."""
    def transform(state: dict) -> dict:
        if state["generation"] != expected_generation:
            raise ValueError("generation changed before finalization")
        if state["state"] != "finalizing":
            raise ValueError(f"strict finalizer requires state=finalizing, got {state['state']!r}")
        open_ids = [x["obligation_id"] for x in state["obligations"] if x["status"] == "open"]
        if open_ids:
            raise ValueError(f"open obligations prevent finalization: {open_ids}")
        state["state"] = "closed"
        state["finalized_at"] = _now()
        return state
    return _update(path, transform)


def reopen_state(path: str, expected_generation: int, reason: str, owner: str) -> dict:
    """Reopen a CLOSED run as a new generation, on the record, without losing its lineage.

    WHY THIS EXISTS. `closed` was terminal and nothing moved a run out of it, so a finalized run
    that turned out to have budget left could not be continued at all: `close` has no `next` and
    `set_deadline` refuses a second call. Measured on one campaign, three owners hit this. The
    only route left was a fresh work root seeded from the champion, which starts a run whose
    canonical record has no link to the one it continues -- so the second session's gain cannot be
    composed with the first without a human vouching for the join, and one owner instead hand-edited
    `run_state.json` and was (correctly) defeated by the state machine.

    WHY IT DOES NOT WEAKEN THE CLOSE. The guardrail worth keeping is that nobody slips PAST a close
    unnoticed, not that a close can never be revisited. So this is not a transition back to the old
    generation: it advances the generation, clears the finalization, and appends to `reopened_from`
    the generation it continues, when that generation closed, who authorized the reopen and why.
    A reader of the reopened state can therefore still see that a close happened and what came
    after it, which a fresh work root does not preserve and a hand edit actively hides.

    The deadline is cleared rather than extended, for the same reason it is immutable within a
    generation: the new session's wall clock is its own fact, and `set_deadline` must be called
    again at entry to state it. Silently inheriting the old window would let a reopen mint time.
    """
    if not _nonempty(reason):
        raise ValueError("a reopen must carry a reason: what the closed generation left unfinished")
    if not _nonempty(owner):
        raise ValueError("reopen owner must be non-empty")

    def transform(state: dict) -> dict:
        if state["generation"] != expected_generation:
            raise ValueError("generation changed before reopen")
        if state["state"] != "closed":
            raise ValueError(
                f"only a closed run is reopened, got {state['state']!r}; a run that is still open, "
                "blocked or finalizing is continued through its own stage graph")
        lineage = state.get("reopened_from")
        state["reopened_from"] = [*(lineage if isinstance(lineage, list) else []), {
            "generation": state["generation"],
            "finalized_at": state.get("finalized_at"),
            "stage": state.get("stage"),
            "reason": reason,
            "owner": owner,
            "reopened_at": _now(),
        }]
        state["generation"] = state["generation"] + 1
        state["state"] = "open"
        state["stage"] = "entry"
        state["finalized_at"] = None
        # Cleared so `set_deadline` can state the new session's own limit; see the docstring.
        state["deadline"] = None
        state["heartbeat"] = None
        return state

    return _update(path, transform)


def _selftest() -> int:
    import shutil
    root = tempfile.mkdtemp(prefix="run_state_selftest_")
    try:
        path = os.path.join(root, "run_state.json")
        state = init_state(path, "run-a")
        assert state["generation"] == 1 and state["state"] == "open"
        state = open_obligation(path, "comparator", "pin comparator")
        g2 = state["generation"]
        oid = state["obligations"][-1]["obligation_id"]
        state = complete_obligation(path, oid, g2, "baseline.json")
        assert state["obligations"][-1]["status"] == "done"
        state = heartbeat(path, "sweep", "triton-sweep", "publish comparator",
                          lease="42", last_artifact="baseline.json")
        assert state["heartbeat"]["lease"] == "42"
        state = transition(path, g2, "entry", "resolve", "captain", "receipt.json",
                           {"resolve.json": "a" * 64})
        assert state["stage"] == "resolve" and state["stage_history"][-1]["to"] == "resolve"
        state = open_obligation(path, "critic", "new counter-evidence")
        assert state["generation"] == g2 + 1
        old = next(x for x in state["obligations"] if x["obligation_id"] == oid)
        assert old["status"] == "superseded", state
        reopened = next(x for x in state["obligations"] if x.get("supersedes") == oid)
        assert reopened["status"] == "open" and reopened["generation"] == state["generation"]
        try:
            open_obligation(path, "sweep", "typed arm-local sweep")
            raise AssertionError("opened a sweep while revalidation remained open")
        except ValueError:
            pass
        try:
            complete_obligation(path, reopened["obligation_id"], g2, "stale.json")
            raise AssertionError("stale generation completed")
        except ValueError:
            pass
        generation = state["generation"]
        for obligation in [x for x in state["obligations"] if x["status"] == "open"]:
            state = complete_obligation(path, obligation["obligation_id"], generation, "evidence.json")
        state = begin_finalization(path, generation)
        assert state["state"] == "finalizing"
        state = finalize_state(path, generation)
        assert state["state"] == "closed"

        # A CLOSED run can be continued, but only as a new generation and only on the record.
        # Before this there was no edge out of `closed` at all, so an owner with budget left had to
        # start a fresh work root -- losing the lineage that lets the two sessions compose -- and
        # one tried to hand-edit this file instead.
        closed_generation = state["generation"]
        for bad, why in ((closed_generation, ""), (closed_generation, "   ")):
            try:
                reopen_state(path, bad, why, "captain")
                raise AssertionError("a reopen without a reason was accepted")
            except ValueError as exc:
                assert "reason" in str(exc), exc
        state = reopen_state(path, closed_generation, "13 rounds of budget left unspent", "captain")
        assert state["generation"] == closed_generation + 1, state
        assert state["state"] == "open" and state["stage"] == "entry", state
        assert state["finalized_at"] is None, state
        # The close is still visible after the reopen -- that is the difference between revisiting
        # a close and hiding one.
        assert state["reopened_from"][-1]["generation"] == closed_generation, state
        assert state["reopened_from"][-1]["reason"].startswith("13 rounds"), state
        assert state["reopened_from"][-1]["owner"] == "captain", state
        assert not validate_state(state), validate_state(state)
        # The new session states its own wall clock; it does not inherit the closed one's window.
        assert state["deadline"] is None, state
        state = set_deadline(path, state["generation"], "1h", "captain")
        assert state["deadline"]["time_limit_s"] == 3600, state
        # ...and reopen is not a way to walk out of a live run.
        try:
            reopen_state(path, state["generation"], "still working", "captain")
            raise AssertionError("an open run was reopened")
        except ValueError as exc:
            assert "only a closed run" in str(exc), exc

        limited = os.path.join(root, "limited_state.json")
        init_state(limited, "run-limited")
        limited_state = set_deadline(limited, 1, "2h", "fleet", 1500)
        assert limited_state["deadline"]["time_limit_s"] == 7200
        assert remaining_seconds(limited_state) is not None
        child_deadline = limited_state["deadline"]["deadline_at"]
        limited_state = dispatch_child(
            limited, 1, "arm-a", "arm", "branch_plan.json", "checkpoint/a", child_deadline)
        limited_state = settle_child(
            limited, 1, "arm-a", "orphaned", "arm_result.json")
        limited_state = settle_child(
            limited, 1, "arm-a", "adopted", "adoption.json", adopter_role="arm")
        assert limited_state["children"][0]["state"] == "adopted"

        # LIVENESS IS READ FROM DECLARATIONS. No case below involves an elapsed-silence threshold,
        # because the rule that used the one an operator picked fired on healthy owners.
        live = os.path.join(root, "liveness.json")
        s = init_state(live, "run-live")
        assert liveness(s)["verdict"] == "unheard", liveness(s)
        assert not liveness(s)["act_now"], "an owner with no heartbeat is not a resume trigger"
        s = heartbeat(live, "sweep", "worker", "measure the anchor")
        assert liveness(s)["verdict"] == "progressing", liveness(s)
        s = set_deadline(live, 1, "2h", "fleet", 1500)
        s = dispatch_child(live, 1, "child-a", "direction", "in.json", "cp/a",
                           s["deadline"]["deadline_at"])
        alive = liveness(s)
        # A parent waiting on a registered child is SUPPOSED to be quiet; the bound is the child's
        # own declared deadline, and this is the case the inferred rule got wrong.
        assert alive["verdict"] == "awaiting_child" and not alive["act_now"], alive
        assert alive["running_children"][0]["attempt_id"] == "child-a", alive
        s = settle_child(live, 1, "child-a", "completed", "child_result.json")
        s = begin_finalization(live, 1)
        fin = liveness(s)
        assert fin["verdict"] == "finalizing" and not fin["act_now"], fin
        # Only the owner's OWN expired deadline says act now...
        past = _parse_timestamp(_now()) - dt.timedelta(hours=1)
        s["deadline"]["deadline_at"] = past.isoformat().replace("+00:00", "Z")
        over = liveness(s)
        assert over["verdict"] == "overdue" and over["act_now"], over
        # ...and a declared end state makes silence correct rather than suspicious.
        s2 = dict(s, state="closed")
        assert liveness(s2)["verdict"] == "terminal" and not liveness(s2)["act_now"]
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("[run_state] SELFTEST PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=False)
    init = sub.add_parser("init")
    init.add_argument("--state", required=True)
    init.add_argument("--run-id", required=True)
    opened = sub.add_parser("open")
    opened.add_argument("--state", required=True)
    opened.add_argument("--kind", required=True, choices=OBLIGATION_KINDS)
    opened.add_argument("--reason", required=True)
    opened.add_argument("--obligation-id")
    done = sub.add_parser("done")
    done.add_argument("--state", required=True)
    done.add_argument("--obligation", required=True)
    done.add_argument("--generation", required=True, type=int)
    done.add_argument("--evidence-ref", required=True)
    beat = sub.add_parser("heartbeat")
    beat.add_argument("--state", required=True)
    beat.add_argument("--phase", required=True)
    beat.add_argument("--owner", required=True)
    beat.add_argument("--next-step", required=True)
    beat.add_argument("--lease")
    beat.add_argument("--last-artifact")
    advance = sub.add_parser("transition")
    advance.add_argument("--state", required=True)
    advance.add_argument("--generation", required=True, type=int)
    advance.add_argument("--from-stage", required=True)
    advance.add_argument("--to-stage", required=True)
    advance.add_argument("--owner", required=True)
    advance.add_argument("--receipt-ref", required=True)
    advance.add_argument("--artifact-hash", action="append", default=[], metavar="NAME=SHA256")
    stop = sub.add_parser("stop")
    stop.add_argument("--state", required=True)
    stop.add_argument("--generation", required=True, type=int)
    stop.add_argument("--as", dest="state_name", required=True,
                      choices=("paused", "crashed", "orphaned", "partial_time_limit",
                               "blocked", "deferred_needs_user"))
    stop.add_argument("--reason", required=True)
    deadline = sub.add_parser("set-deadline")
    deadline.add_argument("--state", required=True)
    deadline.add_argument("--generation", required=True, type=int)
    deadline.add_argument("--time-limit", required=True, help="e.g. 2h, 4h, 8h")
    deadline.add_argument("--owner", required=True)
    deadline.add_argument("--close-reserve-s", type=int,
                          help="optional override; defaults to 25/40/60 minutes for 2/4/8h")
    child_dispatch = sub.add_parser("dispatch-child")
    child_dispatch.add_argument("--state", required=True)
    child_dispatch.add_argument("--generation", required=True, type=int)
    child_dispatch.add_argument("--attempt-id", required=True)
    child_dispatch.add_argument("--role", required=True)
    child_dispatch.add_argument("--input-ref", required=True)
    child_dispatch.add_argument("--checkpoint-ref", required=True)
    child_dispatch.add_argument("--deadline-at", required=True)
    child_settle = sub.add_parser("settle-child")
    child_settle.add_argument("--state", required=True)
    child_settle.add_argument("--generation", required=True, type=int)
    child_settle.add_argument("--attempt-id", required=True)
    child_settle.add_argument("--as", dest="state_name", required=True,
                              choices=CHILD_STATES[1:])
    child_settle.add_argument("--receipt-ref", required=True)
    child_settle.add_argument("--adopter-role")
    begin = sub.add_parser("begin-finalization")
    begin.add_argument("--state", required=True)
    begin.add_argument("--generation", required=True, type=int)
    finalize = sub.add_parser("finalize")
    finalize.add_argument("--state", required=True)
    finalize.add_argument("--generation", required=True, type=int)
    reopen = sub.add_parser("reopen",
                            help="continue a CLOSED run as a new generation, keeping its lineage; "
                                 "the new session must set its own deadline at entry")
    reopen.add_argument("--state", required=True)
    reopen.add_argument("--generation", required=True, type=int)
    reopen.add_argument("--reason", required=True,
                        help="what the closed generation left unfinished")
    reopen.add_argument("--owner", required=True)
    check = sub.add_parser("check")
    check.add_argument("--state", required=True)
    check.add_argument("--strict", action="store_true")
    alive = sub.add_parser("liveness",
                           help="is this owner progressing, waiting, finished, or out of its own "
                                "declared time -- read from the state it declared, never inferred "
                                "from file mtimes or a fixed silence interval")
    alive.add_argument("--state", required=True)
    sub.add_parser("selftest")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest or args.command == "selftest":
        return _selftest()
    try:
        if args.command == "init":
            result = init_state(args.state, args.run_id)
        elif args.command == "open":
            result = open_obligation(args.state, args.kind, args.reason, args.obligation_id)
        elif args.command == "done":
            result = complete_obligation(args.state, args.obligation, args.generation,
                                         args.evidence_ref)
        elif args.command == "heartbeat":
            result = heartbeat(args.state, args.phase, args.owner, args.next_step,
                               lease=args.lease, last_artifact=args.last_artifact)
        elif args.command == "transition":
            hashes = {}
            for item in args.artifact_hash:
                if "=" not in item:
                    raise ValueError("--artifact-hash must use NAME=SHA256")
                name, value = item.split("=", 1)
                hashes[name] = value
            result = transition(args.state, args.generation, args.from_stage, args.to_stage,
                                args.owner, args.receipt_ref, hashes)
        elif args.command == "stop":
            result = set_terminal_state(args.state, args.generation, args.state_name, args.reason)
        elif args.command == "set-deadline":
            result = set_deadline(args.state, args.generation, args.time_limit, args.owner,
                                  args.close_reserve_s)
        elif args.command == "dispatch-child":
            result = dispatch_child(args.state, args.generation, args.attempt_id, args.role,
                                    args.input_ref, args.checkpoint_ref, args.deadline_at)
        elif args.command == "settle-child":
            result = settle_child(args.state, args.generation, args.attempt_id, args.state_name,
                                  args.receipt_ref, adopter_role=args.adopter_role)
        elif args.command == "begin-finalization":
            result = begin_finalization(args.state, args.generation)
        elif args.command == "finalize":
            result = finalize_state(args.state, args.generation)
        elif args.command == "reopen":
            result = reopen_state(args.state, args.generation, args.reason, args.owner)
        elif args.command == "check":
            errors = validate_state(_load(args.state))
            print(json.dumps({"ok": not errors, "findings": errors}, indent=2))
            return 1 if args.strict and errors else 0
        elif args.command == "liveness":
            report = liveness(_load(args.state))
            print(json.dumps(report, indent=2))
            # Non-zero ONLY when the owner's own declaration says something has to happen. A quiet
            # owner waiting on a child exits 0, because a supervisor that treats waiting as failure
            # is the loop this reader replaces.
            return 1 if report["act_now"] else 0
        else:
            parser.error("a subcommand is required")
            return 2
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[run_state] ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
