#!/usr/bin/env python3
"""Ask local Eikos one versioned, typed decision about a frozen state, and record it.

This is the boundary that a carrier agent invokes with ONE fixed command. The command, and the state
inside it, are built by the workflow's code; the carrier only runs it and relays what this prints.
Everything that matters happens here, not in the model:

  - the state arrives as JSON in --state-json and is validated against the decision's required
    fields. The envelope carries its SHA-256 (`state_sha256`), not the state itself: a carrier asked
    to relay a JSON string inside JSON was observed to parse it into an object (synthetic native check
    wf_c010449f-f41), so the caller compares a short hex digest it computes itself. The raw state is
    kept in the attempt receipt;
  - the questions, options and threshold come from a versioned file in eikos_questions/, never from
    the caller or the carrier;
  - the request goes through eikos_router.post_systemone (loopback only, no proxies, no redirects,
    bounded read), and every answer is validated before use;
  - the FIRST valid decision persisted for a logical key is the decision for that key: a later
    execution with the same key gets that same decision back (status ok, reused true) instead of
    asking again, so concurrent or repeated callers can never return different choices;
  - every execution, failed or not, appends an attempt receipt.

Failures are explicit statuses, never a silent default: state_invalid, refused (non-loopback URL),
unavailable, timeout, malformed, write_failed (the decision could not be persisted). A failed attempt is receipted but is NOT persisted as the key's decision,
so a later attempt may still record a real one (shadow output controls nothing, so this costs no
determinism the caller relies on).

Always exits 0 and prints exactly one COMPACT JSON object on stdout (ENVELOPE_FIELDS): only what the
caller acts on, so the carrier copies as little as possible. Everything else (distribution, versions,
timings, the raw state, what this attempt itself observed) is in the attempt receipt, joined by
attempt_id.

WIRE CONTRACT. Every field covered by `decision_sha256` is a string, a boolean or null — never a
number — so both sides hash the same bytes: Python writes 1.0 where JavaScript writes 1, and a hash
over numbers would flag a valid answer as altered. `confidence` therefore travels as Python's repr
string ("0.92", "1.0"); parse it if you need the value. The digest lets the caller detect a carrier
that alters the decision — a carrier was observed to add a field it was asked to relay unchanged
(wf_f81b3428-784).

RECEIPTS. `receipt` reports whether THIS attempt's receipt line was written (written / failed /
not_written), separately from `status`, which is about the decision. A decision can be valid and
reused while its attempt receipt failed; that is reported, never hidden.

    python3 eikos_decide.py --decision round_continue --scope <EVAL_DIR> --round 2 \\
        --state-json '{"round": 2, ...}' [--receipt-dir <dir>] [--timeout-s 30]

Receipts: <receipt-dir or scope/eikos>/<decision>.index.json (first persisted decision per key) and
<decision>.attempts.jsonl (one line per execution, with the full canonical state).
"""
import argparse
import hashlib
import json
import math
import os
import re
import socket
import sys
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eikos_router as er  # noqa: E402  (transport protections, lock, model identity)

QUESTIONS_DIR = os.path.join(HERE, "eikos_questions")
# What the carrier relays. DECISION_FIELDS are covered by decision_sha256 and are str/bool/None only.
DECISION_FIELDS = ("attempt_id", "decision_attempt_id", "logical_key", "status", "choice", "confidence",
                   "would_be_action", "reused", "persisted", "receipt")
ENVELOPE_FIELDS = DECISION_FIELDS + ("state_sha256", "error")
DECISION_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
MAX_STATE_CHARS = 8000
ATTEMPT_ID = re.compile(r"^[0-9a-f]{32}$")


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_spec(decision: str) -> tuple:
    """(spec, sha256 of the spec file's canonical form). The decision name must be a file here."""
    if not isinstance(decision, str) or not DECISION_NAME.match(decision):
        raise ValueError("bad decision name %r" % (decision,))
    with open(os.path.join(QUESTIONS_DIR, decision + ".json"), encoding="utf-8") as fh:
        spec = json.load(fh)
    for key in ("version", "required_state", "primary_question", "options", "questions", "policy"):
        if key not in spec:
            raise ValueError("question file lacks %r" % key)
    return spec, sha256(canonical(spec))


def _prob(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and 0.0 <= x <= 1.0


def validate(answers: dict, spec: dict) -> str:
    """Why these answers are unusable, or "" when every declared question has a valid answer."""
    for qid, q in spec["questions"].items():
        a = answers.get(qid)
        if not isinstance(a, dict):
            return "%s: answer missing" % qid
        if q["type"] == "choice":
            opts = list(q["criteria"])
            probs, choice, conf = a.get("probabilities"), a.get("choice"), a.get("confidence")
            if choice not in opts:
                return "%s: choice %r not offered" % (qid, choice)
            if not isinstance(probs, dict) or set(probs) != set(opts):
                return "%s: probabilities do not cover exactly the offered options" % qid
            if not all(_prob(v) for v in probs.values()):
                return "%s: probabilities are not all numbers in [0, 1]" % qid
            if abs(sum(probs.values()) - 1.0) > 0.03:
                return "%s: probabilities sum to %.4f" % (qid, sum(probs.values()))
            if probs[choice] < max(probs.values()) - 1e-6:
                return "%s: choice is not the most probable option" % qid
            if conf is not None and (not _prob(conf) or abs(conf - probs[choice]) > 1e-3):
                return "%s: confidence disagrees with the chosen option's probability" % qid
        elif q["type"] in ("noul", "boolean"):
            if not _prob(a.get("probability")):
                return "%s: probability missing or not a number in [0, 1]" % qid
        else:
            return "%s: unsupported question type %r" % (qid, q["type"])
    return ""


def _wire(v):
    """Wire form of a digested field: str, bool or None. Numbers become Python repr strings."""
    if v is None or isinstance(v, (str, bool)):
        return v
    if isinstance(v, (int, float)):
        return repr(v)
    raise TypeError("decision field of type %s cannot be put on the wire" % type(v).__name__)


def decision_digest(env: dict) -> str:
    return sha256(canonical({k: _wire(env.get(k)) for k in DECISION_FIELDS}))


def compact(env: dict) -> dict:
    out = {k: (_wire(env.get(k)) if k in DECISION_FIELDS else env.get(k)) for k in ENVELOPE_FIELDS}
    out["decision_sha256"] = decision_digest(out)
    return out


def _reject_constant(name):
    raise ValueError("non-standard JSON constant %s" % name)


def _is_int(x, lo):
    return isinstance(x, int) and not isinstance(x, bool) and x >= lo


def _is_num(x, lo, strict=False):
    if not isinstance(x, (int, float)) or isinstance(x, bool):
        return False
    try:
        x = float(x)                                # an int too large for a float is not a number here
    except OverflowError:
        return False
    return math.isfinite(x) and (x > lo if strict else x >= lo)


def _one_of(x, allowed) -> bool:
    return isinstance(x, str) and x in allowed


def _nonfinite(v) -> bool:
    """True if any number anywhere in v is not a finite float (1e999 parses to inf; 10**400 overflows)."""
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return False
    if isinstance(v, (int, float)):
        return not _is_num(v, -math.inf)
    if isinstance(v, list):
        return any(_nonfinite(x) for x in v)
    if isinstance(v, dict):
        return any(_nonfinite(x) for x in v.values())
    return True


SPECIALTIES = {"algorithm", "memory", "compute", "host_runtime", "deep_explore"}
CLOCK_SOURCES = {"pre_plan_clock", "pre_replan_clock", "no_deadline"}
# The object kernel_lane.js writes after a measured round (see its EIKOS_SHADOW outcome block).
OUTCOME_CHECKS = {
    "verified_candidates": lambda v: _is_int(v, 0),
    "winner_speedup": lambda v: v is None or _is_num(v, 0, strict=True),
    "improved": lambda v: isinstance(v, bool),
    "made_progress": lambda v: isinstance(v, bool),
    "commit_reported": lambda v: v == "not_captured",
    "tracked_incumbent_after": lambda v: _is_num(v, 0, strict=True),
}


def _outcome_ok(o) -> bool:
    if o == "none":
        return True
    return (isinstance(o, dict) and set(o) == set(OUTCOME_CHECKS)
            and all(check(o[k]) for k, check in OUTCOME_CHECKS.items()))


def _round_continue_state(state: dict, round_no: int) -> str:
    """Types, ranges and consistency for round_continue's fixed state. "unknown" is accepted only
    where the lane legitimately lacks a value."""
    checks = [
        ("round", _is_int(state["round"], 1) and state["round"] == round_no, "an integer >= 1 equal to --round"),
        ("directions_used", _is_int(state["directions_used"], 0), "an integer >= 0"),
        ("directions_budget", _is_int(state["directions_budget"], 1), "an integer >= 1"),
        ("tracked_incumbent_speedup", _is_num(state["tracked_incumbent_speedup"], 0, strict=True), "a finite number > 0"),
        ("best_seen_speedup", _is_num(state["best_seen_speedup"], 0), "a finite number >= 0"),
        ("rounds_without_improvement", _is_int(state["rounds_without_improvement"], 0), "an integer >= 0"),
        ("rounds_without_improvement_limit", _is_int(state["rounds_without_improvement_limit"], 1), "an integer >= 1"),
        ("last_round_outcome", _outcome_ok(state["last_round_outcome"]), '"none" or the lane\'s measured-outcome object'),
        ("specialties_dispatched", isinstance(state["specialties_dispatched"], list)
         and all(_one_of(x, SPECIALTIES) for x in state["specialties_dispatched"])
         and len(set(state["specialties_dispatched"])) == len(state["specialties_dispatched"]), "a list of distinct known specialties"),
        ("minutes_left", state["minutes_left"] == "unknown" or _is_int(state["minutes_left"], 0), '"unknown" or an integer >= 0'),
        ("minutes_left_source", _one_of(state["minutes_left_source"], CLOCK_SOURCES), "one of " + ", ".join(sorted(CLOCK_SOURCES))),
        ("noise_band", state["noise_band"] == "unknown" or _is_num(state["noise_band"], 0), '"unknown" or a finite number >= 0'),
        ("min_improve", _is_num(state["min_improve"], 0), "a finite number >= 0"),
        ("progress_delta", _is_num(state["progress_delta"], -1, strict=True), "a finite number > -1 (kernel_lane.js allows negatives)"),
    ]
    bad = ["%s must be %s" % (k, why) for k, ok, why in checks if not ok]
    return "; ".join(bad)


STATE_VALIDATORS = {"round_continue": _round_continue_state}


def derive(answers: dict, spec: dict) -> dict:
    """The decision the policy derives from validated answers."""
    primary = answers[spec["primary_question"]]
    choice = primary["choice"]
    conf = primary["probabilities"][choice]
    min_conf = float(spec["policy"].get("min_confidence", 1.0))
    return {"choice": choice, "confidence": conf, "distribution": dict(primary["probabilities"]),
            "would_be_action": choice if conf >= min_conf else "abstain"}


def valid_persisted(entry, ctx: dict) -> bool:
    """Reuse a persisted decision only if it still matches this request's identity and its own recorded
    answers re-derive exactly the decision it claims. A hand-edited or stale entry is never reused."""
    if not isinstance(entry, dict) or entry.get("status") != "ok":
        return False
    if not isinstance(entry.get("attempt_id"), str) or not ATTEMPT_ID.match(entry["attempt_id"]):
        return False
    if not isinstance(entry.get("round"), int) or isinstance(entry.get("round"), bool):
        return False
    for k in ("scope", "round", "canonical_state", "questions_sha256", "policy_version", "model_id", "endpoint"):
        if entry.get(k) != ctx[k]:
            return False
    # The entry must also sit under the key its own identity produces (no copying between keys).
    if logical_key(ctx["decision"], entry["scope"], entry["round"], entry["canonical_state"],
                   entry["questions_sha256"], entry["policy_version"], entry["model_id"],
                   entry["endpoint"]) != ctx["logical_key"]:
        return False
    answers = entry.get("answers")
    if not isinstance(answers, dict) or validate(answers, ctx["spec"]):
        return False
    again = derive(answers, ctx["spec"])
    return all(entry.get(k) == again[k] for k in ("choice", "confidence", "distribution", "would_be_action"))


def logical_key(decision, scope, round_no, canon, questions_sha, policy_version, model_id, endpoint) -> str:
    return sha256(canonical({
        "decision": decision, "scope": scope, "round": round_no, "state": sha256(canon),
        "questions": questions_sha, "policy": policy_version, "model_id": model_id, "endpoint": endpoint}))


def receipt_paths(receipt_dir: str, decision: str) -> tuple:
    return (os.path.join(receipt_dir, decision + ".index.json"),
            os.path.join(receipt_dir, decision + ".attempts.jsonl"))


def _append(path: str, line: dict) -> bool:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(canonical(line) + "\n")
        return True
    except OSError:
        return False


def _first_persisted(index_path: str, key: str, value: dict, ctx: dict) -> tuple:
    """Insert-if-absent under the lock. A valid prior decision wins; an invalid one is replaced.
    Returns (decision to use, persisted?, reused?)."""
    with er._Lock(index_path) as lock:
        if not lock.held:
            return value, False, False
        index = er.load_cache(index_path)
        prior = index.get(key)
        if valid_persisted(prior, ctx):
            return prior, True, True
        index[key] = value
        try:
            tmp = index_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(index, fh, indent=2, sort_keys=True)
            os.replace(tmp, index_path)
            return value, True, False
        except OSError:
            return value, False, False


def decide(decision: str, scope: str, round_no: int, state_raw: str, receipt_dir: str,
           url: str, timeout_s: float) -> dict:
    t0 = time.monotonic()
    attempt_id = uuid.uuid4().hex
    env = {"status": None, "attempt_id": attempt_id, "decision_attempt_id": None, "logical_key": None,
           "choice": None, "confidence": None, "would_be_action": None, "reused": False,
           "persisted": False, "receipt": "not_written", "state_sha256": sha256(state_raw), "error": None}
    observed = {"transport": "not_attempted"}      # what THIS attempt saw, kept apart from the decision
    extra = {}
    name_ok = isinstance(decision, str) and bool(DECISION_NAME.match(decision))

    def finish(status, error=None):
        env["status"], env["error"] = status, error
        if receipt_dir and name_ok:                 # never build a path from an unvalidated name
            _, attempts_path = receipt_paths(receipt_dir, decision)
            line = dict(env, decision=decision, scope=scope, round=round_no, state_raw=state_raw,
                        elapsed_ms=round((time.monotonic() - t0) * 1000, 1), observed=observed, **extra)
            line["receipt"] = "written"
            env["receipt"] = "written" if _append(attempts_path, line) else "failed"
        return env

    if not name_ok:
        return finish("state_invalid", "bad decision name %r" % (decision,))
    try:
        spec, spec_sha = load_spec(decision)
    except (OSError, ValueError) as exc:
        return finish("state_invalid", "question file: %s" % exc)
    extra.update(questions_version=spec["version"], questions_sha256=spec_sha,
                 policy_version=spec["policy"].get("version"), model_id=er.model_identity(), endpoint=url)
    if len(state_raw) > MAX_STATE_CHARS:
        return finish("state_invalid", "state has %d chars, over %d" % (len(state_raw), MAX_STATE_CHARS))
    try:
        state = json.loads(state_raw, parse_constant=_reject_constant)
    except ValueError as exc:
        return finish("state_invalid", "state is not standard JSON: %s" % exc)
    if not isinstance(state, dict):
        return finish("state_invalid", "state is not a JSON object")
    missing = [k for k in spec["required_state"] if k not in state]
    if missing:
        return finish("state_invalid", "state lacks required field(s): %s" % ", ".join(missing))
    if _nonfinite(state):
        return finish("state_invalid", "state holds a non-finite or out-of-range number")
    checker = STATE_VALIDATORS.get(decision)
    try:
        problem = checker(state, round_no) if checker else ""
    except (TypeError, ValueError, OverflowError) as exc:   # backstop: still a receipted refusal
        problem = "%s: %s" % (type(exc).__name__, exc)
    if problem:
        return finish("state_invalid", "state: " + problem)
    canon = canonical(state)
    extra["canonical_state"] = canon
    ctx = {"spec": spec, "decision": decision, "scope": scope, "round": round_no, "canonical_state": canon,
           "questions_sha256": spec_sha, "policy_version": spec["policy"].get("version"),
           "model_id": extra["model_id"], "endpoint": url}
    env["logical_key"] = ctx["logical_key"] = logical_key(
        decision, scope, round_no, canon, spec_sha, ctx["policy_version"], ctx["model_id"], url)
    index_path = receipt_paths(receipt_dir, decision)[0] if receipt_dir else None

    def adopt(entry, reused, persisted):
        env.update({k: entry[k] for k in ("choice", "confidence", "would_be_action")})
        env.update(decision_attempt_id=entry["attempt_id"], reused=reused, persisted=persisted)
        extra["distribution"] = entry["distribution"]

    # Already decided for this key, and the record still checks out? Return it; never ask twice.
    if index_path:
        prior = er.load_cache(index_path).get(env["logical_key"])
        if valid_persisted(prior, ctx):
            adopt(prior, True, True)
            return finish("ok")
        if prior is not None:
            observed["stale_index_entry"] = "ignored: did not re-validate"

    if not er.is_loopback(url) and os.environ.get("GEAK_EIKOS_ALLOW_REMOTE", "0") != "1":
        return finish("refused", "Eikos URL %s is not a loopback address" % url)
    try:
        body = er.post_systemone(url, canon, spec["questions"], timeout_s)
    except (socket.timeout, TimeoutError) as exc:
        observed["transport"] = "timeout"
        return finish("timeout", "%s: %s" % (type(exc).__name__, exc))
    except OSError as exc:  # URLError and connection failures are OSError subclasses
        reason = getattr(exc, "reason", exc)
        status = "timeout" if isinstance(reason, (socket.timeout, TimeoutError)) else "unavailable"
        observed["transport"] = status
        return finish(status, "%s: %s" % (type(exc).__name__, reason))
    except (ValueError, RuntimeError) as exc:
        observed["transport"] = "malformed"
        return finish("malformed", "%s: %s" % (type(exc).__name__, exc))

    answers = body["answers"]
    observed["transport"] = "answered"
    observed["answers"] = answers
    problem = validate(answers, spec)
    if problem:
        return finish("malformed", "invalid answers: " + problem)
    mine = derive(answers, spec)
    observed.update(choice=mine["choice"], confidence=mine["confidence"])
    decided = dict(mine, status="ok", attempt_id=attempt_id,
                   answers={qid: answers[qid] for qid in spec["questions"]},
                   scope=scope, round=round_no, **{k: ctx[k] for k in
                   ("canonical_state", "questions_sha256", "policy_version", "model_id", "endpoint")})
    if not index_path:
        adopt(decided, False, False)
        return finish("write_failed", "no receipt directory: the decision cannot be persisted")
    used, persisted, reused = _first_persisted(index_path, env["logical_key"], decided, ctx)
    adopt(used, reused, persisted)
    if not persisted:
        return finish("write_failed", "decision could not be persisted")
    return finish("ok")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--decision", required=True)
    ap.add_argument("--scope", required=True, help="run/lane identity, e.g. the lane's EVAL_DIR")
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--state-json", required=True)
    ap.add_argument("--receipt-dir", default=None, help="default: <scope>/eikos")
    ap.add_argument("--timeout-s", type=float, default=30.0)
    try:
        a = ap.parse_args(argv)
    except SystemExit:
        print(canonical(compact({"status": "state_invalid", "error": "bad arguments"})))
        return 0
    receipt_dir = a.receipt_dir or os.path.join(a.scope, "eikos")
    try:
        env = decide(a.decision, a.scope, a.round, a.state_json, receipt_dir, er.eikos_url(), a.timeout_s)
    except Exception as exc:  # noqa: BLE001 - the carrier must always get one JSON object
        env = {"status": "malformed", "error": "%s: %s" % (type(exc).__name__, exc),
               "state_sha256": sha256(a.state_json)}
    print(canonical(compact(env)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
