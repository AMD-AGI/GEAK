#!/usr/bin/env python3
"""Ask local Eikos one versioned, typed decision about a frozen state, and record it.

This is the boundary that a carrier agent invokes with ONE fixed command. The command, and the state
inside it, are built by the workflow's code; the carrier only runs it and relays what this prints.
Everything that matters happens here, not in the model:

  - the state arrives as JSON in --state-json, is validated against the decision's required fields,
    and is echoed back byte-for-byte (`state_raw`) so the caller can check the relay was faithful;
  - the questions, options and threshold come from a versioned file in eikos_questions/, never from
    the caller or the carrier;
  - the request goes through eikos_router.post_systemone (loopback only, no proxies, no redirects,
    bounded read), and every answer is validated before use;
  - the FIRST valid decision persisted for a logical key is the decision for that key: a later
    execution with the same key gets that same decision back (status ok, reused true) instead of
    asking again, so concurrent or repeated callers can never return different choices;
  - every execution, failed or not, appends an attempt receipt.

Failures are explicit statuses, never a silent default: state_invalid, refused (non-loopback URL),
unavailable, timeout, malformed, write_failed. A failed attempt is receipted but is NOT persisted as the key's decision,
so a later attempt may still record a real one (shadow output controls nothing, so this costs no
determinism the caller relies on).

Always exits 0 and prints exactly one JSON object on stdout: a carrier must always have something
faithful to relay.

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
import socket
import sys
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eikos_router as er  # noqa: E402  (transport protections, lock, model identity)

QUESTIONS_DIR = os.path.join(HERE, "eikos_questions")
MAX_STATE_CHARS = 8000


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_spec(decision: str) -> tuple:
    """(spec, sha256 of the spec file's canonical form). The decision name must be a file here."""
    if not decision or "/" in decision or decision.startswith("."):
        raise ValueError("bad decision name %r" % decision)
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


def _first_persisted(index_path: str, key: str, value: dict) -> tuple:
    """Insert-if-absent under the lock. Returns (decision to use, persisted?, reused?)."""
    with er._Lock(index_path) as lock:
        if not lock.held:
            return value, False, False
        index = er.load_cache(index_path)
        prior = index.get(key)
        if isinstance(prior, dict) and prior.get("status") == "ok":
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
    env = {"decision": decision, "status": None, "attempt_id": uuid.uuid4().hex, "logical_key": None,
           "reused": False, "persisted": False, "state_raw": state_raw, "state_sha256": sha256(state_raw),
           "questions_version": None, "questions_sha256": None, "policy_version": None,
           "model_id": er.model_identity(), "options": None, "choice": None, "confidence": None,
           "distribution": None, "would_be_action": None, "error": None}

    def finish(status, error=None, attempt_extra=None):
        env["status"], env["error"] = status, error
        env["elapsed_ms"] = round((time.monotonic() - t0) * 1000, 1)
        if receipt_dir:
            _, attempts_path = receipt_paths(receipt_dir, decision)
            line = {k: env[k] for k in env if k != "state_raw"}
            line.update(attempt_extra or {})
            line["scope"], line["round"] = scope, round_no
            if not _append(attempts_path, line) and status == "ok" and not env["persisted"]:
                env["status"], env["error"] = "write_failed", "attempt receipt could not be written"
        return env

    try:
        spec, spec_sha = load_spec(decision)
    except (OSError, ValueError) as exc:
        return finish("state_invalid", "question file: %s" % exc)
    env.update(questions_version=spec["version"], questions_sha256=spec_sha,
               policy_version=spec["policy"].get("version"), options=list(spec["options"]))
    if len(state_raw) > MAX_STATE_CHARS:
        return finish("state_invalid", "state has %d chars, over %d" % (len(state_raw), MAX_STATE_CHARS))
    try:
        state = json.loads(state_raw)
    except ValueError as exc:
        return finish("state_invalid", "state is not JSON: %s" % exc)
    if not isinstance(state, dict):
        return finish("state_invalid", "state is not a JSON object")
    missing = [k for k in spec["required_state"] if k not in state]
    if missing:
        return finish("state_invalid", "state lacks required field(s): %s" % ", ".join(missing))
    canon = canonical(state)
    env["logical_key"] = sha256(canonical({
        "decision": decision, "scope": scope, "round": round_no, "state": sha256(canon),
        "questions": spec_sha, "policy": spec["policy"].get("version"), "model_id": env["model_id"]}))
    index_path, _ = receipt_paths(receipt_dir, decision) if receipt_dir else (None, None)

    # Already decided for this key? Return that decision; never ask twice.
    if index_path:
        prior = er.load_cache(index_path).get(env["logical_key"])
        if isinstance(prior, dict) and prior.get("status") == "ok":
            env.update({k: prior[k] for k in ("choice", "confidence", "distribution", "would_be_action")})
            env.update(reused=True, persisted=True, first_attempt_id=prior.get("attempt_id"))
            return finish("ok", attempt_extra={"canonical_state": canon})

    if not er.is_loopback(url) and os.environ.get("GEAK_EIKOS_ALLOW_REMOTE", "0") != "1":
        return finish("refused", "Eikos URL %s is not a loopback address" % url, {"canonical_state": canon})
    try:
        body = er.post_systemone(url, canon, spec["questions"], timeout_s)
    except (socket.timeout, TimeoutError) as exc:
        return finish("timeout", "%s: %s" % (type(exc).__name__, exc), {"canonical_state": canon})
    except OSError as exc:  # URLError and connection failures are OSError subclasses
        reason = getattr(exc, "reason", exc)
        status = "timeout" if isinstance(reason, (socket.timeout, TimeoutError)) else "unavailable"
        return finish(status, "%s: %s" % (type(exc).__name__, reason), {"canonical_state": canon})
    except (ValueError, RuntimeError) as exc:
        return finish("malformed", "%s: %s" % (type(exc).__name__, exc), {"canonical_state": canon})

    answers = body["answers"]
    problem = validate(answers, spec)
    if problem:
        return finish("malformed", "invalid answers: " + problem, {"canonical_state": canon})
    primary = answers[spec["primary_question"]]
    choice, conf = primary["choice"], primary["probabilities"][primary["choice"]]
    min_conf = float(spec["policy"].get("min_confidence", 1.0))
    decided = {"status": "ok", "attempt_id": env["attempt_id"], "choice": choice, "confidence": conf,
               "distribution": dict(primary["probabilities"]),
               "would_be_action": choice if conf >= min_conf else "abstain",
               "answers": {qid: answers[qid] for qid in spec["questions"]},
               "canonical_state": canon, "scope": scope, "round": round_no,
               "questions_version": spec["version"], "policy_version": spec["policy"].get("version"),
               "model_id": env["model_id"]}
    used, persisted, reused = (_first_persisted(index_path, env["logical_key"], decided)
                               if index_path else (decided, False, False))
    env.update({k: used[k] for k in ("choice", "confidence", "distribution", "would_be_action")})
    env.update(persisted=persisted, reused=reused)
    if reused:
        env["first_attempt_id"] = used.get("attempt_id")
    if index_path and not persisted:
        return finish("write_failed", "decision could not be persisted", {"canonical_state": canon})
    return finish("ok", attempt_extra={"canonical_state": canon, "answers": decided["answers"]})


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
        print(canonical({"status": "state_invalid", "error": "bad arguments"}))
        return 0
    receipt_dir = a.receipt_dir or os.path.join(a.scope, "eikos")
    try:
        env = decide(a.decision, a.scope, a.round, a.state_json, receipt_dir, er.eikos_url(), a.timeout_s)
    except Exception as exc:  # noqa: BLE001 - the carrier must always get one JSON object
        env = {"decision": a.decision, "status": "malformed", "error": "%s: %s" % (type(exc).__name__, exc),
               "state_raw": a.state_json}
    print(canonical(env))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
