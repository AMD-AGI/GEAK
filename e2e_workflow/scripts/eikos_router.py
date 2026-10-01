#!/usr/bin/env python3
"""Model/effort router for GEAK agent spawns, optionally backed by Eikos, a local typed-decision model.

WHY THIS EXISTS
---------------
`e2e_workflow.js:ablEffortFor` already routes effort with a hardcoded label regex. This module
generalises that single decision into `{effort, model}` and lets an *optional* Eikos evaluation
make it instead of the regex.

EIKOS
-----
Eikos-27B (huggingface.co/caiovicentino1/Eikos-27B) is an open-weights "System-1" decision model:
one forward pass per question, reading logits over the option letters only, so every answer
carries a real probability distribution and a confidence. It speaks the same typed-question
vocabulary Jev did (`boolean`/`noul`, `choice`, `score`) at `POST /v1/systemone` of its
`serve.py` API, and it runs on our own hardware: no API key, no third party, no retention terms.
Caveats that bound what its answers mean here:
  - It was trained for finance and trade-rule decisions. Routing GEAK agent tasks is outside
    that domain; nothing here has measured how well it does.
  - Its context is 16,384 tokens, trained to 12k. The state sent is capped far below that.
  - Its calibration temperature is T=1 by the publisher's choice, not a fit: temperatures fitted
    on their held-out sets made hard items over-confident, so they kept T=1 for want of a
    realistic calibration set. Nothing is calibrated for GEAK; the gates below are starting
    points.
  - It needs vLLM >= 0.30.0; older builds give wrong answers on batched long requests.

THREE PROPERTIES THIS FILE GUARANTEES (see research/09_jev_typesafe_routing.md sections 10-11)
  1. OFF BY DEFAULT. Without GEAK_EIKOS_ROUTER=1 the decision is the static one, byte-identical
     to today, and no request is made.
  2. RECORDED. Every outcome for an eligible request -- an Eikos decision OR a failure -- is
     written to `<cache-dir>/eikos_router_cache.json`, keyed by the policy, question set,
     endpoint, declared model identity and a digest of the FULL task. The first outcome
     persisted for a key wins: concurrent callers evaluating the same key all return that one,
     never their own. Recorded decisions are re-derived from their stored answers before reuse.
     This is NOT a certified replay guarantee: if the cache cannot be written the next call asks
     again and may differ; a swapped checkpoint behind the same URL is only distinguished when
     GEAK_EIKOS_MODEL_ID names it; and whether a workflow runtime re-executes this hook on resume
     is not documented.
  3. NON-FATAL AND NON-INTRUSIVE. Every failure -- disabled, retry, oversized or empty task,
     unreachable server, malformed or out-of-range answer, unreadable cache -- returns "no
     decision" (model and effort null, source "host"), so the caller's own logic decides exactly
     as it would without this router. This router never substitutes a guess of its own.

It also refuses to send GEAK task text anywhere but a literal loopback address (127.0.0.0/8, ::1 or
`localhost`) over http(s) unless GEAK_EIKOS_ALLOW_REMOTE=1, and it neither follows redirects nor
uses proxy settings: the reason to use a local model is that the state never leaves the host.

PROTOCOL
    echo '<request-json>' | python3 eikos_router.py --cache-dir DIR
  request: {"label": str, "phase": str, "attempt": int, "task": str}
  stdout:  {"model": str|null, "effort": str|null, "tier": str, "source": str, ...}
  A null model/effort means "leave the caller's options untouched".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

# --- Tiers -----------------------------------------------------------------------------------
# Haiku 4.5 is the current Haiku; there is no "Haiku 5" in the pricing reference. Note its 4,096
# token cache minimum -- prompts shorter than that do not cache on it at all.
TIERS = {
    "thinker":  {"model": "claude-opus-5-5",            "effort": None},
    "standard": {"model": "claude-sonnet-5",            "effort": None},
    "cheap":    {"model": "claude-haiku-4-5-20251001",  "effort": "low"},
}
TIER_CRITERIA = {
    "thinker":  "Open-ended authoring, design, adjudication between conflicting results, or "
                "anything acting as the oracle for another agent's output.",
    "standard": "A bounded task with a clear specification, where the result is checked by a "
                "later step but the work still needs judgement.",
    "cheap":    "Mechanical, fully specified work whose output is validated objectively and "
                "immediately (a compile, a benchmark, a file reclaim, a fixed-format extraction).",
}

# Which labels are eligible at all is decided deterministically by the host (e2e_workflow.js,
# ABL_CHEAP_LABELS), not here: a model's opinion is never what makes a task eligible.

# Thresholds. Vendor guidance is explicit that real cutoffs must be derived from labeled examples;
# these are the article's illustrative starting points and are NOT calibrated against GEAK.
MIN_TIER_CONFIDENCE = 0.6
MIN_CHOICE_PROBABILITY = 0.7
MIN_REVERSIBLE_PROBABILITY = 0.8

CACHE_BASENAME = "eikos_router_cache.json"
# The Eikos serve.py API (v1.2). One request answers every question against one shared state,
# which the server encodes once. The vLLM engine behind it is not addressed directly.
EIKOS_DEFAULT_URL = "http://127.0.0.1:8000"
EIKOS_PATH = "/v1/systemone"
STATE_TOKEN_BUDGET_CHARS = 8000  # ~2k tokens; Eikos serves 16,384 and was trained to 12k.


POLICY_VERSION = "eikos-router/1"
MAX_RESPONSE_BYTES = 64 * 1024
LOCK_TIMEOUT_S = 5.0


def no_decision(reason: str, **extra) -> dict:
    """Leave the caller's options untouched: its own logic decides, exactly as without us."""
    return {"model": None, "effort": None, "tier": None, "source": "host", "reason": reason,
            "policy": POLICY_VERSION, **extra}


def policy_digest() -> str:
    """Identity of everything that shapes a decision besides the input: questions, tiers, gates."""
    blob = json.dumps({"policy": POLICY_VERSION, "questions": build_questions(), "tiers": TIERS,
                       "gates": [MIN_TIER_CONFIDENCE, MIN_CHOICE_PROBABILITY,
                                 MIN_REVERSIBLE_PROBABILITY]},
                      sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def model_identity() -> str:
    """The checkpoint behind the URL, as declared by the operator (e.g. an HF revision). The
    server's own /health reports only a path, so it cannot identify a swapped checkpoint."""
    return os.environ.get("GEAK_EIKOS_MODEL_ID", "") or "undeclared"


def request_key(req: dict, url: str = "") -> str:
    """Stable, order-independent hash of the routing inputs. The task enters as a digest of its
    FULL text, so two tasks that share a prefix never share a decision."""
    task = req.get("task") or ""
    canonical = json.dumps(
        {
            "policy": policy_digest(),
            "url": url,
            "model_id": model_identity(),
            "label": req.get("label", ""),
            "phase": req.get("phase", ""),
            "attempt": int(req.get("attempt", 0) or 0),
            "task_sha256": hashlib.sha256(task.encode("utf-8")).hexdigest(),
            "task_chars": len(task),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_cache(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


class _Lock:
    """Exclusive advisory lock on `<path>.lock`, so concurrent helpers cannot lose entries.
    Where locking is unavailable the lock is skipped, and that is reported by the caller."""

    def __init__(self, path: str):
        self.path, self.fh, self.held = path + ".lock", None, False

    def __enter__(self):
        try:
            import fcntl  # noqa: PLC0415
            import time  # noqa: PLC0415
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            self.fh = open(self.path, "a")
            deadline = time.monotonic() + LOCK_TIMEOUT_S
            while True:
                try:
                    fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self.held = True
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        break
                    time.sleep(0.05)
        except Exception:
            self.held = False
        return self

    def __exit__(self, *exc):
        try:
            if self.fh:
                self.fh.close()  # closing releases the flock
        except Exception:
            pass
        return False


def record(path: str, key: str, value: dict) -> tuple:
    """Insert-if-absent under the lock. Returns (outcome to use, persisted?). If a valid outcome
    for this key is already stored -- a concurrent caller got there first -- that one is
    returned instead of `value`, so every caller for one key acts on the same decision."""
    with _Lock(path) as lock:
        if not lock.held:
            return value, False
        cache = load_cache(path)
        if valid_recorded(cache.get(key)):
            return cache[key], True
        cache[key] = value
        try:
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(cache, fh, indent=2, sort_keys=True)
            os.replace(tmp, path)
            return value, True
        except Exception:
            return value, False


def valid_recorded(value) -> bool:
    """A recorded outcome is reused only if it has exactly the shape this policy writes, and an
    Eikos decision only if re-deriving it from its own stored answers gives the same result. A
    malformed or hand-edited entry can never authorize a route."""
    if not isinstance(value, dict) or value.get("policy") != POLICY_VERSION:
        return False
    if value.get("source") == "host":
        return value.get("model") is None and value.get("effort") is None
    if value.get("source") != "eikos":
        return False
    try:
        again = decide_from_answers(value.get("answers"), {})
    except Exception:
        return False
    return all(value.get(k) == again.get(k) for k in ("tier", "model", "effort", "escalated"))


def build_questions() -> dict:
    return {
        "tier": {
            "type": "choice",
            "instructions": "Which model tier should handle this agent task?",
            "criteria": dict(TIER_CRITERIA),
        },
        "complexity": {
            "type": "score",
            "instructions": "How much open-ended reasoning does this task require?",
            "criteria": [
                "Mechanical: follow fixed steps, no judgement.",
                "Bounded: one clear goal, the approach is already decided.",
                "Substantive: requires choosing an approach among several.",
                "Open-ended: requires designing, diagnosing, or adjudicating.",
            ],
        },
        "reversible": {
            "type": "boolean",
            "instructions": "If this task is answered badly, will a later automated step detect "
                            "it and allow a cheap retry?",
            "criteria": {
                "true": "A compile, benchmark, correctness gate, or verifier checks the output.",
                "false": "The output is consumed directly, or it is itself the check on "
                         "another agent's work.",
            },
        },
    }


def tier_confidence(tier_ans: dict, meta_confidence: dict, probs: dict) -> tuple:
    """Confidence for the tier question, with the source recorded.

    Eikos sends `confidence` on every answer (the selected option's probability). If a server
    omits it, fall back to the concentration of the distribution itself, so the confidence gate
    is then satisfied by a strictly stronger test rather than silently passing -- and an empty
    `probabilities` yields 0.0, which escalates. `meta_confidence` is kept for callers that pass
    a separate confidence map; Eikos never needs it.
    """
    if isinstance(tier_ans.get("confidence"), (int, float)):
        return float(tier_ans["confidence"]), "answer"
    meta = (meta_confidence or {}).get("tier")
    if isinstance(meta, (int, float)):
        return float(meta), "provider_metadata"
    return (max(probs.values()) if probs else 0.0), "derived_from_probabilities"


def _prob(x) -> bool:
    """A real probability: a finite int/float in [0, 1]. A bool is not a number here."""
    return (isinstance(x, (int, float)) and not isinstance(x, bool)
            and math.isfinite(x) and 0.0 <= x <= 1.0)


def validate_answers(answers) -> str:
    """Return why the answers are unusable, or "" when they are. Untrusted input: a wrong shape
    must never reach the gates, which would otherwise accept out-of-range numbers."""
    if not isinstance(answers, dict):
        return "answers is not an object"
    tier = answers.get("tier")
    if not isinstance(tier, dict):
        return "tier answer missing"
    choice, probs, conf = tier.get("choice"), tier.get("probabilities"), tier.get("confidence")
    if choice not in TIERS:
        return "tier choice %r is not a known tier" % (choice,)
    if not isinstance(probs, dict) or set(probs) != set(TIERS):
        return "tier probabilities do not cover exactly the declared tiers"
    if not all(_prob(v) for v in probs.values()):
        return "tier probabilities are not all numbers in [0, 1]"
    if abs(sum(probs.values()) - 1.0) > 0.03:
        return "tier probabilities sum to %.4f, not ~1" % sum(probs.values())
    if probs[choice] < max(probs.values()) - 1e-6:
        return "tier choice is not the most probable option"
    if conf is not None and (not _prob(conf) or abs(conf - probs[choice]) > 1e-3):
        return "tier confidence disagrees with the chosen option's probability"
    rev = answers.get("reversible")
    if not isinstance(rev, dict) or not _prob(rev.get("probability")):
        return "reversible probability missing or not a number in [0, 1]"
    cx = answers.get("complexity")
    if cx is not None:
        score = (cx or {}).get("score") if isinstance(cx, dict) else None
        if not (isinstance(score, int) and not isinstance(score, bool) and 0 <= score <= 3):
            return "complexity score is not an integer level 0..3"
    return ""


def decide_from_answers(answers: dict, confidence: dict) -> dict:
    """Apply the escalation rule. Anything short of every gate falls back to the thinker.

    Read the distribution defensively: `probabilities` is optional in the contract, values are
    rounded to two decimals so a distribution may total 0.99, and it must not be renormalized.
    """
    problem = validate_answers(answers)
    if problem:
        raise ValueError("invalid Eikos answers: " + problem)
    tier_ans = answers.get("tier") or {}
    choice = tier_ans.get("choice")
    probs = tier_ans.get("probabilities") or {}
    selected_p = probs.get(choice, 0.0) if choice else 0.0
    tier_conf, conf_source = tier_confidence(tier_ans, confidence, probs)
    # A boolean answer carries `probability` = P(true). Eikos also sends a `confidence` for it,
    # max(p, 1 - p), which is NOT what this gate needs; the gate reads P(true).
    reversible_p = (answers.get("reversible") or {}).get("probability", 0.0)

    gates = {
        "known_tier": choice in TIERS,
        "confidence": tier_conf >= MIN_TIER_CONFIDENCE,
        "probability": selected_p >= MIN_CHOICE_PROBABILITY,
    }
    # Only a downgrade needs the reversibility gate; escalating to the thinker is always safe.
    if choice != "thinker":
        gates["reversible"] = reversible_p >= MIN_REVERSIBLE_PROBABILITY

    telemetry = {
        "source": "eikos",
        "confidence": tier_conf,
        "confidence_source": conf_source,
        "selected_probability": selected_p,
        "reversible_probability": reversible_p,
        "complexity": (answers.get("complexity") or {}).get("score"),
        "gates": gates,
    }
    # Only the validated fields the decision depends on, so a recorded decision can be re-derived.
    kept = {"tier": {k: tier_ans[k] for k in ("choice", "probabilities", "confidence") if k in tier_ans},
            "reversible": {"probability": reversible_p}}
    if isinstance(answers.get("complexity"), dict):
        kept["complexity"] = {"score": answers["complexity"].get("score")}
    base = {"policy": POLICY_VERSION, "model_id": model_identity(), "answers": kept, **telemetry}
    if not all(gates.values()):
        return {**TIERS["thinker"], "tier": "thinker", "escalated": True, **base}
    return {**TIERS[choice], "tier": choice, "escalated": False, **base}


def eikos_url() -> str:
    return (os.environ.get("GEAK_EIKOS_URL") or EIKOS_DEFAULT_URL).rstrip("/")


def is_loopback(url: str) -> bool:
    """True only for http(s) to a literal loopback IP or exactly `localhost`. A hostname that
    merely starts with "127." is a name, not an address, and is refused."""
    import ipaddress  # noqa: PLC0415
    from urllib.parse import urlparse  # noqa: PLC0415
    u = urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        return False
    host = u.hostname.lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class _NoRedirect:
    """urllib handler that refuses every redirect: a redirect could move the state off-host."""

    def __new__(cls):
        import urllib.request  # noqa: PLC0415

        class H(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *a, **k):  # noqa: D401
                raise RuntimeError("Eikos server answered with a redirect; refusing to follow it")
        return H()


def build_state(req: dict) -> str:
    """Eikos takes the state as text. One field per line, the task last and capped."""
    return "\n".join([
        "agent_label: %s" % req.get("label", ""),
        "workflow_phase: %s" % req.get("phase", ""),
        "attempt_number: %d" % int(req.get("attempt", 0) or 0),
        "task:",
        (req.get("task") or "")[:STATE_TOKEN_BUDGET_CHARS],
    ])


def post_systemone(url: str, state, questions: dict, timeout_s: float) -> dict:
    """One request to Eikos's serve.py API, with the transport protections every caller needs:
    loopback only (unless GEAK_EIKOS_ALLOW_REMOTE=1), no proxies, no redirects, bounded read.
    Returns the parsed response body; raises on any transport or format failure.

    Standard library only, so the disabled path and the test suite need nothing installed.
    """
    import urllib.request  # noqa: PLC0415

    if not is_loopback(url) and os.environ.get("GEAK_EIKOS_ALLOW_REMOTE", "0") != "1":
        raise RuntimeError(
            "GEAK_EIKOS_URL %s is not a loopback address. The state carries GEAK task text, and "
            "the point of a local model is that it never leaves the host. Set "
            "GEAK_EIKOS_ALLOW_REMOTE=1 to send it there deliberately." % url)
    payload = json.dumps({"state": state, "questions": questions}).encode("utf-8")
    request = urllib.request.Request(url + EIKOS_PATH, data=payload,
                                     headers={"Content-Type": "application/json"})
    # No proxies (an http_proxy variable would route the state through another host) and no
    # redirects: the destination is exactly the loopback address checked above.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    with opener.open(request, timeout=timeout_s) as resp:  # noqa: S310 - scheme checked above
        raw = resp.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise RuntimeError("Eikos response exceeds %d bytes" % MAX_RESPONSE_BYTES)
    body = json.loads(raw.decode("utf-8"))
    if not isinstance(body, dict) or not isinstance(body.get("answers"), dict):
        raise RuntimeError("Eikos response has no `answers` object")
    return body


def call_eikos(req: dict, url: str, timeout_s: float) -> dict:
    """Single evaluation request. All questions are answered against one shared state."""
    body = post_systemone(url, build_state(req), build_questions(), timeout_s)
    return decide_from_answers(body["answers"], {})


def route(req: dict, cache_dir: str, *, enabled: bool, url: str, timeout_s: float) -> dict:
    task = req.get("task") or ""
    # Neither case needs a model, so neither is recorded: both are pure functions of the input.
    if not enabled:
        return no_decision("router disabled")
    if int(req.get("attempt", 0) or 0) > 0:
        return no_decision("retry: a retry is never routed")
    if not task.strip():
        return no_decision("empty task")
    if len(task) > STATE_TOKEN_BUDGET_CHARS:
        # Never judge a task by a prefix: its tail may hold the instruction that matters.
        return no_decision("task has %d chars, over the %d the state may carry; not judged on a "
                           "prefix" % (len(task), STATE_TOKEN_BUDGET_CHARS))

    cache_path = os.path.join(cache_dir or ".", CACHE_BASENAME)
    key = request_key(req, url)
    hit = load_cache(cache_path).get(key)
    if valid_recorded(hit):
        return {**hit, "cached": True, "recorded": True}

    try:
        decision = call_eikos(req, url, timeout_s)
    except Exception as exc:  # noqa: BLE001 - every failure degrades, none propagates
        decision = no_decision("eikos unavailable or invalid", eikos_error=f"{type(exc).__name__}: {exc}"[:500])

    # Failures are recorded too, so a repeat of this request gets the same outcome. If another
    # caller persisted an outcome for this key first, that one is used, not ours.
    used, persisted = record(cache_path, key, decision)
    return {**used, "cached": used is not decision, "recorded": persisted}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default=".", help="directory holding " + CACHE_BASENAME)
    ap.add_argument("--timeout-s", type=float, default=10.0)
    ap.add_argument("--request", default=None, help="request JSON inline (default: read stdin)")
    args = ap.parse_args(argv)

    raw = args.request if args.request is not None else sys.stdin.read()
    try:
        req = json.loads(raw or "{}")
        if not isinstance(req, dict):
            raise ValueError("request must be a JSON object")
    except Exception as exc:  # noqa: BLE001
        # Even a malformed request must not fail the caller: answer with the safe default.
        print(json.dumps(no_decision("malformed request", request_error=str(exc)[:300])))
        return 0

    decision = route(
        req,
        args.cache_dir,
        enabled=os.environ.get("GEAK_EIKOS_ROUTER", "0") == "1",
        url=eikos_url(),
        timeout_s=args.timeout_s,
    )
    print(json.dumps(decision))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
