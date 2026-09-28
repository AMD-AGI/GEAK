#!/usr/bin/env python3
"""Model/effort router for GEAK agent spawns, optionally backed by Jev (TypeSafe "System One").

WHY THIS EXISTS
---------------
`e2e_workflow.js:ablEffortFor` already routes effort with a hardcoded label regex. This module
generalises that single decision into `{effort, model}` and lets an *optional* Jev evaluation
make it instead of the regex.

THREE PROPERTIES THIS FILE GUARANTEES (see research/09_jev_typesafe_routing.md sections 10-11)
  1. OFF BY DEFAULT. Without GEAK_JEV_ROUTER=1 the decision is the static one, byte-identical to
     today. Without GEAK_JEV_API_KEY it is ALSO the static one, so setting the flag alone can
     never produce a network call.
  2. DETERMINISTIC. Workflow scripts are replayed from a journal; an unjournaled non-deterministic
     input would diverge on resume. Every decision is memoized under sha256(request) in
     `<cache-dir>/jev_router_cache.json`, so a replay reads the recorded answer.
  3. NON-FATAL. Every failure path returns the static decision. This router cannot fail a run.

PROTOCOL
    echo '<request-json>' | python3 jev_router.py --cache-dir DIR
  request: {"label": str, "phase": str, "attempt": int, "task": str}
  stdout:  {"model": str|null, "effort": str|null, "tier": str, "source": str, ...}
  A null model/effort means "leave the caller's options untouched".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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

# The static B5 router, mirrored from e2e_workflow.js:1428-1445. This is the fallback everywhere.
CHEAP_LABELS = re.compile(r"^(storage:reclaim|bakeoff |extract_op |roofline )")

# Thresholds. Vendor guidance is explicit that real cutoffs must be derived from labeled examples;
# these are the article's illustrative starting points and are NOT calibrated against GEAK.
MIN_TIER_CONFIDENCE = 0.6
MIN_CHOICE_PROBABILITY = 0.7
MIN_REVERSIBLE_PROBABILITY = 0.8

CACHE_BASENAME = "jev_router_cache.json"
# The native HTTP evaluation API, which the vendor changelog specifies for NEW integrations:
# types boolean|choice|score, answers read at `.probability`, model id `typesafe-ai/jev`.
#
# Deliberately NOT https://ai-gateway.vercel.sh/typesafe -- that is the TypeSafe-COMPATIBLE
# client surface for migrating existing TypeSafe code, and it keeps the legacy vocabulary: a
# boolean question is typed `noul` there and its answer is read at `.noul` rather than
# `.probability`. Both surfaces are correct; they are simply different vocabularies, and mixing
# them is what produces `expected one of 'noul', 'choice', 'score'`.
JEV_ENDPOINT = "https://ai-gateway.vercel.sh/v1/evaluate"
JEV_MODEL = "typesafe-ai/jev"
STATE_TOKEN_BUDGET_CHARS = 8000  # Jev caps state at 32k tokens; stay far below.


def static_decision(label: str, attempt: int) -> dict:
    """Today's behaviour, expressed as a decision. Never routes a retry cheap."""
    if attempt > 0 or not CHEAP_LABELS.match(label or ""):
        return {"model": None, "effort": None, "tier": "thinker", "source": "static"}
    return {"model": None, "effort": "low", "tier": "cheap", "source": "static"}


def request_key(req: dict) -> str:
    """Stable hash of the routing inputs. Determinism depends on this being order-independent."""
    canonical = json.dumps(
        {
            "label": req.get("label", ""),
            "phase": req.get("phase", ""),
            "attempt": int(req.get("attempt", 0) or 0),
            "task": (req.get("task") or "")[:STATE_TOKEN_BUDGET_CHARS],
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


def save_cache(path: str, cache: dict) -> None:
    """Best-effort. A cache that cannot be written costs determinism on resume, not correctness."""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(cache, fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except Exception:
        pass


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

    CORRECTED 2026-09-28 against /docs/ai-gateway/modalities/evaluation. The documented
    `POST /v1/evaluate` response body is `{model, answers, usage, providerMetadata}`, and its
    `providerMetadata.gateway` carries routing and cost ONLY. There is no
    `providerMetadata.typesafe.confidence` on this surface -- that path was assumed from the AI
    SDK docs and was never observed. What the HTTP contract DOES document on a choice/score
    answer is `probabilities`, inline.

    Precedence: an answer-level `confidence` if the server sends one, then the legacy metadata
    map (harmless if absent), then the concentration of the distribution itself. The derived
    value is the selected option's mass, so the confidence gate is then satisfied by a strictly
    stronger test rather than silently passing -- and an empty `probabilities` yields 0.0, which
    escalates. That matches the documented meaning: `confidence: 0` with `probabilities: {}`
    means the value is UNAVAILABLE, not that the model measured zero.
    """
    if isinstance(tier_ans.get("confidence"), (int, float)):
        return float(tier_ans["confidence"]), "answer"
    meta = (meta_confidence or {}).get("tier")
    if isinstance(meta, (int, float)):
        return float(meta), "provider_metadata"
    return (max(probs.values()) if probs else 0.0), "derived_from_probabilities"


def decide_from_answers(answers: dict, confidence: dict) -> dict:
    """Apply the escalation rule. Anything short of every gate falls back to the thinker.

    Read the distribution defensively: `probabilities` is optional in the contract, values are
    rounded to two decimals so a distribution may total 0.99, and it must not be renormalized.
    """
    tier_ans = answers.get("tier") or {}
    choice = tier_ans.get("choice")
    probs = tier_ans.get("probabilities") or {}
    selected_p = probs.get(choice, 0.0) if choice else 0.0
    tier_conf, conf_source = tier_confidence(tier_ans, confidence, probs)
    # A boolean answer carries `probability` = P(true); it is NOT a confidence value, and
    # confidence is not reported for booleans at all.
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
        "source": "jev",
        "confidence": tier_conf,
        "confidence_source": conf_source,
        "selected_probability": selected_p,
        "reversible_probability": reversible_p,
        "complexity": (answers.get("complexity") or {}).get("score"),
        "gates": gates,
    }
    if not all(gates.values()):
        return {**TIERS["thinker"], "tier": "thinker", "escalated": True, **telemetry}
    return {**TIERS[choice], "tier": choice, "escalated": False, **telemetry}


def call_jev(req: dict, api_key: str, timeout_s: float) -> dict:
    """Single evaluation request. All questions are answered in parallel against one state.

    Imported lazily so that the disabled path never needs `requests` installed.
    """
    import requests  # noqa: PLC0415

    state = {
        "agent_label": req.get("label", ""),
        "workflow_phase": req.get("phase", ""),
        "attempt_number": int(req.get("attempt", 0) or 0),
        "task": (req.get("task") or "")[:STATE_TOKEN_BUDGET_CHARS],
    }
    # ZDR and no-training are per-request Gateway options; the catalog listing `zdr: none`
    # describes the catalog entry, not what is available per call. ZDR is DEFAULTED ON because
    # the state carries GEAK task text, and shipping that to a third party without retention
    # controls is the riskier default -- but probed 2026-09-28, ZDR is refused outright on
    # non-Pro plans ("Current plan: hobby", ZdrUnauthorizedError), so it must be disableable
    # or such an account can never make a call at all. Turning it off is a deliberate act.
    payload = {
        "model": JEV_MODEL,
        "state": state,
        "questions": build_questions(),
    }
    # `zeroDataRetention` is the documented gateway privacy control. `noTraining` was sent here
    # until 2026-09-28 and has been REMOVED: it appears in no Vercel doc, and it cannot be
    # verified by probing, because an unknown key under `providerOptions.gateway` is accepted
    # exactly like a real one (a bogus control key returned the same 403, not a 400). A flag
    # that may silently do nothing is worse than no flag, because it reads as protection.
    gw = {}
    if os.environ.get("GEAK_JEV_ZDR", "1") == "1":
        gw["zeroDataRetention"] = True
        # Pinning the provider is REQUIRED for ZDR, not a preference. Probed 2026-09-28 on a
        # funded account: unpinned, the gateway resolves `typesafe-ai/jev` to `digitalocean`,
        # then skips it with reason `zdr_ineligible_model` and returns 403 WITHOUT falling back
        # to the `typesafe-ai` provider it listed as available. Pinning `typesafe-ai` reaches
        # provider routing with ZDR intact. So ZDR-on without this line can never succeed.
        gw["only"] = ["typesafe-ai"]
    payload["providerOptions"] = {"gateway": gw}
    resp = requests.post(
        JEV_ENDPOINT,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout_s,
    )
    if resp.status_code == 403:
        # Three distinct account states produce a 403 here, all observed 2026-09-28. None is a
        # bad request, so name them rather than letting a generic error hide the fix.
        body = resp.text[:300]
        if "customer_verification" in body:
            hint = "the Vercel team has no credit card on file"
        elif "zdr_ineligible_model" in body:
            hint = ("no ZDR-eligible provider was reachable for this model; the request was "
                    "pinned to `typesafe-ai`, so this means that provider became ineligible "
                    "too. GEAK_JEV_ZDR=0 would send WITHOUT retention cover -- a deliberate act")
        elif "ZdrUnauthorized" in body or "Zero Data Retention" in body:
            hint = ("this API key's plan does not permit Zero Data Retention (observed on a "
                    "`hobby` plan); set GEAK_JEV_ZDR=0 to send without it, which means the "
                    "state is NOT covered by ZDR")
        elif "Free tier" in body:
            hint = ("the account has no paid credits, so this model is restricted. NOTE: a ZDR "
                    "routing failure can also surface under this message -- check the routing "
                    "metadata for `zdr_ineligible_model` before assuming it is a billing issue")
        else:
            hint = "unrecognised 403"
        raise RuntimeError(f"gateway refused the request (403): {hint}: {body}")
    resp.raise_for_status()
    body = resp.json()
    answers = body.get("answers") or body
    # Kept only as a fallback source; see tier_confidence(). The documented HTTP response does
    # not carry this path, so it is normally absent and the distribution is used instead.
    confidence = (
        ((body.get("providerMetadata") or {}).get("typesafe") or {}).get("confidence") or {}
    )
    return decide_from_answers(answers, confidence)


def route(req: dict, cache_dir: str, *, enabled: bool, api_key: str, timeout_s: float) -> dict:
    label = str(req.get("label", "") or "")
    attempt = int(req.get("attempt", 0) or 0)
    fallback = static_decision(label, attempt)

    # A retry is never routed cheap, so there is nothing to ask and no reason to pay for a call.
    if attempt > 0:
        return fallback
    if not enabled or not api_key:
        return fallback

    cache_path = os.path.join(cache_dir or ".", CACHE_BASENAME)
    cache = load_cache(cache_path)
    key = request_key(req)
    hit = cache.get(key)
    if isinstance(hit, dict):
        return {**hit, "cached": True}

    try:
        decision = call_jev(req, api_key, timeout_s)
    except Exception as exc:  # noqa: BLE001 - every failure degrades, none propagates
        return {**fallback, "source": "static", "jev_error": f"{type(exc).__name__}: {exc}"}

    cache[key] = decision
    save_cache(cache_path, cache)
    return {**decision, "cached": False}


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
        print(json.dumps({**static_decision("", 0), "request_error": str(exc)}))
        return 0

    decision = route(
        req,
        args.cache_dir,
        enabled=os.environ.get("GEAK_JEV_ROUTER", "0") == "1",
        api_key=os.environ.get("GEAK_JEV_API_KEY", ""),
        timeout_s=args.timeout_s,
    )
    print(json.dumps(decision))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
