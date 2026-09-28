"""Tests for the opt-in Jev model/effort router (`e2e_workflow/scripts/jev_router.py`).

The three properties under test are the three the router promises: it is OFF by default, its
decisions are DETERMINISTIC under replay, and every failure path is NON-FATAL.
"""
import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import jev_router as jr  # noqa: E402

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "jev_router.py")


def run_cli(request, cache_dir, env=None):
    e = dict(os.environ)
    e.pop("GEAK_JEV_ROUTER", None)
    e.pop("GEAK_JEV_API_KEY", None)
    e.update(env or {})
    out = subprocess.run(
        [sys.executable, "-B", SCRIPT, "--cache-dir", str(cache_dir)],
        input=json.dumps(request), capture_output=True, text=True, env=e, timeout=60,
    )
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


# --- Property 1: off by default -------------------------------------------------------------

def test_disabled_by_default_matches_static(tmp_path):
    """No flag => the static B5 decision, unchanged."""
    d = run_cli({"label": "storage:reclaim ws", "attempt": 0}, tmp_path)
    assert d == {"model": None, "effort": "low", "tier": "cheap", "source": "static"}


def test_flag_alone_cannot_call_out(tmp_path):
    """GEAK_JEV_ROUTER=1 without a key must NOT attempt a network call."""
    d = run_cli({"label": "bakeoff gemm", "attempt": 0}, tmp_path,
                env={"GEAK_JEV_ROUTER": "1"})
    assert d["source"] == "static"
    assert "jev_error" not in d
    assert not (tmp_path / jr.CACHE_BASENAME).exists()


def test_non_cheap_label_is_never_routed_cheap(tmp_path):
    d = run_cli({"label": "System Architect", "attempt": 0}, tmp_path)
    assert d["tier"] == "thinker" and d["effort"] is None


def test_retry_never_routes_cheap(tmp_path):
    """attempt > 0 escalates -- preserved from ablEffortFor."""
    d = run_cli({"label": "storage:reclaim ws", "attempt": 1}, tmp_path)
    assert d["tier"] == "thinker" and d["effort"] is None


def test_static_decision_mirrors_the_js_regex():
    """Mirrors ABL_CHEAP_LABELS in e2e_workflow.js:1430 EXACTLY, asymmetry included.

    Note `storage:reclaim` carries no trailing space in the JS alternation while the other
    three do, so `storage:reclaimx` matches and `bakeoff` (bare) does not. That is the shipped
    behaviour; this test pins it so the two copies cannot drift apart silently.
    """
    for label in ("storage:reclaim x", "storage:reclaimx", "bakeoff a", "extract_op k",
                  "roofline r"):
        assert jr.static_decision(label, 0)["tier"] == "cheap", label
    for label in ("bakeoff", "extract_op", "roofline", " storage:reclaim", "Engineer", ""):
        assert jr.static_decision(label, 0)["tier"] == "thinker", label


# --- Property 2: deterministic --------------------------------------------------------------

def test_request_key_is_stable_and_order_independent():
    a = {"label": "x", "phase": "p", "attempt": 0, "task": "t"}
    b = {"task": "t", "attempt": 0, "phase": "p", "label": "x"}
    assert jr.request_key(a) == jr.request_key(b)
    assert jr.request_key({**a, "task": "u"}) != jr.request_key(a)


def test_request_key_ignores_fields_outside_the_contract():
    """Extra keys must not change the hash, or resume would miss its own cache."""
    a = {"label": "x", "phase": "p", "attempt": 0, "task": "t"}
    assert jr.request_key({**a, "nonce": "anything"}) == jr.request_key(a)


def test_cache_hit_replays_without_calling(tmp_path, monkeypatch):
    req = {"label": "bakeoff gemm", "phase": "HeadKernel", "attempt": 0, "task": "bake off"}
    seeded = {"model": "claude-haiku-4-5-20251001", "effort": "low", "tier": "cheap",
              "source": "jev"}
    jr.save_cache(str(tmp_path / jr.CACHE_BASENAME), {jr.request_key(req): seeded})

    def boom(*a, **k):
        raise AssertionError("a cached decision must not trigger a call")
    monkeypatch.setattr(jr, "call_jev", boom)

    d = jr.route(req, str(tmp_path), enabled=True, api_key="k", timeout_s=5)
    assert d["cached"] is True and d["model"] == "claude-haiku-4-5-20251001"


def test_decision_is_persisted_for_replay(tmp_path, monkeypatch):
    req = {"label": "roofline r", "phase": "Profile", "attempt": 0, "task": "roofline"}
    monkeypatch.setattr(jr, "call_jev", lambda *a, **k: {
        **jr.TIERS["cheap"], "tier": "cheap", "source": "jev"})
    first = jr.route(req, str(tmp_path), enabled=True, api_key="k", timeout_s=5)
    assert first["cached"] is False

    def boom(*a, **k):
        raise AssertionError("second identical request must replay from cache")
    monkeypatch.setattr(jr, "call_jev", boom)
    second = jr.route(req, str(tmp_path), enabled=True, api_key="k", timeout_s=5)
    assert second["cached"] is True and second["tier"] == first["tier"]


# --- Property 3: non-fatal ------------------------------------------------------------------

def test_call_failure_degrades_to_static(tmp_path, monkeypatch):
    req = {"label": "bakeoff gemm", "attempt": 0}
    monkeypatch.setattr(jr, "call_jev", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("gateway refused the request (403)")))
    d = jr.route(req, str(tmp_path), enabled=True, api_key="k", timeout_s=5)
    assert d["source"] == "static" and d["effort"] == "low"
    assert "403" in d["jev_error"]


def test_malformed_request_still_answers(tmp_path):
    e = dict(os.environ)
    e.pop("GEAK_JEV_ROUTER", None)
    out = subprocess.run([sys.executable, "-B", SCRIPT, "--cache-dir", str(tmp_path)],
                         input="not json", capture_output=True, text=True, env=e, timeout=60)
    assert out.returncode == 0
    d = json.loads(out.stdout)
    assert d["tier"] == "thinker" and "request_error" in d


def test_unreadable_cache_does_not_raise(tmp_path):
    (tmp_path / jr.CACHE_BASENAME).write_text("{ broken")
    assert jr.load_cache(str(tmp_path / jr.CACHE_BASENAME)) == {}


# --- The escalation rule --------------------------------------------------------------------

def _answers(tier="cheap", p=0.95, rev_p=0.95, score=0.4):
    return {
        "tier": {"type": "choice", "choice": tier, "probabilities": {tier: p}},
        "complexity": {"type": "score", "score": score},
        "reversible": {"type": "boolean", "probability": rev_p},
    }


def test_confident_cheap_answer_is_taken():
    d = jr.decide_from_answers(_answers(), {"tier": 0.9})
    assert d["tier"] == "cheap" and d["model"] == "claude-haiku-4-5-20251001"
    assert d["escalated"] is False


def test_low_confidence_escalates():
    d = jr.decide_from_answers(_answers(), {"tier": 0.4})
    assert d["tier"] == "thinker" and d["escalated"] is True


def test_low_selected_probability_escalates():
    d = jr.decide_from_answers(_answers(p=0.55), {"tier": 0.9})
    assert d["escalated"] is True


def test_irreversible_task_escalates_even_when_confident():
    """The whole point: a confident 'cheap' on unverifiable work is still refused."""
    d = jr.decide_from_answers(_answers(rev_p=0.05), {"tier": 0.95})
    assert d["tier"] == "thinker" and d["escalated"] is True


def test_thinker_choice_needs_no_reversibility():
    """Escalating is always safe, so it must not be gated on reversibility."""
    d = jr.decide_from_answers(_answers(tier="thinker", rev_p=0.05), {"tier": 0.9})
    assert d["tier"] == "thinker" and d["escalated"] is False


def test_unknown_tier_escalates():
    d = jr.decide_from_answers(_answers(tier="turbo"), {"tier": 0.99})
    assert d["tier"] == "thinker" and d["escalated"] is True


def test_missing_probabilities_escalates():
    """`probabilities` is optional in the contract; a missing distribution is an escalation."""
    d = jr.decide_from_answers({"tier": {"type": "choice", "choice": "cheap"}}, {"tier": 0.99})
    assert d["tier"] == "thinker" and d["escalated"] is True


def test_empty_answers_escalate():
    assert jr.decide_from_answers({}, {})["tier"] == "thinker"


def test_rounded_distribution_is_not_renormalized():
    """Vendor rounds to 2dp so a distribution may total 0.99; we must read it as-is."""
    d = jr.decide_from_answers(_answers(p=0.69), {"tier": 0.9})
    assert d["selected_probability"] == 0.69 and d["escalated"] is True


# --- The question payload -------------------------------------------------------------------

def test_only_documented_question_types_are_emitted():
    """/v1/evaluate validates the discriminator; a bogus type is a 400."""
    types = {q["type"] for q in jr.build_questions().values()}
    assert types <= {"boolean", "choice", "score"}


def test_endpoint_is_the_canonical_one_not_the_compat_shim():
    """/typesafe/v1/systemone rejects `boolean` with a corrupted 'noul' message (2026-09-28)."""
    assert jr.JEV_ENDPOINT.endswith("/v1/evaluate")
    assert "systemone" not in jr.JEV_ENDPOINT
    assert jr.JEV_MODEL == "typesafe-ai/jev"


def test_boolean_question_has_no_criteria_requirement_violation():
    """`choice` requires criteria (400 without it); boolean's criteria are optional."""
    for name, q in jr.build_questions().items():
        if q["type"] == "choice":
            assert q.get("criteria"), name


def test_score_levels_within_vendor_limits():
    score = [q for q in jr.build_questions().values() if q["type"] == "score"]
    for q in score:
        assert 2 <= len(q["criteria"]) <= 10


def test_choice_options_within_vendor_limits():
    for q in jr.build_questions().values():
        if q["type"] == "choice":
            assert 1 <= len(q["criteria"]) <= 255


def test_tier_question_covers_exactly_the_declared_tiers():
    assert set(jr.build_questions()["tier"]["criteria"]) == set(jr.TIERS)


def test_state_is_truncated_to_stay_under_the_token_cap():
    req = {"label": "bakeoff", "attempt": 0, "task": "x" * 50_000}
    assert len(json.loads(json.dumps(req))["task"]) == 50_000
    key_a = jr.request_key(req)
    key_b = jr.request_key({**req, "task": "x" * jr.STATE_TOKEN_BUDGET_CHARS})
    assert key_a == key_b, "truncation must happen before hashing, or replay misses"
