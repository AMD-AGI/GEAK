"""Tests for the opt-in Eikos model/effort router (`e2e_workflow/scripts/eikos_router.py`).

Offline: a fake Eikos server stands in for the real one. These establish the helper's contract,
not routing quality and not workflow replay behaviour.
"""
import http.server
import json
import math
import multiprocessing
import os
import subprocess
import sys
import threading

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import eikos_router as er  # noqa: E402

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "eikos_router.py")
TASK = "Run EXACTLY this command and nothing else: bash reclaim.sh --keep-best"


def req(**kw):
    base = {"label": "storage:reclaim r1", "phase": "Optimize", "attempt": 0, "task": TASK}
    base.update(kw)
    return base


def answers(tier="cheap", p=0.95, rev_p=0.95, score=0, conf="match"):
    others = [t for t in er.TIERS if t != tier]
    probs = {tier: p, others[0]: (1 - p) / 2, others[1]: (1 - p) / 2}
    t = {"type": "choice", "choice": tier, "probabilities": probs}
    if conf == "match":
        t["confidence"] = p
    elif conf is not None:
        t["confidence"] = conf
    return {"tier": t,
            "reversible": {"type": "boolean", "probability": rev_p, "noul": rev_p,
                           "value": rev_p >= 0.5, "confidence": max(rev_p, 1 - rev_p)},
            "complexity": {"type": "score", "score": score, "probabilities": {"0": 1.0}}}


def route(tmp_path, r=None, *, enabled=True, url="http://127.0.0.1:1", timeout_s=1.0):
    return er.route(r or req(), str(tmp_path), enabled=enabled, url=url, timeout_s=timeout_s)


def is_host(d):
    return d["source"] == "host" and d["model"] is None and d["effort"] is None


# --------------------------------------------------------------------------------------------
# Off by default, and the Jev configuration cannot switch it on
# --------------------------------------------------------------------------------------------
def run_cli(request, cache_dir, env=None):
    e = {k: v for k, v in os.environ.items() if not k.startswith(("GEAK_EIKOS", "GEAK_JEV"))}
    e.update(env or {})
    out = subprocess.run([sys.executable, "-B", SCRIPT, "--cache-dir", str(cache_dir)],
                         input=json.dumps(request), capture_output=True, text=True, env=e, timeout=30)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_disabled_by_default_leaves_the_host_in_charge(tmp_path):
    d = run_cli(req(), tmp_path)
    assert is_host(d) and d["reason"] == "router disabled"
    assert not (tmp_path / er.CACHE_BASENAME).exists()


def test_old_jev_switches_do_not_enable_eikos(tmp_path):
    d = run_cli(req(), tmp_path, env={"GEAK_JEV_ROUTER": "1", "GEAK_JEV_API_KEY": "k", "GEAK_JEV_ZDR": "0"})
    assert is_host(d) and d["reason"] == "router disabled"


def test_an_old_jev_cache_is_never_read(tmp_path, monkeypatch):
    key = er.request_key(req(), "http://127.0.0.1:1")
    (tmp_path / "jev_router_cache.json").write_text(json.dumps(
        {key: {**er.TIERS["cheap"], "tier": "cheap", "source": "jev"}}))
    monkeypatch.setattr(er, "call_eikos", lambda *a, **k: (_ for _ in ()).throw(OSError("down")))
    d = route(tmp_path)
    assert is_host(d) and "down" in d["eikos_error"]


def test_no_executable_jev_configuration_remains():
    src = open(SCRIPT).read()
    for token in ("GEAK_JEV", "ai-gateway", "typesafe-ai/jev", "api_key", "Authorization", "ZDR"):
        assert token not in src, token


# --------------------------------------------------------------------------------------------
# Cases that need no model at all
# --------------------------------------------------------------------------------------------
def test_retry_is_never_routed(tmp_path):
    assert is_host(route(tmp_path, req(attempt=1)))


def test_empty_task_is_not_judged(tmp_path):
    assert is_host(route(tmp_path, req(task="  ")))


def test_oversized_task_is_deferred_not_judged_on_a_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr(er, "call_eikos", lambda *a, **k: pytest.fail("must not be asked"))
    head = "x" * er.STATE_TOKEN_BUDGET_CHARS
    d = route(tmp_path, req(task=head + " -- and then rewrite the kernel from scratch"))
    assert is_host(d) and "not judged on a prefix" in d["reason"]


# --------------------------------------------------------------------------------------------
# Request identity
# --------------------------------------------------------------------------------------------
def test_request_key_is_stable_and_order_independent():
    a = {"label": "l", "phase": "p", "attempt": 0, "task": "t"}
    b = {"task": "t", "attempt": 0, "phase": "p", "label": "l"}
    assert er.request_key(a, "u") == er.request_key(b, "u")


def test_same_prefix_different_tail_never_share_a_decision():
    """Astra's reproduction: two tasks sharing an 8,000-character prefix used to collide."""
    head = "y" * 8000
    assert er.request_key(req(task=head + "A"), "u") != er.request_key(req(task=head + "B"), "u")


def test_request_key_ignores_fields_outside_the_contract():
    assert er.request_key({**req(), "extra": 1}, "u") == er.request_key(req(), "u")


def test_policy_and_endpoint_are_part_of_the_identity(monkeypatch):
    k = er.request_key(req(), "u")
    assert er.request_key(req(), "v") != k
    monkeypatch.setattr(er, "MIN_TIER_CONFIDENCE", 0.61)
    assert er.request_key(req(), "u") != k


# --------------------------------------------------------------------------------------------
# Recording outcomes
# --------------------------------------------------------------------------------------------
def test_a_decision_is_recorded_and_replayed_without_asking(tmp_path, monkeypatch):
    monkeypatch.setattr(er, "call_eikos", lambda *a, **k: er.decide_from_answers(answers(), {}))
    first = route(tmp_path)
    assert first["source"] == "eikos" and first["recorded"] is True
    monkeypatch.setattr(er, "call_eikos", lambda *a, **k: pytest.fail("must replay"))
    again = route(tmp_path)
    assert again["cached"] is True and again["model"] == first["model"]


def test_a_failure_is_recorded_so_a_repeat_cannot_turn_into_a_route(tmp_path, monkeypatch):
    """Astra's reproduction: a timeout first, then a Haiku route on the same request."""
    monkeypatch.setattr(er, "call_eikos", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("slow")))
    assert is_host(route(tmp_path))
    monkeypatch.setattr(er, "call_eikos", lambda *a, **k: er.decide_from_answers(answers(), {}))
    again = route(tmp_path)
    assert is_host(again) and again["cached"] is True


def test_a_malformed_recorded_value_is_not_trusted(tmp_path, monkeypatch):
    key = er.request_key(req(), "http://127.0.0.1:1")
    (tmp_path / er.CACHE_BASENAME).write_text(json.dumps(
        {key: {"model": "claude-opus-9", "effort": "low", "tier": "cheap", "source": "eikos",
               "policy": er.POLICY_VERSION}}))
    monkeypatch.setattr(er, "call_eikos", lambda *a, **k: er.decide_from_answers(answers(), {}))
    d = route(tmp_path)
    assert d["cached"] is False and d["model"] == er.TIERS["cheap"]["model"]


def test_an_unreadable_cache_does_not_raise(tmp_path, monkeypatch):
    (tmp_path / er.CACHE_BASENAME).write_text("{not json")
    monkeypatch.setattr(er, "call_eikos", lambda *a, **k: er.decide_from_answers(answers(), {}))
    assert route(tmp_path)["source"] == "eikos"


def _write_many(args):
    path, n = args
    for i in range(20):
        er.record(path, "k%d-%d" % (n, i), er.no_decision("t"))


def test_concurrent_writers_lose_no_entries(tmp_path):
    path = str(tmp_path / er.CACHE_BASENAME)
    with multiprocessing.get_context("fork").Pool(6) as pool:
        pool.map(_write_many, [(path, n) for n in range(6)])
    assert len(er.load_cache(path)) == 6 * 20


# --------------------------------------------------------------------------------------------
# Untrusted answers
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mutate, why", [
    (lambda a: a["tier"]["probabilities"].update(cheap=2.0), "numbers in [0, 1]"),
    (lambda a: a["tier"].update(confidence=2.0), "confidence"),
    (lambda a: a["tier"]["probabilities"].update(cheap=True), "numbers in [0, 1]"),
    (lambda a: a["tier"]["probabilities"].update(cheap=float("nan")), "numbers in [0, 1]"),
    (lambda a: a["tier"]["probabilities"].pop("thinker"), "exactly the declared tiers"),
    (lambda a: a["tier"]["probabilities"].update(extra=0.0), "exactly the declared tiers"),
    (lambda a: a["tier"].update(choice="haiku"), "not a known tier"),
    (lambda a: a["tier"]["probabilities"].update(cheap=0.5), "sum to"),
    (lambda a: a["tier"].update(choice="standard"), "most probable"),
    (lambda a: a["tier"].update(confidence=0.5), "confidence disagrees"),
    (lambda a: a.pop("reversible"), "reversible"),
    (lambda a: a["reversible"].update(probability="0.9"), "reversible"),
    (lambda a: a["complexity"].update(score=0.4), "integer level"),
    (lambda a: a["complexity"].update(score=True), "integer level"),
    (lambda a: a.pop("tier"), "tier answer missing"),
])
def test_invalid_answers_are_rejected_before_any_gate(mutate, why):
    a = answers()
    mutate(a)
    assert why in er.validate_answers(a)
    with pytest.raises(ValueError):
        er.decide_from_answers(a, {})


def test_out_of_range_numbers_can_no_longer_pass_a_cheap_gate(tmp_path, monkeypatch):
    """Astra's reproduction: probability/confidence 2.0 used to pass the cheap gate."""
    bad = answers(p=0.95)
    bad["tier"]["probabilities"]["cheap"] = 2.0
    bad["tier"]["confidence"] = 2.0
    monkeypatch.setattr(er, "call_eikos", lambda *a, **k: er.decide_from_answers(bad, {}))
    assert is_host(route(tmp_path))


def test_a_valid_eikos_shaped_answer_passes():
    assert er.validate_answers(answers()) == ""
    assert er.validate_answers(answers(conf=None)) == ""   # confidence is optional


# --------------------------------------------------------------------------------------------
# The gates (unchanged policy; thresholds NOT calibrated against GEAK)
# --------------------------------------------------------------------------------------------
def test_confident_cheap_answer_is_taken():
    d = er.decide_from_answers(answers(), {})
    assert d["tier"] == "cheap" and not d["escalated"] and d["model"] == er.TIERS["cheap"]["model"]


def test_low_confidence_escalates_to_the_thinker():
    d = er.decide_from_answers(answers(p=0.55), {})
    assert d["tier"] == "thinker" and d["escalated"]


def test_irreversible_task_escalates_even_when_confident():
    d = er.decide_from_answers(answers(rev_p=0.4), {})
    assert d["tier"] == "thinker" and d["escalated"]


def test_thinker_choice_needs_no_reversibility():
    d = er.decide_from_answers(answers(tier="thinker", rev_p=0.1), {})
    assert d["tier"] == "thinker" and not d["escalated"]


# --------------------------------------------------------------------------------------------
# Transport, against a fake local server
# --------------------------------------------------------------------------------------------
class _Fake(http.server.BaseHTTPRequestHandler):
    body = b"{}"
    seen = []

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("content-length", 0))
        type(self).seen.append((self.path, json.loads(self.rfile.read(n))))
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(type(self).body)

    def log_message(self, *a):
        pass


@pytest.fixture
def fake_server():
    _Fake.seen = []
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv, "http://127.0.0.1:%d" % srv.server_address[1]
    srv.shutdown()


def test_end_to_end_against_a_fake_eikos(tmp_path, fake_server):
    _, url = fake_server
    _Fake.body = json.dumps({"model": ".", "answers": answers(), "usage": {}}).encode()
    d = route(tmp_path, url=url)
    assert d["source"] == "eikos" and d["tier"] == "cheap"
    path, sent = _Fake.seen[0]
    assert path == "/v1/systemone"
    assert set(sent) == {"state", "questions"}                     # no model/key/gateway fields
    assert isinstance(sent["state"], str) and TASK in sent["state"]


def test_an_oversized_response_is_refused(tmp_path, fake_server):
    _, url = fake_server
    _Fake.body = b" " * (er.MAX_RESPONSE_BYTES + 10) + b"{}"
    d = route(tmp_path, url=url)
    assert is_host(d) and "exceeds" in d["eikos_error"]


def test_a_response_without_answers_leaves_the_host_in_charge(tmp_path, fake_server):
    _, url = fake_server
    _Fake.body = b'{"error": "not found"}'
    assert is_host(route(tmp_path, url=url))


def test_a_non_loopback_url_is_refused_unless_allowed(tmp_path, monkeypatch):
    monkeypatch.delenv("GEAK_EIKOS_ALLOW_REMOTE", raising=False)
    d = route(tmp_path, url="http://10.0.0.5:8000")
    assert is_host(d) and "not a loopback address" in d["eikos_error"]


def test_state_carries_the_whole_task_when_under_the_cap():
    task = "z" * (er.STATE_TOKEN_BUDGET_CHARS - 5) + "TAIL!"
    assert er.build_state(req(task=task)).endswith("TAIL!")


def test_malformed_request_still_answers(tmp_path):
    e = {k: v for k, v in os.environ.items() if not k.startswith("GEAK_EIKOS")}
    out = subprocess.run([sys.executable, "-B", SCRIPT, "--cache-dir", str(tmp_path)],
                         input="not json", capture_output=True, text=True, env=e, timeout=30)
    d = json.loads(out.stdout)
    assert is_host(d) and d["reason"] == "malformed request"


# --------------------------------------------------------------------------------------------
# Astra's second review (2026-09-29, draft-130749): executed findings, now tests
# --------------------------------------------------------------------------------------------
def _race(args):
    cache_dir, tier, delay = args
    import time as _t
    def fake(*a, **k):
        _t.sleep(delay)
        return er.decide_from_answers(answers(tier=tier, rev_p=0.95), {})
    er.call_eikos = fake
    d = er.route(req(), cache_dir, enabled=True, url="http://127.0.0.1:1", timeout_s=1.0)
    return d["tier"], d["recorded"]


def test_concurrent_callers_for_one_key_all_act_on_the_persisted_winner(tmp_path):
    """Two callers, same key, providers disagree (cheap vs thinker). Both must return the one
    outcome that was persisted, and a later replay must agree with it."""
    with multiprocessing.get_context("fork").Pool(2) as pool:
        got = pool.map(_race, [(str(tmp_path), "cheap", 0.0), (str(tmp_path), "thinker", 0.3)])
    assert got[0][0] == got[1][0], got
    replay = er.route(req(), str(tmp_path), enabled=True, url="http://127.0.0.1:1", timeout_s=1.0)
    assert replay["cached"] and replay["tier"] == got[0][0]


def test_a_bare_cheap_entry_cannot_authorize_a_downgrade(tmp_path, monkeypatch):
    key = er.request_key(req(), "http://127.0.0.1:1")
    (tmp_path / er.CACHE_BASENAME).write_text(json.dumps({key: {
        "policy": er.POLICY_VERSION, "source": "eikos", "tier": "cheap",
        "model": er.TIERS["cheap"]["model"], "effort": "low"}}))
    monkeypatch.setattr(er, "call_eikos", lambda *a, **k: er.decide_from_answers(answers(rev_p=0.1), {}))
    d = route(tmp_path)
    assert not d["cached"] and d["tier"] == "thinker"


def test_a_tampered_entry_whose_answers_disagree_is_not_trusted(tmp_path, monkeypatch):
    real = er.decide_from_answers(answers(rev_p=0.1), {})          # escalates to thinker
    forged = {**real, **er.TIERS["cheap"], "tier": "cheap", "escalated": False}
    key = er.request_key(req(), "http://127.0.0.1:1")
    (tmp_path / er.CACHE_BASENAME).write_text(json.dumps({key: forged}))
    monkeypatch.setattr(er, "call_eikos", lambda *a, **k: real)
    d = route(tmp_path)
    assert not d["cached"] and d["tier"] == "thinker"


def test_a_genuine_recorded_decision_is_re_derivable():
    d = er.decide_from_answers(answers(), {})
    assert er.valid_recorded(d)


@pytest.mark.parametrize("url, ok", [
    ("http://127.0.0.1:8000", True), ("http://127.9.9.9:8000", True), ("http://[::1]:8000", True),
    ("http://localhost:8000", True), ("https://127.0.0.1", True),
    ("http://127.example.invalid:8000", False), ("http://127.0.0.1.nip.io:8000", False),
    ("http://10.0.0.5:8000", False), ("ftp://127.0.0.1/x", False), ("file:///etc/passwd", False),
    ("http://:8000", False),
])
def test_loopback_means_a_literal_loopback_address(url, ok):
    assert er.is_loopback(url) is ok


class _Redirect(http.server.BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        self.send_response(307)
        self.send_header("Location", "http://10.0.0.5:9/v1/systemone")
        self.end_headers()

    def log_message(self, *a):
        pass


def test_a_redirect_is_refused_not_followed(tmp_path):
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Redirect)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        d = route(tmp_path, url="http://127.0.0.1:%d" % srv.server_address[1])
    finally:
        srv.shutdown()
    assert is_host(d) and "redirect" in d["eikos_error"]


def test_proxy_settings_are_ignored(tmp_path, fake_server):
    """With a dead proxy configured, the call still reaches the loopback server directly. Run in
    a fresh process: urllib caches its default opener (and the proxies it read) per process, so
    an in-process test would pass whatever the code does."""
    _, url = fake_server
    _Fake.body = json.dumps({"answers": answers()}).encode()
    dead = "http://127.0.0.1:9"
    d = run_cli(req(), tmp_path, env={"GEAK_EIKOS_ROUTER": "1", "GEAK_EIKOS_URL": url,
                                     "http_proxy": dead, "HTTP_PROXY": dead,
                                     "https_proxy": dead, "HTTPS_PROXY": dead,
                                     "no_proxy": "", "NO_PROXY": ""})
    assert d["source"] == "eikos", d

def test_declared_model_identity_is_part_of_the_key_and_the_provenance(monkeypatch):
    monkeypatch.delenv("GEAK_EIKOS_MODEL_ID", raising=False)
    a = er.request_key(req(), "u")
    assert er.decide_from_answers(answers(), {})["model_id"] == "undeclared"
    monkeypatch.setenv("GEAK_EIKOS_MODEL_ID", "hf:103a5647")
    assert er.request_key(req(), "u") != a
    assert er.decide_from_answers(answers(), {})["model_id"] == "hf:103a5647"
