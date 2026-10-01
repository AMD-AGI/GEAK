"""Tests for eikos_decide.py — the shadow pilot's decision boundary. Offline: a fake Eikos server.

These establish the script's contract (state validation, statuses, receipts, first-persisted
decisions, relay echo), not decision quality.
"""
import http.server
import json
import multiprocessing
import os
import subprocess
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import eikos_decide as ed  # noqa: E402

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "eikos_decide.py")
SPEC, _ = ed.load_spec("round_continue")


def state(**kw):
    s = {k: "unknown" for k in SPEC["required_state"]}
    s.update(round=2, directions_used=2, directions_budget=6, tracked_incumbent_speedup=1.21,
             best_seen_speedup=1.234, rounds_without_improvement=1, rounds_without_improvement_limit=2,
             specialties_dispatched=["algorithm", "host_runtime"])
    s.update(kw)
    return json.dumps(s)


def answers(choice="continue", p=0.92, stalled=0.3):
    other = "stop" if choice == "continue" else "continue"
    return {"next_step": {"type": "choice", "choice": choice, "probabilities": {choice: p, other: 1 - p},
                          "confidence": p},
            "stalled": {"type": "noul", "probability": stalled, "noul": stalled, "value": stalled >= 0.5,
                        "confidence": max(stalled, 1 - stalled)}}


class _Fake(http.server.BaseHTTPRequestHandler):
    bodies = []          # served in order; the last one repeats
    seen = []
    delay = 0.0

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("content-length", 0))
        type(self).seen.append(json.loads(self.rfile.read(n)))
        if type(self).delay:
            time.sleep(type(self).delay)
        body = type(self).bodies[min(len(type(self).seen), len(type(self).bodies)) - 1]
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(body if isinstance(body, bytes) else json.dumps(body).encode())

    def log_message(self, *a):
        pass


@pytest.fixture
def fake():
    _Fake.bodies, _Fake.seen, _Fake.delay = [{"answers": answers()}], [], 0.0
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield "http://127.0.0.1:%d" % srv.server_address[1]
    srv.shutdown()


def decide(tmp_path, url, st=None, rnd=2, scope="/eval/lane", timeout_s=5.0):
    return ed.decide("round_continue", scope, rnd, st or state(), str(tmp_path), url, timeout_s)


def attempts(tmp_path):
    p = tmp_path / "round_continue.attempts.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


# ---------------------------------------------------------------------------------- happy path
def test_ok_decision_is_validated_persisted_and_echoed(tmp_path, fake):
    st = state()
    env = decide(tmp_path, fake, st)
    assert env["status"] == "ok" and env["choice"] == "continue" and env["persisted"] and not env["reused"]
    assert env["state_raw"] == st                                         # byte-for-byte echo
    assert env["would_be_action"] == "continue"                           # 0.92 >= 0.8
    assert env["questions_version"] == "round_continue.v1" and env["policy_version"]
    sent = _Fake.seen[0]
    assert set(sent) == {"state", "questions"} and sent["questions"] == SPEC["questions"]
    assert json.loads(sent["state"]) == json.loads(st)                    # canonical, same content
    assert len(attempts(tmp_path)) == 1 and attempts(tmp_path)[0]["canonical_state"]


def test_below_threshold_is_recorded_as_abstain(tmp_path, fake):
    _Fake.bodies = [{"answers": answers(p=0.6)}]
    env = decide(tmp_path, fake)
    assert env["status"] == "ok" and env["choice"] == "continue" and env["would_be_action"] == "abstain"


def test_same_key_returns_the_first_decision_without_asking_again(tmp_path, fake):
    first = decide(tmp_path, fake)
    _Fake.bodies = [{"answers": answers(choice="stop", p=0.99)}]          # would differ if asked
    again = decide(tmp_path, fake)
    assert len(_Fake.seen) == 1
    assert again["reused"] and again["choice"] == first["choice"] and again["first_attempt_id"] == first["attempt_id"]
    assert len(attempts(tmp_path)) == 2                                   # both executions receipted


def test_a_different_round_or_scope_is_a_different_key(tmp_path, fake):
    a = decide(tmp_path, fake, rnd=2)
    b = decide(tmp_path, fake, rnd=3)
    c = decide(tmp_path, fake, scope="/eval/other")
    assert len({a["logical_key"], b["logical_key"], c["logical_key"]}) == 3 and len(_Fake.seen) == 3


def test_declared_model_identity_changes_the_key(tmp_path, fake, monkeypatch):
    a = decide(tmp_path, fake)["logical_key"]
    monkeypatch.setenv("GEAK_EIKOS_MODEL_ID", "hf:103a5647")
    assert decide(tmp_path, fake)["logical_key"] != a


def _race(args):
    receipt_dir, url = args
    return ed.decide("round_continue", "/eval/lane", 2, state(), receipt_dir, url, 10.0)["choice"]


def test_concurrent_callers_for_one_key_return_the_same_decision(tmp_path, fake):
    _Fake.bodies = [{"answers": answers("continue")}, {"answers": answers("stop", p=0.95)}]
    _Fake.delay = 0.2
    with multiprocessing.get_context("fork").Pool(2) as pool:
        got = pool.map(_race, [(str(tmp_path), fake), (str(tmp_path), fake)])
    assert got[0] == got[1], got
    index = json.loads((tmp_path / "round_continue.index.json").read_text())
    assert [v["choice"] for v in index.values()] == [got[0]]


# ---------------------------------------------------------------------------------- failures
@pytest.mark.parametrize("raw, why", [
    ("not json", "not JSON"), ("[1, 2]", "not a JSON object"),
    (json.dumps({"round": 1}), "lacks required field"),
    ("{" + " " * ed.MAX_STATE_CHARS + "}", "over"),
])
def test_bad_state_is_refused_before_any_request(tmp_path, fake, raw, why):
    env = decide(tmp_path, fake, raw)
    assert env["status"] == "state_invalid" and why in env["error"] and not _Fake.seen


def test_unknown_values_are_allowed_but_missing_fields_are_not(tmp_path, fake):
    assert decide(tmp_path, fake, state(noise_band="unknown", minutes_left="unknown"))["status"] == "ok"


def test_unreachable_server_is_unavailable(tmp_path):
    env = decide(tmp_path, "http://127.0.0.1:9")
    assert env["status"] == "unavailable" and env["choice"] is None


def test_slow_server_is_a_timeout(tmp_path, fake):
    _Fake.delay = 2.0
    env = decide(tmp_path, fake, timeout_s=0.3)
    assert env["status"] == "timeout"


@pytest.mark.parametrize("body", [
    b"not json", {"error": "nope"},
    {"answers": {"next_step": {"choice": "maybe", "probabilities": {"maybe": 1.0}}}},
    {"answers": dict(answers(), next_step={"choice": "continue", "probabilities": {"continue": 2.0, "stop": 0.1}})},
])
def test_malformed_answers_are_rejected(tmp_path, fake, body):
    _Fake.bodies = [body]
    assert decide(tmp_path, fake)["status"] == "malformed"


def test_a_failed_attempt_is_receipted_but_not_persisted_as_the_decision(tmp_path, fake):
    _Fake.bodies = [b"not json", {"answers": answers("stop", p=0.9)}]
    first = decide(tmp_path, fake)
    second = decide(tmp_path, fake)
    assert first["status"] == "malformed" and second["status"] == "ok" and second["choice"] == "stop"
    assert [a["status"] for a in attempts(tmp_path)] == ["malformed", "ok"]


def test_a_non_loopback_url_is_refused(tmp_path, monkeypatch):
    monkeypatch.delenv("GEAK_EIKOS_ALLOW_REMOTE", raising=False)
    assert decide(tmp_path, "http://10.0.0.5:8000")["status"] == "refused"


def test_unwritable_receipts_are_write_failed(tmp_path, fake):
    blocker = tmp_path / "is_a_file"
    blocker.write_text("x")
    env = ed.decide("round_continue", "/eval/lane", 2, state(), str(blocker), fake, 5.0)
    assert env["status"] == "write_failed" and env["choice"] == "continue"   # answer shown, not persisted


def test_bad_decision_name_never_reads_outside_the_question_dir(tmp_path, fake):
    env = ed.decide("../../etc/passwd", "/eval", 1, state(), str(tmp_path), fake, 5.0)
    assert env["status"] == "state_invalid" and not _Fake.seen


# ---------------------------------------------------------------------------------- CLI
def test_cli_always_prints_one_json_object_and_exits_zero(tmp_path, fake):
    st = state(last_round_outcome="it's \"odd\"\nwith a newline")
    env = dict(os.environ, GEAK_EIKOS_URL=fake)
    out = subprocess.run([sys.executable, "-B", SCRIPT, "--decision", "round_continue", "--scope", "/e v'al",
                          "--round", "2", "--state-json", st, "--receipt-dir", str(tmp_path)],
                         capture_output=True, text=True, env=env, timeout=60)
    assert out.returncode == 0
    d = json.loads(out.stdout)
    assert d["status"] == "ok" and d["state_raw"] == st
    bad = subprocess.run([sys.executable, "-B", SCRIPT, "--decision", "round_continue"],
                         capture_output=True, text=True, timeout=60)
    assert bad.returncode == 0 and json.loads(bad.stdout)["status"] == "state_invalid"
