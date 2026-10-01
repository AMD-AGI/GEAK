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
    s = dict(round=2, directions_used=2, directions_budget=6, tracked_incumbent_speedup=1.21,
             best_seen_speedup=1.234, rounds_without_improvement=1, rounds_without_improvement_limit=2,
             last_round_outcome="none", specialties_dispatched=["algorithm", "host_runtime"],
             minutes_left="unknown", minutes_left_source="no_deadline", noise_band="unknown",
             min_improve=0.01, progress_delta=0.01)
    assert set(s) == set(SPEC["required_state"])
    s.update(kw)
    return json.dumps(s)


# What kernel_lane.js writes after a measured round.
OUTCOME = {"verified_candidates": 2, "winner_speedup": 1.31, "improved": True, "made_progress": True,
           "commit_reported": "not_captured", "tracked_incumbent_after": 1.31}


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
    assert env["state_sha256"] == ed.sha256(st) and "state_raw" not in env  # digest, not the state
    assert env["would_be_action"] == "continue"                           # 0.92 >= 0.8
    line = attempts(tmp_path)[0]
    assert line["questions_version"] == "round_continue.v1" and line["policy_version"] and line["endpoint"] == fake
    sent = _Fake.seen[0]
    assert set(sent) == {"state", "questions"} and sent["questions"] == SPEC["questions"]
    assert json.loads(sent["state"]) == json.loads(st)                    # canonical, same content
    assert len(attempts(tmp_path)) == 1 and attempts(tmp_path)[0]["canonical_state"]
    assert attempts(tmp_path)[0]["state_raw"] == st                       # raw state kept in the receipt


def test_below_threshold_is_recorded_as_abstain(tmp_path, fake):
    _Fake.bodies = [{"answers": answers(p=0.6)}]
    env = decide(tmp_path, fake)
    assert env["status"] == "ok" and env["choice"] == "continue" and env["would_be_action"] == "abstain"


def test_same_key_returns_the_first_decision_without_asking_again(tmp_path, fake):
    first = decide(tmp_path, fake)
    _Fake.bodies = [{"answers": answers(choice="stop", p=0.99)}]          # would differ if asked
    again = decide(tmp_path, fake)
    assert len(_Fake.seen) == 1
    assert again["reused"] and again["choice"] == first["choice"] and again["decision_attempt_id"] == first["attempt_id"]
    assert len(attempts(tmp_path)) == 2                                   # both executions receipted


def test_a_different_round_or_scope_is_a_different_key(tmp_path, fake):
    a = decide(tmp_path, fake, rnd=2)
    b = decide(tmp_path, fake, state(round=3), rnd=3)
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
    ("not json", "not standard JSON"), ("[1, 2]", "not a JSON object"),
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
    assert env["receipt"] == "failed"


def test_bad_decision_name_writes_nothing_anywhere(tmp_path, fake):
    """Astra: '../escaped' was rejected but its failure receipt was written to ../escaped.attempts.jsonl."""
    inner = tmp_path / "receipts"
    inner.mkdir()
    for name in ("../escaped", "../../etc/passwd", "Round", "a/b", "", "x" * 65):
        env = ed.decide(name, "/eval", 2, state(), str(inner), fake, 5.0)
        assert env["status"] == "state_invalid" and env["receipt"] == "not_written", name
    assert not _Fake.seen
    assert sorted(p.name for p in tmp_path.rglob("*")) == ["receipts"]          # nothing written at all


# ---------------------------------------------------------------------------------- CLI
def test_cli_always_prints_one_json_object_and_exits_zero(tmp_path, fake):
    st = state(last_round_outcome=OUTCOME)
    env = dict(os.environ, GEAK_EIKOS_URL=fake)
    out = subprocess.run([sys.executable, "-B", SCRIPT, "--decision", "round_continue", "--scope", "/e v'al \"odd\"\nnl",
                          "--round", "2", "--state-json", st, "--receipt-dir", str(tmp_path)],
                         capture_output=True, text=True, env=env, timeout=60)
    assert out.returncode == 0
    d = json.loads(out.stdout)
    assert d["status"] == "ok" and d["state_sha256"] == ed.sha256(st) and "state_raw" not in d
    assert set(d) == set(ed.ENVELOPE_FIELDS) | {"decision_sha256"}               # compact: only what the lane uses
    assert d["decision_sha256"] == ed.sha256(ed.canonical({k: d[k] for k in ed.DECISION_FIELDS}))
    assert len(out.stdout) < 700                                                  # small for the carrier to copy
    bad = subprocess.run([sys.executable, "-B", SCRIPT, "--decision", "round_continue"],
                         capture_output=True, text=True, timeout=60)
    assert bad.returncode == 0 and json.loads(bad.stdout)["status"] == "state_invalid"


def test_decision_digest_detects_an_altered_choice():
    env = {"attempt_id": "a", "logical_key": "k", "status": "ok", "choice": "continue", "confidence": 0.9,
           "would_be_action": "continue"}
    c = ed.compact(env)
    tampered = dict(c, choice="stop")
    assert ed.decision_digest(tampered) != c["decision_sha256"]
    assert ed.decision_digest(dict(c, extra_field="ok")) == c["decision_sha256"]   # added fields are not decision fields


# ------------------------------------------------------------------ Astra's pilot review (dcb75a1d)
def test_wire_fields_are_never_numbers_so_1_0_survives_js():
    """Finding 1: Python hashes 1.0, JS hashes 1. The wire carries repr strings instead."""
    for conf in (1.0, 0.92, 0.0078125, 1e-07, 5e-324, 0.0):
        c = ed.compact({"attempt_id": "a", "decision_attempt_id": "a", "logical_key": "k", "status": "ok",
                        "choice": "continue", "confidence": conf, "would_be_action": "continue",
                        "reused": False, "persisted": True, "receipt": "written"})
        assert c["confidence"] == repr(conf)
        assert all(v is None or isinstance(v, (str, bool)) for k, v in c.items() if k in ed.DECISION_FIELDS)
        js_view = json.loads(json.dumps(c))                                       # what a JS parse sees
        assert ed.decision_digest(js_view) == c["decision_sha256"]


def test_a_failed_attempt_receipt_is_reported_even_when_the_decision_is_fine(tmp_path, fake):
    """Finding 2: attempts.jsonl unwritable, index writable -> status ok must not hide the missing row."""
    (tmp_path / "round_continue.attempts.jsonl").mkdir()
    env = decide(tmp_path, fake)
    assert env["status"] == "ok" and env["persisted"] and env["receipt"] == "failed"
    again = decide(tmp_path, fake)                                               # reused decision, same problem
    assert again["reused"] and again["receipt"] == "failed"
    _Fake.bodies = [b"not json"]
    bad = ed.decide("round_continue", "/eval/other", 2, state(), str(tmp_path), fake, 5.0)
    assert bad["status"] == "malformed" and bad["receipt"] == "failed"          # failed attempts too


def _first_index_entry(tmp_path):
    p = tmp_path / "round_continue.index.json"
    idx = json.loads(p.read_text())
    (k, v), = idx.items()
    return p, idx, k, v


def test_a_tampered_cached_decision_is_not_reused(tmp_path, fake):
    """Finding 4: an index entry edited to choice=stop over continue-0.92 answers was reused as ok."""
    decide(tmp_path, fake)
    p, idx, k, v = _first_index_entry(tmp_path)
    v.update(choice="stop", confidence=0.92, would_be_action="stop")
    p.write_text(json.dumps(idx))
    env = decide(tmp_path, fake)
    assert env["status"] == "ok" and not env["reused"] and env["choice"] == "continue"
    assert len(_Fake.seen) == 2                                                  # asked again, not trusted
    _, _, _, healed = _first_index_entry(tmp_path)
    assert healed["choice"] == "continue"                                        # invalid entry replaced


def test_a_changed_endpoint_is_a_different_key(tmp_path, fake):
    """Finding 4: localhost:8000 vs :8999 shared a key and reused the first endpoint's answer."""
    a = decide(tmp_path, fake)
    other = fake.replace("127.0.0.1", "localhost")                               # same server, different endpoint id
    b = decide(tmp_path, other)
    assert a["logical_key"] != b["logical_key"] and not b["reused"] and len(_Fake.seen) == 2


def test_a_losing_racer_records_what_it_saw_apart_from_the_decision(tmp_path, fake, monkeypatch):
    """Finding 5: the loser's stop answers were written next to the winner's continue decision."""
    real_post = ed.er.post_systemone
    won = {}

    def interleave(url, st, qs, t):
        body = real_post(url, st, qs, t)                                        # this attempt sees 'stop'
        ed.er.post_systemone = real_post                                        # the competitor uses the real transport
        try:
            won["env"] = ed.decide("round_continue", "/eval/lane", 2, state(), str(tmp_path), fake, 5.0)
        finally:
            ed.er.post_systemone = interleave
        return body
    first_body = {"answers": answers("stop", p=0.95)}
    _Fake.bodies = [first_body, {"answers": answers("continue", p=0.92)}]
    monkeypatch.setattr(ed.er, "post_systemone", interleave)
    env = decide(tmp_path, fake)
    assert won["env"]["choice"] == "continue" and won["env"]["persisted"]        # competitor committed first
    assert env["choice"] == "continue" and env["reused"]                          # the persisted winner
    loser = [a for a in attempts(tmp_path) if a["attempt_id"] == env["attempt_id"]][0]
    assert loser["observed"]["choice"] == "stop"                                  # what it actually saw
    assert loser["choice"] == "continue" and loser["decision_attempt_id"] != loser["attempt_id"]
    assert "answers" not in loser or loser["answers"] is None


@pytest.mark.parametrize("bad, why", [
    (dict(round=-7), "round must be"), (dict(round=3), "round must be"),
    (dict(directions_used="many"), "directions_used"), (dict(directions_budget=-1), "directions_budget"),
    (dict(tracked_incumbent_speedup=0), "tracked_incumbent_speedup"),
    (dict(specialties_dispatched=["algorithm", "algorithm"]), "specialties"),
    (dict(specialties_dispatched=["magic"]), "specialties"),
    (dict(minutes_left_source="guess"), "minutes_left_source"), (dict(minutes_left=-3), "minutes_left"),
    (dict(noise_band="small"), "noise_band"), (dict(last_round_outcome=7), "last_round_outcome"),
    (dict(rounds_without_improvement=True), "rounds_without_improvement"),
])
def test_state_values_are_checked_not_just_present(tmp_path, fake, bad, why):
    """Finding 6: round=-7, directions_used='many', budget=-1 reached the transport as status ok."""
    env = decide(tmp_path, fake, json.dumps(json.loads(state()) | bad))
    assert env["status"] == "state_invalid" and why in env["error"] and not _Fake.seen


@pytest.mark.parametrize("const", ["NaN", "Infinity", "-Infinity"])
def test_non_standard_json_numbers_are_refused(tmp_path, fake, const):
    raw = state().replace('"tracked_incumbent_speedup":1.21', '"tracked_incumbent_speedup":' + const)
    raw = raw.replace('"tracked_incumbent_speedup": 1.21', '"tracked_incumbent_speedup": ' + const)
    assert const in raw
    env = decide(tmp_path, fake, raw)
    assert env["status"] == "state_invalid" and "non-standard" in env["error"] and not _Fake.seen



# ------------------------------------------------------------- second review (880582df)
def test_a_negative_progress_delta_is_accepted_as_the_lane_allows(tmp_path, fake):
    """kernel_lane.js accepts any finite progress_delta > -1; the validator must not refuse it."""
    assert decide(tmp_path, fake, state(progress_delta=-0.1))["status"] == "ok"
    assert decide(tmp_path, fake, state(progress_delta=-1), rnd=2, scope="/x")["status"] == "state_invalid"


@pytest.mark.parametrize("bad", [
    dict(specialties_dispatched=[{}]), dict(minutes_left_source={}),            # unhashable -> was TypeError
    dict(tracked_incumbent_speedup=10 ** 400),                                  # was OverflowError
    dict(last_round_outcome={"verified_candidates": -3}),                       # any object was accepted
    dict(last_round_outcome=dict(OUTCOME, verified_candidates=-3)),
    dict(last_round_outcome=dict(OUTCOME, extra=1)),
    dict(last_round_outcome=dict(OUTCOME, commit_reported=True)),
])
def test_bad_nested_state_is_a_receipted_refusal(tmp_path, fake, bad):
    env = decide(tmp_path, fake, json.dumps(json.loads(state()) | bad))
    assert env["status"] == "state_invalid" and env["receipt"] == "written" and not _Fake.seen
    assert [a["attempt_id"] for a in attempts(tmp_path)] == [env["attempt_id"]]


def test_real_lane_outcomes_are_accepted(tmp_path, fake):
    for i, o in enumerate(["none", OUTCOME, dict(OUTCOME, winner_speedup=None, verified_candidates=0,
                                                   improved=False, made_progress=False)]):
        assert decide(tmp_path, fake, state(last_round_outcome=o), scope="/s%d" % i)["status"] == "ok"


def test_an_overflowing_nested_number_never_reaches_the_model(tmp_path, fake):
    """1e999 is standard JSON spelling but parses to inf; it must not go out as Infinity."""
    raw = state(last_round_outcome=dict(OUTCOME)).replace('"winner_speedup": 1.31', '"winner_speedup": 1e999')
    assert "1e999" in raw
    env = decide(tmp_path, fake, raw)
    assert env["status"] == "state_invalid" and "non-finite" in env["error"] and not _Fake.seen


@pytest.mark.parametrize("edit", [
    dict(scope="/other/lane"), dict(round=3), dict(attempt_id=""), dict(attempt_id="x" * 32),
])
def test_a_cached_entry_with_the_wrong_identity_is_not_reused(tmp_path, fake, edit):
    decide(tmp_path, fake)
    p, idx, k, v = _first_index_entry(tmp_path)
    v.update(edit)
    p.write_text(json.dumps(idx))
    env = decide(tmp_path, fake)
    assert not env["reused"] and len(_Fake.seen) == 2


def test_an_entry_copied_from_another_scope_is_not_adopted(tmp_path, fake):
    """Valid entry for /source (continue) copied under /target's key must not be adopted by /target."""
    _Fake.bodies = [{"answers": answers("continue")}, {"answers": answers("stop", p=0.95)}]
    src = decide(tmp_path, fake, scope="/source/lane")
    tgt = decide(tmp_path, fake, scope="/target/lane")
    p = tmp_path / "round_continue.index.json"
    idx = json.loads(p.read_text())
    idx[tgt["logical_key"]] = idx[src["logical_key"]]
    p.write_text(json.dumps(idx))
    _Fake.bodies = [{"answers": answers("stop", p=0.95)}]
    again = decide(tmp_path, fake, scope="/target/lane")
    assert again["choice"] == "stop" and again["decision_attempt_id"] != src["attempt_id"]
