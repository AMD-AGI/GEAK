#!/usr/bin/env python3
"""agent_stage.dispatch: the opt-in salvage phase after an exhausted budget.

The runner session and the slot finalize are faked, so this runs on a CPU in under a
second. Run:  python3 -m pytest -q workload/engine/core/tests/test_agent_stage_salvage.py
"""
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

CORE = Path(__file__).resolve().parents[1]
WORKLOAD = CORE.parents[1]
sys.path.insert(0, str(CORE))
import agent_stage  # noqa: E402
import stage as stage_mod  # noqa: E402

SALVAGE_BANNER = "THIS IS A SALVAGE SESSION"


class FakeSessions:
    """Stands in for `timeout <left> bash runner prompt geak_root log`.

    `deliver_in_salvage`: a salvage session leaves a deliverable on disk, which the fake
    finalize then grades ok. A normal session never does, so the normal budget is always
    spent without an ok."""

    def __init__(self, out, deliver_in_salvage=True):
        self.out, self.deliver = out, deliver_in_salvage
        self.calls = []

    def __call__(self, argv, env=None):
        prompt = Path(argv[4]).read_text()
        salvage = SALVAGE_BANNER in prompt
        self.calls.append({"timeout": int(argv[1]), "salvage": salvage, "prompt": prompt})
        Path(argv[6]).write_text("session log\n")
        if salvage and self.deliver:
            (self.out / "DELIVERED").write_text("")
        return 0


def fake_finalize(write=True):
    """write=False: the slot never produces a valid receipt for a normal session (the
    "attempts ended without a valid result.json" path); it still grades a delivery."""
    def finalize(env, log_cb=print):
        out = Path(env["OUT"])
        ok = (out / "DELIVERED").exists()
        if not write and not ok:
            return
        doc = {"stage": "profile", "skill": "kernel-extract",
               "status": "ok" if ok else "blocked", "artifacts": {"marker": "params.json"},
               "metrics": {}, "notes": "" if ok else "task tB has no RETURN.json"}
        (out / "result.json").write_text(json.dumps(doc))
    return finalize


class SalvageDispatchTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.mkdtemp(prefix="agent_salvage_")
        self.addCleanup(shutil.rmtree, tmp)
        self.out = Path(tmp) / "profile" / "out"
        self.out.mkdir(parents=True)
        (self.out / "params.json").write_text("{}")
        runner = Path(tmp) / "runner.sh"
        runner.write_text("exit 0\n")
        self.env = {"OUT": str(self.out), "RUNNER": str(runner), "GEAK_ROOT": tmp,
                    "WORKLOAD_ROOT": str(WORKLOAD), "STAGE": "profile",
                    "STAGE_SKILL": "kernel-extract"}
        self.logs = []

    def lay(self, **params):
        base = {"agent_budget_sec": 900, "agent_attempts": 2}
        base.update(params)
        return {"stage": "profile", "skill": "kernel-extract", "params": base,
                "out": str(self.out), "slot_root": str(WORKLOAD / "stages" / "profile"),
                "skill_md": str(WORKLOAD / "stages" / "profile" / "kernel-extract" /
                                "SKILL.md")}

    def dispatch(self, lay, sessions, finalize=None):
        with mock.patch.object(agent_stage.subprocess, "call", sessions), \
                mock.patch.object(agent_stage, "finalize", finalize or fake_finalize()):
            return agent_stage.dispatch({}, lay, self.env, log_cb=self.logs.append)

    def marker(self):
        return json.loads((self.out / "salvage.json").read_text())

    # -- default: unchanged

    def test_default_budget_spent_returns_1_without_salvage(self):
        sessions = FakeSessions(self.out)
        rc, note = self.dispatch(self.lay(), sessions, fake_finalize(write=False))
        self.assertEqual(rc, 1)
        self.assertIn("2 agent attempts ended without a valid result.json", note)
        self.assertEqual(len(sessions.calls), 2)
        self.assertFalse(any(c["salvage"] for c in sessions.calls))
        self.assertFalse((self.out / "salvage.json").exists())

    def test_default_blocked_verdict_is_the_answer(self):
        sessions = FakeSessions(self.out)
        rc, note = self.dispatch(self.lay(), sessions)
        self.assertEqual((rc, note), (0, None))
        self.assertEqual(json.loads((self.out / "result.json").read_text())["status"],
                         "blocked")
        self.assertFalse((self.out / "salvage.json").exists())

    # -- opted in

    def test_salvage_entered_once_with_its_own_budget_and_prompt(self):
        sessions = FakeSessions(self.out)
        lay = self.lay(on_budget_exhausted="deliver_qualified", salvage_budget_sec=500,
                       salvage_attempts=3)
        rc, note = self.dispatch(lay, sessions, fake_finalize(write=False))
        self.assertEqual((rc, note), (0, None))
        self.assertEqual([c["salvage"] for c in sessions.calls], [False, False, True])
        salvage = sessions.calls[-1]
        self.assertLessEqual(salvage["timeout"], 500)
        self.assertGreater(salvage["timeout"], 400)
        self.assertIn("profile budget exhausted (salvage)", salvage["prompt"])
        self.assertIn("2 agent attempts ended", salvage["prompt"])
        self.assertTrue(any(line.startswith("[agent] budget spent; entering salvage")
                            for line in self.logs))
        doc = self.marker()
        self.assertEqual(doc["schema"], "workload/agent_salvage/v1")
        self.assertEqual(doc["normal_sessions"], 2)
        self.assertEqual([s["session"] for s in doc["sessions"]], [3])
        self.assertEqual(doc["sessions"][0]["status"], "ok")

    def test_salvage_entered_after_blocked_last_attempt(self):
        sessions = FakeSessions(self.out)
        lay = self.lay(on_budget_exhausted="deliver_qualified")
        rc, _ = self.dispatch(lay, sessions)
        self.assertEqual(rc, 0)
        self.assertEqual(sum(c["salvage"] for c in sessions.calls), 1)
        self.assertIn("task tB has no RETURN.json", self.marker()["trigger"])

    def test_salvage_attempts_are_bounded(self):
        sessions = FakeSessions(self.out, deliver_in_salvage=False)
        lay = self.lay(on_budget_exhausted="deliver_qualified", salvage_attempts=2)
        rc, note = self.dispatch(lay, sessions)
        # The slot's blocked verdict stands and is recorded by the driver.
        self.assertEqual((rc, note), (0, None))
        self.assertEqual([c["salvage"] for c in sessions.calls], [False, False, True, True])
        self.assertEqual(len(self.marker()["sessions"]), 2)
        self.assertIn("task tB has no RETURN.json", sessions.calls[-1]["prompt"])

    # -- resume

    def test_marker_resumes_salvage_not_the_normal_loop(self):
        (self.out / "agent_prompt.4.txt").write_text("earlier")
        (self.out / "salvage.json").write_text(json.dumps({
            "schema": "workload/agent_salvage/v1", "trigger": "budget spent",
            "sessions": [{"session": 4, "started_unix": 1, "seconds": 100, "exit": 124,
                          "status": "blocked", "finalize_notes": "kB unaccounted"}]}))
        sessions = FakeSessions(self.out)
        # The mode is not set on this invocation: the marker alone keeps it in salvage.
        rc, _ = self.dispatch(self.lay(salvage_budget_sec=500), sessions)
        self.assertEqual(rc, 0)
        self.assertEqual(len(sessions.calls), 1)
        self.assertTrue(sessions.calls[0]["salvage"])
        self.assertLessEqual(sessions.calls[0]["timeout"], 400)   # 500 - 100 already spent
        self.assertIn("kB unaccounted", sessions.calls[0]["prompt"])
        self.assertTrue((self.out / "agent_prompt.5.txt").exists())
        self.assertEqual(len(self.marker()["sessions"]), 2)

    def test_spent_salvage_starts_nothing_and_rejudges(self):
        (self.out / "salvage.json").write_text(json.dumps({
            "schema": "workload/agent_salvage/v1", "trigger": "budget spent",
            "sessions": [{"session": n, "seconds": 10} for n in (3, 4, 5)]}))
        sessions = FakeSessions(self.out)
        judged = []

        def finalize(env, log_cb=print):
            judged.append(1)
            fake_finalize()(env, log_cb)

        lay = self.lay(on_budget_exhausted="deliver_qualified", salvage_attempts=3)
        rc, _ = self.dispatch(lay, sessions, finalize)
        self.assertEqual(rc, 0)
        self.assertEqual(sessions.calls, [])
        self.assertEqual(judged, [1])
        self.assertEqual(json.loads((self.out / "result.json").read_text())["status"],
                         "blocked")

    def test_lost_session_is_charged(self):
        log = self.out / "agent.3.log"
        log.write_text("x")
        os.utime(log, (1300, 1300))
        (self.out / "salvage.json").write_text(json.dumps({
            "schema": "workload/agent_salvage/v1", "trigger": "budget spent",
            "sessions": [{"session": 3, "started_unix": 1000, "seconds": None}]}))
        sessions = FakeSessions(self.out)
        self.dispatch(self.lay(salvage_budget_sec=500), sessions)
        self.assertLessEqual(sessions.calls[0]["timeout"], 200)
        self.assertEqual(self.marker()["sessions"][0]["seconds"], 300)

    def test_unreadable_marker_starts_nothing(self):
        (self.out / "salvage.json").write_text("{not json")
        sessions = FakeSessions(self.out)
        rc, note = self.dispatch(self.lay(), sessions)
        self.assertEqual(rc, 1)
        self.assertIn("unreadable", note)
        self.assertEqual(sessions.calls, [])

    # -- configuration

    def test_invalid_settings_are_refused(self):
        for params in ({"on_budget_exhausted": "keep_going"},
                       {"on_budget_exhausted": "deliver_qualified", "salvage_attempts": 0},
                       {"salvage_budget_sec": "2h"}):
            with self.assertRaises(ValueError):
                agent_stage.salvage_policy(self.lay(**params))
        lay = dict(self.lay(on_budget_exhausted="deliver_qualified"),
                   slot_root=str(self.out))          # a slot with no salvage.md
        with self.assertRaises(ValueError):
            agent_stage.salvage_policy(lay)
        problems = []
        stage_mod.check_skill(dict(lay, mode="agent", finalize=__file__,
                                   skill_root=str(self.out), run_sh=""), problems)
        self.assertTrue(any("salvage.md" in p for p in problems), problems)

    def test_default_policy_is_stop(self):
        policy = agent_stage.salvage_policy(self.lay())
        self.assertEqual((policy["mode"], policy["budget"], policy["attempts"]),
                         ("stop", 7200, 3))



class PartialDeliveryReportingTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.mkdtemp(prefix="agent_salvage_report_")
        self.addCleanup(shutil.rmtree, tmp)
        self.root = Path(tmp)
        self.out = self.root / "profile" / "out"
        self.out.mkdir(parents=True)
        (self.out / "x.json").write_text("{}")
        self.lay = {"stage": "profile", "skill": "kernel-extract", "out": str(self.out)}

    def result(self, **extra):
        doc = {"stage": "profile", "skill": "kernel-extract", "status": "ok",
               "artifacts": {"x": "x.json"}, "metrics": {}}
        doc.update(extra)
        (self.out / "result.json").write_text(json.dumps(doc))

    def test_partial_is_labelled_in_the_report(self):
        import report
        self.result(delivery="partial", notes="PARTIAL DELIVERY (profile budget salvage)")
        (self.root / "STATE.json").write_text(json.dumps(
            {"stages": {"profile": {"status": "ok", "delivery": "partial"}}}))
        rows = report.stage_rows(str(self.root))
        self.assertEqual(rows[0]["status"], "ok")        # still admitted downstream
        self.assertIn("| ok (partial delivery) |", "\n".join(report.stage_table(rows)))

    def test_full_delivery_row_is_unchanged(self):
        import report
        self.result()
        rows = report.stage_rows(str(self.root))
        self.assertIn("| ok |", "\n".join(report.stage_table(rows)))

    def test_delivery_value_is_validated(self):
        self.result(delivery="partial")
        self.assertEqual(stage_mod.validate_result(self.lay)[1], [])
        self.result(delivery="mostly")
        self.assertTrue(any("delivery" in e for e in stage_mod.validate_result(self.lay)[1]))


if __name__ == "__main__":
    unittest.main()
