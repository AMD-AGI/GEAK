"""Incremental signing in run_points.py: profile_run.json is a 'running' checkpoint that
holds every signed record as soon as its census is complete, so a hard kill keeps the
finished censuses and a rerun recaptures only what the killed invocation left unsigned."""
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_eager_prefix_and_census_knobs import _RunPoints  # noqa: E402
from test_profile_fast_path import (FAST, RANKS, SCRIPTS, FakeServer,  # noqa: E402
                                    write_trace, prefill_events)

import probe_server  # noqa: E402
import run_points  # noqa: E402

ORDER = ["pr_64", "pr_256", "de_1", "de_2"]
# finished pools of the Kimi-K3.1-Pro vLLM capture (read only; skipped where absent)
REAL_POOLS = [
    "/home/chuschen/kimi-geak-build/runs/kimik31_mi355x_vllm_c1_multicfg_r2/profile/out/"
    "sampling/deployment_9e1824f4b14fe413",
    "/home/chuschen/kimi-geak-build/runs/kimik31_mi355x_vllm_dram_multicfg/profile/out/"
    "sampling/deployment_5ec1bc7f46a5a2c9",
]


class HangingServer(FakeServer):
    """A FakeServer whose `hang_at`-th /start_profile never answers (a preempted node)."""

    def __init__(self, hang_at, **kw):
        super().__init__(**kw)
        self.httpd.daemon_threads = True
        self.httpd.handle_error = lambda *a: None     # the killed client's broken pipes
        self.hanging = threading.Event()
        self.release = threading.Event()
        self.starts = 0
        handler = self.httpd.RequestHandlerClass
        orig = handler.do_POST
        fake = self

        def do_POST(h):
            if h.path == "/start_profile":
                fake.starts += 1
                if fake.starts == hang_at:
                    fake.hanging.set()
                    fake.release.wait(120)
                    return
            orig(h)
        handler.do_POST = do_POST

    def close(self):
        self.release.set()
        super().close()


class _Kill(_RunPoints):
    def sign_plan(self):
        plan = json.loads((self.root / "points.json").read_text())
        for p in plan["points"]:
            p["measurement_signature"] = "sig-" + p["point_id"]
        (self.root / "points.json").write_text(json.dumps(plan))

    def env(self, extra=None):
        e = {k: v for k, v in os.environ.items()
             if k not in ("GEAK_CENSUS_SERIAL", "GEAK_CENSUS_WORKERS",
                          "GEAK_BOUNDARY_RESTART_GB", "GEAK_MEMINFO")}
        e.update(extra or {})
        return e

    def start(self, server, *extra, env=None):
        return subprocess.Popen(
            [sys.executable, str(SCRIPTS / "run_points.py"), "--points",
             str(self.root / "points.json"), "--base-url", server.url,
             "--trace-root", str(self.root / "traces"), "--expect-files", str(RANKS),
             "--eval-dir", str(self.ev), *FAST, *map(str, extra)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            env=self.env(env), start_new_session=True)

    def kill_when(self, proc, server, signed, timeout=90):
        """SIGKILL the run (and its children) once it hangs in a window and the
        checkpoint holds exactly `signed`; returns that checkpoint."""
        t0, doc = time.time(), None
        while time.time() - t0 < timeout:
            self.assertIsNone(proc.poll(), proc.stdout.read() if proc.poll() is not None else "")
            try:
                doc = self.run_doc()
            except (OSError, ValueError):
                doc = None
            if (server.hanging.is_set() and doc and doc.get("status") == run_points.RUNNING
                    and set(doc["points"]) == set(signed)):
                break
            time.sleep(0.1)
        else:
            os.killpg(proc.pid, signal.SIGKILL)
            self.fail("checkpoint never held %s: %s" % (signed, doc))
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(30)
        proc.stdout.close()
        return doc

    def probed(self):
        return [pid for s, pid, _a in self.log_cmds() if s == "serving_probe.py"]


class KillMidPoolTest(_Kill):
    def _kill_and_resume(self, env):
        self.sign_plan()
        server = HangingServer(hang_at=3)            # pr_64, pr_256 done; de_1 hangs
        self.addCleanup(server.close)
        proc = self.start(server, "--decode-eager", "--no-evidence", env=env)
        doc = self.kill_when(proc, server, ["pr_64", "pr_256"])
        self.assertEqual(doc["signed_points"], 2)
        for pid in ("pr_64", "pr_256"):
            rec = doc["points"][pid]
            self.assertEqual(rec["measurement_signature"], "sig-" + pid)
            self.assertEqual(rec["census"]["rc"], 0)
            self.assertEqual(rec["census"]["out"],
                             str(self.ev / "evidence" / "census" / (pid + ".json")))
        # what a kill during de_1 can leave: a partial census and a window's traces
        census = self.ev / "evidence" / "census"
        (census / "de_1.json").write_text('{"point_id": "de_1", "per_')
        stale = self.root / "traces" / "de_1"
        stale.mkdir(parents=True, exist_ok=True)
        write_trace(str(stale / "0.0000-TP-0.trace.json.gz"), prefill_events(64))
        self.assertEqual(self.probed(), ["pr_64", "pr_256"])
        # the rerun: the same command on a fresh server
        fresh = FakeServer()
        self.addCleanup(fresh.close)
        p = self.run_points(fresh, "--decode-eager", "--no-evidence", env=env)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("recapturing 1 point(s) it left unsigned: de_1", p.stdout)
        doc = self.run_doc()
        self.assertEqual(doc["status"], "ok")
        self.assertEqual(doc["recaptured_after_kill"], ["de_1"])
        self.assertNotIn("signed_points", doc)
        self.assertTrue(doc["points"]["pr_64"]["resumed"])
        self.assertTrue(doc["points"]["pr_256"]["resumed"])
        self.assertNotIn("resumed", doc["points"]["de_1"])
        self.assertNotIn("resumed", doc["points"]["de_2"])
        self.assertEqual(len(fresh.profile_bodies), 2)
        self.assertEqual(self.probed(), ["pr_64", "pr_256", "de_1", "de_2"])
        tag = doc["points"]["de_1"]["attempts"][0]["tag"]
        self.assertTrue(tag.startswith("de_1_"), tag)      # fresh tag, not the stale dir
        c = json.loads((census / "de_1.json").read_text())
        self.assertEqual(c["trace_dir"], str(self.root / "traces" / tag))
        for pid in ORDER:
            self.assertEqual(doc["points"][pid]["measurement_signature"], "sig-" + pid)
        self.assertEqual(doc["points_without_census"], [])
        # and the next rerun resumes everything
        third = FakeServer()
        self.addCleanup(third.close)
        p = self.run_points(third, "--decode-eager", "--no-evidence", env=env)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertEqual(third.profile_bodies, [])
        self.assertNotIn("recaptured_after_kill", self.run_doc())

    def test_serial_census_signed_points_survive_a_kill(self):
        self._kill_and_resume({"GEAK_CENSUS_SERIAL": "1"})

    def test_parallel_census_signed_points_survive_a_kill(self):
        # pr_256's census completes while de_1's window hangs: signed on the census thread
        self._kill_and_resume({})

    def test_a_recapture_cut_short_by_a_finished_stop_is_still_recaptured(self):
        # A kill leaves de_1 with a partial census; the rerun takes it as a recapture but
        # its server goes down before de_1 is captured, so it finishes "server down" with
        # no signed record of de_1. The next launch must recapture de_1 again, neither
        # refusing the unsigned file nor resuming the partial census.
        self.sign_plan()
        env = {"GEAK_CENSUS_SERIAL": "1"}
        server = HangingServer(hang_at=3)
        self.addCleanup(server.close)
        proc = self.start(server, "--decode-eager", "--no-evidence", env=env)
        self.kill_when(proc, server, ["pr_64", "pr_256"])
        census = self.ev / "evidence" / "census"
        (census / "de_1.json").write_text('{"point_id": "de_1", "per_')
        dead = FakeServer()
        dead.close()
        dead.httpd.server_close()          # nothing listens: /health is refused
        p = self.run_points(dead, "--decode-eager", "--no-evidence", env=env)
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
        doc = self.run_doc()
        self.assertEqual(doc["status"], "server down")
        self.assertEqual(doc["recaptured_after_kill"], ["de_1"])
        self.assertNotIn("census", doc["points"].get("de_1", {}))
        fresh = FakeServer()
        self.addCleanup(fresh.close)
        p = self.run_points(fresh, "--decode-eager", "--no-evidence", env=env)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        doc = self.run_doc()
        self.assertEqual(doc["status"], "ok")
        self.assertEqual(doc["recaptured_after_kill"], ["de_1"])
        self.assertTrue(doc["points"]["de_1"]["attempts"][-1]["tag"].startswith("de_1_"))
        self.assertEqual(doc["points"]["de_1"]["census"]["rc"], 0)
        self.assertEqual(json.loads((census / "de_1.json").read_text())["point_id"], "de_1")
        self.assertEqual(doc["points_without_census"], [])
        # once signed it is resumed like any other point
        third = FakeServer()
        self.addCleanup(third.close)
        p = self.run_points(third, "--decode-eager", "--no-evidence", env=env)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertEqual(third.profile_bodies, [])
        self.assertNotIn("recaptured_after_kill", self.run_doc())

    def test_a_kill_during_a_redo_drops_the_old_record_and_the_rerun_redoes_it(self):
        self.sign_plan()
        server = FakeServer()
        self.addCleanup(server.close)
        self.assertEqual(self.run_points(server, "--decode-eager", "--no-evidence")
                         .returncode, 0)
        hang = HangingServer(hang_at=1)
        self.addCleanup(hang.close)
        proc = self.start(hang, "--decode-eager", "--no-evidence", "--redo", "de_1")
        doc = self.kill_when(proc, hang, ["pr_64", "pr_256", "de_2"])
        self.assertTrue(all(r["resumed"] for r in doc["points"].values()))
        # de_1's old census is still on disk but unsigned in the checkpoint
        self.assertTrue((self.ev / "evidence" / "census" / "de_1.json").exists())
        fresh = FakeServer()
        self.addCleanup(fresh.close)
        p = self.run_points(fresh, "--decode-eager", "--no-evidence", "--redo", "de_1")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertEqual(len(fresh.profile_bodies), 1)
        doc = self.run_doc()
        self.assertEqual(doc["status"], "ok")
        self.assertEqual(doc["points"]["de_1"]["measurement_signature"], "sig-de_1")
        # without --redo the checkpoint alone also makes the rerun recapture it
        hang2 = HangingServer(hang_at=1)
        self.addCleanup(hang2.close)
        proc = self.start(hang2, "--decode-eager", "--no-evidence", "--redo", "de_1")
        self.kill_when(proc, hang2, ["pr_64", "pr_256", "de_2"])
        fresh2 = FakeServer()
        self.addCleanup(fresh2.close)
        p = self.run_points(fresh2, "--decode-eager", "--no-evidence")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertEqual(self.run_doc()["recaptured_after_kill"], ["de_1"])
        self.assertEqual(len(fresh2.profile_bodies), 1)


class StatusTransitionTest(_Kill):
    def test_running_checkpoints_then_the_final_status(self):
        self.sign_plan()
        server = FakeServer()
        self.addCleanup(server.close)
        proc = self.start(server, "--decode-eager", "--no-evidence",
                          env={"GEAK_CENSUS_SERIAL": "1"})
        seen = []
        while proc.poll() is None:
            try:
                d = self.run_doc()
                key = (d.get("status"), tuple(sorted(d.get("points", {}))))
                if not seen or seen[-1] != key:
                    seen.append(key)
            except (OSError, ValueError) as e:      # never a partial file
                if (self.ev / "profile_run.json").exists():
                    self.fail("partial profile_run.json: %s" % e)
            time.sleep(0.02)
        out = proc.stdout.read()
        proc.stdout.close()
        self.assertEqual(proc.returncode, 0, out)
        doc = self.run_doc()
        seen.append((doc["status"], tuple(sorted(doc["points"]))))
        self.assertEqual(doc["status"], "ok")
        self.assertEqual(seen[-1][0], "ok")
        running = [pts for st, pts in seen if st == run_points.RUNNING]
        self.assertTrue(running)
        # signed counts only grow while running
        self.assertEqual(running, sorted(running, key=len))
        self.assertEqual({st for st, _p in seen if st != run_points.RUNNING}, {"ok"})
        self.assertEqual(set(doc), {"started", "points", "server", "decode_eager",
                                    "census_serial", "census_workers", "boundary_restart_gb",
                                    "decode_deferred_to_eager", "capture_and_census_s",
                                    "status", "total_s", "points_without_census", "post"})
        self.assertFalse(list(self.ev.glob("profile_run.json.tmp*")))

    def test_an_unsigned_census_under_a_finished_file_is_refused_before_any_capture(self):
        self.sign_plan()
        server = FakeServer()
        self.addCleanup(server.close)
        self.assertEqual(self.run_points(server, "--decode-eager", "--no-evidence")
                         .returncode, 0)
        doc = self.run_doc()
        del doc["points"]["de_2"]["measurement_signature"]
        (self.ev / "profile_run.json").write_text(json.dumps(doc))
        # pr_256 needs a capture; de_2 (later in order) is unsigned
        (self.ev / "evidence" / "census" / "pr_256.json").unlink()
        second = FakeServer()
        self.addCleanup(second.close)
        p = self.run_points(second, "--decode-eager", "--no-evidence")
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("incompatible or unsigned shared census for de_2", p.stdout + p.stderr)
        self.assertEqual(second.profile_bodies, [])
        self.assertEqual(json.loads((self.ev / "profile_run.json").read_text()), doc)

    def test_probe_server_takes_a_running_file_as_unfinished(self):
        self.ev.mkdir(parents=True)
        (self.ev / "evidence").mkdir()
        (self.ev / "evidence" / "eager_evidence_request.json").write_text(
            json.dumps({"point": "de_1", "kernels": ["k"]}))
        cmd = ["run_points.py", "--eval-dir", str(self.ev)]
        (self.ev / "profile_run.json").write_text(json.dumps({"status": "ok"}))
        self.assertTrue(probe_server.evidence_requested(cmd))
        (self.ev / "profile_run.json").write_text(json.dumps({"status": run_points.RUNNING}))
        self.assertFalse(probe_server.evidence_requested(cmd))


class AtomicWriteTest(unittest.TestCase):
    def test_same_bytes_as_before_and_no_partial_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "profile_run.json")
            doc = {"status": "ok", "points": {"a": {"x": [1, 2]}}}
            run_points.write_json_atomic(path, doc)
            with open(path) as fh:
                self.assertEqual(fh.read(), json.dumps(doc, indent=1))
            with self.assertRaises(TypeError):
                run_points.write_json_atomic(path, {"status": "running", "bad": object()})
            with open(path) as fh:
                self.assertEqual(json.load(fh), doc)          # the old file, intact
            self.assertEqual(os.listdir(d), ["profile_run.json"])


class CensusCompleteTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)
        self.trace = self.d / "traces" / "p"
        self.trace.mkdir(parents=True)
        for r in range(2):
            (self.trace / ("1.0-TP-%d.trace.json.gz" % r)).write_text("x")
        self.out = self.d / "p.json"

    def census(self, **kw):
        doc = {"point_id": "p", "trace_dir": str(self.trace), "instances": 3, "ranks": 2,
               "per_rank": {"0": {}, "1": {}}}
        doc.update(kw)
        self.out.write_text(json.dumps(doc))
        return {"rc": 0, "out": str(self.out), "trace_dir": str(self.trace)}

    def test_complete_census_is_signed(self):
        self.assertIsNone(run_points.census_complete(self.census(), "p", 2))

    def test_incomplete_evidence_is_never_signed(self):
        bad = [
            dict(self.census(), rc=1, out=None),
            dict(self.census(), graph_hidden=True),
            self.census(point_id="q"),
            self.census(trace_dir=str(self.d / "elsewhere")),
            self.census(instances=0),
            self.census(ranks=1, per_rank={"0": {}}),
        ]
        for c in bad:
            self.assertIsNotNone(run_points.census_complete(c, "p", 2), c)
        c = self.census()
        self.out.write_text('{"point_id": "p", "tra')
        self.assertIn("unreadable", run_points.census_complete(c, "p", 2))
        c = self.census()
        (self.trace / "1.0-TP-1.trace.json.gz").unlink()
        self.assertIn("1 of 2 files", run_points.census_complete(c, "p", 2))


class ResumePlanTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)
        self.census, self.traces = self.d / "census", self.d / "traces"
        self.census.mkdir()
        self.traces.mkdir()
        self.points = [{"point_id": p, "phase": "PREFILL", "measurement_signature": "s-" + p}
                       for p in ("a", "b", "c")]
        for p in ("a", "b"):
            (self.census / (p + ".json")).write_text("{}")

    def plan(self, status, signed, redo=()):
        doc = {"status": status, "points": {p: {"measurement_signature": "s-" + p}
                                            for p in signed}}
        return run_points.resume_plan(self.points, doc, str(self.census),
                                      str(self.traces), redo)

    def test_finished_file_keeps_the_old_contract(self):
        self.assertEqual(self.plan("ok", ["a", "b"]), (["a", "b"], []))
        with self.assertRaisesRegex(SystemExit, "unsigned shared census for b"):
            self.plan("server down", ["a"])
        self.assertEqual(self.plan("ok", ["a"], redo=["b"]), (["a"], []))
        (self.traces / "c").mkdir()
        (self.traces / "c" / "x.trace.json").write_text("x")
        self.assertEqual(self.plan("ok", ["a", "b"]), (["a", "b"], []))

    def test_running_file_recaptures_what_it_left_unsigned(self):
        self.assertEqual(self.plan(run_points.RUNNING, ["a"]), (["a"], ["b"]))
        (self.traces / "c_r2").mkdir()
        (self.traces / "c_r2" / "x.trace.json").write_text("x")
        self.assertEqual(self.plan(run_points.RUNNING, ["a"]), (["a"], ["b", "c"]))
        (self.traces / "c").mkdir()        # an empty directory is no evidence
        (self.traces / "c_r2" / "x.trace.json").unlink()
        self.assertEqual(self.plan(run_points.RUNNING, ["a", "b"]), (["a", "b"], []))

    def test_a_pending_recapture_stays_pending_until_it_has_a_census(self):
        # a finished stop after a kill: b's partial census, a record of a failed attempt
        doc = {"status": "server down", "recaptured_after_kill": ["b"],
               "points": {"a": {"measurement_signature": "s-a"},
                          "b": {"measurement_signature": "s-b", "attempts": []}}}
        args = (self.points, doc, str(self.census), str(self.traces))
        self.assertEqual(run_points.resume_plan(*args), (["a"], ["b"]))
        self.assertEqual(run_points.resume_plan(*args, ["b"]), (["a"], ["b"]))
        doc["points"]["b"]["census"] = {"rc": 1, "out": None}
        self.assertEqual(run_points.resume_plan(*args), (["a"], ["b"]))
        doc["points"]["b"]["census"] = {"rc": 0, "out": str(self.census / "b.json")}
        self.assertEqual(run_points.resume_plan(*args), (["a", "b"], []))

    def test_running_file_still_refuses_a_different_signature(self):
        doc = {"status": run_points.RUNNING,
               "points": {"a": {"measurement_signature": "other"}}}
        with self.assertRaisesRegex(SystemExit, "incompatible or unsigned shared census for a"):
            run_points.resume_plan(self.points, doc, str(self.census), str(self.traces))

    def test_finished_pools_still_validate(self):
        found = 0
        for pool in REAL_POOLS:
            run = os.path.join(pool, "profile_run.json")
            if not os.path.exists(run):
                continue
            found += 1
            with open(run) as fh:
                doc = json.load(fh)
            with open(os.path.join(pool, "sampling_points.json")) as fh:
                pts = run_points.ordered(json.load(fh)["points"], None)
            resumed, recapture = run_points.resume_plan(
                pts, doc, os.path.join(pool, "evidence", "census"),
                os.path.join(pool, "probe", "traces"))
            self.assertEqual(recapture, [], pool)
            self.assertEqual(sorted(resumed), sorted(
                p["point_id"] for p in pts
                if os.path.exists(os.path.join(pool, "evidence", "census",
                                               p["point_id"] + ".json"))), pool)
            self.assertNotEqual(doc["status"], run_points.RUNNING)
        if not found:
            self.skipTest("no finished capture pool on this machine")


if __name__ == "__main__":
    unittest.main()
