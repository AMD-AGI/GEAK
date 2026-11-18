"""Eager evidence tag accepted by vLLM, loud eager evidence failure, and the
default-off census / boundary environment knobs of run_points.py."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_profile_fast_path import (FAST, RANKS, SCRIPTS, FakeServer,  # noqa: E402
                                    observed, points_plan)

import build_task_plan  # noqa: E402
import probe_server  # noqa: E402
import run_points  # noqa: E402
import serving_probe  # noqa: E402


def vllm_validate_profile_prefix(prefix):
    """Verbatim logic of vllm/config/profiler.py validate_profile_prefix (image vLLM)."""
    if prefix is None:
        return None
    if (
        not 1 <= len(prefix) <= 128
        or not prefix.isascii()
        or not prefix[0].isalnum()
        or any(not (char.isalnum() or char in "._-") for char in prefix[1:])
    ):
        raise ValueError("profile_prefix must be ...")
    return prefix


class RejectingServer(FakeServer):
    """A server whose /start_profile answers 400, as vLLM does for a bad prefix."""

    def __init__(self, **kw):
        super().__init__(**kw)
        handler = self.httpd.RequestHandlerClass
        orig = handler.do_POST

        def do_POST(h):
            if h.path == "/start_profile":
                ln = int(h.headers.get("Content-Length") or 0)
                h.rfile.read(ln)
                h._send('{"error": "profile_prefix must be 1-128 ASCII characters"}', code=400)
                return
            orig(h)
        handler.do_POST = do_POST


class EagerTagTest(unittest.TestCase):
    PIDS = ["mc_6d36006e54e2dd61ad7f", "de_1", "mc_4c308a553a497b555310"]

    def test_eager_tag_is_a_valid_vllm_prefix(self):
        for pid in self.PIDS:
            tag = run_points.eager_tag(pid)
            self.assertEqual(vllm_validate_profile_prefix(tag), tag)
            self.assertIsNone(serving_probe.vllm_prefix_problem(tag))
            self.assertNotEqual(tag, pid)
            self.assertTrue(run_points.is_eager_shapes_file(tag + ".json"))
            self.assertFalse(run_points.is_eager_shapes_file(pid + ".json"))
            self.assertEqual(build_task_plan.base_point_id(tag), pid)

    def test_the_old_tag_is_refused_by_vllm_and_by_the_local_check(self):
        old = "mc_6d36006e54e2dd61ad7f@eager"
        with self.assertRaises(ValueError):
            vllm_validate_profile_prefix(old)
        self.assertIn("'@'", serving_probe.vllm_prefix_problem(old))
        # read-side compatibility with files written under the old tag
        self.assertTrue(run_points.is_eager_shapes_file(old + ".json"))
        self.assertEqual(build_task_plan.base_point_id(old), "mc_6d36006e54e2dd61ad7f")

    def test_local_check_matches_vllm_on_edge_cases(self):
        for p in ["a", "a" * 128, "a" * 129, "", "_x", ".x", "x y", "x/y", "x.y_z-1",
                  "é", "1@2", "mc_1_r2_123456"]:
            try:
                vllm_validate_profile_prefix(p)
                ok = True
            except ValueError:
                ok = False
            self.assertEqual(serving_probe.vllm_prefix_problem(p) is None, ok, p)

    def test_redo_and_recapture_tags_stay_valid(self):
        for tag in ["mc_6d36006e54e2dd61ad7f_r2", "mc_6d36006e54e2dd61ad7f_r2_123456",
                    "mc_6d36006e54e2dd61ad7f_123456"]:
            self.assertIsNone(serving_probe.vllm_prefix_problem(tag))

    def test_evidence_requested_reads_new_and_legacy_names(self):
        with tempfile.TemporaryDirectory() as d:
            ev = Path(d)
            (ev / "evidence" / "shapes").mkdir(parents=True)
            (ev / "evidence" / "eager_evidence_request.json").write_text(
                json.dumps({"point": "de_1", "kernels": ["k1", "k2"]}))
            cmd = ["run_points.py", "--eval-dir", str(ev)]
            self.assertTrue(probe_server.evidence_requested(cmd))
            for name in ("de_1.eager.json", "de_1@eager.json"):
                f = ev / "evidence" / "shapes" / name
                f.write_text(json.dumps({"requested": ["k1", "k2"]}))
                self.assertFalse(probe_server.evidence_requested(cmd), name)
                f.write_text(json.dumps({"requested": ["k1"]}))
                self.assertTrue(probe_server.evidence_requested(cmd), name)
                f.unlink()

    def test_no_eager_restart_when_the_capture_did_not_finish(self):
        with tempfile.TemporaryDirectory() as d:
            ev = Path(d)
            (ev / "evidence").mkdir(parents=True)
            (ev / "evidence" / "eager_evidence_request.json").write_text(
                json.dumps({"point": "de_1", "kernels": ["k1"]}))
            cmd = ["run_points.py", "--eval-dir", str(ev)]
            for status, want in (("server down", False), (run_points.MEMORY_RESTART, False),
                                 ("ok", True), ("incomplete", True)):
                (ev / "profile_run.json").write_text(json.dumps({"status": status}))
                self.assertEqual(probe_server.evidence_requested(cmd), want, status)


class ServingProbePrefixTest(unittest.TestCase):
    def test_vllm_probe_with_an_invalid_prefix_fails_before_any_request(self):
        server = FakeServer()
        self.addCleanup(server.close)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            points_plan(d / "points.json")
            td = d / "traces" / "de_1@eager"
            td.mkdir(parents=True)
            p = subprocess.run(
                [sys.executable, str(SCRIPTS / "serving_probe.py"), "--points",
                 str(d / "points.json"), "--point-id", "de_1", "--base-url", server.url,
                 "--trace-dir", str(td), "--out", str(d / "out.json"),
                 "--framework", "vllm", "--served-model", "m"],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
            res = json.loads((d / "out.json").read_text())
            self.assertEqual(res["status"], "failed")
            self.assertIn("not a valid vLLM profile_prefix", res["verify_reason"])
            self.assertEqual(server.requests, [])
            self.assertEqual(server.profile_bodies, [])


class _RunPoints(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="knobs ")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        points_plan(self.root / "points.json")
        observed(self.root / "observed.json")
        self.ev = self.root / "eval"

    def run_points(self, server, *extra, env=None):
        e = {k: v for k, v in os.environ.items()
             if k not in ("GEAK_CENSUS_SERIAL", "GEAK_CENSUS_WORKERS",
                          "GEAK_BOUNDARY_RESTART_GB", "GEAK_MEMINFO")}
        e.update(env or {})
        return subprocess.run(
            [sys.executable, str(SCRIPTS / "run_points.py"), "--points",
             str(self.root / "points.json"), "--base-url", server.url,
             "--trace-root", str(self.root / "traces"), "--expect-files", str(RANKS),
             "--eval-dir", str(self.ev), *FAST, *map(str, extra)],
            capture_output=True, text=True, timeout=300, env=e)

    def run_doc(self):
        return json.loads((self.ev / "profile_run.json").read_text())

    def log_cmds(self):
        """(script, point id) per command, in completion order (run() logs on exit)."""
        out = []
        for line in (self.ev / "run_points.log").read_text().splitlines():
            if "] $ " not in line:
                continue
            argv = line.split("] $ ", 1)[1].split()
            script = os.path.basename(argv[1]) if len(argv) > 1 else ""
            pid = argv[argv.index("--point-id") + 1] if "--point-id" in argv else None
            out.append((script, pid, argv))
        return out

    def meminfo(self, gb):
        f = self.root / "meminfo"
        f.write_text("MemTotal: 3000000000 kB\nMemAvailable: %d kB\n" % int(gb * 1048576))
        return str(f)


class EagerEvidenceFailureTest(_RunPoints):
    def test_a_failed_eager_window_is_not_ok(self):
        graph = FakeServer(replay=True)
        self.addCleanup(graph.close)
        p = self.run_points(graph, "--observed", self.root / "observed.json")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertTrue((self.ev / "evidence" / "eager_evidence_request.json").exists())
        eager = RejectingServer()
        self.addCleanup(eager.close)
        p = self.run_points(eager, "--observed", self.root / "observed.json",
                            "--evidence-capture")
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
        self.assertIn("EAGER EVIDENCE FAILED", p.stdout)
        doc = self.run_doc()
        self.assertEqual(doc["status"], "eager evidence failed")
        self.assertEqual(doc["eager_evidence"]["status"], "failed")
        self.assertEqual(doc["eager_evidence"]["tag"], "de_1.eager")
        self.assertIn("HTTP Error 400", doc["eager_evidence"]["reason"])
        self.assertTrue(any("EAGER EVIDENCE FAILED" in n for n in doc["notes"]))
        self.assertTrue(doc["eager_evidence_needed"])
        probe = json.loads((self.ev / "evidence" / "probe" / "de_1.eager.json").read_text())
        self.assertIn("HTTP Error 400", probe["verify_reason"])
        self.assertFalse((self.ev / "evidence" / "shapes" / "de_1.eager.json").exists())

    def test_a_successful_eager_window_is_ok_and_named_with_the_new_tag(self):
        graph = FakeServer(replay=True)
        self.addCleanup(graph.close)
        self.assertEqual(self.run_points(graph, "--observed", self.root / "observed.json")
                         .returncode, 0)
        eager = FakeServer()
        self.addCleanup(eager.close)
        p = self.run_points(eager, "--observed", self.root / "observed.json",
                            "--evidence-capture")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        doc = self.run_doc()
        self.assertEqual(doc["status"], "ok")
        self.assertEqual(doc["eager_evidence"]["status"], "ok")
        self.assertNotIn("notes", doc)
        self.assertNotIn("eager_evidence_needed", doc)
        self.assertTrue((self.root / "traces" / "de_1.eager").is_dir())
        self.assertTrue((self.ev / "evidence" / "census_eager" / "de_1.json").exists())
        self.assertTrue((self.ev / "evidence" / "shapes" / "de_1.eager.json").exists())


class CensusKnobTest(_RunPoints):
    def test_defaults_are_recorded_and_unchanged(self):
        server = FakeServer()
        self.addCleanup(server.close)
        p = self.run_points(server, "--decode-eager", "--no-evidence")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        doc = self.run_doc()
        self.assertEqual((doc["census_serial"], doc["census_workers"],
                          doc["boundary_restart_gb"]), (False, None, None))
        census = [a for s, _p, a in self.log_cmds() if s == "point_census.py"]
        self.assertEqual(len(census), 4)
        self.assertFalse(any("--workers" in a for a in census))
        self.assertFalse(any("census_serial_wait_s" in r for r in doc["points"].values()))

    def test_serial_census_finishes_before_the_next_window(self):
        server = FakeServer()
        self.addCleanup(server.close)
        p = self.run_points(server, "--decode-eager", "--no-evidence",
                            env={"GEAK_CENSUS_SERIAL": "1", "GEAK_CENSUS_WORKERS": "2"})
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        doc = self.run_doc()
        self.assertEqual((doc["census_serial"], doc["census_workers"]), (True, 2))
        seq = [(s, pid) for s, pid, _a in self.log_cmds()
               if s in ("serving_probe.py", "point_census.py")]
        order = ["pr_64", "pr_256", "de_1", "de_2"]
        want = [x for pid in order for x in (("serving_probe.py", pid), ("point_census.py", pid))]
        self.assertEqual(seq, want)
        census = [a for s, _p, a in self.log_cmds() if s == "point_census.py"]
        self.assertTrue(all(a[a.index("--workers") + 1] == "2" for a in census))
        self.assertTrue(all("census_serial_wait_s" in doc["points"][p] for p in order))

    def test_invalid_knob_values_are_refused(self):
        server = FakeServer()
        self.addCleanup(server.close)
        for env in ({"GEAK_CENSUS_WORKERS": "0"}, {"GEAK_CENSUS_WORKERS": "x"},
                    {"GEAK_CENSUS_SERIAL": "yes"}, {"GEAK_BOUNDARY_RESTART_GB": "-1"}):
            p = self.run_points(server, "--no-evidence", env=env)
            self.assertNotEqual(p.returncode, 0, env)
            self.assertIn("run_points: GEAK_", p.stdout + p.stderr)
            self.assertEqual(server.profile_bodies, [])

    def test_env_knobs_parse(self):
        self.assertEqual(run_points.env_knobs({}), {"census_serial": False,
                         "census_workers": None, "boundary_restart_gb": None})
        self.assertEqual(run_points.env_knobs({"GEAK_CENSUS_SERIAL": "0"})["census_serial"],
                         False)
        self.assertEqual(run_points.env_knobs({"GEAK_BOUNDARY_RESTART_GB": "650"})
                         ["boundary_restart_gb"], 650.0)


class BoundaryRestartTest(_RunPoints):
    def test_low_memory_at_a_boundary_stops_and_a_rerun_resumes_without_redoing(self):
        server = FakeServer()
        self.addCleanup(server.close)
        env = {"GEAK_CENSUS_SERIAL": "1", "GEAK_BOUNDARY_RESTART_GB": "650",
               "GEAK_MEMINFO": self.meminfo(600)}
        p = self.run_points(server, "--decode-eager", "--no-evidence", env=env)
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
        doc = self.run_doc()
        self.assertEqual(doc["status"], run_points.MEMORY_RESTART)
        self.assertEqual(doc["boundary_stop"]["before_point"], "pr_256")
        self.assertEqual(doc["boundary_stop"]["mem_available_gb"], 600.0)
        self.assertEqual(list(doc["points"]), ["pr_64"])
        self.assertEqual(len(server.profile_bodies), 1)
        self.assertTrue((self.ev / "evidence" / "census" / "pr_64.json").exists())
        # the next server: memory is back; the same command resumes after pr_64
        fresh = FakeServer()
        self.addCleanup(fresh.close)
        env["GEAK_MEMINFO"] = self.meminfo(800)
        p = self.run_points(fresh, "--decode-eager", "--no-evidence", env=env)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        doc = self.run_doc()
        self.assertEqual(doc["status"], "ok")
        self.assertTrue(doc["points"]["pr_64"]["resumed"])
        self.assertEqual(len(fresh.profile_bodies), 3)
        probed = [pid for s, pid, _a in self.log_cmds() if s == "serving_probe.py"]
        self.assertEqual(probed.count("pr_64"), 1)

    def test_a_stop_keeps_the_records_of_later_points_censused_earlier(self):
        # Signed plan; an earlier invocation censused every point, then pr_256 and de_1
        # lost theirs (failed attempts). Order is pr_64, pr_256, de_1, de_2. The rerun stops at
        # the boundary before de_1, so it never visits de_2: de_2's signed record must
        # survive in profile_run.json, or the next rerun refuses de_2's census as unsigned.
        plan = json.loads((self.root / "points.json").read_text())
        for p in plan["points"]:
            p["measurement_signature"] = "sig-" + p["point_id"]
        (self.root / "points.json").write_text(json.dumps(plan))
        server = FakeServer()
        self.addCleanup(server.close)
        p = self.run_points(server, "--decode-eager", "--no-evidence")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        census = self.ev / "evidence" / "census"
        for pid in ("pr_256", "de_1"):
            (census / (pid + ".json")).unlink()
            for t in (self.root / "traces").glob(pid + "*"):
                for f in t.iterdir():
                    f.unlink()
                t.rmdir()
        env = {"GEAK_CENSUS_SERIAL": "1", "GEAK_BOUNDARY_RESTART_GB": "650",
               "GEAK_MEMINFO": self.meminfo(600)}
        second = FakeServer()
        self.addCleanup(second.close)
        p = self.run_points(second, "--decode-eager", "--no-evidence", env=env)
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
        doc = self.run_doc()
        self.assertEqual(doc["status"], run_points.MEMORY_RESTART)
        self.assertEqual(doc["boundary_stop"]["before_point"], "de_1")
        self.assertEqual(doc["points"]["de_2"]["measurement_signature"], "sig-de_2")
        self.assertTrue(doc["points"]["de_2"]["resumed"])
        self.assertEqual(doc["points_without_census"], ["de_1"])
        third = FakeServer()
        self.addCleanup(third.close)
        env["GEAK_MEMINFO"] = self.meminfo(800)
        p = self.run_points(third, "--decode-eager", "--no-evidence", env=env)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertEqual(self.run_doc()["status"], "ok")
        self.assertEqual(len(third.profile_bodies), 1)      # de_1 only

    def test_probe_server_reads_the_run_status(self):
        cmd = ["run_points.py", "--eval-dir", str(self.ev)]
        self.assertIsNone(probe_server.run_status(cmd))
        self.ev.mkdir(parents=True)
        (self.ev / "profile_run.json").write_text(json.dumps({"status": run_points.MEMORY_RESTART}))
        self.assertEqual(probe_server.run_status(cmd), run_points.MEMORY_RESTART)
        self.assertIsNone(probe_server.run_status(["run_points.py"]))

    def test_the_first_point_of_an_invocation_is_never_refused(self):
        server = FakeServer()
        self.addCleanup(server.close)
        env = {"GEAK_BOUNDARY_RESTART_GB": "650", "GEAK_MEMINFO": self.meminfo(100)}
        p = self.run_points(server, "--decode-eager", "--no-evidence", env=env)
        self.assertEqual(p.returncode, 1)
        self.assertEqual(len(server.profile_bodies), 1)

    def test_unset_threshold_ignores_memory(self):
        server = FakeServer()
        self.addCleanup(server.close)
        p = self.run_points(server, "--decode-eager", "--no-evidence",
                            env={"GEAK_MEMINFO": self.meminfo(1)})
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertEqual(len(server.profile_bodies), 4)


if __name__ == "__main__":
    unittest.main()
