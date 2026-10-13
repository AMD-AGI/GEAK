"""The serving-GPU lease: register what you start, and the workflow reaps only that.

Uses real short-lived processes (sleep / sh) -- the whole point of the helper is the
/proc identity it records, so a mocked /proc would test the mock.
"""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fusion_gpu_lease as lease  # noqa: E402


def _run(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = lease.main(argv)
    return code, json.loads(buf.getvalue().strip().splitlines()[-1])


class GpuLeaseTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = os.path.join(self.tmp.name, "gpu_lease")
        self.lock = os.path.join(self.tmp.name, "serving.lock")
        self.procs = []

    def tearDown(self):
        for p in self.procs:
            try:
                p.kill()
                p.wait(timeout=5)
            except Exception:
                pass
        # Anything launched in its own session by `launch` that a test left alive.
        for doc in lease.all_leases(self.dir):
            for rec in doc.get("processes", []):
                if lease.identity_alive(rec):
                    try:
                        os.kill(rec["pid"], 9)
                    except OSError:
                        pass
        self.tmp.cleanup()

    def _acquire(self, lease_id, **kw):
        argv = ["acquire", "--lease-dir", self.dir, "--lease-id", lease_id,
                "--serving-gpu", "0,1", "--serving-lock", self.lock]
        for k, v in kw.items():
            argv += ["--" + k.replace("_", "-"), str(v)]
        return _run(argv)

    def _launch_sleep(self, lease_id, secs=60):
        return _run(["launch", "--lease-dir", self.dir, "--lease-id", lease_id,
                     "--log", os.path.join(self.tmp.name, lease_id + ".log"),
                     "--role", "ab_driver", "--", "sleep", str(secs)])

    def test_acquire_on_an_idle_box_succeeds(self):
        code, out = self._acquire("apply_e01")
        self.assertEqual(code, 0)
        self.assertTrue(out["ok"])
        self.assertEqual(lease.load(self.dir, "apply_e01")["state"], "held")

    def test_launch_registers_its_own_group(self):
        self._acquire("apply_e03")
        code, out = self._launch_sleep("apply_e03")
        self.assertEqual(code, 0)
        self.assertEqual(out["pid"], out["pgid"])     # own session => group is ours
        doc = lease.load(self.dir, "apply_e03")
        self.assertTrue(doc["processes"][0]["own_group"])
        self.assertTrue(doc["processes"][0]["start_ticks"])

    def test_a_live_registered_process_makes_the_next_acquire_busy(self):
        """The e03 -> e04 failure: e04 must see WHO holds the GPUs, not just 'busy'."""
        self._acquire("apply_e03")
        self._launch_sleep("apply_e03")
        code, out = self._acquire("apply_e04")
        self.assertEqual(code, 2)
        self.assertFalse(out["ok"])
        self.assertEqual([h["lease_id"] for h in out["busy_holders"]], ["apply_e03"])
        self.assertIn("gpu_busy", out["reason"])

    def test_reap_frees_the_lease_and_the_next_acquire_succeeds(self):
        self._acquire("apply_e03")
        _, launched = self._launch_sleep("apply_e03")
        code, out = _run(["reap", "--lease-dir", self.dir, "--lease-id", "apply_e03",
                          "--grace-s", "5"])
        self.assertEqual(code, 0, out)
        self.assertEqual(out["alive_count"], 0)
        self.assertTrue(any(s["mode"] == "group" for s in out["signalled"]))
        self.assertEqual(lease.load(self.dir, "apply_e03")["state"], "released")
        rec = lease.load(self.dir, "apply_e03")["processes"][0]
        self.assertEqual(rec["pid"], launched["pid"])
        self.assertFalse(lease.identity_alive(rec))
        code, _ = self._acquire("apply_e04")
        self.assertEqual(code, 0)

    def test_wait_returns_when_the_registered_work_finishes(self):
        self._acquire("apply_e02")
        self._launch_sleep("apply_e02", secs=1)
        code, out = _run(["wait", "--lease-dir", self.dir, "--lease-id", "apply_e02",
                          "--max-s", "20", "--poll-s", "1", "--release"])
        self.assertEqual(code, 0)
        self.assertEqual(out["released"], ["apply_e02"])

    def test_wait_is_bounded_and_reports_survivors(self):
        self._acquire("apply_e05")
        self._launch_sleep("apply_e05", secs=60)
        t0 = time.time()
        code, out = _run(["wait", "--lease-dir", self.dir, "--lease-id", "apply_e05",
                          "--max-s", "1", "--poll-s", "1"])
        self.assertLess(time.time() - t0, 10)
        self.assertEqual(code, 2)
        self.assertEqual(out["alive_count"], 1)

    def test_wait_is_capped_below_the_bash_tool_limit(self):
        p = lease.build_parser().parse_args(
            ["wait", "--lease-dir", self.dir, "--lease-id", "x", "--max-s", "99999"])
        self.assertEqual(min(float(p.max_s), lease.WAIT_CAP_S), lease.WAIT_CAP_S)
        self.assertLess(lease.WAIT_CAP_S, 600)

    def test_descendants_are_adopted_before_anything_is_signalled(self):
        """A server bench_e2e.sh starts is a CHILD of the registered driver."""
        self._acquire("apply_e06")
        _, out = _run(["launch", "--lease-dir", self.dir, "--lease-id", "apply_e06",
                       "--log", os.path.join(self.tmp.name, "e06.log"), "--",
                       "sh", "-c", "sleep 60 & sleep 60; wait"])
        time.sleep(0.5)
        code, st = _run(["status", "--lease-dir", self.dir, "--lease-id", "apply_e06"])
        self.assertEqual(code, 2)
        doc = lease.load(self.dir, "apply_e06")
        adopted = [r for r in doc["processes"] if r.get("adopted_from")]
        self.assertGreaterEqual(len(adopted), 1)
        code, _ = _run(["reap", "--lease-dir", self.dir, "--lease-id", "apply_e06",
                        "--grace-s", "5"])
        self.assertEqual(code, 0)

    def test_unregistered_processes_are_never_touched(self):
        bystander = subprocess.Popen(["sleep", "60"], start_new_session=True)
        self.procs.append(bystander)
        self._acquire("apply_e07")
        self._launch_sleep("apply_e07")
        _run(["reap", "--lease-dir", self.dir, "--lease-id", "apply_e07", "--grace-s", "5"])
        self.assertIsNone(bystander.poll())     # still running

    def test_a_recycled_pid_is_a_stranger(self):
        rec = {"pid": os.getpid(), "start_ticks": "1", "own_group": False}
        self.assertFalse(lease.identity_alive(rec))

    def test_register_refuses_pid1_and_our_own_ancestors(self):
        code, out = _run(["register", "--lease-dir", self.dir, "--lease-id", "x",
                          "--pid", "1"])
        self.assertEqual(code, 1)
        self.assertIn("init", out["reason"])
        code, out = _run(["register", "--lease-dir", self.dir, "--lease-id", "x",
                          "--pid", str(os.getpid())])
        self.assertEqual(code, 1)
        self.assertIn("ancestor", out["reason"])

    def test_register_records_identity(self):
        p = subprocess.Popen(["sleep", "60"], start_new_session=True)
        self.procs.append(p)
        code, out = _run(["register", "--lease-dir", self.dir, "--lease-id", "apply_e08",
                          "--pid", str(p.pid), "--role", "server"])
        self.assertEqual(code, 0)
        self.assertEqual(out["process"]["pgid"], p.pid)
        self.assertEqual(out["process"]["role"], "server")

    def test_release_is_refused_while_registered_work_is_alive(self):
        self._acquire("apply_e09")
        self._launch_sleep("apply_e09")
        code, out = _run(["release", "--lease-dir", self.dir, "--lease-id", "apply_e09"])
        self.assertEqual(code, 2)
        self.assertFalse(out["ok"])

    def test_a_held_lease_with_nothing_alive_is_stale_not_busy(self):
        self._acquire("apply_e01")          # returned without releasing, nothing launched
        code, out = self._acquire("apply_e02")
        self.assertEqual(code, 0)
        self.assertEqual(lease.load(self.dir, "apply_e01")["state"], "released")

    def test_a_foreign_serving_lock_is_busy_but_has_no_reapable_holder(self):
        import fcntl
        fh = open(self.lock, "a+")
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            code, out = self._acquire("apply_e01")
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            fh.close()
        self.assertEqual(code, 2)
        self.assertTrue(out["serving_lock_held"])
        self.assertEqual(out["busy_holders"], [])

    def test_all_scope_waits_for_every_held_lease(self):
        self._acquire("unit_c01")
        self._launch_sleep("unit_c01", secs=1)
        _run(["register", "--lease-dir", self.dir, "--lease-id", "unit_c02",
              "--pid", str(subprocess.Popen(["sleep", "1"]).pid)])
        code, out = _run(["wait", "--lease-dir", self.dir, "--all", "--max-s", "20",
                          "--poll-s", "1", "--release"])
        self.assertEqual(code, 0)
        self.assertEqual(sorted(out["released"]), ["unit_c01", "unit_c02"])


class ReapRecordedTest(unittest.TestCase):
    """reap-recorded stops servers this run recorded, and nothing else."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.eval = os.path.join(self.tmp.name, "eval")
        self.dir = os.path.join(self.eval, "fusion", "gpu_lease")
        self.procs = []

    def tearDown(self):
        for p in self.procs:
            try:
                p.kill()
                p.wait(timeout=5)
            except Exception:
                pass
        self.tmp.cleanup()

    def _server(self, out_dir, record=True, ticks=None):
        p = subprocess.Popen(["sleep", "60"], start_new_session=True)
        self.procs.append(p)
        os.makedirs(out_dir, exist_ok=True)
        if record:
            with open(os.path.join(out_dir, "server_identity.json"), "w") as fh:
                json.dump({"pid": p.pid, "pgid": p.pid,
                           "start_ticks": ticks if ticks is not None else lease.start_ticks(p.pid),
                           "port": "30000", "backend": "sglang", "log": "x"}, fh)
        return p

    def _reap(self):
        return _run(["reap-recorded", "--lease-dir", self.dir, "--eval-dir", self.eval,
                     "--grace-s", "2"])

    def test_a_recorded_leftover_server_is_stopped(self):
        p = self._server(os.path.join(self.eval, "baseline"))
        code, out = self._reap()
        self.assertEqual(code, 0)
        self.assertEqual(out["alive_count"], 0)
        p.wait(timeout=10)
        self.assertIsNotNone(p.returncode)

    def test_an_unrecorded_process_is_never_touched(self):
        p = self._server(os.path.join(self.eval, "baseline"), record=False)
        code, out = self._reap()
        self.assertEqual(code, 0)
        self.assertEqual(out["signalled"], [])
        self.assertIsNone(p.poll())

    def test_a_record_with_a_stale_start_time_is_a_stranger(self):
        p = self._server(os.path.join(self.eval, "baseline"), ticks="1")
        code, out = self._reap()
        self.assertEqual(out["signalled"], [])
        self.assertIsNone(p.poll())

    def test_a_record_without_start_time_is_ignored(self):
        p = self._server(os.path.join(self.eval, "baseline"), ticks="")
        code, out = self._reap()
        self.assertEqual(out["signalled"], [])
        self.assertIsNone(p.poll())


if __name__ == "__main__":
    unittest.main()
