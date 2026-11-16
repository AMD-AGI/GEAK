#!/usr/bin/env python3
"""serving_watch: a server that dies without its container exiting fails the leg in seconds.

Run:  python3 -I workload/stages/applyback/tests/test_serving_watch.py   (stdlib, CPU only)
"""
import importlib.util
import json
import os
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

SLOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("serving_watch", SLOT.parent / "_shared" / "serving_watch.py")
W = importlib.util.module_from_spec(spec)
spec.loader.exec_module(W)

HEALTHY = """[2026-11-06 14:45:09 TP0 EP0] Decode batch, #running-req: 12, cuda graph: True
[2026-11-06 14:45:10] INFO:     127.0.0.1:52884 - "GET /metrics HTTP/1.1" 200 OK
[aiter] Registering 260 cuda graph addresses
"""
# the three ways the 2026-11-06 legs died, verbatim
DEATHS = [
    ":0:rocdevice.cpp            :3586: 566015967851 us:  Callback: Queue 0x768a65800000 aborting "
    "with error : HSA_STATUS_ERROR_EXCEPTION: An HSAIL operation resulted in a hardware exception. code: 0x1016",
    ":0:rocdevice.cpp            :3582: 582249003559 us:  Callback: Queue 0x761c30200000 Aborting "
    "with error : HSA_STATUS_ERROR_OUT_OF_RESOURCES: The runtime failed to allocate the necessary resources.",
    "Fatal Python error: Aborted",
    "[2026-11-06 10:15:03] Subprocess scheduler_1 (pid=398) crashed with exit code -6. Triggering SIGQUIT for cleanup...",
]


class FatalLineTest(unittest.TestCase):
    def test_a_healthy_log_is_alive(self):
        self.assertIsNone(W.fatal_line(HEALTHY))

    def test_each_observed_death_is_recognized(self):
        for d in DEATHS:
            self.assertIsNotNone(W.fatal_line(HEALTHY + d + "\n"), d)


class WatchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.log = self.tmp / "server.log"
        self.log.write_text(HEALTHY)
        fake = self.tmp / "docker"
        # `docker inspect` says running; `docker logs` prints the current log file
        fake.write_text("#!/bin/sh\ncase \"$1\" in\n  inspect) echo true ;;\n"
                        "  logs) cat '%s' ;;\nesac\n" % self.log)
        fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
        self.env = dict(os.environ, PATH="%s:%s" % (self.tmp, os.environ["PATH"]))

    def run_watch(self, client_seconds, die_after=None):
        client = subprocess.Popen(["sleep", str(client_seconds)], start_new_session=True)
        out = self.tmp / "watch.json"
        w = subprocess.Popen([sys.executable, str(SLOT.parent / "_shared" / "serving_watch.py"), "watch",
                              "--container", "c", "--client-pid", str(client.pid),
                              "--out", str(out), "--interval", "0.5"], env=self.env)
        if die_after is not None:
            time.sleep(die_after)
            with open(self.log, "a") as fh:
                fh.write(DEATHS[1] + "\n")
        t0 = time.time()
        rc = w.wait(timeout=90)
        client.wait(timeout=60)
        return rc, json.loads(out.read_text()), time.time() - t0

    def test_a_healthy_window_passes(self):
        rc, doc, _ = self.run_watch(2)
        self.assertEqual(rc, 0)
        self.assertFalse(doc["dead"])

    def test_a_death_mid_window_stops_the_client_and_fails(self):
        rc, doc, elapsed = self.run_watch(600, die_after=1.5)
        self.assertEqual(rc, 3)
        self.assertTrue(doc["dead"])
        self.assertIn("HSA_STATUS_ERROR_OUT_OF_RESOURCES", doc["line"])
        self.assertLess(elapsed, 60)          # not the client's 600 s

    def test_a_container_that_exited_is_dead(self):
        (self.tmp / "docker").write_text("#!/bin/sh\necho false\n")
        r = subprocess.run([sys.executable, str(SLOT.parent / "_shared" / "serving_watch.py"), "scan", "--container", "c"],
                           env=self.env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 3)
        self.assertEqual(json.loads(r.stdout)["reason"], "container not running")


class FinalizeAdmissionTest(unittest.TestCase):
    """Evaluate's server_death: the leg's watch record, else its saved server log."""
    def setUp(self):
        spec = importlib.util.spec_from_file_location("evaluate_finalize",
                                                      SLOT.parent / "evaluate" / "finalize.py")
        self.F = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.F)
        self.out = Path(tempfile.mkdtemp())

    def test_a_watch_record_that_saw_a_death_refuses_the_leg(self):
        (self.out / "server_watch.json").write_text(json.dumps({"dead": True, "line": DEATHS[0]}))
        self.assertIn("HSA_STATUS_ERROR_EXCEPTION", self.F.server_death(str(self.out)))

    def test_a_leg_from_before_the_watch_is_judged_by_its_server_log(self):
        (self.out / "server.log").write_text(HEALTHY + DEATHS[1] + "\n" + HEALTHY)
        self.assertIn("OUT_OF_RESOURCES", self.F.server_death(str(self.out)))

    def test_a_clean_leg_is_admitted(self):
        (self.out / "server_watch.json").write_text(json.dumps({"dead": False}))
        (self.out / "server.log").write_text(HEALTHY)
        self.assertIsNone(self.F.server_death(str(self.out)))


if __name__ == "__main__":
    unittest.main()
