#!/usr/bin/env python3
"""Real-process test: teardown reaps a Magpie-launched ATOM group without the supervisor.

Under the Magpie launcher the ATOM server leads its own process group and SERVER_PID is
GEAK's supervisor in another group, so teardown's group kill never reaches ATOM. Reaping it
used to depend on the supervisor's TERM trap, which bash cannot install when TERM arrives
already ignored -- reproduced here -- and which never runs when the supervisor is SIGKILLed.
"""
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
import unittest

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.path.join(SCRIPTS_DIR, "server_teardown.sh")
MAGPIE = os.path.join(SCRIPTS_DIR, "adapters", "launchers", "magpie.sh")
BASH = shutil.which("bash")
SETSID = shutil.which("setsid")


def _group_members(pgid: int) -> list[int]:
    out = subprocess.run(["ps", "-eo", "pid=,pgid="], capture_output=True, text=True).stdout
    return [int(p) for p, g in (line.split() for line in out.splitlines()) if int(g) == pgid]


@unittest.skipIf(BASH is None or SETSID is None, "bash and setsid are required")
class MagpieInnerGroupReapTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="magpie_reap_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        # The "ATOM server": a leader that exits on TERM plus a rank worker that ignores it.
        self.atom = subprocess.Popen(
            [BASH, "-c", '(trap "" TERM; exec sleep 300) & exec sleep 300'],
            start_new_session=True,
        )
        self.addCleanup(self._kill_group, self.atom.pid)
        time.sleep(0.3)
        self.assertEqual(len(_group_members(self.atom.pid)), 2)
        supervisor = re.search(r"_MAGPIE_ATOM_SUPERVISOR='(.*?)\n'\n", open(MAGPIE).read(), re.S).group(1)
        # The supervisor, started the way magpie.sh does -- but with TERM inherited ignored.
        self.supervisor = subprocess.Popen(
            [SETSID, BASH, "-c", supervisor, "atom-magpie-supervisor", str(self.atom.pid), str(self.atom.pid), "1"],
            preexec_fn=lambda: signal.signal(signal.SIGTERM, signal.SIG_IGN),
        )
        self.addCleanup(self._kill_group, self.supervisor.pid)
        time.sleep(0.3)

    @staticmethod
    def _kill_group(pgid: int) -> None:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except OSError:
            pass

    def _teardown(self, inner: str) -> subprocess.CompletedProcess:
        driver = (
            f'source "{LIB}"\n'
            f"server_record_identity {self.supervisor.pid}\n"
            f"export MAGPIE_INNER_SERVER_PID={inner}\n"
            "server_teardown\n"
        )
        return subprocess.run([BASH, "-c", driver], env={**os.environ, "SERVER_STOP_GRACE_S": "2"},
                              capture_output=True, text=True, timeout=60)

    def test_supervisor_alone_leaks_the_atom_group(self):
        """The bug this guards against, reproduced: without the inner pid the group survives."""
        self._teardown("")
        self.supervisor.wait(timeout=10)
        self.assertEqual(len(_group_members(self.atom.pid)), 2)

    def test_teardown_reaps_the_atom_group_directly(self):
        proc = self._teardown(str(self.atom.pid))
        self.assertIn("Reaping Magpie inner server group", proc.stdout)
        self.atom.wait(timeout=10)
        deadline = time.time() + 5
        while _group_members(self.atom.pid) and time.time() < deadline:
            time.sleep(0.1)
        self.assertEqual(_group_members(self.atom.pid), [])

    def test_an_empty_inner_group_sends_nothing(self):
        self._kill_group(self.atom.pid)
        self.atom.wait(timeout=10)
        time.sleep(0.2)
        proc = self._teardown(str(self.atom.pid))
        self.assertNotIn("Reaping Magpie inner server group", proc.stdout)


if __name__ == "__main__":
    unittest.main()
