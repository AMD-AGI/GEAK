#!/usr/bin/env python3
"""Unit tests for server_teardown.sh -- the server kill contract.

Run:  python3 -m unittest discover -s e2e_workflow/scripts/tests -v
  or: python3 e2e_workflow/scripts/tests/test_server_teardown.py

WHY THESE EXIST: the previous teardown resolved the server's process group AT KILL
TIME and group-killed whenever it differed from the benchmark shell's own group. That
guard proves nothing about OWNERSHIP -- a pid that already exited and was recycled
resolves to a stranger's group, and a group TERM aimed at it can reach the caller's
orchestrator or container PID 1 (observed: a capture teardown TERMed the coordinator
running as PID 1, which failed the whole task 72 minutes later).

So the behaviour under test is WHICH TERMINATION MODE IS CHOSEN, asserted from the
signals actually sent rather than from a log line:

  group kill allowed        only when the launch proved pgid == pid
  bench-inherited group     -> pid kill (never signal our own group)
  protected pgid (caller)   -> refused, even when pgid == pid
  group holds pid 1         -> refused
  pgid == 1                 -> refused (`kill -TERM -1` is a BROADCAST, not a group)
  pid start time moved      -> the pid was reused; send NOTHING at all
  pid-kill path             -> still reaps verified descendants (no leaked VRAM)

`ps`, `kill` and `sleep` are replaced by fakes on PATH (the `kill` builtin is disabled
with `enable -n` so the fake is reachable), and every process table is a fixture, so
no test signals a real process.
"""
import os
import shutil
import subprocess
import tempfile
import textwrap
import unittest

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.path.join(SCRIPTS_DIR, "server_teardown.sh")

BASH = shutil.which("bash")

# Fake `ps`: answers only the queries the library makes, from fixture files.
FAKE_PS = """#!/usr/bin/env bash
case "$*" in
  "-o pgid= -p "*)
    _pid="${*##* }"
    cat "$FIX/pgid_$_pid" 2>/dev/null || cat "$FIX/pgid_default" 2>/dev/null || exit 1 ;;
  "-o args= -p "*)
    _pid="${*##* }"; cat "$FIX/args_$_pid" 2>/dev/null || true ;;
  "-eo pid=,pgid=") cat "$FIX/pid_pgid" 2>/dev/null || true ;;
  "-eo pid=,ppid=") cat "$FIX/pid_ppid" 2>/dev/null || true ;;
  *) exit 1 ;;
esac
"""

# Fake `kill`: records every invocation and answers -0 liveness from $FIX/alive. A
# TERM/KILL marks its target(s) dead so the library's grace loop exits immediately.
FAKE_KILL = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$FIX/kill.log"
_target="${2:-}"
if [ "$1" = "-0" ]; then
  grep -qx -- "$_target" "$FIX/alive" 2>/dev/null && exit 0
  exit 1
fi
_reap() {
  grep -vx -- "$1" "$FIX/alive" > "$FIX/alive.next" 2>/dev/null
  mv "$FIX/alive.next" "$FIX/alive"
}
case "$_target" in
  -*)
    for _p in $(awk -v g="${_target#-}" '$2==g{print $1}' "$FIX/pid_pgid" 2>/dev/null); do
      _reap "$_p"
    done ;;
  *) _reap "$_target" ;;
esac
exit 0
"""

FAKE_SLEEP = "#!/usr/bin/env bash\nexit 0\n"


@unittest.skipIf(BASH is None, "bash is required to exercise the shell teardown library")
class ServerTeardownTest(unittest.TestCase):
    def setUp(self):
        self.fix = tempfile.mkdtemp(prefix="teardown_fix_")
        self.addCleanup(shutil.rmtree, self.fix, True)
        bindir = os.path.join(self.fix, "bin")
        os.makedirs(bindir)
        for name, body in (("ps", FAKE_PS), ("kill", FAKE_KILL), ("sleep", FAKE_SLEEP)):
            path = os.path.join(bindir, name)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(body)
            os.chmod(path, 0o755)
        # Defaults: the benchmark shell sits in pgid 4242; the server pid is 500 and
        # leads its own group; only the server is alive.
        self.write("pgid_default", "4242\n")
        self.write("pgid_500", "500\n")
        self.write("pid_pgid", "500 500\n4242 4242\n1 1\n")
        self.write("pid_ppid", "500 4242\n4242 1\n")
        self.write("alive", "500\n")
        self.write("kill.log", "")

    def write(self, name, text):
        with open(os.path.join(self.fix, name), "w", encoding="utf-8") as fh:
            fh.write(text)

    def run_body(self, body, env=None):
        """Source the library with the fakes on PATH and run `body`; return kill args."""
        driver = os.path.join(self.fix, "driver.sh")
        with open(driver, "w", encoding="utf-8") as fh:
            fh.write(
                "#!/usr/bin/env bash\nset -uo pipefail\n"
                'export PATH="$FIX/bin:$PATH"\n'
                "enable -n kill\n"          # else the builtin shadows the fake
                'source "$LIB"\n'
                # Deterministic start times: the real reader needs a live /proc.
                "_start_ticks_of() { printf '%s\\n' \"${FAKE_TICKS:-}\"; }\n"
                + textwrap.dedent(body)
            )
        run_env = dict(os.environ, FIX=self.fix, LIB=LIB, FAKE_TICKS="100")
        run_env.update(env or {})
        proc = subprocess.run(
            [BASH, driver], env=run_env, capture_output=True, text=True, timeout=60
        )
        self.assertEqual(proc.returncode, 0, f"driver failed: {proc.stderr}")
        with open(os.path.join(self.fix, "kill.log"), encoding="utf-8") as fh:
            calls = [line.strip() for line in fh if line.strip()]
        # Liveness probes are noise for mode assertions; keep the real signals.
        return [c for c in calls if not c.startswith("-0 ")], proc.stdout + proc.stderr

    # ---- mode selection -------------------------------------------------------
    def test_own_group_leader_allows_group_kill(self):
        """pgid == pid PROVES the group holds only our descendants -> group kill."""
        signals, out = self.run_body('server_record_identity 500\nserver_teardown\n')
        self.assertIn("-TERM -500", signals)
        self.assertIn("teardown_mode=group", out)

    def test_group_inherited_from_bench_falls_back_to_pid_kill(self):
        """A native launch that shares our group must never be group-killed."""
        self.write("pgid_500", "4242\n")
        self.write("pid_pgid", "500 4242\n4242 4242\n1 1\n")
        signals, out = self.run_body('server_record_identity 500\nserver_teardown\n')
        self.assertIn("-TERM 500", signals)
        self.assertFalse([s for s in signals if " -" in s], f"group-killed: {signals}")
        self.assertIn("group kill REFUSED", out)

    def test_protected_pgid_is_never_group_killed(self):
        """The caller exports its own pgid; a group matching it is refused outright."""
        signals, out = self.run_body(
            'server_record_identity 500\nserver_teardown\n',
            env={"GEAK_PROTECTED_PGIDS": "500 7"},
        )
        self.assertIn("-TERM 500", signals)
        self.assertFalse([s for s in signals if " -" in s], f"group-killed: {signals}")
        self.assertIn("is protected", out)

    def test_group_containing_pid_1_is_refused(self):
        """Defence in depth: a group holding container init is never signalled."""
        self.write("pid_pgid", "500 500\n1 500\n4242 4242\n")
        signals, out = self.run_body('server_record_identity 500\nserver_teardown\n')
        self.assertIn("-TERM 500", signals)
        self.assertFalse([s for s in signals if " -" in s], f"group-killed: {signals}")
        self.assertIn("contains pid 1", out)

    def test_pgid_one_never_becomes_a_broadcast(self):
        """`kill -TERM -1` signals EVERY permitted process; it must be unreachable.

        White-box: pin SERVER_PGID=1 so a future regression that trusts a pgid of 1
        (an empty or inherited lookup) is caught at the guard rather than in prod.
        """
        signals, out = self.run_body(
            'server_record_identity 500\nSERVER_PGID=1\nserver_teardown\n'
        )
        self.assertNotIn("-TERM -1", signals)
        self.assertIn("-TERM 500", signals)
        self.assertIn("would broadcast", out)

    def test_pid_one_is_never_signalled_on_the_pid_path(self):
        """The pid-only fallback must be as unable to reach init as the group path."""
        self.write("pgid_1", "1\n")
        self.write("alive", "1\n")
        signals, out = self.run_body('server_record_identity 1\nserver_teardown\n')
        self.assertEqual(signals, [], f"signalled init: {signals}")
        self.assertIn("refusing to signal pid 1", out)

    def test_launcher_marked_group_unverified_uses_pid_kill(self):
        """An externally-supplied pid (magpie pid file) never earns a group kill."""
        signals, _ = self.run_body(
            'SERVER_GROUP_UNVERIFIED=1\nserver_record_identity 500\nserver_teardown\n'
        )
        self.assertIn("-TERM 500", signals)
        self.assertFalse([s for s in signals if " -" in s], f"group-killed: {signals}")

    # ---- identity gates -------------------------------------------------------
    def test_reused_pid_is_not_signalled_at_all(self):
        """A pid whose start time moved is a DIFFERENT process: send nothing."""
        signals, _ = self.run_body(
            'server_record_identity 500\nFAKE_TICKS=999\nserver_teardown\n'
        )
        self.assertEqual(signals, [], f"signalled a recycled pid: {signals}")

    def test_dead_pid_is_not_signalled(self):
        self.write("alive", "")
        signals, out = self.run_body('server_record_identity 500\nserver_teardown\n')
        self.assertEqual(signals, [])
        self.assertIn("already gone", out)

    def test_no_server_pid_is_a_noop(self):
        """REUSE_SERVER=1 leaves SERVER_PID empty; the EXIT trap must do nothing."""
        signals, _ = self.run_body('SERVER_PID=""\nserver_teardown\n')
        self.assertEqual(signals, [])

    # ---- anti-orphan behaviour on the pid path --------------------------------
    def test_pid_kill_still_reaps_descendants(self):
        """The group kill existed to catch workers outside SERVER_PID; keep that."""
        self.write("pgid_500", "4242\n")
        self.write("pid_pgid", "500 4242\n4242 4242\n1 1\n")
        self.write("pid_ppid", "500 4242\n610 500\n611 610\n4242 1\n999 1\n")
        self.write("alive", "500\n610\n611\n")
        signals, _ = self.run_body('server_record_identity 500\nserver_teardown\n')
        self.assertIn("-TERM 500", signals)
        self.assertIn("-TERM 610", signals)
        self.assertIn("-TERM 611", signals, "grandchild worker leaked")
        self.assertNotIn("-TERM 999", signals, "signalled an unrelated process")
        self.assertNotIn("-TERM 1", signals)

    # ---- /proc parsing --------------------------------------------------------
    def test_start_ticks_parses_comm_with_spaces_and_parens(self):
        """comm is attacker-shaped free text; mis-parsing silently voids the gate."""
        proc_root = os.path.join(self.fix, "proc", "777")
        os.makedirs(proc_root)
        filler = " ".join(str(i) for i in range(1, 19))   # fields 4..21
        with open(os.path.join(proc_root, "stat"), "w", encoding="utf-8") as fh:
            fh.write(f"777 (vllm serve (worker)) S {filler} 555777 0 0 0\n")
        driver = os.path.join(self.fix, "ticks.sh")
        with open(driver, "w", encoding="utf-8") as fh:
            fh.write(
                "#!/usr/bin/env bash\nset -uo pipefail\n"
                'export PATH="$FIX/bin:$PATH"\n'
                'source "$LIB"\n_start_ticks_of 777\n'
            )
        env = dict(
            os.environ, FIX=self.fix, LIB=LIB,
            SERVER_PROC_ROOT=os.path.join(self.fix, "proc"),
        )
        out = subprocess.run(
            [BASH, driver], env=env, capture_output=True, text=True, timeout=60
        )
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "555777")


if __name__ == "__main__":
    unittest.main(verbosity=2)
