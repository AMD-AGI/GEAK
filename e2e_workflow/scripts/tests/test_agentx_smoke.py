#!/usr/bin/env python3
"""The readiness check must catch the failures that otherwise arrive hours late.

An AgentX prerequisite cannot be noticed by the client until it has already
launched and warmed a server, so a missing one surfaces 20+ minutes into a leg
in the middle of a long run. ``agentx_smoke.sh`` front-loads those checks.

The one that matters most is not a missing file. A stock aiperf has no
``--scenario`` and cannot replay the corpus at all, but pointed at a server it
will still run something and report a throughput -- a "successful" run that
measured a workload nobody asked for. That has to be a blocker, and these tests
pin that it is.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

BASH = shutil.which("bash")
SCRIPTS = Path(__file__).resolve().parents[1]
SMOKE = SCRIPTS / "agentx_smoke.sh"

STOCK_AIPERF_HELP = """\
usage: aiperf profile [-h] --model MODEL --url URL
  --concurrency N
  --request-count N
"""

AGENTX_AIPERF_HELP = """\
usage: aiperf profile [-h] --model MODEL --url URL
  --scenario SCENARIO
  --scenario-dataset DATASET
  --warmup-requests-per-lane N
  --failed-request-threshold F
  --unsafe-override
"""


@unittest.skipIf(BASH is None, "bash is required to run the readiness check")
class AgentXSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="agentx_smoke_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.bin = self.tmp / "bin"
        self.bin.mkdir()

    def _stub_aiperf(self, help_text: str, name: str = "aiperf") -> Path:
        path = self.bin / name
        path.write_text(
            textwrap.dedent(
                f"""\
                #!/usr/bin/env bash
                if [ "${{1:-}}" = "--version" ]; then echo "aiperf 9.9.9-stub"; exit 0; fi
                cat <<'EOF'
                {help_text}
                EOF
                """
            )
        )
        path.chmod(0o755)
        return path

    def _run(self, *, with_path: bool = True, **env) -> subprocess.CompletedProcess:
        run_env = {
            k: v for k, v in os.environ.items()
            if k not in ("AIPERF_BIN", "BASE_URL", "MODEL", "INFERENCEX_PATH")
        }
        # A real PATH is needed for python3/mktemp; prepend the stub dir.
        run_env["PATH"] = (
            f"{self.bin}:{os.environ.get('PATH', '')}" if with_path
            else os.environ.get("PATH", "")
        )
        run_env.update({k: str(v) for k, v in env.items()})
        return subprocess.run(
            [BASH, str(SMOKE)], env=run_env, capture_output=True, text=True
        )

    # ── the trap: a client that "works" but replays nothing ──
    def test_an_aiperf_without_scenario_support_is_a_blocker(self):
        self._stub_aiperf(STOCK_AIPERF_HELP)
        proc = self._run()
        self.assertIn("CANNOT replay the AgentX corpus", proc.stdout)
        self.assertEqual(proc.returncode, 1)

    def test_the_stock_aiperf_message_explains_why_it_is_dangerous(self):
        """An operator seeing "found: /usr/bin/aiperf" needs to know why that is
        not enough, or they will assume the check is being pedantic."""
        self._stub_aiperf(STOCK_AIPERF_HELP)
        proc = self._run()
        self.assertIn("still produce a number", proc.stdout)
        self.assertIn("AIPERF_BIN", proc.stdout)

    def test_an_agentx_capable_aiperf_passes(self):
        self._stub_aiperf(AGENTX_AIPERF_HELP)
        proc = self._run()
        self.assertIn("supports --scenario", proc.stdout)
        self.assertNotIn("CANNOT replay", proc.stdout)

    def test_a_help_text_larger_than_the_pipe_buffer_still_passes(self):
        """The gate used to be ``printf '%s' "$help" | grep -q`` under
        ``set -o pipefail``: grep -q exits at the first match, printf then dies
        of SIGPIPE, and pipefail returns 141 -- so a capable aiperf was reported
        as a blocker, and only on the builds worth approving. It reproduces only
        past the 64KiB pipe buffer, which is why the fixtures above missed it;
        real aiperf help is ~169KiB. --scenario must stay near the top so grep
        quits while printf still has data to write."""
        padding = "\n".join(f"  --unrelated-option-{i} VALUE" for i in range(5000))
        self._stub_aiperf(AGENTX_AIPERF_HELP + padding)
        proc = self._run()
        self.assertGreater(len(padding), 64 * 1024, "padding must exceed the pipe buffer")
        self.assertIn("supports --scenario", proc.stdout)
        self.assertNotIn("CANNOT replay", proc.stdout)

    def test_a_missing_aiperf_names_what_it_looked_for(self):
        proc = self._run(AIPERF_BIN="aiperf-agentx")
        self.assertIn("aiperf-agentx", proc.stdout)
        self.assertEqual(proc.returncode, 1)

    def test_aiperf_bin_is_honoured(self):
        self._stub_aiperf(AGENTX_AIPERF_HELP, name="my-aiperf")
        proc = self._run(AIPERF_BIN="my-aiperf")
        self.assertIn("my-aiperf", proc.stdout)
        self.assertIn("supports --scenario", proc.stdout)

    def test_a_partially_capable_aiperf_warns_without_blocking(self):
        """Missing optional flags may just mean a different build; say so."""
        self._stub_aiperf("usage: aiperf profile\n  --scenario S\n")
        proc = self._run()
        self.assertIn("no --warmup-requests-per-lane", proc.stdout)
        self.assertIn("supports --scenario", proc.stdout)

    # ── the mapper, exercised rather than merely located ──
    def test_the_vendored_mapper_satisfies_the_check(self):
        self._stub_aiperf(AGENTX_AIPERF_HELP)
        proc = self._run()
        self.assertIn("vendored copy", proc.stdout)
        self.assertIn("maps an export", proc.stdout)

    def test_the_mapper_check_confirms_the_noncanonical_stamp(self):
        self._stub_aiperf(AGENTX_AIPERF_HELP)
        proc = self._run()
        self.assertIn("force submission_valid=false", proc.stdout)

    def test_an_inferencex_checkout_is_preferred_and_said_so(self):
        self._stub_aiperf(AGENTX_AIPERF_HELP)
        bench = self.tmp / "ix" / "benchmarks"
        bench.mkdir(parents=True)
        shutil.copy(
            SCRIPTS / "adapters" / "clients" / "map_aiperf.py", bench / "map_aiperf.py"
        )
        proc = self._run(INFERENCEX_PATH=str(self.tmp / "ix"))
        self.assertIn("InferenceX checkout wins", proc.stdout)

    # ── the plumbing ──
    def test_the_profile_window_placement_is_reported_as_steady_state(self):
        self._stub_aiperf(AGENTX_AIPERF_HELP)
        proc = self._run()
        self.assertIn("profile window opens 2700s into the 3600s", proc.stdout)
        self.assertIn("steady state", proc.stdout)

    def test_the_adapter_hooks_are_checked_by_name(self):
        self._stub_aiperf(AGENTX_AIPERF_HELP)
        proc = self._run()
        for fn in ("adapter_bench", "adapter_profile_warmup_s", "_agentx_duration"):
            self.assertIn(f"defines {fn}", proc.stdout)

    def test_a_clean_environment_exits_zero(self):
        self._stub_aiperf(AGENTX_AIPERF_HELP)
        proc = self._run()
        self.assertEqual(proc.returncode, 0, msg=proc.stdout)
        self.assertIn("can run an AgentX trace replay", proc.stdout)

    def test_a_missing_model_path_is_a_blocker_but_an_unset_one_is_not(self):
        self._stub_aiperf(AGENTX_AIPERF_HELP)
        self.assertEqual(self._run().returncode, 0)
        proc = self._run(MODEL=str(self.tmp / "no_such_model"))
        self.assertIn("does not exist", proc.stdout)
        self.assertEqual(proc.returncode, 1)

    # ── the live leg is opt-in and never launches anything ──
    def test_the_live_replay_is_skipped_without_a_base_url(self):
        self._stub_aiperf(AGENTX_AIPERF_HELP)
        proc = self._run()
        self.assertIn("skipped (set BASE_URL", proc.stdout)

    def test_an_unreachable_base_url_is_reported_not_launched(self):
        self._stub_aiperf(AGENTX_AIPERF_HELP)
        # Port 1 is reserved and never listening.
        proc = self._run(BASE_URL="http://127.0.0.1:1")
        self.assertIn("no server at http://127.0.0.1:1", proc.stdout)
        self.assertIn("does not launch one", proc.stdout)
        self.assertEqual(proc.returncode, 1)

    def test_the_verdict_names_the_next_step_on_success(self):
        self._stub_aiperf(AGENTX_AIPERF_HELP)
        proc = self._run()
        self.assertIn("agentx_trace_replay", proc.stdout)


class SmokeScriptShapeTest(unittest.TestCase):
    def test_the_script_is_executable(self):
        self.assertTrue(os.access(SMOKE, os.X_OK))

    def test_it_never_launches_a_server(self):
        """The check must be safe to run on a busy box: no server lifecycle, no
        GPU allocation, no teardown of somebody else's process."""
        text = SMOKE.read_text()
        for forbidden in ("adapter_launch", "vllm serve", "server_teardown.sh ",
                          "pkill", "kill -9"):
            self.assertNotIn(forbidden, text, msg=f"smoke check must not {forbidden}")

    def test_it_documents_its_own_invocation(self):
        self.assertIn("Usage:", SMOKE.read_text())


if __name__ == "__main__":
    unittest.main()
