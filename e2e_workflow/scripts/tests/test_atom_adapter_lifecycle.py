#!/usr/bin/env python3
"""Focused, GPU-free tests for ATOM readiness and distributed profiling."""
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
ATOM_ADAPTER = SCRIPTS_DIR / "adapters" / "atom.sh"
BENCH_E2E = SCRIPTS_DIR / "bench_e2e.sh"
BASH = shutil.which("bash")


@unittest.skipIf(BASH is None, "bash is required")
class AtomAdapterLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="atom_adapter_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _write(self, path: Path, body: str) -> None:
        path.write_text(body, encoding="utf-8")
        path.chmod(0o755)

    def _profile_driver(self, ranks: int) -> subprocess.CompletedProcess:
        profile = self.tmp / "profile"
        profile.mkdir()
        sleep_log = self.tmp / "sleep.log"
        driver = self.tmp / "profile_driver.sh"
        self._write(driver, f'''#!/usr/bin/env bash
set -uo pipefail
source "{ATOM_ADAPTER}"
sleep() {{ printf '%s\n' "$1" >> "{sleep_log}"; /bin/sleep 0.05; }}
curl() {{
  case "$*" in
    *stop_profile*)
      for r in $(seq 0 $((TRACE_RANKS-1))); do
        mkdir -p "$PROFILE_DIR/rank_$r"
        printf '{{}}' | gzip > "$PROFILE_DIR/rank_$r/r$r.pt.trace.json.gz"
      done ;;
  esac
  return 0
}}
adapter_profile_window
''')
        env = dict(os.environ)
        env.update(
            PROFILE_DIR=str(profile), BASE_URL="http://127.0.0.1:8888", TP="2",
            TRACE_RANKS=str(ranks), PROFILE_WINDOW_SEC="17", PROFILE_WINDOW_TIMEOUT="1",
            ATOM_PROFILE_FINALIZE_TIMEOUT="1",
        )
        return subprocess.run([BASH, str(driver)], env=env, capture_output=True,
                              text=True, timeout=10)

    def test_profile_keeps_atom_size_cap_and_waits_for_all_tp_ranks(self):
        proc = self._profile_driver(ranks=2)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        sleeps = (self.tmp / "sleep.log").read_text(encoding="utf-8").splitlines()
        self.assertEqual(sleeps[0], "6")
        manifest = json.loads((self.tmp / "profile" / "atom_profile_manifest.json").read_text())
        self.assertEqual(manifest["expected_ranks"], 2)
        self.assertEqual(manifest["completed_ranks"], ["rank_0", "rank_1"])

    def test_profile_rejects_a_partial_tp_capture(self):
        proc = self._profile_driver(ranks=1)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("finalized 1/2 expected rank traces", proc.stderr)
        manifest = json.loads((self.tmp / "profile" / "atom_profile_manifest.json").read_text())
        self.assertEqual(manifest["status"], "incomplete")
        self.assertFalse(list((self.tmp / "profile").rglob("*.trace.json.gz")))
        self.assertTrue(list((self.tmp / "profile").rglob("*.PARTIAL_missing_ranks")))

    def test_readiness_requires_engine_marker_and_runs_isolated_warmup(self):
        log = self.tmp / "server.log"
        log.write_text("All 1 EngineCores initialized and ready\n", encoding="utf-8")
        called = self.tmp / "warmup.called"
        driver = self.tmp / "ready_driver.sh"
        self._write(driver, f'''#!/usr/bin/env bash
set -uo pipefail
source "{ATOM_ADAPTER}"
adapter_health() {{ return 0; }}
adapter_bench() {{ touch "{called}"; return 0; }}
adapter_prepare_measurement
''')
        env = dict(os.environ)
        env.update(LOG=str(log), CONC="4", GEAK_ISOLATED_REPLICA="1",
                   ATOM_READY_STABLE_SEC="1", ATOM_READY_TIMEOUT_SEC="3",
                   ATOM_READY_POLL_SEC="0.1")
        proc = subprocess.run([BASH, str(driver)], env=env, capture_output=True,
                              text=True, timeout=10)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(called.exists())

    def test_readiness_fails_closed_without_engine_marker(self):
        log = self.tmp / "server.log"
        log.write_text("HTTP server listening\n", encoding="utf-8")
        driver = self.tmp / "ready_fail.sh"
        self._write(driver, f'''#!/usr/bin/env bash
source "{ATOM_ADAPTER}"
adapter_health() {{ return 0; }}
adapter_prepare_measurement
''')
        env = dict(os.environ)
        env.update(LOG=str(log), GEAK_ISOLATED_REPLICA="0")
        proc = subprocess.run([BASH, str(driver)], env=env, capture_output=True,
                              text=True, timeout=10)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("no fully-initialized engine marker", proc.stderr)


@unittest.skipIf(BASH is None, "bash is required")
class BenchPrepareHookTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="bench_prepare_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.events = self.tmp / "events"
        self.adapter = self.tmp / "fake.sh"
        self.adapter.write_text(f'''adapter_default_port() {{ echo 9999; }}
adapter_launch() {{ SERVER_PID=""; }}
adapter_health() {{ return 0; }}
adapter_bench() {{
  echo bench >> "{self.events}"
  printf '{{"output_throughput":1,"median_ttft_ms":1,"median_tpot_ms":1}}\n' >> "$RESULT_JSONL"
}}
adapter_prepare_measurement() {{ echo prepare >> "{self.events}"; [ "${{PREP_FAIL:-0}}" != 1 ]; }}
''', encoding="utf-8")

    def _run(self, fail=False):
        env = dict(os.environ)
        env.update(ADAPTER=str(self.adapter), BACKEND="atom", MODEL="/models/x",
                   REUSE_SERVER="1", OUT_DIR=str(self.tmp / ("out_fail" if fail else "out")),
                   REPEATS="1", ISL="1", OSL="1", CONC="1", NUM_PROMPTS="1",
                   PREP_FAIL="1" if fail else "0")
        return subprocess.run([BASH, str(BENCH_E2E)], env=env, capture_output=True,
                              text=True, timeout=20)

    def test_hook_runs_between_warmup_and_timed_bench(self):
        proc = self._run()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(self.events.read_text().splitlines(), ["bench", "prepare", "bench"])

    def test_hook_failure_prevents_timed_bench(self):
        proc = self._run(fail=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(self.events.read_text().splitlines(), ["bench", "prepare"])
        self.assertIn("refusing to time it", proc.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
