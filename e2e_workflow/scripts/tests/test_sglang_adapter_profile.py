"""GPU-free tests for the sglang profile window: it must not return before every rank's trace is written."""
import gzip
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
SGLANG_ADAPTER = SCRIPTS_DIR / "adapters" / "sglang.sh"
BASH = shutil.which("bash")


@unittest.skipIf(BASH is None, "bash is required")
class SglangProfileWindowTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="sglang_profile_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.profile = self.tmp / "profile"
        self.profile.mkdir()

    def _run(self, finishing_ranks: str) -> subprocess.CompletedProcess:
        """Each rank's export starts truncated; the ranks in ``finishing_ranks`` complete one second later."""
        driver = self.tmp / "driver.sh"
        driver.write_text(f'''#!/usr/bin/env bash
set -uo pipefail
source "{SGLANG_ADAPTER}"
sleep() {{ /bin/sleep 0.2; }}
curl() {{
  case "$*" in
    *start_profile*)
      for r in 0 1; do
        printf '{{"traceEvents": [' | gzip | head -c 20 > "$PROFILE_DIR/123.4-TP-$r-EP-$r.trace.json.gz"
      done
      ( /bin/sleep 1
        for r in {finishing_ranks}; do
          printf '{{"traceEvents": [{{"cat": "kernel"}}]}}' | gzip > "$PROFILE_DIR/.w$r"
          mv "$PROFILE_DIR/.w$r" "$PROFILE_DIR/123.4-TP-$r-EP-$r.trace.json.gz"
        done ) &
      ;;
  esac
  return 0
}}
adapter_profile_window
rc=$?
for f in "$PROFILE_DIR"/*.trace.json.gz; do [ -e "$f" ] && {{ gzip -t "$f" 2>/dev/null && echo "VALID $f" || echo "TRUNCATED $f"; }}; done
exit $rc
''', encoding="utf-8")
        env = {
            "PATH": "/usr/bin:/bin",
            "PROFILE_DIR": str(self.profile),
            "BASE_URL": "http://127.0.0.1:8888",
            "PROFILE_NUM_STEPS": "4",
            "TP": "2",
            "PROFILE_WINDOW_TIMEOUT": "2",
            "SGLANG_PROFILE_FINALIZE_TIMEOUT": "8",
        }
        return subprocess.run([BASH, str(driver)], env=env, capture_output=True, text=True, timeout=30, check=False)

    def test_returns_only_after_every_rank_trace_is_a_complete_gzip(self):
        proc = self._run("0 1")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertNotIn("TRUNCATED", proc.stdout)
        self.assertEqual(proc.stdout.count("VALID"), 2, proc.stdout)
        for trace in self.profile.glob("*.trace.json.gz"):
            self.assertIn(b"kernel", gzip.decompress(trace.read_bytes()))

    def test_refuses_a_capture_where_one_rank_never_finishes(self):
        proc = self._run("0")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("finalized 1/2 expected rank traces", proc.stderr)
        self.assertFalse(list(self.profile.glob("*.trace.json.gz")))
        self.assertEqual(len(list(self.profile.glob("*.PARTIAL_missing_ranks"))), 2)


if __name__ == "__main__":
    unittest.main()
