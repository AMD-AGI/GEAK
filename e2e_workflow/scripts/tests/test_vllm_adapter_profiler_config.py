#!/usr/bin/env python3
"""Unit tests for adapters/vllm.sh's version-portable torch-profiler wiring (issue #398).

Run:  python3 -m unittest discover -s e2e_workflow/scripts/tests -v
  or: python3 e2e_workflow/scripts/tests/test_vllm_adapter_profiler_config.py

WHY THESE EXIST: the #398 memory bound lives entirely in the exact JSON adapter_launch
hands to `vllm serve --profiler-config`. That JSON is computed by shell from a capability
probe, so it is asserted at the shell layer with fakes rather than trusted:

  * The ProfilerConfig schema is strict (pydantic extra=forbid) and ABORTS the server on
    an unknown key, so adapter_launch may only emit fields the INSTALLED build declares.
    A fake `python3` stands in for the probe and prints the field set we want to model
    (0.26+, 0.19-era, or an import failure), and a fake `vllm` echoes the argv it was
    handed so we can read back the emitted JSON.

No GPU or real vLLM is needed: everything the adapter touches is a fake on PATH.
"""
import os
import shutil
import subprocess
import tempfile
import unittest

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VLLM_ADAPTER = os.path.join(SCRIPTS_DIR, "adapters", "vllm.sh")

BASH = shutil.which("bash")

# A representative 0.26+ field set (superset of the 0.19-era one) and a 0.19-era set
# that lacks the step knobs. Kept here so a rename upstream shows up as a test edit.
FIELDS_026 = (
    "profiler torch_profiler_dir torch_profiler_record_shapes torch_profiler_with_stack "
    "max_iterations delay_iterations ignore_frontend detailed_trace_annotation "
    "capture_torch_profiler warmup_iterations active_iterations wait_iterations"
)
FIELDS_019 = "profiler torch_profiler_dir torch_profiler_record_shapes torch_profiler_with_stack"


@unittest.skipIf(BASH is None, "bash is required to exercise the shell adapter")
class VllmProfilerConfigTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="vllm_prof_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.bin = os.path.join(self.tmp, "bin")
        os.makedirs(self.bin)
        self.profile_dir = os.path.join(self.tmp, "profile")
        os.makedirs(self.profile_dir)
        self.log = os.path.join(self.tmp, "server.log")

        # Fake `python3`: ignore the probe heredoc on stdin, print the modelled field set
        # (empty => import failed => old build). Command substitution strips the newline.
        self._write(os.path.join(self.bin, "python3"),
                    '#!/usr/bin/env bash\nprintf \'%s\\n\' "${PROBE_FIELDS:-}"\n')
        # Fake `vllm`: echo the argv AND the profiler env var it was handed (the <0.19 path
        # passes the dir as VLLM_TORCH_PROFILER_DIR in the env, not on argv). Both land in
        # $LOG via the launch redirect.
        self._write(os.path.join(self.bin, "vllm"),
                    '#!/usr/bin/env bash\n'
                    'printf \'VLLM_ARGV: %s\\n\' "$*"\n'
                    'printf \'VLLM_ENV: VLLM_TORCH_PROFILER_DIR=%s\\n\' "${VLLM_TORCH_PROFILER_DIR:-}"\n')

    def _write(self, path, body):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(body)
        os.chmod(path, 0o755)

    def _run(self, func, probe_fields="", **env):
        """Source the real adapter and invoke one of its functions under fakes."""
        driver = os.path.join(self.tmp, "driver.sh")
        self._write(driver,
                    "#!/usr/bin/env bash\n"
                    "set -uo pipefail\n"
                    f'source "{VLLM_ADAPTER}"\n'
                    f"{func}\n"
                    'wait "${SERVER_PID:-}" 2>/dev/null || true\n')
        run_env = dict(os.environ)
        run_env["PATH"] = self.bin + os.pathsep + run_env.get("PATH", "")
        run_env.update(
            PROBE_FIELDS=probe_fields,
            PROFILE_DIR=self.profile_dir,
            MODEL="/models/x", HOST="127.0.0.1", PORT="8000", TP="1", GPU="0",
            MEM_FRACTION="0.9", GPU_ARCHS="gfx90a",
            SERVER_LAUNCH_PREFIX="", EXTRA_ENV="", EXTRA_SERVER_ARGS="",
            OVERLAY_PYTHONPATH="", LOG=self.log,
            BASE_URL="http://127.0.0.1:8000",
            PROFILE_WINDOW_SEC="0", PROFILE_WINDOW_TIMEOUT="10",
        )
        run_env.update({k: str(v) for k, v in env.items()})
        proc = subprocess.run([BASH, driver], env=run_env,
                              capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        return proc

    def _log(self):
        with open(self.log, encoding="utf-8") as fh:
            return fh.read()

    def _argv(self):
        for line in self._log().splitlines():
            if line.startswith("VLLM_ARGV:"):
                return line
        self.fail("fake vllm was never reached (no VLLM_ARGV line in the log)")

    # ---- config JSON emission -------------------------------------------------

    def test_026_probe_emits_step_bound_and_shape_flags(self):
        self._run("adapter_launch", probe_fields=FIELDS_026, PROFILE_MAX_ITERS="48")
        argv = self._argv()
        self.assertIn("--profiler-config", argv)
        self.assertIn('"torch_profiler_with_stack":false', argv)  # the primary #398 fix
        self.assertIn('"max_iterations":48', argv)                 # step-bounds the buffer
        self.assertIn('"detailed_trace_annotation":true', argv)    # shape param riding along
        self.assertIn('"ignore_frontend":true', argv)
        self.assertIn('"torch_profiler_record_shapes":true', argv)
        # capture stays OFF unless explicitly opted in.
        self.assertNotIn("capture_torch_profiler", argv)
        # never falls back to the env path when the CLI flag is available.
        self.assertNotIn("VLLM_TORCH_PROFILER_DIR", argv)

    def test_026_capture_opt_in(self):
        self._run("adapter_launch", probe_fields=FIELDS_026, PROFILE_CAPTURE_TRACES="1")
        self.assertIn('"capture_torch_profiler":true', self._argv())

    def test_019_probe_omits_026_only_fields(self):
        self._run("adapter_launch", probe_fields=FIELDS_019)
        argv = self._argv()
        self.assertIn("--profiler-config", argv)
        self.assertIn('"torch_profiler_with_stack":false', argv)  # present on 0.19 too
        self.assertNotIn("max_iterations", argv)                   # 0.26-only, correctly gated
        self.assertNotIn("ignore_frontend", argv)
        self.assertNotIn("detailed_trace_annotation", argv)

    def test_old_build_uses_env_fallback(self):
        # Empty probe => import failed => the CLI flag does not exist; must use the env var.
        self._run("adapter_launch", probe_fields="")
        self.assertNotIn("--profiler-config", self._argv())
        self.assertIn(f"VLLM_TORCH_PROFILER_DIR={self.profile_dir}", self._log())


if __name__ == "__main__":
    unittest.main(verbosity=2)
