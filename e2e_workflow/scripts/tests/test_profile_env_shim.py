#!/usr/bin/env python3
"""Unit tests for the profile_env start-up shim and its place on the vLLM server's PYTHONPATH.

Run:  python3 -m unittest discover -s e2e_workflow/scripts/tests -v

A server process spawned after its parent touched the GPU (vLLM's EngineCore) inherits
ROCPROFILER_REGISTER_LIBRARY, and its torch profiler then records no GPU events. The shim
drops the variable at interpreter start-up and then loads the overlay's sitecustomize, which
does not chain, so it must come first on PYTHONPATH.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIM_DIR = os.path.join(SCRIPTS_DIR, "adapters", "profile_env")
VLLM_ADAPTER = os.path.join(SCRIPTS_DIR, "adapters", "vllm.sh")

BASH = shutil.which("bash")


class ProfileEnvShimTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="profile_env_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.overlay = os.path.join(self.tmp, "overlay")
        os.makedirs(self.overlay)
        with open(os.path.join(self.overlay, "sitecustomize.py"), "w", encoding="utf-8") as fh:
            fh.write("import os\nos.environ['OVERLAY_RUNS'] = os.environ.get('OVERLAY_RUNS', '') + 'x'\n")

    def test_shim_drops_the_variable_and_loads_the_overlay_once(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([SHIM_DIR, self.overlay])
        env["ROCPROFILER_REGISTER_LIBRARY"] = "/opt/rocm/lib/librocprofiler-sdk.so.1"
        out = subprocess.run(
            [sys.executable, "-c",
             "import os; print(os.environ.get('ROCPROFILER_REGISTER_LIBRARY', '<unset>'), os.environ.get('OVERLAY_RUNS'))"],
            env=env, cwd=self.tmp, capture_output=True, text=True, timeout=60, check=True,
        )
        self.assertEqual(out.stdout.split(), ["<unset>", "x"])

    @unittest.skipIf(BASH is None, "bash is required to exercise the shell adapter")
    def test_vllm_adapter_puts_the_shim_ahead_of_the_overlay(self):
        bin_dir = os.path.join(self.tmp, "bin")
        os.makedirs(bin_dir)
        for name, body in (("python3", "#!/usr/bin/env bash\n"),
                           ("vllm", "#!/usr/bin/env bash\nprintf 'VLLM_PYTHONPATH: %s\\n' \"$PYTHONPATH\"\n")):
            path = os.path.join(bin_dir, name)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(body)
            os.chmod(path, 0o755)
        log = os.path.join(self.tmp, "server.log")
        driver = os.path.join(self.tmp, "driver.sh")
        with open(driver, "w", encoding="utf-8") as fh:
            fh.write(f'set -uo pipefail\nsource "{VLLM_ADAPTER}"\nadapter_launch\nwait "${{SERVER_PID:-}}" 2>/dev/null || true\n')
        env = dict(os.environ)
        env.update(
            PATH=bin_dir + os.pathsep + env.get("PATH", ""), PYTHONPATH="/opt/base",
            PROFILE_PYTHONPATH=SHIM_DIR, OVERLAY_PYTHONPATH=self.overlay, PROFILE_DIR=self.tmp,
            MODEL="/models/x", HOST="127.0.0.1", PORT="8000", TP="1", GPU="0", MEM_FRACTION="0.9",
            GPU_ARCHS="gfx1201", SERVER_LAUNCH_PREFIX="", EXTRA_ENV="", EXTRA_SERVER_ARGS="", LOG=log,
        )
        proc = subprocess.run([BASH, driver], env=env, capture_output=True, text=True, timeout=60, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        with open(log, encoding="utf-8") as fh:
            self.assertIn(f"VLLM_PYTHONPATH: {SHIM_DIR}:{self.overlay}:/opt/base", fh.read())


if __name__ == "__main__":
    unittest.main(verbosity=2)
