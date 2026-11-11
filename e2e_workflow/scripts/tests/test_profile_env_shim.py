#!/usr/bin/env python3
"""Unit tests for the profile_env start-up shim and its place on the vLLM server's PYTHONPATH.

Run:  python3 -m unittest discover -s e2e_workflow/scripts/tests -v

A server process spawned after its parent touched the GPU (vLLM's EngineCore) inherits
ROCPROFILER_REGISTER_LIBRARY, and its torch profiler then records no GPU events. The shim
drops the variable at interpreter start-up and then loads the overlay's sitecustomize, which
does not chain, so it must come first on PYTHONPATH.
"""
import json
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

    def test_an_overlay_defining_a_dataclass_loads_through_the_shim(self):
        with open(os.path.join(self.overlay, "sitecustomize.py"), "w", encoding="utf-8") as fh:
            fh.write(
                "from __future__ import annotations\n"
                "import dataclasses, os\n"
                "@dataclasses.dataclass\n"
                "class Hook:\n"
                "    name: str = 'overlay'\n"
                "os.environ['OVERLAY_HOOK'] = Hook().name\n"
            )
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([SHIM_DIR, self.overlay])
        env["ROCPROFILER_REGISTER_LIBRARY"] = "/opt/rocm/lib/librocprofiler-sdk.so.1"
        out = subprocess.run(
            [sys.executable, "-c",
             "import os; print(os.environ.get('ROCPROFILER_REGISTER_LIBRARY', '<unset>'), os.environ.get('OVERLAY_HOOK'))"],
            env=env, cwd=self.tmp, capture_output=True, text=True, timeout=60, check=True,
        )
        self.assertEqual(out.stderr, "")
        self.assertEqual(out.stdout.split(), ["<unset>", "overlay"])

    def test_an_overlay_that_raises_is_reported_and_the_variable_is_still_dropped(self):
        with open(os.path.join(self.overlay, "sitecustomize.py"), "w", encoding="utf-8") as fh:
            fh.write("raise RuntimeError('overlay boom')\n")
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([SHIM_DIR, self.overlay])
        env["ROCPROFILER_REGISTER_LIBRARY"] = "/opt/rocm/lib/librocprofiler-sdk.so.1"
        out = subprocess.run(
            [sys.executable, "-c",
             "import os, sys; print(os.environ.get('ROCPROFILER_REGISTER_LIBRARY', '<unset>'),"
             " '_geak_following_sitecustomize' in sys.modules)"],
            env=env, cwd=self.tmp, capture_output=True, text=True, timeout=60, check=True,
        )
        self.assertIn("overlay boom", out.stderr)
        self.assertEqual(out.stdout.split(), ["<unset>", "False"])

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

    @unittest.skipIf(BASH is None, "bash is required to exercise the shell adapter")
    def test_server_from_task_dir_can_import_stdlib_unittest(self):
        """Exercise the real launcher with the task/unittest.py collision seen on GPU."""
        bin_dir = os.path.join(self.tmp, "bin")
        os.makedirs(bin_dir)
        with open(os.path.join(self.tmp, "unittest.py"), "w", encoding="utf-8") as fh:
            fh.write("raise RuntimeError('task unittest.py shadowed stdlib')\n")
        server = os.path.join(bin_dir, "vllm")
        with open(server, "w", encoding="utf-8") as fh:
            fh.write(f"#!{sys.executable}\nimport json, os, subprocess, sys, unittest.mock\n"
                     "if sys.version_info >= (3, 11):\n"
                     "    subprocess.run([sys.executable, '-c', 'import unittest.mock'], check=True)\n"
                     "print(json.dumps(os.environ.get('PYTHONPATH', '')))\n")
        os.chmod(server, 0o755)
        log = os.path.join(self.tmp, "server.log")
        for profile, overlay, inherited in (
            (SHIM_DIR, self.overlay, None),
            ("", self.overlay, ""),
            (SHIM_DIR, "", None),
            ("", "", None),
            (SHIM_DIR, self.overlay, "/opt/base:/opt/extra"),
        ):
            with self.subTest(profile=profile, overlay=overlay, inherited=inherited):
                env = dict(os.environ)
                env.pop("PYTHONPATH", None)
                env.pop("PYTHONSAFEPATH", None)
                if inherited is not None:
                    env["PYTHONPATH"] = inherited
                env.update(
                    PATH=bin_dir + os.pathsep + env.get("PATH", ""),
                    PROFILE_PYTHONPATH=profile, OVERLAY_PYTHONPATH=overlay, PROFILE_DIR="",
                    MODEL="/models/x", HOST="127.0.0.1", PORT="8000", TP="1", GPU="0",
                    MEM_FRACTION="0.9", GPU_ARCHS="gfx950", SERVER_LAUNCH_PREFIX="",
                    EXTRA_ENV="", EXTRA_SERVER_ARGS="", LOG=log,
                )
                proc = subprocess.run(
                    [BASH, "-c", 'set -uo pipefail\nsource "$1"\nadapter_launch\nwait "$SERVER_PID"',
                     "adapter-test", VLLM_ADAPTER], env=env, cwd=self.tmp,
                    capture_output=True, text=True, timeout=60,
                )
                with open(log, encoding="utf-8") as fh:
                    output = fh.read()
                self.assertEqual(proc.returncode, 0, proc.stderr + output)
                self.assertEqual(json.loads(output), ":".join(
                    part for part in (profile, overlay, inherited) if part))


if __name__ == "__main__":
    unittest.main(verbosity=2)
