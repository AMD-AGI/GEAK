#!/usr/bin/env python3
"""Behavioral tests for the fail-soft rocprof-compute installer."""

import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTALLER = os.path.join(ROOT, "scripts", "install_rocprof_compute.sh")


def _write_executable(path, body):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("#!/bin/sh\n")
        handle.write(body)
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)


class InstallerTests(unittest.TestCase):
    def _environment(self, directory):
        environment = os.environ.copy()
        environment.update(
            {
                "GEAK_ROOFLINE_COMPUTE_PATH": os.path.join(
                    directory, "rocprof-compute"
                ),
                "GEAK_ROOFLINE_APT_BIN": os.path.join(directory, "fake-apt"),
                "GEAK_ROOFLINE_PYTHON": os.path.join(directory, "fake-python"),
                "GEAK_ROOFLINE_SUDO_BIN": os.path.join(directory, "missing-sudo"),
                "APT_LOG": os.path.join(directory, "apt.log"),
                "TOOL_TARGET": os.path.join(directory, "rocprof-compute"),
                "REAL_PYTHON": sys.executable,
            }
        )
        _write_executable(
            environment["GEAK_ROOFLINE_PYTHON"],
            'if [ "$1" = "-c" ]; then exit 1; fi\n'
            'exec "$REAL_PYTHON" "$@"\n',
        )
        return environment

    def _run(self, environment, *arguments):
        completed = subprocess.run(
            ["bash", INSTALLER] + list(arguments),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        return completed, result

    def test_existing_tool_skips_apt(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = self._environment(directory)
            _write_executable(
                environment["GEAK_ROOFLINE_COMPUTE_PATH"],
                'echo "rocprofiler-compute version: 3.4.0"\n',
            )
            _write_executable(
                environment["GEAK_ROOFLINE_APT_BIN"],
                'echo "$*" >> "$APT_LOG"\nexit 99\n',
            )
            completed, result = self._run(environment, "--check", "--required")

            self.assertEqual(completed.returncode, 0)
            self.assertEqual(result["status"], "present")
            self.assertEqual(
                result["path"], environment["GEAK_ROOFLINE_COMPUTE_PATH"]
            )
            self.assertFalse(os.path.exists(environment["APT_LOG"]))

    def test_check_mode_never_installs_missing_tool(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = self._environment(directory)
            _write_executable(
                environment["GEAK_ROOFLINE_APT_BIN"],
                'echo "$*" >> "$APT_LOG"\nexit 0\n',
            )
            completed, result = self._run(environment, "--check")

            self.assertEqual(completed.returncode, 0)
            self.assertEqual(result["status"], "missing")
            self.assertEqual(result["reason"], "rocprof_compute_unavailable")
            self.assertFalse(os.path.exists(environment["APT_LOG"]))

    def test_install_mode_installs_package_and_writes_json(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = self._environment(directory)
            _write_executable(
                environment["GEAK_ROOFLINE_APT_BIN"],
                'echo "$*" >> "$APT_LOG"\n'
                'if [ "$1" = "install" ]; then\n'
                '  printf \'#!/bin/sh\\necho "rocprofiler-compute version: 3.4.0"\\n\' > "$TOOL_TARGET"\n'
                '  chmod +x "$TOOL_TARGET"\n'
                "fi\n"
                "exit 0\n",
            )
            json_path = os.path.join(directory, "result", "install.json")
            completed, result = self._run(
                environment,
                "--install",
                "--required",
                "--json-out",
                json_path,
            )

            self.assertEqual(completed.returncode, 0)
            self.assertEqual(result["status"], "installed")
            self.assertTrue(result["installed"])
            with open(json_path, "r", encoding="utf-8") as handle:
                self.assertEqual(json.load(handle), result)
            with open(environment["APT_LOG"], "r", encoding="utf-8") as handle:
                apt_calls = handle.read()
            self.assertIn("update -qq", apt_calls)
            self.assertIn(
                "install -y --no-install-recommends rocprofiler-compute",
                apt_calls,
            )

    def test_install_failure_is_fail_soft_unless_required(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = self._environment(directory)
            _write_executable(
                environment["GEAK_ROOFLINE_APT_BIN"],
                'echo "$*" >> "$APT_LOG"\nexit 7\n',
            )
            soft, soft_result = self._run(environment, "--install")
            required, required_result = self._run(
                environment, "--install", "--required"
            )

            self.assertEqual(soft.returncode, 0)
            self.assertEqual(soft_result["status"], "missing")
            self.assertEqual(soft_result["reason"], "apt_install_failed")
            self.assertNotEqual(required.returncode, 0)
            self.assertEqual(required_result["reason"], "apt_install_failed")

    def test_required_fails_when_pandas_repair_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = self._environment(directory)
            _write_executable(
                environment["GEAK_ROOFLINE_COMPUTE_PATH"],
                'echo "rocprofiler-compute version: 3.4.0"\n',
            )
            _write_executable(
                environment["GEAK_ROOFLINE_PYTHON"],
                'if [ "$1" = "-c" ]; then exit 0; fi\n'
                'if [ "$1" = "-m" ] && [ "$2" = "pip" ] && '
                '[ "$3" = "install" ] && [ "$4" = "--help" ]; then exit 0; fi\n'
                'if [ "$1" = "-m" ] && [ "$2" = "pip" ]; then exit 9; fi\n'
                'exec "$REAL_PYTHON" "$@"\n',
            )

            completed, result = self._run(
                environment, "--install", "--required"
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["reason"], "python_dependency_repair_failed")
            self.assertTrue(result["dependency_repair_failed"])

    def test_required_check_validates_profile_and_analyze_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = self._environment(directory)
            _write_executable(
                environment["GEAK_ROOFLINE_COMPUTE_PATH"],
                'if [ "$1" = "--version" ]; then echo "version 3.4.0"; exit 0; fi\n'
                'if [ "$1" = "profile" ]; then exit 0; fi\n'
                'if [ "$1" = "analyze" ]; then exit 7; fi\n',
            )

            completed, result = self._run(
                environment, "--check", "--required"
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["reason"], "command_health_check_failed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
