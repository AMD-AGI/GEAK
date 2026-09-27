# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test the thin kernel entrypoint without starting a native client."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from interface import run_kernel_native as runner


class KernelEntrypointTests(unittest.TestCase):
    def test_kernel_request_uses_the_existing_sdk_runner(self):
        arguments = {"kernel_path": "/synthetic/kernel", "workflow_dir": str(runner.GEAK_ROOT / "kernel_workflow"),
            "budget": 2, "target_language": "triton"}
        result = {"eval_dir": "/synthetic/result", "validation_status": "pass"}
        with patch.object(runner, "_invoke_via_sdk", return_value=json.dumps(result)) as invoke:
            self.assertEqual(runner.invoke_kernel(arguments, 60), result)
        args, kwargs = invoke.call_args
        self.assertEqual(args[1], 60)
        self.assertIn(json.dumps(kwargs["workflow_request"]), args[0])
        self.assertIs(kwargs["workflow_request"]["args"], arguments)
        self.assertEqual(kwargs["workflow_request"]["scriptPath"], str(runner.GEAK_ROOT / "kernel_workflow" / "kernel_workflow.js"))

    def test_invalid_kernel_arguments_do_not_start_the_sdk(self):
        with patch.object(runner, "_invoke_via_sdk") as invoke:
            for arguments in [None, [], {}, {"kernel_path": "/synthetic/kernel"}]:
                with self.subTest(arguments=arguments), self.assertRaises((TypeError, ValueError)):
                    runner.invoke_kernel(arguments, 60)
        invoke.assert_not_called()

    def test_cli_reads_arguments_and_writes_the_complete_result(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "args.json", Path(directory) / "result.json"
            source.write_text('{"kernel_path":"/synthetic/kernel","workflow_dir":"/synthetic/workflow"}')
            expected = {"eval_dir": "/synthetic/result", "final_geomean": 1.0}
            with patch("sys.argv", ["run_kernel_native.py", str(source), str(output), "--timeout", "15"]), \
                    patch.object(runner, "invoke_kernel", return_value=expected) as invoke:
                runner.main()
            self.assertEqual(json.loads(output.read_text()), expected)
            self.assertEqual(invoke.call_args.args[1], 15)

    def test_nonpositive_timeout_stops_before_the_sdk(self):
        with patch("sys.argv", ["run_kernel_native.py", "args.json", "result.json", "--timeout", "0"]), \
                patch.object(runner, "invoke_kernel") as invoke, patch("sys.stderr"), self.assertRaises(SystemExit):
            runner.main()
        invoke.assert_not_called()

    def test_isolated_profile_is_explicit_and_keeps_workflow_arguments(self):
        arguments = {"kernel_path": "/synthetic/kernel", "workflow_dir": "/synthetic/workflow"}
        with patch.object(runner, "_invoke_via_sdk", return_value='{"eval_dir":"/synthetic/result"}') as invoke:
            runner.invoke_kernel(arguments, 10, settings_profile="isolated")
        self.assertEqual(invoke.call_args.kwargs["settings_profile"], "isolated")
        self.assertIs(invoke.call_args.kwargs["workflow_request"]["args"], arguments)
        with self.assertRaises(ValueError):
            runner.invoke_kernel(arguments, 10, settings_profile="unknown")

    def test_explicit_native_directory_keeps_workflow_arguments_unchanged(self):
        arguments = {"kernel_path": "/synthetic/kernel", "workflow_dir": "/synthetic/workflow"}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner, "_invoke_via_sdk", return_value='{"eval_dir":"/synthetic/result"}') as invoke:
            runner.invoke_kernel(arguments, 10, settings_profile="isolated", working_directory=directory)
            self.assertEqual(invoke.call_args.kwargs["native_cwd"], str(Path(directory).resolve()))
            self.assertIs(invoke.call_args.kwargs["workflow_request"]["args"], arguments)
            with self.assertRaises(ValueError):
                runner.invoke_kernel(arguments, 10, working_directory=Path(directory) / "missing")
