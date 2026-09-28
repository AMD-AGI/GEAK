# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check controller admission through the public native entrypoint."""

import importlib
import json
import os
from types import SimpleNamespace
from unittest.mock import Mock, patch

from interface import run_kernel_native as kernel
from interface.test_native_helpers import UserMessage
from interface.test_run_e2e_dispatch import (
    ResultMessage,
    TaskNotificationMessage,
    TaskStartedMessage,
    _make_fake_anyio,
    _make_fake_sdk,
    _RunE2ECase,
    rx,
)


class QualityEntrypointTests(_RunE2ECase):
    def setUp(self):
        super().setUp()
        self.arguments = {"kernel_path": "/fixture/candidate", "workflow_dir": str(kernel.GEAK_ROOT / "kernel_workflow")}
        self.controller = SimpleNamespace(public_config={"protocol": "fixture-public-config"}, confirm_native_return=Mock())
        self.request = {"scriptPath": str(kernel.GEAK_ROOT / "kernel_workflow" / "kernel_workflow.js"),
                        "args": {**self.arguments, "quality_stop": self.controller.public_config}}
        os.environ.clear()
        self.install_module("anyio", _make_fake_anyio())
        self.install_module("claude_agent_sdk", _make_fake_sdk())

    def test_kernel_binds_host_public_config_without_changing_input(self):
        original = dict(self.arguments)
        with patch.object(kernel, "_invoke_via_sdk", return_value='{"eval_dir":"/fixture/result"}') as invoke:
            result = kernel.invoke_kernel(self.arguments, 10, settings_profile="isolated", quality_stop_controller=self.controller)
        self.assertEqual(result["eval_dir"], "/fixture/result")
        self.assertEqual(self.arguments, original)
        self.assertIs(invoke.call_args.kwargs["quality_stop_controller"], self.controller)
        self.assertEqual(invoke.call_args.kwargs["workflow_request"], self.request)

    def test_json_cannot_enable_or_change_controller_authority(self):
        with patch.object(kernel, "_invoke_via_sdk") as invoke:
            for kwargs, arguments in (
                ({}, {**self.arguments, "quality_stop": {}}),
                ({"quality_stop_controller": self.controller}, self.arguments),
                ({"quality_stop_controller": self.controller, "settings_profile": "isolated"},
                 {**self.arguments, "quality_stop": {"public_key": "model-key"}}),
            ):
                with self.subTest(kwargs=tuple(kwargs)), self.assertRaises(ValueError):
                    kernel.invoke_kernel(arguments, 10, **kwargs)
        invoke.assert_not_called()

    def test_sdk_rejects_cost_controls_nonisolated_and_missing_root(self):
        for environment, kwargs in (
            ({"GEAK_LOCAL_HELPERS": "1"}, {"workflow_request": self.request, "settings_profile": "isolated"}),
            ({"GEAK_SHARED_TOOL_CACHE": "true"}, {"workflow_request": self.request, "settings_profile": "isolated"}),
            ({}, {"workflow_request": self.request}),
            ({}, {"settings_profile": "isolated"}),
        ):
            os.environ.clear()
            os.environ.update(environment)
            with self.subTest(environment=environment, kwargs=tuple(kwargs)), self.assertRaises(ValueError):
                rx._invoke_via_sdk("fixture", 10, quality_stop_controller=self.controller, **kwargs)

    def test_sdk_rejects_unbacked_stopping_args_and_legacy_query(self):
        with self.assertRaisesRegex(ValueError, "trusted host controller"):
            rx._invoke_via_sdk("fixture", 10, workflow_request=self.request, settings_profile="isolated")
        self.install_module("claude_agent_sdk", _make_fake_sdk(with_client=False))
        with self.assertRaisesRegex(ValueError, "isolated native settings"):
            rx._invoke_via_sdk("fixture", 10, workflow_request=self.request, settings_profile="isolated",
                               quality_stop_controller=self.controller)

    def module(self):
        try:
            return importlib.import_module("native_cost_controls.sdk_quality_stop")
        except ModuleNotFoundError:
            return importlib.import_module("interface.native_cost_controls.sdk_quality_stop")

    def messages(self, actual=None):
        output = self.tmp / "quality-native-output.json"
        if actual is not None:
            output.write_text(json.dumps({"result": actual, "workflowProgress": []}))
        return [TaskStartedMessage(task_id="wf", task_type="local_workflow", tool_use_id="root"),
                UserMessage(content=[{"tool_use_id": "root", "is_error": False}], tool_use_result={
                    "taskId": "wf", "taskType": "local_workflow", "status": "async_launched",
                    "scriptPath": self.request["scriptPath"]}),
                ResultMessage(result='{"eval_dir":"/fixture/model-claim","qualifying":true}'),
                TaskNotificationMessage(task_id="wf", status="completed", output_file=str(output), summary="Complete.")]

    def test_sdk_uses_controller_wrapper_and_confirms_bound_native_result(self):
        actual = {"eval_dir": "/fixture/result", "quality_stop": {"qualifying": False}}
        self.install_module("claude_agent_sdk", _make_fake_sdk(self.messages(actual)))
        clients = []
        def wrapper(factory, options, **_kwargs):
            client = factory(options=options)
            clients.append(client)
            return client
        self.controller.confirm_native_return.side_effect = lambda _result: self.assertFalse(clients[0].closed)
        with patch.object(self.module(), "QualityStopSDKClient", side_effect=wrapper) as managed:
            result = rx._invoke_via_sdk("fixture", 10, workflow_request=self.request,
                                       settings_profile="isolated", quality_stop_controller=self.controller)
        self.assertEqual(json.loads(result), actual)
        self.assertEqual(managed.call_args.kwargs["workflow_request"], self.request)
        self.assertIs(managed.call_args.kwargs["controller"], self.controller)
        self.controller.confirm_native_return.assert_called_once_with(actual)
        self.assertTrue(clients[0].closed)

    def test_missing_native_output_and_synchronous_model_claim_cannot_qualify(self):
        for messages in (self.messages(), [ResultMessage(result='{"eval_dir":"/fixture/model","qualifying":true}')]):
            self.install_module("claude_agent_sdk", _make_fake_sdk(messages))
            def wrapper(factory, options, **_kwargs):
                return factory(options=options)
            with patch.object(self.module(), "QualityStopSDKClient", side_effect=wrapper), self.assertRaises(rx.WorkflowParseError):
                rx._invoke_via_sdk("fixture", 10, workflow_request=self.request, settings_profile="isolated",
                                   quality_stop_controller=self.controller)
        self.controller.confirm_native_return.assert_not_called()
