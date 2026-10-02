# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the real run_e2e SDK entrypoint with a synthetic loopback provider."""

import http.client
import importlib
import json
import os
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch
from urllib.parse import urlsplit

from interface.test_run_e2e_dispatch import (
    ResultMessage,
    TaskNotificationMessage,
    TaskStartedMessage,
    _make_fake_anyio,
    _make_fake_sdk,
    _RunE2ECase,
    rx,
)
from interface.test_shared_tool_cache import INSERTION, request
from interface.test_system_envelope import synthetic_system_policy


@dataclass
class FakeOptions:
    model: object = None
    allowed_tools: object = None
    permission_mode: object = None
    settings: object = None
    extra_args: object = None
    cwd: object = None
    env: dict = field(default_factory=dict)
    cli_path: object = None
    session_id: object = None
    hooks: dict = field(default_factory=dict)
    session_store: object = None
    session_store_flush: str = "batched"
    setting_sources: list = None


class NativeCostEntryTests(_RunE2ECase):
    def test_opt_in_changes_only_the_supported_request_marker(self):
        calls = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                return
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                calls.append(body)
                result = b'{"usage":{"input_tokens":7,"output_tokens":3}}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(result)))
                self.end_headers()
                self.wfile.write(result)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        endpoint = "http://127.0.0.1:" + str(server.server_port)
        os.environ.clear()
        os.environ["ANTHROPIC_BASE_URL"] = endpoint
        anyio = _make_fake_anyio()
        sdk = _make_fake_sdk([ResultMessage(result="done")])
        sdk.ClaudeAgentOptions = FakeOptions
        inputs = []
        replies = []
        async def query(client, prompt):
            body = request()
            body["messages"].insert(0, {
                "role": "system", "content": [{"type": "text", "text": "Synthetic workflow notification."}]
            })
            body["metadata"] = {"user_id": json.dumps({"session_id": client.options.session_id})}
            raw = json.dumps(body, indent=2).encode()
            inputs.append(raw)
            url = urlsplit(client.options.env.get("ANTHROPIC_BASE_URL", endpoint))
            connection = http.client.HTTPConnection(url.hostname, url.port, timeout=3)
            connection.request("POST", "/v1/messages", raw, {"Content-Type": "application/json"})
            response = connection.getresponse()
            replies.append(response.read())
            connection.close()
        sdk.ClaudeSDKClient.query = query
        self.install_module("anyio", anyio)
        self.install_module("claude_agent_sdk", sdk)
        try:
            self.assertEqual(rx._invoke_via_sdk("synthetic", 10), "done")
            baseline = sdk.state["clients"][-1].options
            os.environ["GEAK_SHARED_TOOL_CACHE"] = "1"
            self.assertEqual(rx._invoke_via_sdk("synthetic", 10), "done")
            treatment = sdk.state["clients"][-1].options
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], inputs[0])
        self.assertEqual(calls[1].replace(INSERTION, b"", 1), inputs[1])
        self.assertEqual(replies[0], replies[1])
        for name in ("model", "allowed_tools", "permission_mode", "settings", "extra_args", "cwd", "cli_path"):
            self.assertEqual(getattr(treatment, name), getattr(baseline, name))
        self.assertTrue(all(client.closed for client in sdk.state["clients"]))

    def test_legacy_options_keep_the_original_query_path(self):
        os.environ.clear()
        os.environ["GEAK_SHARED_TOOL_CACHE"] = "1"
        self.install_module("anyio", _make_fake_anyio())
        sdk = _make_fake_sdk(with_client=False, query_script=[ResultMessage(result="legacy")])
        self.install_module("claude_agent_sdk", sdk)
        self.assertEqual(rx._invoke_via_sdk("synthetic", 10), "legacy")
        self.assertEqual(sdk.state["prompts"], ["synthetic"])

    def test_kernel_return_comes_from_the_bound_native_output(self):
        from interface.test_native_helpers import UserMessage

        os.environ.clear()
        request_value = {"scriptPath": "/synthetic/kernel_workflow.js", "args": {}}
        actual = {"eval_dir": "/synthetic/actual", "final_geomean": 2.0, "nested": {"complete": True}}
        output = self.tmp / "native-output.json"
        output.write_text(json.dumps({"result": actual, "workflowProgress": []}))
        messages = [TaskStartedMessage(task_id="wf", task_type="local_workflow", tool_use_id="root"),
            UserMessage(content=[{"tool_use_id": "root", "is_error": False}], tool_use_result={
                "taskId": "wf", "taskType": "local_workflow", "status": "async_launched",
                "scriptPath": request_value["scriptPath"]}),
            ResultMessage(result='{"eval_dir":"/synthetic/planned","incomplete":true}'),
            TaskNotificationMessage(task_id="wf", status="completed", output_file=str(output), summary="Complete.")]
        self.install_module("anyio", _make_fake_anyio())
        self.install_module("claude_agent_sdk", _make_fake_sdk(messages))
        self.assertEqual(json.loads(rx._invoke_via_sdk("kernel", 10, workflow_request=request_value)), actual)

    def test_planned_model_json_cannot_replace_missing_native_kernel_output(self):
        from interface.test_native_helpers import UserMessage

        os.environ.clear()
        request_value = {"scriptPath": "/synthetic/kernel_workflow.js", "args": {}}
        messages = [TaskStartedMessage(task_id="wf", task_type="local_workflow", tool_use_id="root"),
            UserMessage(content=[{"tool_use_id": "root", "is_error": False}], tool_use_result={
                "taskId": "wf", "taskType": "local_workflow", "status": "async_launched",
                "scriptPath": request_value["scriptPath"]}),
            ResultMessage(result='{"eval_dir":"/synthetic/planned","incomplete":true}'),
            TaskNotificationMessage(task_id="wf", status="completed", output_file=str(self.tmp / "missing"), summary="Complete.")]
        self.install_module("anyio", _make_fake_anyio())
        self.install_module("claude_agent_sdk", _make_fake_sdk(messages))
        with self.assertRaises(rx.WorkflowParseError):
            rx._invoke_via_sdk("kernel", 10, workflow_request=request_value)

    def test_kernel_entry_uses_the_combined_native_client_and_explicit_profile(self):
        from interface.native_cost_controls.sdk_helpers import WorkflowSDKClient
        from interface.test_native_helpers import Contract
        from interface.test_sdk_helpers import Client, Matcher

        calls, clients = [], []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                return
            def do_POST(self):
                calls.append(self.rfile.read(int(self.headers["Content-Length"])))
                result = b'{"usage":{"input_tokens":7,"output_tokens":3}}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(result)))
                self.end_headers()
                self.wfile.write(result)
        @dataclass
        class TaskNotificationMessage:
            task_id: str = "root-task"
            status: str = "completed"
            output_file: str = ""
        class KernelClient(Client):
            async def receive_messages(self):
                async for message in super().receive_messages():
                    yield message
                output = self.journal_root / "native-workflow-output.json"
                output.write_text(json.dumps({"result": {"eval_dir": "/synthetic/result", "native_value": 42},
                    "workflowProgress": [], "summary": "A synthetic native completion."}))
                yield TaskNotificationMessage(output_file=str(output))
                yield ResultMessage(result="kernel-done")
        def factory(*, options):
            client = KernelClient(options=options)
            client.journal_root = self.tmp
            clients.append(client)
            return client
        def managed(client_factory, options, **kwargs):
            options.session_id = "fixture-session"
            return WorkflowSDKClient(client_factory, options, contract_factory=Contract,
                hook_matcher_factory=Matcher, **kwargs)
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        os.environ.clear()
        os.environ.update(ANTHROPIC_BASE_URL="http://127.0.0.1:" + str(server.server_port),
            GEAK_SHARED_TOOL_CACHE="1", GEAK_LOCAL_HELPERS="1")
        sdk = _make_fake_sdk()
        sdk.ClaudeAgentOptions = FakeOptions
        sdk.ClaudeSDKClient = factory
        self.install_module("anyio", _make_fake_anyio())
        self.install_module("claude_agent_sdk", sdk)
        try:
            helper_module = importlib.import_module("native_cost_controls.sdk_helpers")
        except ModuleNotFoundError:
            helper_module = importlib.import_module("interface.native_cost_controls.sdk_helpers")
        try:
            with patch.object(helper_module, "WorkflowSDKClient", side_effect=managed), synthetic_system_policy():
                result = rx._invoke_via_sdk("kernel-fixture", 10, workflow_request=Contract().root_request,
                    settings_profile="isolated", native_cwd=str(self.tmp))
            self.assertEqual(json.loads(result), {"eval_dir": "/synthetic/result", "native_value": 42})
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        self.assertEqual(len(calls), 1)
        self.assertEqual(clients[0].options.setting_sources, [])
        self.assertEqual(clients[0].options.cwd, str(self.tmp))
        self.assertEqual(clients[0].options.permission_mode, "bypassPermissions")
        self.assertTrue(clients[0].closed)


if __name__ == "__main__":
    import unittest
    unittest.main()
