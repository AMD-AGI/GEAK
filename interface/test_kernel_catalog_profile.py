# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test source selection and the exact synthetic Workflow dispatch."""

import asyncio
import http.client
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import nullcontext, redirect_stderr
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from urllib.parse import urlsplit

from interface.native_cost_controls import kernel_catalog_profile as kernel
from interface.native_cost_controls.native_prefix_marker import DECODER, wire


def definition(name):
    return {"name": name, "input_schema": {"type": "object", "properties": {}}}


def body(*, session="session", prompt=kernel.CHILD_PROMPT, tools=None):
    return wire({"model": "synthetic-model", "metadata": {
        "user_id": json.dumps({"session_id": session})},
        "tools": tools or [definition("Read"), definition("StructuredOutput")],
        "messages": [{"role": "user", "content": prompt}]})


class KernelCaptureTests(unittest.TestCase):
    def setUp(self):
        self.capture = kernel.KernelCapture(1, Path("/synthetic/workflow.js"), "session")

    def grant(self):
        return asyncio.run(self.capture.pre_tool({"tool_name": "Workflow", "session_id": "session",
            "tool_input": self.capture.workflow_input}, "toolu_synthetic_catalog_workflow", None))

    def child_headers(self, agent="child"):
        self.capture.validate_headers({"x-claude-code-session-id": "session", "x-claude-code-agent-id": agent})

    def test_only_the_exact_single_workflow_dispatch_receives_a_grant(self):
        self.assertEqual(self.grant()["hookSpecificOutput"]["permissionDecision"], "allow")
        self.assertEqual(self.grant()["hookSpecificOutput"]["permissionDecision"], "deny")
        for name in ("Bash", "Write", "TaskOutput", "StructuredOutput", "Agent"):
            decision = asyncio.run(self.capture.pre_tool({"tool_name": name, "session_id": "session",
                "tool_input": self.capture.workflow_input}, "tool-2", None))
            self.assertEqual(decision["hookSpecificOutput"]["permissionDecision"], "deny")

    def test_changed_dispatch_input_or_session_receives_no_grant(self):
        for session, value in [("another", self.capture.workflow_input), ("session", {"scriptPath": "/changed"})]:
            decision = asyncio.run(self.capture.pre_tool({"tool_name": "Workflow", "session_id": session,
                "tool_input": value}, "tool", None))
            self.assertEqual(decision["hookSpecificOutput"]["permissionDecision"], "deny")
        self.assertEqual(self.capture.permitted, 0)

    def test_unbound_tool_identifier_receives_no_grant(self):
        decision = asyncio.run(self.capture.pre_tool({"tool_name": "Workflow", "session_id": "session",
            "tool_input": self.capture.workflow_input}, "another-tool-id", None))
        self.assertEqual(decision["hookSpecificOutput"]["permissionDecision"], "deny")

    def test_root_returns_only_the_bound_workflow_then_terminal_text(self):
        self.capture.validate_headers({"x-claude-code-session-id": "session"})
        initial = self.capture.message(body(prompt="root"))
        events = [json.loads(line[6:]) for line in initial.splitlines() if line.startswith(b"data: ")]
        delta = next(item["delta"] for item in events if item["type"] == "content_block_delta")
        self.assertEqual(json.loads(delta["partial_json"]), self.capture.workflow_input)
        for _ in range(2):
            self.assertNotIn(b"tool_use", self.capture.message(body(prompt="root")))
        with self.assertRaises(ValueError):
            self.capture.message(body(prompt="root"))
        self.assertIsNone(self.capture.catalog)

    def test_child_requires_session_prompt_and_prior_dispatch(self):
        with self.assertRaises(ValueError):
            self.capture.validate_headers({"x-claude-code-session-id": "wrong"})
        self.child_headers()
        with self.assertRaises(ValueError):
            self.capture.message(body())
        self.grant()
        with self.assertRaises(ValueError):
            self.capture.message(body(prompt="wrong child"))
        with self.assertRaises(ValueError):
            self.capture.message(body(prompt="another request " + kernel.CHILD_PROMPT))

    def test_selected_child_checks_portable_policy_and_preserves_raw_bytes(self):
        self.grant()
        self.child_headers()
        response = self.capture.message(body())
        self.assertNotIn(b"tool_use", response)
        self.assertEqual(DECODER.decode(self.capture.catalog.decode()), [definition("Read")])
        evidence = self.capture.request["portable_policy"]
        self.assertTrue(evidence["applied"])
        self.assertTrue(evidence["only_marker_bytes_changed"])
        self.assertEqual(self.capture.request["role"], "native_workflow_agent")
        self.assertNotIn("child", self.capture.request.values())
        with self.assertRaises(ValueError):
            self.child_headers("second-child")

    def test_metadata_session_mismatch_rejects_the_catalog(self):
        self.grant()
        self.child_headers()
        with self.assertRaisesRegex(ValueError, "session_mismatch"):
            self.capture.message(body(session="other"))
        self.assertIsNone(self.capture.catalog)

    def test_exact_native_date_envelope_is_supported(self):
        self.assertTrue(kernel._prompt_matches([kernel._date_reminder(), kernel.CHILD_PROMPT]))
        self.assertFalse(kernel._prompt_matches([kernel._date_reminder() + "extra", kernel.CHILD_PROMPT]))
        self.assertFalse(kernel._prompt_matches([kernel._date_reminder(), "prefix " + kernel.CHILD_PROMPT]))

    def test_count_alone_cannot_select_the_child_catalog(self):
        self.grant()
        self.child_headers()
        with self.assertRaises(ValueError):
            self.capture.message(body(tools=[definition("Read"), definition("Write")]))

    def test_public_prefix_can_precede_extra_tools_and_the_output_tool(self):
        self.grant()
        self.child_headers()
        self.capture.message(body(tools=[definition("Read"), definition("Write"), definition("StructuredOutput")]))
        self.assertEqual(self.capture.request["tool_count"], 3)
        self.assertEqual(self.capture.request["prefix_count"], 1)

    def test_frozen_profile_requires_the_measured_output_tool_position(self):
        self.capture.require_exact_suffix = True
        self.grant()
        self.child_headers()
        with self.assertRaises(ValueError):
            self.capture.message(body(tools=[definition("Read"), definition("Write"), definition("StructuredOutput")]))


class SourceBindingTests(unittest.TestCase):
    def test_current_public_wrapper_is_copied_without_translation(self):
        source = (kernel.ROOT / "kernel_workflow/kernel_lane.js").read_text()
        wrapper, variant = kernel._extract_wrapper(source)
        self.assertEqual(variant, "agentT")
        self.assertIn(wrapper, source)
        self.assertIn(wrapper, kernel._workflow(source))
        self.assertNotIn("function tlAgent", wrapper)

    def test_frozen_wrapper_retains_its_native_parent_function(self):
        source = "function tlAgent(prompt, o, attempt) {\n  return agent(prompt, o);\n}\n" \
                 "async function agentT(p, o) {\n  return tlAgent(p, o, 1);\n}\n"
        wrapper, variant = kernel._extract_wrapper(source)
        self.assertEqual(variant, "tlAgent_and_agentT")
        self.assertEqual(wrapper + "\n", source)

    def test_external_source_mutation_fails_before_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kernel_lane.js"
            path.write_text("async function agentT(p, o) {\n return agent(p, o);\n}\n")
            options = SimpleNamespace(kernel_source=path, kernel_source_sha256=kernel._file_sha(path), profile="kernel-child")
            before = kernel.bindings(options)
            self.assertEqual(before["kernel_source_sha256"], options.kernel_source_sha256)
            path.write_text(path.read_text() + "// changed\n")
            with self.assertRaises(ValueError):
                kernel.bindings(options)


FROZEN_OPTIONS_SOURCE = '''
TOOLS = ["Workflow", "Read"]
SYSTEM_APPEND = "Synthetic source safety rule."
def construct(config):
    return ClaudeAgentOptions(
        model=config["model"], effort=config["effort"], thinking={"type": "adaptive"},
        allowed_tools=[*TOOLS, "ToolSearch"], permission_mode="bypassPermissions",
        settings=json.dumps({"enableWorkflows": True, "ultracode": True, **config.get("settings", {})}),
        system_prompt={"type": "preset", "preset": "claude_code", "append": SYSTEM_APPEND},
        include_partial_messages=True, cwd="ignored source workspace")
'''


class FrozenOptionsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "options.py"

    def options(self, source=FROZEN_OPTIONS_SOURCE):
        self.source.write_text(source)
        return SimpleNamespace(kernel_options_source=self.source,
            kernel_options_sha256=kernel._file_sha(self.source), model="selected-model", effort="high")

    def test_source_values_preserve_model_effort_filters_and_system_context(self):
        values = kernel._frozen_values(self.options())
        self.assertEqual(values["model"], "selected-model")
        self.assertEqual(values["effort"], "high")
        self.assertEqual(values["thinking"], {"type": "adaptive"})
        self.assertEqual(values["allowed_tools"], ["Workflow", "Read", "ToolSearch"])
        self.assertEqual(values["permission_mode"], "bypassPermissions")
        self.assertEqual(json.loads(values["settings"]), {"enableWorkflows": True, "ultracode": True})
        self.assertEqual(values["system_prompt"]["append"], "Synthetic source safety rule.")
        self.assertTrue(values["include_partial_messages"])
        self.assertNotIn("cwd", values)

    def test_source_pin_and_explicit_native_effort_are_required(self):
        options = self.options()
        options.kernel_options_sha256 = "0" * 64
        with self.assertRaisesRegex(ValueError, "explicit hash"):
            kernel._frozen_values(options)
        options = self.options()
        options.effort = "ultracode"
        with self.assertRaisesRegex(ValueError, "explicit native effort"):
            kernel._frozen_values(options)

    def test_missing_constants_or_ambiguous_constructors_are_rejected(self):
        cases = [
            (FROZEN_OPTIONS_SOURCE.replace('SYSTEM_APPEND = "Synthetic source safety rule."', ""), "constants"),
            (FROZEN_OPTIONS_SOURCE.replace("ClaudeAgentOptions(", "OtherOptions("), "one native options call"),
            (FROZEN_OPTIONS_SOURCE + "\nother = ClaudeAgentOptions()\n", "one native options call"),
            (FROZEN_OPTIONS_SOURCE.replace('thinking={"type": "adaptive"},', ""), "unsupported shape"),
        ]
        for source, reason in cases:
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                kernel._frozen_values(self.options(source))

    def test_options_source_cannot_execute_unapproved_expressions(self):
        cases = [
            ('config["model"] + "changed"', "expression is unsupported"),
            ("unknown_value", "unknown value"),
            ("config.__class__", "unsupported attribute"),
            ("config()", "unsupported call"),
        ]
        for expression, reason in cases:
            source = FROZEN_OPTIONS_SOURCE.replace('model=config["model"]', "model=" + expression)
            with self.subTest(expression=expression), self.assertRaisesRegex(ValueError, reason):
                kernel._frozen_values(self.options(source))

    def test_frozen_binding_requires_both_sources_and_records_their_hashes(self):
        options = self.options()
        options.profile = "kernel-frozen"
        options.kernel_source = None
        with self.assertRaisesRegex(ValueError, "explicitly pinned kernel source"):
            kernel.bindings(options)
        options.kernel_source = self.root / "kernel_lane.js"
        options.kernel_source.write_text("async function agentT(p, o) {\n return agent(p, o);\n}\n")
        options.kernel_source_sha256 = kernel._file_sha(options.kernel_source)
        bound = kernel.bindings(options)
        self.assertEqual(bound["kernel_source_sha256"], options.kernel_source_sha256)
        self.assertEqual(bound["options_source_sha256"], options.kernel_options_sha256)
        self.assertEqual(bound["wrapper_variant"], "agentT")

    def test_ambiguous_native_wrapper_is_rejected(self):
        source = "async function agentT(p, o) {\n return agent(p, o);\n}\n"
        with self.assertRaisesRegex(ValueError, "one native agentT"):
            kernel._extract_wrapper(source + source)


@dataclass
class _SDKOptions:
    model: str = "source-model"
    effort: object = None
    thinking: object = None
    allowed_tools: list = field(default_factory=list)
    permission_mode: str = "bypassPermissions"
    settings: str = '{"enableWorkflows":true,"ultracode":true}'
    extra_args: dict = field(default_factory=dict)
    env: dict = field(default_factory=dict)
    cwd: object = None
    cli_path: object = None
    setting_sources: object = None
    session_id: object = None
    strict_mcp_config: bool = False
    mcp_servers: dict = field(default_factory=dict)
    hooks: dict = field(default_factory=dict)
    stderr: object = None
    tools: object = None
    system_prompt: object = None
    include_partial_messages: bool = False


@dataclass
class _HookMatcher:
    matcher: str
    hooks: list
    timeout: int


class KernelLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.clients = []
        self.deadlines = []
        self.mode = "success"
        self.options = SimpleNamespace(profile="kernel-child", kernel_source=None,
            kernel_source_sha256=None, kernel_options_source=None, kernel_options_sha256=None,
            model="source-model", effort="high", cli=Path("/synthetic/native-cli"), prefix_count=1, timeout=7)
        self.sdk = ModuleType("claude_agent_sdk")
        self.sdk.ClaudeAgentOptions = _SDKOptions
        self.sdk.HookMatcher = _HookMatcher
        owner = self

        class Client:
            def __init__(self, *, options):
                self.options = options
                self.closed = False
                owner.clients.append(self)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                self.closed = True

            async def query(self, prompt):
                owner.assertEqual(prompt, "GEAK_SYNTHETIC_KERNEL_ROOT. Invoke the exact local Workflow once.")
                if owner.mode == "query_error":
                    raise RuntimeError("Synthetic client failure.")
                if owner.mode == "no_request":
                    return
                address = urlsplit(os.environ["ANTHROPIC_BASE_URL"])

                def post(raw, child=False):
                    connection = http.client.HTTPConnection(address.hostname, address.port, timeout=2)
                    headers = {"x-claude-code-session-id": self.options.session_id}
                    if child:
                        headers["x-claude-code-agent-id"] = "synthetic-child"
                    try:
                        connection.request("POST", "/v1/messages", body=raw, headers=headers)
                        response = connection.getresponse()
                        return response.status, response.read()
                    finally:
                        connection.close()

                raw = json.loads(body(session=self.options.session_id, prompt="root"))
                raw["model"] = self.options.model
                status, response = post(wire(raw))
                owner.assertEqual(status, 200)
                events = [json.loads(line[6:]) for line in response.splitlines() if line.startswith(b"data: ")]
                delta = next(event["delta"] for event in events if event["type"] == "content_block_delta")
                workflow_input = json.loads(delta["partial_json"])
                workflow = Path(workflow_input["scriptPath"])
                self.workflow = workflow
                self.workflow_bytes = workflow.read_bytes()
                hook = self.options.hooks["PreToolUse"][0].hooks[0]
                name = "Bash" if owner.mode == "wrong_tool" else "Workflow"
                decision = await hook({"session_id": self.options.session_id, "tool_name": name,
                    "tool_input": workflow_input}, "toolu_synthetic_catalog_workflow", None)
                if owner.mode == "wrong_tool":
                    owner.assertEqual(decision["hookSpecificOutput"]["permissionDecision"], "deny")
                    return
                owner.assertEqual(decision["hookSpecificOutput"]["permissionDecision"], "allow")
                raw = json.loads(body(session=self.options.session_id))
                raw["model"] = self.options.model
                raw["messages"][0]["content"] = [
                    {"type": "text", "text": kernel._date_reminder()},
                    {"type": "text", "text": kernel.CHILD_PROMPT},
                ]
                if owner.mode == "wrong_body_session":
                    raw["metadata"]["user_id"] = json.dumps({"session_id": "another-session"})
                status, response = post(wire(raw), child=True)
                owner.assertEqual(status, 400 if owner.mode == "wrong_body_session" else 200)
                owner.assertNotIn(b"tool_use", response)

        self.sdk.ClaudeSDKClient = Client
        self.anyio = ModuleType("anyio")

        def run(function):
            return asyncio.run(function())

        def fail_after(timeout):
            self.deadlines.append(timeout)
            return nullcontext()

        async def run_sync(function, timeout):
            self.assertEqual(timeout, self.options.timeout)
            if self.mode == "no_request":
                return False
            if self.mode == "wrong_tool":
                return True
            return function(1)

        self.anyio.run = run
        self.anyio.fail_after = fail_after
        self.anyio.to_thread = SimpleNamespace(run_sync=run_sync)
        from interface import run_e2e
        self.runner = run_e2e
        for patcher in (
            patch.dict(sys.modules, {"claude_agent_sdk": self.sdk, "anyio": self.anyio}),
            patch.object(run_e2e, "CLAUDE_MODEL", "source-model"),
            patch.object(run_e2e, "CLAUDE_EFFORT", "high"),
            patch.object(run_e2e, "CLAUDE_BIN", "/source/native-cli"),
            patch.dict(os.environ, {"GEAK_SHARED_TOOL_CACHE": "0", "GEAK_LOCAL_HELPERS": "0"}),
            patch("subprocess.Popen", side_effect=AssertionError("A native process must not start.")),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_public_options_use_the_existing_source_without_starting_a_client(self):
        original_client = self.sdk.ClaudeSDKClient
        selected = kernel._public_options(self.options)
        self.assertEqual(selected.model, "source-model")
        self.assertEqual(selected.extra_args, {"effort": "high"})
        self.assertEqual(selected.allowed_tools, self.runner.ALLOWED_TOOLS)
        self.assertEqual(selected.permission_mode, "bypassPermissions")
        self.assertEqual(selected.setting_sources, [])
        self.assertEqual(selected.cli_path, "/source/native-cli")
        self.assertIs(self.sdk.ClaudeSDKClient, original_client)
        self.assertEqual(self.clients, [])

    def test_public_options_reject_a_runner_that_constructs_no_options(self):
        with patch.object(self.runner, "_invoke_via_sdk", return_value="unused"), \
                self.assertRaisesRegex(ValueError, "one SDK options object"):
            kernel._public_options(self.options)

    def test_public_capture_uses_real_source_options_and_local_policy(self):
        environment = {key: os.environ.get(key) for key in ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY")}
        recorded, profile = kernel.capture(self.options)
        client = self.clients[-1]
        self.assertTrue(client.closed)
        self.assertFalse(Path(client.options.cwd).exists())
        self.assertFalse(client.workflow.exists())
        self.assertEqual(profile["model"], "source-model")
        self.assertEqual(profile["extra_args"], {"effort": "high"})
        self.assertEqual(profile["allowed_tools"], self.runner.ALLOWED_TOOLS)
        self.assertEqual(profile["permission_mode"], "bypassPermissions")
        self.assertIsNone(profile["output_token_cap"])
        self.assertIsNone(profile["tool_filters"]["ENABLE_TOOL_SEARCH"])
        self.assertEqual(profile["workflow_sha256"], kernel.sha(client.workflow_bytes))
        self.assertEqual(profile["dispatches_allowed"], 1)
        self.assertEqual(profile["dispatches_denied"], 0)
        self.assertTrue(recorded.request["portable_policy"]["only_marker_bytes_changed"])
        self.assertEqual(recorded.request["prompt_envelope"], "exact_prompt_with_optional_native_date_block")
        self.assertEqual(client.options.cli_path, str(self.options.cli))
        self.assertTrue(client.options.strict_mcp_config)
        self.assertEqual(client.options.mcp_servers, {})
        self.assertEqual(client.options.env["IS_SANDBOX"], "1")
        self.assertEqual({key: os.environ.get(key) for key in environment}, environment)

    def test_frozen_capture_preserves_its_pinned_source_profile(self):
        self.options.profile = "kernel-frozen"
        self.options.kernel_source = self.root / "kernel_lane.js"
        self.options.kernel_source.write_text("async function agentT(p, o) {\n return agent(p, o);\n}\n")
        self.options.kernel_source_sha256 = kernel._file_sha(self.options.kernel_source)
        self.options.kernel_options_source = self.root / "options.py"
        self.options.kernel_options_source.write_text(FROZEN_OPTIONS_SOURCE)
        self.options.kernel_options_sha256 = kernel._file_sha(self.options.kernel_options_source)
        recorded, profile = kernel.capture(self.options)
        self.assertEqual(profile["effort"], "high")
        self.assertEqual(profile["thinking"], {"type": "adaptive"})
        self.assertEqual(profile["allowed_tools"], ["Workflow", "Read", "ToolSearch"])
        self.assertEqual(profile["output_token_cap"], "64000")
        self.assertEqual(profile["tool_filters"]["ENABLE_TOOL_SEARCH"], "false")
        self.assertTrue(recorded.require_exact_suffix)
        self.assertTrue(recorded.request["portable_policy"]["applied"])
        self.assertTrue(self.clients[-1].closed)

    def test_unsupported_settings_stop_before_native_client_construction(self):
        with patch.object(self.runner, "WORKFLOW_SETTINGS", '{"hooks":{"SessionStart":[]}}'), \
                self.assertRaisesRegex(ValueError, "source settings"):
            kernel.capture(self.options)
        self.assertEqual(self.clients, [])

    def test_missing_request_or_client_error_closes_the_client_and_reports_only_status(self):
        for mode in ("no_request", "query_error"):
            with self.subTest(mode=mode), redirect_stderr(io.StringIO()) as errors:
                self.mode = mode
                with self.assertRaises(RuntimeError):
                    kernel.capture(self.options)
                client = self.clients[-1]
                self.assertTrue(client.closed)
                self.assertFalse(Path(client.options.cwd).exists())
                message = errors.getvalue()
                self.assertIn("GEAK_CAPTURE_STATUS", message)
                self.assertNotIn("Synthetic client failure", message)
                self.assertNotIn(kernel.CHILD_PROMPT, message)
                self.assertNotIn(str(client.options.cwd), message)

    def test_protocol_failures_return_no_capture_and_close_the_client(self):
        for mode in ("wrong_tool", "wrong_body_session"):
            with self.subTest(mode=mode), redirect_stderr(io.StringIO()) as errors:
                self.mode = mode
                with self.assertRaisesRegex(RuntimeError, "protocol checks"):
                    kernel.capture(self.options)
                self.assertTrue(self.clients[-1].closed)
                self.assertFalse(self.clients[-1].workflow.exists())
                self.assertIn("GEAK_CAPTURE_STATUS", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
