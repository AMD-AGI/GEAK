# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture one source-bound native Workflow child without external tool calls."""

import ast
import json
import os
import sys
import tempfile
import uuid
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from .native_prefix_marker import DECODER, sha, wire
from .produce_catalog import ROOT, SyntheticCapture, _file_sha, _sse

CHILD_PROMPT = "GEAK_SYNTHETIC_KERNEL_CATALOG_CHILD. Return the fixed local terminal text."


def _date_reminder():
    # Match the native CLI's local calendar date with an aware datetime.
    return ("<system-reminder>\nAs you answer the user's questions, you can use the following context:\n"
            "# currentDate\nToday's date is " + datetime.now().astimezone().date().isoformat()
            + ".\n\n      IMPORTANT: this context may or may not be relevant to your tasks. "
            "You should not respond to this context unless it is highly relevant to your task.\n"
            "</system-reminder>\n\n")


def _prompt_matches(texts):
    # The pinned native CLI adds this exact date context as a separate block.
    return texts == [CHILD_PROMPT] or texts == [_date_reminder(), CHILD_PROMPT]


def add_arguments(parser):
    parser.add_argument("--kernel-source", type=Path,
                        help="Use this kernel_lane.js source instead of the current public file.")
    parser.add_argument("--kernel-source-sha256", help="Require this hash for an external kernel source.")
    parser.add_argument("--kernel-options-source", type=Path,
                        help="Read the frozen profile's ClaudeAgentOptions call from this Python source.")
    parser.add_argument("--kernel-options-sha256", help="Require this hash for the frozen options source.")


def _source(options):
    selected = options.kernel_source or ROOT / "kernel_workflow/kernel_lane.js"
    if options.kernel_source and _file_sha(selected) != options.kernel_source_sha256:
        raise ValueError("The external kernel source does not match its explicit hash.")
    return selected


def _extract_wrapper(source):
    """Copy whole source functions without translating their JavaScript."""
    start = source.index("async function agentT(p, o) {")
    if source.count("async function agentT(p, o) {") != 1:
        raise ValueError("The source must define one native agentT function.")
    end = source.index("\n}\n", start) + 2
    variant = "agentT"
    if "function tlAgent(prompt, o, attempt) {" in source[:start]:
        start = source.index("function tlAgent(prompt, o, attempt) {")
        variant = "tlAgent_and_agentT"
    return source[start:end], variant


def _frozen_values(options):
    path = options.kernel_options_source
    if not path or _file_sha(path) != options.kernel_options_sha256:
        raise ValueError("The frozen options source requires its explicit hash.")
    if options.effort == "ultracode":
        raise ValueError("The frozen profile requires an explicit native effort value.")
    tree = ast.parse(path.read_text())
    constants = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in {"TOOLS", "SYSTEM_APPEND"}):
            constants[node.targets[0].id] = ast.literal_eval(node.value)
    if set(constants) != {"TOOLS", "SYSTEM_APPEND"}:
        raise ValueError("The frozen source omits its tool or system constants.")
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "ClaudeAgentOptions"]
    if len(calls) != 1:
        raise ValueError("The frozen source must contain one native options call.")
    selected = {"model", "effort", "thinking", "allowed_tools", "permission_mode", "settings",
                "system_prompt", "include_partial_messages"}
    values = {}
    scope = {**constants, "json": json,
             "config": {"model": options.model, "effort": options.effort, "settings": {}}}
    for item in calls[0].keywords:
        if item.arg in selected:
            expression = ast.Expression(item.value)
            for node in ast.walk(expression):
                if not isinstance(node, (ast.Expression, ast.Constant, ast.Name, ast.Load, ast.Dict, ast.List,
                                         ast.Starred, ast.Subscript, ast.Attribute, ast.Call, ast.keyword)):
                    # Invalid source expressions retain the ValueError contract.
                    raise ValueError("The frozen options expression is unsupported.")  # noqa: TRY004
                if isinstance(node, ast.Name) and node.id not in scope:
                    raise ValueError("The frozen options expression uses an unknown value.")
                if isinstance(node, ast.Attribute) and not (
                    isinstance(node.value, ast.Name) and (node.value.id, node.attr) in
                    {("config", "get"), ("json", "dumps")}
                ):
                    raise ValueError("The frozen options expression uses an unsupported attribute.")
                if isinstance(node, ast.Call) and not isinstance(node.func, ast.Attribute):
                    # Invalid source calls retain the ValueError contract.
                    raise ValueError("The frozen options expression uses an unsupported call.")  # noqa: TRY004
            values[item.arg] = eval(compile(expression, "<pinned-native-options>", "eval"),
                                    {"__builtins__": {}}, scope)
    if set(values) != selected:
        raise ValueError("The frozen options source has an unsupported shape.")
    return values


def bindings(options):
    source = _source(options)
    wrapper, variant = _extract_wrapper(source.read_text())
    result = {"kernel_source_sha256": _file_sha(source), "wrapper_sha256": sha(wrapper.encode()),
              "wrapper_variant": variant, "profile_module_sha256": _file_sha(Path(__file__))}
    if options.profile == "kernel-frozen":
        if options.kernel_source is None:
            raise ValueError("The frozen profile requires an explicitly pinned kernel source.")
        _frozen_values(options)
        result["options_source_sha256"] = _file_sha(options.kernel_options_source)
    else:
        result["options_source_sha256"] = _file_sha(ROOT / "interface/run_e2e.py")
    return result


def _workflow(source):
    wrapper, _ = _extract_wrapper(source)
    return ("export const meta={name:'synthetic-kernel-catalog',description:'Synthetic catalog',"
            "phases:[{title:'Optimize',detail:'One native child'}]};\n"
            "const LLM_STATS=false, ABL_B6=false, AGENT_RETRIES=1, AGENT_TIMEOUT_MS=0;\n"
            + wrapper + "\nconst schema={type:'object',properties:{ok:{type:'boolean'}},required:['ok']};\n"
            + "return await agentT(" + json.dumps(CHILD_PROMPT)
            + ",{phase:'Optimize',label:'catalog child',schema});\n")


def _tool_sse(model, tool_input):
    response = _sse(model).decode()
    # Build the same native SSE envelope with one exact Workflow call.
    events = [json.loads(line[6:]) for line in response.splitlines() if line.startswith("data: ")]
    for event in events:
        if event["type"] == "content_block_start":
            event["content_block"] = {"type": "tool_use", "id": "toolu_synthetic_catalog_workflow",
                                      "name": "Workflow", "input": {}}
        elif event["type"] == "content_block_delta":
            event["delta"] = {"type": "input_json_delta", "partial_json": json.dumps(tool_input)}
        elif event["type"] == "message_delta":
            event["delta"]["stop_reason"] = "tool_use"
    return b"".join(b"event: " + event["type"].encode() + b"\ndata: " + wire(event) + b"\n\n"
                    for event in events)


class KernelCapture(SyntheticCapture):
    def __init__(self, prefix_count, workflow, session_id):
        super().__init__(prefix_count)
        self.workflow_input = {"scriptPath": str(workflow), "args": {}}
        self.expected_session_id = session_id
        self.root_requests = 0
        self.child_id = None
        self.current_child = None
        self.permitted = 0
        self.denied = 0
        self.require_exact_suffix = False
        self.stop_after_capture = True

    def validate_headers(self, headers):
        session = headers.get("x-claude-code-session-id")
        if session != self.expected_session_id:
            raise ValueError("The Workflow capture received another session.")
        self.session_id = session
        self.current_child = headers.get("x-claude-code-agent-id")
        if self.current_child and self.child_id and self.current_child != self.child_id:
            raise ValueError("The Workflow capture received a second child.")

    def message(self, raw):
        body = DECODER.decode(raw.decode())
        if not self.current_child:
            self.root_requests += 1
            if self.root_requests == 1:
                return _tool_sse(body["model"], self.workflow_input)
            if self.root_requests > 3:
                raise ValueError("The root exceeded its synthetic response limit.")
            return _sse(body["model"])
        texts = []
        for message in body.get("messages", []):
            if message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str):
                    texts.append(content)
                elif isinstance(content, list):
                    texts.extend(block.get("text", "") for block in content if isinstance(block, dict))
        if self.permitted != 1 or not _prompt_matches(texts):
            raise ValueError("The child request lacks its exact synthetic dispatch binding.")
        tools = body.get("tools", [])
        structured = [index for index, tool in enumerate(tools) if tool.get("name") == "StructuredOutput"]
        if (structured != [len(tools) - 1] or self.prefix_count >= len(tools)
                or (self.require_exact_suffix and len(tools) != self.prefix_count + 1)):
            raise ValueError("The Workflow child has an unsupported output-tool layout: "
                             + json.dumps({"tool_count": len(tools), "output_tool_indices": structured}))
        self.child_id = self.current_child
        response = super().message(raw)
        self.request.update(role="native_workflow_agent", child_id_sha256=sha(self.child_id.encode()),
                            dispatch_prompt_sha256=sha(CHILD_PROMPT.encode()),
                            prompt_envelope="exact_prompt_with_optional_native_date_block")
        return response

    async def pre_tool(self, data, _tool_id, _context):
        allowed = (self.permitted == 0 and data.get("tool_name") == "Workflow"
                   and _tool_id == "toolu_synthetic_catalog_workflow"
                   and data.get("tool_input") == self.workflow_input
                   and data.get("session_id") == self.expected_session_id)
        if allowed:
            self.permitted += 1
        else:
            self.denied += 1
        return {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                "permissionDecision": "allow" if allowed else "deny"}}


def _public_options(options):
    import claude_agent_sdk

    from interface import run_e2e
    selected = []

    class Captured(Exception):
        pass

    class ObserveClient:
        def __init__(self, *, options):
            selected.append(options)

        async def __aenter__(self):
            raise Captured()

        async def __aexit__(self, *_):
            return False

    with patch.object(claude_agent_sdk, "ClaudeSDKClient", ObserveClient):
        try:
            run_e2e._invoke_via_sdk("synthetic options only", 1, settings_profile="isolated")
        except Captured:
            pass
    if len(selected) != 1:
        raise ValueError("The public source did not construct one SDK options object.")
    return selected[0]


def capture(options):
    import anyio
    from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, HookMatcher
    bindings(options)
    session = str(uuid.uuid4())
    with tempfile.TemporaryDirectory(prefix="geak-kernel-catalog-") as directory:
        temporary = Path(directory)
        work = temporary / "work"
        work.mkdir()
        workflow = temporary / "workflow.js"
        workflow.write_text(_workflow(_source(options).read_text()))
        captured = KernelCapture(options.prefix_count, workflow, session)
        captured.require_exact_suffix = options.profile == "kernel-frozen"
        if options.profile == "kernel-frozen":
            original = ClaudeAgentOptions(**_frozen_values(options))
        else:
            original = _public_options(options)
        settings = json.loads(original.settings)
        if set(settings) - {"enableWorkflows", "ultracode"}:
            raise ValueError("The source settings exceed the synthetic kernel profile.")
        environment = {"CLAUDE_CONFIG_DIR": str(temporary / "config"), "IS_SANDBOX": "1"}
        if options.profile == "kernel-frozen":
            environment.update(ENABLE_TOOL_SEARCH="false", CLAUDE_CODE_MAX_OUTPUT_TOKENS="64000")
        prepared = replace(original, cli_path=str(options.cli), cwd=str(work), session_id=session,
                           setting_sources=[], strict_mcp_config=True, mcp_servers={},
                           hooks={"PreToolUse": [HookMatcher(matcher=".*", hooks=[captured.pre_tool], timeout=10)]},
                           env={**(original.env or {}), **environment}, stderr=lambda _line: None)
        captured.expected_model = prepared.model

        async def run():
            with anyio.fail_after(options.timeout):
                async with ClaudeSDKClient(options=prepared) as client:
                    await client.query("GEAK_SYNTHETIC_KERNEL_ROOT. Invoke the exact local Workflow once.")
                    completed = await anyio.to_thread.run_sync(captured.completed.wait, options.timeout)
                    if not completed:
                        raise RuntimeError("The source-bound Workflow child did not reach the local endpoint.")

        with captured.server() as endpoint, patch.dict(os.environ, {
            "ANTHROPIC_BASE_URL": endpoint, "ANTHROPIC_API_KEY": "synthetic-local-key",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "DISABLE_AUTOUPDATER": "1",
            "GEAK_SHARED_TOOL_CACHE": "0", "GEAK_LOCAL_HELPERS": "0",
        }):
            try:
                anyio.run(run)
            except Exception as error:
                print("GEAK_CAPTURE_STATUS " + json.dumps({"profile": options.profile,
                    "error_class": type(error).__name__, "requests": captured.requests,
                    "root_requests": captured.root_requests, "errors": captured.failure_reasons,
                    "permitted": captured.permitted, "denied": captured.denied}), file=sys.stderr)
                raise
        if captured.errors or captured.denied or captured.permitted != 1 or captured.catalog is None:
            print("GEAK_CAPTURE_STATUS " + json.dumps({"profile": options.profile,
                "requests": captured.requests, "root_requests": captured.root_requests,
                "errors": captured.failure_reasons, "permitted": captured.permitted,
                "denied": captured.denied}), file=sys.stderr)
            raise RuntimeError("The source-bound kernel capture failed its protocol checks.")
        profile = {"scope": "one_source_bound_native_workflow_child", "role": "native_workflow_agent",
                   "model": prepared.model, "effort": prepared.effort, "extra_args": prepared.extra_args,
                   "thinking": prepared.thinking, "settings": settings,
                   "permission_mode": prepared.permission_mode,
                   "allowed_tools": prepared.allowed_tools, "tool_filters": {
                       "allowed_tools": prepared.allowed_tools, "tools": prepared.tools,
                       "ENABLE_TOOL_SEARCH": environment.get("ENABLE_TOOL_SEARCH")},
                   "setting_sources": [], "strict_mcp_config": True,
                   "output_token_cap": environment.get("CLAUDE_CODE_MAX_OUTPUT_TOKENS"),
                   "workflow_sha256": _file_sha(workflow), "dispatches_allowed": captured.permitted,
                   "dispatches_denied": captured.denied, "tool_use_responses": 1,
                   "completion": "Capture one child request, return local text, then disconnect the SDK.",
                   "differences_from_default_runner": [
                       "Run a synthetic Workflow containing only the source-extracted native agent wrapper.",
                       "Use an empty temporary work directory and configuration directory.",
                       "Disable filesystem settings and use strict empty MCP configuration.",
                       "Install one SDK hook that permits only the exact synthetic Workflow dispatch.",
                       "Use the selected source profile's model, effort, filters, and permission mode.",
                       "Disconnect after the child response without requiring StructuredOutput or TaskOutput."],
                   "limits": ["The producer runs only the extracted native agent wrapper.",
                              "It does not run the complete kernel Workflow or any kernel tool.",
                              "The selected source and options can differ from the frozen experiment.",
                              "Only an explicit fingerprint comparison establishes prefix parity.",
                              "Synthetic usage does not measure token cost or cache benefit."]}
        return captured, profile
