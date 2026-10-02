# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test the combined native SDK adapter with loopback traffic only."""

import asyncio
import hashlib
import http.client
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import urlsplit

from interface.native_cost_controls.registered_prefix import POLICY_NAME
from interface.native_cost_controls.sdk_helpers import WorkflowSDKClient
from interface.test_native_helpers import (
    Contract,
    TaskProgressMessage,
    TaskStartedMessage,
    UserMessage,
)
from interface.test_native_sdk_cache import Options as CacheOptions
from interface.test_native_sdk_cache import source_catalog_environment
from interface.test_shared_tool_cache import INSERTION, request
from interface.test_system_envelope import synthetic_system_policy

NODE = shutil.which("node")

@dataclass
class Options(CacheOptions):
    cwd: str = None
    session_store: object = None
    session_store_flush: str = "batched"
    settings: str = '{"enableWorkflows":true,"ultracode":true}'
    setting_sources: list = field(default_factory=list)
    plugins: list = None
    can_use_tool: object = None
    system_prompt: object = None
    allowed_tools: list = field(default_factory=lambda: ["Workflow", "Bash"])
    extra_args: dict = field(default_factory=lambda: {"effort": "high"})


@dataclass
class Matcher:
    matcher: str = None
    hooks: list = field(default_factory=list)


class Client:
    retry_unsupported = False
    post_feedback = None
    def __init__(self, *, options):
        self.options = options
        self.closed = False
        self.responses = []
        self.hook_replies = []
        self.provider_body = None
        self.journal_root = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        self.closed = True

    async def query(self, prompt):
        self.prompt = prompt

    async def hook(self, event, name, value, tool_id, agent=None, response=None):
        data = {"hook_event_name": event, "session_id": self.options.session_id, "cwd": self.options.cwd,
            "tool_name": name, "tool_input": value, "agent_id": agent, "tool_response": response}
        for matcher in self.options.hooks[event]:
            for callback in matcher.hooks:
                self.hook_replies.append(await callback(data, tool_id, {}))

    def send(self, body, agent):
        endpoint = urlsplit(self.options.env["ANTHROPIC_BASE_URL"])
        connection = http.client.HTTPConnection(endpoint.hostname, endpoint.port, timeout=3)
        headers = {"Content-Type": "application/json", "x-claude-code-session-id": self.options.session_id,
            "x-claude-code-agent-id": agent, "Authorization": "Bearer public-test-value"}
        raw = json.dumps(body, indent=2).encode()
        if agent == "engineer":
            self.provider_body = raw
        connection.request("POST", "/v1/messages", raw, headers)
        response = connection.getresponse()
        value = response.status, json.loads(response.read())
        connection.close()
        self.responses.append(value)
        return value[1]

    async def receive_messages(self):
        if not self.options.session_store:
            yield "unchanged-message"
            return
        contract = Contract()
        await self.hook("PreToolUse", "Workflow", contract.root_request, "root-tool")
        yield TaskStartedMessage()
        run_id = "wf_" + uuid.uuid4().hex[:12]
        directory = self.journal_root / "projects/fixture" / self.options.session_id / "subagents/workflows" / run_id
        directory.mkdir(parents=True)
        journal = directory / "journal.jsonl"
        journal.write_text("")
        yield UserMessage(content=[{"tool_use_id": "root-tool", "is_error": False}],
            tool_use_result={"status": "async_launched", "taskType": "local_workflow", "taskId": "root-task",
                "scriptPath": contract.root_request["scriptPath"], "runId": run_id, "transcriptDir": str(directory)})
        def append_result(kind, agent, result=None):
            row = {"type": kind, "key": "v2:" + hashlib.sha256(agent.encode()).hexdigest(), "agentId": agent}
            if kind == "result":
                row["result"] = result
            with journal.open("a") as handle:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        nodes = [{"type": "workflow_phase", "index": 5, "title": "▸ kernel-lane"}]
        for index, (agent, label) in enumerate([("helper", "clock pre-r1"), ("engineer", "engineer:test")], 1):
            append_result("started", agent)
            nodes.append({"type": "workflow_agent", "phaseIndex": 5, "index": index, "label": label,
                "agentId": agent, "attempt": 1, "state": "running"})
        yield TaskProgressMessage(data={"workflow_progress": nodes})
        for agent, label in [("helper", "clock pre-r1"), ("engineer", "engineer:test")]:
            await self.options.session_store.append({"session_id": self.options.session_id, "project_key": "fixture",
                "subpath": "subagents/workflows/" + run_id + "/agent-" + agent}, [{"type": "user", "parentUuid": None,
                    "cwd": self.options.cwd, "uuid": agent, "message": {"content": label}}])
        body = request()
        body["metadata"] = {"user_id": json.dumps({"session_id": self.options.session_id})}
        body["tools"].insert(0, {"name": "Bash", "input_schema": {"type": "object"}})
        body["stream"] = False
        body["messages"][0]["content"][0]["text"] = "clock pre-r1"
        first = (await asyncio.to_thread(self.send, body, "helper"))["content"][0]
        await self.hook("PreToolUse", "Bash", first["input"], first["id"], "helper")
        await self.hook("PostToolUse", "Bash", first["input"], first["id"], "helper",
            {"stdout": "42\n", "stderr": "", "interrupted": False})
        if self.retry_unsupported:
            nodes.append({"type": "workflow_agent", "phaseIndex": 5, "index": 3, "label": "clock pre-r1 (retry 1)",
                "agentId": "retry", "attempt": 2, "state": "running"})
            yield TaskProgressMessage(data={"workflow_progress": nodes})
            await self.options.session_store.append({"session_id": self.options.session_id, "project_key": "fixture",
                "subpath": "subagents/workflows/" + run_id + "/agent-retry"}, [{"type": "user", "parentUuid": None,
                    "cwd": self.options.cwd, "uuid": "retry", "message": {"content": "unsupported retry prompt"}}])
            body["messages"][0]["content"][0]["text"] = "unsupported retry prompt"
            await asyncio.to_thread(self.send, body, "retry")
            yield "unchanged-message"
            return
        body["messages"].append({"role": "assistant", "content": [first]})
        body["messages"].append({"role": "user", "content": [{"type": "tool_result",
            "tool_use_id": first["id"], "content": "42\n", "is_error": False}]})
        if self.post_feedback:
            body["messages"][-1]["content"].append({"type": "text", "text": self.post_feedback})
            await asyncio.to_thread(self.send, body, "helper")
            yield "unchanged-message"
            return
        second = (await asyncio.to_thread(self.send, body, "helper"))["content"][0]
        await self.hook("PreToolUse", "StructuredOutput", second["input"], second["id"], "helper")
        await self.hook("PostToolUse", "StructuredOutput", second["input"], second["id"], "helper")
        append_result("result", "helper", second["input"])
        nodes[1].update(state="done", lastToolName="StructuredOutput", resultPreview=json.dumps(second["input"], separators=(",", ":")))
        yield TaskProgressMessage(data={"workflow_progress": nodes})
        body["system"] = request()["system"]
        body["messages"] = [{"role": "user", "content": [{"type": "text", "text": "engineer:test",
            "cache_control": {"type": "ephemeral", "ttl": "5m"}}]}]
        await asyncio.to_thread(self.send, body, "engineer")
        yield "unchanged-message"

    async def receive_response(self):
        async for message in self.receive_messages():
            yield message


class SDKHelperTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.system_policy = synthetic_system_policy()
        self.system_policy.start()
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name)
        self.calls = []
        calls = self.calls

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                return
            def do_POST(self):
                calls.append(self.rfile.read(int(self.headers["Content-Length"])))
                payload = b'{"usage":{"input_tokens":7,"output_tokens":3}}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.options = Options(cwd=str(self.workspace), session_id="fixture-session",
            env={"ANTHROPIC_BASE_URL": "http://127.0.0.1:" + str(self.server.server_port)})
        self.clients = []

    def tearDown(self):
        self.system_policy.stop()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temporary.cleanup()
        self.environment.stop()

    def factory(self, *, options):
        client = Client(options=options)
        client.journal_root = self.workspace
        self.clients.append(client)
        return client

    def managed(self, **kwargs):
        return WorkflowSDKClient(self.factory, kwargs.pop("options", self.options),
            workflow_request=Contract().root_request, source_root=self.workspace,
            contract_factory=Contract, hook_matcher_factory=Matcher, **kwargs)

    async def test_disabled_keeps_identical_options_and_no_listener(self):
        async with self.managed() as client:
            self.assertEqual([message async for message in client.receive_messages()], ["unchanged-message"])
            self.assertIs(self.clients[0].options, self.options)
            self.assertEqual(client.helper_status, "disabled")
        self.assertTrue(self.clients[0].closed)
        self.assertEqual(self.calls, [])

    async def test_response_iterator_observes_the_same_native_lifecycle(self):
        async with self.managed(helpers_enabled=True) as client:
            messages = [message async for message in client.receive_response()]
            self.assertEqual(messages[-1], "unchanged-message")
        self.assertEqual(len(self.calls), 1)

    async def test_combined_session_keeps_permissions_and_forwards_only_scientific_request(self):
        async with self.managed(helpers_enabled=True, cache_enabled=True) as client:
            await client.query("unchanged-prompt")
            messages = [message async for message in client.receive_messages()]
            self.assertEqual(messages[-1], "unchanged-message")
            self.assertEqual(client.helper_status, "active")
            options = self.clients[0].options
            for key in ["model", "effort", "permission_mode", "allowed_tools", "extra_args", "settings", "can_use_tool"]:
                self.assertEqual(getattr(options, key), getattr(self.options, key))
            self.assertTrue(all(reply == {} for reply in self.clients[0].hook_replies))
            self.assertEqual(self.clients[0].responses[1][1]["content"][0]["input"], {"epoch": 42})
            self.assertEqual(self.clients[0].responses[2][1]["usage"], {"input_tokens": 7, "output_tokens": 3})
            self.assertEqual(len(self.calls), 1)
            self.assertEqual(self.calls[0].replace(INSERTION, b"", 1), self.clients[0].provider_body)
        self.assertTrue(self.clients[0].closed)

    async def test_helpers_work_with_cache_disabled_and_preserve_provider_bytes(self):
        async with self.managed(helpers_enabled=True) as client:
            [message async for message in client.receive_messages()]
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0], self.clients[0].provider_body)

    def registered_environment(self):
        tools = [{"name": "Bash", "input_schema": {"type": "object"}}] + request()["tools"][:-1]
        return {**self.options.env, **source_catalog_environment(self.workspace / "catalog.json", tools)}

    async def test_registered_cache_keeps_helpers_local_and_preserves_native_options(self):
        options = replace(self.options, env={**self.registered_environment(),
            "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE": "75"},
            settings='{"enableWorkflows":true,"ultracode":true,"autoCompactEnabled":true}')
        async with self.managed(options=options, helpers_enabled=True, cache_enabled=True) as client:
            messages = [message async for message in client.receive_messages()]
            self.assertEqual(messages[-1], "unchanged-message")
            self.assertEqual(client.helper_status, "active")
            self.assertEqual(client.cache_session.policy_name, POLICY_NAME)
            self.assertEqual(client.cache_session.registration.count, 3)
            prepared = self.clients[-1].options
            for key in ("model", "effort", "permission_mode", "allowed_tools", "extra_args", "settings",
                    "can_use_tool", "system_prompt", "setting_sources"):
                self.assertIs(getattr(prepared, key), getattr(options, key), key)
            self.assertEqual(prepared.env["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"], "75")
            self.assertTrue(all(reply == {} for reply in self.clients[-1].hook_replies))
            self.assertEqual(self.clients[-1].responses[1][1]["content"][0]["input"], {"epoch": 42})
            self.assertEqual(len(self.calls), 1)
            self.assertEqual(self.calls[0].replace(INSERTION, b"", 1), self.clients[-1].provider_body)
        self.assertIsNone(client.registry)

    async def test_helpers_ignore_invalid_registered_catalog_when_cache_is_disabled(self):
        options = replace(self.options, env={**self.options.env, "GEAK_SHARED_TOOL_CACHE_POLICY": POLICY_NAME,
            "GEAK_SHARED_TOOL_CATALOG": "unused-relative-path.json",
            "GEAK_SHARED_TOOL_CATALOG_SHA256": "invalid-unused-hash"})
        async with self.managed(options=options, helpers_enabled=True) as client:
            [message async for message in client.receive_messages()]
            self.assertEqual(client.helper_status, "active")
            self.assertIsNone(client.cache_session.registration)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0], self.clients[-1].provider_body)

    async def test_invalid_registration_stops_combined_client_and_removes_registry(self):
        options = replace(self.options, env={**self.registered_environment(),
            "GEAK_SHARED_TOOL_CATALOG_SHA256": "0" * 64})

        def proxy(*_args, **_kwargs):
            self.fail("The proxy must not start after invalid registration.")

        managed = self.managed(options=options, helpers_enabled=True, cache_enabled=True, proxy_factory=proxy)
        with self.assertRaises(ValueError):
            await managed.__aenter__()
        self.assertEqual(self.clients, [])
        self.assertEqual(self.calls, [])
        self.assertIsNone(managed.registry)
        self.assertIsNone(managed._temporary)

    @staticmethod
    @contextmanager
    def inactive_proxy(endpoint, policy, *, enabled, transport):
        try:
            yield SimpleNamespace(active=False, status="unsupported_configuration", decisions=())
        finally:
            transport.close()

    async def test_disabled_cache_ignores_invalid_registration_in_helper_fallbacks(self):
        base = replace(self.options, env={**self.options.env, "GEAK_SHARED_TOOL_CACHE_POLICY": POLICY_NAME,
            "GEAK_SHARED_TOOL_CATALOG": "unused-relative-path.json"})
        cases = [
            (base, {"helpers_enabled": False}, "disabled"),
            (replace(base, system_prompt="Original system policy."), {"helpers_enabled": True},
                "unsupported_system_prompt"),
            (base, {"helpers_enabled": True, "proxy_factory": self.inactive_proxy}, "unsupported_configuration"),
        ]
        for options, settings, reason in cases:
            with self.subTest(reason=reason):
                async with self.managed(options=options, **settings) as client:
                    self.assertIs(self.clients[-1].options, options)
                    self.assertEqual(client.helper_status, reason)
                    self.assertEqual(client.cache_session.status, "disabled")
                    self.assertIsNone(client.cache_session.registration)
                    self.assertIsNone(client.registry)
        self.assertEqual(self.calls, [])

    async def test_registered_cache_remains_active_in_both_helper_fallbacks(self):
        base = replace(self.options, env=self.registered_environment())
        for unsupported_options in (True, False):
            with self.subTest(unsupported_options=unsupported_options):
                options = replace(base, system_prompt="Original system policy.") if unsupported_options else base
                settings = {} if unsupported_options else {"proxy_factory": self.inactive_proxy}
                async with self.managed(options=options, helpers_enabled=True, cache_enabled=True, **settings) as client:
                    self.assertNotEqual(client.helper_status, "active")
                    self.assertIsNone(client.registry)
                    self.assertEqual(client.cache_session.status, "active")
                    self.assertEqual(client.cache_session.policy_name, POLICY_NAME)
                    self.assertEqual(client.cache_session.registration.count, 3)
                    prepared = self.clients[-1].options
                    self.assertIs(prepared.hooks, options.hooks)
                    self.assertIs(prepared.system_prompt, options.system_prompt)
                    self.assertIsNone(prepared.session_store)
                    body = request()
                    body["tools"].insert(0, {"name": "Bash", "input_schema": {"type": "object"}})
                    body["metadata"] = {"user_id": json.dumps({"session_id": prepared.session_id})}
                    await asyncio.to_thread(self.clients[-1].send, body, "engineer")
                    self.assertEqual(self.calls[-1].replace(INSERTION, b"", 1), self.clients[-1].provider_body)
        self.assertEqual(len(self.calls), 2)

    async def test_combined_session_blocks_unsupported_retry_after_native_bash(self):
        with patch.object(Client, "retry_unsupported", True):
            async with self.managed(helpers_enabled=True, cache_enabled=True) as client:
                messages = [message async for message in client.receive_messages()]
                self.assertEqual(messages[-1], "unchanged-message")
                self.assertEqual(self.clients[0].responses[-1][0], 502)
                self.assertEqual(len(client.registry.operations), 1)
                driver = next(iter(client.registry.operations.values()))
                self.assertEqual(json.loads(driver.path.read_bytes())["bash_emissions"], 1)
        self.assertEqual(self.calls, [])

    async def test_combined_session_rejects_managed_block_or_context_after_bash(self):
        for feedback in ("A managed hook blocked this result.", "A managed hook supplied extra context."):
            with self.subTest(feedback=feedback), patch.object(Client, "post_feedback", feedback):
                async with self.managed(helpers_enabled=True, cache_enabled=True) as client:
                    [message async for message in client.receive_messages()]
                    self.assertEqual(self.clients[-1].responses[-1][0], 502)
                    driver = next(iter(client.registry.operations.values()))
                    state = json.loads(driver.path.read_bytes())
                    self.assertEqual(state["stage"], "UNKNOWN")
                    self.assertEqual(state["bash_emissions"], 1)
        self.assertEqual(self.calls, [])

    async def test_unsupported_input_callbacks_retain_original_objects_and_decisions(self):
        async def permission(*_args):
            return "original-deny"
        cases = [replace(self.options, can_use_tool=permission),
            replace(self.options, hooks={"PreToolUse": [Matcher(hooks=[permission])]}),
            replace(self.options, setting_sources=None),
            replace(self.options, setting_sources=["project"]),
            replace(self.options, settings="/fixture/settings.json")]
        cases.append(replace(self.options, extra_args={"plugin-dir": "/synthetic/plugin"}))
        cases.append(replace(self.options, plugins=[{"type": "local", "path": "/synthetic/plugin"}]))
        cases.append(replace(self.options, system_prompt="A caller-supplied system policy."))
        for options in cases:
            with self.subTest(options=options):
                async with self.managed(options=options, helpers_enabled=True) as client:
                    self.assertIs(self.clients[-1].options, options)
                    self.assertNotEqual(client.helper_status, "active")
                if options.can_use_tool is not None:
                    self.assertEqual(await options.can_use_tool(), "original-deny")
        self.assertEqual(self.calls, [])

    async def test_existing_eager_store_keeps_its_identity(self):
        class Store:
            async def append(self, key, entries):
                return
            async def load(self, key):
                return None
        store = Store()
        options = replace(self.options, session_store=store, session_store_flush="eager")
        async with self.managed(options=options, helpers_enabled=True) as client:
            prepared = self.clients[-1].options
            self.assertIs(prepared.session_store.original, store)
            self.assertEqual(client.helper_status, "active")

    async def test_existing_post_hook_rejection_retains_the_original_client_path(self):
        async def post(*_args):
            return {"decision": "block", "reason": "A synthetic original rejection."}
        original = Matcher(hooks=[post])
        for event in ("PostToolUse", "PostToolUseFailure"):
            options = replace(self.options, hooks={event: [original]})
            async with self.managed(options=options, helpers_enabled=True) as client:
                self.assertIs(self.clients[-1].options, options)
                self.assertEqual(client.helper_status, "unsupported_tool_hooks")
                self.assertEqual(await options.hooks[event][0].hooks[0](), await post())

    async def test_unsupported_transport_removes_helper_hooks_before_client_start(self):
        options = replace(self.options, env={**self.options.env, "HTTPS_PROXY": "http://fixture.invalid"})
        async with self.managed(options=options, helpers_enabled=True) as client:
            self.assertIs(self.clients[-1].options, options)
            self.assertEqual(client.helper_status, "unsupported_transport_settings")
        self.assertEqual(self.calls, [])

    async def test_failed_sdk_start_closes_proxy_and_removes_registry(self):
        class Failed(Client):
            async def __aenter__(self):
                raise RuntimeError("A synthetic startup failure.")
        managed = WorkflowSDKClient(Failed, self.options, helpers_enabled=True,
            workflow_request=Contract().root_request, source_root=self.workspace,
            contract_factory=Contract, hook_matcher_factory=Matcher)
        with self.assertRaises(RuntimeError):
            await managed.__aenter__()
        self.assertIsNone(managed.registry)
        self.assertIsNone(managed._temporary)
        self.assertFalse(managed.cache_session.proxy.active)

    async def test_unsupported_session_store_and_resume_keep_original_options(self):
        for options in [replace(self.options, resume="old-session"),
                replace(self.options, session_store=object()), replace(self.options, settings='{"hooks":{"PreToolUse":[]}}')]:
            with self.subTest(options=options):
                async with self.managed(options=options, helpers_enabled=True) as client:
                    self.assertIs(self.clients[-1].options, options)
                    self.assertNotEqual(client.helper_status, "active")

    async def test_unavailable_source_contract_retains_original_client_options(self):
        def unavailable(*_args):
            raise ValueError("The source layout is unsupported.")
        managed = WorkflowSDKClient(self.factory, self.options, helpers_enabled=True,
            workflow_request=Contract().root_request, source_root=self.workspace,
            contract_factory=unavailable, hook_matcher_factory=Matcher)
        async with managed as client:
            self.assertIs(self.clients[-1].options, self.options)
            self.assertEqual(client.helper_status, "unsupported_helper_configuration")
        self.assertIsNone(managed.registry)

    async def test_missing_workflow_request_retains_original_client_options(self):
        managed = WorkflowSDKClient(self.factory, self.options, helpers_enabled=True)
        async with managed as client:
            self.assertIs(self.clients[-1].options, self.options)
            self.assertEqual(client.helper_status, "unsupported_workflow_request")

    async def test_invalid_enable_values_and_old_options_keep_helpers_inactive(self):
        with self.assertRaises(ValueError):
            self.managed(helpers_enabled="true")
        options = CacheOptions()
        async with self.managed(options=options, helpers_enabled=True) as client:
            self.assertIs(self.clients[-1].options, options)
            self.assertEqual(client.helper_status, "unsupported_sdk_options")

    @unittest.skipUnless(NODE, "The public source renderer requires Node.")
    async def test_default_contract_and_sdk_matcher_activate_for_an_explicit_public_request(self):
        from interface.test_native_source_contract import SOURCE, root_request

        managed = WorkflowSDKClient(self.factory, self.options, helpers_enabled=True,
            workflow_request=root_request(), source_root=SOURCE)
        self.assertFalse(hasattr(managed, "query"))
        with patch.dict(sys.modules, {"claude_agent_sdk": SimpleNamespace(HookMatcher=Matcher)}), \
                patch.dict(os.environ, {"PATH": str(Path(NODE).parent) + os.pathsep + os.defpath}):
            async with managed as client:
                self.assertEqual(client.helper_status, "active")
                self.assertIsNotNone(client.registry)
        self.assertEqual(self.calls, [])
