# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test native helper joins and fallback with synthetic SDK observations."""

import hashlib
import json
import shutil
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

from interface.native_cost_controls.helper_driver import Unsupported
from interface.native_cost_controls.native_helpers import (
    HelperMirror,
    HelperTransport,
    NativeHelperRegistry,
)
from interface.test_shared_tool_cache import request
from interface.test_system_envelope import synthetic_system_policy


@dataclass
class TaskStartedMessage:
    session_id: str = "fixture-session"
    tool_use_id: str = "root-tool"
    task_id: str = "root-task"
    task_type: str = "local_workflow"
    data: dict = field(default_factory=dict)


@dataclass
class TaskProgressMessage(TaskStartedMessage):
    pass


@dataclass
class UserMessage:
    content: list = field(default_factory=list)
    tool_use_result: dict = field(default_factory=dict)
    parent_tool_use_id: object = None


class Contract:
    """Use explicit synthetic tasks. This class never runs in production."""

    def __init__(self, root_request=None, source_root=None):
        self.root_request = root_request or {"scriptPath": "/synthetic/kernel_workflow.js", "args": {}}
        self.source_bindings = {}
        self.setup = None
        self.changed = False
        self.eligible = True

    def verify_sources(self):
        if self.changed:
            raise Unsupported("source_changed")

    def bind_setup(self, value, attestation):
        self.setup = value

    def resolve(self, label, task, schema):
        role = {"clock pre-r1": "clock_reader", "warm_start:resolve": "warm_start_resolver",
            "clock replan-r1": "clock_reader",
            "storage:reclaim r1": "storage_reclaim", "kb:cite": "citation_writer", "kb:write": "experience_writer"}[label]
        return {"eligible": self.eligible and task == label, "role": role, "command": "date +%s" if role == "clock_reader" else "fixture-only",
            "completion_marker": "STORAGE_RECLAIM_DONE round=1" if role == "storage_reclaim" else None}


class Upstream:
    def __init__(self):
        self.calls = []
        self.closed = False

    @contextmanager
    def send(self, method, target, headers, body):
        self.calls.append((method, target, headers, body))
        yield object()

    def close(self):
        self.closed = True


class NativeHelperTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.system_policy = synthetic_system_policy()
        self.system_policy.start()
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name)
        self.contract = Contract()
        self.registry = NativeHelperRegistry(self.workspace / "registry", "fixture-session", self.workspace,
            self.contract, wait_seconds=0.01)
        self.journal_dir = self.workspace / "native/projects/fixture-project/fixture-session/subagents/workflows/wf_fixture"
        self.journal_dir.mkdir(parents=True)
        self.journal_file = self.journal_dir / "journal.jsonl"
        self.journal_file.write_text("")
        self.started_agents = set()
        self.headers = [("Content-Type", "application/json"), ("x-claude-code-session-id", "fixture-session"),
            ("x-claude-code-agent-id", "clock-agent")]
        self.body = request()
        self.body["metadata"] = {"user_id": json.dumps({"session_id": "fixture-session"})}
        self.body["tools"].insert(0, {"name": "Bash", "input_schema": {"type": "object"}})
        self.body["messages"][0]["content"][0]["text"] = "clock pre-r1"
        self.upstream = Upstream()
        self.transport = HelperTransport(self.upstream, self.registry)

    def tearDown(self):
        self.system_policy.stop()
        self.registry.close()
        self.temporary.cleanup()

    def raw(self):
        return json.dumps(self.body, indent=2).encode()

    async def register(self, label="clock pre-r1", agent="clock-agent", index=1, task=None):
        await self.registry.pre({"session_id": "fixture-session", "tool_name": "Workflow",
            "tool_input": self.contract.root_request}, "root-tool")
        self.registry.observe(TaskStartedMessage())
        self.registry.observe(UserMessage(content=[{"tool_use_id": "root-tool", "is_error": False}],
            tool_use_result={"status": "async_launched", "taskType": "local_workflow", "taskId": "root-task",
                "scriptPath": self.contract.root_request["scriptPath"], "runId": "wf_fixture", "transcriptDir": str(self.journal_dir)}))
        if agent not in self.started_agents:
            self.append_journal("started", agent)
            self.started_agents.add(agent)
        self.registry.observe(TaskProgressMessage(data={"workflow_progress": [
            {"type": "workflow_phase", "index": 5, "title": "▸ kernel-lane"},
            {"type": "workflow_agent", "phaseIndex": 5, "index": index, "label": label,
                "agentId": agent, "attempt": 1, "state": "running"}]}))
        self.registry.mirror_append({"session_id": "fixture-session", "project_key": "fixture-project",
            "subpath": "subagents/workflows/wf_fixture/agent-" + agent}, [{"type": "user", "parentUuid": None,
                "agentId": agent, "sessionId": "fixture-session", "cwd": str(self.workspace), "uuid": agent,
                "message": {"content": task or label}}])

    def append_journal(self, kind, agent, value=None):
        record = {"type": kind, "key": "v2:" + hashlib.sha256(agent.encode()).hexdigest(), "agentId": agent}
        if kind == "result":
            record["result"] = value
        with self.journal_file.open("a") as handle:
            handle.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n")

    def terminal(self, agent, value, *, status="done", preview=None):
        node = self.registry.nodes[agent]
        raw = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
        if preview is None:
            units = raw.encode("utf-16-le")
            preview = units[:800].decode("utf-16-le", errors="surrogatepass") + "…" if len(units) > 800 else raw
        self.registry.observe(TaskProgressMessage(data={"workflow_progress": [
            {"type": "workflow_phase", "index": 5, "title": "▸ kernel-lane"},
            {"type": "workflow_agent", **node, "state": status, "lastToolName": "StructuredOutput", "resultPreview": preview}]}))

    def finish(self, agent, value):
        self.append_journal("result", agent, value)
        self.terminal(agent, value)

    def hook(self, block, **extra):
        return {"session_id": "fixture-session", "agent_id": self.headers[-1][1], "cwd": str(self.workspace),
            "tool_name": block["name"], "tool_input": block["input"], **extra}

    async def complete(self, native_stdout="42\n", *, finish=True):
        first = self.registry.response(self.headers, self.raw())
        self.assertEqual(first["name"], "Bash")
        self.assertEqual(await self.registry.pre(self.hook(first), first["id"]), {})
        await self.registry.post(self.hook(first, hook_event_name="PostToolUse",
            tool_response={"stdout": native_stdout, "stderr": "", "interrupted": False}), first["id"])
        self.body["messages"].append({"role": "assistant", "content": [first]})
        self.body["messages"].append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": first["id"], "content": native_stdout, "is_error": False}]})
        structured = self.registry.response(self.headers, self.raw())
        await self.registry.post(self.hook(structured, hook_event_name="PostToolUse"), structured["id"])
        if finish:
            self.finish(self.headers[-1][1], structured["input"])
        return first, structured

    async def test_clock_executes_via_native_hooks_and_replays_only_its_result(self):
        await self.register()
        first, structured = await self.complete()
        self.assertEqual(first["input"]["command"], "date +%s")
        self.assertEqual(structured["input"], {"epoch": 42})
        await self.register("clock pre-r1 (retry 1)", "retry-clock", 2, "clock pre-r1")
        self.headers[-1] = ("x-claude-code-agent-id", "retry-clock")
        self.body["messages"] = [{"role": "user", "content": [{"type": "text", "text": "clock pre-r1"}]}]
        self.assertEqual(self.registry.response(self.headers, self.raw())["name"], "StructuredOutput")
        driver = self.registry._driver("clock-agent")
        self.assertEqual(json.loads(driver.path.read_bytes())["bash_emissions"], 1)
        self.assertEqual(self.upstream.calls, [])

    async def test_all_five_roles_project_the_actual_native_result(self):
        cases = [("clock pre-r1", "42\n", {"epoch": 42}),
            ("warm_start:resolve", '{"candidates": []}', {"candidates": []}),
            ("storage:reclaim r1", "STORAGE_RECLAIM_DONE round=1\n", {"ok": True, "note": "reclaimed"}),
            ("kb:cite", '{"citations": 3}', {"filed": 3}),
            ("kb:write", '{"written": false}', {"written": False})]
        for index, (label, stdout, expected) in enumerate(cases, 1):
            with self.subTest(label=label):
                agent = "helper-" + str(index)
                await self.register(label, agent, index)
                self.headers[-1] = ("x-claude-code-agent-id", agent)
                self.body["messages"] = [{"role": "user", "content": [{"type": "text", "text": label}]}]
                _, result = await self.complete(stdout)
                self.assertEqual(result["input"], expected)

    async def test_request_cannot_register_itself_without_native_observations(self):
        raw = self.raw()
        with self.transport.send("POST", "/v1/messages", self.headers, raw):
            pass
        self.assertIs(self.upstream.calls[0][-1], raw)
        await self.register()
        self.assertIsNone(self.registry.response(self.headers, raw))
        self.assertEqual(self.registry.operations, {})

    async def test_unsupported_template_retains_exact_bytes_and_provider_ownership(self):
        await self.register()
        self.contract.eligible = False
        raw = self.raw()
        with self.transport.send("POST", "/v1/messages", self.headers, raw):
            pass
        self.assertIs(self.upstream.calls[0][-1], raw)
        self.contract.eligible = True
        self.assertIsNone(self.registry.response(self.headers, raw))

    async def test_denied_or_missing_result_cannot_repeat_a_local_action(self):
        await self.register()
        block = self.registry.response(self.headers, self.raw())
        await self.registry.pre(self.hook(block), block["id"])
        await self.registry.post(self.hook(block, hook_event_name="PostToolUseFailure", error="denied"), block["id"])
        with self.assertRaises(Unsupported), self.transport.send("POST", "/v1/messages", self.headers, self.raw()):
            pass
        self.assertEqual(self.upstream.calls, [])

    async def test_changed_local_arguments_receive_no_permission_grant(self):
        await self.register()
        block = self.registry.response(self.headers, self.raw())
        block["input"]["command"] = "changed-command"
        result = await self.registry.pre(self.hook(block), block["id"])
        self.assertEqual(result["hookSpecificOutput"]["permissionDecision"], "deny")
        with self.assertRaises(Unsupported):
            self.registry.response(self.headers, self.raw())

    async def test_changed_source_after_execution_prevents_provider_fallback(self):
        await self.register()
        await self.complete()
        self.registry.errors.append("source_changed")
        with self.assertRaises(Unsupported):
            self.registry.response(self.headers, self.raw())
        self.assertEqual(self.upstream.calls, [])

    async def test_local_stream_uses_explicit_local_origin_and_synthetic_zero_usage(self):
        await self.register()
        with self.transport.send("POST", "/v1/messages", self.headers, self.raw()) as response:
            events = [json.loads(line[6:]) for line in b"".join(response.iter_bytes()).splitlines() if line.startswith(b"data: ")]
            self.assertEqual(events[0]["message"]["model"], "local-deterministic-helper-v1")
            self.assertEqual(events[0]["message"]["usage"]["input_tokens"], 0)
            self.assertEqual(events[-1]["type"], "message_stop")
            self.assertEqual(dict(response.headers)["x-geak-local-origin"], "local_deterministic_driver")
        self.assertEqual(self.upstream.calls, [])

    async def test_missing_or_duplicate_child_identity_after_emission_cannot_forward(self):
        await self.register()
        block = self.registry.response(self.headers, self.raw())
        self.body["messages"].append({"role": "user", "content": [{"type": "tool_result",
            "tool_use_id": block["id"], "content": "42"}]})
        for headers in [self.headers[:-1], self.headers + [self.headers[-1]], [],
                [("x-claude-code-session-id", "different-session")]]:
            with self.subTest(headers=headers), self.assertRaises(Unsupported):
                self.registry.response(headers, self.raw())

    async def test_changed_encoding_after_local_emission_cannot_forward(self):
        await self.register()
        self.registry.response(self.headers, self.raw())
        with self.assertRaises(Unsupported), self.transport.send("POST", "/v1/messages",
                self.headers + [("Content-Encoding", "gzip")], b"compressed-fixture"):
            pass
        self.assertEqual(self.upstream.calls, [])

    async def test_unsupported_retry_alias_cannot_reopen_provider_fallback(self):
        await self.register()
        self.registry.response(self.headers, self.raw())
        await self.register("clock pre-r1 (retry 1)", "retry-agent", 2)
        self.headers[-1] = ("x-claude-code-agent-id", "retry-agent")
        self.body["messages"][0]["content"][0]["text"] = "clock pre-r1 (retry 1)"
        with self.assertRaises(Unsupported), self.transport.send("POST", "/v1/messages", self.headers, self.raw()):
            pass
        self.assertEqual(self.upstream.calls, [])

    async def test_malformed_retry_schema_cannot_reopen_provider_fallback(self):
        await self.register()
        self.registry.response(self.headers, self.raw())
        await self.register("clock pre-r1 (retry 1)", "retry-agent", 2)
        self.headers[-1] = ("x-claude-code-agent-id", "retry-agent")
        self.body["tools"] = []
        with self.assertRaises(Unsupported), self.transport.send("POST", "/v1/messages", self.headers, self.raw()):
            pass
        self.assertEqual(self.upstream.calls, [])

    async def test_mirror_preserves_original_store_and_its_optional_methods(self):
        class Store:
            def __init__(self):
                self.values = []
            async def append(self, key, entries):
                self.values.append((key, entries))
            async def load(self, key):
                return self.values
            async def list_subkeys(self, key):
                return [key]
        original = Store()
        mirror = HelperMirror(self.registry, original)
        key, entries = {"session_id": "other"}, []
        await mirror.append(key, entries)
        self.assertIs(original.values[0][0], key)
        self.assertIs(original.values[0][1], entries)
        self.assertIs(await mirror.load(key), original.values)
        self.assertEqual(await mirror.list_subkeys(key), [key])

    async def test_provider_setup_result_binds_the_contract_only_after_native_acceptance(self):
        await self.register("director:setup", "setup-agent")
        self.headers[-1] = ("x-claude-code-agent-id", "setup-agent")
        self.body["messages"][0]["content"][0]["text"] = "director:setup"
        self.assertIsNone(self.registry.response(self.headers, self.raw()))
        block = {"name": "StructuredOutput", "input": {"eval_dir": "/synthetic/experiment"}}
        await self.registry.post(self.hook(block, hook_event_name="PostToolUseFailure"), "setup-output")
        self.assertIsNone(self.contract.setup)
        await self.registry.post(self.hook(block, hook_event_name="PostToolUse"), "setup-output")
        self.assertEqual(self.contract.setup, block["input"])

    async def test_replan_clock_requires_a_provider_owned_replan_anchor(self):
        await self.register("tech_lead:replan r1#1", "planner", 1)
        self.headers[-1] = ("x-claude-code-agent-id", "planner")
        self.body["messages"][0]["content"][0]["text"] = "tech_lead:replan r1#1"
        self.assertIsNone(self.registry.response(self.headers, self.raw()))
        await self.register("clock replan-r1", "clock", 2)
        self.headers[-1] = ("x-claude-code-agent-id", "clock")
        self.body["messages"][0]["content"][0]["text"] = "clock replan-r1"
        self.assertEqual(self.registry.response(self.headers, self.raw())["name"], "Bash")

    async def test_bad_progress_before_emission_preserves_exact_provider_request(self):
        await self.register()
        self.registry.observe(TaskProgressMessage(data={"workflow_progress": "invalid"}))
        raw = self.raw()
        with self.transport.send("POST", "/v1/messages", self.headers, raw):
            pass
        self.assertIs(self.upstream.calls[0][-1], raw)
        self.assertEqual(self.registry.operations, {})

    async def test_bad_initial_mirror_before_emission_preserves_provider_request(self):
        await self.register()
        self.registry.mirror_append({"session_id": "fixture-session", "subpath": "subagents/workflows/root-task/agent-clock-agent"},
            [{"type": "user", "parentUuid": None, "cwd": "/wrong/workspace", "message": {"content": "clock pre-r1"}}])
        self.assertIsNone(self.registry.response(self.headers, self.raw()))
        self.assertEqual(self.registry.operations, {})

    async def test_wrong_hook_workspace_and_extra_local_tool_cannot_execute(self):
        await self.register()
        block = self.registry.response(self.headers, self.raw())
        for data in [self.hook(block, cwd="/wrong/workspace"), self.hook(block, tool_name="Read")]:
            with self.subTest(data=data):
                reply = await self.registry.pre(data, block["id"])
                self.assertEqual(reply["hookSpecificOutput"]["permissionDecision"], "deny")

    async def test_changed_native_result_input_latches_a_failed_continuation(self):
        await self.register()
        block = self.registry.response(self.headers, self.raw())
        await self.registry.pre(self.hook(block), block["id"])
        changed = dict(block, input={**block["input"], "command": "rewritten-native-command"})
        await self.registry.post(self.hook(changed, hook_event_name="PostToolUse",
            tool_response={"stdout": "42\n", "interrupted": False}), block["id"])
        self.assertEqual(self.registry.errors, ["native_helper_outcome_conflict"])
        with self.assertRaises(Unsupported):
            self.registry.response(self.headers, self.raw())

    async def test_native_schema_rejection_prevents_another_local_emission(self):
        await self.register()
        _, structured = await self.complete()
        await self.registry.post(self.hook(structured, hook_event_name="PostToolUseFailure"), structured["id"])
        with self.assertRaises(Unsupported):
            self.registry.response(self.headers, self.raw())

    async def test_observer_ignores_unrelated_sessions_and_non_workflow_transcripts(self):
        self.registry.observe(object())
        self.registry.observe(TaskStartedMessage(tool_use_id="unrelated"))
        self.registry.mirror_append({"session_id": "fixture-session", "subpath": "subagents/agent-unrelated"}, [])
        self.assertEqual(self.registry.nodes, {})
        self.assertEqual(self.registry.mirrors, {})
        self.assertEqual(self.registry.errors, [])
        mirror = HelperMirror(self.registry)
        self.assertIsNone(await mirror.load({}))
        self.assertFalse(hasattr(mirror, "list_subkeys"))

    async def test_unattributed_root_and_invalid_pre_emission_requests_preserve_bytes(self):
        for headers, raw in [([], self.raw()), (self.headers[:-1], b'{"tools":[]}'),
                (self.headers + [self.headers[-1]], self.raw()), (self.headers, b'[]'),
                (self.headers, b'{"tools":[],"tools":[]}')]:
            with self.subTest(headers=headers, raw=raw):
                self.assertIsNone(self.registry.response(headers, raw))
        self.assertEqual(self.registry.operations, {})

    async def test_invalid_json_after_local_emission_cannot_reach_provider(self):
        await self.register()
        self.registry.response(self.headers, self.raw())
        for raw in [b'{"unreadable":', b'[]', b'{"tools":[],"tools":[]}']:
            with self.subTest(raw=raw), self.assertRaises(Unsupported):
                self.registry.response([], raw)

    async def test_root_request_after_local_emission_still_reaches_provider(self):
        await self.register()
        self.registry.response(self.headers, self.raw())
        self.assertIsNone(self.registry.response(self.headers[:-1], b'{"tools":[]}'))

    async def test_pending_and_unrelated_workflow_nodes_cannot_register_helpers(self):
        await self.register()
        self.registry.observe(TaskProgressMessage(data={"workflow_progress": [
            {"type": "workflow_phase", "index": 5, "title": "▸ kernel-lane"},
            {"type": "workflow_agent", "phaseIndex": 5, "state": "queued"},
            {"type": "workflow_agent", "phaseIndex": 9, "agentId": "other"}]}))
        self.registry.mirror_append({"session_id": "fixture-session", "subpath": "subagents/workflows/root-task/agent-clock-agent"},
            [{"type": "assistant", "message": {"content": "This cannot replace the initial task."}}])
        self.assertEqual(set(self.registry.nodes), {"clock-agent"})
        self.assertEqual(self.registry.mirrors["clock-agent"]["task"], "clock pre-r1")

    async def test_changed_native_node_identity_after_local_emission_stops_unknown_alias(self):
        await self.register()
        self.registry.response(self.headers, self.raw())
        await self.register("clock pre-r1", "conflicting-alias", 1)
        self.headers[-1] = ("x-claude-code-agent-id", "conflicting-alias")
        with self.assertRaises(Unsupported):
            self.registry.response(self.headers, self.raw())

    async def test_source_read_failure_returns_explicit_native_denial(self):
        await self.register()
        block = self.registry.response(self.headers, self.raw())
        driver = self.registry._driver("clock-agent")
        with patch.object(driver, "verify", side_effect=OSError("A synthetic source read failure.")):
            result = await self.registry.pre(self.hook(block), block["id"])
        self.assertEqual(result["hookSpecificOutput"]["permissionDecision"], "deny")
        self.assertNotIn("synthetic source", str(result))

    async def test_native_result_write_failure_prevents_provider_fallback(self):
        await self.register()
        block = self.registry.response(self.headers, self.raw())
        await self.registry.pre(self.hook(block), block["id"])
        driver = self.registry._driver("clock-agent")
        with patch.object(driver, "capture_bash", side_effect=OSError("A synthetic ledger write failure.")):
            await self.registry.post(self.hook(block, hook_event_name="PostToolUse",
                tool_response={"stdout": "42\n", "interrupted": False}), block["id"])
        with self.assertRaises(Unsupported):
            self.registry.response(self.headers, self.raw())

    async def test_acknowledgement_without_native_done_cannot_replay(self):
        await self.register()
        await self.complete(finish=False)
        driver = self.registry._driver("clock-agent")
        self.assertEqual(json.loads(driver.path.read_bytes())["stage"], "ACKNOWLEDGED")
        with self.assertRaises(Unsupported):
            self.registry.response(self.headers, self.raw())
        self.assertEqual(json.loads(driver.path.read_bytes())["bash_emissions"], 1)

    async def test_done_without_a_full_journal_result_stays_non_replayable(self):
        await self.register()
        await self.complete(finish=False)
        self.terminal("clock-agent", {"epoch": 42})
        driver = self.registry._driver("clock-agent")
        self.assertEqual(json.loads(driver.path.read_bytes())["stage"], "ACKNOWLEDGED")
        self.append_journal("result", "clock-agent", {"epoch": 42})
        self.registry.refresh_native_completions()
        self.assertEqual(json.loads(driver.path.read_bytes())["stage"], "COMPLETE")

    async def test_native_error_after_acknowledgement_latches_unknown(self):
        await self.register()
        await self.complete(finish=False)
        self.terminal("clock-agent", {"epoch": 42}, status="error")
        driver = self.registry._driver("clock-agent")
        self.assertEqual(json.loads(driver.path.read_bytes())["stage"], "UNKNOWN")

    async def test_conflicting_native_preview_cannot_override_full_value(self):
        await self.register()
        await self.complete(finish=False)
        self.append_journal("result", "clock-agent", {"epoch": 42})
        self.terminal("clock-agent", {"epoch": 43})
        driver = self.registry._driver("clock-agent")
        self.assertEqual(json.loads(driver.path.read_bytes())["stage"], "UNKNOWN")

    async def test_full_large_result_can_complete_despite_a_truncated_preview(self):
        await self.register("warm_start:resolve", "resolver", 1)
        self.headers[-1] = ("x-claude-code-agent-id", "resolver")
        self.body["messages"][0]["content"][0]["text"] = "warm_start:resolve"
        value = {"payload": "a" * 700}
        await self.complete(json.dumps(value), finish=False)
        self.finish("resolver", value)
        driver = self.registry._driver("resolver")
        self.assertEqual(json.loads(driver.path.read_bytes())["stage"], "COMPLETE")

    async def test_matching_preview_prefix_cannot_hide_a_changed_full_result(self):
        await self.register("warm_start:resolve", "resolver", 1)
        self.headers[-1] = ("x-claude-code-agent-id", "resolver")
        self.body["messages"][0]["content"][0]["text"] = "warm_start:resolve"
        await self.complete(json.dumps({"payload": "a" * 700}), finish=False)
        self.finish("resolver", {"payload": "a" * 500 + "changed"})
        driver = self.registry._driver("resolver")
        self.assertEqual(json.loads(driver.path.read_bytes())["stage"], "UNKNOWN")

    async def test_wrong_root_tool_result_cannot_change_the_journal_descriptor(self):
        await self.register()
        original = dict(self.registry.journal_descriptor)
        self.registry.observe(UserMessage(content=[{"tool_use_id": "root-tool", "is_error": False}],
            tool_use_result={"status": "async_launched", "taskType": "local_workflow", "taskId": "wrong-task",
                "scriptPath": self.contract.root_request["scriptPath"], "runId": "wf_fixture", "transcriptDir": str(self.journal_dir)}))
        self.assertEqual(self.registry.journal_descriptor, original)
        self.assertIsNone(self.registry.response(self.headers, self.raw()))
        self.assertEqual(self.registry.operations, {})

    async def test_later_missing_terminal_preview_invalidates_prior_completion(self):
        await self.register()
        await self.complete()
        node = self.registry.nodes["clock-agent"]
        self.registry.observe(TaskProgressMessage(data={"workflow_progress": [
            {"type": "workflow_phase", "index": 5, "title": "▸ kernel-lane"},
            {"type": "workflow_agent", **node, "state": "done", "lastToolName": "StructuredOutput", "resultPreview": None}]}))
        driver = self.registry._driver("clock-agent")
        self.assertEqual(json.loads(driver.path.read_bytes())["stage"], "UNKNOWN")
        with self.assertRaises(Unsupported):
            self.registry.response(self.headers, self.raw())
        self.assertEqual(json.loads(driver.path.read_bytes())["bash_emissions"], 1)

    @unittest.skipUnless(shutil.which("node"), "The public source renderer requires Node.")
    async def test_public_source_contract_joins_native_setup_and_clock_dispatch(self):
        from interface.native_cost_controls.source_contract import (
            KernelWorkflowContract,
        )
        from interface.test_native_source_contract import SOURCE, root_request

        self.contract = KernelWorkflowContract(root_request(exp_root=str(self.workspace / "experiment")), SOURCE)
        self.registry.close()
        self.registry = NativeHelperRegistry(self.workspace / "public-registry", "fixture-session", self.workspace,
            self.contract, wait_seconds=0.01)
        await self.register("director:setup", "setup", 1)
        self.headers[-1] = ("x-claude-code-agent-id", "setup")
        self.body["messages"][0]["content"][0]["text"] = "director:setup"
        self.assertIsNone(self.registry.response(self.headers, self.raw()))
        setup = {"eval_dir": str(self.workspace / "experiment/run"),
            "workspace": str(self.workspace / "experiment/run/workspace"), "kernel_name": "public_fixture_task"}
        await self.registry.post(self.hook({"name": "StructuredOutput", "input": setup},
            hook_event_name="PostToolUse"), "native-setup-result")
        captured = self.contract._render("render", "clock", {"round": 1, "tag": "pre-r1"})
        await self.register(captured["options"]["label"], "clock", 2, captured["task"])
        self.headers[-1] = ("x-claude-code-agent-id", "clock")
        self.body["messages"][0]["content"][0]["text"] = captured["task"]
        self.body["tools"][-1]["input_schema"] = captured["options"]["schema"]
        first, result = await self.complete()
        self.assertEqual(first["input"]["command"], "date +%s")
        self.assertEqual(result["input"], {"epoch": 42})
        self.assertEqual(self.upstream.calls, [])
