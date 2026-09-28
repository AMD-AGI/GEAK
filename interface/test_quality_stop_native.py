# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test stopping authority with local files and synthetic native events."""

import asyncio
import gzip
import hashlib
import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

from interface.native_cost_controls.quality_stop_controller import (
    PREFIX,
    PROTOCOL,
    SCHEMA,
    StopRejected,
    canonical,
)
from interface.native_cost_controls.quality_stop_native import (
    CheckpointResponse,
    CheckpointTransport,
    NativeProducerCensus,
    _object,
)


@dataclass
class TaskStartedMessage:
    session_id: str = "stop-session"
    tool_use_id: str = "root-tool"
    task_id: str = "root-task"
    task_type: str = "local_workflow"
    data: dict = field(default_factory=dict)


@dataclass
class TaskProgressMessage(TaskStartedMessage):
    pass


@dataclass
class TaskNotificationMessage(TaskStartedMessage):
    status: str = "completed"


@dataclass
class TaskUpdatedMessage(TaskStartedMessage):
    status: str = "completed"
    patch: dict = field(default_factory=lambda: {"status": "completed"})


@dataclass
class UserMessage:
    content: list = field(default_factory=list)
    tool_use_result: dict = field(default_factory=dict)
    parent_tool_use_id: object = None


@dataclass
class AssistantMessage:
    content: list = field(default_factory=list)
    parent_tool_use_id: object = None
    session_id: str = "stop-session"


class Controller:
    """Keep the test independent of timing, signing, and GPU adapters."""

    def __init__(self):
        self.public_config = {"protocol": PROTOCOL, "trial_id": "fixture_trial_001",
                              "public_key": {"kty": "RSA", "n": "fixture", "e": "AQAB"},
                              "candidate_root": "/fixture/candidate"}
        self.calls = []
        self.envelope = {"payload": "fixture-payload", "signature": "fixture-signature"}
        self.callback = None

    def checkpoint(self, task, binding, *, census):
        self.calls.append((task, binding))
        census()
        if self.callback:
            self.callback()
        census()
        return deepcopy(self.envelope)

    def bind_native_closure(self, callback):
        self.closure = callback


class NativeFixture:
    """Create one complete producer census around a pending checkpoint."""

    def __init__(self, directory, *, populate=True):
        self.root = Path(directory)
        self.source = self.root / "workflow"
        self.source.mkdir()
        for name in ("kernel_workflow.js", "kernel_lane.js", "quality_stop_verify.js"):
            (self.source / name).write_text("// isolated synthetic source\n")
        self.controller = Controller()
        self.request = {"scriptPath": str(self.source / "kernel_workflow.js"), "args": {
            "mode": "optimize", "workflow_dir": str(self.source),
            "kernel_lane_script": str(self.source / "kernel_lane.js"),
            "quality_stop": deepcopy(self.controller.public_config)}}
        self.registry = NativeProducerCensus(session_id="stop-session", root_request=self.request,
            source_root=self.source, workspace=self.root, controller=self.controller, wait_seconds=0)
        self.registry.bind_root_prompt("Synthetic host root prompt.")
        self.task = PREFIX + canonical({"protocol": PROTOCOL, "trial_id": "fixture_trial_001",
            "stage": "boundary", "look_index": 1, "round": 1, "dispatched": 2,
            "budget": 6, "no_improve": 0, "max_no_improve": 2, "forced_replans": 0,
            "deadline_epoch": 2000000000, "candidate_root": "/fixture/candidate"})
        self.reclaim_command = "printf 'STORAGE_RECLAIM_DONE round=1\\n'"
        self.reclaim_task = "Reclaim storage.\n```bash\n" + self.reclaim_command + "\n```"
        self.headers = [("Content-Type", "application/json"),
            ("x-claude-code-session-id", "stop-session"), ("x-claude-code-agent-id", "checkpoint")]
        self.body = {"tools": [{"name": "StructuredOutput", "input_schema": deepcopy(SCHEMA)}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": self.task}]}], "stream": False}
        self.directory = self.root / "projects/fixture/stop-session/subagents/workflows/wf_fixture"
        self.directory.mkdir(parents=True)
        self.journal = self.directory / "journal.jsonl"
        self.journal.write_text("")
        self.nodes = []
        if populate:
            self.start()
            self.add("engineer", "engineer:test", {"ok": True})
            self.add("reclaim", "storage:reclaim r1", {"done": True}, task=self.reclaim_task)
            self.add("checkpoint", "quality_stop:boundary r1 l1", task=self.task)
            self.progress()
            self.hook("PreToolUse", "Bash", {"command": self.reclaim_command}, "reclaim-tool", "reclaim")
            self.hook("PostToolUse", "Bash", {"command": self.reclaim_command}, "reclaim-tool", "reclaim",
                response={"stdout": "STORAGE_RECLAIM_DONE round=1\n", "stderr": "", "interrupted": False})

    def start(self):
        self.registry.observe(AssistantMessage(content=[{"id": "root-tool", "name": "Workflow", "input": self.request}]))
        self.hook("PreToolUse", "Workflow", self.request, "root-tool")
        self.registry.observe(TaskStartedMessage())
        self.registry.observe(self.root_result())

    def root_result(self):
        return UserMessage(content=[{"tool_use_id": "root-tool", "is_error": False}], tool_use_result={
            "status": "async_launched", "taskType": "local_workflow", "taskId": "root-task",
            "scriptPath": self.request["scriptPath"], "runId": "wf_fixture", "transcriptDir": str(self.directory)})

    def append(self, kind, agent, result=None):
        row = {"type": kind, "key": "v2:" + hashlib.sha256(agent.encode()).hexdigest(), "agentId": agent}
        if kind == "result":
            row["result"] = result
        with self.journal.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")

    def add(self, agent, label, result=None, *, task="Synthetic native task."):
        self.append("started", agent)
        node = {"type": "workflow_agent", "phaseIndex": 1, "index": len(self.nodes) + 1,
                "attempt": 1, "agentId": agent, "label": label, "state": "running"}
        self.nodes.append(node)
        self.mirror(agent, task)
        if result is not None:
            self.finish(agent, result)
        return node

    def finish(self, agent, result):
        self.append("result", agent, result)
        node = next(node for node in self.nodes if node.get("agentId") == agent)
        raw = json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        units = raw.encode("utf-16-le", errors="surrogatepass")
        node.update(state="done", lastToolName="StructuredOutput",
            resultPreview=units[:800].decode("utf-16-le", errors="surrogatepass") + "…" if len(units) > 800 else raw)

    def mirror(self, agent, task, **changes):
        entry = {"type": "user", "parentUuid": None, "agentId": agent,
            "sessionId": "stop-session", "cwd": str(self.root), "message": {"content": task}}
        entry.update(changes)
        self.registry.mirror_append({"session_id": "stop-session", "project_key": "fixture",
            "subpath": "subagents/workflows/wf_fixture/agent-" + agent}, [entry])

    def progress(self, **changes):
        event = TaskProgressMessage(data={"workflow_progress": deepcopy(self.nodes)})
        for key, value in changes.items():
            setattr(event, key, value)
        self.registry.observe(event)

    def hook(self, event, name, value, identity, agent=None, *, response=None, **changes):
        data = {"session_id": "stop-session", "cwd": str(self.root), "hook_event_name": event,
                "tool_name": name, "tool_input": value, "agent_id": agent, "tool_response": response}
        data.update(changes)
        return asyncio.run((self.registry.pre if event == "PreToolUse" else self.registry.post)(data, identity, {}))

    def response(self):
        return self.registry.response(self.headers, canonical(self.body).encode())


class NativeCensusTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = NativeFixture(self.temporary.name)
        self.registry = self.fixture.registry
        self.addCleanup(self.registry.close)

    def rejected(self, code):
        return self.assertRaisesRegex(StopRejected, "^" + code + "$")

    def test_complete_journal_and_native_terminals_allow_checkpoint(self):
        self.fixture.nodes.insert(0, {"type": "workflow_phase", "index": 1, "title": "kernel lane"})
        self.fixture.progress()
        self.registry.assert_quiescent("checkpoint")
        block = self.fixture.response()
        self.assertEqual(block["input"], self.fixture.controller.envelope)
        self.assertEqual(block["name"], "StructuredOutput")
        self.assertEqual(len(self.fixture.controller.calls), 1)
        self.assertEqual(self.fixture.controller.calls[0][1], {"session_id": "stop-session", "run_id": "wf_fixture",
            "root_tool": "root-tool", "root_task": "root-task", "agent_id": "checkpoint"})
        block["input"]["payload"] = "caller changed copy"
        self.assertEqual(self.registry.emissions["checkpoint"]["block"]["input"], self.fixture.controller.envelope)

    def test_queued_producer_without_agent_id_blocks_certificate(self):
        self.fixture.nodes.append({"type": "workflow_agent", "index": 4, "attempt": 1,
                                  "phaseIndex": 1, "label": "queued", "state": "queued"})
        self.fixture.progress()
        with self.rejected("native_queued_producer_unknown"):
            self.registry.assert_quiescent("checkpoint")

    def test_complete_bytes_do_not_prove_a_producer_finished(self):
        self.fixture.add("late", "engineer:late")
        self.fixture.progress()
        self.assertEqual(self.registry.journal.snapshot()["status"], "complete")
        with self.rejected("native_producer_result_missing"):
            self.registry.assert_quiescent("checkpoint")

    def test_started_and_observed_producer_sets_must_match(self):
        self.fixture.append("started", "unobserved")
        with self.rejected("native_producer_sets_differ"):
            self.registry.assert_quiescent("checkpoint")

    def test_journal_success_requires_independent_done_event(self):
        self.fixture.add("late", "engineer:late")
        self.fixture.append("result", "late", {"ok": True})
        self.fixture.progress()
        with self.rejected("native_producer_not_successful_terminal"):
            self.registry.assert_quiescent("checkpoint")

    def test_journal_error_cannot_authorize_stopping(self):
        node = self.fixture.add("late", "engineer:late")
        self.fixture.append("error", "late")
        node["state"] = "done"
        self.fixture.progress()
        with self.rejected("native_producer_not_successful_terminal"):
            self.registry.assert_quiescent("checkpoint")

    def test_result_preview_and_structured_output_must_match(self):
        for name, value in (("resultPreview", '{"ok":false}'), ("lastToolName", "Bash")):
            with self.subTest(field=name):
                original = self.registry.nodes[(1, 1, 1)][name]
                self.registry.nodes[(1, 1, 1)][name] = value
                with self.rejected("native_terminal_result_changed"):
                    self.registry.assert_quiescent("checkpoint")
                self.registry.nodes[(1, 1, 1)][name] = original

    def test_preview_uses_native_utf16_limit(self):
        self.fixture.add("long", "engineer:long", {"text": "x" * 390 + "😀" * 20})
        self.fixture.progress()
        self.registry.assert_quiescent("checkpoint")

    def test_partial_journal_and_already_terminal_checkpoint_are_rejected(self):
        with self.journal_partial(), self.rejected("native_journal_not_stable"):
            self.registry.assert_quiescent("checkpoint")
        self.fixture.finish("checkpoint", {"model": "claimed certificate"})
        self.fixture.progress()
        with self.rejected("checkpoint_already_terminal"):
            self.registry.assert_quiescent("checkpoint")

    @contextmanager
    def journal_partial(self):
        raw = self.fixture.journal.read_bytes()
        self.fixture.journal.write_bytes(raw[:-1])
        try:
            yield
        finally:
            self.fixture.journal.write_bytes(raw)

    def test_checkpoint_journal_result_is_not_a_running_checkpoint(self):
        self.fixture.append("result", "checkpoint", {"claimed": True})
        with self.rejected("checkpoint_journal_state_invalid"):
            self.registry.assert_quiescent("checkpoint")

    def test_unknown_agent_and_duplicate_agent_node_are_rejected(self):
        with self.rejected("native_agent_node_missing_or_duplicate"):
            self.registry.assert_quiescent("unknown")
        self.fixture.nodes.append({**self.fixture.nodes[-1], "index": 4})
        self.fixture.progress()
        with self.rejected("native_agent_node_missing_or_duplicate"):
            self.registry.assert_quiescent("checkpoint")

    def test_active_tool_blocks_quiescence_until_native_outcome(self):
        self.fixture.hook("PreToolUse", "Write", {"file_path": "candidate"}, "write", "engineer")
        with self.rejected("native_background_producer_active"):
            self.registry.assert_quiescent("checkpoint")
        self.fixture.hook("PostToolUse", "Write", {"file_path": "candidate"}, "write", "engineer", response={})
        self.registry.assert_quiescent("checkpoint")

    def test_background_task_requires_terminal_notification(self):
        self.fixture.hook("PreToolUse", "Bash", {"command": "true"}, "bash", "engineer")
        self.registry.observe(TaskStartedMessage(task_id="b00000001", tool_use_id="bash", task_type="local_bash"))
        with self.rejected("native_background_producer_active"):
            self.registry.assert_quiescent("checkpoint")
        self.registry.observe(TaskUpdatedMessage(task_id="b00000001", tool_use_id="bash", task_type="local_bash",
                                                status="running", patch={"status": "running"}))
        self.registry.observe(TaskNotificationMessage(task_id="b00000001", tool_use_id="bash", task_type="local_bash", status="completed"))
        self.fixture.hook("PostToolUse", "Bash", {"command": "true"}, "bash", "engineer", response={})
        self.registry.assert_quiescent("checkpoint")

    def test_duplicate_background_start_fails_closed(self):
        self.fixture.hook("PreToolUse", "Bash", {"command": "true"}, "bash", "engineer")
        event = TaskStartedMessage(task_id="b00000001", tool_use_id="bash", task_type="local_bash")
        self.registry.observe(event)
        self.registry.observe(event)
        self.assertEqual(self.registry.error, "native_census_event_conflict")

    def test_progress_conflicts_fail_closed(self):
        cases = [None, [None], [{"type": "workflow_agent", "index": True, "attempt": 1, "label": "bad"}],
                 [{**self.fixture.nodes[0], "cached": True}], self.fixture.nodes + [self.fixture.nodes[0]],
                 self.fixture.nodes[1:], [{**self.fixture.nodes[0], "label": "changed"}, *self.fixture.nodes[1:]],
                 [{**self.fixture.nodes[0], "state": "running"}, *self.fixture.nodes[1:]]]
        for progress in cases:
            with self.subTest(progress=progress):
                self.registry.error = None
                self.registry.observe(TaskProgressMessage(data={"workflow_progress": progress}))
                self.assertEqual(self.registry.error, "native_census_event_conflict")

    def test_event_session_root_task_and_tool_are_bound(self):
        for changes in ({"session_id": "other"}, {"task_id": "other"}, {"tool_use_id": "other"}):
            with self.subTest(changes=changes):
                self.registry.error = None
                self.fixture.progress(**changes)
                self.assertEqual(self.registry.error, "native_census_event_conflict")

    def test_changed_mirror_fails_closed(self):
        self.fixture.mirror("checkpoint", "Changed task.")
        self.assertEqual(self.registry.error, "native_census_mirror_conflict")

    def test_unrelated_mirrors_and_nested_messages_are_ignored(self):
        for key in ({"session_id": "other", "subpath": "x"}, {"session_id": "stop-session"},
                    {"session_id": "stop-session", "subpath": "not-a-workflow"}):
            self.registry.mirror_append(key, [None])
        self.fixture.mirror("checkpoint", "Ignored nested task.", parentUuid="parent")
        self.assertIsNone(self.registry.error)
        self.registry.assert_quiescent("checkpoint")

    def test_mirror_workspace_identity_and_task_are_required(self):
        for changes in ({"cwd": "/other"}, {"agentId": "other"}, {"sessionId": "other"}, {"message": {}}):
            with self.subTest(changes=changes):
                self.registry.error = None
                self.fixture.mirror("checkpoint", self.fixture.task, **changes)
                self.assertEqual(self.registry.error, "native_census_mirror_conflict")

    def test_changed_workflow_source_is_rejected_before_controller_call(self):
        (self.fixture.source / "kernel_lane.js").write_text("// changed\n")
        with self.rejected("stopping_workflow_source_changed"):
            self.fixture.response()
        self.assertEqual(self.fixture.controller.calls, [])

    def test_duplicate_checkpoint_delivery_never_repeats_controller(self):
        self.fixture.response()
        with self.rejected("checkpoint_continuation_forbidden"):
            self.fixture.response()
        self.assertEqual(len(self.fixture.controller.calls), 1)

    def test_producer_arriving_during_measurement_invalidates_local_output(self):
        def late_producer():
            self.fixture.add("late", "engineer:late")
            self.fixture.progress()
        self.fixture.controller.callback = late_producer
        with self.rejected("native_census_unavailable"):
            self.fixture.response()
        self.assertEqual(self.registry.error, "native_census_event_conflict")
        self.assertEqual(self.registry.emissions["checkpoint"], {"pending": True, "accepted": False})
        with self.rejected("native_census_unavailable"):
            self.fixture.response()

    def test_event_lock_is_released_during_controller_measurement(self):
        events = []
        def event_during_measurement():
            thread = threading.Thread(target=lambda: (self.registry.observe(TaskProgressMessage(
                data={"workflow_progress": deepcopy(self.fixture.nodes)})), events.append("observed")))
            thread.start()
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
        self.fixture.controller.callback = event_during_measurement
        self.fixture.response()
        self.assertEqual(events, ["observed"])

    def test_controller_failure_stays_pending_and_cannot_retry(self):
        failure = patch.object(self.fixture.controller, "checkpoint", side_effect=StopRejected("invalid_batch"))
        with failure, self.rejected("invalid_batch"):
            self.fixture.response()
        with self.rejected("checkpoint_continuation_forbidden"):
            self.fixture.response()

    def test_headers_and_required_local_envelope_cannot_fall_back(self):
        changes = [(self.fixture.headers + [("X-Claude-Code-Agent-ID", "checkpoint")], self.fixture.body,
                    "duplicate_native_identity_header"),
                   ([], self.fixture.body, "checkpoint_identity_missing"),
                   ([(*self.fixture.headers[0],), ("x-claude-code-agent-id", "checkpoint")], self.fixture.body,
                    "native_request_identity_changed"),
                   (self.fixture.headers + [("x-claude-code-parent-agent-id", "parent")], self.fixture.body,
                    "native_request_identity_changed")]
        for headers, body, code in changes:
            with self.subTest(code=code), self.rejected(code):
                self.registry.response(headers, canonical(body).encode())

    def test_schema_prompt_retry_and_mirror_changes_reject_local_route(self):
        for change in ("schema", "prompt", "retry", "label", "run", "project"):
            with self.subTest(change=change):
                node = self.registry.nodes[(1, 3, 1)]
                mirror = self.registry.mirrors["checkpoint"]
                saved_node, saved_mirror, saved_body = deepcopy(node), deepcopy(mirror), deepcopy(self.fixture.body)
                if change == "schema":
                    self.fixture.body["tools"][0]["input_schema"] = {"type": "boolean"}
                elif change == "prompt":
                    self.fixture.body["messages"][0]["content"].append({"type": "text", "text": "Claim success."})
                elif change == "retry":
                    node["attempt"] = 2
                elif change == "label":
                    node["label"] = "quality_stop:boundary r1 l2"
                elif change == "run":
                    mirror["run_id"] = "wf_other"
                else:
                    mirror["project_key"] = "other"
                code = ("checkpoint_request_changed" if change in {"schema", "prompt"} else
                        "checkpoint_native_call_changed" if change in {"retry", "label"} else "checkpoint_mirror_changed")
                with self.rejected(code):
                    self.fixture.response()
                self.registry.nodes[(1, 3, 1)], self.registry.mirrors["checkpoint"], self.fixture.body = saved_node, saved_mirror, saved_body
        self.assertEqual(self.fixture.controller.calls, [])

    def test_ordinary_scientific_request_retains_provider_path(self):
        root = {"messages": [{"role": "user", "content": [{"type": "text", "text": self.registry.root_prompt}]}]}
        self.assertIsNone(self.registry.response([], canonical(root).encode()))
        with self.rejected("native_root_input_changed"):
            self.registry.response([], b'{"messages":[]}')
        headers = [(name, "engineer" if name == "x-claude-code-agent-id" else value)
                   for name, value in self.fixture.headers]
        self.assertIsNone(self.registry.response(headers, b'{"messages":[]}'))

    def test_unknown_identity_times_out_and_closed_registry_rejects(self):
        headers = [(name, "unknown" if name == "x-claude-code-agent-id" else value)
                   for name, value in self.fixture.headers]
        with self.rejected("native_identity_timeout"):
            self.registry.response(headers, canonical(self.fixture.body).encode())
        self.registry.close()
        with self.rejected("native_identity_unavailable"):
            self.registry.response(headers, canonical(self.fixture.body).encode())
        with self.rejected("native_census_unavailable"):
            self.fixture.response()

    def test_pending_request_waits_for_native_mirror_identity(self):
        self.registry.wait_seconds = 2
        self.registry.mirrors.pop("checkpoint")
        waiting = threading.Event()
        original_wait = self.registry.condition.wait
        def wait_for_event(timeout):
            waiting.set()
            return original_wait(timeout)
        with patch.object(self.registry.condition, "wait", wait_for_event), ThreadPoolExecutor(max_workers=1) as executor:
            response = executor.submit(self.fixture.response)
            self.assertTrue(waiting.wait(timeout=2))
            self.fixture.mirror("checkpoint", self.fixture.task)
            self.assertEqual(response.result(timeout=2)["input"], self.fixture.controller.envelope)

    def test_emitted_output_requires_exact_hooks_and_native_journal(self):
        block = self.fixture.response()
        self.assertEqual(self.fixture.hook("PreToolUse", "StructuredOutput", block["input"], block["id"], "checkpoint"), {})
        self.fixture.hook("PostToolUse", "StructuredOutput", block["input"], block["id"], "checkpoint", response={})
        self.fixture.finish("checkpoint", block["input"])
        self.fixture.progress()
        self.registry.confirm_return({"quality_stop": {"qualifying": True}})
        self.assertTrue(self.registry.emissions["checkpoint"]["accepted"])

    def test_model_claim_cannot_replace_signed_local_output(self):
        block = self.fixture.response()
        changed = {**block["input"], "payload": "model claims a certificate"}
        reply = self.fixture.hook("PreToolUse", "StructuredOutput", changed, block["id"], "checkpoint")
        self.assertEqual(reply["hookSpecificOutput"]["permissionDecision"], "deny")
        self.assertFalse(self.registry.emissions["checkpoint"]["accepted"])
        self.assertEqual(self.registry.error, "native_tool_admission_failed")

    def test_post_hook_failure_cannot_accept_local_output(self):
        block = self.fixture.response()
        self.fixture.hook("PreToolUse", "StructuredOutput", block["input"], block["id"], "checkpoint")
        self.fixture.hook("PostToolUseFailure", "StructuredOutput", block["input"], block["id"], "checkpoint")
        self.assertEqual(self.registry.error, "native_tool_outcome_unknown")
        self.assertFalse(self.registry.emissions["checkpoint"]["accepted"])

    def test_pending_checkpoint_pre_hook_cannot_be_delivered_twice(self):
        block = self.fixture.response()
        self.fixture.hook("PreToolUse", "StructuredOutput", block["input"], block["id"], "checkpoint")
        reply = self.fixture.hook("PreToolUse", "StructuredOutput", block["input"], block["id"], "checkpoint")
        self.assertEqual(reply["hookSpecificOutput"]["permissionDecision"], "deny")
        self.assertEqual(self.registry.error, "native_tool_admission_failed")
        self.assertFalse(self.registry.emissions["checkpoint"]["accepted"])
        self.assertEqual(len(self.fixture.controller.calls), 1)

    def test_accepted_checkpoint_tool_cannot_execute_twice(self):
        block = self.fixture.response()
        self.fixture.hook("PreToolUse", "StructuredOutput", block["input"], block["id"], "checkpoint")
        self.fixture.hook("PostToolUse", "StructuredOutput", block["input"], block["id"], "checkpoint")
        reply = self.fixture.hook("PreToolUse", "StructuredOutput", block["input"], block["id"], "checkpoint")
        self.assertEqual(reply["hookSpecificOutput"]["permissionDecision"], "deny")
        self.assertEqual(self.registry.error, "native_tool_admission_failed")

    def test_checkpoint_post_hook_requires_unique_pre_and_post_delivery(self):
        block = self.fixture.response()
        self.fixture.hook("PreToolUse", "StructuredOutput", block["input"], block["id"], "checkpoint")
        self.fixture.hook("PostToolUse", "StructuredOutput", block["input"], block["id"], "checkpoint")
        self.assertIsNone(self.registry.error)
        self.fixture.hook("PostToolUse", "StructuredOutput", block["input"], block["id"], "checkpoint")
        self.assertEqual(self.registry.error, "native_tool_outcome_unknown")
        self.assertEqual(len(self.fixture.controller.calls), 1)

    def test_checkpoint_post_hook_without_pre_hook_is_rejected(self):
        block = self.fixture.response()
        self.fixture.hook("PostToolUse", "StructuredOutput", block["input"], block["id"], "checkpoint")
        self.assertEqual(self.registry.error, "native_tool_outcome_unknown")
        self.assertFalse(self.registry.emissions["checkpoint"]["accepted"])

    def test_managed_result_change_after_local_emission_rejects_closure(self):
        block = self.fixture.response()
        self.fixture.hook("PreToolUse", "StructuredOutput", block["input"], block["id"], "checkpoint")
        self.fixture.hook("PostToolUse", "StructuredOutput", block["input"], block["id"], "checkpoint")
        self.fixture.finish("checkpoint", {**block["input"], "payload": "managed hook changed this"})
        self.fixture.progress()
        with self.rejected("native_closure_checkpoint_changed"):
            self.registry.confirm_return({"quality_stop": {"qualifying": True}})

    def test_previous_checkpoint_output_change_blocks_next_census(self):
        block = self.fixture.response()
        self.fixture.hook("PreToolUse", "StructuredOutput", block["input"], block["id"], "checkpoint")
        self.fixture.hook("PostToolUse", "StructuredOutput", block["input"], block["id"], "checkpoint")
        self.fixture.finish("checkpoint", {**block["input"], "payload": "changed after hook"})
        self.fixture.add("checkpoint2", "quality_stop:boundary r1 l1", task=self.fixture.task)
        self.fixture.progress()
        with self.rejected("checkpoint_native_result_changed"):
            self.registry.assert_quiescent("checkpoint2")

    def test_closure_requires_all_producers_and_terminal_records(self):
        with self.rejected("native_closure_sets_differ"):
            self.registry.confirm_return({"quality_stop": {"qualifying": True}})
        self.registry.confirm_return(None)
        self.registry.confirm_return({"quality_stop": {"qualifying": False}})
        self.fixture.finish("checkpoint", {"done": True})
        self.fixture.progress()
        self.registry.confirm_return({"quality_stop": {"qualifying": True}})
        self.registry.nodes[(1, 3, 1)]["state"] = "running"
        with self.rejected("native_closure_not_done"):
            self.registry.confirm_return({"quality_stop": {"qualifying": True}})

    def test_closure_census_binds_exact_journal_bridge_and_mirrored_tasks(self):
        self.fixture.controller.state_dir = self.fixture.root / "controller"
        self.fixture.controller.state_dir.mkdir()
        for event in ("started", "forwarding", "finished"):
            self.registry.record_transport({"event": event, "request_id": "fixture"})
        self.fixture.finish("checkpoint", {"done": True})
        self.fixture.progress()
        census = self.registry.confirm_return({"quality_stop": {"enabled": True, "qualifying": False}})
        self.assertEqual(census["root_tool_id"], "root-tool")
        self.assertEqual(census["root_task_id"], "root-task")
        self.assertEqual(census["session_id"], "stop-session")
        self.assertEqual(census["journal"]["raw_utf8"].encode(), self.fixture.journal.read_bytes())
        bridge = self.fixture.controller.state_dir / "bridge.jsonl"
        self.assertEqual(census["bridge"]["raw_utf8"].encode(), bridge.read_bytes())
        self.assertEqual(census["bridge"]["sha256"], hashlib.sha256(bridge.read_bytes()).hexdigest())
        self.assertEqual({node["agentId"] for node in census["nodes"]}, set(census["mirrors"]))
        self.assertEqual(census["mirrors"]["checkpoint"]["task"], self.fixture.task)
        self.assertEqual(census["mirrors"]["checkpoint"]["initial_entry"]["message"]["content"], self.fixture.task)
        self.assertEqual(census["active_tools"], {})
        self.assertEqual(census["active_background"], [])
        bridge.write_text(bridge.read_text().replace("finished", "altered_"))
        with self.assertRaisesRegex(StopRejected, "native_closure_bridge_changed"):
            self.registry.confirm_return({"quality_stop": {"enabled": True, "qualifying": False}})

    def test_closure_waits_for_the_delayed_bridge_terminal_and_seals_new_sends(self):
        self.fixture.controller.state_dir = self.fixture.root / "controller"
        self.fixture.controller.state_dir.mkdir()
        self.fixture.finish("checkpoint", {"done": True})
        self.fixture.progress()
        for event in ("started", "forwarding"):
            self.registry.record_transport({"event": event, "request_id": "fixture"})
        self.registry.wait_seconds = 2
        waiting = threading.Event()
        original_wait = self.registry.condition.wait
        def wait(timeout):
            waiting.set()
            return original_wait(timeout)
        with patch.object(self.registry.condition, "wait", side_effect=wait), ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.registry.confirm_return, {"quality_stop": {"enabled": True, "qualifying": False}})
            self.assertTrue(waiting.wait(timeout=1))
            self.assertFalse(future.done())
            self.registry.record_transport({"event": "finished", "request_id": "fixture"})
            census = future.result(timeout=2)
        self.assertTrue(census["bridge_all_terminal"])
        self.assertEqual(census["request_records"][-1]["event"], "finished")
        self.assertEqual(census["bridge"]["sha256"], hashlib.sha256(
            (self.fixture.controller.state_dir / "bridge.jsonl").read_bytes()).hexdigest())
        with self.assertRaisesRegex(StopRejected, "native_transport_after_closure"):
            self.registry.record_transport({"event": "started", "request_id": "late"})

    def test_closure_rejects_an_unclosed_or_malformed_bridge_sequence(self):
        self.fixture.finish("checkpoint", {"done": True})
        self.fixture.progress()
        self.registry.record_transport({"event": "started", "request_id": "fixture"})
        with self.assertRaisesRegex(StopRejected, "native_closure_bridge_pending"):
            self.registry.confirm_return({"quality_stop": {"enabled": True, "qualifying": False}})
        self.registry.record_transport({"event": "started", "request_id": "fixture"})
        with self.assertRaisesRegex(StopRejected, "native_closure_bridge_sequence_invalid"):
            self.registry.confirm_return({"quality_stop": {"enabled": True, "qualifying": False}})

    def test_closure_requires_every_initial_mirror(self):
        self.fixture.finish("checkpoint", {"done": True})
        self.fixture.progress()
        self.registry.mirror_inputs.pop("engineer")
        with self.assertRaisesRegex(StopRejected, "native_closure_mirrors_differ"):
            self.registry.confirm_return({"quality_stop": {"enabled": True, "qualifying": False}})

    def test_duplicate_active_tool_start_and_changed_result_are_rejected(self):
        self.fixture.hook("PreToolUse", "Bash", {"command": "true"}, "bash", "engineer")
        reply = self.fixture.hook("PreToolUse", "Bash", {"command": "true"}, "bash", "engineer")
        self.assertEqual(reply["hookSpecificOutput"]["permissionDecision"], "deny")
        self.registry.error = None
        self.fixture.hook("PostToolUse", "Bash", {"command": "false"}, "bash", "engineer", response={})
        self.assertEqual(self.registry.error, "native_tool_outcome_unknown")

    def test_completed_tool_id_cannot_be_reused(self):
        reply = self.fixture.hook("PreToolUse", "Bash", {"command": self.fixture.reclaim_command}, "reclaim-tool", "reclaim")
        self.assertEqual(reply.get("hookSpecificOutput", {}).get("permissionDecision"), "deny")
        self.assertIsNotNone(self.registry.error)

    def test_hook_identity_changes_and_missing_tool_id_are_rejected(self):
        for changes in ({"session_id": "other"}, {"cwd": "/other"}, {}):
            with self.subTest(changes=changes):
                self.registry.error = None
                reply = self.fixture.hook("PreToolUse", "Bash", {}, None, "engineer", **changes)
                self.assertEqual(reply["hookSpecificOutput"]["permissionDecision"], "deny")

    def test_native_task_inspection_does_not_create_producers(self):
        for name in ("TaskOutput", "TaskList", "TaskGet"):
            identity = "inspect-" + name
            self.registry.observe(AssistantMessage(content=[{"id": identity, "name": name, "input": {}}]))
            self.fixture.hook("PreToolUse", name, {}, identity, None)
            self.fixture.hook("PostToolUse", name, {}, identity, None)
        self.assertEqual(self.registry.tools, {})
        self.assertIsNone(self.registry.error)

    def test_background_bash_flag_or_task_result_fails_closed(self):
        for response, inputs in [({}, {"command": "true", "run_in_background": True}),
                                 ({"task_id": "background"}, {"command": "true"}),
                                 ({"backgroundTaskId": "background"}, {"command": "true"})]:
            with self.subTest(response=response, inputs=inputs):
                self.registry.error = None
                identity = "background" + str(len(self.registry.completed_tools))
                self.fixture.hook("PreToolUse", "Bash", inputs, identity, "engineer")
                self.fixture.hook("PostToolUse", "Bash", inputs, identity, "engineer", response=response)
                self.assertEqual(self.registry.error, "native_tool_outcome_unknown")

    def test_reclaim_needs_exact_command_and_clean_native_completion(self):
        operation = self.registry.completed_tools["reclaim-tool"]
        for changes in ({"interrupted": True}, {"stdout": "unknown"}, {"truncated": True},
                        {"background_task_id": "background"}, {"isImage": True}, {"stdout": None}):
            with self.subTest(changes=changes):
                saved = deepcopy(operation["response"])
                operation["response"].update(changes)
                with self.rejected("storage_reclaim_completion_unknown"):
                    self.registry.assert_quiescent("checkpoint")
                operation["response"] = saved
        operation["input"] = {"command": "different command"}
        with self.rejected("storage_reclaim_execution_missing_or_duplicate"):
            self.registry.assert_quiescent("checkpoint")

    def test_reclaim_prompt_and_execution_are_unique(self):
        mirror = self.registry.mirrors["reclaim"]
        for task in ("No command.", self.fixture.reclaim_task + self.fixture.reclaim_task):
            mirror["task"] = task
            with self.rejected("storage_reclaim_command_missing"):
                self.registry.assert_quiescent("checkpoint")
        mirror["task"] = self.fixture.reclaim_task
        self.registry.completed_tools["duplicate"] = deepcopy(self.registry.completed_tools["reclaim-tool"])
        with self.rejected("storage_reclaim_execution_missing_or_duplicate"):
            self.registry.assert_quiescent("checkpoint")


class NativeFailureEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = NativeFixture(self.temporary.name)
        self.registry = self.fixture.registry
        self.directory = Path(self.temporary.name) / "private_controller"
        self.directory.mkdir(mode=0o700)
        self.fixture.controller.state_dir = self.directory
        self.path = self.directory / "native_census_first_failure.json"

    def conflict(self):
        self.fixture.nodes[0]["agentId"] = "unqualified_replacement"
        self.fixture.progress()

    def test_first_progress_failure_keeps_leaf_and_both_identity_views(self):
        self.conflict()
        value = json.loads(self.path.read_text())
        self.assertEqual(value["reason"], "native_census_event_conflict")
        self.assertEqual(value["exception"]["code"], "native_node_changed")
        self.assertEqual(value["nodes"][0]["agentId"], "engineer")
        incoming = value["trigger"]["event"]["data"]["workflow_progress"]
        self.assertEqual(incoming[0]["agentId"], "unqualified_replacement")
        self.assertTrue(self.registry.first_failure_persisted)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        first = self.path.read_bytes()
        self.fixture.hook("PreToolUse", "Bash", {}, "later", cwd="/wrong")
        self.assertEqual(self.path.read_bytes(), first)
        self.assertEqual(self.registry.error, "native_census_event_conflict")

    def test_hook_payload_is_hashed_without_retaining_credential_text(self):
        payload = {"command": "export API_KEY=synthetic-private-value", "run_in_background": True}
        self.fixture.hook("PreToolUse", "Bash", payload, "secret-tool", cwd="/wrong")
        raw = self.path.read_bytes()
        self.assertNotIn(b"synthetic-private-value", raw)
        value = json.loads(raw)
        self.assertEqual(value["exception"]["code"], "native_tool_hook_identity_changed")
        self.assertEqual(value["trigger"]["event"]["tool_input_sha256"],
                         hashlib.sha256(canonical(payload).encode("ascii")).hexdigest())
        self.assertTrue(value["trigger"]["event"]["tool_input.run_in_background"])

    def test_existing_failure_file_is_never_replaced(self):
        self.path.write_bytes(b"Preserved earlier evidence.\n")
        self.conflict()
        self.assertEqual(self.path.read_bytes(), b"Preserved earlier evidence.\n")
        self.assertFalse(self.registry.first_failure_persisted)
        self.assertEqual(self.registry.error, "native_census_event_conflict")

    def test_diagnostic_write_failure_does_not_restore_identity_admission(self):
        for error in (OSError("Synthetic write failure."), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                self.registry.error = None
                self.registry.first_failure = None
                with patch("interface.native_cost_controls.quality_stop_native.os.open", side_effect=error):
                    self.conflict()
                self.assertEqual(self.registry.error, "native_census_event_conflict")
                self.assertFalse(self.registry.first_failure_persisted)
                headers = [(key, "unknown" if key == "x-claude-code-agent-id" else value)
                           for key, value in self.fixture.headers]
                with self.assertRaisesRegex(StopRejected, "native_identity_unavailable"):
                    self.registry.response(headers, canonical(self.fixture.body).encode())

    def test_closed_registry_failure_records_the_distinct_closed_state(self):
        self.registry.close()
        headers = [(key, "unknown" if key == "x-claude-code-agent-id" else value)
                   for key, value in self.fixture.headers]
        with self.assertRaisesRegex(StopRejected, "native_identity_unavailable"):
            self.registry.response(headers, canonical(self.fixture.body).encode())
        value = json.loads(self.path.read_text())
        self.assertEqual(value["reason"], "native_registry_closed")
        self.assertEqual(value["exception"]["code"], "native_identity_unavailable")
        self.assertTrue(value["closed"])
        self.assertIsNotNone(value["closed_at"])
        self.assertIsNone(self.registry.error)


class NativeAdmissionTests(unittest.TestCase):
    def test_root_request_is_exact_and_does_not_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = NativeFixture(directory, populate=False)
            self.addCleanup(fixture.registry.close)
            variants = [("unsupported_stopping_root", {**fixture.request, "extra": True}),
                        ("stopping_root_changed", {**fixture.request, "args": {**fixture.request["args"], "mode": "evaluate"}}),
                        ("stopping_root_changed", {**fixture.request, "args": {**fixture.request["args"], "quality_stop": {}}}),
                        ("stopping_resume_unsupported", {**fixture.request, "args": {**fixture.request["args"], "resume": "session"}})]
            for code, request in variants:
                with self.subTest(code=code), self.assertRaisesRegex(StopRejected, code):
                    NativeProducerCensus(session_id="stop-session", root_request=request, source_root=fixture.source,
                        workspace=fixture.root, controller=fixture.controller)

    def test_root_admission_and_descriptor_cannot_change(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = NativeFixture(directory)
            self.addCleanup(fixture.registry.close)
            for changes in ({"agent_id": "child"}, {"tool_input": {"changed": True}}):
                with self.subTest(changes=changes):
                    fixture.registry.error = None
                    reply = fixture.hook("PreToolUse", "Workflow", fixture.request, "root-tool", **changes)
                    self.assertEqual(reply["hookSpecificOutput"]["permissionDecision"], "deny")
            for change in ("error", "parent", "task", "directory", "descriptor"):
                with self.subTest(change=change):
                    fixture.registry.error = None
                    event = fixture.root_result()
                    if change == "error":
                        event.content[0]["is_error"] = True
                    elif change == "parent":
                        event.parent_tool_use_id = "parent"
                    elif change == "task":
                        event.tool_use_result["taskId"] = "other"
                    elif change == "directory":
                        event.tool_use_result["transcriptDir"] = "relative"
                    else:
                        event.tool_use_result["runId"] = "wf_other"
                    fixture.registry.observe(event)
                    self.assertEqual(fixture.registry.error, "native_census_event_conflict")
            fixture.registry.error = None
            fixture.registry.observe(UserMessage(content=[]))
            fixture.registry.observe(UserMessage(content="ignored"))
            fixture.registry.observe(TaskStartedMessage(task_id="other-root"))
            self.assertEqual(fixture.registry.error, "native_census_event_conflict")

    def test_json_object_parser_rejects_ambiguous_values(self):
        for raw in (b'[]', b'{"a":1,"a":2}', b'{"nested":{"a":1,"a":2}}', b'{"value":NaN}', b'invalid'):
            with self.subTest(raw=raw), self.assertRaises((StopRejected, ValueError)):
                _object(raw)
        self.assertEqual(_object(b'{"value":1}'), {"value": 1})


class Upstream:
    def __init__(self):
        self.calls = []
        self.closed = False
        self.response = object()

    @contextmanager
    def send(self, *arguments):
        self.calls.append(arguments)
        yield self.response

    def close(self):
        self.closed = True


class CheckpointTransportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = NativeFixture(self.temporary.name)
        self.upstream = Upstream()
        self.transport = CheckpointTransport(self.upstream, self.fixture.registry, base_path="/api/")
        self.addCleanup(self.transport.close)

    def test_local_checkpoint_never_uses_upstream(self):
        with self.transport.send("POST", "/api/v1/messages", self.fixture.headers, canonical(self.fixture.body).encode()) as response:
            value = json.loads(b"".join(response.iter_bytes()))
        self.assertEqual(value["model"], "local-quality-stop-controller-v2")
        self.assertEqual(value["content"][0]["input"], self.fixture.controller.envelope)
        self.assertEqual(self.upstream.calls, [])
        final = self.fixture.registry.request_records[-1]
        self.assertEqual(final["route"], "local_controller")
        self.assertFalse(final["upstream_bridge_attempted"])

    def assert_forwarded(self, method, target, headers, raw):
        actual_method, actual_target, actual_headers, actual_raw = self.upstream.calls[-1]
        self.assertEqual((actual_method, actual_target, actual_raw), (method, target, raw))
        traced = [value for key, value in actual_headers if key.lower() == "x-geak-quality-request-id"]
        self.assertEqual(len(traced), 1)
        self.assertTrue(traced[0].startswith("quality_bridge_"))
        self.assertEqual([(key, value) for key, value in actual_headers if key.lower() != "x-geak-quality-request-id"],
                         [(key, value) for key, value in headers if key.lower() != "x-geak-quality-request-id"])
        self.assertEqual(self.fixture.registry.request_records[-1]["request_id"], traced[0])

    def test_scientific_body_and_nonmessage_route_are_forwarded_unchanged(self):
        raw = b'{  "messages": [] }'
        headers = [("Content-Type", "application/json"), ("Authorization", "public-fixture"),
                   ("x-claude-code-session-id", "stop-session"), ("x-claude-code-agent-id", "engineer")]
        for method, target in (("POST", "/api/v1/messages"), ("GET", "/api/models")):
            with self.transport.send(method, target, headers, raw) as response:
                self.assertIs(response, self.upstream.response)
            self.assert_forwarded(method, target, headers, raw)

    def test_nonmessage_route_preserves_opaque_or_empty_body(self):
        headers = [("x-claude-code-session-id", "stop-session"), ("x-claude-code-agent-id", "engineer")]
        for raw in (b"opaque fixture body", b""):
            with self.subTest(raw=raw):
                with self.transport.send("GET", "/api/models", headers, raw) as response:
                    self.assertIs(response, self.upstream.response)
                self.assert_forwarded("GET", "/api/models", headers, raw)

    def test_bridge_trace_is_fresh_and_cannot_be_supplied_by_the_client(self):
        headers = [("Content-Type", "application/json"), ("x-geak-quality-request-id", "forged"),
                   ("x-claude-code-session-id", "stop-session"), ("x-claude-code-agent-id", "engineer")]
        raw = b'{"messages":[]}'
        ids = []
        for _ in range(2):
            with self.transport.send("POST", "/api/v1/messages", headers, raw):
                pass
            self.assert_forwarded("POST", "/api/v1/messages", headers, raw)
            ids.append(self.fixture.registry.request_records[-1]["request_id"])
        self.assertEqual(len(set(ids)), 2)
        self.assertNotIn("forged", ids)

    def test_private_bridge_ledger_contains_hashes_and_no_credentials(self):
        state = self.fixture.root / "controller-state"
        state.mkdir()
        self.fixture.controller.state_dir = state
        headers = [("Content-Type", "application/json"), ("Authorization", "synthetic-private-token"),
                   ("x-claude-code-session-id", "stop-session"), ("x-claude-code-agent-id", "engineer")]
        raw = b'{"messages":[],"private_text":"private-message-text"}'
        with self.transport.send("POST", "/api/v1/messages", headers, raw):
            pass
        saved = (state / "bridge.jsonl").read_text()
        records = [json.loads(line) for line in saved.splitlines()]
        self.assertEqual([row["event"] for row in records], ["started", "forwarding", "finished"])
        self.assertTrue(all(row["body_sha256"] == hashlib.sha256(raw).hexdigest() for row in records))
        self.assertNotIn("synthetic-private-token", saved)
        self.assertNotIn("private-message-text", saved)
        self.assertIsNone(records[-1]["provider_attempted"])

    def test_bridge_ledger_link_cannot_receive_a_request(self):
        state = self.fixture.root / "controller-state"
        state.mkdir()
        self.fixture.controller.state_dir = state
        foreign = self.fixture.root / "foreign-file"
        foreign.write_text("unchanged")
        (state / "bridge.jsonl").symlink_to(foreign)
        with self.assertRaisesRegex(StopRejected, "bridge_ledger_unavailable"), \
                self.transport.send("GET", "/api/models", [], b""):
            self.fail("The request used an unsafe ledger.")
        self.assertEqual(foreign.read_text(), "unchanged")
        self.assertEqual(self.upstream.calls, [])

    def test_invalid_local_request_never_falls_back_to_provider(self):
        for headers in ([], [("Content-Type", "text/plain")],
                        [("Content-Type", "application/json"), ("Content-Encoding", "gzip")]):
            rejection = self.assertRaisesRegex(StopRejected, "checkpoint_transport_encoding_unsupported")
            request = self.transport.send("POST", "/api/v1/messages", headers, canonical(self.fixture.body).encode())
            with self.subTest(headers=headers), rejection, request:
                self.fail("The invalid route returned a response.")
        request = self.transport.send("POST", "/api/v1/messages", [("Content-Type", "application/json")],
                                      canonical(self.fixture.body).encode())
        with self.assertRaisesRegex(StopRejected, "checkpoint_identity_missing"), request:
            self.fail("The invalid identity returned a response.")
        self.assertEqual(self.upstream.calls, [])

    def test_checkpoint_body_on_unsupported_endpoint_cannot_reach_provider(self):
        raw = canonical(self.fixture.body).encode()
        headers = [("Content-Type", "application/json")]
        for method, target in (("POST", "/api/v1/messages/count_tokens"),
                               ("POST", "/other/v1/messages"), ("POST", "/api/unknown"),
                               ("GET", "/api/v1/messages")):
            request = self.transport.send(method, target, headers, raw)
            with self.subTest(method=method, target=target), self.assertRaises(StopRejected), request:
                self.fail("The checkpoint reached an unsupported endpoint.")
        self.assertEqual(self.upstream.calls, [])
        self.assertEqual(self.fixture.controller.calls, [])

    def test_known_checkpoint_identity_rejects_opaque_compressed_and_empty_bodies(self):
        cases = [("opaque", b"opaque fixture body", []),
                 ("compressed", gzip.compress(canonical(self.fixture.body).encode()), [("Content-Encoding", "gzip")]),
                 ("empty", b"", [])]
        for name, body, extra_headers in cases:
            for method, target in (("POST", "/api/v1/messages/count_tokens"), ("GET", "/api/v1/messages")):
                request = self.transport.send(method, target, self.fixture.headers + extra_headers, body)
                rejection = self.assertRaisesRegex(StopRejected, "checkpoint_endpoint_unsupported")
                with self.subTest(body=name, method=method, target=target), rejection, request:
                    self.fail("The known checkpoint reached an unsupported endpoint.")
        self.assertEqual(self.upstream.calls, [])
        self.assertEqual(self.fixture.controller.calls, [])

    def test_emitted_checkpoint_keeps_required_route_without_task_mirror(self):
        self.fixture.response()
        self.fixture.registry.mirrors.pop("checkpoint")
        request = self.transport.send("POST", "/api/unknown", self.fixture.headers, b"")
        with self.assertRaisesRegex(StopRejected, "checkpoint_endpoint_unsupported"), request:
            self.fail("The emitted checkpoint reached an unsupported endpoint.")
        self.assertEqual(self.upstream.calls, [])
        self.assertEqual(len(self.fixture.controller.calls), 1)

    def test_close_releases_upstream_and_registry(self):
        self.transport.close()
        self.assertTrue(self.upstream.closed)
        self.assertTrue(self.fixture.registry.closed)

    def test_json_and_stream_responses_preserve_envelope_and_zero_usage(self):
        block = {"type": "tool_use", "id": "signed-call", "name": "StructuredOutput",
                 "input": {"payload": "literal\\ntext", "signature": "signed"}}
        for stream in (False, True):
            with self.subTest(stream=stream):
                response = CheckpointResponse(block, stream)
                raw = b"".join(response.iter_bytes())
                headers = dict(response.headers)
                self.assertEqual((response.status, response.reason), (200, "OK"))
                self.assertEqual(headers["Content-Length"], str(len(raw)))
                self.assertEqual(headers["x-geak-local-origin"], "quality_stop_controller")
                self.assertEqual(headers["request-id"], block["id"])
                if stream:
                    events = [json.loads(part.split(b"\ndata: ", 1)[1]) for part in raw.strip().split(b"\n\n")]
                    self.assertEqual([event["type"] for event in events], ["message_start", "content_block_start",
                        "content_block_delta", "content_block_stop", "message_delta", "message_stop"])
                    self.assertEqual(json.loads(events[2]["delta"]["partial_json"]), block["input"])
                    message = events[0]["message"]
                    self.assertEqual(headers["Content-Type"], "text/event-stream")
                else:
                    message = json.loads(raw)
                    self.assertEqual(message["content"], [block])
                    self.assertEqual(headers["Content-Type"], "application/json")
                self.assertEqual(message["usage"], {"input_tokens": 0, "output_tokens": 0,
                    "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0})


if __name__ == "__main__":
    unittest.main()
