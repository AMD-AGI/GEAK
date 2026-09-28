# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qualify retirement with synthetic events and protected private files."""

import asyncio
import hashlib
import json
import os
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from unittest.mock import patch

from interface.native_cost_controls.quality_stop_controller import StopRejected, canonical
from interface.native_cost_controls.quality_stop_native import CheckpointTransport
from interface.native_cost_controls.quality_stop_retirement import INTERRUPTION, canonical_bytes
from interface.native_cost_controls.native_journal import JournalError
from interface.test_quality_stop_native import (
    NativeFixture, Upstream, AssistantMessage, TaskStartedMessage, TaskNotificationMessage,
)


class RetirementFixture:
    def __init__(self, directory, *, label="engineer:replacement"):
        self.fixture = f = NativeFixture(directory, populate=False)
        self.registry = r = f.registry
        r.retirement_wait_seconds = .15
        f.controller.state_dir = f.root / "controller-state"
        f.controller.state_dir.mkdir(mode=0o700)
        f.start()
        self.task = f.reclaim_task if label.startswith("storage:reclaim") else "Synthetic replacement task."
        self.prompt = "11111111-1111-1111-1111-111111111111"
        self.old = {"type": "workflow_agent", "phaseIndex": 1, "index": 1, "attempt": 1,
                    "agentId": "old", "label": label, "state": "progress"}
        f.nodes = [deepcopy(self.old)]
        f.append("started", "old")
        f.mirror("old", self.task, promptId=self.prompt, uuid="22222222-2222-2222-2222-222222222222",
                 message={"role": "user", "content": self.task})
        f.progress()
        self.new = {**self.old, "attempt": 2, "agentId": "new", "state": "start",
                    "label": self.old["label"] + " (retry 1)"}
        self.initial = deepcopy(r.mirror_inputs["old"]["initial_entry"])
        self.previous = {"type": "assistant", "uuid": "33333333-3333-3333-3333-333333333333",
                         "message": {"role": "assistant", "content": []}}
        self.terminal = {"parentUuid": self.previous["uuid"], "isSidechain": True, "promptId": self.prompt,
            "agentId": "old", "type": "user", "message": deepcopy(INTERRUPTION),
            "uuid": "44444444-4444-4444-4444-444444444444", "timestamp": "2026-09-27T20:00:00.000Z",
            "userType": "external", "entrypoint": "sdk-py", "cwd": str(f.root), "sessionId": "stop-session",
            "version": "2.1.221", "gitBranch": "synthetic"}
        self.transcript = f.directory / "agent-old.jsonl"
        self.write_transcript()
        self.upstream = Upstream()
        self.transport = CheckpointTransport(self.upstream, r)

    def write_transcript(self, rows=None):
        self.transcript.write_bytes(canonical_bytes(rows or [self.initial, self.previous, self.terminal]))

    def replace(self):
        f = self.fixture
        with f.journal.open("a") as stream:
            stream.write(canonical({"type": "started", "agentId": "new",
                "key": "v2:" + hashlib.sha256(b"old").hexdigest()}) + "\n")
        f.mirror("new", self.task, promptId=self.prompt, uuid="55555555-5555-5555-5555-555555555555",
                 message={"role": "user", "content": self.task})
        f.nodes = [deepcopy(self.new)]
        f.progress()

    def headers(self, agent):
        return [("Content-Type", "application/json"), ("x-claude-code-session-id", "stop-session"),
                ("x-claude-code-agent-id", agent)]

    def send(self, agent="new", *, method="POST", path="/v1/messages", body=b'{"messages":[]}'):
        with self.transport.send(method, path, self.headers(agent), body):
            return True

    def bridge(self, event, request="old-request"):
        self.registry.record_transport({"request_id": request, "event": event, "agent_id": "old",
            "session_id": "stop-session", "method": "POST", "path": "/v1/messages",
            "body_sha256": hashlib.sha256(b"{}").hexdigest(), "body_bytes": 2,
            **({"route": "provider_recorder", "upstream_bridge_attempted": True, "provider_attempted": None}
               if event != "started" else {})})

    def retire(self):
        self.replace()
        self.send()


class NativeRetirementTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fx = RetirementFixture(self.temporary.name)
        self.r = self.fx.registry
        self.addCleanup(self.fx.transport.close)

    def test_proof_binds_old_attempt_and_fence_precedes_successor_forwarding(self):
        self.fx.bridge("started")
        self.fx.bridge("forwarding")
        self.fx.bridge("finished")
        self.fx.retire()
        self.assertEqual(list(self.r.retired), ["old"])
        fence = self.r.retirement_events[0]
        proof = fence["proof"]
        self.assertEqual(proof["old_bridge_request_ids"], ["old-request"])
        self.assertEqual(proof["old_node"], self.fx.old)
        self.assertEqual(proof["new_node"], self.fx.new)
        self.assertEqual(proof["logical_label"], self.fx.old["label"])
        self.assertEqual(proof["activity_evidence"], "trusted_host_state_under_shared_admission_lock")
        successor = [row for row in self.r.request_records if row.get("agent_id") == "new"]
        self.assertEqual([row["event"] for row in successor], ["started", "forwarding", "finished"])
        self.assertGreater(successor[1]["monotonic_time"], fence["monotonic_time"])
        self.assertEqual(proof["bridge_event_count"], 4)
        raw = canonical_bytes(self.r.request_records[:4])
        self.assertEqual(proof["bridge_prefix_sha256"], hashlib.sha256(raw).hexdigest())
        saved = (self.fx.fixture.controller.state_dir / "native_retirements.jsonl").read_bytes()
        self.assertEqual(saved, canonical_bytes(self.r.retirement_events))
        self.assertEqual(set(self.r.mirrors), {"old", "new"})
        self.assertEqual(self.r.journal.snapshot()["started_agents"], ["new", "old"])

    def test_successful_closure_preserves_current_and_retired_populations(self):
        self.fx.retire()
        result = {"ok": True}
        with self.fx.fixture.journal.open("a") as stream:
            stream.write(canonical({"type": "result", "agentId": "new",
                "key": "v2:" + hashlib.sha256(b"old").hexdigest(), "result": result}) + "\n")
        self.fx.fixture.nodes[0].update(state="done", lastToolName="StructuredOutput", resultPreview=canonical(result))
        self.fx.fixture.progress()
        census = self.r.confirm_return({"quality_stop": {"enabled": True}})
        self.assertEqual([node["agentId"] for node in census["nodes"]], ["new"])
        self.assertEqual(set(census["mirrors"]), {"old", "new"})
        self.assertEqual(set(census["journal"]["snapshot"]["results"]), {"new"})
        self.assertEqual(census["journal"]["snapshot"]["started_agents"], ["new", "old"])
        self.assertEqual(census["retirements"]["events"][0]["old_agent_id"], "old")
        self.assertTrue(self.r.transport_sealed)

    def test_retry_label_remains_raw_and_logical_role_checks_still_work(self):
        with tempfile.TemporaryDirectory() as directory:
            fx = RetirementFixture(directory, label="storage:reclaim r1")
            try:
                fx.retire()
                f = fx.fixture
                f.hook("PreToolUse", "Bash", {"command": f.reclaim_command}, "reclaim-execution", "new")
                f.hook("PostToolUse", "Bash", {"command": f.reclaim_command}, "reclaim-execution", "new",
                       response={"stdout": "STORAGE_RECLAIM_DONE round=1\n", "stderr": "", "interrupted": False})
                result = {"done": True}
                with f.journal.open("a") as stream:
                    stream.write(canonical({"type": "result", "agentId": "new",
                        "key": "v2:" + hashlib.sha256(b"old").hexdigest(), "result": result}) + "\n")
                f.nodes[0].update(state="done", lastToolName="StructuredOutput", resultPreview=canonical(result))
                f.add("checkpoint", "quality_stop:boundary r1 l1", task=f.task)
                f.progress()
                fx.registry.assert_quiescent("checkpoint")
                self.assertEqual(fx.registry._node("new")["label"], "storage:reclaim r1 (retry 1)")
                self.assertEqual(fx.registry.retired["old"]["logical_label"], "storage:reclaim r1")
            finally:
                fx.transport.close()

    def test_retry_suffix_must_match_the_exact_attempt(self):
        labels = ("engineer:replacement", "engineer:replacement (retry 2)",
                  "engineer:replacement (retry 01)", "engineer:replacement (retry 1) changed")
        for label in labels:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                fx = RetirementFixture(directory)
                try:
                    fx.new["label"] = label
                    fx.replace()
                    self.assertEqual(fx.registry.error, "native_census_event_conflict")
                    self.assertEqual(fx.registry.first_failure["exception"]["code"], "native_retry_label_changed")
                    self.assertEqual(fx.registry.retirement_events, [])
                finally:
                    fx.transport.close()

    def test_raw_retry_label_cannot_disappear_after_admission(self):
        self.fx.retire()
        self.fx.fixture.nodes[0].update(state="progress", label=self.fx.old["label"])
        self.fx.fixture.progress()
        self.assertEqual(self.r.error, "native_census_event_conflict")
        self.assertEqual(self.r.first_failure["exception"]["code"], "native_node_changed")

    def test_open_old_bridge_blocks_successor_until_terminal(self):
        self.r.retirement_wait_seconds = 2
        self.fx.bridge("started")
        self.fx.bridge("forwarding")
        self.fx.replace()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.fx.send)
            with self.r.condition:
                self.assertTrue(self.r.condition.wait_for(lambda: len(self.r.request_records) == 3, timeout=1))
            self.assertFalse(future.done())
            self.assertEqual(self.fx.upstream.calls, [])
            self.assertEqual(self.r.retired, {})
            self.fx.bridge("finished")
            self.assertTrue(future.result(timeout=2))
        self.assertIsNone(self.r.error)

    def test_automatic_bash_blocks_retirement_until_native_terminal(self):
        self.r.retirement_wait_seconds = 2
        f = self.fx.fixture
        inputs = {"command": "sleep 135"}
        f.hook("PreToolUse", "Bash", inputs, "old-bash", "old")
        self.r.observe(TaskStartedMessage(task_id="b12345678", tool_use_id="old-bash", task_type="local_bash"))
        f.hook("PostToolUse", "Bash", inputs, "old-bash", "old", response={
            "stdout": "", "stderr": "", "interrupted": False,
            "backgroundTaskId": "b12345678", "timedOutAfterMs": 120000})
        self.fx.replace()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.fx.send)
            with self.r.condition:
                self.assertTrue(self.r.condition.wait_for(lambda: len(self.r.request_records) == 1, timeout=1))
            self.assertFalse(future.done())
            self.assertEqual(self.r.retired, {})
            self.assertEqual(self.fx.upstream.calls, [])
            self.r.observe(TaskNotificationMessage(task_id="b12345678", tool_use_id="old-bash",
                                                   task_type="local_bash", status="completed"))
            self.assertTrue(future.result(timeout=2))
        self.assertEqual(list(self.r.retired), ["old"])
        self.assertEqual(self.r.background["b12345678"], "completed")
        self.assertIsNone(self.r.error)

    def test_concurrent_journal_append_defers_the_retirement_snapshot(self):
        self.fx.replace()
        self.fx.fixture.nodes.append({"type": "workflow_agent", "phaseIndex": 1, "index": 2, "attempt": 1,
                                      "agentId": "parallel", "label": "engineer:parallel", "state": "progress"})
        self.fx.fixture.mirror("parallel", "Synthetic parallel task.")
        self.fx.fixture.progress()
        original = self.r.journal.bound_bytes
        def append_after_snapshot(snapshot):
            self.fx.fixture.append("started", "parallel")
            return original(snapshot)
        with patch.object(self.r.journal, "bound_bytes", side_effect=append_after_snapshot):
            with self.r.condition:
                self.r._try_retirements()
        self.assertIsNone(self.r.error)
        self.assertEqual(self.r.retirement_events, [])
        self.assertIn("new", self.r.pending_retirements)
        self.assertTrue(self.fx.send())

    def test_journal_prefix_errors_are_never_treated_as_pending(self):
        self.fx.replace()
        with patch.object(self.r.journal, "bound_bytes", side_effect=JournalError("journal_prefix_changed")):
            with self.assertRaisesRegex(JournalError, "journal_prefix_changed"):
                self.fx.send()
        self.assertEqual(self.r.error, "native_retirement_evidence_failed")
        self.assertEqual(self.r.retirement_events, [])
        self.assertEqual(self.fx.upstream.calls, [])

    def test_changed_journal_identity_prefix_and_size_reject_retirement(self):
        for change in ("inode", "prefix", "truncate"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                fx = RetirementFixture(directory)
                try:
                    fx.replace()
                    self.assertEqual(fx.registry.journal.snapshot()["status"], "complete")
                    journal = fx.fixture.journal
                    raw = journal.read_bytes()
                    if change == "inode":
                        journal.rename(journal.with_name("saved-journal"))
                        journal.write_bytes(raw)
                    elif change == "prefix":
                        journal.write_bytes(raw.replace(b'"old"', b'"xxx"', 1))
                    else:
                        journal.write_bytes(raw[:-1])
                    with self.assertRaisesRegex(StopRejected, "retirement_journal_error"):
                        fx.send()
                    self.assertEqual(fx.registry.retirement_events, [])
                    self.assertEqual(fx.upstream.calls, [])
                finally:
                    fx.transport.close()

    def test_started_but_not_forwarded_old_request_also_blocks_retirement(self):
        self.fx.bridge("started")
        self.fx.replace()
        with self.assertRaisesRegex(StopRejected, "native_retirement_timeout"):
            self.fx.send()
        self.assertEqual(self.fx.upstream.calls, [])
        self.assertEqual(self.r.retirement_events, [])
        self.assertEqual(self.r.error, "native_retirement_timeout")
        self.assertEqual(self.r.request_records[-1]["route"], "unclassified")
        self.assertFalse(self.r.request_records[-1]["upstream_bridge_attempted"])

    def test_pending_successor_chain_refuses_intermediate_and_gates_latest(self):
        self.r.retirement_wait_seconds = 2
        self.fx.bridge("started")
        self.fx.bridge("forwarding")
        self.fx.replace()
        with ThreadPoolExecutor(max_workers=2) as pool:
            future = pool.submit(self.fx.send)
            with self.r.condition:
                self.assertTrue(self.r.condition.wait_for(lambda: len(self.r.request_records) == 3, timeout=1))
            initial = deepcopy(self.r.mirror_inputs["new"]["initial_entry"])
            previous = {**self.fx.previous, "uuid": "66666666-6666-6666-6666-666666666666"}
            terminal = {**self.fx.terminal, "agentId": "new", "parentUuid": previous["uuid"],
                        "uuid": "77777777-7777-7777-7777-777777777777"}
            (self.fx.fixture.directory / "agent-new.jsonl").write_bytes(canonical_bytes([initial, previous, terminal]))
            with self.fx.fixture.journal.open("a") as stream:
                stream.write(canonical({"type": "started", "agentId": "third",
                    "key": "v2:" + hashlib.sha256(b"old").hexdigest()}) + "\n")
            self.fx.fixture.mirror("third", self.fx.task, promptId=self.fx.prompt,
                uuid="88888888-8888-8888-8888-888888888888", message={"role": "user", "content": self.fx.task})
            self.fx.fixture.nodes = [{**self.fx.new, "attempt": 3, "agentId": "third", "state": "start",
                                      "label": self.fx.old["label"] + " (retry 2)"}]
            self.fx.fixture.progress()
            with self.assertRaisesRegex(StopRejected, "native_pending_successor_superseded"):
                future.result(timeout=1)
            self.assertIsNone(self.r.error)
            latest = pool.submit(self.fx.send, "third")
            with self.r.condition:
                self.assertTrue(self.r.condition.wait_for(lambda: len(self.r.request_records) == 5, timeout=1))
            self.assertFalse(latest.done())
            self.assertEqual(self.fx.upstream.calls, [])
            self.assertEqual(self.r.retirement_events, [])
            self.fx.bridge("finished")
            self.assertTrue(latest.result(timeout=1))
        self.assertEqual([row["old_agent_id"] for row in self.r.retirement_events], ["old", "new"])
        proof = self.r.retired["new"]
        refused = [row for row in self.r.request_records if row.get("agent_id") == "new"]
        self.assertEqual([row["event"] for row in refused], ["started", "failed"])
        self.assertFalse(refused[-1]["provider_attempted"])
        self.assertFalse(refused[-1]["upstream_bridge_attempted"])
        self.assertEqual(proof["old_unforwarded_request_ids"], [refused[0]["request_id"]])
        self.assertEqual(set(self.r.mirrors), {"old", "new", "third"})
        self.assertEqual(len(self.fx.upstream.calls), 1)
        self.assertTrue(self.r._bridge_all_terminal())
        with self.assertRaisesRegex(StopRejected, "native_retired_identity_denied"):
            self.fx.send("new")

    def test_old_model_request_denial_latches_failure_and_preserves_refusal(self):
        self.fx.retire()
        calls = len(self.fx.upstream.calls)
        with self.assertRaisesRegex(StopRejected, "native_retired_identity_denied"):
            self.fx.send("old")
        self.assertEqual(len(self.fx.upstream.calls), calls)
        self.assertEqual(self.r.retirement_events[-1]["event"], "denied_model")
        self.assertEqual(self.r.error, "native_retired_identity_denied")
        refusal = self.r.request_records[-1]
        self.assertEqual(refusal["route"], "unclassified")
        self.assertFalse(refusal["upstream_bridge_attempted"])
        with self.assertRaisesRegex(StopRejected, "native_census_unavailable"):
            self.fx.send()

    def test_all_old_tool_names_are_denied_before_effects(self):
        self.fx.retire()
        for index, name in enumerate(("Workflow", "StructuredOutput", "TaskOutput", "TaskList", "TaskGet", "Bash", "Read")):
            identity = "old-tool-" + str(index)
            reply = self.fx.fixture.hook("PreToolUse", name, {}, identity, "old")
            self.assertEqual(reply["hookSpecificOutput"]["permissionDecision"], "deny")
            self.assertNotIn(identity, self.r.tools)
            self.assertNotIn(identity, self.r.completed_tools)
        self.assertEqual([event["event"] for event in self.r.retirement_events], ["fence_installed"] + ["denied_tool"] * 7)

    def test_tool_admission_rechecks_fence_after_async_identity_wait(self):
        original = self.r._wait_hook_identity
        def retire_after_identity(data, tool_id):
            original(data, tool_id)
            self.fx.retire()
        with patch.object(self.r, "_wait_hook_identity", side_effect=retire_after_identity):
            reply = self.fx.fixture.hook("PreToolUse", "Bash", {"command": "true"}, "race-tool", "old")
        self.assertEqual(reply["hookSpecificOutput"]["permissionDecision"], "deny")
        self.assertEqual(self.r.retirement_events[-1]["event"], "denied_tool")
        self.assertNotIn("race-tool", self.r.tools)
        self.assertNotIn("race-tool", self.r.completed_tools)

    def test_every_superseded_pending_tool_denial_retains_its_identity(self):
        self.fx.replace()
        self.fx.fixture.nodes = [{**self.fx.new, "attempt": 3, "agentId": "third", "state": "start",
                                  "label": self.fx.old["label"] + " (retry 2)"}]
        self.fx.fixture.progress()
        for index, name in enumerate(("TaskGet", "Bash", "StructuredOutput")):
            tool_id = "pending-denial-" + str(index)
            reply = self.fx.fixture.hook("PreToolUse", name, {}, tool_id, "new")
            self.assertEqual(reply["hookSpecificOutput"]["permissionDecision"], "deny")
            event = self.r.retirement_events[-1]
            self.assertEqual((event["event"], event["old_agent_id"], event["new_agent_id"]), ("denied_tool", "new", "third"))
            self.assertEqual(event["detail"]["tool_use_id"], tool_id)
        self.assertEqual(len(self.r.retirement_events), 3)
        self.assertEqual(self.r.error, "native_superseded_tool_denied")
        self.assertEqual(self.r.tools, {})
        self.assertEqual(self.r.completed_tools, {})

    def test_old_auxiliary_and_malformed_requests_hit_the_fence(self):
        self.fx.retire()
        for method, path, body in (("GET", "/v1/models", b""), ("POST", "/v1/messages", b"invalid")):
            with self.assertRaisesRegex(StopRejected, "native_retired_identity_denied"):
                self.fx.send("old", method=method, path=path, body=body)
        self.assertEqual(len(self.fx.upstream.calls), 1)
        self.assertEqual([row["event"] for row in self.r.retirement_events], ["fence_installed", "denied_model", "denied_model"])

    def test_mixed_case_duplicate_auxiliary_identity_cannot_hide_old_agent(self):
        self.fx.retire()
        headers = self.fx.headers("old") + [("X-Claude-Code-Agent-Id", "new")]
        with self.assertRaisesRegex(StopRejected, "duplicate_native_identity_header"):
            with self.fx.transport.send("GET", "/v1/models", headers, b""):
                self.fail("The duplicate identity reached the provider.")
        self.assertEqual(len(self.fx.upstream.calls), 1)
        self.assertEqual(self.r.request_records[-1]["route"], "unclassified")

    def test_child_task_inspection_blocks_retirement_until_post_hook(self):
        self.r.retirement_wait_seconds = 2
        self.assertEqual(self.fx.fixture.hook("PreToolUse", "TaskGet", {"task_id": "synthetic"}, "inspect", "old"), {})
        self.fx.replace()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.fx.send)
            with self.r.condition:
                self.assertTrue(self.r.condition.wait_for(lambda: self.r.request_records, timeout=1))
            self.assertEqual(self.r.retired, {})
            self.fx.fixture.hook("PostToolUse", "TaskGet", {"task_id": "synthetic"}, "inspect", "old", response={})
            self.assertTrue(future.result(timeout=2))
        self.assertIn("inspect", self.r.completed_tools)

    def test_pending_successor_hook_uses_short_wait_and_returns_explicit_denial(self):
        self.r.wait_seconds = .03
        self.r.retirement_wait_seconds = 600
        self.fx.bridge("started")
        self.fx.replace()
        start = time.monotonic()
        reply = self.fx.fixture.hook("PreToolUse", "Bash", {"command": "true"}, "pending-tool", "new")
        self.assertLess(time.monotonic() - start, 1)
        self.assertEqual(reply["hookSpecificOutput"]["permissionDecision"], "deny")
        self.assertEqual(self.r.error, "native_retirement_timeout")
        self.assertEqual(self.r.tools, {})
        self.assertEqual(self.r.retired, {})

    def test_cancelled_hook_latches_failure_and_returns_explicit_denial(self):
        self.r.wait_seconds = .2
        data = {"session_id": "stop-session", "cwd": str(self.fx.fixture.root), "hook_event_name": "PreToolUse",
                "tool_name": "TaskGet", "tool_input": {}, "agent_id": None}
        async def exercise():
            task = asyncio.create_task(self.r.pre(data, "never-observed-root-tool"))
            await asyncio.sleep(.01)
            task.cancel()
            reply = await task
            self.assertEqual(reply["hookSpecificOutput"]["permissionDecision"], "deny")
        asyncio.run(exercise())
        self.assertEqual(self.r.error, "native_tool_admission_failed")
        self.assertEqual(self.r.tools, {})

    def test_worker_start_exception_returns_explicit_denial_before_admission(self):
        with patch("interface.native_cost_controls.quality_stop_native.asyncio.to_thread",
                   side_effect=RuntimeError("Synthetic worker start failure.")):
            reply = self.fx.fixture.hook("PreToolUse", "Bash", {"command": "true"}, "worker-failure", "old")
        self.assertEqual(reply["hookSpecificOutput"]["permissionDecision"], "deny")
        self.assertEqual(self.r.error, "native_tool_admission_failed")
        self.assertEqual(self.r.first_failure["exception"]["type"], "RuntimeError")
        self.assertEqual(self.r.first_failure["trigger"]["tool_use_id"], "worker-failure")
        self.assertEqual(self.r.tools, {})
        self.assertEqual(self.r.completed_tools, {})

    def test_pre_hook_entry_exception_returns_explicit_denial(self):
        reply = asyncio.run(self.r.pre(None))
        self.assertEqual(reply["hookSpecificOutput"]["permissionDecision"], "deny")
        self.assertEqual(self.r.error, "native_tool_admission_failed")
        self.assertEqual(self.r.first_failure["exception"]["type"], "AttributeError")
        self.assertEqual(self.r.tools, {})

    def test_post_hook_entry_exception_latches_unknown_outcome(self):
        self.assertEqual(asyncio.run(self.r.post(None)), {})
        self.assertEqual(self.r.error, "native_tool_outcome_unknown")
        self.assertEqual(self.r.first_failure["exception"]["type"], "AttributeError")

    def test_post_hook_exception_after_pop_latches_unknown_outcome(self):
        command = {"command": "true"}
        self.assertEqual(self.fx.fixture.hook("PreToolUse", "Bash", command, "post-failure", "old"), {})
        response = {"synthetic_response": True}
        original_copy = deepcopy
        def fail_response_copy(value, *args, **kwargs):
            if value is response:
                raise RuntimeError("Synthetic outcome copy failure.")
            return original_copy(value, *args, **kwargs)
        with patch("interface.native_cost_controls.quality_stop_native.deepcopy", side_effect=fail_response_copy):
            self.fx.fixture.hook("PostToolUse", "Bash", command, "post-failure", "old", response=response)
        self.assertNotIn("post-failure", self.r.tools)
        self.assertNotIn("post-failure", self.r.completed_tools)
        self.assertEqual(self.r.error, "native_tool_outcome_unknown")
        self.assertEqual(self.r.first_failure["exception"]["type"], "RuntimeError")
        with self.assertRaisesRegex(StopRejected, "native_census_unavailable"):
            self.fx.send("old")
        self.assertEqual(self.fx.upstream.calls, [])

    def test_trial_deadline_cannot_extend_replacement_wait(self):
        self.r.retirement_wait_seconds = 600
        self.fx.fixture.controller.deadline_epoch = time.time() - 1
        self.fx.replace()
        with self.assertRaisesRegex(StopRejected, "native_retirement_timeout"):
            self.fx.send()
        self.assertEqual(self.r.retired, {})
        self.assertEqual(self.fx.upstream.calls, [])

    def test_completed_old_structured_output_blocks_retirement(self):
        self.fx.fixture.hook("PreToolUse", "StructuredOutput", {}, "output", "old")
        self.fx.fixture.hook("PostToolUse", "StructuredOutput", {}, "output", "old", response={})
        self.fx.replace()
        with self.assertRaisesRegex(StopRejected, "retirement_old_structured_output_already_completed"):
            self.fx.send()
        self.assertEqual(self.r.retirement_events, [])
        self.assertEqual(self.fx.upstream.calls, [])

    def test_completed_old_journal_result_blocks_retirement(self):
        self.fx.fixture.append("result", "old", {})
        self.fx.replace()
        with self.assertRaisesRegex(StopRejected, "retirement_agent_already_has_result"):
            self.fx.send()
        self.assertEqual(self.r.retirement_events, [])

    def test_late_old_journal_result_fails_revalidation(self):
        self.fx.retire()
        self.fx.fixture.append("result", "old", {})
        with self.r.condition, self.assertRaisesRegex(StopRejected, "retired_agent_late_journal_result"):
            self.r._validate_retirements(self.r.journal.snapshot())

    def test_late_old_transcript_append_fails_revalidation(self):
        self.fx.retire()
        with self.fx.transcript.open("ab") as stream:
            stream.write(b'{"type":"assistant"}\n')
        with self.r.condition, self.assertRaisesRegex(StopRejected, "retired_agent_transcript_changed"):
            self.r._validate_retirements(self.r.journal.snapshot())

    def test_missing_control_frame_waits_without_forwarding(self):
        self.fx.write_transcript([self.fx.initial, self.fx.previous])
        self.fx.replace()
        with self.assertRaisesRegex(StopRejected, "native_retirement_timeout"):
            self.fx.send()
        self.assertEqual(self.fx.upstream.calls, [])

    def test_control_text_inside_tool_result_is_not_retirement_authority(self):
        fake = {"type": "user", "message": {"role": "user", "content": [{"type": "tool_result",
            "tool_use_id": "synthetic", "content": "[Request interrupted by user]"}]}}
        self.fx.write_transcript([self.fx.initial, self.fx.previous, fake])
        self.fx.replace()
        with self.assertRaisesRegex(StopRejected, "native_retirement_timeout"):
            self.fx.send()
        self.assertEqual(self.r.retirement_events, [])

    def test_changed_interruption_parent_rejects(self):
        self.fx.terminal["parentUuid"] = "66666666-6666-6666-6666-666666666666"
        self.fx.write_transcript()
        self.fx.replace()
        with self.assertRaisesRegex(StopRejected, "retirement_interruption_entry_changed"):
            self.fx.send()

    def test_unsupported_replacement_nodes_never_admit_successor(self):
        for change in ({"attempt": 3}, {"phaseIndex": 2}, {"phaseIndex": True}, {"phaseIndex": 1.0},
                       {"index": 2}, {"label": "changed"},
                       {"agentId": "old"}, {"state": "running"}, {"state": "done"}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                fx = RetirementFixture(directory)
                try:
                    fx.new.update(change)
                    fx.replace()
                    self.assertEqual(fx.registry.error, "native_census_event_conflict")
                    with self.assertRaises(StopRejected):
                        fx.send()
                    self.assertEqual(fx.upstream.calls, [])
                    self.assertEqual(fx.registry.retirement_events, [])
                finally:
                    fx.transport.close()

    def test_two_event_refusal_without_retirement_proof_cannot_close(self):
        self.r.record_transport({"event": "started", "request_id": "unknown-refusal", "agent_id": "old"})
        self.r.record_transport({"event": "failed", "request_id": "unknown-refusal", "agent_id": "old",
            "route": "unclassified", "upstream_bridge_attempted": False, "provider_attempted": False,
            "reason": "native_identity_unavailable"})
        with self.assertRaisesRegex(StopRejected, "native_closure_unqualified_local_refusal"):
            self.r._bridge_all_terminal()

    def test_mismatched_identity_evidence_never_admits_successor(self):
        for field in ("task", "prompt", "session", "descriptor", "key"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                fx = RetirementFixture(directory)
                try:
                    fx.replace()
                    if field == "task":
                        fx.registry.mirrors["new"]["task"] = "Changed task."
                    elif field == "prompt":
                        fx.registry.mirror_inputs["new"]["initial_entry"]["promptId"] = "66666666-6666-6666-6666-666666666666"
                    elif field == "session":
                        fx.registry.mirror_inputs["new"]["initial_entry"]["sessionId"] = "other-session"
                    elif field == "descriptor":
                        fx.registry.mirror_inputs["new"]["descriptor"]["project_key"] = "other-project"
                    else:
                        rows = [json.loads(line) for line in fx.fixture.journal.read_bytes().splitlines()]
                        rows[-1]["key"] = "v2:" + "0" * 64
                        fx.fixture.journal.write_bytes(canonical_bytes(rows))
                    with self.assertRaises(StopRejected):
                        fx.send()
                    self.assertEqual(fx.upstream.calls, [])
                    self.assertEqual(fx.registry.retirement_events, [])
                finally:
                    fx.transport.close()

    def test_changed_control_frame_fields_never_admit_successor(self):
        variants = ({"sessionId": "other"}, {"agentId": "other"}, {"isSidechain": False},
                    {"version": "other"}, {"userType": "ordinary"}, {"entrypoint": "cli"},
                    {"extra": True}, {"timestamp": "invalid"}, {"timestamp": "2026-09-27T20:00:00"})
        for change in variants:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                fx = RetirementFixture(directory)
                try:
                    fx.terminal.update(change)
                    fx.write_transcript()
                    fx.replace()
                    with self.assertRaises(StopRejected):
                        fx.send()
                    self.assertEqual(fx.upstream.calls, [])
                finally:
                    fx.transport.close()

    def test_hardlink_transcript_rejects(self):
        os.link(self.fx.transcript, self.fx.transcript.with_name("extra-link"))
        self.fx.replace()
        with self.assertRaisesRegex(StopRejected, "retirement_transcript_identity_invalid"):
            self.fx.send()

    def test_symlink_transcript_rejects(self):
        original = self.fx.transcript.with_name("original")
        self.fx.transcript.rename(original)
        self.fx.transcript.symlink_to(original)
        self.fx.replace()
        with self.assertRaises(OSError):
            self.fx.send()
        self.assertEqual(self.r.error, "native_retirement_evidence_failed")

    def test_retirement_ledger_failure_never_admits_successor(self):
        self.fx.replace()
        foreign = self.fx.fixture.root / "foreign-ledger"
        foreign.write_bytes(b"unchanged")
        (self.fx.fixture.controller.state_dir / "native_retirements.jsonl").symlink_to(foreign)
        with self.assertRaises(OSError):
            self.fx.send()
        self.assertEqual(self.fx.upstream.calls, [])
        self.assertEqual(self.r.retired, {})
        self.assertEqual(self.r.error, "native_retirement_ledger_unavailable")
        self.assertEqual(foreign.read_bytes(), b"unchanged")

    def test_headerless_child_request_cannot_become_root(self):
        self.fx.retire()
        body = canonical({"messages": [{"role": "user", "content": [{"type": "text", "text": self.fx.task}]}]}).encode()
        with self.assertRaisesRegex(StopRejected, "native_root_input_changed"):
            with self.fx.transport.send("POST", "/v1/messages", [("Content-Type", "application/json")], body):
                self.fail("The child input acquired root authority.")
        self.assertEqual(len(self.fx.upstream.calls), 1)

    def test_only_one_exact_initial_bootstrap_request_can_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            f = NativeFixture(directory, populate=False)
            upstream = Upstream()
            transport = CheckpointTransport(upstream, f.registry, base_path="/api")
            try:
                with transport.send("HEAD", "/api/api/hello", [], b""):
                    pass
                self.assertEqual(len(upstream.calls), 1)
                self.assertTrue(f.registry.bootstrap_admitted)
                with self.assertRaisesRegex(StopRejected, "native_root_endpoint_unsupported"):
                    with transport.send("HEAD", "/api/api/hello", [], b""):
                        self.fail("A second bootstrap request reached the provider.")
            finally:
                transport.close()

    def test_fixed_bootstrap_path_also_works_without_endpoint_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            f = NativeFixture(directory, populate=False)
            upstream = Upstream()
            transport = CheckpointTransport(upstream, f.registry)
            try:
                with transport.send("HEAD", "/api/hello", [], b""):
                    pass
                self.assertEqual(len(upstream.calls), 1)
            finally:
                transport.close()

    def test_root_workflow_post_accepts_only_the_exact_resolved_source(self):
        for changed in (False, True):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                f = NativeFixture(directory, populate=False)
                try:
                    f.start()
                    script = (f.source / "kernel_workflow.js").read_bytes().decode("utf-8")
                    f.hook("PostToolUse", "Workflow", {**f.request, "script": script + ("changed" if changed else "")},
                           "root-tool", response={})
                    if changed:
                        self.assertEqual(f.registry.error, "native_tool_outcome_unknown")
                        self.assertEqual(f.registry.first_failure["exception"]["code"], "native_root_resolved_script_changed")
                    else:
                        self.assertIsNone(f.registry.error)
                        self.assertIn("root-tool", f.registry.root_tool_completed)
                finally:
                    f.registry.close()

    def test_bootstrap_rejects_unqualified_paths_bodies_and_native_headers(self):
        variants = [("HEAD", "/api/api/hello?extra=1", [], b""), ("GET", "/api/api/hello", [], b""),
                    ("HEAD", "/api/api/hello", [], b"extra"), ("HEAD", "/api/models", [], b""),
                    ("HEAD", "/api/api/hello", [("x-claude-code-session-id", "stop-session")], b"")]
        for method, target, headers, body in variants:
            with self.subTest(target=target, method=method, body=body), tempfile.TemporaryDirectory() as directory:
                f = NativeFixture(directory, populate=False)
                upstream = Upstream()
                transport = CheckpointTransport(upstream, f.registry, base_path="/api")
                try:
                    with self.assertRaisesRegex(StopRejected, "native_root_endpoint_unsupported"):
                        with transport.send(method, target, headers, body):
                            self.fail("An unknown bootstrap request reached the provider.")
                    self.assertEqual(upstream.calls, [])
                finally:
                    transport.close()

    def test_bootstrap_rechecks_after_concurrent_root_forwarding(self):
        with tempfile.TemporaryDirectory() as directory:
            f = NativeFixture(directory, populate=False)
            upstream = Upstream()
            transport = CheckpointTransport(upstream, f.registry, base_path="/api")
            checked, root_forwarded = threading.Event(), threading.Event()
            original = f.registry._bootstrap
            calls = []
            def interleaved_check(*args):
                original(*args)
                calls.append(True)
                if len(calls) == 1:
                    checked.set()
                    f.registry.condition.release()
                    try:
                        self.assertTrue(root_forwarded.wait(1))
                    finally:
                        f.registry.condition.acquire()
            def probe():
                with self.assertRaisesRegex(StopRejected, "native_root_endpoint_unsupported"):
                    with transport.send("HEAD", "/api/api/hello", [], b""):
                        self.fail("The probe forwarded after root inference.")
            try:
                with patch.object(f.registry, "_bootstrap", side_effect=interleaved_check), ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(probe)
                    self.assertTrue(checked.wait(1))
                    body = canonical({"messages": [{"role": "user", "content": [{"type": "text", "text": f.registry.root_prompt}]}]}).encode()
                    with transport.send("POST", "/api/v1/messages", [("Content-Type", "application/json")], body):
                        pass
                    root_forwarded.set()
                    future.result(timeout=1)
                self.assertEqual(len(upstream.calls), 1)
                self.assertEqual(upstream.calls[0][0], "POST")
                self.assertFalse(f.registry.bootstrap_admitted)
            finally:
                root_forwarded.set()
                transport.close()

    def test_root_continuation_accepts_only_qualified_notice_representation_change(self):
        notice = "Synthetic registered root notice."
        initial = [{"role": "user", "content": [{"type": "text", "text": self.r.root_prompt}]},
                   {"role": "system", "content": [{"type": "text", "text": notice,
                                                    "cache_control": {"type": "ephemeral"}}]}]
        continuation = [deepcopy(initial[0]), {"role": "system", "content": notice},
                        {"role": "assistant", "content": [{"type": "text", "text": "Synthetic continuation."}]}]
        bodies = [canonical({"messages": value}).encode() for value in (initial, continuation)]
        with patch("interface.native_cost_controls.quality_stop_native.ROOT_ULTRACODE_NOTICE_SHA256",
                   hashlib.sha256(notice.encode()).hexdigest()):
            for body in bodies:
                with self.fx.transport.send("POST", "/v1/messages", [("Content-Type", "application/json")], body):
                    pass
            self.assertEqual([call[3] for call in self.fx.upstream.calls], bodies)
            changed = deepcopy(continuation)
            changed[1]["content"] += " changed"
            with self.assertRaisesRegex(StopRejected, "native_root_input_changed"):
                self.r.response([], canonical({"messages": changed}).encode())
            changed = deepcopy(continuation)
            changed[0]["content"][0]["text"] += " changed"
            with self.assertRaisesRegex(StopRejected, "native_root_input_changed"):
                self.r.response([], canonical({"messages": changed}).encode())
            changed = deepcopy(initial)
            del changed[1]["content"][0]["cache_control"]
            with self.assertRaisesRegex(StopRejected, "native_root_input_changed"):
                self.r.response([], canonical({"messages": changed}).encode())
            from interface.native_cost_controls.quality_stop_native import checkpoint_initial_messages
            self.assertFalse(checkpoint_initial_messages(initial, self.r.root_prompt))

    def test_headerless_old_tool_cannot_become_root(self):
        self.fx.retire()
        reply = self.fx.fixture.hook("PreToolUse", "TaskGet", {}, "missing-child-identity")
        self.assertEqual(reply["hookSpecificOutput"]["permissionDecision"], "deny")
        self.assertEqual(self.r.error, "native_tool_admission_failed")
        self.assertEqual(self.r.tools, {})

    def test_root_hook_can_wait_for_message_without_blocking_event_loop(self):
        self.r.wait_seconds = 1
        data = {"session_id": "stop-session", "cwd": str(self.fx.fixture.root), "hook_event_name": "PreToolUse",
                "tool_name": "TaskOutput", "tool_input": {"task_id": "root-task"}, "agent_id": None}
        async def exercise():
            task = asyncio.create_task(self.r.pre(data, "root-inspection"))
            await asyncio.sleep(.02)
            self.assertFalse(task.done())
            self.r.observe(AssistantMessage(content=[{"id": "root-inspection", "name": "TaskOutput",
                                                     "input": data["tool_input"]}]))
            self.assertEqual(await asyncio.wait_for(task, 1), {})
        asyncio.run(exercise())
        self.assertEqual(self.r.tools, {})


if __name__ == "__main__":
    unittest.main()
