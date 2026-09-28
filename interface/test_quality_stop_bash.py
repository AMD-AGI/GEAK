# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qualify automatic Bash task bindings without changing native timeouts."""

import tempfile
import unittest
from copy import deepcopy

from interface.native_cost_controls.quality_stop_controller import StopRejected
from interface.test_quality_stop_native import (
    NativeFixture, TaskStartedMessage, TaskUpdatedMessage, TaskNotificationMessage,
)


class AutomaticBashTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.f = NativeFixture(self.tmp.name)
        self.r = self.f.registry
        self.task = "b12345678"
        self.tool = "bash-tool"
        self.inputs = {"command": "sleep 135", "description": "Exercise native timeout"}
        self.response = {"stdout": "", "stderr": "", "interrupted": False, "isImage": False,
                         "noOutputExpected": False, "backgroundTaskId": self.task, "timedOutAfterMs": 120000}
        self.pre()

    def pre(self):
        return self.f.hook("PreToolUse", "Bash", self.inputs, self.tool, "engineer")

    def start(self, **changes):
        fields = {"task_id": self.task, "tool_use_id": self.tool, "task_type": "local_bash"}
        fields.update(changes)
        self.r.observe(TaskStartedMessage(**fields))

    def post(self, **changes):
        response = {**self.response, **changes}
        return self.f.hook("PostToolUse", "Bash", self.inputs, self.tool, "engineer", response=response)

    def terminal(self, status="completed", *, kind="updated", **changes):
        fields = {"task_id": self.task, "tool_use_id": None, "task_type": None, "status": status}
        if kind == "updated":
            fields["patch"] = {"status": status, "end_time": 1790612000000,
                               "result": {"code": 0, "interrupted": False}}
            cls = TaskUpdatedMessage
        else:
            fields["tool_use_id"] = self.tool
            cls = TaskNotificationMessage
        fields.update(changes)
        self.r.observe(cls(**fields))

    def assert_blocked(self):
        with self.assertRaisesRegex(StopRejected, "native_background_producer_active"):
            self.r.assert_quiescent("checkpoint")

    def test_start_before_automatic_result_binds_original_input_and_terminal(self):
        self.start()
        self.assert_blocked()
        self.post()
        self.assert_blocked()
        self.terminal()
        self.r.assert_quiescent("checkpoint")
        proof = self.r._automatic_bash_proof()[self.task]
        self.assertEqual([e["kind"] for e in proof["events"]],
                         ["TaskStartedMessage", "PostToolUse", "TaskUpdatedMessage"])
        self.assertEqual(proof["agent_id"], "engineer")
        self.assertEqual(proof["tool_use_id"], self.tool)
        self.assertEqual(proof["result"]["input_sha256"], proof["input_sha256"])
        self.assertEqual(self.r.completed_tools[self.tool]["input"], self.inputs)
        self.assertIsNone(self.r.error)

    def test_automatic_result_before_start_stays_active_until_complete_binding(self):
        self.post()
        self.assertEqual(self.r.background[self.task], "active")
        self.assert_blocked()
        self.start()
        self.assert_blocked()
        self.terminal(kind="notification")
        self.r.assert_quiescent("checkpoint")
        self.assertEqual([e["kind"] for e in self.r._automatic_bash_proof()[self.task]["events"]],
                         ["PostToolUse", "TaskStartedMessage", "TaskNotificationMessage"])

    def test_foreground_registration_and_completion_remain_supported(self):
        self.start()
        self.terminal(kind="notification")
        self.f.hook("PostToolUse", "Bash", self.inputs, self.tool, "engineer",
                    response={"stdout": "complete", "stderr": "", "interrupted": False})
        self.r.assert_quiescent("checkpoint")
        self.assertEqual(self.r._automatic_bash_proof(), {})

    def test_foreground_post_can_precede_task_start(self):
        self.f.hook("PostToolUse", "Bash", self.inputs, self.tool, "engineer", response={})
        self.start()
        self.terminal(kind="notification")
        self.r.assert_quiescent("checkpoint")

    def test_two_native_terminal_kinds_require_equivalent_status(self):
        self.start()
        self.post()
        self.terminal("killed")
        self.terminal("stopped", kind="notification")
        self.assertEqual(self.r.background[self.task], "stopped")
        self.r.assert_quiescent("checkpoint")
        self.assertEqual(len(self.r._automatic_bash_proof()[self.task]["events"]), 4)

    def test_changed_terminal_status_fails_closed(self):
        self.start()
        self.post()
        self.terminal()
        self.terminal("failed", kind="notification")
        self.assertEqual(self.r.first_failure["exception"]["code"], "background_bash_terminal_changed")

    def test_duplicate_terminal_kind_fails_closed(self):
        self.start()
        self.post()
        self.terminal()
        self.terminal()
        self.assertEqual(self.r.first_failure["exception"]["code"], "background_bash_duplicate_terminal")

    def test_running_patch_does_not_clear_active_state(self):
        self.start()
        self.post()
        self.terminal("running")
        self.assert_blocked()
        self.assertIsNone(self.r.error)

    def test_late_running_patch_cannot_reopen_terminal_task(self):
        self.start()
        self.post()
        self.terminal()
        self.terminal("running")
        self.assertEqual(self.r.first_failure["exception"]["code"], "background_bash_terminal_changed")

    def test_patch_without_status_does_not_clear_active_state(self):
        self.start()
        self.post()
        self.terminal(None, patch={"notified": True})
        self.assert_blocked()
        self.terminal()
        self.terminal(None, patch={"notified": True})
        self.r.assert_quiescent("checkpoint")

    def test_terminal_before_start_is_an_orphan_even_after_result(self):
        self.post()
        self.terminal()
        self.assertEqual(self.r.first_failure["exception"]["code"], "background_bash_orphan_event")
        self.assertEqual(self.r.background[self.task], "active")

    def test_terminal_before_automatic_result_preserves_closed_task_and_active_tool(self):
        self.start()
        self.terminal()
        self.assert_blocked()
        self.post()
        self.assertEqual(self.r.background[self.task], "completed")
        self.r.assert_quiescent("checkpoint")
        proof = self.r._automatic_bash_proof()[self.task]
        self.assertEqual([event["kind"] for event in proof["events"]],
                         ["TaskStartedMessage", "TaskUpdatedMessage", "PostToolUse"])

    def test_start_must_name_one_admitted_bash_tool(self):
        self.start(tool_use_id="unknown")
        self.assertEqual(self.r.first_failure["exception"]["code"], "background_bash_binding_changed")

    def test_one_tool_cannot_bind_two_task_ids(self):
        self.start()
        self.start(task_id="b87654321")
        self.assertEqual(self.r.first_failure["exception"]["code"], "background_bash_binding_changed")

    def test_one_task_cannot_bind_two_tools(self):
        self.start()
        self.f.hook("PreToolUse", "Bash", self.inputs, "other-tool", "engineer")
        self.start(tool_use_id="other-tool")
        self.assertEqual(self.r.first_failure["exception"]["code"], "background_bash_binding_changed")

    def test_event_identity_aliases_and_malformed_data_fail_closed(self):
        for changes in ({"data": None}, {"data": {"task_id": "b87654321"}}, {"task_id": "bad"},
                        {"task_type": "local_agent"}, {"data": {"agent_id": "other"}}, {"tool_use_id": []},
                        {"data": {"taskId": self.task}}, {"data": {"unserializable": object()}}):
            with self.subTest(changes=changes):
                self.r.error = self.r.first_failure = None
                self.start(**changes)
                self.assertEqual(self.r.error, "native_census_event_conflict")

    def test_terminal_tool_agent_and_patch_must_match(self):
        self.start()
        self.post()
        for changes in ({"tool_use_id": "other"}, {"data": {"agent_id": "other"}},
                        {"patch": {"status": "failed"}}, {"patch": None}, {"status": "cancelled"}):
            with self.subTest(changes=changes):
                self.r.error = self.r.first_failure = None
                self.terminal(**changes)
                self.assertEqual(self.r.error, "native_census_event_conflict")

    def test_terminal_patch_cannot_change_any_native_identity(self):
        self.start()
        self.post()
        for key in ("task_id", "id", "tool_use_id", "agent_id", "session_id", "task_type", "type"):
            with self.subTest(key=key):
                self.r.error = self.r.first_failure = None
                self.terminal(patch={"status": "completed", key: "changed"})
                self.assertEqual(self.r.first_failure["exception"]["code"], "background_bash_binding_changed")
                self.assertEqual(self.r.background[self.task], "active")

    def test_terminal_patch_rejects_identity_aliases(self):
        self.start()
        self.post()
        for key in ("taskId", "toolUseId", "agentId", "sessionId", "taskType", "backgroundTaskId", "background_task_id"):
            with self.subTest(key=key):
                self.r.error = self.r.first_failure = None
                self.terminal(patch={"status": "completed", key: self.task})
                self.assertEqual(self.r.first_failure["exception"]["code"], "background_bash_event_invalid")

    def test_terminal_patch_preserves_matching_identity_and_nonidentity_metadata(self):
        self.start()
        self.post()
        self.terminal(patch={"status": "completed", "task_id": self.task, "id": self.task,
                            "tool_use_id": self.tool, "agent_id": "engineer", "session_id": "stop-session",
                            "task_type": "local_bash", "type": "local_bash", "end_time": 1790612000000,
                            "result": {"code": 0, "interrupted": False}, "notified": True})
        self.r.assert_quiescent("checkpoint")
        self.assertTrue(self.r._automatic_bash_proof()[self.task]["terminal"]["patch"]["notified"])

    def test_result_requires_pinned_automatic_timeout_shape(self):
        cases = ({"backgroundedByUser": True}, {"backgroundedByUser": False}, {"task_id": self.task},
                 {"taskId": self.task}, {"background_task_id": self.task}, {"backgroundTaskId": "bad"},
                 {"backgroundTaskId": None}, {"interrupted": True}, {"isImage": True},
                 {"timedOutAfterMs": 600000}, {"timedOutAfterMs": True}, {"timedOutAfterMs": None},
                 {"stdout": None}, {"stderr": None}, {"unknown": "field"}, {"noOutputExpected": "false"},
                 {"backgroundCwdHint": {}}, {"persistedOutputSize": -1}, {"dangerouslyDisableSandbox": 0})
        for changes in cases:
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as directory:
                f = NativeFixture(directory)
                f.hook("PreToolUse", "Bash", self.inputs, self.tool, "engineer")
                f.hook("PostToolUse", "Bash", self.inputs, self.tool, "engineer", response={**self.response, **changes})
                self.assertEqual(f.registry.error, "native_tool_outcome_unknown")
                self.assertEqual(f.registry.background, {})

    def test_nonboolean_background_request_is_denied_before_execution(self):
        result = self.f.hook("PreToolUse", "Bash", {**self.inputs, "run_in_background": "true"}, "explicit", "engineer")
        self.assertEqual(result["hookSpecificOutput"]["permissionDecision"], "deny")
        self.assertNotIn("explicit", self.r.tools)

    def test_explicit_false_retains_foreground_semantics(self):
        self.inputs["run_in_background"] = False
        self.r.tools[self.tool]["input"] = deepcopy(self.inputs)
        self.start()
        self.post()
        self.terminal()
        self.r.assert_quiescent("checkpoint")

    def test_requested_timeout_matches_native_clamp(self):
        self.inputs["timeout"] = 900000
        self.r.tools[self.tool]["input"] = deepcopy(self.inputs)
        self.start()
        self.post(timedOutAfterMs=600000)
        self.terminal()
        self.r.assert_quiescent("checkpoint")

    def test_failure_hook_cannot_claim_automatic_background(self):
        self.f.hook("PostToolUseFailure", "Bash", self.inputs, self.tool, "engineer", response=self.response)
        self.assertEqual(self.r.first_failure["exception"]["code"], "background_bash_result_unsupported")

    def test_background_metadata_without_task_id_fails_closed(self):
        self.f.hook("PostToolUse", "Bash", self.inputs, self.tool, "engineer",
                    response={"stdout": "", "stderr": "", "interrupted": False, "timedOutAfterMs": 120000})
        self.assertEqual(self.r.first_failure["exception"]["code"], "background_bash_result_unsupported")

    def test_result_cannot_switch_the_task_id_of_a_native_start(self):
        self.start()
        self.post(backgroundTaskId="b87654321")
        self.assertEqual(self.r.first_failure["exception"]["code"], "background_bash_binding_changed")

    def test_supplied_agent_id_and_duplicate_raw_event_fields_must_agree(self):
        self.start(data={"agent_id": "engineer", "task_id": self.task})
        self.post()
        self.terminal(data={"agent_id": "engineer", "patch": {"status": "failed"}})
        self.assertEqual(self.r.first_failure["exception"]["code"], "background_bash_event_invalid")

    def test_proof_rechecks_completed_input_and_response(self):
        self.start()
        self.post()
        self.terminal()
        for key in ("input", "response"):
            saved = deepcopy(self.r.completed_tools[self.tool][key])
            self.r.completed_tools[self.tool][key]["changed"] = True
            with self.assertRaisesRegex(StopRejected, "background_bash_binding_changed"):
                self.r.assert_quiescent("checkpoint")
            self.r.completed_tools[self.tool][key] = saved

    def test_closure_preserves_proof_and_requires_no_active_tasks(self):
        self.start()
        self.post()
        with self.assertRaisesRegex(StopRejected, "native_closure_producer_active"):
            self.r.confirm_return({"quality_stop": {"enabled": True}})
        self.terminal()
        self.f.finish("checkpoint", {"ok": True})
        self.f.progress()
        proof = self.r.confirm_return({"quality_stop": {"enabled": True}})
        self.assertEqual(proof["automatic_bash"][self.task], self.r.bash_tasks[self.task])
        self.assertEqual(proof["active_background"], [])
        self.assertGreater(proof["automatic_bash_closed_monotonic_time"],
                           max(event["monotonic_time"] for event in proof["automatic_bash"][self.task]["events"]))

    def test_foreground_only_closure_does_not_claim_an_automatic_clock(self):
        self.f.hook("PostToolUse", "Bash", self.inputs, self.tool, "engineer", response={})
        self.f.finish("checkpoint", {"ok": True})
        self.f.progress()
        census = self.r.confirm_return({"quality_stop": {"enabled": True}})
        self.assertNotIn("automatic_bash", census)
        self.assertNotIn("automatic_bash_closed_monotonic_time", census)

    def test_missing_binding_cannot_be_hidden_by_terminal_status_map(self):
        self.post()
        self.r.background[self.task] = "completed"
        with self.assertRaisesRegex(StopRejected, "background_bash_binding_incomplete"):
            self.r.assert_quiescent("checkpoint")


class ExplicitBashTests(unittest.TestCase):
    def fixture(self, directory):
        f = NativeFixture(directory)
        inputs = {"command": "sleep 20", "description": "Run an explicit background worker", "run_in_background": True}
        response = {"stdout": "", "stderr": "", "interrupted": False, "isImage": False,
                    "noOutputExpected": False, "backgroundTaskId": "b12345678"}
        reply = f.hook("PreToolUse", "Bash", inputs, "explicit-tool", "engineer")
        self.assertEqual(reply, {})
        return f, inputs, response

    def apply(self, f, inputs, response, event):
        if event == "start":
            f.registry.observe(TaskStartedMessage(task_id="b12345678", tool_use_id="explicit-tool", task_type="local_bash"))
        elif event == "post":
            f.hook("PostToolUse", "Bash", inputs, "explicit-tool", "engineer", response=response)
        else:
            f.registry.observe(TaskNotificationMessage(task_id="b12345678", tool_use_id="explicit-tool",
                                                       task_type="local_bash", status="completed"))

    def test_three_explicit_orders_block_until_the_binding_is_complete(self):
        for order in (("start", "post", "terminal"), ("post", "start", "terminal"), ("start", "terminal", "post")):
            with self.subTest(order=order), tempfile.TemporaryDirectory() as directory:
                f, inputs, response = self.fixture(directory)
                for event in order[:-1]:
                    self.apply(f, inputs, response, event)
                    with self.assertRaisesRegex(StopRejected, "native_background_producer_active"):
                        f.registry.assert_quiescent("checkpoint")
                self.apply(f, inputs, response, order[-1])
                f.registry.assert_quiescent("checkpoint")
                proof = f.registry._automatic_bash_proof()["b12345678"]
                self.assertEqual(proof["mode"], "explicit_request")
                self.assertIsNone(proof["result"]["timed_out_after_ms"])
                self.assertEqual(f.registry.completed_tools["explicit-tool"]["input"], inputs)
                self.assertIsNone(f.registry.error)

    def test_explicit_response_requires_task_and_rejects_timeout_or_manual_metadata(self):
        cases = [None, {}, {"backgroundTaskId": "bad"}, {"timedOutAfterMs": 120000}, {"timedOutAfterMs": None},
                 {"backgroundedByUser": True}, {"backgroundedByUser": False}, {"code": 0},
                 {"task_id": "b12345678"}, {"interrupted": True}, {"isImage": True}]
        for change in cases:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                f, inputs, response = self.fixture(directory)
                value = change if change is None or change == {} else {**response, **change}
                f.hook("PostToolUse", "Bash", inputs, "explicit-tool", "engineer", response=value)
                self.assertEqual(f.registry.error, "native_tool_outcome_unknown")
                self.assertEqual(f.registry.first_failure["exception"]["code"], "background_bash_result_unsupported")

    def test_foreground_input_cannot_claim_an_explicit_result(self):
        with tempfile.TemporaryDirectory() as directory:
            f = NativeFixture(directory)
            inputs = {"command": "sleep 20", "run_in_background": False}
            f.hook("PreToolUse", "Bash", inputs, "explicit-tool", "engineer")
            f.hook("PostToolUse", "Bash", inputs, "explicit-tool", "engineer", response={
                "stdout": "", "stderr": "", "interrupted": False, "backgroundTaskId": "b12345678"})
            self.assertEqual(f.registry.error, "native_tool_outcome_unknown")

    def test_failed_post_cannot_admit_an_explicit_task(self):
        with tempfile.TemporaryDirectory() as directory:
            f, inputs, response = self.fixture(directory)
            f.hook("PostToolUseFailure", "Bash", inputs, "explicit-tool", "engineer", response=response)
            self.assertEqual(f.registry.error, "native_tool_outcome_unknown")

    def test_explicit_terminal_before_start_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            f, inputs, response = self.fixture(directory)
            self.apply(f, inputs, response, "post")
            self.apply(f, inputs, response, "terminal")
            self.assertEqual(f.registry.first_failure["exception"]["code"], "background_bash_orphan_event")

    def test_explicit_duplicate_events_fail_closed(self):
        for duplicate, code in (("start", "duplicate_background_task_start"), ("terminal", "background_bash_duplicate_terminal")):
            with self.subTest(duplicate=duplicate), tempfile.TemporaryDirectory() as directory:
                f, inputs, response = self.fixture(directory)
                for event in ("start", "post", "terminal") if duplicate == "terminal" else ("start",):
                    self.apply(f, inputs, response, event)
                self.apply(f, inputs, response, duplicate)
                self.assertEqual(f.registry.first_failure["exception"]["code"], code)

    def test_explicit_timeout_input_does_not_create_timeout_result_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            f, inputs, response = self.fixture(directory)
            inputs["timeout"] = 400000
            f.registry.tools["explicit-tool"]["input"] = deepcopy(inputs)
            for event in ("start", "post", "terminal"):
                self.apply(f, inputs, response, event)
            f.registry.assert_quiescent("checkpoint")
            self.assertIsNone(f.registry._automatic_bash_proof()["b12345678"]["result"]["timed_out_after_ms"])

    def test_mixed_automatic_and_explicit_tasks_share_the_closure_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            f, inputs, response = self.fixture(directory)
            auto_inputs = {"command": "sleep 135"}
            f.hook("PreToolUse", "Bash", auto_inputs, "automatic-tool", "engineer")
            f.registry.observe(TaskStartedMessage(task_id="b87654321", tool_use_id="automatic-tool", task_type="local_bash"))
            f.hook("PostToolUse", "Bash", auto_inputs, "automatic-tool", "engineer", response={
                "stdout": "", "stderr": "", "interrupted": False, "backgroundTaskId": "b87654321", "timedOutAfterMs": 120000})
            f.registry.observe(TaskNotificationMessage(task_id="b87654321", tool_use_id="automatic-tool",
                                                       task_type="local_bash", status="completed"))
            self.apply(f, inputs, response, "start")
            self.apply(f, inputs, response, "post")
            with self.assertRaisesRegex(StopRejected, "native_background_producer_active"):
                f.registry.assert_quiescent("checkpoint")
            self.apply(f, inputs, response, "terminal")
            f.registry.assert_quiescent("checkpoint")
            proof = f.registry._automatic_bash_proof()
            self.assertEqual(set(proof), {"b12345678", "b87654321"})
            self.assertNotIn("mode", proof["b87654321"])
            self.assertEqual(proof["b12345678"]["mode"], "explicit_request")

    def test_explicit_closure_has_the_same_clock_and_rechecks_its_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            f, inputs, response = self.fixture(directory)
            for event in ("start", "post", "terminal"):
                self.apply(f, inputs, response, event)
            proof = f.registry.bash_tasks["b12345678"]
            proof["mode"] = "automatic"
            with self.assertRaisesRegex(StopRejected, "background_bash_binding_changed"):
                f.registry.assert_quiescent("checkpoint")
            proof["mode"] = "explicit_request"
            f.finish("checkpoint", {"ok": True})
            f.progress()
            census = f.registry.confirm_return({"quality_stop": {"enabled": True}})
            self.assertEqual(census["automatic_bash"]["b12345678"]["mode"], "explicit_request")
            self.assertGreater(census["automatic_bash_closed_monotonic_time"], proof["terminal"]["monotonic_time"])


if __name__ == "__main__":
    unittest.main()
