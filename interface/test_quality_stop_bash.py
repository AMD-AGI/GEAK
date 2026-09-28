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

    def test_explicit_background_request_is_denied_before_execution(self):
        result = self.f.hook("PreToolUse", "Bash", {**self.inputs, "run_in_background": True}, "explicit", "engineer")
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


if __name__ == "__main__":
    unittest.main()
