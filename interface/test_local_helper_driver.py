# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test local helper ownership without a model, network, GPU, or shell."""

import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from interface.native_cost_controls import native_envelope
from interface.native_cost_controls.helper_driver import (
    LocalHelperDriver,
    Unsupported,
    digest,
    json_values_equal,
    project,
)
from interface.native_cost_controls.native_preamble import BASH_LIBRARY_WARNING
from interface.test_system_envelope import SYNTHETIC_SYSTEM, synthetic_system_policy


class HelperDriverTests(unittest.TestCase):
    def setUp(self):
        self.system_policy = synthetic_system_policy()
        self.system_policy.start()
        self.temp = tempfile.TemporaryDirectory(prefix="geak-helper-")
        self.root = Path(self.temp.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.source = self.root / "workflow.js"
        self.source.write_text("// Explicit test workflow source.\n")
        stat = self.workspace.stat()
        self.schema = {
            "type": "object",
            "properties": {"epoch": {"type": "number"}},
            "required": ["epoch"],
            "additionalProperties": True,
        }
        self.binding = {
            "operation_id": "a" * 64,
            "role": "clock_reader",
            "gate": True,
            "command": "date +%s",
            "command_sha256": hashlib.sha256(b"date +%s").hexdigest(),
            "schema": self.schema,
            "schema_sha256": digest(self.schema),
            "prompt": "Read the clock once.",
            "workspace": str(self.workspace),
            "workspace_identity": [stat.st_dev, stat.st_ino],
            "source_bindings": {
                str(self.source): hashlib.sha256(self.source.read_bytes()).hexdigest()
            },
            "native_session": "public-fixture-session",
            "native_preamble": None,
            "completion_marker": None,
        }
        self.driver = LocalHelperDriver(self.root / "ledger", self.binding)
        self.driver.attest_dispatch("root/child", "root/child")
        self.body = {
            "system": deepcopy(SYNTHETIC_SYSTEM),
            "metadata": {"user_id": json.dumps({"session_id": "public-fixture-session"})},
            "tools": [{"name": "StructuredOutput", "input_schema": self.schema}],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": self.binding["prompt"]}]}
            ],
        }

    def tearDown(self):
        self.system_policy.stop()
        self.temp.cleanup()

    def complete_bash(self, stdout="42\n"):
        first = self.driver.response(self.body, self.binding)
        self.assertEqual(first["name"], "Bash")
        self.driver.before_bash(first["id"], first["input"])
        native = {"stdout": stdout, "stderr": "", "interrupted": False, "isImage": False}
        self.driver.capture_bash(first["id"], native)
        self.body["messages"].append({"role": "assistant", "content": [deepcopy(first)]})
        self.body["messages"].append(
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": first["id"], "content": stdout, "is_error": False}
            ]}
        )
        return first

    def finish_native(self, value):
        state = json.loads(self.driver.path.read_bytes())
        evidence = {"source": "native_workflow_journal", "status": "done",
            "native_session": self.binding["native_session"], "operation_id": self.binding["operation_id"],
            "entry_sha256": hashlib.sha256(state["structured_id"].encode()).hexdigest()}
        self.driver.confirm_completion(value, evidence)

    def test_native_result_requires_its_own_structured_acknowledgement(self):
        self.complete_bash()
        result = self.driver.response(self.body, self.binding)
        self.assertEqual(result["name"], "StructuredOutput")
        self.assertEqual(result["input"], {"epoch": 42})
        self.assertEqual(json.loads(self.driver.path.read_bytes())["stage"], "STRUCTURED_PENDING")
        self.driver.accept(result["id"], {"epoch": 42})
        self.assertEqual(json.loads(self.driver.path.read_bytes())["stage"], "ACKNOWLEDGED")
        self.finish_native({"epoch": 42})
        self.assertEqual(json.loads(self.driver.path.read_bytes())["stage"], "COMPLETE")

    def test_qualified_warning_preserves_raw_continuation_and_terminal_requirement(self):
        self.binding["native_preamble"] = BASH_LIBRARY_WARNING
        raw = BASH_LIBRARY_WARNING + "42\n"
        self.complete_bash(raw)
        state = json.loads(self.driver.path.read_bytes())
        self.assertEqual(state["native_result"]["stdout"], raw)
        self.assertTrue(state["recognized_native_preamble"])
        result = self.driver.response(self.body, self.binding)
        self.assertEqual(result["input"], {"epoch": 42})
        self.driver.accept(result["id"], result["input"])
        self.assertEqual(json.loads(self.driver.path.read_bytes())["stage"], "ACKNOWLEDGED")
        self.finish_native(result["input"])
        self.assertEqual(json.loads(self.driver.path.read_bytes())["stage"], "COMPLETE")

    def test_removing_the_warning_from_native_continuation_is_a_conflict(self):
        self.binding["native_preamble"] = BASH_LIBRARY_WARNING
        self.complete_bash(BASH_LIBRARY_WARNING + "42\n")
        self.body["messages"][-1]["content"][0]["content"] = "42\n"
        with self.assertRaises(Unsupported):
            self.driver.response(self.body, self.binding)
        state = json.loads(self.driver.path.read_bytes())
        self.assertEqual(state["stage"], "UNKNOWN")
        self.assertEqual(state["bash_emissions"], 1)

    def test_changed_warning_runtime_cannot_repeat_a_command(self):
        profile = {"prefix": BASH_LIBRARY_WARNING}
        self.binding.update(native_warning_profile=profile, native_preamble=BASH_LIBRARY_WARNING, native_shell="/bin/bash")
        with patch("interface.native_cost_controls.helper_driver.qualified_bash_warning", return_value=profile):
            self.complete_bash(BASH_LIBRARY_WARNING + "42\n")
        with patch("interface.native_cost_controls.helper_driver.qualified_bash_warning", return_value=None), self.assertRaises(Unsupported):
            self.driver.response(self.body, self.binding)
        state = json.loads(self.driver.path.read_bytes())
        self.assertEqual(state["stage"], "UNKNOWN")
        self.assertEqual(state["bash_emissions"], 1)

    def test_exact_warning_projects_each_nonstorage_role_without_changing_raw_output(self):
        cases = [("clock_reader", "42\n", {"epoch": 42}),
                 ("warm_start_resolver", '{"candidates":[]}\n', {"candidates": []}),
                 ("citation_writer", '{"citations":2}\n', {"filed": 2}),
                 ("experience_writer", '{"written":true}\n', {"written": True})]
        for role, text, expected in cases:
            with self.subTest(role=role):
                binding = {**self.binding, "role": role, "native_preamble": BASH_LIBRARY_WARNING}
                native = {"stdout": BASH_LIBRARY_WARNING + text, "stderr": "", "interrupted": False}
                original = deepcopy(native)
                self.assertEqual(project(binding, native), expected)
                self.assertEqual(native, original)

    def test_near_repeated_and_additional_warning_text_stays_unsupported(self):
        binding = {**self.binding, "role": "warm_start_resolver", "native_preamble": BASH_LIBRARY_WARNING}
        value = '{"candidates":[]}\n'
        cases = [BASH_LIBRARY_WARNING.replace("no version", "other version") + value,
                 BASH_LIBRARY_WARNING.rstrip() + value,
                 BASH_LIBRARY_WARNING + BASH_LIBRARY_WARNING + value,
                 "extra feedback\n" + BASH_LIBRARY_WARNING + value,
                 BASH_LIBRARY_WARNING + "extra feedback\n" + value,
                 BASH_LIBRARY_WARNING + value + "extra feedback\n"]
        for text in cases:
            with self.subTest(text=text), self.assertRaises((Unsupported, ValueError)):
                project(binding, {"stdout": text, "stderr": "", "interrupted": False})

    def test_completed_operation_replays_value_without_another_bash(self):
        self.complete_bash()
        result = self.driver.response(self.body, self.binding)
        self.driver.accept(result["id"], result["input"])
        self.finish_native(result["input"])
        replay = self.driver.response(self.body, self.binding)
        self.assertEqual(replay["name"], "StructuredOutput")
        self.assertNotEqual(result["id"], replay["id"])
        self.driver.accept(replay["id"], replay["input"])
        self.finish_native(replay["input"])
        state = json.loads(self.driver.path.read_bytes())
        self.assertEqual(state["bash_emissions"], 1)
        self.assertEqual(len(state["accepted_ids"]), 2)

    def test_repeat_before_native_result_cannot_repeat_the_command(self):
        self.driver.response(self.body, self.binding)
        with self.assertRaises(Unsupported) as caught:
            self.driver.response(self.body, self.binding)
        self.assertTrue(caught.exception.may_have_executed)
        original = b' {"native": "request"} '
        fallback = self.driver.fallback_decision(original, caught.exception)
        self.assertEqual(fallback["route"], "original_failure_path")
        self.assertIs(fallback["original_request_bytes"], original)
        self.assertFalse(fallback["caller_must_forward_unchanged"])

    def test_unsupported_task_before_emission_preserves_original_model_path(self):
        body = deepcopy(self.body)
        body["tools"][0]["input_schema"] = {"type": "string"}
        with self.assertRaises(Unsupported) as caught:
            self.driver.response(body, self.binding)
        self.assertFalse(caught.exception.may_have_executed)
        original = b' {"exact": [1, 2]}\n'
        fallback = self.driver.fallback_decision(original, caught.exception)
        self.assertEqual(fallback["route"], "unchanged_provider_passthrough")
        self.assertIs(fallback["original_request_bytes"], original)
        self.assertTrue(fallback["caller_must_forward_unchanged"])
        with self.assertRaises(Unsupported):
            self.driver.response(self.body, self.binding)

    def test_denied_native_permission_never_becomes_unrestricted_fallback(self):
        emitted = self.driver.response(self.body, self.binding)
        # The caller retains its original denial and does not call before_bash.
        decision = self.driver.fallback_decision(b"unchanged", Unsupported("permission_denied"))
        self.assertEqual(decision["route"], "original_failure_path")
        state = json.loads(self.driver.path.read_bytes())
        self.assertEqual(state["bash_id"], emitted["id"])
        self.assertNotIn("bash_started", state)

    def test_duplicate_native_bash_acknowledgement_is_rejected(self):
        first = self.driver.response(self.body, self.binding)
        self.driver.before_bash(first["id"], first["input"])
        with self.assertRaisesRegex(Unsupported, "duplicate_bash_execution"):
            self.driver.before_bash(first["id"], first["input"])

    def test_changed_source_before_dispatch_is_unsupported(self):
        self.source.write_text("// Changed workflow.\n")
        with self.assertRaisesRegex(Unsupported, "script_changed") as caught:
            self.driver.response(self.body, self.binding)
        self.assertFalse(caught.exception.may_have_executed)

    def test_changed_source_after_emission_cannot_reopen_fallback(self):
        self.driver.response(self.body, self.binding)
        self.source.write_text("// Changed workflow.\n")
        with self.assertRaises(Unsupported) as caught:
            self.driver.response(self.body, self.binding)
        self.assertTrue(caught.exception.may_have_executed)
        self.assertEqual(
            self.driver.fallback_decision(b"original", caught.exception)["route"],
            "original_failure_path",
        )

    def test_changed_native_result_latches_unknown(self):
        first = self.complete_bash()
        with self.assertRaisesRegex(Unsupported, "conflicting_native_result"):
            self.driver.capture_bash(
                first["id"], {"stdout": "43\n", "stderr": "", "interrupted": False}
            )
        self.assertEqual(json.loads(self.driver.path.read_bytes())["stage"], "UNKNOWN")

    def test_bool_is_not_accepted_as_a_numeric_result(self):
        self.complete_bash()
        result = self.driver.response(self.body, self.binding)
        with self.assertRaisesRegex(Unsupported, "accepted_value_changed"):
            self.driver.accept(result["id"], {"epoch": True})
        self.assertEqual(json.loads(self.driver.path.read_bytes())["stage"], "UNKNOWN")

    def test_finite_integer_and_float_json_values_match(self):
        self.complete_bash()
        result = self.driver.response(self.body, self.binding)
        self.driver.accept(result["id"], {"epoch": 42.0})
        self.finish_native({"epoch": 42.0})
        self.assertEqual(json.loads(self.driver.path.read_bytes())["stage"], "COMPLETE")

    def test_cross_session_request_is_unsupported(self):
        self.body["metadata"]["user_id"] = json.dumps({"session_id": "another-session"})
        with self.assertRaisesRegex(Unsupported, "native_session_changed"):
            self.driver.response(self.body, self.binding)
        self.assertFalse(self.driver.path.exists())

    def test_malformed_native_request_does_not_execute(self):
        for body in [None, {"metadata": {"user_id": "not json"}}, {"metadata": {}}]:
            with self.subTest(body=body), self.assertRaises(Unsupported):
                self.driver.response(body, self.binding)
        self.assertFalse(self.driver.path.exists())

    def test_untrusted_dispatch_is_rejected(self):
        with self.assertRaisesRegex(Unsupported, "untrusted_native_dispatch"):
            self.driver.attest_dispatch("other/child", "root/child")

    def test_operation_identifier_cannot_escape_the_ledger(self):
        binding = dict(self.binding, operation_id="../outside")
        with self.assertRaisesRegex(Unsupported, "invalid_operation_id"):
            LocalHelperDriver(self.root / "ledger", binding)

    def test_result_projection_rejects_interruption_and_truncation(self):
        for native in [
            {"stdout": "42", "interrupted": True},
            {"stdout": "Output truncated", "interrupted": False},
            {"stdout": "42", "interrupted": False, "backgroundTaskId": "job"},
        ]:
            with self.subTest(native=native), self.assertRaises(Unsupported):
                project(self.binding, native)

    def test_json_result_rejects_duplicate_keys_and_nonfinite_numbers(self):
        binding = dict(self.binding, role="warm_start_resolver")
        for stdout in ['{"candidates":[],"candidates":[1]}', '{"score":Infinity}', '{"score":NaN}']:
            with self.subTest(stdout=stdout), self.assertRaises(ValueError):
                project(binding, {"stdout": stdout, "interrupted": False})
        self.assertFalse(json_values_equal({"score": float("inf")}, {"score": float("inf")}))

    def test_only_one_exact_registered_preamble_can_precede_native_output(self):
        binding = dict(self.binding, native_preamble="Synthetic registered preamble.\n")
        native = {"stdout": binding["native_preamble"] + "42\n", "interrupted": False}
        self.assertEqual(project(binding, native), {"epoch": 42})
        native["stdout"] = binding["native_preamble"] + native["stdout"]
        with self.assertRaisesRegex(Unsupported, "repeated_native_preamble"):
            project(binding, native)

    def test_extra_hook_feedback_or_changed_native_continuation_latches_unknown(self):
        initial = deepcopy(self.body)
        mutations = [
            lambda body: body["messages"][-1]["content"].append({"type": "text", "text": "PostToolUse hook blocked this result."}),
            lambda body: body["messages"].append({"role": "user", "content": [{"type": "text", "text": "Extra managed context."}]}),
            lambda body: body["messages"][-1]["content"][0].update(content="changed output"),
            lambda body: body["messages"][-1]["content"][0].update(is_error=True),
            lambda body: body["messages"][-1]["content"][0].update(is_error=0),
            lambda body: body["messages"][-2]["content"].append({"type": "text", "text": "Unexpected assistant text."}),
            lambda body: body["messages"][0]["content"][0].update(text="Changed initial task."),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                self.driver = LocalHelperDriver(self.root / ("case_" + str(index)), self.binding)
                self.driver.attest_dispatch("root/child", "root/child")
                self.body = deepcopy(initial)
                self.complete_bash()
                mutate(self.body)
                with self.assertRaises(Unsupported) as caught:
                    self.driver.response(self.body, self.binding)
                state = json.loads(self.driver.path.read_bytes())
                self.assertEqual(state["stage"], "UNKNOWN")
                self.assertEqual(state["bash_emissions"], 1)
                self.assertFalse(self.driver.fallback_decision(b"unchanged", caught.exception)["caller_must_forward_unchanged"])

    def test_native_cache_marker_can_move_without_changing_the_message_prefix(self):
        self.body["messages"][0]["content"][0]["cache_control"] = {"type": "ephemeral"}
        self.complete_bash()
        self.body["messages"][0]["content"][0].pop("cache_control")
        self.body["messages"][-1]["content"][0]["cache_control"] = {"type": "ephemeral", "ttl": "5m"}
        self.assertEqual(self.driver.response(self.body, self.binding)["input"], {"epoch": 42})

    def test_hook_context_after_structured_acknowledgement_cannot_replay_success(self):
        self.complete_bash()
        structured = self.driver.response(self.body, self.binding)
        self.driver.accept(structured["id"], structured["input"])
        self.finish_native(structured["input"])
        self.body["messages"].append({"role": "user", "content": [{"type": "text", "text": "A managed hook blocked completion."}]})
        with self.assertRaisesRegex(Unsupported, "native_replay_continuation_changed"):
            self.driver.response(self.body, self.binding)
        self.assertEqual(json.loads(self.driver.path.read_bytes())["stage"], "UNKNOWN")

    def test_additional_initial_policy_text_preserves_provider_path_before_emission(self):
        self.body["messages"][0]["content"].append({"type": "text", "text": "A managed hook adds a restriction."})
        with self.assertRaisesRegex(Unsupported, "unqualified_initial_context") as caught:
            self.driver.response(self.body, self.binding)
        self.assertFalse(caught.exception.may_have_executed)
        original = json.dumps(self.body).encode()
        fallback = self.driver.fallback_decision(original, caught.exception)
        self.assertTrue(fallback["caller_must_forward_unchanged"])
        self.assertIs(fallback["original_request_bytes"], original)
        self.assertEqual(json.loads(self.driver.path.read_bytes())["bash_emissions"], 0)

    def test_qualified_native_notice_can_change_only_its_serialization(self):
        notice = "A synthetic native notice with fixed content."
        registration = frozenset({hashlib.sha256(notice.encode()).hexdigest()})
        with patch.object(native_envelope, "QUALIFIED_SKILL_NOTICES", registration):
            self.body["messages"].append({"role": "system", "content": [{"type": "text", "text": notice}]})
            self.complete_bash()
            self.body["messages"][1]["content"] = notice
            self.assertEqual(self.driver.response(self.body, self.binding)["input"], {"epoch": 42})
            self.body["messages"][1]["content"] += " Added policy."
            with self.assertRaises(Unsupported):
                self.driver.response(self.body, self.binding)
            self.assertEqual(json.loads(self.driver.path.read_bytes())["stage"], "UNKNOWN")

    def test_added_initial_system_policy_cannot_emit_bash(self):
        self.body["system"] = [{"type": "text", "text": "Do not run a Bash tool."}]
        with self.assertRaisesRegex(Unsupported, "unqualified_system_context") as caught:
            self.driver.response(self.body, self.binding)
        self.assertFalse(caught.exception.may_have_executed)
        self.assertTrue(self.driver.fallback_decision(b"original", caught.exception)["caller_must_forward_unchanged"])

    def test_changed_system_policy_after_bash_cannot_emit_structured_output(self):
        self.complete_bash()
        self.body["system"] = [{"type": "text", "text": "A managed policy now blocks this result."}]
        with self.assertRaises(Unsupported):
            self.driver.response(self.body, self.binding)
        self.assertEqual(json.loads(self.driver.path.read_bytes())["stage"], "UNKNOWN")

    def test_missing_or_empty_system_cannot_emit_bash(self):
        for missing in (True, False):
            body = deepcopy(self.body)
            if missing:
                body.pop("system")
            else:
                body["system"] = []
            with self.subTest(missing=missing), self.assertRaisesRegex(Unsupported, "unqualified_system_context") as caught:
                self.driver.response(body, self.binding)
            self.assertFalse(caught.exception.may_have_executed)
        self.assertFalse(self.driver.path.exists())


if __name__ == "__main__":
    unittest.main()
