# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qualify only the observed native skill-notice cache decoration."""

import hashlib
import tempfile
import unittest
from copy import deepcopy
from unittest.mock import patch

from interface.native_cost_controls import native_envelope
from interface.native_cost_controls.quality_stop_controller import StopRejected
from interface.native_cost_controls.quality_stop_native import (
    checkpoint_initial_messages,
)
from interface.test_quality_stop_native import NativeFixture


class CheckpointEnvelopeTests(unittest.TestCase):
    def setUp(self):
        self.task = "GEAK_QUALITY_STOP_V2\n{}"
        self.notice = "Registered synthetic native skill notice."
        self.messages = [{"role": "user", "content": [{"type": "text", "text": self.task}]},
                         {"role": "system", "content": [{"type": "text", "text": self.notice,
                                                           "cache_control": {"type": "ephemeral"}}]}]
        self.registration = patch.object(native_envelope, "QUALIFIED_SKILL_NOTICES",
                                         frozenset({hashlib.sha256(self.notice.encode()).hexdigest()}))
        self.registration.start()
        self.addCleanup(self.registration.stop)

    def test_exact_decoration_qualifies_without_changing_request(self):
        original = deepcopy(self.messages)
        self.assertTrue(checkpoint_initial_messages(self.messages, self.task))
        self.assertEqual(self.messages, original)

    def test_exact_task_decoration_qualifies_with_or_without_skill_notice(self):
        for messages in (self.messages, self.messages[:1]):
            changed = deepcopy(messages)
            changed[0]["content"][-1]["cache_control"] = {"type": "ephemeral"}
            original = deepcopy(changed)
            self.assertTrue(checkpoint_initial_messages(changed, self.task))
            self.assertEqual(changed, original)

    def test_unknown_notice_fields_and_markers_do_not_qualify(self):
        cases = []
        for marker in ({"type": "ephemeral", "ttl": "5m"}, {"type": "ephemeral", "ttl": "1h"},
                       {"type": "permanent"}, None, 1):
            changed = deepcopy(self.messages)
            changed[1]["content"][0]["cache_control"] = marker
            cases.append(changed)
        for change in ({"text": self.notice + " changed"}, {"extra": True}, {"type": "image"}):
            changed = deepcopy(self.messages)
            changed[1]["content"][0].update(change)
            cases.append(changed)
        for messages in cases:
            with self.subTest(messages=messages):
                self.assertFalse(checkpoint_initial_messages(messages, self.task))

    def test_no_general_message_shape_or_task_cache_exception(self):
        values = [None, [], [{}, None], [{}, {"role": "assistant", "content": []}],
                  [{}, {"role": "system", "content": "text"}], [{}, {"role": "system", "content": [None]}]]
        changed = deepcopy(self.messages)
        changed[0]["content"][0]["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
        values.append(changed)
        for messages in values:
            with self.subTest(messages=messages):
                self.assertFalse(checkpoint_initial_messages(messages, self.task))

    def test_rejection_ledger_records_only_fixed_codes(self):
        for error, expected in ((StopRejected("fixed_failure"), "fixed_failure"),
                                (StopRejected("untrusted arbitrary details"), "native_checkpoint_rejected"),
                                (ValueError("untrusted arbitrary details"), "invalid_native_checkpoint_request")):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as directory:
                fixture = NativeFixture(directory)
                with patch.object(fixture.controller, "checkpoint", side_effect=error), self.assertRaises(type(error)):
                    fixture.response()
                self.assertEqual(fixture.registry.rejections[-1]["reason"], expected)
                self.assertNotIn("untrusted", str(fixture.registry.rejections))
                self.assertFalse(fixture.controller.calls)
                fixture.registry.close()
