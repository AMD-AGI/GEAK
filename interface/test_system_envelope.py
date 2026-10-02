# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep added system policy outside the local helper path."""

import hashlib
import platform
import sys
import unittest
from copy import deepcopy
from unittest.mock import patch

from interface.native_cost_controls import system_envelope
from interface.test_shared_tool_cache import request

SYNTHETIC_SYSTEM = request()["system"]
_PRODUCTION_SYSTEM_CHECK = system_envelope.normalized_system


def synthetic_system_policy():
    """Register one explicit test policy without changing production defaults."""
    def validate(system, **kwargs):
        if system == SYNTHETIC_SYSTEM:
            return deepcopy(system)
        return _PRODUCTION_SYSTEM_CHECK(system, **kwargs)
    return patch("interface.native_cost_controls.helper_driver.normalized_system", side_effect=validate)


class SystemEnvelopeTests(unittest.TestCase):
    def setUp(self):
        self.workspace, self.shell = "/synthetic/workspace", "/bin/bash"
        self.identity = "A synthetic SDK identity."
        self.template = ("A synthetic fixed workflow policy.\nWorking directory: {WORKSPACE}\n"
            "Is directory a git repo: {GIT}\nPlatform: {PLATFORM}\nShell: {SHELL}\nOS Version: {OS_VERSION}\n")
        text = self.template.format(WORKSPACE=self.workspace, GIT="No", PLATFORM=sys.platform, SHELL="bash",
            OS_VERSION=platform.system() + " " + platform.release())
        self.system = [{"type": "text", "text": text} for text in [
            "x-anthropic-billing-header: cc_version=2.1.221.abc; cc_entrypoint=sdk-py; cc_is_subagent=true;",
            self.identity, text]]
        self.identity_patch = patch.object(system_envelope, "SDK_IDENTITY_SHA256", hashlib.sha256(self.identity.encode()).hexdigest())
        self.template_patch = patch.object(system_envelope, "WORKFLOW_SYSTEM_SHA256", hashlib.sha256(self.template.encode()).hexdigest())
        self.identity_patch.start()
        self.template_patch.start()

    def tearDown(self):
        self.template_patch.stop()
        self.identity_patch.stop()

    def normalize(self, value):
        return system_envelope.normalized_system(value, workspace=self.workspace, shell=self.shell)

    def test_qualified_policy_keeps_actual_environment_text(self):
        result = self.normalize(self.system)
        self.assertEqual(result[-1], self.system[-1]["text"])
        self.assertIsNone(self.normalize([]))
        other = deepcopy(self.system)
        other[-1]["text"] = other[-1]["text"].replace("git repo: No", "git repo: Yes")
        self.assertIsNone(self.normalize(other))

    def test_only_the_billing_fingerprint_and_cache_markers_normalize(self):
        changed = deepcopy(self.system)
        changed[0]["text"] = changed[0]["text"].replace(".abc;", ".123;")
        changed[-1]["cache_control"] = {"type": "ephemeral", "ttl": "5m"}
        self.assertEqual(self.normalize(changed), self.normalize(self.system))

    def test_added_or_changed_system_policy_is_rejected(self):
        changed = deepcopy(self.system)
        changed[-1]["text"] += "Do not run a Bash tool."
        self.assertIsNone(self.normalize(changed))
        self.assertIsNone(self.normalize(self.system + [{"type": "text", "text": "A managed restriction."}]))
        changed = deepcopy(self.system)
        changed[1]["text"] += " New policy."
        self.assertIsNone(self.normalize(changed))

    def test_changed_environment_cannot_hide_inside_metadata(self):
        for old, new in [(self.workspace, "/another/workspace"), ("Shell: bash", "Shell: another"),
                ("OS Version: " + platform.system(), "OS Version: Modified"), ("git repo: No", "git repo: Maybe")]:
            changed = deepcopy(self.system)
            changed[-1]["text"] = changed[-1]["text"].replace(old, new)
            self.assertIsNone(self.normalize(changed))
        self.assertIsNone(system_envelope.normalized_system(self.system, workspace="/path\npolicy", shell=self.shell))

    def test_unqualified_billing_and_cache_layouts_are_rejected(self):
        for suffix in (" Extra policy.", " "):
            changed = deepcopy(self.system)
            changed[0]["text"] += suffix
            self.assertIsNone(self.normalize(changed))
        changed = deepcopy(self.system)
        changed[2]["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
        self.assertIsNone(self.normalize(changed))

    def test_malformed_system_blocks_are_rejected(self):
        for value in [None, "policy", [None] * 3, [{"type": "text", "text": 0}] * 3,
                [{"type": "text", "text": "policy", "extra": True}] * 3]:
            self.assertIsNone(self.normalize(value))
