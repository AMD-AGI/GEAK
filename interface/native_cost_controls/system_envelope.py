# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qualify native system instructions before bypassing helper inference."""

import hashlib
import platform
import re
import sys

SDK_IDENTITY_SHA256 = "0d7062851dd7bd7e66d4be4f12ac4951e3d2f587ec408295333a49963bd3f6b7"
WORKFLOW_SYSTEM_SHA256 = "0e2afada6b27a21f0521958464fc5ee160de7c6ca45d2d2b5f6cbf4949c1b205"
_BILLING = re.compile(
    r"x-anthropic-billing-header: cc_version=2\.1\.221\.[0-9a-f]{3}; cc_entrypoint=sdk-py; cc_is_subagent=true;\Z"
)


def normalized_system(system, *, workspace, shell):
    """Return a bound representation, or None for an unqualified system.

    The template hash covers the qualified Opus 4.8 instructions and cutoff.
    Environment fields must match the local runtime. Git status is unsupported.
    The actual environment text remains in the returned representation.
    Only the billing fingerprint and five-minute cache markers are omitted.
    """
    if not isinstance(system, list) or len(system) != 3:
        return None
    texts = []
    for block in system:
        if (not isinstance(block, dict) or not {"type", "text"} <= set(block)
                or set(block) - {"type", "text", "cache_control"} or block.get("type") != "text"
                or not isinstance(block.get("text"), str)):
            return None
        if "cache_control" in block and block["cache_control"] not in (
                {"type": "ephemeral"}, {"type": "ephemeral", "ttl": "5m"}):
            return None
        texts.append(block["text"])
    if not _BILLING.fullmatch(texts[0]) or hashlib.sha256(texts[1].encode()).hexdigest() != SDK_IDENTITY_SHA256:
        return None
    if (not isinstance(workspace, str) or any(char in workspace for char in "\r\n")
            or not isinstance(shell, str) or not re.fullmatch(r"[A-Za-z0-9/_+.-]{1,256}", shell)):
        return None
    shell = shell.rsplit("/", 1)[-1]
    template = texts[2]
    metadata = [("Working directory", workspace, "WORKSPACE"), ("Platform", sys.platform, "PLATFORM"),
        ("Shell", shell, "SHELL"), ("OS Version", platform.system() + " " + platform.release(), "OS_VERSION")]
    for label, value, token in metadata:
        line = label + ": " + value
        if len(re.findall(r"^" + re.escape(line) + r"$", template, re.MULTILINE)) != 1:
            return None
        template = template.replace(line, label + ": {" + token + "}")
    git_lines = re.findall(r"^Is directory a git repo: (Yes|No)$", template, re.MULTILINE)
    if git_lines != ["No"]:
        return None
    template = template.replace("Is directory a git repo: " + git_lines[0], "Is directory a git repo: {GIT}")
    if hashlib.sha256(template.encode()).hexdigest() != WORKFLOW_SYSTEM_SHA256:
        return None
    return ["qualified-native-billing-header", texts[1], texts[2]]
