# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recognize the qualified native envelope around an exact helper task.

The notice hash comes from a synthetic session under SDK 0.2.128. Unknown
notices and added policy text retain the provider path before local emission.
The module contains no captured request body or private fixture text.
"""

import datetime
import hashlib
import re

QUALIFIED_SKILL_NOTICES = frozenset({
    "daea6ca8b72e69932aa8a5d38fd1ac9a8b2087e969eac4410175f069a54c3607",
})
_DATE_PREFIX = (
    "<system-reminder>\nAs you answer the user's questions, you can use the following context:\n"
    "# currentDate\nToday's date is "
)
_DATE_SUFFIX = (
    ".\n\n      IMPORTANT: this context may or may not be relevant to your tasks. "
    "You should not respond to this context unless it is highly relevant to your task.\n"
    "</system-reminder>\n\n"
)


def qualified_notice_text(text):
    return isinstance(text, str) and hashlib.sha256(text.encode()).hexdigest() in QUALIFIED_SKILL_NOTICES


def _date_reminder(block):
    if not isinstance(block, dict) or set(block) != {"type", "text"} or block.get("type") != "text":
        return False
    text = block.get("text")
    if not isinstance(text, str):
        return False
    match = re.fullmatch(re.escape(_DATE_PREFIX) + r"(\d{4}-\d{2}-\d{2})" + re.escape(_DATE_SUFFIX), text)
    if match is None:
        return False
    try:
        datetime.date.fromisoformat(match[1])
    except ValueError:
        return False
    return True


def qualified_initial_messages(messages, prompt):
    """Accept the exact task with only qualified native context blocks."""
    if not isinstance(messages, list) or len(messages) not in (1, 2):
        return False
    first = messages[0]
    if not isinstance(first, dict) or set(first) != {"role", "content"} or first.get("role") != "user":
        return False
    content = first.get("content")
    if not isinstance(content, list) or len(content) not in (1, 2):
        return False
    if content[-1] != {"type": "text", "text": prompt}:
        return False
    if len(content) == 2 and not _date_reminder(content[0]):
        return False
    if len(messages) == 1:
        return True
    notice = messages[1]
    if not isinstance(notice, dict) or set(notice) != {"role", "content"} or notice.get("role") != "system":
        return False
    blocks = notice.get("content")
    if not isinstance(blocks, list) or len(blocks) != 1:
        return False
    block = blocks[0]
    return (isinstance(block, dict) and set(block) == {"type", "text"}
        and block.get("type") == "text" and isinstance(block.get("text"), str)
        and qualified_notice_text(block["text"]))
