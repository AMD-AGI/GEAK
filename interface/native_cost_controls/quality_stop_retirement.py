# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read protected evidence for host-enforced native attempt retirement.

An interruption control frame does not prove OS-process termination. The caller
must also close old transport work and install permanent admission fences.
"""

from __future__ import annotations

import datetime
import hashlib
import os
import re
import stat
from copy import deepcopy

from .helper_driver import json_values_equal
from .native_journal import NativeJournal, MAX_JOURNAL_BYTES, _object as journal_object
from .quality_stop_controller import canonical, require

MODE = "host_enforced_not_observed_os_termination"
UUID = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\Z")
TERMINAL_FIELDS = {"parentUuid", "isSidechain", "promptId", "agentId", "type", "message", "uuid", "timestamp",
                   "userType", "entrypoint", "cwd", "sessionId", "version", "gitBranch"}
INTERRUPTION = {"role": "user", "content": [{"type": "text", "text": "[Request interrupted by user]"}]}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def initial_record(mirror, initial, agent, *, descriptor, session_id, workspace):
    """Require exact native identity fields for a replacement candidate."""
    key, entry = initial.get("descriptor"), initial.get("initial_entry")
    require(isinstance(key, dict) and key.get("session_id") == session_id
            and key.get("project_key") == descriptor["project_key"]
            and key.get("subpath") == "subagents/workflows/" + descriptor["run_id"] + "/agent-" + agent,
            "retirement_initial_descriptor_changed")
    require(isinstance(entry, dict) and entry.get("type") == "user" and entry.get("parentUuid") is None
            and entry.get("agentId") == agent and entry.get("sessionId") == session_id
            and entry.get("cwd") == str(workspace) and isinstance(entry.get("message"), dict)
            and entry["message"].get("role") == "user" and entry["message"].get("content") == mirror["task"]
            and isinstance(entry.get("promptId"), str) and UUID.fullmatch(entry["promptId"]),
            "retirement_initial_entry_changed")
    return deepcopy(initial)


def interruption_binding(raw, initial, agent, *, session_id, workspace):
    """Return a proof only for the final exact interruption control frame."""
    if not raw or not raw.endswith(b"\n"):
        return None
    rows = [journal_object(line) for line in raw.splitlines()]
    require(rows and json_values_equal(rows[0], initial["initial_entry"]), "retirement_initial_file_changed")
    controls = [index for index, row in enumerate(rows)
                if row.get("type") == "user" and json_values_equal(row.get("message"), INTERRUPTION)]
    if not controls:
        return None
    require(controls == [len(rows) - 1] and len(rows) >= 2, "retirement_interruption_entry_not_unique_or_final")
    terminal = rows[-1]
    require(set(terminal) == TERMINAL_FIELDS and terminal["isSidechain"] is True
            and terminal["agentId"] == agent and terminal["sessionId"] == session_id
            and terminal["cwd"] == str(workspace)
            and terminal["promptId"] == initial["initial_entry"]["promptId"]
            and terminal["userType"] == "external" and terminal["entrypoint"] == "sdk-py"
            and terminal["version"] == "2.1.221" and isinstance(terminal["gitBranch"], str)
            and isinstance(terminal["uuid"], str) and UUID.fullmatch(terminal["uuid"])
            and isinstance(terminal["parentUuid"], str) and UUID.fullmatch(terminal["parentUuid"])
            and terminal["parentUuid"] == rows[-2].get("uuid")
            and terminal["uuid"] not in {row.get("uuid") for row in rows[:-1]},
            "retirement_interruption_entry_changed")
    require(isinstance(terminal["timestamp"], str), "retirement_interruption_time_invalid")
    try:
        stamp = datetime.datetime.fromisoformat(terminal["timestamp"].replace("Z", "+00:00"))
    except ValueError:
        require(False, "retirement_interruption_time_invalid")
    require(stamp.tzinfo is not None and stamp.utcoffset() == datetime.timedelta(0),
            "retirement_interruption_time_invalid")
    return {"raw_utf8": raw.decode("utf-8"), "sha256": sha(raw), "bytes": len(raw),
            "terminal_entry": deepcopy(terminal)}


class ProtectedTranscript(NativeJournal):
    """Bind stable append-only bytes through a protected directory descriptor."""

    def __init__(self, directory, *, session_id, run_id, agent_id):
        super().__init__(directory, session_id=session_id, run_id=run_id)
        require(isinstance(agent_id, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", agent_id),
                "retirement_agent_identity_invalid")
        self.filename = "agent-" + agent_id + ".jsonl"

    def read(self):
        """Return stable bytes, or wait for a complete concurrent append."""
        directory_fd, directory_identity = self._open_directory()
        try:
            require(self._directory_identity in (None, directory_identity), "retirement_directory_replaced")
            self._directory_identity = directory_identity
            try:
                descriptor = os.open(self.filename, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
            except FileNotFoundError:
                require(self._file_identity is None, "retirement_transcript_disappeared")
                return None
            try:
                before = os.fstat(descriptor)
                identity = (before.st_dev, before.st_ino, before.st_uid)
                require(stat.S_ISREG(before.st_mode) and before.st_uid == os.geteuid() and before.st_nlink == 1,
                        "retirement_transcript_identity_invalid")
                require(self._file_identity in (None, identity), "retirement_transcript_replaced")
                require(self._previous_size <= before.st_size <= MAX_JOURNAL_BYTES, "retirement_transcript_size_invalid")
                chunks, remaining = [], before.st_size
                while remaining:
                    chunk = os.read(descriptor, min(remaining, 1024 * 1024))
                    require(bool(chunk), "retirement_transcript_truncated")
                    chunks.append(chunk)
                    remaining -= len(chunk)
                raw = b"".join(chunks)
                after = os.fstat(descriptor)
                current = os.stat(self.filename, dir_fd=directory_fd, follow_symlinks=False)
                fields = ("st_dev", "st_ino", "st_uid", "st_nlink")
                require(all(getattr(before, field) == getattr(after, field) == getattr(current, field) for field in fields)
                        and stat.S_ISREG(current.st_mode), "retirement_transcript_replaced_during_read")
                require(sha(raw[:self._previous_size]) == self._previous_hash, "retirement_transcript_prefix_changed")
                canonical_fd, current_directory = self._open_directory()
                os.close(canonical_fd)
                require(current_directory == directory_identity, "retirement_directory_replaced_during_read")
                self._file_identity = identity
                self._previous_size, self._previous_hash = len(raw), sha(raw)
                if any(getattr(before, field) != getattr(after, field) or getattr(after, field) != getattr(current, field)
                       for field in ("st_size", "st_mtime_ns", "st_ctime_ns")):
                    return None
                return raw
            finally:
                os.close(descriptor)
        finally:
            os.close(directory_fd)


def journal_starts(raw):
    """Read start keys without discarding old attempts or terminal rows."""
    return {row["agentId"]: row["key"] for row in (journal_object(line) for line in raw.splitlines())
            if row["type"] == "started"}


def canonical_bytes(rows):
    return b"".join((canonical(row) + "\n").encode("ascii") for row in rows)
