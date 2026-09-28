# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read full native child results from a trusted Workflow journal descriptor.

The caller binds the descriptor to its root tool result and session mirror.
This reader creates no file. A result also needs a matching native done event
before the caller can make it replayable.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import threading
from copy import deepcopy
from pathlib import Path

MAX_JOURNAL_BYTES = 64 * 1024 * 1024
_IDENTITY = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_KEY = re.compile(r"v2:[0-9a-f]{64}\Z")


class JournalError(ValueError):
    """Report a fixed code without returning native journal content."""


def _require(condition, code):
    if not condition:
        raise JournalError(code)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _identity(info):
    return info.st_dev, info.st_ino, info.st_uid


def _object(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            _require(key not in value, "duplicate_journal_json_key")
            value[key] = item
        return value

    value = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs)
    _require(isinstance(value, dict), "journal_record_not_object")
    # Reject NaN, Infinity, and finite spellings that overflow float parsing.
    json.dumps(value, allow_nan=False)
    return value


def _result_json(raw):
    """Extract one top-level value span after the complete object validates."""
    text = raw.decode('utf-8')
    decoder = json.JSONDecoder()

    def space(offset):
        while offset < len(text) and text[offset] in ' \t\r\n':
            offset += 1
        return offset

    cursor = space(0) + 1
    while True:
        cursor = space(cursor)
        key, cursor = decoder.raw_decode(text, cursor)
        cursor = space(cursor)
        _require(text[cursor] == ':', 'invalid_journal_field_boundary')
        start = space(cursor + 1)
        _, cursor = decoder.raw_decode(text, start)
        if key == 'result':
            return text[start:cursor]
        cursor = space(cursor)
        _require(text[cursor] == ',', 'journal_result_field_missing')
        cursor += 1


class NativeJournal:
    """Read an append-only journal inside one already bound native run."""

    def __init__(self, transcript_dir, *, session_id, run_id):
        _require(isinstance(session_id, str) and _IDENTITY.fullmatch(session_id), "invalid_journal_session")
        _require(isinstance(run_id, str) and _IDENTITY.fullmatch(run_id)
                 and run_id.startswith("wf_"), "invalid_journal_run")
        raw_path = os.fspath(transcript_dir)
        _require(isinstance(raw_path, str) and ".." not in raw_path.split("/"), "invalid_journal_directory")
        directory = Path(raw_path)
        _require(directory.is_absolute()
                 and directory.parts[-4:] == (session_id, "subagents", "workflows", run_id),
                 "journal_directory_identity_mismatch")
        _require(hasattr(os, "O_NOFOLLOW") and hasattr(os, "O_DIRECTORY"), "unsupported_journal_platform")
        self.directory = directory
        self.session_id = session_id
        self.run_id = run_id
        self._directory_identity = None
        self._file_identity = None
        self._previous_size = 0
        self._previous_hash = _sha(b"")
        self._failure = None
        self._lock = threading.RLock()

    def _open_directory(self):
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        descriptor = os.open("/", flags)
        try:
            for part in self.directory.parts[1:]:
                child = os.open(part, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
            info = os.fstat(descriptor)
            _require(info.st_uid == os.geteuid(), "journal_directory_owner_mismatch")
            return descriptor, _identity(info)
        except BaseException:
            os.close(descriptor)
            raise

    def _read(self):
        directory_fd, directory_identity = self._open_directory()
        try:
            _require(self._directory_identity in (None, directory_identity), "journal_directory_replaced")
            self._directory_identity = directory_identity
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
            descriptor = os.open("journal.jsonl", flags, dir_fd=directory_fd)
            try:
                before = os.fstat(descriptor)
                _require(stat.S_ISREG(before.st_mode), "journal_not_regular_file")
                _require(before.st_uid == os.geteuid(), "journal_file_owner_mismatch")
                _require(self._file_identity in (None, _identity(before)), "journal_file_replaced")
                _require(before.st_size <= MAX_JOURNAL_BYTES, "journal_size_limit")
                _require(before.st_size >= self._previous_size, "journal_truncated")
                chunks, remaining = [], before.st_size
                while remaining:
                    chunk = os.read(descriptor, min(remaining, 1024 * 1024))
                    _require(bool(chunk), "journal_truncated_during_read")
                    chunks.append(chunk)
                    remaining -= len(chunk)
                raw = b"".join(chunks)
                after = os.fstat(descriptor)
                current = os.stat("journal.jsonl", dir_fd=directory_fd, follow_symlinks=False)
                _require(_identity(current) == _identity(before) and stat.S_ISREG(current.st_mode),
                         "journal_file_replaced_during_read")
                _require(after.st_size >= before.st_size, "journal_truncated_during_read")
                _require(_sha(raw[:self._previous_size]) == self._previous_hash, "journal_prefix_changed")
                canonical_fd, canonical_identity = self._open_directory()
                try:
                    _require(canonical_identity == directory_identity, "journal_directory_replaced_during_read")
                finally:
                    os.close(canonical_fd)
                self._file_identity = _identity(before)
                self._previous_size, self._previous_hash = len(raw), _sha(raw)
                # A concurrent append is valid. Retry the complete snapshot on
                # the next native event. Its observed prefix must remain intact.
                if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                        after.st_size, after.st_mtime_ns, after.st_ctime_ns) or (
                        after.st_size, after.st_mtime_ns, after.st_ctime_ns) != (
                        current.st_size, current.st_mtime_ns, current.st_ctime_ns):
                    return None
                return raw
            finally:
                os.close(descriptor)
        finally:
            os.close(directory_fd)

    def snapshot(self):
        """Return the current read status and independently joined agent results."""
        with self._lock:
            if self._failure:
                return {"status": "error", "reason": self._failure, "results": {}}
            try:
                raw = self._read()
                if raw is None:
                    return {"status": "pending", "reason": "journal_changed_during_read", "results": {}}
                if not raw or not raw.endswith(b"\n"):
                    return {"status": "pending", "reason": "journal_record_incomplete", "results": {}}
                starts, results = {}, {}
                for number, line in enumerate(raw.splitlines(), 1):
                    record = _object(line)
                    agent, key, kind = record.get("agentId"), record.get("key"), record.get("type")
                    _require(isinstance(agent, str) and _IDENTITY.fullmatch(agent), "invalid_journal_agent")
                    _require(isinstance(key, str) and _KEY.fullmatch(key), "invalid_journal_key")
                    _require(kind in ("started", "result", "error"), "unsupported_journal_record")
                    if kind == "started":
                        _require(set(record) == {"type", "key", "agentId"}, "unsupported_journal_start_fields")
                        _require(agent not in starts, "duplicate_journal_start")
                        starts[agent] = {"key": key, "entry_sha256": _sha(line), "line": number}
                        continue
                    _require(agent in starts and starts[agent]["key"] == key, "journal_result_without_matching_start")
                    _require(agent not in results, "duplicate_journal_terminal_result")
                    if kind == "error":
                        results[agent] = {"status": "error", "reason": "native_child_journal_error", "key": key}
                        continue
                    _require(set(record) == {"type", "key", "agentId", "result"}, "unsupported_journal_result_fields")
                    results[agent] = {
                        "status": "complete", "value": record["result"], "key": key,
                        "result_json": _result_json(line),
                        "entry_sha256": _sha(line), "started_entry_sha256": starts[agent]["entry_sha256"],
                        "line": number,
                    }
                return {"status": "complete", "results": deepcopy(results), "started_agents": sorted(starts),
                        "journal_sha256": _sha(raw), "journal_bytes": len(raw),
                        "directory_identity": self._directory_identity, "file_identity": self._file_identity}
            except FileNotFoundError:
                if self._file_identity is None:
                    return {"status": "pending", "reason": "journal_not_created", "results": {}}
                self._failure = "journal_disappeared"
            except JournalError as error:
                self._failure = str(error)
            except (OSError, ValueError, TypeError, RecursionError):
                self._failure = "invalid_or_unreadable_journal"
            return {"status": "error", "reason": self._failure, "results": {}}

    def result_for(self, agent_id):
        """Return complete only for a unique full result for this native agent."""
        with self._lock:
            _require(isinstance(agent_id, str) and _IDENTITY.fullmatch(agent_id), "invalid_journal_agent")
            snapshot = self.snapshot()
            if snapshot["status"] != "complete":
                return {"status": snapshot["status"], "reason": snapshot["reason"], "value": None}
            result = snapshot["results"].get(agent_id)
            if result is None:
                return {"status": "pending", "reason": "native_child_result_missing", "value": None}
            return {**result, "journal_sha256": snapshot["journal_sha256"], "journal_bytes": snapshot["journal_bytes"]}

    def bound_bytes(self, snapshot):
        """Return exact bytes only while they match the caller's complete snapshot."""
        with self._lock:
            raw = self._read()
            _require(raw is not None and snapshot.get("status") == "complete"
                     and len(raw) == snapshot.get("journal_bytes")
                     and _sha(raw) == snapshot.get("journal_sha256"), "journal_closure_bytes_changed")
            return raw
