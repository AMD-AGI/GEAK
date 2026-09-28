# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bind stopping calls to the complete native producer census.

This registry is independent of NativeHelperRegistry. Helper and cache controls
remain disabled. Native journal parsing, native terminal state, tool activity,
background task state, and the host process boundary must all agree.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import stat
import threading
import time
import uuid
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from pathlib import Path
from urllib.parse import urlsplit

from .helper_driver import json_values_equal
from .native_envelope import qualified_initial_messages, qualified_notice_text
from .native_journal import JournalError, NativeJournal
from .quality_stop_retirement import (
    MODE as RETIREMENT_MODE, ProtectedTranscript, canonical_bytes,
    initial_record, interruption_binding, journal_starts,
)
from .quality_stop_controller import (
    PREFIX,
    SCHEMA,
    StopRejected,
    canonical,
    parse_task,
    require,
)

ROOT_ULTRACODE_NOTICE_SHA256 = "a5f29abc2627e9d881289565aa239bed2d9feba821998d863a5c0eff788e0626"
MAX_NATIVE_ATTEMPTS = 6
BASH_TASK_ID = re.compile(r"b[0-9a-z]{8}")
BASH_TERMINAL = {"completed": "completed", "failed": "failed", "stopped": "stopped", "killed": "stopped"}
BASH_RESULT_FIELDS = frozenset({"stdout", "stderr", "interrupted", "isImage", "returnCodeInterpretation",
    "noOutputExpected", "backgroundTaskId", "timedOutAfterMs", "backgroundCwdHint", "dangerouslyDisableSandbox",
    "persistedOutputPath", "persistedOutputSize", "staleReadFileStateHint", "ghRateLimitHint"})


def native_attempt_label(logical_label, attempt):
    require(isinstance(logical_label, str) and logical_label and type(attempt) is int
            and 1 <= attempt <= MAX_NATIVE_ATTEMPTS, "native_attempt_label_identity_invalid")
    return logical_label if attempt == 1 else logical_label + " (retry " + str(attempt - 1) + ")"


def _root_notice_text(text):
    return (qualified_notice_text(text) or isinstance(text, str)
            and hashlib.sha256(text.encode("utf-8")).hexdigest() == ROOT_ULTRACODE_NOTICE_SHA256)


def _object(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate_native_request_key")
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=pairs)
    require(isinstance(value, dict), "native_request_not_object")
    canonical(value)
    return value


def _contains_checkpoint(value):
    if isinstance(value, str):
        return PREFIX in value
    if isinstance(value, dict):
        return any(_contains_checkpoint(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_checkpoint(item) for item in value)
    return False


def _initial_messages_view(messages, task, *, root=False):
    """Recognize the measured cache decoration on an exact task or skill notice.

    The scientific request bytes remain unchanged. The native CLI decorates its
    known skill notice with a five-minute cache marker even with our cache
    control disabled. Unknown notices, extra fields, and other markers fail.
    """
    view = deepcopy(messages)
    if isinstance(view, list) and view and isinstance(view[0], dict) and view[0].get("role") == "user":
        blocks = view[0].get("content")
        if isinstance(blocks, list) and blocks and isinstance(blocks[-1], dict):
            block = blocks[-1]
            if (set(block) == {"type", "text", "cache_control"} and block.get("type") == "text"
                    and block.get("text") == task and block["cache_control"] == {"type": "ephemeral"}):
                del block["cache_control"]
    if isinstance(view, list) and len(view) == 2:
        notice = view[1]
        if (root and isinstance(notice, dict) and notice.get("role") == "system"
                and _root_notice_text(notice.get("content"))):
            notice["content"] = [{"type": "text", "text": notice["content"]}]
        blocks = notice.get("content") if isinstance(notice, dict) and notice.get("role") == "system" else None
        if isinstance(blocks, list) and len(blocks) == 1 and isinstance(blocks[0], dict):
            block = blocks[0]
            if (set(block) == {"type", "text", "cache_control"} and block.get("type") == "text"
                    and (qualified_notice_text(block.get("text")) or root and _root_notice_text(block.get("text")))
                    and block["cache_control"] == {"type": "ephemeral"}):
                del block["cache_control"]
    return view


def checkpoint_initial_messages(messages, task):
    return qualified_initial_messages(_initial_messages_view(messages, task), task)


def _root_initial_messages(view, task):
    if checkpoint_initial_messages(view, task):
        return True
    if not (isinstance(view, list) and len(view) == 2 and checkpoint_initial_messages(view[:1], task)):
        return False
    notice = view[1]
    if not (isinstance(notice, dict) and set(notice) == {"role", "content"} and notice["role"] == "system"):
        return False
    blocks = notice["content"]
    if isinstance(blocks, str):
        return hashlib.sha256(blocks.encode("utf-8")).hexdigest() == ROOT_ULTRACODE_NOTICE_SHA256
    return (isinstance(blocks, list) and len(blocks) == 1 and isinstance(blocks[0], dict)
            and set(blocks[0]) == {"type", "text", "cache_control"} and blocks[0]["type"] == "text"
            and isinstance(blocks[0]["text"], str)
            and hashlib.sha256(blocks[0]["text"].encode("utf-8")).hexdigest() == ROOT_ULTRACODE_NOTICE_SHA256
            and blocks[0]["cache_control"] == {"type": "ephemeral"})


def _evidence_digest(value):
    """Hash native text and tool payloads without copying possible credentials."""
    try:
        return hashlib.sha256(canonical(value).encode("ascii")).hexdigest()
    except BaseException:
        return None


def _identity_evidence(value, depth=0):
    """Retain protocol fields, with hashes for unbounded native text."""
    try:
        if depth > 6:
            return {"value_type": type(value).__name__, "depth_limit": True}
        return _identity_evidence_value(value, depth)
    except BaseException as error:
        return {"value_type": type(value).__name__, "evidence_error_type": type(error).__name__}


def _identity_evidence_value(value, depth):
    if not isinstance(value, dict):
        return {"value_type": type(value).__name__}
    result = {}
    fields = ("session_id", "task_id", "tool_use_id", "task_type", "status", "agent_id",
              "hook_event_name", "tool_name", "cwd", "type", "phaseIndex", "index", "attempt",
              "label", "agentId", "state", "cached", "lastToolName", "parentUuid", "sessionId",
              "project_key", "subpath", "uuid", "promptId", "entrypoint", "userType")
    for key in fields:
        if key in value:
            item = value[key]
            if item is None or type(item) in (str, int, bool):
                result[key] = item
            else:
                result[key] = {"value_type": type(item).__name__}
    for key in ("resultPreview", "tool_input", "tool_response", "message", "content"):
        if key in value:
            result[key + "_sha256"] = _evidence_digest(value[key])
    for key in ("run_in_background", "task_id", "taskId", "backgroundTaskId", "background_task_id",
                "interrupted", "truncated"):
        for parent in ("tool_input", "tool_response"):
            child = value.get(parent)
            if isinstance(child, dict) and key in child:
                item = child[key]
                result[parent + "." + key] = item if item is None or type(item) in (str, int, bool) else {
                    "value_type": type(item).__name__}
    progress = value.get("workflow_progress")
    if isinstance(progress, list):
        result["workflow_progress"] = [_identity_evidence(node, depth + 1) for node in progress]
    elif "workflow_progress" in value:
        result["workflow_progress_type"] = type(progress).__name__
    if isinstance(value.get("data"), dict):
        result["data"] = _identity_evidence(value["data"], depth + 1)
    return result


class NativeProducerCensus:
    """Join every native producer, including queued nodes without agent IDs."""

    def __init__(self, *, session_id, root_request, source_root, workspace, controller, wait_seconds=10,
                 retirement_wait_seconds=600):
        self.session_id, self.root_request = session_id, deepcopy(root_request)
        self.workspace, self.controller = Path(workspace), controller
        self.source_root = Path(source_root).resolve()
        require(json_values_equal(root_request, {"scriptPath": str(self.source_root / "kernel_workflow.js"),
                                "args": root_request.get("args")}), "unsupported_stopping_root")
        arguments = root_request["args"]
        require(arguments.get("mode", "optimize") == "optimize"
                and arguments.get("workflow_dir") == str(self.source_root)
                and arguments.get("kernel_lane_script", str(self.source_root / "kernel_lane.js"))
                    == str(self.source_root / "kernel_lane.js")
                and json_values_equal(arguments.get("quality_stop"), controller.public_config), "stopping_root_changed")
        require(not any(arguments.get(key) for key in ("state_dir", "resume", "continue_conversation", "session_import")),
                "stopping_resume_unsupported")
        self.sources = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in
                        (self.source_root / "kernel_workflow.js", self.source_root / "kernel_lane.js",
                         self.source_root / "quality_stop_verify.js", Path(__file__).resolve(),
                         Path(__file__).with_name("quality_stop_retirement.py").resolve(),
                         Path(__file__).with_name("sdk_quality_stop.py").resolve())}
        self.root_tool = self.root_task = self.journal = self.journal_descriptor = None
        self.nodes, self.mirrors, self.tools, self.background, self.emissions = {}, {}, {}, {}, {}
        self.mirror_inputs = {}
        self.completed_tools = {}
        # Native Bash also registers foreground commands after its progress
        # hint. Retain their lifecycle so an eventual automatic timeout can
        # join the exact already-admitted tool without guessing an identity.
        self.bash_tasks = {}
        self.bash_task_tools = {}
        self.root_prompt = None
        self.root_messages = None
        self.root_tool_bindings = {}
        self.root_tool_admitted = set()
        self.root_tool_completed = set()
        self.bootstrap_admitted = False
        self.pending_retirements = {}
        self.superseded_pending = set()
        self.logical_labels = {}
        self.retired = {}
        self.retirement_events = []
        self.retirement_readers = {}
        self.last_progress = None
        self.checkpoint_active = None
        self.error = None
        self.first_failure = None
        self.first_failure_persisted = False
        self.rejections = []
        self.request_records = []
        self.transport_sealed = False
        self.closed = False
        self.closed_at = None
        self.wait_seconds = wait_seconds
        self.retirement_wait_seconds = retirement_wait_seconds
        self.condition = threading.Condition(threading.RLock())

    def bind_root_prompt(self, prompt, session_id="default"):
        with self.condition:
            require(isinstance(prompt, str) and prompt and session_id == "default"
                    and self.root_prompt is None, "native_root_query_changed")
            self.root_prompt = prompt

    def _root_model(self, fields, body):
        require(fields.get("x-claude-code-session-id") in (None, self.session_id)
                and not fields.get("x-claude-code-parent-agent-id"), "native_request_identity_changed")
        require(self.root_prompt is not None and isinstance(body.get("messages"), list), "native_root_input_unbound")
        messages = body["messages"]
        # The qualified SDK changes the known system notice from a cached
        # text-block list to a string on continuation. Normalize only that
        # measured decoration, and keep the transmitted bytes unchanged.
        count = len(messages) if self.root_messages is None else len(self.root_messages)
        require(_root_initial_messages(messages[:count], self.root_prompt), "native_root_input_changed")
        initial = _initial_messages_view(messages[:count], self.root_prompt, root=True)
        if self.root_messages is None:
            self.root_messages = deepcopy(initial)
        else:
            require(len(messages) >= len(self.root_messages)
                    and json_values_equal(initial, self.root_messages), "native_root_input_changed")

    def _bootstrap(self, method, target, headers, body, base_path):
        parts = urlsplit(target)
        require(method == "HEAD" and target == base_path + "/api/hello" and not parts.query and not parts.fragment
                and not body and not any(name.lower().startswith("x-claude-code-") for name, _ in headers)
                and self.root_tool is None and not self.bootstrap_admitted
                and not any(row.get("event") == "forwarding" for row in self.request_records),
                "native_root_endpoint_unsupported")

    def _observe_root_tools(self, value):
        require(value.get("parent_tool_use_id") is None and not value.get("agent_id")
                and value.get("session_id") == self.session_id and self.root_prompt is not None,
                "native_root_message_identity_changed")
        for block in value.get("content", []):
            if not isinstance(block, dict) or block.get("type") not in (None, "tool_use") or "name" not in block:
                continue
            name, tool_id = block.get("name"), block.get("id")
            require(name in {"Workflow", "TaskOutput", "TaskList", "TaskGet"} and isinstance(tool_id, str) and tool_id,
                    "native_root_tool_unsupported")
            binding = {"name": name, "input": deepcopy(block.get("input"))}
            require(tool_id not in self.root_tool_bindings or self.root_tool_bindings[tool_id] == binding,
                    "native_root_tool_binding_changed")
            self.root_tool_bindings[tool_id] = binding

    def _guard_retired(self, agent, event, *, detail=None):
        if agent not in self.retired:
            return
        old = self.retired[agent]
        self._record_retirement(event, agent, old["new_node"]["agentId"], detail=detail)
        self.fail("native_retired_identity_denied", trigger={"kind": event, "agent_id": agent,
                                                            "detail": _identity_evidence(detail)})
        raise StopRejected("native_retired_identity_denied")

    def _guard_superseded_tool(self, agent, detail):
        if agent not in self.superseded_pending:
            return
        successors = [pair["new"]["agentId"] for pair in self.pending_retirements.values()
                      if pair["old"]["agentId"] == agent]
        require(len(successors) == 1, "native_superseded_tool_successor_unknown")
        self._record_retirement("denied_tool", agent, successors[0], detail=detail)
        self.fail("native_superseded_tool_denied", trigger={"kind": "denied_tool", "agent_id": agent,
                                                          "event": _identity_evidence(detail)})
        raise StopRejected("native_superseded_tool_denied")

    def _retirement_path(self):
        directory = getattr(self.controller, "state_dir", None)
        require(directory is not None, "native_retirement_state_directory_missing")
        return Path(directory) / "native_retirements.jsonl"

    def _record_retirement(self, event, old, new, *, proof=None, detail=None):
        record = {"schema": "geak-native-retirement-event-v1", "sequence": len(self.retirement_events) + 1,
            "monotonic_time": time.monotonic(), "event": event, "old_agent_id": old, "new_agent_id": new}
        if proof is not None:
            record["proof"] = deepcopy(proof)
        if detail is not None:
            record["detail"] = _identity_evidence(detail)
        raw = canonical_bytes([record])
        try:
            path = self._retirement_path()
            directory = path.parent
            require(directory.is_absolute() and directory.resolve() == directory, "native_retirement_directory_changed")
            directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                require(os.fstat(directory_fd).st_uid == os.geteuid(), "native_retirement_directory_changed")
                descriptor = os.open(path.name, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
                                     0o600, dir_fd=directory_fd)
                with os.fdopen(descriptor, "ab") as stream:
                    info = os.fstat(stream.fileno())
                    require(stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid() and info.st_nlink == 1
                            and info.st_size == len(canonical_bytes(self.retirement_events)), "native_retirement_ledger_changed")
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException as error:
            self.fail("native_retirement_ledger_unavailable", error=error,
                      trigger={"kind": event, "agent_id": old})
            raise
        self.retirement_events.append(deepcopy(record))
        self.condition.notify_all()

    def _old_bridge_closed(self, agent):
        rows = [row for row in self.request_records if row.get("agent_id") == agent]
        groups = {}
        for row in rows:
            groups.setdefault(row["request_id"], []).append(row)
        for records in groups.values():
            names = [row["event"] for row in records]
            if names in (["started"], ["started", "forwarding"]):
                return None
            if agent in self.superseded_pending:
                last = records[-1]
                require(names == ["started", "failed"]
                        and last.get("reason") == "native_pending_successor_superseded"
                        and last.get("route") == "unclassified" and last.get("upstream_bridge_attempted") is False
                        and last.get("provider_attempted") is False, "retirement_superseded_bridge_invalid")
                continue
            require(names in (["started", "forwarding", "finished"], ["started", "forwarding", "failed"]),
                    "retirement_old_bridge_invalid")
        return sorted(groups)

    def _retirement_remaining(self, pair):
        remaining = pair["observed_monotonic"] + self.retirement_wait_seconds - time.monotonic()
        deadline = getattr(self.controller, "deadline_epoch", 0)
        if deadline:
            remaining = min(remaining, deadline - getattr(self.controller, "clock", time.time)())
        return remaining

    def _try_retirements(self):
        if not self.pending_retirements:
            return
        require(self.error is None and not self.closed, "native_census_unavailable")
        if self.journal is None:
            return
        snapshot = self.journal.snapshot()
        require(snapshot["status"] != "error", "retirement_journal_error")
        if snapshot["status"] != "complete":
            return
        try:
            raw_journal = self.journal.bound_bytes(snapshot)
        except JournalError as error:
            # The descriptor reader already checks unchanged earlier bytes.
            # A concurrent valid append needs another complete snapshot.
            if str(error) == "journal_closure_bytes_changed":
                return
            raise
        starts = journal_starts(raw_journal)
        for new_agent, pair in list(self.pending_retirements.items()):
            old, new = pair["old"], pair["new"]
            old_agent = old["agentId"]
            require(self._retirement_remaining(pair) > 0, "native_retirement_timeout")
            # Every ancestor fence must precede this fence. An intermediate
            # successor can remain unforwarded and then become an old attempt.
            if old_agent in self.pending_retirements:
                continue
            if not all(agent in self.mirrors and agent in self.mirror_inputs and agent in starts
                       for agent in (old_agent, new_agent)):
                continue
            require(old_agent not in snapshot["results"] and new_agent not in snapshot["results"],
                    "retirement_agent_already_has_result")
            require(starts[old_agent] == starts[new_agent], "retirement_journal_key_changed")
            old_initial = initial_record(self.mirrors[old_agent], self.mirror_inputs[old_agent], old_agent,
                descriptor=self.journal_descriptor, session_id=self.session_id, workspace=self.workspace)
            new_initial = initial_record(self.mirrors[new_agent], self.mirror_inputs[new_agent], new_agent,
                descriptor=self.journal_descriptor, session_id=self.session_id, workspace=self.workspace)
            task = self.mirrors[old_agent]["task"]
            require(task == self.mirrors[new_agent]["task"] and not task.startswith(PREFIX)
                    and old_initial["initial_entry"]["promptId"] == new_initial["initial_entry"]["promptId"],
                    "retirement_task_or_prompt_changed")
            reader = self.retirement_readers.get(old_agent)
            if reader is None:
                reader = ProtectedTranscript(self.journal_descriptor["directory"], session_id=self.session_id,
                    run_id=self.journal_descriptor["run_id"], agent_id=old_agent)
                self.retirement_readers[old_agent] = reader
            raw = reader.read()
            binding = None if raw is None else interruption_binding(raw, old_initial, old_agent,
                session_id=self.session_id, workspace=self.workspace)
            if binding is None:
                continue
            require(not any(operation["agent"] == old_agent and operation["name"] == "StructuredOutput"
                            for operation in self.completed_tools.values()), "retirement_old_structured_output_already_completed")
            requests = self._old_bridge_closed(old_agent)
            if (requests is None or any(operation["agent"] == old_agent for operation in self.tools.values())
                    or any(status == "active" for status in self.background.values())):
                continue
            self._automatic_bash_proof()
            require(self._retirement_remaining(pair) > 0, "native_retirement_timeout")
            # Hooks and forwarding use this same condition lock. The empty
            # activity arrays are a trusted host-state attestation at this point.
            proof = {"schema": "geak-native-enforced-retirement-proof-v1", "mode": RETIREMENT_MODE,
                "activity_evidence": "trusted_host_state_under_shared_admission_lock",
                "logical_label": self._logical_label(old),
                "old_node": deepcopy(old), "new_node": deepcopy(new),
                "old_initial_sha256": hashlib.sha256(canonical(old_initial).encode("ascii")).hexdigest(),
                "new_initial_sha256": hashlib.sha256(canonical(new_initial).encode("ascii")).hexdigest(),
                "task_sha256": hashlib.sha256(task.encode("utf-8")).hexdigest(),
                "prompt_id": old_initial["initial_entry"]["promptId"], "journal_key": starts[old_agent],
                "old_mirror": binding, "bridge_prefix_sha256": hashlib.sha256(canonical_bytes(self.request_records)).hexdigest(),
                "bridge_event_count": len(self.request_records), "old_bridge_request_ids": requests,
                "old_unforwarded_request_ids": requests if old_agent in self.superseded_pending else [],
                "active_old_tool_ids": [], "active_background": []}
            self._record_retirement("fence_installed", old_agent, new_agent, proof=proof)
            self.retired[old_agent] = proof
            del self.pending_retirements[new_agent]

    def _validate_retirements(self, snapshot):
        require(not self.pending_retirements, "native_retirement_pending")
        for agent, proof in self.retired.items():
            require(agent not in snapshot["results"], "retired_agent_late_journal_result")
            require(self.retirement_readers[agent].read() == proof["old_mirror"]["raw_utf8"].encode("utf-8"),
                    "retired_agent_transcript_changed")
            require(self._old_bridge_closed(agent) == proof["old_bridge_request_ids"]
                    and not any(row["agent"] == agent for row in self.tools.values())
                    and not any(row["agent"] == agent and row["name"] == "StructuredOutput"
                                for row in self.completed_tools.values()), "retired_agent_late_activity")

    def fail(self, reason, *, error=None, trigger=None):
        first = self.error is None
        self.error = self.error or reason
        if first:
            self._record_first_failure(reason, error, trigger)
        self.condition.notify_all()

    def _record_first_failure(self, reason, error, trigger):
        """Attempt durable first-failure evidence without changing rejection."""
        try:
            leaf = str(error) if isinstance(error, StopRejected) else None
            if leaf is not None and re.fullmatch(r"[a-z0-9_]{1,128}", leaf) is None:
                leaf = None
            record = {"schema": "geak-native-census-first-failure-v1", "reason": reason,
                "at_unix": time.time(), "monotonic_time": time.monotonic(),
                "exception": None if error is None else {"module": type(error).__module__,
                    "type": type(error).__name__, "code": leaf},
                "trigger": deepcopy(trigger), "session_id": self.session_id,
                "root_tool": self.root_tool, "root_task": self.root_task,
                "journal_descriptor": deepcopy(self.journal_descriptor), "closed": self.closed,
                "closed_at": self.closed_at,
                "checkpoint_active": self.checkpoint_active,
                "nodes": [_identity_evidence(node) for node in self.nodes.values()],
                "pending_retirements": {agent: {"old": _identity_evidence(pair["old"]),
                    "new": _identity_evidence(pair["new"]), "observed_monotonic": pair["observed_monotonic"]}
                    for agent, pair in self.pending_retirements.items()},
                "retired_agents": sorted(self.retired),
                "superseded_pending_agents": sorted(self.superseded_pending),
                "mirrors": {agent: {"run_id": mirror.get("run_id"), "project_key": mirror.get("project_key"),
                    "task_sha256": _evidence_digest(mirror.get("task"))} for agent, mirror in self.mirrors.items()},
                "active_tools": {identity: {"name": operation.get("name"), "agent": operation.get("agent"),
                    "input_sha256": _evidence_digest(operation.get("input"))} for identity, operation in self.tools.items()},
                "background": deepcopy(self.background)}
            self.first_failure = record
            directory = getattr(self.controller, "state_dir", None)
            if directory is None:
                return
            directory = Path(directory)
            info = directory.lstat()
            if not (directory.is_absolute() and directory.resolve() == directory and stat.S_ISDIR(info.st_mode)
                    and info.st_uid == os.geteuid()):
                return
            raw = (canonical(record) + "\n").encode("ascii")
            descriptor = os.open(directory / "native_census_first_failure.json",
                                 os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            self.first_failure_persisted = True
        except BaseException:
            # Diagnostic errors must not clear the failure or replace its cause.
            pass

    def verify_sources(self):
        for path, digest in self.sources.items():
            require(path.resolve() == path and hashlib.sha256(path.read_bytes()).hexdigest() == digest,
                    "stopping_workflow_source_changed")

    def close(self):
        with self.condition:
            if not self.closed:
                self.closed_at = {"at_unix": time.time(), "monotonic_time": time.monotonic()}
            self.closed = True
            self.condition.notify_all()

    def record_transport(self, record):
        """Persist a private transport census without request text or credentials."""
        with self.condition:
            require(not self.transport_sealed, "native_transport_after_closure")
            record = {"schema": "geak-quality-bridge-event-v1", "monotonic_time": time.monotonic(), **record}
            raw = (canonical(record) + "\n").encode("ascii")
            directory = getattr(self.controller, "state_dir", None)
            if directory is not None:
                try:
                    descriptor = os.open(Path(directory) / "bridge.jsonl",
                                         os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
                    with os.fdopen(descriptor, "ab") as stream:
                        info = os.fstat(stream.fileno())
                        require(stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid() and info.st_nlink == 1,
                                "bridge_ledger_identity_changed")
                        stream.write(raw)
                        stream.flush()
                        os.fsync(stream.fileno())
                except (OSError, StopRejected) as error:
                    self.fail("bridge_ledger_unavailable", error=error,
                              trigger={"kind": "transport_record", "record_sha256": _evidence_digest(record)})
                    raise StopRejected("bridge_ledger_unavailable") from None
            self.request_records.append(deepcopy(record))
            self.condition.notify_all()

    def pin_root(self, data, tool_id):
        require(data.get("session_id") == self.session_id and not data.get("agent_id")
                and data.get("tool_name") == "Workflow" and tool_id, "stopping_root_hook_identity")
        require(json_values_equal(data.get("tool_input"), self.root_request) and self.root_tool in (None, tool_id), "stopping_root_request_changed")
        self.verify_sources()
        self.root_tool = tool_id

    def _bash_event(self, kind, value):
        raw = value.get("data", {})
        require(isinstance(raw, dict), "background_bash_event_invalid")
        aliases = {"taskId", "toolUseId", "agentId", "sessionId", "taskType", "backgroundTaskId", "background_task_id"}
        require(not aliases.intersection(value) and not aliases.intersection(raw), "background_bash_event_invalid")
        event = {"kind": kind, "monotonic_time": time.monotonic()}
        for key in ("session_id", "task_id", "tool_use_id", "task_type", "agent_id", "uuid", "status"):
            if key in value and key in raw:
                require(json_values_equal(value[key], raw[key]), "background_bash_event_invalid")
            item = value.get(key, raw.get(key))
            require(item is None or isinstance(item, str), "background_bash_event_invalid")
            event[key] = item
        if kind == "TaskUpdatedMessage":
            patch = value.get("patch", raw.get("patch"))
            require(isinstance(patch, dict) and ("patch" not in raw or json_values_equal(patch, raw["patch"]))
                    and event["status"] == patch.get("status"), "background_bash_event_invalid")
            require(not aliases.intersection(patch), "background_bash_event_invalid")
            event["patch"] = deepcopy(patch)
        require(event["session_id"] == self.session_id and isinstance(event["task_id"], str)
                and BASH_TASK_ID.fullmatch(event["task_id"]), "background_bash_event_invalid")
        event["event_sha256"] = _evidence_digest(value)
        require(event["event_sha256"] is not None, "background_bash_event_invalid")
        event["native_event"] = deepcopy(value)
        return event

    def _bash_binding(self, task, tool):
        operation = self.tools.get(tool, self.completed_tools.get(tool))
        require(isinstance(operation, dict) and operation.get("name") == "Bash"
                and isinstance(operation.get("input"), dict) and operation.get("agent"),
                "background_bash_binding_changed")
        require(operation["input"].get("run_in_background", False) is False,
                "background_bash_requires_qualified_adapter")
        require(self.bash_task_tools.get(tool, task) == task, "background_bash_binding_changed")
        binding = {"schema": "geak-native-automatic-bash-v1", "task_id": task, "tool_use_id": tool,
                   "agent_id": operation["agent"], "input_sha256": _evidence_digest(operation["input"])}
        require(binding["input_sha256"] is not None, "background_bash_binding_changed")
        record = self.bash_tasks.get(task)
        if record is None:
            record = {**binding, "start": None, "result": None, "terminal": None, "events": []}
            self.bash_tasks[task] = record
            self.bash_task_tools[tool] = task
        else:
            require(all(record[key] == item for key, item in binding.items()), "background_bash_binding_changed")
        return record

    def _bash_started(self, value):
        event = self._bash_event("TaskStartedMessage", value)
        task, tool = event["task_id"], event["tool_use_id"]
        require(event["task_type"] == "local_bash" and isinstance(tool, str) and tool,
                "background_bash_event_invalid")
        record = self._bash_binding(task, tool)
        require(record["start"] is None, "duplicate_background_task_start")
        require(event["agent_id"] in (None, record["agent_id"]), "background_bash_binding_changed")
        record["start"] = event
        record["events"].append(deepcopy(event))
        self.background[task] = "active"

    def _bash_terminal(self, kind, value):
        event = self._bash_event(kind, value)
        task = event["task_id"]
        require(task in self.bash_tasks and self.bash_tasks[task]["start"] is not None,
                "background_bash_orphan_event")
        record = self.bash_tasks[task]
        require(event["tool_use_id"] in (None, record["tool_use_id"])
                and event["agent_id"] in (None, record["agent_id"])
                and event["task_type"] in (None, "local_bash"), "background_bash_binding_changed")
        patch = event.get("patch", {})
        identity = {"task_id": task, "id": task, "tool_use_id": record["tool_use_id"],
                    "agent_id": record["agent_id"], "session_id": self.session_id,
                    "task_type": "local_bash", "type": "local_bash"}
        require(all(patch[key] == expected for key, expected in identity.items() if key in patch),
                "background_bash_binding_changed")
        if kind == "TaskNotificationMessage":
            require(event["tool_use_id"] == record["tool_use_id"] and event["status"] in BASH_TERMINAL,
                    "background_bash_event_invalid")
        status = event["status"]
        require(status is None or status in {"pending", "running", *BASH_TERMINAL}, "background_bash_event_invalid")
        if status not in BASH_TERMINAL:
            require(record["terminal"] is None or status is None, "background_bash_terminal_changed")
            return
        require(not any(row["kind"] == kind and row.get("status") in BASH_TERMINAL for row in record["events"]),
                "background_bash_duplicate_terminal")
        terminal = record["terminal"]
        require(terminal is None or BASH_TERMINAL[terminal["status"]] == BASH_TERMINAL[status],
                "background_bash_terminal_changed")
        record["terminal"] = terminal or event
        record["events"].append(deepcopy(event))
        self.background[task] = BASH_TERMINAL[status]

    def _bash_result(self, tool, operation, response, failed):
        inputs = operation["input"]
        require(isinstance(inputs, dict) and inputs.get("run_in_background", False) is False,
                "background_bash_requires_qualified_adapter")
        if not isinstance(response, dict):
            return
        require(not any(key in response for key in ("task_id", "taskId", "background_task_id", "backgroundedByUser")),
                "background_bash_result_unsupported")
        if "backgroundTaskId" not in response:
            require(not any(key in response for key in ("timedOutAfterMs", "backgroundCwdHint")),
                    "background_bash_result_unsupported")
            return
        task = response["backgroundTaskId"]
        timeout = inputs.get("timeout", 120000)
        require(not failed and isinstance(task, str) and BASH_TASK_ID.fullmatch(task)
                and type(timeout) is int and timeout > 0
                and set(response) <= BASH_RESULT_FIELDS
                and isinstance(response.get("stdout"), str) and isinstance(response.get("stderr"), str)
                and response.get("interrupted") is False and response.get("isImage", False) is False
                and type(response.get("timedOutAfterMs")) is int
                and response["timedOutAfterMs"] == min(timeout, 600000), "background_bash_result_unsupported")
        require(all(isinstance(response[key], str) for key in ("returnCodeInterpretation", "backgroundCwdHint",
                    "persistedOutputPath", "staleReadFileStateHint", "ghRateLimitHint") if key in response)
                and all(type(response[key]) is bool for key in ("noOutputExpected", "dangerouslyDisableSandbox") if key in response)
                and ("persistedOutputSize" not in response or type(response["persistedOutputSize"]) is int
                     and response["persistedOutputSize"] >= 0), "background_bash_result_unsupported")
        record = self._bash_binding(task, tool)
        require(record["result"] is None, "background_bash_binding_changed")
        result = {"kind": "PostToolUse", "task_id": task, "tool_use_id": tool,
                  "agent_id": operation["agent"], "input_sha256": record["input_sha256"],
                  "response_sha256": _evidence_digest(response), "timed_out_after_ms": response["timedOutAfterMs"],
                  "monotonic_time": time.monotonic()}
        record["result"] = result
        record["events"].append(deepcopy(result))
        self.background[task] = (BASH_TERMINAL[record["terminal"]["status"]]
                                 if record["terminal"] is not None else "active")

    def _automatic_bash_proof(self):
        proof = {}
        for task, record in self.bash_tasks.items():
            if record["result"] is None:
                continue
            require(record["start"] is not None and record["terminal"] is not None
                    and self.background.get(task) in set(BASH_TERMINAL.values()), "background_bash_binding_incomplete")
            operation = self.completed_tools.get(record["tool_use_id"], {})
            require(operation.get("name") == "Bash" and operation.get("agent") == record["agent_id"]
                    and operation.get("failed") is False
                    and _evidence_digest(operation.get("input")) == record["input_sha256"]
                    and _evidence_digest(operation.get("response")) == record["result"]["response_sha256"],
                    "background_bash_binding_changed")
            proof[task] = deepcopy(record)
        return proof

    def observe(self, message):
        kind = type(message).__name__
        value = asdict(message) if is_dataclass(message) else vars(message)
        data = value.get("data", {})
        with self.condition:
            try:
                if kind == "UserMessage":
                    self._root_result(value)
                elif kind == "AssistantMessage" and value.get("parent_tool_use_id") is None:
                    self._observe_root_tools(value)
                elif kind in {"TaskStartedMessage", "TaskProgressMessage", "TaskNotificationMessage", "TaskUpdatedMessage"}:
                    session = value.get("session_id", data.get("session_id"))
                    require(session == self.session_id, "native_event_session_changed")
                    task = value.get("task_id", data.get("task_id"))
                    tool = value.get("tool_use_id", data.get("tool_use_id"))
                    if kind == "TaskStartedMessage":
                        require(isinstance(task, str) and task, "native_task_identity_missing")
                        if tool == self.root_tool and value.get("task_type", data.get("task_type")) == "local_workflow":
                            require(self.root_task in (None, task), "second_native_root_task")
                            self.root_task = task
                        else:
                            require(self.checkpoint_active is None, "producer_started_during_checkpoint")
                            self._bash_started(value)
                    elif kind in {"TaskNotificationMessage", "TaskUpdatedMessage"} and task != self.root_task:
                        self._bash_terminal(kind, value)
                    if "workflow_progress" in data:
                        require(task == self.root_task and tool == self.root_tool, "native_progress_root_changed")
                        self._progress(data["workflow_progress"])
            except (StopRejected, OSError, ValueError, TypeError, KeyError, AttributeError) as error:
                self.fail("native_census_event_conflict", error=error,
                          trigger={"kind": kind, "event": _identity_evidence(value)})
            self.condition.notify_all()

    def _root_result(self, value):
        blocks = value.get("content", [])
        if not isinstance(blocks, list) or not self.root_tool:
            return
        matching = [block for block in blocks if isinstance(block, dict) and block.get("tool_use_id") == self.root_tool]
        if not matching:
            return
        result = value.get("tool_use_result")
        require(len(matching) == 1 and matching[0].get("is_error") is False and value.get("parent_tool_use_id") is None,
                "native_root_result_changed")
        require(isinstance(result, dict) and result.get("status") == "async_launched"
                and result.get("taskType") == "local_workflow" and result.get("taskId") == self.root_task
                and result.get("scriptPath") == self.root_request["scriptPath"], "native_root_result_identity")
        directory = Path(result["transcriptDir"])
        require(directory.is_absolute() and len(directory.parts) >= 6 and directory.parts[-6] == "projects",
                "native_journal_directory_invalid")
        descriptor = {"directory": str(directory), "run_id": result["runId"], "project_key": directory.parts[-5]}
        require(self.journal_descriptor in (None, descriptor), "native_journal_descriptor_changed")
        if self.journal is None:
            self.journal = NativeJournal(directory, session_id=self.session_id, run_id=result["runId"])
            self.journal_descriptor = descriptor

    def _progress(self, progress):
        require(isinstance(progress, list) and all(isinstance(item, dict) for item in progress), "native_progress_invalid")
        current = {}
        logical_slots = set()
        for node in progress:
            if node.get("type") != "workflow_agent":
                continue
            require(type(node.get("phaseIndex")) is int and node["phaseIndex"] > 0
                    and type(node.get("index")) is int and type(node.get("attempt")) is int
                    and node["index"] > 0 and 1 <= node["attempt"] <= MAX_NATIVE_ATTEMPTS
                    and isinstance(node.get("label"), str) and node["label"],
                    "native_node_identity_invalid")
            require(not node.get("cached") and node.get("state") != "cached", "cached_producer_unsupported")
            key = (node.get("phaseIndex"), node["index"], node["attempt"])
            require(key not in current, "duplicate_native_node")
            require(key[:2] not in logical_slots, "duplicate_native_logical_slot")
            logical_slots.add(key[:2])
            require(node.get("agentId") not in self.retired, "retired_agent_reappeared_in_progress")
            if node.get("agentId") in self.pending_retirements:
                require(node.get("state") in {"start", "progress"}, "retirement_pending_successor_became_terminal")
            if key not in self.nodes and not any(old_key[:2] == key[:2] for old_key in self.nodes):
                require(node["attempt"] == 1, "native_initial_attempt_not_one")
                require(key[:2] not in self.logical_labels, "native_logical_slot_reused")
                self.logical_labels[key[:2]] = node["label"]
            if key in self.nodes:
                old = self.nodes[key]
                require(old.get("label") == node["label"] and old.get("agentId") in (None, node.get("agentId")),
                        "native_node_changed")
                if old.get("state") in {"done", "error", "failed", "stopped", "cancelled"}:
                    terminal_fields = ("state", "agentId", "label", "index", "attempt", "phaseIndex", "lastToolName", "resultPreview")
                    require(all(old.get(field) == node.get(field) for field in terminal_fields), "native_terminal_node_changed")
            self._logical_label(node)
            current[key] = deepcopy(node)
        if self.checkpoint_active is not None:
            require(set(current) == set(self.nodes), "native_admission_during_checkpoint")
        missing = set(self.nodes) - set(current)
        for key in missing:
            old = self.nodes[key]
            successors = [node for candidate, node in current.items() if candidate[:2] == key[:2]]
            require(len(successors) == 1, "native_producer_disappeared")
            new = successors[0]
            require(old.get("agentId") and new.get("agentId") and old["agentId"] != new["agentId"]
                    and new["attempt"] == old["attempt"] + 1
                    and self._logical_label(old) == self._logical_label(new)
                    and not self._logical_label(old).startswith("quality_stop:")
                    and old.get("state") in {"start", "progress"} and new.get("state") in {"start", "progress"}
                    and type(old.get("phaseIndex")) is int and old["phaseIndex"] > 0,
                    "native_replacement_unsupported")
            require(new["agentId"] not in self.retired and new["agentId"] not in self.pending_retirements
                    and new["agentId"] not in self.superseded_pending
                    and not any(pair["old"]["agentId"] == new["agentId"] for pair in self.pending_retirements.values())
                    and not any(node.get("agentId") == new["agentId"] for node in self.nodes.values()),
                    "native_replacement_identity_reused")
            if old["agentId"] in self.pending_retirements:
                require(not any(row.get("agent_id") == old["agentId"] and row.get("event") in {"forwarding", "local_prepared"}
                                for row in self.request_records)
                        and not any(row["agent"] == old["agentId"] for row in self.tools.values())
                        and not any(row["agent"] == old["agentId"] for row in self.completed_tools.values()),
                        "retirement_pending_successor_already_active")
                self.superseded_pending.add(old["agentId"])
            self.pending_retirements[new["agentId"]] = {"old": deepcopy(old), "new": deepcopy(new),
                                                       "observed_monotonic": time.monotonic()}
        self.nodes = current
        self.last_progress = deepcopy(progress)

    def mirror_append(self, key, entries):
        with self.condition:
            try:
                if key.get("session_id") != self.session_id or not key.get("subpath"):
                    return
                match = re.fullmatch(r"subagents/workflows/([^/]+)/agent-([^/]+)", key["subpath"])
                if match is None:
                    return
                for entry in entries:
                    if entry.get("type") != "user" or entry.get("parentUuid") is not None:
                        continue
                    agent = match[2]
                    require(entry.get("agentId", agent) == agent and entry.get("sessionId", self.session_id) == self.session_id
                            and entry.get("cwd") == str(self.workspace), "native_mirror_identity_changed")
                    task = entry.get("message", {}).get("content")
                    require(isinstance(task, str) and task, "native_mirror_task_missing")
                    record = {"task": task, "run_id": match[1], "project_key": key.get("project_key")}
                    require(agent not in self.mirrors or self.mirrors[agent] == record, "native_mirror_task_changed")
                    self.mirrors[agent] = record
                    initial = {"descriptor": deepcopy(key), "initial_entry": deepcopy(entry)}
                    require(agent not in self.mirror_inputs or self.mirror_inputs[agent] == initial,
                            "native_mirror_initial_entry_changed")
                    self.mirror_inputs[agent] = initial
            except (StopRejected, ValueError, TypeError, KeyError, AttributeError) as error:
                self.fail("native_census_mirror_conflict", error=error,
                          trigger={"kind": "mirror_append", "descriptor": _identity_evidence(key),
                                   "entries_sha256": _evidence_digest(entries)})
            self.condition.notify_all()

    def _node(self, agent):
        nodes = [node for node in self.nodes.values() if node.get("agentId") == agent]
        nodes += [pair["old"] for pair in self.pending_retirements.values() if pair["old"].get("agentId") == agent]
        require(len(nodes) == 1, "native_agent_node_missing_or_duplicate")
        return nodes[0]

    def _logical_label(self, node):
        logical = self.logical_labels.get((node.get("phaseIndex"), node.get("index")))
        require(node.get("label") == native_attempt_label(logical, node.get("attempt")), "native_retry_label_changed")
        return logical

    def assert_quiescent(self, checkpoint_agent):
        with self.condition:
            self._assert_quiescent(checkpoint_agent)

    def confirm_return(self, result):
        """Check full native closure after the Workflow returns to its runner."""
        with self.condition:
            quality = result.get("quality_stop") if isinstance(result, dict) else None
            if not isinstance(quality, dict) or (quality.get("qualifying") is not True and quality.get("enabled") is not True):
                return
            require(self.error is None and self.journal is not None, "native_closure_unknown")
            deadline = time.monotonic() + self.wait_seconds
            while not self._bridge_all_terminal():
                remaining = deadline - time.monotonic()
                require(remaining > 0, "native_closure_bridge_pending")
                self.condition.wait(timeout=min(remaining, .1))
            self.verify_sources()
            require(not self.tools and all(status != "active" for status in self.background.values()), "native_closure_producer_active")
            automatic_bash = self._automatic_bash_proof()
            snapshot = self.journal.snapshot()
            require(snapshot["status"] == "complete", "native_closure_journal_unstable")
            self._validate_retirements(snapshot)
            agents = {node.get("agentId"): node for node in self.nodes.values() if node.get("agentId")}
            all_agents = set(agents) | set(self.retired)
            require(len(agents) == len(self.nodes) and all_agents == set(snapshot["started_agents"])
                    and set(agents) == set(snapshot["results"]), "native_closure_sets_differ")
            require(all_agents == set(self.mirrors) == set(self.mirror_inputs), "native_closure_mirrors_differ")
            for agent, node in agents.items():
                outcome = snapshot["results"][agent]
                require(node.get("state") == "done" and outcome.get("status") == "complete", "native_closure_not_done")
                raw = outcome["result_json"]
                units = raw.encode("utf-16-le", errors="surrogatepass")
                preview = units[:800].decode("utf-16-le", errors="surrogatepass") + "…" if len(units) > 800 else raw
                require(node.get("resultPreview") == preview and node.get("lastToolName") == "StructuredOutput",
                        "native_closure_result_changed")
                if agent in self.emissions:
                    emitted = self.emissions[agent]
                    require(emitted.get("accepted") is True and json_values_equal(outcome["value"], emitted["block"]["input"]),
                            "native_closure_checkpoint_changed")
            raw_journal = self.journal.bound_bytes(snapshot)
            raw_bridge = b"".join((canonical(row) + "\n").encode("ascii") for row in self.request_records)
            directory = getattr(self.controller, "state_dir", None)
            bridge_path = Path(directory) / "bridge.jsonl" if directory is not None else None
            if bridge_path is not None:
                descriptor = os.open(bridge_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                with os.fdopen(descriptor, "rb") as stream:
                    before = os.fstat(stream.fileno())
                    require(stat.S_ISREG(before.st_mode) and before.st_uid == os.geteuid() and before.st_nlink == 1
                            and before.st_size == len(raw_bridge), "native_closure_bridge_changed")
                    observed = stream.read()
                    after = os.fstat(stream.fileno())
                    current = bridge_path.stat(follow_symlinks=False)
                fields = ("st_dev", "st_ino", "st_uid", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
                require(observed == raw_bridge and all(getattr(before, key) == getattr(after, key) == getattr(current, key)
                        for key in fields), "native_closure_bridge_changed")
            census = {"schema": "geak-quality-native-census-v1", "session_id": self.session_id,
                "root_tool_id": self.root_tool, "root_task_id": self.root_task,
                "root_request": deepcopy(self.root_request), "workspace": str(self.workspace),
                "stopping_enabled": self.controller.public_config.get("stopping_enabled"),
                "nodes": deepcopy(list(self.nodes.values())),
                "mirrors": {agent: {**deepcopy(mirror), **deepcopy(self.mirror_inputs[agent])}
                            for agent, mirror in self.mirrors.items()},
                "completed_tools": deepcopy(self.completed_tools), "emissions": deepcopy(self.emissions),
                "background": deepcopy(self.background), "active_tools": {}, "active_background": [],
                "request_records": deepcopy(self.request_records), "bridge_all_terminal": True,
                "bridge": {"path": str(bridge_path) if bridge_path is not None else None,
                    "sha256": hashlib.sha256(raw_bridge).hexdigest(), "bytes": len(raw_bridge),
                    "raw_utf8": raw_bridge.decode("ascii")},
                "journal": {"descriptor": deepcopy(self.journal_descriptor), "snapshot": deepcopy(snapshot),
                            "raw_utf8": raw_journal.decode("utf-8")},
                "sources_sha256": {str(path): value for path, value in self.sources.items()}}
            if automatic_bash:
                census["automatic_bash"] = automatic_bash
            if self.retirement_events:
                path = self._retirement_path()
                raw = canonical_bytes(self.retirement_events)
                descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                with os.fdopen(descriptor, "rb") as stream:
                    before = os.fstat(stream.fileno())
                    require(stat.S_ISREG(before.st_mode) and before.st_uid == os.geteuid() and before.st_nlink == 1,
                            "native_retirement_ledger_changed")
                    observed = stream.read()
                    after = os.fstat(stream.fileno())
                    current = path.stat(follow_symlinks=False)
                fields = ("st_dev", "st_ino", "st_uid", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
                require(observed == raw and all(getattr(before, key) == getattr(after, key) == getattr(current, key)
                        for key in fields), "native_retirement_ledger_changed")
                census["retirements"] = {"schema": "geak-native-enforced-retirements-v1", "mode": RETIREMENT_MODE,
                    "events": deepcopy(self.retirement_events), "path": str(path), "raw_utf8": raw.decode("ascii"),
                    "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            self.transport_sealed = True
            if automatic_bash:
                census["automatic_bash_closed_monotonic_time"] = time.monotonic()
            return census

    def _bridge_all_terminal(self):
        groups = {}
        for record in self.request_records:
            key = record.get("request_id")
            require(isinstance(key, str) and key, "native_closure_bridge_sequence_invalid")
            groups.setdefault(key, []).append(record)
        for key, records in groups.items():
            events = [row.get("event") for row in records]
            terminals = [name for name in events if name in {"finished", "failed"}]
            routed = [name for name in events if name in {"forwarding", "local_prepared"}]
            require(events[0] == "started" and events.count("started") == 1 and len(terminals) <= 1
                    and len(routed) <= 1 and all(name in {"started", "forwarding", "local_prepared", "finished", "failed"}
                                               for name in events), "native_closure_bridge_sequence_invalid")
            if not terminals:
                return False
            require(events[-1] == terminals[0] and (len(routed) == 1 or events == ["started", "failed"]),
                    "native_closure_bridge_sequence_invalid")
            if events == ["started", "failed"]:
                last = records[-1]
                proof = self.retired.get(records[0].get("agent_id"), {})
                require(key in proof.get("old_unforwarded_request_ids", [])
                        and last.get("reason") == "native_pending_successor_superseded"
                        and last.get("route") == "unclassified" and last.get("upstream_bridge_attempted") is False
                        and last.get("provider_attempted") is False, "native_closure_unqualified_local_refusal")
        return True

    def _assert_quiescent(self, checkpoint_agent):
        require(not self.closed and self.error is None and self.journal is not None, "native_census_unavailable")
        self.verify_sources()
        active = self._node(checkpoint_agent)
        require(active.get("state") not in {"done", "error", "failed", "stopped", "cancelled"}, "checkpoint_already_terminal")
        require(not self.tools and all(status != "active" for status in self.background.values()), "native_background_producer_active")
        self._automatic_bash_proof()
        snapshot = self.journal.snapshot()
        require(snapshot["status"] == "complete", "native_journal_not_stable")
        self._validate_retirements(snapshot)
        started, results = set(snapshot["started_agents"]), snapshot["results"]
        nodes = {node.get("agentId"): node for node in self.nodes.values() if node.get("agentId")}
        require(len(nodes) == len(self.nodes), "native_queued_producer_unknown")
        require(set(nodes) | set(self.retired) == started, "native_producer_sets_differ")
        require(checkpoint_agent in started and checkpoint_agent not in results, "checkpoint_journal_state_invalid")
        require(set(results) == set(nodes) - {checkpoint_agent}, "native_producer_result_missing")
        for agent in set(nodes) - {checkpoint_agent}:
            node, outcome = nodes[agent], results[agent]
            require(node.get("state") == "done" and outcome.get("status") == "complete", "native_producer_not_successful_terminal")
            raw = outcome.get("result_json")
            require(isinstance(raw, str), "native_full_result_missing")
            units = raw.encode("utf-16-le", errors="surrogatepass")
            preview = units[:800].decode("utf-16-le", errors="surrogatepass") + "…" if len(units) > 800 else raw
            require(node.get("lastToolName") == "StructuredOutput" and node.get("resultPreview") == preview,
                    "native_terminal_result_changed")
            if agent in self.emissions:
                emitted = self.emissions[agent]
                require(emitted.get("accepted") is True and json_values_equal(outcome["value"], emitted["block"]["input"]),
                        "checkpoint_native_result_changed")
        request = parse_task(self.mirrors[checkpoint_agent]["task"])
        if request["stage"] == "seed":
            require([self._logical_label(node) for agent, node in nodes.items() if agent != checkpoint_agent] == ["director:setup"],
                    "seed_handoff_after_optimizer_work")
            return
        reclaim = [node for node in nodes.values() if self._logical_label(node) == "storage:reclaim r" + str(request["round"])]
        require(len(reclaim) == 1, "completed_storage_reclaim_missing")
        reclaim_agent = reclaim[0]["agentId"]
        prompt = self.mirrors.get(reclaim_agent, {}).get("task", "")
        commands = re.findall(r"```bash\n(.*?)\n```", prompt, flags=re.DOTALL)
        require(len(commands) == 1, "storage_reclaim_command_missing")
        executions = [operation for operation in self.completed_tools.values()
                      if operation["agent"] == reclaim_agent and operation["name"] == "Bash"
                      and operation["input"].get("command") == commands[0]]
        require(len(executions) == 1, "storage_reclaim_execution_missing_or_duplicate")
        outcome = executions[0]
        native = outcome["response"]
        require(not outcome["failed"] and isinstance(native, dict) and native.get("interrupted") is False
                and not any(native.get(key) for key in ("backgroundTaskId", "background_task_id", "truncated", "isImage"))
                and isinstance(native.get("stdout"), str)
                and native["stdout"].rstrip().endswith("STORAGE_RECLAIM_DONE round=" + str(request["round"])),
                "storage_reclaim_completion_unknown")

    def _wait_identity(self, agent, denial_event="denied_model", maximum_wait_seconds=None, denial_detail=None):
        deadline = time.monotonic() + self.wait_seconds
        maximum_deadline = None if maximum_wait_seconds is None else time.monotonic() + maximum_wait_seconds
        while True:
            self._guard_retired(agent, denial_event, detail=denial_detail)
            if denial_event == "denied_tool":
                self._guard_superseded_tool(agent, denial_detail)
            require(agent not in self.superseded_pending, "native_pending_successor_superseded")
            pending = self.pending_retirements.get(agent)
            if pending is not None:
                deadline = pending["observed_monotonic"] + self.retirement_wait_seconds
                if (self._retirement_remaining(pending) <= 0
                        or maximum_deadline is not None and time.monotonic() >= maximum_deadline):
                    self.fail("native_retirement_timeout", trigger={"kind": "retirement_wait", "agent_id": agent})
                    raise StopRejected("native_retirement_timeout")
                try:
                    self._try_retirements()
                except BaseException as error:
                    self.fail("native_retirement_evidence_failed", error=error,
                              trigger={"kind": "retirement_wait", "agent_id": agent})
                    raise
            if (self.root_task and self.journal and agent in self.mirrors
                    and agent not in self.pending_retirements
                    and (any(node.get("agentId") == agent for node in self.nodes.values())
                         or any(pair["old"].get("agentId") == agent for pair in self.pending_retirements.values()))):
                break
            if (self.closed or self.error is not None) and self.first_failure is None:
                self._record_first_failure(self.error or "native_registry_closed",
                    StopRejected("native_identity_unavailable"), {"kind": "identity_wait", "agent_id": agent})
            require(not self.closed and self.error is None, "native_identity_unavailable")
            remaining = self._retirement_remaining(pending) if pending is not None else deadline - time.monotonic()
            if maximum_deadline is not None:
                remaining = min(remaining, maximum_deadline - time.monotonic())
            if remaining <= 0 and pending is not None:
                self.fail("native_retirement_timeout", trigger={"kind": "retirement_wait", "agent_id": agent})
                raise StopRejected("native_retirement_timeout")
            require(remaining > 0, "native_identity_timeout")
            self.condition.wait(min(remaining, .05))

    def response(self, headers, raw):
        """Return only local signed checkpoint output, with no provider fallback."""
        try:
            return self._response(headers, raw)
        except (StopRejected, OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError) as error:
            reason = str(error) if isinstance(error, StopRejected) else "invalid_native_checkpoint_request"
            if re.fullmatch(r"[a-z0-9_]{1,128}", reason) is None:
                reason = "native_checkpoint_rejected"
            with self.condition:
                self.rejections.append({"reason": reason, "time": time.monotonic()})
            raise

    def _response(self, headers, raw):
        with self.condition:
            fields = {}
            for key, value in headers:
                key = key.lower()
                if key.startswith("x-claude-code-"):
                    require(key not in fields, "duplicate_native_identity_header")
                    fields[key] = value
            body = _object(raw)
            agent = fields.get("x-claude-code-agent-id")
            self._guard_retired(agent, "denied_model")
            if not agent:
                # Missing child headers must never route a checkpoint to inference.
                require(not _contains_checkpoint(body), "checkpoint_identity_missing")
                require(self.error is None and not self.closed, "native_census_unavailable")
                self._root_model(fields, body)
                return None
            require(fields.get("x-claude-code-session-id") == self.session_id
                    and not fields.get("x-claude-code-parent-agent-id"), "native_request_identity_changed")
            self._wait_identity(agent)
            require(self.error is None and not self.closed, "native_census_unavailable")
            node, mirror = self._node(agent), self.mirrors[agent]
            marker = node["label"].startswith("quality_stop:") or mirror["task"].startswith(PREFIX)
            if not marker:
                return None
            require(agent not in self.emissions, "checkpoint_continuation_forbidden")
            require(self.error is None and not self.closed, "native_census_unavailable")
            self.verify_sources()
            request = parse_task(mirror["task"])
            label = "quality_stop:seed" if request["stage"] == "seed" else (
                "quality_stop:" + request["stage"] + " r" + str(request["round"]) + " l" + str(request["look_index"]))
            require(node["label"] == label and node["attempt"] == 1, "checkpoint_native_call_changed")
            require(mirror["run_id"] == self.journal_descriptor["run_id"]
                    and mirror["project_key"] == self.journal_descriptor["project_key"], "checkpoint_mirror_changed")
            schemas = [tool.get("input_schema") for tool in body.get("tools", []) if tool.get("name") == "StructuredOutput"]
            require(json_values_equal(schemas, [SCHEMA]) and checkpoint_initial_messages(body.get("messages"), mirror["task"]),
                    "checkpoint_request_changed")
            # A route error consumes no GPU process. A started controller batch
            # consumes a look even when census or evaluator checks later fail.
            binding = {"session_id": self.session_id, "run_id": mirror["run_id"], "root_tool": self.root_tool,
                       "root_task": self.root_task, "agent_id": agent}
            self.emissions[agent] = {"pending": True, "accepted": False}
            require(self.checkpoint_active is None, "checkpoint_overlap")
            self.checkpoint_active = agent
            # Native events must continue during measurement. Holding this lock
            # would hide late producers from the second census.
            self.condition.release()
            try:
                envelope = self.controller.checkpoint(mirror["task"], binding,
                                                      census=lambda: self.assert_quiescent(agent))
            except BaseException:
                with self.condition:
                    self.checkpoint_active = None
                raise
            finally:
                self.condition.acquire()
            identity = "quality_stop_" + hashlib.sha256((agent + mirror["task"]).encode()).hexdigest()
            block = {"type": "tool_use", "id": identity, "name": "StructuredOutput", "input": envelope}
            self.emissions[agent] = {"block": block, "started": False, "accepted": False}
            return deepcopy(block)

    def _wait_hook_identity(self, data, tool_id):
        with self.condition:
            agent, name = data.get("agent_id"), data.get("tool_name")
            detail = {**data, "tool_use_id": tool_id}
            self._guard_retired(agent, "denied_tool", detail=detail)
            self._guard_superseded_tool(agent, detail)
            require(data.get("session_id") == self.session_id and data.get("cwd") == str(self.workspace),
                    "native_tool_hook_identity_changed")
            if agent:
                # The SDK receives timeout=30, but its enforcement is not
                # established. The hook uses its own short identity wait.
                self._wait_identity(agent, "denied_tool", maximum_wait_seconds=min(self.wait_seconds, 10), denial_detail=detail)
            else:
                deadline = time.monotonic() + min(self.wait_seconds, 10)
                require(name in {"Workflow", "TaskOutput", "TaskList", "TaskGet"}, "native_root_tool_unsupported")
                while tool_id not in self.root_tool_bindings:
                    require(self.error is None and not self.closed, "native_census_unavailable")
                    remaining = deadline - time.monotonic()
                    require(remaining > 0, "native_root_tool_identity_timeout")
                    self.condition.wait(min(remaining, .05))
                require(json_values_equal(self.root_tool_bindings[tool_id],
                        {"name": name, "input": data.get("tool_input")}), "native_root_tool_binding_changed")

    async def pre(self, data, tool_use_id=None, context=None):
        tool_id = tool_use_id
        try:
            tool_id = tool_id or data.get("tool_use_id")
            # Native hook callbacks can precede the trusted root message. This
            # worker releases the event loop while it waits for that message.
            await asyncio.to_thread(self._wait_hook_identity, data, tool_id)
            with self.condition:
                detail = {**data, "tool_use_id": tool_id}
                self._guard_retired(data.get("agent_id"), "denied_tool", detail=detail)
                self._guard_superseded_tool(data.get("agent_id"), detail)
                require(self.error is None and not self.closed, "native_census_unavailable")
                require(data.get("session_id") == self.session_id and data.get("cwd") == str(self.workspace),
                        "native_tool_hook_identity_changed")
                name, agent = data.get("tool_name"), data.get("agent_id")
                require(not agent or self.checkpoint_active in (None, agent), "tool_admitted_during_checkpoint")
                if not agent:
                    require(tool_id not in self.root_tool_admitted and tool_id not in self.root_tool_completed,
                            "duplicate_native_root_tool_start")
                    self.root_tool_admitted.add(tool_id)
                if name == "Workflow":
                    self.pin_root(data, tool_id)
                elif agent in self.emissions:
                    block = self.emissions[agent]["block"]
                    require(name == "StructuredOutput" and tool_id == block["id"] and json_values_equal(data.get("tool_input"), block["input"]),
                            "checkpoint_structured_output_changed")
                    require(not self.emissions[agent].get("started") and not self.emissions[agent].get("accepted"),
                            "duplicate_checkpoint_tool_execution")
                    self.emissions[agent]["started"] = True
                elif agent:
                    require(tool_id and tool_id not in self.tools and tool_id not in self.completed_tools,
                            "duplicate_native_tool_start")
                    if name == "Bash":
                        inputs = data.get("tool_input")
                        require(isinstance(inputs, dict) and inputs.get("run_in_background", False) is False,
                                "background_bash_requires_qualified_adapter")
                    self.tools[tool_id] = {"name": name, "agent": agent, "input": deepcopy(data.get("tool_input"))}
                return {}
        except (Exception, asyncio.CancelledError) as error:
            with self.condition:
                self.fail("native_tool_admission_failed", error=error,
                          trigger={"kind": "pre_hook", "tool_use_id": tool_id, "event": _identity_evidence(data)})
            return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                    "permissionDecisionReason": "The stopping controller rejected the native tool binding."}}

    async def post(self, data, tool_use_id=None, context=None):
        tool_id = tool_use_id
        with self.condition:
            try:
                tool_id = tool_id or data.get("tool_use_id")
                require(data.get("session_id") == self.session_id and data.get("cwd") == str(self.workspace),
                        "native_tool_hook_identity_changed")
                agent, name = data.get("agent_id"), data.get("tool_name")
                detail = {**data, "tool_use_id": tool_id}
                self._guard_retired(agent, "denied_tool", detail=detail)
                self._guard_superseded_tool(agent, detail)
                if not agent:
                    observed_input = data.get("tool_input")
                    if name == "Workflow" and isinstance(observed_input, dict) and "script" in observed_input:
                        self.verify_sources()
                        script = (self.source_root / "kernel_workflow.js").read_bytes().decode("utf-8")
                        require(json_values_equal(observed_input, {**self.root_request, "script": script}),
                                "native_root_resolved_script_changed")
                        observed_input = self.root_request
                    require(tool_id in self.root_tool_admitted and tool_id not in self.root_tool_completed
                            and json_values_equal(self.root_tool_bindings.get(tool_id),
                                {"name": name, "input": observed_input}), "native_root_tool_result_changed")
                    self.root_tool_completed.add(tool_id)
                if agent in self.emissions:
                    emitted = self.emissions[agent]
                    require(name == "StructuredOutput" and tool_id == emitted["block"]["id"]
                            and json_values_equal(data.get("tool_input"), emitted["block"]["input"])
                            and data.get("hook_event_name") != "PostToolUseFailure"
                            and emitted.get("started") is True and not emitted.get("accepted"), "checkpoint_result_rejected")
                    emitted["accepted"] = True
                    self.checkpoint_active = None
                elif agent:
                    require(tool_id in self.tools and json_values_equal(self.tools[tool_id],
                            {"name": name, "agent": agent, "input": data.get("tool_input")}), "native_tool_result_changed")
                    operation = self.tools.pop(tool_id)
                    response = data.get("tool_response")
                    self.completed_tools[tool_id] = {**operation, "response": deepcopy(response),
                                                    "failed": data.get("hook_event_name") == "PostToolUseFailure"}
                    if name == "Bash":
                        self._bash_result(tool_id, operation, response, data.get("hook_event_name") == "PostToolUseFailure")
            except Exception as error:
                self.fail("native_tool_outcome_unknown", error=error,
                          trigger={"kind": "post_hook", "tool_use_id": tool_id, "event": _identity_evidence(data)})
            self.condition.notify_all()
            return {}


class CheckpointTransport:
    """Forward scientific calls unchanged and reject unsupported local routes."""

    def __init__(self, upstream, registry, *, base_path=""):
        self.upstream, self.registry, self.base_path = upstream, registry, base_path.rstrip("/")

    @contextmanager
    def send(self, method, target, headers, body):
        fields = {name.lower(): value for name, value in headers}
        request_id = "quality_bridge_" + uuid.uuid4().hex
        identity = {"request_id": request_id, "method": method, "path": urlsplit(target).path,
                    "body_sha256": hashlib.sha256(body).hexdigest(), "body_bytes": len(body),
                    "session_id": fields.get("x-claude-code-session-id"),
                    "agent_id": fields.get("x-claude-code-agent-id")}
        self.registry.record_transport({**identity, "event": "started"})
        upstream_attempted = False
        route = "unclassified"
        bootstrap = False
        try:
            native_names = [name.lower() for name, _ in headers if name.lower().startswith("x-claude-code-")]
            require(len(native_names) == len(set(native_names)), "duplicate_native_identity_header")
            with self.registry.condition:
                self.registry._guard_retired(fields.get("x-claude-code-agent-id"), "denied_model", detail=identity)
            if method == "POST" and urlsplit(target).path == self.base_path + "/v1/messages":
                require(fields.get("content-encoding", "identity").lower() == "identity"
                        and fields.get("content-type", "").split(";", 1)[0].strip().lower() == "application/json",
                        "checkpoint_transport_encoding_unsupported")
                block = self.registry.response(headers, body)
                if block is not None:
                    route = "local_controller"
                    self.registry.record_transport({**identity, "event": "local_prepared", "route": route,
                        "origin": "quality_stop_controller", "upstream_bridge_attempted": False,
                        "provider_attempted": False,
                        "tool_use_id": block["id"], "envelope_sha256": hashlib.sha256(canonical(block["input"]).encode()).hexdigest()})
                    yield CheckpointResponse(block, _object(body).get("stream") is True)
                    self.registry.record_transport({**identity, "event": "finished", "route": route,
                        "origin": "quality_stop_controller", "upstream_bridge_attempted": False, "provider_attempted": False})
                    return
            else:
                with self.registry.condition:
                    agent = fields.get("x-claude-code-agent-id")
                    mirror = self.registry.mirrors.get(agent, {})
                    local = agent in self.registry.emissions or mirror.get("task", "").startswith(PREFIX)
                    require(not local, "checkpoint_endpoint_unsupported")
                    if agent:
                        require(fields.get("x-claude-code-session-id") == self.registry.session_id
                                and not fields.get("x-claude-code-parent-agent-id"), "native_request_identity_changed")
                        self.registry._wait_identity(agent)
                    else:
                        self.registry._bootstrap(method, target, headers, body, self.base_path)
                        bootstrap = True
                if body:
                    try:
                        decoded = _object(body)
                    except (StopRejected, ValueError, TypeError, UnicodeError):
                        decoded = None
                    require(not _contains_checkpoint(decoded), "checkpoint_endpoint_unsupported")
            # Retirement and forwarding must share one atomic admission lock.
            # A started old request blocks retirement, including this interval.
            with self.registry.condition:
                agent = fields.get("x-claude-code-agent-id")
                self.registry._guard_retired(agent, "denied_model", detail=identity)
                require(self.registry.error is None and not self.registry.closed, "native_census_unavailable")
                if agent:
                    self.registry._wait_identity(agent)
                if bootstrap:
                    self.registry._bootstrap(method, target, headers, body, self.base_path)
                    self.registry.bootstrap_admitted = True
                self.registry.record_transport({**identity, "event": "forwarding", "route": "provider_recorder",
                    "upstream_bridge_attempted": True, "provider_attempted": None})
                route, upstream_attempted = "provider_recorder", True
            forwarded_headers = [(name, value) for name, value in headers if name.lower() != "x-geak-quality-request-id"]
            forwarded_headers.append(("x-geak-quality-request-id", request_id))
            with self.upstream.send(method, target, forwarded_headers, body) as response:
                yield response
            self.registry.record_transport({**identity, "event": "finished", "route": route,
                "upstream_bridge_attempted": True, "provider_attempted": None})
        except BaseException as error:
            reason = str(error) if isinstance(error, StopRejected) else "bridge_transport_error"
            if re.fullmatch(r"[a-z0-9_]{1,128}", reason) is None:
                reason = "bridge_transport_error"
            self.registry.record_transport({**identity, "event": "failed", "route": route,
                "upstream_bridge_attempted": upstream_attempted, "provider_attempted": None if upstream_attempted else False,
                "reason": reason})
            raise

    def close(self):
        self.registry.close()
        self.upstream.close()


class CheckpointResponse:
    """Label a signed controller reply separately from administrative helpers."""

    status, reason = 200, "OK"

    def __init__(self, block, stream):
        usage = {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
        message = {"id": block["id"], "type": "message", "role": "assistant", "model": "local-quality-stop-controller-v2",
                   "content": [block], "stop_reason": "tool_use", "stop_sequence": None, "usage": usage}
        if stream:
            events = [{"type": "message_start", "message": {**message, "content": [], "stop_reason": None}},
                      {"type": "content_block_start", "index": 0, "content_block": {**block, "input": {}}},
                      {"type": "content_block_delta", "index": 0,
                       "delta": {"type": "input_json_delta", "partial_json": canonical(block["input"])}},
                      {"type": "content_block_stop", "index": 0},
                      {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                       "usage": {"output_tokens": 0}}, {"type": "message_stop"}]
            self.body = b"".join(("event: " + event["type"] + "\ndata: " + canonical(event) + "\n\n").encode() for event in events)
        else:
            self.body = canonical(message).encode()
        self.headers = [("Content-Type", "text/event-stream" if stream else "application/json"),
                        ("Content-Length", str(len(self.body))), ("request-id", block["id"]),
                        ("x-geak-local-origin", "quality_stop_controller")]

    def iter_bytes(self):
        yield self.body
