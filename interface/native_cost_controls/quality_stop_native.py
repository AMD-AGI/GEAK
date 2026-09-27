# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bind stopping calls to the complete native producer census.

This registry is independent of NativeHelperRegistry. Helper and cache controls
remain disabled. Native journal parsing, native terminal state, tool activity,
background task state, and the host process boundary must all agree.
"""

from __future__ import annotations

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
from .native_journal import NativeJournal
from .quality_stop_controller import (
    PREFIX,
    SCHEMA,
    StopRejected,
    canonical,
    parse_task,
    require,
)


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


def checkpoint_initial_messages(messages, task):
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
        blocks = notice.get("content") if isinstance(notice, dict) and notice.get("role") == "system" else None
        if isinstance(blocks, list) and len(blocks) == 1 and isinstance(blocks[0], dict):
            block = blocks[0]
            if (set(block) == {"type", "text", "cache_control"} and block.get("type") == "text"
                    and qualified_notice_text(block.get("text")) and block["cache_control"] == {"type": "ephemeral"}):
                del block["cache_control"]
    return qualified_initial_messages(view, task)


class NativeProducerCensus:
    """Join every native producer, including queued nodes without agent IDs."""

    def __init__(self, *, session_id, root_request, source_root, workspace, controller, wait_seconds=10):
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
                         self.source_root / "quality_stop_verify.js", Path(__file__).resolve())}
        self.root_tool = self.root_task = self.journal = self.journal_descriptor = None
        self.nodes, self.mirrors, self.tools, self.background, self.emissions = {}, {}, {}, {}, {}
        self.mirror_inputs = {}
        self.completed_tools = {}
        self.last_progress = None
        self.checkpoint_active = None
        self.error = None
        self.rejections = []
        self.request_records = []
        self.transport_sealed = False
        self.closed = False
        self.wait_seconds = wait_seconds
        self.condition = threading.Condition(threading.RLock())

    def fail(self, reason):
        self.error = self.error or reason
        self.condition.notify_all()

    def verify_sources(self):
        for path, digest in self.sources.items():
            require(path.resolve() == path and hashlib.sha256(path.read_bytes()).hexdigest() == digest,
                    "stopping_workflow_source_changed")

    def close(self):
        with self.condition:
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
                except (OSError, StopRejected):
                    self.fail("bridge_ledger_unavailable")
                    raise StopRejected("bridge_ledger_unavailable") from None
            self.request_records.append(deepcopy(record))
            self.condition.notify_all()

    def pin_root(self, data, tool_id):
        require(data.get("session_id") == self.session_id and not data.get("agent_id")
                and data.get("tool_name") == "Workflow" and tool_id, "stopping_root_hook_identity")
        require(json_values_equal(data.get("tool_input"), self.root_request) and self.root_tool in (None, tool_id), "stopping_root_request_changed")
        self.verify_sources()
        self.root_tool = tool_id

    def observe(self, message):
        kind = type(message).__name__
        value = asdict(message) if is_dataclass(message) else vars(message)
        data = value.get("data", {})
        with self.condition:
            try:
                if kind == "UserMessage":
                    self._root_result(value)
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
                            require(task not in self.background, "duplicate_background_task_start")
                            self.background[task] = "active"
                    elif kind in {"TaskNotificationMessage", "TaskUpdatedMessage"} and task in self.background:
                        status = value.get("status", data.get("status"))
                        if status in {"completed", "failed", "stopped", "cancelled"}:
                            self.background[task] = status
                    if "workflow_progress" in data:
                        require(task == self.root_task and tool == self.root_tool, "native_progress_root_changed")
                        self._progress(data["workflow_progress"])
            except (StopRejected, OSError, ValueError, TypeError, KeyError, AttributeError):
                self.fail("native_census_event_conflict")
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
        for node in progress:
            if node.get("type") != "workflow_agent":
                continue
            require(type(node.get("index")) is int and type(node.get("attempt")) is int
                    and node["index"] > 0 and node["attempt"] > 0 and isinstance(node.get("label"), str),
                    "native_node_identity_invalid")
            require(not node.get("cached") and node.get("state") != "cached", "cached_producer_unsupported")
            key = (node.get("phaseIndex"), node["index"], node["attempt"])
            require(key not in current, "duplicate_native_node")
            if key in self.nodes:
                old = self.nodes[key]
                require(old.get("label") == node["label"] and old.get("agentId") in (None, node.get("agentId")),
                        "native_node_changed")
                if old.get("state") in {"done", "error", "failed", "stopped", "cancelled"}:
                    terminal_fields = ("state", "agentId", "label", "index", "attempt", "phaseIndex", "lastToolName", "resultPreview")
                    require(all(old.get(field) == node.get(field) for field in terminal_fields), "native_terminal_node_changed")
            current[key] = deepcopy(node)
        if self.checkpoint_active is not None:
            require(set(current) == set(self.nodes), "native_admission_during_checkpoint")
        require(set(self.nodes) <= set(current), "native_producer_disappeared")
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
            except (StopRejected, ValueError, TypeError, KeyError, AttributeError):
                self.fail("native_census_mirror_conflict")
            self.condition.notify_all()

    def _node(self, agent):
        nodes = [node for node in self.nodes.values() if node.get("agentId") == agent]
        require(len(nodes) == 1, "native_agent_node_missing_or_duplicate")
        return nodes[0]

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
            snapshot = self.journal.snapshot()
            require(snapshot["status"] == "complete", "native_closure_journal_unstable")
            agents = {node.get("agentId"): node for node in self.nodes.values() if node.get("agentId")}
            require(len(agents) == len(self.nodes) and set(agents) == set(snapshot["started_agents"])
                    and set(agents) == set(snapshot["results"]), "native_closure_sets_differ")
            require(set(agents) == set(self.mirrors) == set(self.mirror_inputs), "native_closure_mirrors_differ")
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
            self.transport_sealed = True
            return {"schema": "geak-quality-native-census-v1", "session_id": self.session_id,
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

    def _bridge_all_terminal(self):
        groups = {}
        for record in self.request_records:
            key = record.get("request_id")
            require(isinstance(key, str) and key, "native_closure_bridge_sequence_invalid")
            groups.setdefault(key, []).append(record)
        for records in groups.values():
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
        return True

    def _assert_quiescent(self, checkpoint_agent):
        require(not self.closed and self.error is None and self.journal is not None, "native_census_unavailable")
        self.verify_sources()
        active = self._node(checkpoint_agent)
        require(active.get("state") not in {"done", "error", "failed", "stopped", "cancelled"}, "checkpoint_already_terminal")
        require(not self.tools and all(status != "active" for status in self.background.values()), "native_background_producer_active")
        snapshot = self.journal.snapshot()
        require(snapshot["status"] == "complete", "native_journal_not_stable")
        started, results = set(snapshot["started_agents"]), snapshot["results"]
        nodes = {node.get("agentId"): node for node in self.nodes.values() if node.get("agentId")}
        require(len(nodes) == len(self.nodes), "native_queued_producer_unknown")
        require(set(nodes) == started, "native_producer_sets_differ")
        require(checkpoint_agent in started and checkpoint_agent not in results, "checkpoint_journal_state_invalid")
        require(set(results) == started - {checkpoint_agent}, "native_producer_result_missing")
        for agent in started - {checkpoint_agent}:
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
            require([node["label"] for agent, node in nodes.items() if agent != checkpoint_agent] == ["director:setup"],
                    "seed_handoff_after_optimizer_work")
            return
        reclaim = [node for node in nodes.values() if node["label"] == "storage:reclaim r" + str(request["round"])]
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

    def _wait_identity(self, agent):
        deadline = time.monotonic() + self.wait_seconds
        while not (self.root_task and self.journal and agent in self.mirrors
                   and any(node.get("agentId") == agent for node in self.nodes.values())):
            require(not self.closed and self.error is None, "native_identity_unavailable")
            remaining = deadline - time.monotonic()
            require(remaining > 0, "native_identity_timeout")
            self.condition.wait(remaining)

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
            if not agent:
                # Missing child headers must never route a checkpoint to inference.
                require(not _contains_checkpoint(body), "checkpoint_identity_missing")
                return None
            require(fields.get("x-claude-code-session-id") == self.session_id
                    and not fields.get("x-claude-code-parent-agent-id"), "native_request_identity_changed")
            self._wait_identity(agent)
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

    async def pre(self, data, tool_use_id=None, context=None):
        tool_id = tool_use_id or data.get("tool_use_id")
        with self.condition:
            try:
                require(data.get("session_id") == self.session_id and data.get("cwd") == str(self.workspace),
                        "native_tool_hook_identity_changed")
                name, agent = data.get("tool_name"), data.get("agent_id")
                require(self.checkpoint_active in (None, agent), "tool_admitted_during_checkpoint")
                if name == "Workflow":
                    self.pin_root(data, tool_id)
                elif agent in self.emissions:
                    block = self.emissions[agent]["block"]
                    require(name == "StructuredOutput" and tool_id == block["id"] and json_values_equal(data.get("tool_input"), block["input"]),
                            "checkpoint_structured_output_changed")
                    require(not self.emissions[agent].get("started") and not self.emissions[agent].get("accepted"),
                            "duplicate_checkpoint_tool_execution")
                    self.emissions[agent]["started"] = True
                elif name not in {"TaskOutput", "TaskList", "TaskGet"}:
                    require(tool_id and tool_id not in self.tools and tool_id not in self.completed_tools,
                            "duplicate_native_tool_start")
                    self.tools[tool_id] = {"name": name, "agent": agent, "input": deepcopy(data.get("tool_input"))}
                return {}
            except (StopRejected, OSError, ValueError, TypeError, KeyError, AttributeError):
                self.fail("native_tool_admission_failed")
                return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                        "permissionDecisionReason": "The stopping controller rejected the native tool binding."}}

    async def post(self, data, tool_use_id=None, context=None):
        tool_id = tool_use_id or data.get("tool_use_id")
        with self.condition:
            try:
                require(data.get("session_id") == self.session_id and data.get("cwd") == str(self.workspace),
                        "native_tool_hook_identity_changed")
                agent, name = data.get("agent_id"), data.get("tool_name")
                if agent in self.emissions:
                    emitted = self.emissions[agent]
                    require(name == "StructuredOutput" and tool_id == emitted["block"]["id"]
                            and json_values_equal(data.get("tool_input"), emitted["block"]["input"])
                            and data.get("hook_event_name") != "PostToolUseFailure"
                            and emitted.get("started") is True and not emitted.get("accepted"), "checkpoint_result_rejected")
                    emitted["accepted"] = True
                    self.checkpoint_active = None
                elif name not in {"Workflow", "TaskOutput", "TaskList", "TaskGet"}:
                    require(tool_id in self.tools and json_values_equal(self.tools[tool_id],
                            {"name": name, "agent": agent, "input": data.get("tool_input")}), "native_tool_result_changed")
                    operation = self.tools.pop(tool_id)
                    response = data.get("tool_response")
                    self.completed_tools[tool_id] = {**operation, "response": deepcopy(response),
                                                    "failed": data.get("hook_event_name") == "PostToolUseFailure"}
                    if name == "Bash":
                        require(not (operation["input"] or {}).get("run_in_background"), "background_bash_requires_qualified_adapter")
                        if isinstance(response, dict):
                            require(not any(response.get(key) for key in ("task_id", "taskId", "backgroundTaskId", "background_task_id")),
                                    "background_bash_result_unsupported")
            except (StopRejected, OSError, ValueError, TypeError, KeyError, AttributeError):
                self.fail("native_tool_outcome_unknown")
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
        try:
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
                if body:
                    try:
                        decoded = _object(body)
                    except (StopRejected, ValueError, TypeError, UnicodeError):
                        decoded = None
                    require(not _contains_checkpoint(decoded), "checkpoint_endpoint_unsupported")
            route, upstream_attempted = "provider_recorder", True
            self.registry.record_transport({**identity, "event": "forwarding", "route": route,
                "upstream_bridge_attempted": True, "provider_attempted": None})
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
