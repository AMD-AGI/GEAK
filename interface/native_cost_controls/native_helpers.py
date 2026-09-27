# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Join native Workflow identities before emitting a local helper tool call.

The registry receives identities from SDK events, session mirrors, and hooks.
An HTTP request cannot register a task. The native CLI still executes Bash and
validates StructuredOutput through its ordinary permission and tool paths.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from urllib.parse import urlsplit

from .helper_driver import (
    LOCAL_MODEL,
    LocalHelperDriver,
    Unsupported,
    canonical,
    check,
    digest,
    json_values_equal,
)
from .native_journal import NativeJournal


def normalized(label):
    return re.sub(r" (?:\(retry [0-9]+\)|\(throttle-retry\))$", "", label)


def role_slot(label, prior):
    label = normalized(label)
    fixed = {"warm_start:resolve": ("warm_start_resolver", "initial"),
        "kb:cite": ("citation_writer", "final"), "kb:write": ("experience_writer", "final")}
    if label in fixed:
        return fixed[label]
    match = re.fullmatch(r"storage:reclaim r([1-9][0-9]*)", label)
    if match:
        return "storage_reclaim", "round_" + match[1]
    match = re.fullmatch(r"clock pre-r([1-9][0-9]*)", label)
    if match:
        return "clock_reader", "pre_" + match[1]
    match = re.fullmatch(r"clock replan-r([1-9][0-9]*)", label)
    if match:
        anchors = [(node["index"], normalized(node["label"])) for node in prior
            if re.fullmatch(r"tech_lead:replan r" + match[1] + r"#[1-9][0-9]*", normalized(node["label"]))]
        check(bool(anchors), "replan_anchor_missing")
        return "clock_reader", max(anchors)[1]
    return None, None


def _object(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key.")
            result[key] = value
        return result

    value = json.loads(raw, object_pairs_hook=pairs)
    if not isinstance(value, dict):
        raise TypeError("A JSON object is required.")
    canonical(value)
    return value


class NativeHelperRegistry:
    """Bind one fresh native kernel workflow to a public source contract."""

    def __init__(self, directory, session_id, workspace, contract, *, wait_seconds=1.0, native_shell=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, mode=0o700)
        check(not any(self.directory.iterdir()), "fresh_helper_directory_required")
        self.workspace = Path(workspace).resolve()
        self.session_id = session_id
        self.native_shell = native_shell or os.environ.get("SHELL") or "unknown"
        self.contract = contract
        contract.verify_sources()
        self.root_tool = None
        self.root_task = None
        self.nodes = {}
        self.mirrors = {}
        self.dispatches = {}
        self.operations = {}
        self.local_ids = {}
        self.journal = None
        self.journal_descriptor = None
        self.native_terminals = {}
        self.finalized_agents = {}
        self.errors = []
        self.wait_seconds = wait_seconds
        self.condition = threading.Condition(threading.RLock())
        self.closed = False

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()

    def pin_root(self, data, tool_id):
        with self.condition:
            check(data.get("session_id") == self.session_id and not data.get("agent_id"), "root_hook_identity")
            check(data.get("tool_name") == "Workflow" and bool(tool_id), "root_hook_tool")
            check(data.get("tool_input") == self.contract.root_request, "root_request_changed")
            check(self.root_tool in (None, tool_id), "second_root_workflow")
            self.contract.verify_sources()
            self.root_tool = tool_id
            self.condition.notify_all()

    def observe(self, message):
        kind = type(message).__name__
        if kind not in {"UserMessage", "TaskStartedMessage", "TaskProgressMessage", "TaskNotificationMessage", "TaskUpdatedMessage"}:
            return
        value = asdict(message) if is_dataclass(message) else vars(message)
        if kind == "UserMessage":
            self._observe_root_result(value)
            return
        data = value.get("data", {})
        if kind in {"TaskNotificationMessage", "TaskUpdatedMessage"}:
            if (value.get("task_id", data.get("task_id")) == self.root_task
                    and value.get("session_id", data.get("session_id")) == self.session_id):
                self.refresh_native_completions()
            return
        tool = value.get("tool_use_id", data.get("tool_use_id"))
        if not self.root_tool or tool != self.root_tool:
            return
        with self.condition:
            try:
                check(value.get("session_id", data.get("session_id")) == self.session_id, "event_session_mismatch")
                task = value.get("task_id", data.get("task_id"))
                check(isinstance(task, str) and bool(task), "event_task_missing")
                if kind == "TaskStartedMessage":
                    check(value.get("task_type", data.get("task_type")) == "local_workflow", "root_task_type")
                    check(self.root_task in (None, task), "root_task_conflict")
                    self.root_task = task
                elif "workflow_progress" in data:
                    check(task == self.root_task, "progress_task_mismatch")
                    progress = data["workflow_progress"]
                    check(isinstance(progress, list) and all(isinstance(n, dict) for n in progress), "progress_shape")
                    phases = {n.get("index"): n.get("title") for n in progress if n.get("type") == "workflow_phase"}
                    for node in progress:
                        if node.get("type") != "workflow_agent":
                            continue
                        # The first supported contract has one nested kernel lane.
                        if phases.get(node.get("phaseIndex")) != "▸ kernel-lane":
                            continue
                        agent = node.get("agentId")
                        if not agent and node.get("state") == "queued":
                            continue
                        check(isinstance(agent, str) and bool(agent), "node_agent_missing")
                        check(type(node.get("index")) is int and node["index"] > 0, "node_index_invalid")
                        check(type(node.get("attempt")) is int and node["attempt"] > 0, "node_attempt_invalid")
                        check(isinstance(node.get("label"), str), "node_label_invalid")
                        check(node.get("state") != "cached" and not node.get("cached"), "cached_workflow_unsupported")
                        identity = {key: node[key] for key in ("index", "label", "phaseIndex", "agentId", "attempt")}
                        check(agent not in self.nodes or self.nodes[agent] == identity, "agent_node_conflict")
                        previous = [item for item in self.nodes.values() if item["index"] == node["index"]
                            and item["phaseIndex"] == node["phaseIndex"] and item["agentId"] != agent]
                        for item in previous:
                            check(node["attempt"] > item["attempt"]
                                and normalized(node["label"]) == normalized(item["label"]), "native_retry_conflict")
                        self.nodes[agent] = identity
                        if node.get("state") in {"done", "error", "failed", "stopped", "cancelled"}:
                            terminal = {key: node.get(key) for key in ("state", "lastToolName", "resultPreview")}
                            if self.native_terminals.get(agent) != terminal:
                                if agent in self.finalized_agents:
                                    driver = self._driver(agent)
                                    if driver is not None:
                                        driver.reject_completion("native_terminal_evidence_changed")
                                self.finalized_agents.pop(agent, None)
                            self.native_terminals[agent] = terminal
            except (Unsupported, OSError, ValueError, TypeError, KeyError, AttributeError):
                self.errors.append("workflow_identity_conflict")
            self._refresh_native_completions()
            self.condition.notify_all()

    def _observe_root_result(self, value):
        with self.condition:
            content = value.get("content", [])
            if not self.root_tool or not isinstance(content, list):
                return
            blocks = [block for block in content if isinstance(block, dict) and block.get("tool_use_id") == self.root_tool]
            if not blocks or not self.root_task:
                return
            try:
                result = value.get("tool_use_result")
                check(len(blocks) == 1 and blocks[0].get("is_error") is False
                    and value.get("parent_tool_use_id") is None, "root_result_identity")
                check(isinstance(result, dict) and result.get("status") == "async_launched"
                    and result.get("taskType") == "local_workflow" and result.get("taskId") == self.root_task
                    and result.get("scriptPath") == self.contract.root_request.get("scriptPath"), "root_result_binding")
                directory = Path(result["transcriptDir"])
                check(directory.is_absolute() and ".." not in directory.parts and len(directory.parts) >= 6
                    and directory.parts[-6] == "projects", "root_transcript_directory")
                descriptor = {"directory": str(directory), "run_id": result["runId"],
                    "project_key": directory.parts[-5]}
                check(self.journal_descriptor in (None, descriptor), "root_journal_changed")
                if self.journal is None:
                    self.journal = NativeJournal(directory, session_id=self.session_id, run_id=result["runId"])
                    self.journal_descriptor = descriptor
                self._refresh_native_completions()
            except (Unsupported, OSError, ValueError, TypeError, KeyError, AttributeError):
                self.errors.append("root_journal_identity_conflict")
            self.condition.notify_all()

    def refresh_native_completions(self):
        with self.condition:
            self._refresh_native_completions()

    def _refresh_native_completions(self):
        try:
            self._read_native_completions()
        except (Unsupported, OSError, ValueError, TypeError, KeyError, AttributeError):
            if "native_completion_unavailable" not in self.errors:
                self.errors.append("native_completion_unavailable")

    def _read_native_completions(self):
        if self.journal is None:
            return
        snapshot = self.journal.snapshot()
        if snapshot["status"] == "error":
            for driver in self.operations.values():
                if driver.path.exists():
                    driver.reject_completion("native_journal_changed")
            if "native_journal_changed" not in self.errors:
                self.errors.append("native_journal_changed")
            return
        for agent, terminal in self.native_terminals.items():
            driver = self._driver(agent)
            if driver is None or not driver.path.exists():
                continue
            if terminal["state"] != "done":
                driver.reject_completion("native_child_failed")
                continue
            if agent in self.finalized_agents or snapshot["status"] != "complete":
                continue
            outcome = self.journal.result_for(agent)
            if outcome["status"] == "pending":
                continue
            if outcome["status"] == "error":
                driver.reject_completion("native_child_journal_error")
                continue
            raw_value = outcome.get("result_json")
            if not isinstance(raw_value, str) or not isinstance(terminal.get("resultPreview"), str):
                continue
            units = raw_value.encode("utf-16-le", errors="surrogatepass")
            preview = (units[:800].decode("utf-16-le", errors="surrogatepass") + "…") if len(units) > 800 else raw_value
            with driver.state() as state:
                if state["stage"] in {"NEW", "MAY_HAVE_EXECUTED", "RESULT", "STRUCTURED_PENDING"}:
                    continue
                valid = (terminal.get("lastToolName") == "StructuredOutput"
                    and terminal["resultPreview"] == preview and json_values_equal(outcome["value"], state.get("value")))
            if not valid:
                driver.reject_completion("native_terminal_outcome_changed")
                continue
            evidence = {"source": "native_workflow_journal", "status": "done", "native_session": self.session_id,
                "operation_id": driver.approved["operation_id"], "agent_id": agent,
                "root_task": self.root_task, "root_tool": self.root_tool,
                "run_id": self.journal_descriptor["run_id"], "key": outcome["key"],
                "entry_sha256": outcome["entry_sha256"], "started_entry_sha256": outcome["started_entry_sha256"]}
            try:
                driver.confirm_completion(outcome["value"], evidence)
            except (Unsupported, OSError, ValueError, TypeError, KeyError):
                if "native_terminal_outcome_changed" not in self.errors:
                    self.errors.append("native_terminal_outcome_changed")
            else:
                self.finalized_agents[agent] = outcome["entry_sha256"]

    def mirror_append(self, key, entries):
        with self.condition:
            try:
                if key.get("session_id") != self.session_id or not key.get("subpath"):
                    return
                match = re.fullmatch(r"subagents/workflows/([^/]+)/agent-([^/]+)", key["subpath"])
                if match is None:
                    return
                agent = match[2]
                for entry in entries:
                    if entry.get("type") != "user" or entry.get("parentUuid") is not None:
                        continue
                    check(entry.get("agentId", agent) == agent, "mirror_agent_mismatch")
                    check(entry.get("sessionId", self.session_id) == self.session_id, "mirror_session_mismatch")
                    check(entry.get("cwd") == str(self.workspace), "mirror_workspace_mismatch")
                    task = entry.get("message", {}).get("content")
                    check(isinstance(task, str) and bool(task), "mirror_initial_task_invalid")
                    record = {"task": task, "uuid": entry.get("uuid"), "run_id": match[1], "project_key": key.get("project_key")}
                    check(agent not in self.mirrors or self.mirrors[agent] == record, "mirror_initial_task_conflict")
                    self.mirrors[agent] = record
            except (Unsupported, ValueError, TypeError, KeyError, AttributeError):
                self.errors.append("mirror_identity_conflict")
            self.condition.notify_all()

    def _driver(self, agent):
        dispatch = self.dispatches.get(agent, {})
        return self.operations.get(dispatch.get("operation_id")) if dispatch.get("ownership") == "local" else None

    def _headers(self, headers):
        fields = {}
        for key, value in headers:
            key = key.lower()
            if key.startswith("x-claude-code-"):
                check(key not in fields, "duplicate_identity_header")
                fields[key] = value
        return fields

    def response(self, headers, raw):
        """Return a local block, or None for exact provider passthrough."""
        with self.condition:
            self._refresh_native_completions()
            try:
                fields = self._headers(headers)
            except Unsupported:
                check(not self.operations, "local_identity_changed", True)
                return None
            agent = fields.get("x-claude-code-agent-id")
            driver = self._driver(agent)
            identity_matches = (fields.get("x-claude-code-session-id") == self.session_id
                and not fields.get("x-claude-code-parent-agent-id"))
            if self.operations:
                try:
                    decoded = _object(raw)
                except (ValueError, TypeError, RecursionError, UnicodeError):
                    raise Unsupported("local_continuation_unreadable", True) from None
                references = self._local_references(decoded)
                if references:
                    candidate = driver or self._operation_driver(agent)
                    check(identity_matches and candidate is not None
                        and references == {candidate.approved["operation_id"]}, "local_identity_changed", True)
                if not identity_matches or not agent:
                    check(not any(isinstance(tool, dict) and tool.get("name") == "StructuredOutput"
                        for tool in decoded.get("tools", [])), "local_child_identity_missing", True)
            if not identity_matches:
                check(driver is None, "local_identity_changed", True)
                return None
            if not agent:
                return None
            previous = self.dispatches.get(agent, {})
            if previous.get("ownership") == "provider":
                return None
            try:
                body = _object(raw)
                deadline = time.monotonic() + self.wait_seconds
                while not (self.root_task and agent in self.nodes and agent in self.mirrors):
                    check(not self.closed, "helper_registry_closed", driver is not None)
                    remaining = deadline - time.monotonic()
                    check(remaining > 0, "helper_identity_unavailable", driver is not None)
                    self.condition.wait(remaining)
                check(not self.errors, "helper_identity_conflict", driver is not None)
                check(not self.closed, "helper_registry_closed", driver is not None)
                self.contract.verify_sources()
                if driver is None:
                    driver = self._admit(agent, body)
                if driver is None:
                    return None
                # A completed CLI child cannot start a new turn. Native retry
                # aliases need their own observed identity and terminal result.
                check(agent not in self.native_terminals, "completed_native_agent_reused", True)
                check(any(t.get("name") == "Bash" for t in body.get("tools", [])), "native_bash_not_advertised")
                block = driver.response(body, driver.approved)
                self.local_ids[block["id"]] = driver.approved["operation_id"]
                return block
            except (Unsupported, OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError) as error:
                failure = error if isinstance(error, Unsupported) else Unsupported("unsupported_native_request", driver is not None)
                if driver is not None:
                    fallback = driver.fallback_decision(raw, failure)
                    check(fallback["caller_must_forward_unchanged"], failure.code, True)
                else:
                    # A native retry can receive a new agent ID for an old
                    # operation. A malformed retry must not reopen inference.
                    prior_driver = self._operation_driver(agent)
                    if prior_driver is not None:
                        fallback = prior_driver.fallback_decision(raw, failure)
                        check(fallback["caller_must_forward_unchanged"], failure.code, True)
                    if self.errors:
                        for operation_driver in self.operations.values():
                            fallback = operation_driver.fallback_decision(raw, failure)
                            check(fallback["caller_must_forward_unchanged"], failure.code, True)
                    check(not failure.may_have_executed, failure.code, True)
                self.dispatches[agent] = {**previous, "ownership": "provider"}
                return None

    def _local_references(self, value):
        pending, operations = [value], set()
        while pending:
            item = pending.pop()
            if isinstance(item, dict):
                pending.extend(item.values())
            elif isinstance(item, list):
                pending.extend(item)
            elif isinstance(item, str) and item in self.local_ids:
                operations.add(self.local_ids[item])
        return operations

    def _operation_driver(self, agent):
        node = self.nodes.get(agent)
        if node is None:
            return None
        prior = [n for n in self.nodes.values() if n["index"] < node["index"]]
        role, slot = role_slot(node["label"], prior)
        operation = digest({"session": self.session_id, "root_tool": self.root_tool, "root_task": self.root_task,
            "lane": node["phaseIndex"], "role": role, "slot": slot})
        return self.operations.get(operation)

    def _admit(self, agent, body):
        node, mirror = self.nodes[agent], self.mirrors[agent]
        dispatch = {**node, "ownership": "provider"}
        previous = self._operation_driver(agent)
        self.contract.verify_sources()
        prior = [n for n in self.nodes.values() if n["index"] < node["index"]]
        role, slot = role_slot(node["label"], prior)
        if role is None:
            self.dispatches[agent] = dispatch
            return None
        check(self.journal_descriptor is not None and mirror["run_id"] == self.journal_descriptor["run_id"]
            and mirror["project_key"] == self.journal_descriptor["project_key"], "native_journal_unavailable")
        if normalized(node["label"]).startswith("clock replan-"):
            anchor = next(n for n in prior if normalized(n["label"]) == slot)
            check(self.dispatches.get(anchor["agentId"], {}).get("ownership") == "provider", "replan_anchor_unadmitted")
        prompt = mirror["task"]
        schemas = [t["input_schema"] for t in body.get("tools", []) if t.get("name") == "StructuredOutput"]
        check(len(schemas) == 1, "native_schema_missing")
        source = self.contract.resolve(normalized(node["label"]), prompt, schemas[0])
        if source.get("eligible") is not True or source.get("role") != role:
            if previous is not None:
                fallback = previous.fallback_decision(canonical(body), Unsupported("unsupported_retry_template"))
                check(fallback["caller_must_forward_unchanged"], "unsupported_retry_template", True)
            self.dispatches[agent] = dispatch
            return None
        operation = digest({"session": self.session_id, "root_tool": self.root_tool, "root_task": self.root_task,
            "lane": node["phaseIndex"], "role": role, "slot": slot})
        stat = self.workspace.stat()
        binding = {"operation_id": operation, "role": role, "gate": True,
            "command": source["command"], "command_sha256": hashlib.sha256(source["command"].encode()).hexdigest(),
            "schema": schemas[0], "schema_sha256": digest(schemas[0]), "prompt": prompt,
            "workspace": str(self.workspace), "workspace_identity": [stat.st_dev, stat.st_ino],
            "source_bindings": self.contract.source_bindings, "native_session": self.session_id,
            "native_shell": self.native_shell,
            "completion_marker": source.get("completion_marker"), "native_preamble": None}
        if operation not in self.operations:
            self.operations[operation] = LocalHelperDriver(self.directory / "ledger", binding)
            self.operations[operation].attest_dispatch(self.contract.root_request, self.contract.root_request)
        driver = self.operations[operation]
        check(driver.approved == binding, "retry_binding_changed", driver.path.exists())
        self.dispatches[agent] = {**dispatch, "ownership": "local", "operation_id": operation}
        return driver

    async def pre(self, data, tool_use_id=None, context=None):
        """Return no permission grant. Reject changed local helper inputs."""
        tool_use_id = tool_use_id or data.get("tool_use_id")
        with self.condition:
            driver = self._driver(data.get("agent_id"))
            try:
                if not data.get("agent_id") and data.get("tool_name") == "Workflow":
                    # A different root request keeps the original provider path.
                    if data.get("tool_input") == self.contract.root_request:
                        self.pin_root(data, tool_use_id)
                    return {}
                if driver is None:
                    return {}
                check(data.get("session_id") == self.session_id, "hook_session_mismatch", True)
                check(data.get("cwd") == str(self.workspace), "hook_workspace_mismatch", True)
                check(not self.errors and not self.closed, "helper_registry_unavailable", True)
                self.contract.verify_sources()
                if data.get("tool_name") == "Bash":
                    driver.before_bash(tool_use_id, data.get("tool_input"))
                else:
                    check(data.get("tool_name") == "StructuredOutput", "local_helper_tool_forbidden", True)
                return {}
            except (Unsupported, OSError, ValueError, TypeError, KeyError, AttributeError):
                return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                    "permissionDecisionReason": "The local helper input differs from its source binding."}}

    async def post(self, data, tool_use_id=None, context=None):
        tool_use_id = tool_use_id or data.get("tool_use_id")
        with self.condition:
            driver = self._driver(data.get("agent_id"))
            dispatch = self.dispatches.get(data.get("agent_id"), {})
            failed = data.get("hook_event_name") == "PostToolUseFailure"
            try:
                if (not failed and data.get("tool_name") == "StructuredOutput"
                        and normalized(dispatch.get("label", "")) == "director:setup"):
                    check(data.get("session_id") == self.session_id, "setup_session_mismatch")
                    check(data.get("cwd") == str(self.workspace), "setup_workspace_mismatch")
                    self.contract.bind_setup(data.get("tool_input"), {"agent_id": data.get("agent_id"),
                        "tool_use_id": tool_use_id, "root_tool": self.root_tool, "root_task": self.root_task})
                if driver is None:
                    return {}
                check(data.get("session_id") == self.session_id, "hook_session_mismatch", True)
                check(data.get("cwd") == str(self.workspace), "hook_workspace_mismatch", True)
                if data.get("tool_name") == "Bash":
                    check(data.get("tool_input") == driver.bash_input(), "native_bash_input_changed", True)
                    driver.capture_bash(tool_use_id, data.get("tool_response", {"error": data.get("error")}), failed)
                elif data.get("tool_name") == "StructuredOutput":
                    if failed:
                        driver.reject_structured(tool_use_id, "native_schema_rejected")
                    else:
                        driver.accept(tool_use_id, data.get("tool_input"))
                        self._refresh_native_completions()
            except (Unsupported, OSError, ValueError, TypeError, KeyError, AttributeError):
                self.errors.append("native_helper_outcome_conflict")
            return {}


class HelperMirror:
    """Preserve an existing mirror and observe only initial Workflow tasks."""

    def __init__(self, registry, original=None):
        self.registry, self.original = registry, original

    async def append(self, key, entries):
        if self.original is not None:
            await self.original.append(key, entries)
        self.registry.mirror_append(key, entries)

    async def load(self, key):
        return await self.original.load(key) if self.original is not None else None

    def __getattr__(self, name):
        if self.original is None:
            raise AttributeError(name)
        return getattr(self.original, name)


class _LocalResponse:
    status = 200
    reason = "OK"

    def __init__(self, block, stream):
        usage = {"input_tokens": 0, "output_tokens": 0,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
        message = {"id": block["id"], "type": "message", "role": "assistant", "model": LOCAL_MODEL,
            "content": [block], "stop_reason": "tool_use", "stop_sequence": None, "usage": usage}
        self.headers = [("Content-Type", "text/event-stream" if stream else "application/json"),
            ("request-id", block["id"]), ("x-geak-local-origin", "local_deterministic_driver")]
        if stream:
            events = [
                {"type": "message_start", "message": {**message, "content": [], "stop_reason": None}},
                {"type": "content_block_start", "index": 0, "content_block": {**block, "input": {}}},
                {"type": "content_block_delta", "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": canonical(block["input"]).decode()}},
                {"type": "content_block_stop", "index": 0},
                {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                    "usage": {"output_tokens": 0}},
                {"type": "message_stop"},
            ]
            self.body = b"".join(b"event: " + event["type"].encode() + b"\ndata: " + canonical(event) + b"\n\n" for event in events)
        else:
            self.body = canonical(message)
        self.headers.append(("Content-Length", str(len(self.body))))

    def iter_bytes(self):
        yield self.body


class HelperTransport:
    """Use one upstream send for fallback and no upstream send for a helper."""

    def __init__(self, upstream, registry, *, base_path=""):
        self.upstream, self.registry, self.base_path = upstream, registry, base_path.rstrip("/")

    @contextmanager
    def send(self, method, target, headers, body):
        fields = {name.lower(): value for name, value in headers}
        supported = (method == "POST" and urlsplit(target).path == self.base_path + "/v1/messages"
            and fields.get("content-encoding", "identity").lower() == "identity"
            and fields.get("content-type", "").split(";", 1)[0].strip().lower() == "application/json")
        if method == "POST" and urlsplit(target).path == self.base_path + "/v1/messages" and not supported:
            check(not self.registry.operations, "local_request_encoding_changed", True)
        block = self.registry.response(headers, body) if supported else None
        if block is not None:
            yield _LocalResponse(block, _object(body).get("stream") is True)
        else:
            with self.upstream.send(method, target, headers, body) as response:
                yield response

    def close(self):
        self.registry.close()
        self.upstream.close()
