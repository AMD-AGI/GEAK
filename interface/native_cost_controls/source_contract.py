# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bind exact administrative helper tasks to one installed kernel workflow.

The caller supplies trusted public workflow source and native Setup evidence.
A small Node renderer evaluates selected declarations and task expressions.
It runs no workflow, helper command, provider request, or GPU operation.
The native registry retains identity checks, permissions, and tool execution.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path, PurePosixPath

from .helper_driver import Unsupported, canonical, check, file_digest, json_values_equal

_RENDERER = Path(__file__).with_name("source_templates.cjs")
_JSON_STRING = r'"(?:[^"\\\x00-\x1f]|\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4}))*"'
_NUMBER = r'-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?'
_CITATION_FIELDS = ("card", "round", "direction", "specialty", "cited_then_verified", "became_winner")
_ROLES = {"clock": "clock_reader", "resolver": "warm_start_resolver", "storage": "storage_reclaim",
          "citation": "citation_writer", "writer": "experience_writer"}


def _path(value, parent=None):
    check(isinstance(value, str) and value.startswith("/") and len(value) <= 4096, "unsupported_helper_path")
    check(all(character.isalnum() or character in "/_.-" for character in value), "unsafe_helper_path")
    path = PurePosixPath(value)
    check(".." not in path.parts and str(path) == value and path != PurePosixPath("/"), "noncanonical_helper_path")
    if parent is not None:
        check(path != PurePosixPath(parent) and PurePosixPath(parent) in path.parents, "helper_path_outside_run")
    return value


def _text(value, limit=4096):
    check(isinstance(value, str) and len(value) <= limit, "invalid_helper_text")
    check(not any(character in value for character in '\x00\r\n$`\\"'), "unsafe_helper_text")
    return value


def _finite(value, *, positive=False):
    check(type(value) in (int, float) and math.isfinite(value), "invalid_helper_number")
    if positive:
        check(value > 0, "invalid_helper_number")
    return value


def _no_resume(value):
    if isinstance(value, dict):
        for key, item in value.items():
            check(isinstance(key, str), "invalid_root_json")
            if ("resume" in key.lower() or key in {"state_dir", "session_import", "continue_conversation", "fork_session"}) and item:
                raise Unsupported("resumed_workflow_unsupported")
            _no_resume(item)
    elif isinstance(value, list):
        for item in value:
            _no_resume(item)


class KernelWorkflowContract:
    """Support one fresh optimize or author lane from the public dispatcher.

    ``source_root`` contains kernel_workflow.js and kernel_lane.js.
    Node must be available when this opt-in contract is constructed.
    Source changes, resumes, alternate lanes, and unsafe shell paths remain
    unsupported. The caller can retain its original provider path in those cases.
    """

    def __init__(self, root_request, source_root):
        check(isinstance(root_request, dict) and set(root_request) == {"scriptPath", "args"}, "unsupported_root_request")
        check(isinstance(root_request["args"], dict), "unsupported_root_arguments")
        try:
            canonical(root_request)
        except (TypeError, ValueError, UnicodeError, RecursionError):
            raise Unsupported("invalid_root_json") from None
        _no_resume(root_request)
        self.source_root = Path(source_root).absolute()
        check(not any(path.is_symlink() for path in (self.source_root, *self.source_root.parents)), "workflow_source_symlink")
        _path(str(self.source_root))
        check(root_request["scriptPath"] == str(self.source_root / "kernel_workflow.js"), "unsupported_root_script")
        args = root_request["args"]
        check(str(args.get("workflow_dir", "")).rstrip("/") == str(self.source_root), "workflow_directory_mismatch")
        check(args.get("kernel_lane_script", str(self.source_root / "kernel_lane.js")) == str(self.source_root / "kernel_lane.js"),
              "alternate_lane_unsupported")
        self._root_request = deepcopy(root_request)
        self._source_bindings = {}
        paths = [self.source_root / name for name in ("kernel_workflow.js", "kernel_lane.js", "scripts/experience_store.py",
                                                     "scripts/kb.py", "scripts/reclaim_eval_artifacts.sh")]
        self._kb_root = self.source_root.parent / "kb"
        check((self._kb_root / "__init__.py").is_file(), "workflow_source_missing")
        self._kb_sources = {str(path) for path in self._kb_root.rglob("*.py")}
        paths += [Path(path) for path in sorted(self._kb_sources)]
        paths += [Path(__file__).resolve(), _RENDERER]
        try:
            for path in paths:
                check(not any(item.is_symlink() for item in (path, *path.parents)), "workflow_source_symlink")
                self._source_bindings[str(path)] = file_digest(path)
        except OSError:
            raise Unsupported("workflow_source_missing") from None
        self._node = shutil.which("node")
        check(self._node is not None, "node_renderer_unavailable")
        self._setup = None
        self.setup_attestations = []
        self._configuration = self._render("inspect")
        configuration = self._configuration
        check(configuration["mode"] in {"optimize", "author"}, "multiple_or_unknown_lanes_unsupported")
        check(type(configuration["budget"]) is int and 1 <= configuration["budget"] <= 1000000, "unsupported_direction_budget")
        for name in ("workflow_dir", "exp_root", "kernel_path", "kb_artifacts", "kb_store"):
            _path(configuration[name])
        _text(configuration["language"], 128)
        _text(configuration["dtype"], 128)
        _text(configuration["version"], 128)

    @property
    def root_request(self):
        return deepcopy(self._root_request)

    @property
    def source_bindings(self):
        return dict(self._source_bindings)

    def verify_sources(self):
        """Reject changed source bytes before a local helper receives ownership."""
        try:
            check({str(path) for path in self._kb_root.rglob("*.py")} == self._kb_sources, "workflow_source_changed")
            for name, expected in self._source_bindings.items():
                path = Path(name)
                check(not any(item.is_symlink() for item in (path, *path.parents)) and file_digest(path) == expected,
                      "workflow_source_changed")
        except OSError:
            raise Unsupported("workflow_source_changed") from None

    def _render(self, operation, site=None, dynamic=None):
        self.verify_sources()
        data = {"operation": operation, "source_root": str(self.source_root), "root_args": self._root_request["args"],
                "source_hashes": {name: self._source_bindings[str(self.source_root / name)]
                                  for name in ("kernel_workflow.js", "kernel_lane.js")},
                "setup": self._setup, "site": site, "dynamic": dynamic or {}}
        # Citation objects use the source's insertion order in JSON.stringify.
        payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
        check(len(payload) <= 1048576, "helper_render_input_too_large")
        try:
            completed = subprocess.run([self._node, str(_RENDERER)], input=payload, capture_output=True, timeout=3,
                env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}, check=False)
            check(completed.returncode == 0 and len(completed.stdout) <= 1048576, "source_template_layout_unsupported")
            result = json.loads(completed.stdout)
            canonical(result)
        except (OSError, subprocess.TimeoutExpired, ValueError, TypeError, UnicodeError, RecursionError):
            raise Unsupported("source_template_render_failed") from None
        self.verify_sources()
        return result

    def bind_setup(self, value, attestation):
        """Accept Setup data only from the caller's trusted native hook evidence."""
        self.verify_sources()
        check(isinstance(value, dict) and value.get("resumed", False) is False, "unsupported_setup_result")
        check(isinstance(attestation, dict) and all(isinstance(attestation.get(key), str) and attestation[key]
              for key in ("agent_id", "tool_use_id", "root_tool", "root_task")), "setup_attestation_missing")
        try:
            canonical(value)
        except (TypeError, ValueError, UnicodeError, RecursionError):
            raise Unsupported("invalid_setup_json") from None
        _no_resume(value)
        eval_dir = _path(value.get("eval_dir"), self._configuration["exp_root"])
        check(value.get("workspace") == eval_dir + "/workspace", "setup_workspace_mismatch")
        name = _text(value.get("kernel_name"), 128)
        check(bool(name), "setup_kernel_name_missing")
        override = self._root_request["args"].get("eval_dir")
        check(not override or override == eval_dir, "setup_eval_override_mismatch")
        if self._setup is not None:
            check(json_values_equal(self._setup, value), "setup_result_conflict")
            check(all(attestation[key] == self.setup_attestations[0][key] for key in ("root_tool", "root_task")),
                  "setup_root_attestation_conflict")
        self._setup = deepcopy(value)
        self.setup_attestations.append(deepcopy(attestation))

    def _site(self, label):
        fixed = {"warm_start:resolve": "resolver", "kb:cite": "citation", "kb:write": "writer"}
        if label in fixed:
            return fixed[label], {}
        match = re.fullmatch(r"clock (pre|replan)-r([1-9][0-9]*)", label)
        if match:
            return "clock", {"round": int(match[2]), "tag": label[6:]}
        match = re.fullmatch(r"storage:reclaim r([1-9][0-9]*)", label)
        if match:
            return "storage", {"round": int(match[1])}
        raise Unsupported("unsupported_helper_label")

    def _dynamic(self, site, task, initial):
        dynamic = dict(initial)
        if site in {"resolver", "writer"}:
            matches = set(re.findall(r"--gfx (gfx[0-9]{3,6})\b", task))
            check(len(matches) == 1, "helper_gfx_missing_or_conflicting")
            dynamic["gfx"] = matches.pop()
        if site == "citation":
            matches = re.findall(r"--citations -\n([^\n]+)\nCITEJSON\n", task)
            check(len(matches) == 1, "citation_payload_missing")
            rows = json.loads(matches[0])
            check(isinstance(rows, list) and 1 <= len(rows) <= 1024, "unsupported_citation_population")
            normalized = []
            for row in rows:
                check(isinstance(row, dict) and set(row) == set(_CITATION_FIELDS), "unsupported_citation_fields")
                check(all(isinstance(row[key], str) and row[key] for key in ("card", "direction", "specialty")), "invalid_citation_text")
                check(type(row["round"]) is int and 1 <= row["round"] <= self._configuration["budget"], "invalid_citation_round")
                if row["cited_then_verified"] is not None:
                    _finite(row["cited_then_verified"])
                check(type(row["became_winner"]) is bool, "invalid_citation_outcome")
                normalized.append({key: row[key] for key in _CITATION_FIELDS})
            dynamic["citations"] = normalized
        if site == "writer":
            pattern = (r"--gfx gfx[0-9]{3,6} --kernel-class (" + _JSON_STRING + r") \\\n"
                       r"  --speedup (" + _NUMBER + r") --baseline-wall-ms (" + _NUMBER + r") \\\n"
                       r"  --patch (" + _JSON_STRING + r") --eval-dir (" + _JSON_STRING + r") \\\n"
                       r"  --report (" + _JSON_STRING + r") --metric-kind ([a-z_]+) \\\n"
                       r"  --direction (" + _JSON_STRING + r") --case-names (" + _JSON_STRING + r")"
                       r"(?: \\\n  --precision (" + _JSON_STRING + r"))?"
                       r"(?: \\\n  --parent (" + _JSON_STRING + r"))?\n```")
            matches = list(re.finditer(pattern, task))
            check(len(matches) == 1, "writer_arguments_missing")
            klass, speedup, baseline, patch_path, eval_dir, report_path, metric, direction, cases, precision, parent = matches[0].groups()
            dynamic.update(kernel_class=_text(json.loads(klass), 128), speedup=float(speedup), baseline_ms=float(baseline),
                           patch=_path(json.loads(patch_path), self._setup["eval_dir"]),
                           report=_path(json.loads(report_path), self._setup["eval_dir"]),
                           direction=_text(json.loads(direction), 60), parent=_path(json.loads(parent)) if parent else "")
            _finite(dynamic["speedup"], positive=True)
            _finite(dynamic["baseline_ms"], positive=True)
            check(dynamic["speedup"] > 1, "writer_gate_closed")
            check(json.loads(eval_dir) == self._setup["eval_dir"], "writer_eval_directory_mismatch")
            check(metric == ("time_weighted" if self._configuration["has_workload"] else "geomean"), "writer_metric_mismatch")
            check((json.loads(precision) if precision else "") == self._configuration["dtype"], "writer_precision_mismatch")
            check(re.fullmatch(r"(?:[a-z0-9]+(?:-[a-z0-9]+)*)?", dynamic["direction"]) is not None, "writer_direction_mismatch")
            names = _text(json.loads(cases))
            dynamic["case_names"] = names.split(",") if names else []
        return dynamic

    def resolve(self, label, task, schema):
        """Return an eligible command only after complete source-task equality."""
        self.verify_sources()
        result = {"eligible": False, "role": None, "command": None, "completion_marker": None, "reason": None}
        try:
            check(isinstance(label, str) and isinstance(task, str) and len(task.encode()) <= 262144, "invalid_helper_input")
            site, initial = self._site(label)
            result["role"] = _ROLES[site]
            check(self._setup is not None, "trusted_setup_missing")
            check(initial.get("round", 1) <= self._configuration["budget"], "helper_round_outside_budget")
            enabled = {"clock": self._configuration["deadline_enabled"], "resolver": self._configuration["warm_start_enabled"],
                       "citation": not self._configuration["held_out"], "writer": self._configuration["writer_enabled"], "storage": True}
            check(enabled[site], "helper_site_disabled")
            dynamic = self._dynamic(site, task, initial)
            expected = self._render("render", site, dynamic)
            check(canonical(schema) == canonical(expected["options"]["schema"]), "helper_schema_changed")
            check(task == expected["task"], "helper_task_changed")
            commands = re.findall(r"```bash\n([\s\S]*?)\n```", task)
            check(len(commands) == 1, "helper_command_ambiguous")
            result.update(eligible=True, command=commands[0], reason="exact_public_source_template",
                          completion_marker="STORAGE_RECLAIM_DONE round=" + str(initial["round"]) if site == "storage" else None)
        except Unsupported as error:
            result["reason"] = error.code
        except (ValueError, TypeError, KeyError, OverflowError, UnicodeError, RecursionError):
            result["reason"] = "invalid_helper_arguments"
        return result
