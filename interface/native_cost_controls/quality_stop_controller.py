# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the fixed-floor stopping protocol behind a trusted host boundary.

The evaluator and isolation boundary are host objects, never Workflow inputs.
There is deliberately no default evaluator or permissive isolation boundary.
Neither an agent result nor a numeric certificate establishes source authority.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import secrets
import stat
import subprocess
import threading
import time
import uuid
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path, PurePosixPath

from .helper_driver import json_values_equal
from .quality_stop_failure_codes import CHECK_FAILURE_CODES
from .quality_stop_stats import assess_scores

PROTOCOL = "geak-fixed-floor-stop-v2"
PREFIX = "GEAK_QUALITY_STOP_V2\n"
SCHEMA = {"type": "object", "additionalProperties": False,
          "properties": {"payload": {"type": "string"}, "signature": {"type": "string"}},
          "required": ["payload", "signature"]}
_FIELDS = {"protocol", "trial_id", "stage", "look_index", "round", "dispatched", "budget",
           "no_improve", "max_no_improve", "forced_replans", "deadline_epoch", "candidate_root"}
_FINAL_FIELDS = {"final_patch", "director_final_patch", "export_root"}
_SEED_FIELDS = {"protocol", "trial_id", "stage", "candidate_root", "seed_manifest_sha256"}


class StopRejected(ValueError):
    """Return a fixed rejection code without private host data."""


def require(condition, reason):
    if not condition:
        raise StopRejected(reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def check_failure_code(error):
    if isinstance(error, StopRejected):
        return str(error) if str(error) in CHECK_FAILURE_CODES else "unregistered_stop_rejection"
    if isinstance(error, OSError):
        return "check_os_error"
    if isinstance(error, subprocess.SubprocessError):
        return "check_subprocess_error"
    if isinstance(error, (ValueError, TypeError, KeyError)):
        return "check_record_error"
    return "check_runtime_error"


def parse_task(task):
    require(isinstance(task, str) and task.startswith(PREFIX), "checkpoint_task_missing")
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "checkpoint_duplicate_key")
            result[key] = value
        return result
    try:
        value = json.loads(task[len(PREFIX):], object_pairs_hook=pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(StopRejected("nonfinite_request")))
        canonical(value)
    except (TypeError, ValueError, UnicodeError) as error:
        if isinstance(error, StopRejected):
            raise
        raise StopRejected("invalid_checkpoint_json") from None
    require(isinstance(value, dict), "checkpoint_fields_changed")
    if value.get("stage") == "seed":
        require(set(value) == _SEED_FIELDS and value["protocol"] == PROTOCOL
                and isinstance(value["seed_manifest_sha256"], str)
                and re.fullmatch(r"[0-9a-f]{64}", value["seed_manifest_sha256"]), "seed_request_invalid")
        return value
    expected = _FIELDS | _FINAL_FIELDS if value.get("stage") == "finalize" else _FIELDS
    require(set(value) == expected, "checkpoint_fields_changed")
    require(value["protocol"] == PROTOCOL and value["stage"] in {"boundary", "consume", "finalize"}, "checkpoint_protocol")
    for name in ("look_index", "round", "dispatched", "budget", "no_improve", "max_no_improve", "forced_replans"):
        require(type(value[name]) is int and value[name] >= 0, "invalid_checkpoint_counter")
    require(1 <= value["look_index"] <= 3 and value["round"] >= 1, "invalid_checkpoint_look")
    require(type(value["deadline_epoch"]) in (int, float) and math.isfinite(value["deadline_epoch"])
            and value["deadline_epoch"] >= 0, "invalid_checkpoint_deadline")
    return value


class ExecutionBoundary(ABC):
    """Check OS authority outside the actor's filesystem and process namespace.

    Implementations must inspect the actor's actual sandbox and process state.
    They must protect workflow/controller code, native journals, signing memory,
    state files, and snapshots. They must include descendant and background
    processes that can affect source or the GPU. A caller-supplied boolean is
    not evidence. The native census runs separately in the checkpoint registry.
    """

    @abstractmethod
    def check(self, *, stage, candidate_root, protected_paths):
        """Raise StopRejected unless authority and process quiescence hold."""

    @abstractmethod
    def lease(self, *, candidate_root, protected_paths):
        """Hold continuous OS exclusion in a context manager.

        Prevent competing source/GPU producers and protect source, snapshots,
        native journals, controller state, and signing memory for the entire
        context. A pre/post observation pair does not implement this contract.
        Raise StopRejected if exclusion cannot start or loses authority.
        """

    def prepare_sdk_options(self, options):
        """Keep fixture options unchanged. Production boundaries override this."""
        return options

    def native_started(self):
        """Let production boundaries bind the actual native process before query."""

    def read_native_output(self, path):
        """Read local fixture output. Production boundaries restrict its root."""
        return Path(path).read_text(encoding="utf-8")


class ProcessEvaluator(ABC):
    """Measure one fresh paired process using only the protected snapshot.

    The implementation must check exit status, parity, output schema, record
    hashes, source identity, and freshness from its own child process. It returns
    process_id, snapshot_sha256, order, and buckets. Each bucket contains only
    bucket_id, reference_ms, and candidate_ms. The controller supplies call
    weights. Model-authored measurements are never valid evaluator records.
    """

    @abstractmethod
    def run_process(self, *, snapshot, snapshot_sha256, process_id, order):
        """Start exactly one process, then return its checked record."""


class RSASigner:
    """Use the cryptography package's standard RSA PKCS#1 v1.5 implementation."""

    def __init__(self):
        from cryptography.hazmat.primitives.asymmetric import rsa
        self._key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        numbers = self._key.public_key().public_numbers()
        def encoded(number):
            return base64.urlsafe_b64encode(number.to_bytes((number.bit_length() + 7) // 8, "big")).decode().rstrip("=")
        self.public_jwk = {"kty": "RSA", "n": encoded(numbers.n), "e": encoded(numbers.e),
                           "alg": "RS256", "ext": True, "key_ops": ["verify"]}

    def sign(self, value):
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding
        payload = canonical(value)
        signature = self._key.sign(payload.encode("ascii"), padding.PKCS1v15(), hashes.SHA256())
        return {"payload": payload, "signature": base64.b64encode(signature).decode("ascii")}


def _git(root, *arguments):
    environment = {"PATH": "/usr/bin:/bin", "LC_ALL": "C", "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0",
                   "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null",
                   "GIT_NO_LAZY_FETCH": "1", "GIT_NO_REPLACE_OBJECTS": "1"}
    if "HOME" in os.environ:
        environment["HOME"] = os.environ["HOME"]
    result = subprocess.run(["/usr/bin/git", "--no-replace-objects", "-c", "core.fsmonitor=false", "-C", str(root), *arguments],
                            stdin=subprocess.DEVNULL, capture_output=True,
                            check=False, timeout=30, env=environment)
    require(result.returncode == 0, "candidate_git_failed")
    return result.stdout


def _check_git_layout(root):
    metadata = root / ".git"
    require(metadata.is_dir() and not metadata.is_symlink(), "candidate_git_directory_unprotected")
    for name in ("commondir", "gitdir", "shallow", "info/grafts", "objects/info/alternates",
                 "objects/info/http-alternates", "refs/replace", "worktrees"):
        path = metadata / name
        require(not path.exists() and not path.is_symlink(), "candidate_git_indirection_unsupported")
    for directory, folders, names in os.walk(metadata, followlinks=False):
        for name in folders + names:
            path = Path(directory) / name
            info = path.lstat()
            require(not stat.S_ISLNK(info.st_mode), "candidate_git_link_unsupported")
            require(stat.S_ISDIR(info.st_mode) or (stat.S_ISREG(info.st_mode) and info.st_nlink == 1),
                    "candidate_git_external_metadata")
            require(not name.endswith(".promisor"), "candidate_promisor_unsupported")
    packed = metadata / "packed-refs"
    require(not packed.exists() or b" refs/replace/" not in packed.read_bytes(), "candidate_replace_refs_unsupported")
    config = _git(root, "config", "--local", "--no-includes", "--null", "--list")
    allowed = {"core.repositoryformatversion": {"0"}, "core.filemode": {"true", "false"}, "core.bare": {"false"},
               "core.logallrefupdates": {"true", "false"}, "core.ignorecase": {"true", "false"},
               "user.name": None, "user.email": None}
    seen = set()
    for row in config.split(b"\0"):
        if not row:
            continue
        pair = row.decode("utf-8").split("\n", 1)
        require(len(pair) == 2 and pair[0] in allowed and pair[0] not in seen, "candidate_git_config_unsupported")
        name, value = pair
        require(allowed[name] is None or value in allowed[name], "candidate_git_config_unsupported")
        seen.add(name)


def _absolute_path(value):
    path = Path(value)
    require(path.is_absolute() and path.resolve() == path, "selected_host_path_not_canonical")
    return path


def _actor_path(value):
    require(isinstance(value, str) and PurePosixPath(value).is_absolute()
            and str(PurePosixPath(value)) == value and ".." not in PurePosixPath(value).parts,
            "selected_actor_path_not_canonical")
    return value


def _relative_path(value):
    require(isinstance(value, str) and value not in {"", "."} and not PurePosixPath(value).is_absolute()
            and str(PurePosixPath(value)) == value and ".." not in PurePosixPath(value).parts,
            "selected_relative_path_invalid")
    return value


def _file_record(record):
    require(isinstance(record, dict) and set(record) == {"mode", "sha256", "bytes"}
            and record["mode"] in {"100644", "100755"}
            and isinstance(record["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
            and type(record["bytes"]) is int and record["bytes"] >= 0, "invalid_pinned_file_record")
    return deepcopy(record)


def _metadata_records(value):
    value = {} if value is None else value
    require(isinstance(value, dict) and set(value) <= {".geak/workspace.json"}, "metadata_allowance_unsupported")
    records = {path: _file_record(record) for path, record in value.items()}
    require(all(record["mode"] == "100644" for record in records.values()), "metadata_executable_forbidden")
    return records


class SelectedArtifacts:
    """Pin the seed and every exported source path through trusted launch data.

    The initial profile accepts byte-exact canonical Git patches only. It does
    not apply an actor-written patch in the trusted host process. The deployed
    native writer must qualify against this narrower profile before launch.
    """

    def __init__(self, *, baseline_commit, patch_path, actor_patch_path, export_root, actor_export_root, files,
                 expected_seed_files=None, metadata_files=None):
        require(baseline_commit is None or (isinstance(baseline_commit, str)
                and re.fullmatch(r"[0-9a-f]{40}", baseline_commit)), "selected_baseline_commit_invalid")
        self.baseline_commit = baseline_commit
        self._launch_baseline_commit = baseline_commit
        self.metadata_files = _metadata_records(metadata_files)
        self.expected_seed_files = None
        self.expected_seed_sha256 = None
        self._baseline_records = None
        if baseline_commit is None:
            require(isinstance(expected_seed_files, list) and bool(expected_seed_files), "expected_seed_files_required")
            checked = []
            for record in expected_seed_files:
                require(isinstance(record, dict) and "path" in record, "invalid_seed_file_record")
                path = _relative_path(record["path"])
                require(path not in self.metadata_files, "seed_metadata_is_not_executable_source")
                checked.append({"path": path, **_file_record({key: value for key, value in record.items() if key != "path"})})
            require(len({record["path"] for record in checked}) == len(checked), "duplicate_seed_file")
            self.expected_seed_files = sorted(checked, key=lambda record: record["path"])
            self._baseline_records = deepcopy(self.expected_seed_files)
            self.expected_seed_sha256 = sha(canonical({"files": self.expected_seed_files, "metadata": self.metadata_files}).encode("ascii"))
        else:
            require(expected_seed_files is None, "ambiguous_seed_binding")
        self.patch_path, self.export_root = _absolute_path(patch_path), _absolute_path(export_root)
        self.actor_patch_path, self.actor_export_root = _actor_path(actor_patch_path), _actor_path(actor_export_root)
        require(self.patch_path != self.export_root and self.export_root not in self.patch_path.parents,
                "selected_patch_inside_exports")
        require(isinstance(files, dict) and bool(files), "selected_export_map_required")
        self.files = {_relative_path(source): _relative_path(export) for source, export in files.items()}
        require(len(set(self.files.values())) == len(self.files), "selected_export_collision")

    @property
    def public_config(self):
        return {"profile": "canonical-git-diff-v1", "baseline_commit": self._launch_baseline_commit,
                "seed_manifest_sha256": self.expected_seed_sha256,
                "final_patch": self.actor_patch_path, "export_root": self.actor_export_root,
                "files": deepcopy(self.files)}

    def bind_seed(self, manifest):
        require(self.baseline_commit is None and self.expected_seed_files is not None, "seed_already_bound")
        require(json_values_equal(manifest["files"], self.expected_seed_files)
                and json_values_equal(manifest.get("metadata", {}), self.metadata_files), "seed_source_mismatch")
        self.baseline_commit = manifest["commit"]

    def validate_candidate(self, *, candidate_root, manifest):
        require(self.baseline_commit is not None, "seed_not_bound")
        _check_git_layout(candidate_root)
        _git(candidate_root, "merge-base", "--is-ancestor", self.baseline_commit, manifest["commit"])
        changed = _git(candidate_root, "diff-tree", "--no-commit-id", "--name-only", "--no-renames",
                       "--no-ext-diff", "--no-textconv", "-r", "-z", self.baseline_commit, manifest["commit"], "--")
        paths = {name.decode("utf-8") for name in changed.split(b"\0") if name}
        require(paths <= set(self.files), "immutable_seed_file_changed")
        require(set(self.files) <= {record["path"] for record in manifest["files"]}, "selected_source_missing")

    def check_immutable_worktree(self, candidate_root):
        require(self.baseline_commit is not None, "seed_not_bound")
        if self._baseline_records is None:
            _check_git_layout(candidate_root)
            records = []
            for row in _git(candidate_root, "ls-tree", "-rz", "--full-tree", self.baseline_commit).split(b"\0"):
                if not row:
                    continue
                metadata, name = row.split(b"\t", 1)
                mode, kind, blob = metadata.decode("ascii").split()
                require(kind == "blob" and mode in {"100644", "100755"}, "seed_external_source_unsupported")
                data = _git(candidate_root, "cat-file", "blob", blob)
                records.append({"path": _relative_path(name.decode("utf-8")), "mode": mode, "sha256": sha(data), "bytes": len(data)})
            self._baseline_records = records
        for record in self._baseline_records:
            if record["path"] in self.files:
                continue
            path = candidate_root / record["path"]
            require(path.exists() and path.resolve() == path, "immutable_seed_file_changed")
            info = path.lstat()
            require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                    and bool(info.st_mode & 0o111) == (record["mode"] == "100755"), "immutable_seed_file_changed")
            data = path.read_bytes()
            require(len(data) == record["bytes"] and sha(data) == record["sha256"], "immutable_seed_file_changed")

    def verify(self, *, candidate_root, manifest, request):
        require(isinstance(request, dict), "selected_artifact_request_invalid")
        self.validate_candidate(candidate_root=candidate_root, manifest=manifest)
        require(request.get("final_patch") == self.actor_patch_path
                and request.get("director_final_patch") == self.actor_patch_path
                and request.get("export_root") == self.actor_export_root, "selected_artifact_path_changed")
        require(_git(candidate_root, "cat-file", "-t", self.baseline_commit) == b"commit\n", "selected_seed_not_commit")
        _git(candidate_root, "merge-base", "--is-ancestor", self.baseline_commit, manifest["commit"])
        expected = _git(candidate_root, "diff", "--no-ext-diff", "--no-textconv", "--no-color",
                        "--src-prefix=a/", "--dst-prefix=b/", self.baseline_commit, manifest["commit"], "--")
        require(b"\nBinary files " not in expected and b"\nGIT binary patch\n" not in expected,
                "selected_binary_patch_unsupported")
        require(self.patch_path.resolve() == self.patch_path, "selected_patch_redirected")
        info = self.patch_path.lstat()
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_size <= 32 * 1024 * 1024,
                "selected_patch_not_owned_regular_file")
        actual = self.patch_path.read_bytes()
        require(actual == expected, "selected_patch_differs_from_certificate")
        require(self.export_root.resolve() == self.export_root and self.export_root.is_dir(), "selected_export_root_redirected")
        observed = set()
        for directory, folders, names in os.walk(self.export_root, followlinks=False):
            for name in folders:
                require(not (Path(directory) / name).is_symlink(), "selected_export_directory_link")
            for name in names:
                observed.add((Path(directory) / name).relative_to(self.export_root).as_posix())
        require(observed == set(self.files.values()), "selected_export_set_changed")
        entries = {entry["path"]: entry for entry in manifest["files"]}
        receipts = []
        for source, export in sorted(self.files.items()):
            require(source in entries, "selected_source_not_certified")
            path, expected_source = self.export_root / export, entries[source]
            require(path.resolve() == path, "selected_export_redirected")
            info = path.lstat()
            require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                    and bool(info.st_mode & 0o111) == (expected_source["mode"] == "100755"), "selected_export_mode_changed")
            data = path.read_bytes()
            require(len(data) == expected_source["bytes"] and sha(data) == expected_source["sha256"], "selected_export_source_changed")
            receipts.append({"candidate_path": source, "path": self.actor_export_root + "/" + export,
                             "sha256": sha(data), "bytes": len(data), "mode": expected_source["mode"]})
        return {"profile": "canonical-git-diff-v1", "baseline_commit": self.baseline_commit,
                "final_patch": {"path": self.actor_patch_path, "sha256": sha(actual), "bytes": len(actual)},
                "export_root": self.actor_export_root, "files": receipts}


def source_manifest(root, *, metadata_files=None):
    """Hash the complete clean committed tree and reject external source links."""
    root = Path(root)
    metadata = _metadata_records(metadata_files)
    require(root.is_absolute() and root.resolve() == root, "candidate_root_not_canonical")
    _check_git_layout(root)
    require(_git(root, "rev-parse", "--show-toplevel").decode().strip() == str(root), "candidate_not_repository_root")
    commit = _git(root, "rev-parse", "HEAD").decode().strip()
    raw = _git(root, "ls-tree", "-rz", "--full-tree", commit)
    files, index = [], []
    for item in raw.split(b"\0"):
        if not item:
            continue
        tree_metadata, name = item.split(b"\t", 1)
        mode, kind, blob = tree_metadata.decode("ascii").split()
        name = name.decode("utf-8")
        require(name not in metadata, "metadata_must_remain_untracked")
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "candidate_path_escape")
        require(kind == "blob" and mode in {"100644", "100755"}, "candidate_external_source")
        index.append(mode.encode() + b" " + blob.encode() + b" 0\t" + name.encode() + b"\0")
        path = root / name
        info = path.lstat()
        require(path.resolve() == path and stat.S_ISREG(info.st_mode), "candidate_source_link")
        require(info.st_nlink == 1, "candidate_source_hardlink")
        require(bool(info.st_mode & 0o111) == (mode == "100755"), "candidate_executable_mode_changed")
        data = path.read_bytes()
        require(data == _git(root, "cat-file", "blob", blob), "candidate_worktree_not_committed")
        files.append({"path": name, "mode": mode, "sha256": sha(data), "bytes": len(data)})
    require(bool(files), "candidate_empty")
    # Do not use git status or a worktree diff. Those commands can invoke
    # repository-controlled clean filters in the trusted host process.
    require(_git(root, "ls-files", "--stage", "-z") == b"".join(index), "candidate_index_not_committed")
    actual_files = set()
    for directory, folders, names in os.walk(root, followlinks=False):
        if Path(directory) == root:
            folders[:] = [name for name in folders if name != ".git"]
        for name in folders:
            require(not (Path(directory) / name).is_symlink(), "candidate_directory_link")
        for name in names:
            path = Path(directory) / name
            require(not path.is_symlink(), "candidate_source_link")
            actual_files.add(path.relative_to(root).as_posix())
    require(actual_files == {entry["path"] for entry in files} | set(metadata), "candidate_not_clean")
    for name, record in metadata.items():
        path = root / name
        info = path.lstat()
        require(path.resolve() == path and stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                and not info.st_mode & 0o111, "metadata_file_not_regular")
        data = path.read_bytes()
        require(len(data) == record["bytes"] and sha(data) == record["sha256"], "metadata_file_changed")
    require(_git(root, "rev-parse", "HEAD").decode().strip() == commit
            and _git(root, "ls-files", "--stage", "-z") == b"".join(index), "candidate_changed_during_freeze")
    manifest = {"commit": commit, "files": files}
    if metadata:
        manifest["metadata"] = metadata
    return {**manifest, "sha256": sha(canonical(manifest).encode("ascii"))}


class QualityStopController:
    """Consume at most three boundary looks within one non-resumable trial."""

    def __init__(self, *, state_dir, candidate_root, trial_id, budget, max_no_improve,
                 deadline_epoch, buckets, boundary, evaluator, selected_artifacts, actor_candidate_root=None,
                 stopping_enabled=True, native_session_id=None, signer=None, clock=time.time):
        require(isinstance(boundary, ExecutionBoundary) and isinstance(evaluator, ProcessEvaluator), "trusted_host_adapters_required")
        require(isinstance(selected_artifacts, SelectedArtifacts), "trusted_selected_artifacts_required")
        require(type(stopping_enabled) is bool, "stopping_enabled_must_be_boolean")
        if native_session_id is not None:
            try:
                require(isinstance(native_session_id, str) and str(uuid.UUID(native_session_id)) == native_session_id,
                        "native_session_id_invalid")
            except ValueError:
                raise StopRejected("native_session_id_invalid") from None
        self.native_session_id = native_session_id
        require(type(budget) is int and budget > 0 and type(max_no_improve) is int and max_no_improve > 0,
                "invalid_native_limits")
        require(isinstance(trial_id, str) and 16 <= len(trial_id) <= 128 and trial_id.isascii()
                and all(c.isalnum() or c in "_-" for c in trial_id), "invalid_trial_id")
        require(type(deadline_epoch) in (int, float) and math.isfinite(deadline_epoch) and deadline_epoch >= 0,
                "invalid_native_deadline")
        require(isinstance(buckets, dict) and bool(buckets)
                and all(isinstance(k, str) and k and type(v) in (int, float) and math.isfinite(v) and v > 0
                        for k, v in buckets.items()), "invalid_workload_weights")
        self.state_dir, self.candidate_root = Path(state_dir), Path(candidate_root)
        self.actor_candidate_root = _actor_path(actor_candidate_root if actor_candidate_root is not None else str(self.candidate_root))
        require(self.state_dir.is_absolute() and self.candidate_root.is_absolute(), "host_paths_not_absolute")
        require(self.state_dir.resolve() == self.state_dir and self.candidate_root.resolve() == self.candidate_root,
                "host_path_symlink")
        require(self.candidate_root not in self.state_dir.parents and self.state_dir not in self.candidate_root.parents
                and self.state_dir != self.candidate_root, "controller_inside_candidate")
        self.state_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
        self.boundary, self.evaluator = boundary, evaluator
        self.selected_artifacts = selected_artifacts
        self.metadata_files = deepcopy(selected_artifacts.metadata_files)
        self.stopping_enabled = stopping_enabled
        self.seed_bound = selected_artifacts.baseline_commit is not None
        self.seed_attempt = self.seed_binding = None
        self.closure_receipt = None
        self.closure_failure_code = None
        self.signer, self.clock = signer or RSASigner(), clock
        module_dir = Path(__file__).resolve().parent
        repository = module_dir.parent.parent
        sources = sorted(module_dir.glob("*.py")) + [module_dir.parent / name for name in ("run_e2e.py", "run_kernel_native.py")]
        sources += [repository / "kernel_workflow" / name for name in ("kernel_workflow.js", "kernel_lane.js", "quality_stop_verify.js")]
        sources += [repository / "kernel_workflow" / name for name in
                    ("roles/director.md", "scripts/materialize_workspace.sh", "scripts/workspace_sources.py")]
        self._source_bindings = {path: sha(path.read_bytes()) for path in sources}
        self.protected_paths = (self.state_dir, *self._source_bindings, selected_artifacts.patch_path, selected_artifacts.export_root)
        self.trial_id, self.budget, self.max_no_improve = trial_id, budget, max_no_improve
        self.deadline_epoch, self.buckets = deadline_epoch, deepcopy(buckets)
        self.looks, self.certificate = {}, None
        self.native_scope = None
        self.failed, self.finalized = False, False
        self.consumed = False
        self._native_closure = None
        self.lock = threading.RLock()
        self._persist()

    @property
    def public_config(self):
        return {"protocol": PROTOCOL, "trial_id": self.trial_id, "public_key": deepcopy(self.signer.public_jwk),
                "candidate_root": self.actor_candidate_root, "selected_artifacts": self.selected_artifacts.public_config,
                "seed_manifest_sha256": self.selected_artifacts.expected_seed_sha256,
                "stopping_enabled": self.stopping_enabled}

    def _persist(self):
        data = canonical({"protocol": PROTOCOL, "trial_id": self.trial_id, "looks": self.looks,
                          "native_scope": self.native_scope, "certificate": self.certificate,
                          "failed": self.failed, "finalized": self.finalized, "consumed": self.consumed,
                          "seed_bound": self.seed_bound, "seed_binding": self.seed_binding, "seed_attempt": self.seed_attempt,
                          "stopping_enabled": self.stopping_enabled, "closure_receipt": self.closure_receipt,
                          "closure_failure_code": self.closure_failure_code}).encode("ascii")
        temporary = self.state_dir / ".state.tmp"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.state_dir / "state.json")
            directory = os.open(self.state_dir, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except BaseException:
            self.failed = True
            raise

    def _authority(self, stage):
        for path, digest in self._source_bindings.items():
            require(path.resolve() == path and sha(path.read_bytes()) == digest, "trusted_controller_source_changed")
        self.boundary.check(stage=stage, candidate_root=self.candidate_root,
                            protected_paths=self.protected_paths)

    def bind_native_closure(self, callback):
        require(self._native_closure is None and callable(callback), "native_closure_already_bound")
        self._native_closure = callback

    def confirm_native_return(self, result):
        """Reject a claimed success unless native closure and final source agree."""
        with self.lock:
            try:
                self._confirm_native_return(result)
            except BaseException as error:
                self.failed = True
                self.closure_failure_code = check_failure_code(error)
                self._persist()
                raise

    def _confirm_native_return(self, result):
        require(self._native_closure is not None, "native_closure_not_bound")
        quality = result.get("quality_stop") if isinstance(result, dict) else None
        require(isinstance(quality, dict) and quality.get("enabled") is True, "native_quality_result_missing")
        with self.boundary.lease(candidate_root=self.candidate_root, protected_paths=self.protected_paths):
            census = self._native_closure(result)
            require(isinstance(census, dict) and census.get("schema") == "geak-quality-native-census-v1",
                    "native_closure_census_missing")
            require(self.seed_bound and not self.failed, "native_seed_or_trial_unconfirmed")
            require(json_values_equal(quality.get("stopping_enabled"), self.stopping_enabled),
                    "native_stopping_profile_changed")
            if self.seed_attempt is not None:
                require(json_values_equal(quality.get("seed"), json.loads(self.seed_attempt["envelope"]["payload"])),
                        "native_seed_result_changed")
            if quality.get("qualifying") is not True:
                self._authority("native_return")
                require(quality.get("qualifying") is False and result.get("stopped_by") in
                        {"budget", "deadline", "no_improve", "tech_lead_stop"}, "native_ordinary_exit_unconfirmed")
                require(type(result.get("budget_used")) is int and 0 <= result["budget_used"] <= self.budget
                        and json_values_equal(result.get("budget_total"), self.budget), "native_budget_return_changed")
                manifest = self._current_source()
                selected = self.selected_artifacts.verify(candidate_root=self.candidate_root, manifest=manifest,
                                                          request=quality.get("final_artifacts", {}))
                require(result.get("final_patch") == self.selected_artifacts.actor_patch_path, "native_selected_artifact_changed")
                require(self.closure_receipt is None, "native_closure_repeated")
                snapshot, frozen = self._snapshot("final")
                require(frozen == manifest, "source_changed_during_closure")
                self._record_closure(result, manifest, selected, snapshot.name, census)
                return
            require(self.finalized and self.consumed and not self.failed and self.certificate is not None, "native_quality_claim_unconfirmed")
            self._authority("native_return")
            require(self._current_source() == self.certificate["manifest"], "final_source_changed_after_checkpoint")
            final_payload = json.loads(self.certificate["final_envelope"]["payload"])
            look = self.looks[str(self.certificate["look_index"])]
            selected = self.selected_artifacts.verify(candidate_root=self.candidate_root, manifest=self.certificate["manifest"],
                                                       request=parse_task(final_payload["task"]))
            require(json_values_equal(selected, final_payload["selected_artifacts"])
                    and result.get("final_patch") == self.selected_artifacts.actor_patch_path
                    and final_payload["consumption_sha256"] == sha(self.certificate["consume_envelope"]["payload"].encode("ascii")),
                    "native_selected_artifact_changed")
            require(json_values_equal(quality.get("final_check"), final_payload) and final_payload["qualifying"] is True
                and json_values_equal(quality.get("certificate"), {"request": look["request"], "decision": json.loads(look["envelope"]["payload"]),
                    "consumption": json.loads(self.certificate["consume_envelope"]["payload"])})
                    and result.get("stopped_by") == "quality_certificate"
                    and json_values_equal(result.get("budget_used"), look["request"]["dispatched"])
                and json_values_equal(result.get("budget_total"), self.budget), "native_quality_return_changed")
            self._record_closure(result, self.certificate["manifest"], selected,
                                 "snapshot_" + str(self.certificate["look_index"]), census)

    def _record_closure(self, result, manifest, selected, snapshot_name, census):
        require(self.closure_receipt is None, "native_closure_repeated")
        raw = (canonical(census) + "\n").encode("ascii")
        path = self.state_dir / "native_census.json"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        self.closure_receipt = self.signer.sign({"protocol": PROTOCOL, "stage": "native_closed",
            "trial_id": self.trial_id, "native_scope": self.native_scope, "seed_commit": self.selected_artifacts.baseline_commit,
            "stopping_enabled": self.stopping_enabled, "source_manifest": manifest, "selected_artifacts": selected,
            "snapshot": snapshot_name, "native_return_sha256": sha(canonical(result).encode("ascii")),
            "native_census": {"path": path.name, "sha256": sha(raw), "bytes": len(raw), "schema": census["schema"]},
            "qualifying": result["quality_stop"]["qualifying"], "issued_at": self.clock()})
        self._persist()

    def _admissible(self, request):
        return (request["dispatched"] < self.budget and request["no_improve"] < self.max_no_improve
                and (not self.deadline_epoch or self.clock() < self.deadline_epoch))

    def _current_source(self):
        if self.seed_bound:
            self.selected_artifacts.check_immutable_worktree(self.candidate_root)
        return source_manifest(self.candidate_root, metadata_files=self.metadata_files)

    def _validate(self, request):
        require(request["trial_id"] == self.trial_id and request["candidate_root"] == self.actor_candidate_root,
                "checkpoint_trial_changed")
        if request["stage"] == "seed":
            require(request["seed_manifest_sha256"] == self.selected_artifacts.expected_seed_sha256, "seed_manifest_binding_changed")
            return
        require(request["budget"] == self.budget and request["max_no_improve"] == self.max_no_improve
                and request["deadline_epoch"] == self.deadline_epoch, "checkpoint_limits_changed")

    def _snapshot(self, look):
        manifest = self._current_source()
        self.selected_artifacts.validate_candidate(candidate_root=self.candidate_root, manifest=manifest)
        target = self.state_dir / ("snapshot_" + str(look))
        target.mkdir(mode=0o700)
        for entry in manifest["files"]:
            source, destination = self.candidate_root / entry["path"], target / entry["path"]
            data = source.read_bytes()
            require(sha(data) == entry["sha256"], "candidate_changed_during_freeze")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            destination.chmod(0o500 if entry["mode"] == "100755" else 0o400)
        require(self._current_source() == manifest, "candidate_changed_during_freeze")
        return target, manifest

    def _score(self, result, *, identity, snapshot_sha256, order):
        require(isinstance(result, dict) and set(result) == {"process_id", "snapshot_sha256", "order", "buckets"}, "invalid_process_record")
        require(result["process_id"] == identity and result["snapshot_sha256"] == snapshot_sha256
                and result["order"] == order, "process_identity_changed")
        rows = result["buckets"]
        require(isinstance(rows, list) and len(rows) == len(self.buckets), "process_bucket_count")
        seen, reference, candidate = set(), [], []
        for row in rows:
            require(isinstance(row, dict) and set(row) == {"bucket_id", "reference_ms", "candidate_ms"}, "invalid_bucket_record")
            key = row["bucket_id"]
            require(isinstance(key, str) and key in self.buckets and key not in seen, "process_bucket_identity")
            seen.add(key)
            for field in ("reference_ms", "candidate_ms"):
                require(type(row[field]) in (int, float) and math.isfinite(row[field]) and row[field] > 0,
                        "invalid_process_timing")
            reference.append(self.buckets[key] * row["reference_ms"])
            candidate.append(self.buckets[key] * row["candidate_ms"])
        try:
            score = math.fsum(reference) / math.fsum(candidate)
        except (ValueError, OverflowError, ZeroDivisionError):
            raise StopRejected("undefined_process_score") from None
        require(math.isfinite(score) and score > 0, "undefined_process_score")
        return score

    def _verify_snapshot(self, snapshot, manifest):
        actual = set()
        for path in snapshot.rglob("*"):
            require(not path.is_symlink(), "snapshot_source_link")
            if path.is_file():
                actual.add(path.relative_to(snapshot).as_posix())
        require(actual == {entry["path"] for entry in manifest["files"]}, "snapshot_file_set_changed")
        for entry in manifest["files"]:
            require(sha((snapshot / entry["path"]).read_bytes()) == entry["sha256"], "snapshot_source_changed")

    def checkpoint(self, task, native_binding, *, census):
        """Bind a local call to checked native evidence and a protected snapshot."""
        with self.lock:
            request = parse_task(task)
            try:
                self._validate(request)
            except StopRejected:
                if request["stage"] == "seed":
                    self.failed = True
                    self._persist()
                raise
            require(isinstance(native_binding, dict) and set(native_binding) ==
                    {"session_id", "run_id", "root_tool", "root_task", "agent_id"}
                    and all(isinstance(value, str) and value for value in native_binding.values()), "native_binding_missing")
            require(self.native_session_id is None or native_binding["session_id"] == self.native_session_id,
                    "native_declared_session_changed")
            require(callable(census), "native_census_required")
            scope = {key: value for key, value in native_binding.items() if key != "agent_id"}
            require(self.native_scope is None or self.native_scope == scope, "native_trial_scope_changed")
            if request["stage"] == "seed":
                return self._seed(task, request, native_binding, scope, census)
            require(self.seed_bound, "trusted_seed_handoff_required")
            look = str(request["look_index"])
            if request["stage"] == "consume":
                return self._consume(task, request, native_binding, census)
            if request["stage"] == "finalize":
                return self._finalize(task, request, native_binding, census)
            require(not self.failed, "trial_outcome_unknown")
            require(self.stopping_enabled, "quality_stopping_disabled")
            prior = self.looks.get(look)
            if prior is not None:
                require(prior["task"] == task and prior["native_binding"] == native_binding, "duplicate_delivery_changed")
                require("envelope" in prior, "look_outcome_unknown")
                if json.loads(prior["envelope"]["payload"])["certified"] is True:
                    try:
                        with self.boundary.lease(candidate_root=self.candidate_root,
                                                 protected_paths=self.protected_paths):
                            census()
                            self._authority("duplicate")
                            require(self._admissible(request), "late_certificate_delivery")
                            require(self._current_source() == prior["manifest"], "duplicate_certificate_source_changed")
                    except BaseException:
                        self.failed = True
                        self._persist()
                        raise
                return deepcopy(prior["envelope"])
            require(not self.failed and not self.finalized and self.certificate is None, "trial_not_open")
            require(request["look_index"] == len(self.looks) + 1, "look_counter_changed")
            require(not self.looks or request["round"] > max(item["request"]["round"] for item in self.looks.values()),
                    "boundary_round_reused")
            require(self._admissible(request), "native_search_not_admissible")
            if self.native_scope is None:
                self.native_scope = scope
            entry = {"task": task, "request": request, "native_binding": deepcopy(native_binding),
                     "status": "started", "processes": [], "started_at": self.clock()}
            self.looks[look] = entry
            self._persist()  # A started invalid batch consumes this look.
            try:
                with self.boundary.lease(candidate_root=self.candidate_root,
                                         protected_paths=self.protected_paths):
                    reason, certified, manifest = "unknown", False, None
                    check_stage = "census"
                    try:
                        census()
                        check_stage = "authority"
                        self._authority("freeze")
                        check_stage = "source"
                        snapshot, manifest = self._snapshot(request["look_index"])
                        entry["manifest"] = manifest
                        # Independent fair bits. Draw only after the candidate freezes.
                        orders = ["reference_first" if secrets.randbits(1) == 0 else "candidate_first" for _ in range(3)]
                        entry["orders"] = orders
                        self._persist()  # Persist the entire vector before any process.
                        scores = []
                        for index, order in enumerate(orders):
                            check_stage = "snapshot"
                            self._verify_snapshot(snapshot, manifest)
                            identity = self.trial_id + ":look:" + look + ":process:" + str(index + 1)
                            entry["processes"].append({"process_id": identity, "status": "started", "order": order})
                            self._persist()
                            check_stage = "measurement"
                            result = self.evaluator.run_process(snapshot=snapshot, snapshot_sha256=manifest["sha256"],
                                                                process_id=identity, order=order)
                            check_stage = "snapshot"
                            self._verify_snapshot(snapshot, manifest)
                            check_stage = "measurement"
                            score = self._score(result, identity=identity, snapshot_sha256=manifest["sha256"], order=order)
                            scores.append(score)
                            entry["processes"][-1].update(status="complete", record=result, score=score)
                            self._persist()
                        check_stage = "census"
                        census()
                        check_stage = "authority"
                        self._authority("certify")
                        check_stage = "source"
                        require(self._current_source() == manifest, "candidate_changed_after_measurement")
                        require(self._admissible(request), "native_search_expired_during_measurement")
                        statistics = assess_scores(scores, look_index=request["look_index"], floor=1.05)
                        entry["statistics"] = statistics
                        certified = statistics["valid"] is True and statistics["certified"] is True
                        reason = statistics["reason"]
                    except (StopRejected, OSError, ValueError, TypeError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
                        # Do not retry a process, replace a process, or convert unknown data.
                        reason = "checkpoint_checks_failed"
                        certified = False
                        entry["check_failure_code"] = check_failure_code(error)
                        entry["check_failure_stage"] = check_stage
                        if check_stage in {"authority", "snapshot"} or entry["check_failure_code"] in {
                            "immutable_seed_file_changed", "metadata_file_changed", "metadata_file_not_regular",
                            "metadata_must_remain_untracked", "trusted_controller_source_changed",
                            "candidate_changed_during_freeze", "candidate_changed_after_measurement",
                        }:
                            self.failed = True
                    entry.update(status="certified" if certified else "rejected", reason=reason, finished_at=self.clock())
                    payload = {"protocol": PROTOCOL, "task": task, "native_binding": deepcopy(native_binding),
                               "certified": certified, "qualifying": False, "reason": reason,
                               "snapshot_sha256": manifest["sha256"] if manifest else None,
                               "look_index": request["look_index"], "round": request["round"],
                               "issued_at": self.clock(), "deadline_epoch": self.deadline_epoch}
                    entry["envelope"] = self.signer.sign(payload)
                    if certified:
                        self.certificate = {"look_index": request["look_index"], "task": task,
                                            "manifest": manifest, "native_binding": deepcopy(native_binding)}
                    self._persist()
                    return deepcopy(entry["envelope"])
            except BaseException as error:
                self.failed = True
                entry["check_failure_code"] = check_failure_code(error)
                entry["check_failure_stage"] = "lease_or_persistence"
                entry["status"] = "unknown"
                self._persist()
                raise

    def _seed(self, task, request, native_binding, scope, census):
        try:
            require(not self.failed and not self.seed_bound and self.seed_attempt is None and not self.looks,
                    "seed_handoff_not_admitted")
            self.native_scope = scope
            self.seed_attempt = {"task": task, "native_binding": deepcopy(native_binding), "status": "started"}
            self._persist()
            with self.boundary.lease(candidate_root=self.candidate_root, protected_paths=self.protected_paths):
                bound, reason, manifest = False, "seed_handoff_checks_failed", None
                try:
                    census()
                    self._authority("seed")
                    manifest = self._current_source()
                    require(_git(self.candidate_root, "rev-list", "--count", "HEAD") == b"1\n", "seed_history_not_fresh")
                    self.selected_artifacts.bind_seed(manifest)
                    self.seed_bound = True
                    bound, reason = True, "trusted_seed_bound"
                    self.seed_binding = {"native_binding": deepcopy(native_binding), "manifest": manifest,
                                         "expected_seed_sha256": request["seed_manifest_sha256"]}
                except (StopRejected, OSError, ValueError, TypeError, RuntimeError, subprocess.SubprocessError):
                    self.failed = True
                payload = {"protocol": PROTOCOL, "task": task, "native_binding": deepcopy(native_binding),
                           "certified": False, "qualifying": False, "seed_bound": bound, "reason": reason,
                           "seed_commit": manifest["commit"] if bound else None,
                           "seed_manifest_sha256": request["seed_manifest_sha256"], "issued_at": self.clock(),
                           "deadline_epoch": self.deadline_epoch}
                envelope = self.signer.sign(payload)
                self.seed_attempt.update(status="bound" if bound else "rejected", envelope=envelope)
                self._persist()
                return deepcopy(envelope)
        except BaseException:
            self.failed = True
            self._persist()
            raise

    def _consume(self, task, request, native_binding, census):
        require(self.certificate is not None and not self.failed and not self.finalized, "certificate_consumption_not_admitted")
        certificate = self.certificate
        original = self.looks[str(certificate["look_index"])]["request"]
        require({**request, "stage": "boundary"} == original, "certificate_consumption_binding_changed")
        previous = certificate.get("consume_attempt")
        if previous is not None:
            require(previous["task"] == task and previous["native_binding"] == native_binding
                    and "consume_envelope" in certificate, "duplicate_consumption_changed_or_unknown")
        else:
            certificate["consume_attempt"] = {"task": task, "native_binding": deepcopy(native_binding), "status": "started"}
            self._persist()
        try:
            with self.boundary.lease(candidate_root=self.candidate_root, protected_paths=self.protected_paths):
                consumed, reason = False, "certificate_consumption_checks_failed"
                try:
                    census()
                    self._authority("consume")
                    require(self._admissible(request), "late_certificate_consumption")
                    require(self._current_source() == certificate["manifest"], "source_changed_before_consumption")
                    consumed, reason = True, "certificate_consumed_for_early_exit"
                except (StopRejected, OSError, ValueError, TypeError, RuntimeError, subprocess.SubprocessError):
                    self.failed = True
                if previous is not None:
                    require(consumed, "duplicate_consumption_invalidated")
                    return deepcopy(certificate["consume_envelope"])
                self.consumed = consumed
                payload = {"protocol": PROTOCOL, "task": task, "native_binding": deepcopy(native_binding),
                           "certified": consumed, "consumed": consumed, "qualifying": False, "reason": reason,
                           "snapshot_sha256": certificate["manifest"]["sha256"], "look_index": request["look_index"],
                           "round": request["round"], "issued_at": self.clock(), "deadline_epoch": self.deadline_epoch}
                certificate["consume_envelope"] = self.signer.sign(payload)
                certificate["consume_attempt"]["status"] = "consumed" if consumed else "rejected"
                self._persist()
                return deepcopy(certificate["consume_envelope"])
        except BaseException:
            self.failed = True
            self._persist()
            raise

    def _finalize(self, task, request, native_binding, census):
        require(self.certificate is not None and self.consumed and not self.failed, "finalization_not_admitted")
        certificate = self.certificate
        original = self.looks[str(certificate["look_index"])]["request"]
        require({**{key: value for key, value in request.items() if key not in _FINAL_FIELDS}, "stage": "boundary"}
                == original, "finalization_binding_changed")
        if self.finalized:
            envelope = certificate["final_envelope"]
            payload = json.loads(envelope["payload"])
            require(payload["task"] == task and payload["native_binding"] == native_binding, "duplicate_final_delivery_changed")
            try:
                with self.boundary.lease(candidate_root=self.candidate_root, protected_paths=self.protected_paths):
                    census()
                    self._authority("duplicate_finalize")
                    require(self._current_source() == certificate["manifest"], "duplicate_final_source_changed")
                    selected = self.selected_artifacts.verify(candidate_root=self.candidate_root, manifest=certificate["manifest"], request=request)
                    require(json_values_equal(selected, payload["selected_artifacts"]), "duplicate_final_artifact_changed")
            except BaseException:
                self.failed = True
                self._persist()
                raise
            return deepcopy(envelope)
        require(not self.failed, "finalization_not_admitted")
        try:
            with self.boundary.lease(candidate_root=self.candidate_root,
                                     protected_paths=self.protected_paths):
                qualifying, reason, selected = False, "final_source_or_authority_changed", None
                try:
                    census()
                    self._authority("finalize")
                    require(self._current_source() == certificate["manifest"], "final_source_changed")
                    selected = self.selected_artifacts.verify(candidate_root=self.candidate_root, manifest=certificate["manifest"], request=request)
                    qualifying, reason = True, "final_source_equals_certified_snapshot"
                except (StopRejected, OSError, ValueError, TypeError, RuntimeError, subprocess.SubprocessError):
                    self.failed = True
                self.finalized = True
                envelope = self.signer.sign({"protocol": PROTOCOL, "task": task, "native_binding": deepcopy(native_binding),
                                            "certified": False, "qualifying": qualifying, "reason": reason,
                                            "snapshot_sha256": certificate["manifest"]["sha256"],
                                            "look_index": request["look_index"], "round": request["round"],
                                            "issued_at": self.clock(), "deadline_epoch": self.deadline_epoch,
                                            "selected_artifacts": selected,
                                            "consumption_sha256": sha(certificate["consume_envelope"]["payload"].encode("ascii"))})
                self.certificate["final_envelope"] = envelope
                self._persist()
                return deepcopy(envelope)
        except BaseException:
            self.failed = True
            self._persist()
            raise
