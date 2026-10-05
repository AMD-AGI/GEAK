# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bind one fresh process from the frozen paired scorer to host evidence.

The host must supply a qualified runner. No default runner exposes a GPU or
executes candidate Python in the recorder process. The frozen scorer imports
the candidate into its interpreter. This adapter does not claim resistance to
arbitrary in-process monkeypatching. Exact-source review, allowed-file scope,
and immutable harness bindings remain conditions of a qualifying result.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .quality_stop_controller import ProcessEvaluator, StopRejected, canonical, require

SCORER_SHA256 = "57ac7eae7c4572f85392e6372cd98dbdc17a460dc9c8fc2d640b0c098964f068"
# All six frozen R results share this contract. It binds the exact tolerance,
# random/eager counts, preserved exclusions, report groups, and case order.
# Timing values, error magnitudes, and correctness speedups do not enter it.
CORRECTNESS_CONTRACT_SHA256 = "170b5199918530eaaa65d1cffbc260e1c934cf499297372352e2457dc5e9dd25"
IMAGE_ID = "sha256:1b87d6d12bb329f2314467a61569d85225eafbb93603fef1b43c44e64b87ba06"
IMAGE_TAG = "lmsysorg/sglang-rocm:v0.5.18-rocm724-mi35x-20260825"
PROCESS_TIMEOUT_SECONDS = 600
CALLS = {
    "qkv_proj_decode_M1": 1, "qkv_proj_decode_M64": 1024, "qkv_proj_prefill_M1024": 64,
    "o_proj_decode_M1": 1024, "o_proj_prefill_M1024": 64,
    "router_gate_decode_M1": 1, "router_gate_decode_M64": 1024, "router_gate_prefill_M1024": 64,
}
ORDER_ARGUMENTS = {"reference_first": "ref-first", "candidate_first": "candidate-first"}
REQUIRED_TASK_FILES = frozenset(("unittest.py", "harness_lib.py", "meta.json", "workload.json",
                               "baseline_ms.json", "reference_io.pt", "baseline_ref/gemm_kernels.py.orig"))
MAX_RESULT_BYTES = 16 * 1024 * 1024
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path(value):
    path = Path(value)
    require(path.is_absolute() and path.resolve() == path, "evaluator_path_not_canonical")
    return path


def _relative(value):
    require(isinstance(value, str), "evaluator_relative_path_invalid")
    path = PurePosixPath(value)
    require(value not in {"", "."} and str(path) == value and not path.is_absolute()
            and ".." not in path.parts, "evaluator_relative_path_invalid")
    return value


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _same_number(value, expected):
    return _number(value) and math.isclose(value, expected, rel_tol=1e-12, abs_tol=0)


def _tree(root):
    root = _path(root)
    require(root.is_dir(), "evaluator_source_directory_missing")
    files = {}
    for directory, folders, names in os.walk(root, followlinks=False):
        for name in folders:
            require(not (Path(directory) / name).is_symlink(), "evaluator_source_link")
        for name in names:
            path = Path(directory) / name
            info = path.lstat()
            require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and path.resolve() == path,
                    "evaluator_source_not_regular")
            files[path.relative_to(root).as_posix()] = {
                "sha256": _sha(path), "bytes": info.st_size,
                "mode": "100755" if info.st_mode & 0o111 else "100644",
            }
    require(files, "evaluator_source_empty")
    return dict(sorted(files.items()))


def _json(raw):
    require(isinstance(raw, bytes) and 0 < len(raw) <= MAX_RESULT_BYTES, "paired_result_size_invalid")
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "paired_duplicate_json_key")
            result[key] = value
        return result
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
        canonical(value)
    except (ValueError, UnicodeError, TypeError):
        raise StopRejected("paired_result_json_invalid") from None
    require(isinstance(value, dict), "paired_result_not_object")
    return value


def _task_scorer_hashes(hashes):
    names = REQUIRED_TASK_FILES | {name for name in hashes
        if name.startswith(("baseline_ref/", "baseline_overlay/", "_cand_overlay/"))
        and "__pycache__" not in PurePosixPath(name).parts and not name.endswith(".pyc")}
    return {name: hashes[name] for name in sorted(names)}


def decode_paired(raw, *, request, child_started_unix=None, child_ended_unix=None):
    """Check scorer records without granting source or runtime authority."""
    value = _json(raw)
    for field in ("image", "source_hashes", "reference_binding", "candidate_binding", "correctness", "timing_policy", "score"):
        require(isinstance(value.get(field), dict), "paired_result_schema_invalid")
    require(value.get("schema") == "geak.gptoss.paired_timing/1" and value.get("status") == "pass"
            and "error" not in value, "paired_quality_failed")
    require(value.get("order") == request["scorer_order"] and value.get("input_seed") == 7
            and type(value.get("input_seed")) is int, "paired_order_or_seed_changed")
    require(value.get("image", {}).get("id") == IMAGE_ID
            and value.get("image", {}).get("tag") == IMAGE_TAG, "paired_image_changed")
    require(value.get("task_dir") == request["actor_task_dir"]
            and value.get("candidate_source") == request["actor_candidate_source"], "paired_source_path_changed")
    expected_hashes = {"scorer": request["scorer_sha256"], "frozen_task": request["scorer_task_hashes"],
                       "candidate": request["candidate_hashes"]}
    require(value.get("source_hashes") == expected_hashes, "paired_source_hashes_changed")
    reference = value.get("reference_binding", {})
    require(reference.get("module") == "aiter.ops.flydsl.gemm_kernels"
            and reference.get("qualname") == "flydsl_hgemm"
            and reference.get("sha256") == request["reference_sha256"]
            and reference.get("snapshot_source_sha256") == request["reference_sha256"]
            and reference.get("source") == request["actor_reference_source"], "paired_reference_changed")
    candidate = value.get("candidate_binding", {})
    require(candidate == {"module": "geak_paired_candidate", "qualname": "flydsl_hgemm",
                          "source": request["actor_candidate_source"]}, "paired_candidate_binding_changed")
    correctness = value.get("correctness", {})
    require(correctness.get("passed") is True, "paired_correctness_failed")
    reports = correctness.get("report", {})
    require(isinstance(reports, dict) and all(isinstance(rows, list) and rows
                and all(isinstance(row, dict) and row.get("correct") is True for row in rows)
                for rows in reports.values()) and reports, "paired_correctness_report_invalid")
    require(set(correctness) == {"passed", "tolerance", "random_draws", "eager_cases", "report", "preserved_oracle_exclusions"},
            "paired_correctness_schema_changed")
    contract = {name: correctness.get(name) for name in
                ("tolerance", "random_draws", "eager_cases", "preserved_oracle_exclusions")}
    contract["report_cases"] = {name: [row.get("case") for row in rows] for name, rows in reports.items()}
    require(hashlib.sha256(canonical(contract).encode("ascii")).hexdigest() == CORRECTNESS_CONTRACT_SHA256,
            "paired_correctness_contract_changed")
    policy = value.get("timing_policy", {})
    for name, expected in {"function": "frozen harness_lib.time_op", "graph": True, "warmup": 10,
                           "repeats": 50, "inner": 1, "flush_cache": True,
                           "cache_flush_mb": 512, "detail": True, "raw_event_samples_available": False}.items():
        require(type(policy.get(name)) is type(expected) and policy[name] == expected, "paired_timing_policy_changed")
    started, ended = value.get("started_unix"), value.get("finished_unix")
    require(_positive(started) and _positive(ended) and ended >= started, "paired_clock_invalid")
    if child_started_unix is not None:
        require(started >= child_started_unix, "paired_record_not_fresh")
    if child_ended_unix is not None:
        require(ended <= child_ended_unix, "paired_record_after_exit")
    score = value.get("score", {})
    rows = score.get("per_case", [])
    require(isinstance(rows, list) and len(rows) == 8 and all(isinstance(row, dict) for row in rows)
            and [row.get("sig") for row in rows] == list(CALLS)
            and type(score.get("included_buckets")) is int and score["included_buckets"] == 8,
            "paired_bucket_set_changed")
    buckets, reference_terms, candidate_terms = [], [], []
    for row in rows:
        name = row["sig"]
        require(row.get("included") is True and type(row.get("calls")) is int
                and row["calls"] == request["call_weights"][name] == CALLS[name], "paired_foreign_call_weight")
        reference_ms, candidate_ms = row.get("baseline_ms"), row.get("optimized_ms")
        require(_positive(reference_ms) and _positive(candidate_ms), "paired_latency_invalid")
        require(_same_number(row.get("speedup"), reference_ms / candidate_ms)
                and _same_number(row.get("weight"), CALLS[name] * reference_ms), "paired_bucket_score_changed")
        require(row.get("measured_tie") is (reference_ms == candidate_ms), "paired_tie_policy_changed")
        buckets.append({"bucket_id": name, "reference_ms": reference_ms, "candidate_ms": candidate_ms})
        reference_terms.append(CALLS[name] * reference_ms)
        candidate_terms.append(CALLS[name] * candidate_ms)
    ref_total, cand_total = math.fsum(reference_terms), math.fsum(candidate_terms)
    require(_same_number(score.get("reference_lifecycle_ms"), ref_total)
            and _same_number(score.get("candidate_lifecycle_ms"), cand_total)
            and _same_number(score.get("weighted_speedup"), ref_total / cand_total), "paired_aggregate_score_changed")
    observations = value.get("observations", [])
    legs = ("reference", "candidate") if request["scorer_order"] == "ref-first" else ("candidate", "reference")
    expected_order = [(name, leg) for name in CALLS for leg in legs]
    require(isinstance(observations, list) and len(observations) == 16
            and all(isinstance(row, dict) for row in observations)
            and [(row.get("sig"), row.get("leg")) for row in observations] == expected_order,
            "paired_timing_leg_order_changed")
    by_name = {row["bucket_id"]: row for row in buckets}
    prior_time, unprimed = started, []
    for row in observations:
        when, receipt = row.get("started_unix"), row.get("receipt", {})
        require(_number(when) and prior_time <= when <= ended, "paired_observation_clock_invalid")
        prior_time = when
        require(isinstance(receipt, dict) and receipt.get("timer") == "cuda_event_graph"
                and _positive(receipt.get("ms")) and type(receipt.get("primed")) is bool,
                "paired_timing_receipt_invalid")
        field = "reference_ms" if row["leg"] == "reference" else "candidate_ms"
        require(receipt["ms"] == by_name[row["sig"]][field], "paired_latency_receipt_changed")
        if receipt["primed"] is not True:
            unprimed.append({"sig": row["sig"], "leg": row["leg"]})
    require(value.get("unprimed_observations") == unprimed
            and value.get("device_only_interpretation_supported") is (not unprimed), "paired_priming_summary_changed")
    inputs = value.get("timed_input_hashes", {})
    require(isinstance(inputs, dict) and set(inputs) == set(CALLS), "paired_input_hashes_missing")
    for tensors in inputs.values():
        require(isinstance(tensors, dict) and tensors, "paired_input_hashes_invalid")
        for tensor in tensors.values():
            require(isinstance(tensor, dict) and isinstance(tensor.get("sha256"), str)
                    and _SHA.fullmatch(tensor["sha256"]) and isinstance(tensor.get("dtype"), str)
                    and isinstance(tensor.get("shape"), list)
                    and all(type(size) is int and size >= 0 for size in tensor["shape"]), "paired_input_hashes_invalid")
    return buckets


@dataclass(frozen=True)
class LaunchPlan:
    """Bind the actual host child command to the protected scorer launch."""

    argv: tuple[str, ...]
    environment: tuple[tuple[str, str], ...]
    cwd: str
    runtime_id: str
    details: dict

    def record(self):
        require(isinstance(self.argv, tuple) and self.argv and all(isinstance(value, str) for value in self.argv),
                "evaluator_launch_argv_invalid")
        environment = dict(self.environment)
        require(len(environment) == len(self.environment) and all(isinstance(key, str) and isinstance(value, str)
                for key, value in environment.items()), "evaluator_launch_environment_invalid")
        require(isinstance(self.runtime_id, str) and self.runtime_id and isinstance(self.details, dict)
                and self.details, "evaluator_runtime_binding_missing")
        value = {"argv": list(self.argv), "environment": environment, "cwd": str(_path(self.cwd)),
                 "runtime_id": self.runtime_id, "details": deepcopy(self.details)}
        canonical(value)
        return value


class ProtectedPairedRunner(ABC):
    """Supply reviewed OS controls, never a model-authored result or boolean.

    Prepare one fresh owned runtime with read-only snapshot, task, and scorer
    mounts. Keep the host recorder, signing state, receipts, and authentication
    outside that runtime. The scorer command/environment must equal request.
    Start returns the actual owned host Popen object. Attest must inspect its
    real namespaces, mounts, command, environment, and container state.

    At stage ``stopped``, attest must prove the owned container and every child
    stopped, independently of the docker/client process. Stop must terminate
    that exact runtime. Unknown cleanup blocks every later process admission.

    A GPU implementation must share the trial's exclusive reservation owner.
    It must not reacquire a lock still held by the paused actor. It must verify
    actor producer/context/queue closure before admitting the evaluator. The
    reservation and device-admission exclusion continue until cleanup closes.
    R's writable /workspace/controller mount cannot implement this contract.
    """

    @abstractmethod
    def prepare(self, request, *, record_dir):
        """Return a LaunchPlan before starting any scorer process."""

    @abstractmethod
    def start(self, request, plan, *, stdout, stderr):
        """Return the actual subprocess.Popen with host-owned output streams."""

    @abstractmethod
    def attest(self, request, plan, process, *, stage):
        """Check OS evidence, then return its complete JSON-safe host record."""

    @abstractmethod
    def stop(self, request, plan, process):
        """Stop the exact owned runtime and its descendants without retrying."""


class FrozenPairedEvaluator(ProcessEvaluator):
    """Run one unique paired process and preserve every attempted outcome."""

    def __init__(self, *, state_dir, scorer_path, task_dir, task_hashes, call_weights, runner,
                 environment, actor_reference_source="/sgl-workspace/aiter/aiter/ops/flydsl/gemm_kernels.py"):
        require(isinstance(runner, ProtectedPairedRunner), "trusted_paired_runner_required")
        require(call_weights == CALLS and all(type(value) is int for value in call_weights.values()),
                "evaluator_foreign_call_weights")
        self.scorer_path, self.task_dir, self.state_dir = _path(scorer_path), _path(task_dir), _path(state_dir)
        require(_sha(self.scorer_path) == SCORER_SHA256, "evaluator_frozen_scorer_changed")
        require(isinstance(task_hashes, dict) and REQUIRED_TASK_FILES <= set(task_hashes)
                and all(_relative(name) and isinstance(digest, str) and _SHA.fullmatch(digest)
                        for name, digest in task_hashes.items()), "evaluator_task_manifest_invalid")
        self.task_hashes = dict(sorted(task_hashes.items()))
        self.task_tree = _tree(self.task_dir)
        require({name: row["sha256"] for name, row in self.task_tree.items()} == self.task_hashes,
                "evaluator_frozen_task_changed")
        require(self.state_dir != self.task_dir and self.task_dir not in self.state_dir.parents
                and self.state_dir not in self.task_dir.parents, "evaluator_state_inside_task")
        require(isinstance(environment, dict) and all(isinstance(key, str) and isinstance(value, str)
                for key, value in environment.items()), "evaluator_environment_invalid")
        for key, value in {"GEAK_FREEZE_BASELINE": "0", "HARNESS_CACHE_FLUSH_MB": "512",
                           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}.items():
            require(environment.get(key) == value, "evaluator_timing_environment_changed")
        require(environment.get("HOME") == os.environ.get("HOME"), "evaluator_home_value_changed")
        require(not any(key.startswith(("ANTHROPIC_", "OPENAI_")) for key in environment), "evaluator_provider_environment_forbidden")
        require(isinstance(actor_reference_source, str) and actor_reference_source.startswith("/"),
                "evaluator_reference_path_invalid")
        self.environment, self.actor_reference_source = deepcopy(environment), actor_reference_source
        self.runner, self.lock, self.blocked = runner, threading.RLock(), False
        self.state_dir.mkdir(mode=0o700, parents=False, exist_ok=False)

    @staticmethod
    def _write(path, value):
        data = canonical(value).encode("ascii") + b"\n"
        temporary = path.with_name(path.name + ".tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def _verify_sources(self, request):
        require(_sha(self.scorer_path) == request["scorer_sha256"]
                and _tree(self.task_dir) == self.task_tree
                and _tree(Path(request["snapshot"])) == request["snapshot_tree"], "paired_sources_changed")

    def _attest(self, request, plan, process, *, stage):
        evidence = self.runner.attest(deepcopy(request), plan, process, stage=stage)
        require(isinstance(evidence, dict) and evidence and evidence.get("stage") == stage
                and evidence.get("runtime_id") == plan.runtime_id
                and evidence.get("host_pid") == (process.pid if process is not None else None)
                and evidence.get("request_sha256") == hashlib.sha256(canonical(request).encode()).hexdigest()
                and isinstance(evidence.get("os_evidence"), dict) and evidence["os_evidence"],
                "paired_runtime_evidence_missing")
        require(evidence.get("source_protection") == {"snapshot": "read_only", "task": "read_only",
                    "scorer": "read_only", "recorder": "unmounted", "host_controller": "unmounted"},
                "paired_runtime_protection_missing")
        if stage == "stopped":
            require(evidence.get("runtime_stopped") is True and evidence.get("active_descendants") == []
                    and (process is None or type(evidence.get("runtime_exit_code")) is int),
                    "paired_runtime_stop_unconfirmed")
        canonical(evidence)
        return evidence

    def run_process(self, *, snapshot, snapshot_sha256, process_id, order):
        with self.lock:
            require(not self.blocked, "paired_cleanup_unknown_blocks_admission")
            require(isinstance(process_id, str) and 1 <= len(process_id) <= 256 and process_id.isascii()
                    and all(32 <= ord(char) < 127 for char in process_id), "paired_process_identity_invalid")
            require(order in ORDER_ARGUMENTS and isinstance(snapshot_sha256, str) and _SHA.fullmatch(snapshot_sha256),
                    "paired_request_binding_invalid")
            directory = self.state_dir / hashlib.sha256(process_id.encode("ascii")).hexdigest()
            try:
                directory.mkdir(mode=0o700)
            except FileExistsError:
                raise StopRejected("paired_process_already_attempted") from None
            receipt = {"schema": "geak.fixed_floor.paired_process/1", "process_id": process_id,
                       "status": "started", "order": order, "snapshot_sha256": snapshot_sha256,
                       "started_unix": time.time(), "started_monotonic_ns": time.monotonic_ns(),
                       "timeout_seconds": PROCESS_TIMEOUT_SECONDS, "cleanup": "not_started"}
            self._write(directory / "receipt.json", receipt)
            request = plan = process = None
            stopped = False
            try:
                snapshot = _path(snapshot)
                require(snapshot != self.state_dir and self.state_dir not in snapshot.parents
                        and snapshot not in self.state_dir.parents, "paired_snapshot_inside_recorder")
                tree = _tree(snapshot)
                candidates = [name for name in ("kernel_src/flydsl_hgemm_impl.py", "flydsl_hgemm_impl.py") if name in tree]
                require(len(candidates) == 1, "paired_candidate_source_ambiguous")
                source = candidates[0]
                parent = PurePosixPath(source).parent
                candidate_hashes = {str(PurePosixPath(name).relative_to(parent)): row["sha256"]
                    for name, row in tree.items() if PurePosixPath(name).is_relative_to(parent)
                    and "__pycache__" not in PurePosixPath(name).parts and not name.endswith(".pyc")}
                request = {"process_id": process_id, "snapshot": str(snapshot), "snapshot_sha256": snapshot_sha256,
                    "snapshot_tree": tree, "snapshot_tree_sha256": hashlib.sha256(canonical(tree).encode()).hexdigest(),
                    "scorer_path": str(self.scorer_path), "scorer_sha256": SCORER_SHA256,
                    "task_dir": str(self.task_dir), "task_hashes": self.task_hashes,
                    "scorer_task_hashes": _task_scorer_hashes(self.task_hashes), "candidate_hashes": candidate_hashes,
                    "candidate_relative_path": source, "reference_sha256": self.task_hashes["baseline_ref/gemm_kernels.py.orig"],
                    "actor_task_dir": "/frozen_task", "actor_candidate_source": "/snapshot/" + source,
                    "actor_reference_source": self.actor_reference_source, "order": order, "scorer_order": ORDER_ARGUMENTS[order],
                    "call_weights": dict(CALLS), "image_id": IMAGE_ID, "timeout_seconds": PROCESS_TIMEOUT_SECONDS,
                    "environment": self.environment,
                    "argv": ["python3", "-B", "/study_runtime/paired_timing.py", "--task-dir", "/frozen_task",
                             "--candidate-dir", "/snapshot", "--order", ORDER_ARGUMENTS[order], "--input-seed", "7",
                             "--image", IMAGE_TAG, "--image-id", IMAGE_ID]}
                self._write(directory / "request.json", request)
                receipt["request_sha256"] = _sha(directory / "request.json")
                self._verify_sources(request)
                receipt["cleanup"] = "prepare_pending"
                self._write(directory / "receipt.json", receipt)
                plan = self.runner.prepare(deepcopy(request), record_dir=directory)
                require(isinstance(plan, LaunchPlan), "paired_launch_plan_missing")
                receipt["launch"] = plan.record()
                self._write(directory / "receipt.json", receipt)
                with (directory / "stdout.json").open("xb") as stdout, (directory / "stderr.log").open("xb") as stderr:
                    receipt["child_started_unix"] = time.time()
                    child_started = time.monotonic()
                    receipt["child_started_monotonic"] = child_started
                    process = self.runner.start(deepcopy(request), plan, stdout=stdout, stderr=stderr)
                    require(isinstance(process, subprocess.Popen) and list(process.args) == list(plan.argv),
                            "paired_owned_process_missing")
                    receipt["host_pid"] = process.pid
                    receipt["running_evidence"] = self._attest(request, plan, process, stage="running")
                    self._write(directory / "receipt.json", receipt)
                    try:
                        remaining = PROCESS_TIMEOUT_SECONDS - (time.monotonic() - child_started)
                        if remaining <= 0:
                            raise subprocess.TimeoutExpired(plan.argv, PROCESS_TIMEOUT_SECONDS)
                        exit_code = process.wait(timeout=remaining)
                    except subprocess.TimeoutExpired:
                        receipt["status"] = "unknown"
                        receipt["reason"] = "paired_process_timeout"
                        raise StopRejected("paired_process_timeout") from None
                    receipt["child_ended_unix"] = time.time()
                    receipt["child_elapsed_seconds"] = time.monotonic() - child_started
                    receipt["actual_host_exit_code"] = exit_code
                    receipt["stopped_evidence"] = self._attest(request, plan, process, stage="stopped")
                    stopped = True
                    receipt["cleanup"] = "verified_stopped"
                    require(type(exit_code) is int and exit_code == 0 and process.returncode == 0
                            and receipt["stopped_evidence"]["runtime_exit_code"] == 0, "paired_process_exit_failed")
                    require(receipt["child_elapsed_seconds"] <= PROCESS_TIMEOUT_SECONDS, "paired_process_timeout")
                self._verify_sources(request)
                raw = (directory / "stdout.json").read_bytes()
                receipt["raw_json_sha256"] = hashlib.sha256(raw).hexdigest()
                receipt["raw_json_bytes"] = len(raw)
                buckets = decode_paired(raw, request=request, child_started_unix=receipt["child_started_unix"],
                                        child_ended_unix=receipt["child_ended_unix"])
                result = {"process_id": process_id, "snapshot_sha256": snapshot_sha256, "order": order, "buckets": buckets}
                receipt.update(status="complete", result=result)
                return deepcopy(result)
            except BaseException as error:
                receipt["status"] = "unknown" if receipt.get("status") == "unknown" else "invalid"
                receipt["reason"] = str(error) if isinstance(error, StopRejected) else "paired_host_operation_failed"
                if plan is None and receipt["cleanup"] == "prepare_pending":
                    self.blocked = True
                    receipt.update(status="unknown", cleanup="unknown", reason="paired_prepare_outcome_unknown")
                raise
            finally:
                if plan is not None and not stopped:
                    try:
                        self.runner.stop(deepcopy(request), plan, process)
                        if process is not None:
                            process.wait(timeout=20)
                        receipt["stopped_evidence"] = self._attest(request, plan, process, stage="stopped")
                        receipt["cleanup"] = "verified_stopped"
                    except BaseException:
                        self.blocked = True
                        receipt.update(status="unknown", cleanup="unknown", reason="paired_cleanup_unknown")
                receipt["ended_unix"] = time.time()
                receipt["ended_monotonic_ns"] = time.monotonic_ns()
                for name in ("stdout.json", "stderr.log"):
                    path = directory / name
                    if path.is_file():
                        receipt[name] = {"sha256": _sha(path), "bytes": path.stat().st_size}
                self._write(directory / "receipt.json", receipt)
