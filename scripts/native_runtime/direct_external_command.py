#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The stock measurement/read ABI inside an already owned native container.

The operator owns the container, GPU grant and its deadline. This command only
captures task bytes, calls the existing complete-UT runner, and reads its raw
evidence. Measurements and stock profiling use the same --gpu-lock-file.
"""

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from geak.native import external
from geak.native._util import canonical_json
from geak.native.contract import job_digest
from geak.native.external import read_external_report, validate_external_task
from geak.native.external_runner import run as run_native
from geak.native.external_runner import shared_input_digest
from geak.native.source import _patched, _read, _snapshot
from kernel_workflow.scripts.task_shared_inputs import load_shared_inputs


def need(value, message):
    if not value:
        raise ValueError(message)


GPU_LOCK_WAIT_SECONDS = 1200
GPU_LOCK_SCRIPT = Path(__file__).resolve().parents[2] / "kernel_workflow/scripts/gpu_lock.sh"


def acquire_gpu_lock(lock):
    # Use stock flock's blocking wait. pass_fds shares the open file description,
    # so this process keeps the lock after the utility exits, until lock closes.
    result = subprocess.run(["flock", "-x", "-w", str(GPU_LOCK_WAIT_SECONDS),
                             str(lock.fileno())], pass_fds=(lock.fileno(),), check=False)
    need(result.returncode == 0,
         f"GPU lock unavailable after {GPU_LOCK_WAIT_SECONDS:g}s; "
         "invoke measurement directly, without gpu_lock.sh")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def emit(value):
    print(canonical_json(value).decode(), flush=True)


def snapshot(workspace, paths):
    first = _snapshot(workspace, paths)
    need(first == _snapshot(workspace, paths), "workspace changed during capture")
    return first


def retain_source(root, captured):
    """Publish only captured editable bytes; protected task inputs stay separate."""
    root.mkdir()
    for name, (raw, mode) in captured.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)
        path.chmod(0o444 | mode)


def identities(contract, epoch, captured):
    # Preserve source/candidate object shapes so existing retained identities
    # stay exact, without importing any archive or search lifecycle.
    source = {"kind": "source", "files": [
        {"path": name, "blob_id": digest(raw), "executable": mode}
        for name, (raw, mode) in sorted(captured.items())]}
    source_id = digest(canonical_json(source))
    candidate = {"kind": "candidate", "job_id": job_digest(contract),
                 "measurement_epoch_id": epoch, "source_id": source_id,
                 "parents": [], "donors": [], "contribution_claims": []}
    return source, source_id, digest(canonical_json(candidate))


def check_protected(contract, workspace, task_root=None, shared=()):
    for name, expected in contract["artifacts"].items():
        if name not in contract["source_paths"]:
            observed = (shared_input_digest(workspace, name, task_root)
                        if name in shared and (workspace / name).is_symlink()
                        else digest(_read(workspace, name)[0]))
            need(observed == expected,
                 "protected task artifact changed: " + name)


def create_request(contract, task_root, epoch, evaluation_id, captured, workspace=None):
    source, source_id, candidate_id = identities(contract, epoch, captured)
    return {"schema": "geak-arena-evaluation-request-v1", "operation_id": evaluation_id,
            "binding": {"job_id": job_digest(contract), "measurement_epoch_id": epoch,
                        "candidate_id": candidate_id, "candidate_source_id": source_id},
            "evaluator_sha256": digest(Path(external.__file__).read_bytes()),
            "contract": contract, "artifact_root": str(task_root), "source": source,
            "source_objects": {row["blob_id"]: str((workspace or task_root) / row["path"])
                               for row in source["files"]},
            "correctness_case_ids": contract["correctness_case_ids"],
            "benchmark_case_ids": contract["benchmark_case_ids"],
            "benchmark_report_ids": contract["benchmark_report_ids"],
            "phases": contract["phases"], "measurement_valid": False,
            "custody_proven": False, "reward_eligible": False}


def check_request(request, contract, epoch, evaluation_id, candidate_id):
    need(request["schema"] == "geak-arena-evaluation-request-v1"
         and request["contract"] == contract and request["operation_id"] == evaluation_id
         and request["binding"]["job_id"] == job_digest(contract)
         and request["binding"]["measurement_epoch_id"] == epoch
         and request["binding"]["candidate_id"] == candidate_id,
         "evaluation request identity or task differs")
    need(request["evaluator_sha256"] == digest(Path(external.__file__).read_bytes()),
         "native reader changed since capture")


def checked_result(request, native_root, workspace, patch=None):
    paths = request["contract"]["source_paths"]
    before = snapshot(workspace, paths)
    patch_bytes = _read(patch.parent, patch.name) if patch is not None else None
    candidate = _patched(before, patch_bytes[0]) if patch_bytes is not None else before
    source, source_id, candidate_id = identities(
        request["contract"], request["binding"]["measurement_epoch_id"], candidate)
    workspace_source_id = (identities(request["contract"], request["binding"]["measurement_epoch_id"], before)[1]
                           if patch_bytes is not None else source_id)
    need(source == request["source"] and source_id == request["binding"]["candidate_source_id"]
         and candidate_id == request["binding"]["candidate_id"],
         "workspace or patch differs from measured source")
    names = ["START.json"] + [f"process_{index:02d}/{name}" for index in range(1, 4)
        for name in ("RECEIPT.json", "ut_result.json", "_runs_v2.jsonl")]
    raw = {name: _read(native_root, name)[0] for name in names}
    observed = read_external_report(request, raw)
    need(snapshot(workspace, paths) == before, "workspace changed while reading evidence")
    if patch is not None:
        need(_read(patch.parent, patch.name) == patch_bytes, "patch changed while reading evidence")
    pairs = observed["paired_cases"]
    per_case = [{"name": name, **pairs[name]} for name in request["contract"]["benchmark_case_ids"]]
    speedups = [row["speedup"] for row in per_case]
    need(speedups and all(type(v) in (int, float) and math.isfinite(v) and v > 0
                         for v in speedups), "invalid checked case speedup")
    return {"objective": {"kind": "external_primary", "primary_regime": observed["primary_regime"]},
            **{key: observed[key] for key in ("primary_regime", "primary_speedup",
                 "native_reproducible_gain", "native_reproducible_regression", "phase_speedups")},
            "per_case": per_case,
            "development_geomean_speedup": math.exp(sum(map(math.log, speedups)) / len(speedups)),
            "external_evaluation": {"candidate_id": candidate_id, "evaluation_id": request["operation_id"]},
            "source_id": source_id, "workspace_source_id": workspace_source_id, "source_paths": list(paths),
            "patch_sha256": digest(patch_bytes[0]) if patch_bytes is not None else None,
            "receipt_ref": {"path": str(native_root / "START.json"), "sha256": digest(raw["START.json"])},
            "timing_basis": "external_whole_call_ut", "external_measurement_valid": True,
            "measurement_valid": True, "correctness": "pass", "status": "verified",
            "validation_status": "accepted" if observed["native_reproducible_gain"] else "no_improvement"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", required=True, type=Path)
    parser.add_argument("--state-root", required=True, type=Path)
    gpu = parser.add_mutually_exclusive_group(required=True)
    gpu.add_argument("--gpu-lock-file", type=Path)
    gpu.add_argument("--gpu-pool", help="allocated local GPU IDs, using the stock GPU lock wrapper")
    parser.add_argument("--epoch", default="direct-native-v1")
    parser.add_argument("--shared-input-manifest", type=Path,
                        default=os.environ.get("GEAK_SHARED_INPUT_MANIFEST"))
    parser.add_argument("--shared-input-sha256", default=os.environ.get("GEAK_SHARED_INPUT_SHA256"))
    parser.add_argument("action", choices=("measure", "read"))
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--candidate-id")
    parser.add_argument("--evaluation-id")
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument("--captured", action="store_true",
                        help="measure/read the retained source for an existing candidate/evaluation pair")
    parser.add_argument("--patch", type=Path)
    args = parser.parse_args(argv)
    need(bool(args.candidate_id) == bool(args.evaluation_id), "candidate and evaluation IDs must be paired")
    need(args.action != "read" or (args.candidate_id and not args.capture_only), "read requires captured IDs")
    need(args.patch is None or args.action == "read", "patch is only valid for read")
    need(not args.capture_only or not args.candidate_id, "capture allocates a fresh evaluation identity")
    need((args.captured and args.candidate_id and args.workspace is None
          and args.patch is None and not args.capture_only)
         or (not args.captured and args.workspace is not None),
         "use --workspace, or --captured with paired IDs and no workspace/patch")
    need(bool(args.epoch.strip()), "measurement epoch is required")
    need(args.gpu_pool is None or re.fullmatch(r"[0-9]+(?:,[0-9]+)*", args.gpu_pool),
         "GPU pool must be a comma-separated list of allocated local IDs")
    need(bool(args.shared_input_manifest) == bool(args.shared_input_sha256),
         "shared-input manifest and SHA256 must be supplied together")
    task_root = args.task_root.absolute()
    contract = validate_external_task(json.loads(_read(task_root, "task.json")[0]), task_root)
    shared = None
    if args.shared_input_manifest is not None:
        shared = load_shared_inputs(args.shared_input_manifest, args.shared_input_sha256, task_root, contract)
    workspace = args.workspace.absolute() if args.workspace is not None else None
    if workspace is not None:
        check_protected(contract, workspace, task_root, shared["shared_files"] if shared else ())
    state = args.state_root.absolute()
    if args.action == "measure":
        state.mkdir(parents=True, exist_ok=True, mode=0o700)
    need(state.resolve(strict=True) == state, "state root must be canonical")
    evaluations = state / "evaluations"
    if args.action == "measure":
        evaluations.mkdir(exist_ok=True)
    need(evaluations.resolve(strict=True) == evaluations, "evaluation root must be canonical")
    evaluation_id = args.evaluation_id or "native-" + uuid.uuid4().hex
    need(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", evaluation_id), "invalid evaluation ID")
    root = evaluations / evaluation_id
    source_root = root / "source"
    if args.evaluation_id:
        request = json.loads(_read(root, "request.json")[0])
        check_request(request, contract, args.epoch, evaluation_id, args.candidate_id)
    else:
        captured = snapshot(workspace, contract["source_paths"])
        root.mkdir()
        retain_source(source_root, captured)
        request = create_request(contract, task_root, args.epoch, evaluation_id, captured, source_root)
        with (root / "request.json").open("xb") as stream:
            stream.write(canonical_json(request) + b"\n")
    binding = request["binding"]
    retained_source = request["source_objects"] == {row["blob_id"]: str(source_root / row["path"])
        for row in request["source"]["files"]}
    identity = {"candidate_id": binding["candidate_id"], "evaluation_id": evaluation_id,
                "source_id": binding["candidate_source_id"], "source_paths": list(contract["source_paths"])}
    if retained_source:
        identity["source_workspace"] = str(source_root)
    if args.captured:
        need(retained_source, "evaluation has no retained source snapshot")
        workspace = source_root
    if args.capture_only:
        emit({"schema": "geak-native-measurement-identity-v1", "capture_only": True, **identity})
        return 0
    native_root = root / "native"
    if args.action == "measure":
        source, source_id, candidate_id = identities(contract, args.epoch,
            snapshot(workspace, contract["source_paths"]))
        need(source == request["source"] and source_id == binding["candidate_source_id"]
             and candidate_id == binding["candidate_id"], "workspace changed since source capture")
        emit({"schema": "geak-native-measurement-identity-v1", **identity})
        if args.gpu_pool is not None:
            if not native_root.exists():
                # The stock wrapper chooses and locks a GPU once for the full UT.
                # The runner itself has no GPU lock, so profiling shares the same
                # lock namespace without recursive acquisition.
                package_root = str(Path(external.__file__).resolve().parents[2])
                env = {**os.environ, "PYTHONPATH": os.pathsep.join(
                    filter(None, (package_root, os.environ.get("PYTHONPATH")))),
                    # These caches belong to this immutable evaluation, so the
                    # wrapper's optional workspace defaults must preserve them.
                    "GEAK_ISOLATE_WORKSPACE_CACHES": "0"}
                for key, name in (("TORCH_EXTENSIONS_DIR", "torch_extensions"), ("TRITON_CACHE_DIR", "triton"),
                                  ("XDG_CACHE_HOME", "xdg"), ("AITER_JIT_DIR", "aiter")):
                    cache = root / "cache" / name
                    cache.mkdir(parents=True, exist_ok=True)
                    env[key] = str(cache)
                result = subprocess.run(["bash", str(GPU_LOCK_SCRIPT), args.gpu_pool,
                    sys.executable, "-m", "geak.native.external_runner", "--request", str(root / "request.json"),
                    "--task", str(workspace), "--out", str(native_root), "--protected-root", str(task_root),
                    "--shared-files-json", json.dumps(shared["shared_files"] if shared else {})],
                    cwd=workspace, env=env, check=False)
                need(result.returncode == 0, "native UT failed; retained operation cannot be silently rerun")
        else:
            # Existing pinned-GPU command: own the lock here, without an outer wrapper.
            lock_path = args.gpu_lock_file.absolute()
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with lock_path.open("a+b") as lock:
                acquire_gpu_lock(lock)
                if not native_root.exists():
                    options = ({"protected_root": task_root, "shared_files": shared["shared_files"] if shared else {}}
                               if args.captured or shared else {})
                    rc = run_native(request, workspace, native_root, **options)
                    need(rc == 0, "native UT failed; retained operation cannot be silently rerun")
        value = checked_result(request, native_root, workspace)
    else:
        value = checked_result(request, native_root, workspace,
                               args.patch.absolute() if args.patch is not None else None)
    if args.captured:
        value["source_workspace"] = str(source_root)
    emit(value)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
