"""Run a combined external UT inside an already owned evaluation operation.

The host supplies isolation, device mapping, the deadline and final cleanup.
This entry does not allocate GPUs or create an Arena settlement receipt.
"""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import time

from .._util import canonical_json, sha256_hex
from .external import need, read_external_report, validate_external_task
from .jobs import _decode, _read


# Preserve only the installed runtime, declared device, cache and compiler inputs.
# Credentials, proxies, user-home settings and unrelated workflow variables do
# not belong in a task process or its public evidence. Absent values stay absent.
TASK_ENVIRONMENT = (
    "PATH", "LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED",
    "TMPDIR", "LANG", "LC_ALL", "OMP_NUM_THREADS", "MAX_JOBS",
    "TRITON_CACHE_DIR", "TORCH_EXTENSIONS_DIR", "XDG_CACHE_HOME", "AITER_JIT_DIR", "AITER_ROOT_DIR",
    "HARNESS_CACHE_FLUSH_MB", "IMAGE", "CONTAINER_IMAGE_DIGEST",
    "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
    "GEAK_PHYSICAL_DEVICE_MAP", "GEAK_GPU_SYSFS_BDFS", "GEAK_GPU_ALLOWED", "GEAK_NATIVE_GPU_LIST",
    "GPU_ARCHS", "PYTORCH_ROCM_ARCH", "ROCM_HOME", "ROCM_PATH", "HIP_PATH", "CUDA_HOME",
    "CC", "CXX", "CFLAGS", "CXXFLAGS", "HIPFLAGS", "LIBRARY_PATH", "CPATH", "CPLUS_INCLUDE_PATH",
    "PYTORCH_CUDA_ALLOC_CONF", "SGLANG_USE_AITER", "KF_GRID_MULT", "KF_CHUNK_K",
)


def save(path, value):
    path.write_bytes(canonical_json(value) + b"\n")


def run(request, task, output):
    """One evaluation, three fresh UT processes, and an initially empty history."""
    contract = validate_external_task(request["contract"])
    source = {row["path"]: row["blob_id"] for row in request["source"]["files"]}
    need(set(source) == set(contract["source_paths"]) and all(source.values()), "candidate source closure differs")
    files = {**contract["artifacts"], **source}
    output.mkdir(parents=True, exist_ok=False)
    work = output / "work"
    for name, digest in files.items():
        raw = _read(task / name)
        need(sha256_hex(raw) == digest, "evaluation source differs: " + name)
        path = work / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    env = {key: os.environ[key] for key in TASK_ENVIRONMENT if key in os.environ}
    env["GLM_UT_REQUIRE_REPRODUCIBLE_GAIN"] = "0"
    start = {"parent_evaluation": {"operation_id": request["operation_id"],
        "request_sha256": sha256_hex(canonical_json(request)), "binding": request["binding"]},
        "source_files": files, "declared_processes": 3, "gain_required_exit": False,
        "environment": env}
    save(output / "START.json", start)
    deadline = time.monotonic() + contract["phases"]["evaluate"]["timeout_seconds"]
    for index in range(1, 4):
        leg = output / f"process_{index:02d}"
        leg.mkdir()
        record = {"index": index, "started_utc": datetime.now(timezone.utc).isoformat()}
        save(leg / "START.json", record)
        remaining = deadline-time.monotonic()
        need(remaining > 0, "external UT operation deadline elapsed")
        with (leg / "stdout.log").open("xb") as stdout, (leg / "stderr.log").open("xb") as stderr:
            try:
                result = subprocess.run([sys.executable, "-u", "ut_main.py"], cwd=work, env=env,
                    stdout=stdout, stderr=stderr, timeout=remaining, check=False)
                record["returncode"] = result.returncode
            except subprocess.TimeoutExpired:
                record["returncode"] = None
                record["timed_out"] = True
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        artifact_errors = []
        for name in ("ut_result.json", "_runs_v2.jsonl", "stdout.log", "stderr.log"):
            try:
                report = name in ("ut_result.json", "_runs_v2.jsonl")
                raw = _read((work if report else leg) / name)
                if report:
                    with (leg / name).open("xb") as stream:
                        stream.write(raw)
                record[name+"_sha256"] = sha256_hex(raw)
            except (OSError, ValueError) as error:
                artifact_errors.append(name + ": " + str(error))
        if artifact_errors:
            record["artifact_errors"] = artifact_errors
        # Preserve the prior result in its process directory. A failed later
        # process must not inherit that result; the cumulative history stays.
        if (work / "ut_result.json").exists() or (work / "ut_result.json").is_symlink():
            (work / "ut_result.json").unlink()
        try:
            record["sources_unchanged"] = all(sha256_hex(_read(work / name)) == digest
                                                for name, digest in files.items())
        except (OSError, ValueError):
            record["sources_unchanged"] = False
        save(leg / "RECEIPT.json", record)
        if record["returncode"] != 0 or not record["sources_unchanged"] or artifact_errors:
            return 1
    names = ["START.json"] + [f"process_{index:02d}/{name}" for index in range(1, 4)
        for name in ("RECEIPT.json", "ut_result.json", "_runs_v2.jsonl")]
    artifacts = {name: _read(output / name) for name in names}
    observed = read_external_report(request, artifacts)
    save(output / "OBSERVATIONS.json", observed)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--task", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    return run(_decode(_read(args.request)), args.task, args.out)


if __name__ == "__main__":
    sys.exit(main())
