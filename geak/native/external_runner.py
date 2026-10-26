# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a combined external UT inside an already owned evaluation operation.

The host supplies isolation, device mapping, the deadline and final cleanup.
This entry does not allocate GPUs or create an Arena settlement receipt.
"""

import argparse
import contextlib
import os
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from ._util import canonical_json, sha256_hex
from .contract import _decode, _read
from .external import need, read_external_report, validate_external_task

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


# Static-inspection copies only: never scored, never proof that a build ran, and
# taken once after every UT process, so no process timeout or receipt changes.
# One operation can compile ORIGINAL, candidate and helper kernels; all are kept.
COMPILED_SUFFIXES = (".amdgcn", ".ttgir", ".llir", ".hsaco", ".co")
COMPILED_ROOTS = ("TRITON_CACHE_DIR", "TORCH_EXTENSIONS_DIR", "XDG_CACHE_HOME", "AITER_JIT_DIR")
COMPILED_CAPS = {"files": 128, "bytes": 32 * 1024**2, "file_bytes": 8 * 1024**2, "entries": 50_000, "seconds": 10}


def export_compiled(output, work, began_ns, deadline):
    """Copy compiler outputs created after runner start into output/compiled with an INDEX."""
    stop = min(deadline, time.monotonic() + COMPILED_CAPS["seconds"])
    target, rows, omitted, seen, visited, total = output / "compiled", [], [], set(), 0, 0
    try:
        target.mkdir()  # Never write into a directory the task created.
        (target / "files").mkdir()
    except OSError:
        return
    roots = [("work", Path(work))] + [(key, Path(os.environ[key])) for key in COMPILED_ROOTS if key in os.environ]
    try:
        for label, root in roots:
            pending = [""] if root.is_absolute() and not root.is_symlink() else []
            while pending:
                relative_dir = pending.pop()
                entries = []
                with os.scandir(root / relative_dir) as listing:
                    for entry in listing:  # Caps apply while listing, not after.
                        visited += 1
                        if visited > COMPILED_CAPS["entries"] or time.monotonic() >= stop:
                            raise TimeoutError("entry_or_time_cap")
                        entries.append(entry)
                entries.sort(key=lambda entry: entry.name)
                stems = {entry.name[:-len(suffix)] for entry in entries for suffix in COMPILED_SUFFIXES
                         if entry.name.endswith(suffix)}
                for entry in entries:
                    if time.monotonic() >= stop:  # Before every stat, read and copy.
                        raise TimeoutError("entry_or_time_cap")
                    relative = os.path.join(relative_dir, entry.name)
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(relative)
                        continue
                    info = entry.stat(follow_symlinks=False)
                    wanted = entry.name.endswith(COMPILED_SUFFIXES) or (
                        entry.name.endswith(".json") and entry.name[:-5] in stems)
                    if (not wanted or not stat.S_ISREG(info.st_mode) or info.st_ctime_ns < began_ns
                            or (info.st_dev, info.st_ino) in seen):
                        continue
                    code = ("file_cap" if info.st_size > COMPILED_CAPS["file_bytes"]
                            else "count_cap" if len(rows) >= COMPILED_CAPS["files"]
                            else "total_cap" if total + info.st_size > COMPILED_CAPS["bytes"] else None)
                    if code is None:
                        fd = os.open(entry.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                        with os.fdopen(fd, "rb") as stream:
                            raw = stream.read(COMPILED_CAPS["file_bytes"] + 1)
                        code = "changed" if len(raw) != info.st_size else None
                    if code:
                        omitted.append({"root": label, "relative": relative, "code": code})
                        continue
                    path = "files/%03d-%s" % (len(rows), entry.name)
                    with (target / path).open("xb") as stream:
                        stream.write(raw)
                    seen.add((info.st_dev, info.st_ino))
                    total += len(raw)
                    rows.append({"path": path, "root": label, "relative": relative, "size": len(raw),
                                 "sha256": sha256_hex(raw)})
    except Exception as error:  # Evidence never replaces the UT outcome.
        omitted.append({"root": None, "relative": None, "code": type(error).__name__})
    try:
        with (target / "INDEX.json").open("xb") as stream:
            stream.write(canonical_json({"schema": "geak-external-compiled-evidence-v1", "files": rows,
                "omitted": omitted[:COMPILED_CAPS["files"]], "omitted_count": len(omitted),
                "attribution": "created_or_changed_after_runner_start", "execution_proven": False,
                "caps": COMPILED_CAPS}) + b"\n")
    except Exception:
        pass


def shared_input_digest(work, name, protected_root):
    """Check one declared leaf link without following any work-directory links."""
    path, target = (work / name).absolute(), protected_root / name
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    with contextlib.ExitStack() as stack:
        parent = os.open(path.anchor, flags)
        stack.callback(os.close, parent)
        for part in path.parts[1:-1]:
            parent = os.open(part, flags, dir_fd=parent)
            stack.callback(os.close, parent)
        before = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        need(os.readlink(path.name, dir_fd=parent) == str(target), "shared input link changed: " + name)
        digest = sha256_hex(_read(target))
        after = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode")
        need(all(getattr(before, key) == getattr(after, key) for key in fields)
             and os.readlink(path.name, dir_fd=parent) == str(target), "shared input link changed: " + name)
        return digest


def run(request, task, output, *, protected_root=None, shared_files=None):
    """Run three fresh UT processes; optional protected inputs remain separate from candidate sources."""
    contract = validate_external_task(request["contract"])
    source = {row["path"]: row["blob_id"] for row in request["source"]["files"]}
    need(set(source) == set(contract["source_paths"]) and all(source.values()), "candidate source closure differs")
    need(shared_files is None or (type(shared_files) is dict and protected_root is not None),
         "shared files require a protected root and mapping")
    shared = {} if shared_files is None else dict(shared_files)
    need(all(name in contract["artifacts"] and name not in source and contract["artifacts"][name] == digest
             for name, digest in shared.items()), "shared files must match protected task artifacts")
    if shared_files is not None:
        protected_root = Path(protected_root).absolute()
    files = {**contract["artifacts"], **source}
    output.mkdir(parents=True, exist_ok=False)
    work = output / "work"
    for name, digest in files.items():
        input_root = protected_root if protected_root is not None and name not in source else task
        raw = _read(input_root / name)
        need(sha256_hex(raw) == digest, "evaluation source differs: " + name)
        path = work / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name in shared:
            path.symlink_to(protected_root / name)
        else:
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
            record["sources_unchanged"] = all(
                (shared_input_digest(work, name, protected_root) if name in shared
                 else sha256_hex(_read(work / name))) == digest for name, digest in files.items())
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
    parser.add_argument("--protected-root", type=Path)
    parser.add_argument("--shared-files-json", help="validated protected input path-to-digest mapping")
    args = parser.parse_args(argv)
    request, began, began_ns = _decode(_read(args.request)), time.monotonic(), time.time_ns()
    fresh = not os.path.lexists(args.out)
    try:
        return run(request, args.task, args.out, protected_root=args.protected_root,
                   shared_files=_decode(args.shared_files_json) if args.shared_files_json is not None else None)
    finally:
        if fresh and os.path.lexists(args.out / "START.json"):
            try:
                deadline = began + request["contract"]["phases"]["evaluate"]["timeout_seconds"]
            except Exception:
                deadline = began
            export_compiled(args.out, args.out / "work", began_ns, deadline)


if __name__ == "__main__":
    sys.exit(main())
