#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the existing task shared-input manifest for a relocated read-only task.

The manifest's task_root records its producer location. The explicit task_root
argument names that same pinned task in this launch (normally /task); artifact
hashes bind the two. Only exact manifest-listed fixture leaves may be linked into
a private workspace; candidate source is never shared.
"""

import argparse
import hashlib
import os
import stat
import sys
from pathlib import Path

# Materialization can run without PYTHONPATH. Append the workflow fallback so an
# explicitly selected native /code package keeps precedence in the adapter.
sys.path.append(str(Path(__file__).resolve().parents[2]))
from geak.native.contract import _decode, _path, _read, _sha
from geak.native.external import need, validate_external_task


def load_shared_inputs(manifest_path, expected_sha256, task_root, contract=None):
    _sha(expected_sha256)
    raw = _read(Path(manifest_path))
    need(hashlib.sha256(raw).hexdigest() == expected_sha256, "shared-input manifest changed")
    manifest = _decode(raw)
    need(type(manifest) is dict and set(manifest) == {
        "schema", "task_root", "artifacts", "sizes", "shared_directories", "shared_files"},
        "shared-input manifest fields differ")
    need(manifest["schema"] == "geak-native-shared-inputs-v1", "unsupported shared-input manifest")
    original_root = manifest["task_root"]
    need(type(original_root) is str and Path(original_root).is_absolute()
         and str(Path(original_root)) == original_root and ".." not in Path(original_root).parts,
         "canonical manifest task root required")
    task_root = Path(task_root)
    need(task_root.is_absolute() and task_root.resolve(strict=True) == task_root,
         "canonical shared task root required")
    need(os.statvfs(task_root).f_flag & os.ST_RDONLY, "shared task root must be read-only")
    if contract is None:
        contract = validate_external_task(_decode(_read(task_root / "task.json")), task_root)
    need(manifest["artifacts"] == contract["artifacts"], "shared manifest task artifacts differ")
    shared, directories = manifest["shared_files"], manifest["shared_directories"]
    need(type(shared) is dict and bool(shared) and type(directories) is list and bool(directories)
         and all(type(name) is str for name in directories)
         and directories == sorted(set(directories)), "explicit shared task closure required")
    for name, digest in shared.items():
        _path(name)
        _sha(digest)
        need(contract["artifacts"].get(name) == digest and name not in contract["source_paths"],
             "shared inputs must be protected artifacts, never candidate source")
    for name in directories:
        _path(name)
        need(not any(name.startswith(other + "/") for other in directories if other != name),
             "overlapping shared directories")
        descendants = {key: value for key, value in contract["artifacts"].items()
                       if key.startswith(name + "/")}
        need(descendants and descendants == {key: value for key, value in shared.items()
                                             if key.startswith(name + "/")},
             "shared directory must cover exact artifact descendants")
    need(all(any(name.startswith(directory + "/") for directory in directories) for name in shared),
         "shared file lacks a declared directory")
    sizes = manifest["sizes"]
    need(type(sizes) is dict and set(sizes) == set(contract["artifacts"])
         and all(type(size) is int and size >= 0 for size in sizes.values()),
         "exact task artifact sizes required")
    need(all((task_root / name).stat().st_size == size for name, size in sizes.items()),
         "shared manifest task file size differs")
    for name in shared:
        target = task_root / name
        need(target.resolve(strict=True) == target and stat.S_ISREG(target.lstat().st_mode),
             "shared input must be a canonical regular file")
        need(os.statvfs(target).f_flag & os.ST_RDONLY, "shared input must be read-only")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("excludes", "link"))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--task-root", required=True, type=Path)
    parser.add_argument("--workspace", type=Path)
    args = parser.parse_args()
    manifest = load_shared_inputs(args.manifest, args.sha256, args.task_root)
    if args.action == "excludes":
        for name in sorted(manifest["shared_files"]):
            sys.stdout.buffer.write(("--exclude=./" + name).encode() + b"\0")
    else:
        need(args.workspace is not None and args.workspace.is_absolute()
             and args.workspace.resolve(strict=True) == args.workspace,
             "canonical private workspace required")
        for name in sorted(manifest["shared_files"]):
            link = args.workspace / name
            need(link.parent.resolve(strict=False) == link.parent,
                 "shared input workspace parent cannot be an alias")
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(args.task_root / name)


if __name__ == "__main__":
    main()
