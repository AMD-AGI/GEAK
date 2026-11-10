#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read current source identity and exact-path metadata, without attributing profiles."""

import sys

sys.dont_write_bytecode = True

import contextlib
import json
import os
import stat
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from geak.native._util import canonical_json, sha256_hex
from geak.native.source import _path, _read, _snapshot


MAX_SOURCE_FILES = 1024
MAX_ENTRIES = 4096
MAX_BYTES = 64 * 1024 * 1024
MAX_KNOWN_PATHS = 32
DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def source_paths(root, scope):
    """Expand only explicit source scopes, rejecting links and special files."""
    if not isinstance(scope, list) or len(scope) > MAX_SOURCE_FILES:
        raise ValueError("invalid_source_scope")
    names = sorted({_path(name) for name in scope})
    root = Path(root)
    if not root.is_absolute() or root.resolve(strict=True) != root:
        raise ValueError("noncanonical_workspace")
    files, visited, total = {}, set(), 0

    def visit(parent, leaf, name):
        nonlocal total
        if name in visited:
            return
        visited.add(name)
        if len(visited) > MAX_ENTRIES:
            raise ValueError("source_entry_cap")
        info = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
        if stat.S_ISREG(info.st_mode):
            total += info.st_size
            files[name] = info.st_size
            if len(files) > MAX_SOURCE_FILES or total > MAX_BYTES:
                raise ValueError("source_size_cap")
        elif stat.S_ISDIR(info.st_mode):
            directory = os.open(leaf, DIR_FLAGS, dir_fd=parent)
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        visit(directory, entry.name, name + "/" + entry.name)
            finally:
                os.close(directory)
        else:
            raise ValueError("nonregular_source")

    for name in names:
        with contextlib.ExitStack() as stack:
            parent = os.open(root, DIR_FLAGS)
            stack.callback(os.close, parent)
            parts = name.split("/")
            for part in parts[:-1]:
                parent = os.open(part, DIR_FLAGS, dir_fd=parent)
                stack.callback(os.close, parent)
            visit(parent, parts[-1], name)
    return sorted(files)


def current_source(workspace, scope):
    result = {"current_source_sha256": None, "source_paths": [],
              "source_status": "unknown", "source_reason": "no_source_scope"}
    if scope is None or scope == []:
        return result
    try:
        root = Path(workspace)
        paths = source_paths(root, scope)
        if not paths:
            raise ValueError("empty_source_scope")
        first = _snapshot(root, paths)
        if (sum(len(raw) for raw, _ in first.values()) > MAX_BYTES
                or paths != source_paths(root, scope)
                or first != _snapshot(root, paths)
                or paths != source_paths(root, scope)):
            raise ValueError("source_changed")
        source = {"kind": "source", "files": [
            {"path": name, "blob_id": sha256_hex(raw), "executable": mode}
            for name, (raw, mode) in sorted(first.items())]}
        result.update(current_source_sha256=sha256_hex(canonical_json(source)),
                      source_paths=paths, source_status="verified", source_reason=None)
    except Exception:
        result["source_reason"] = "source_capture_failed"
    return result


def known_paths(paths):
    if (not isinstance(paths, list) or len(paths) > MAX_KNOWN_PATHS
            or not all(isinstance(path, str) for path in paths)):
        raise ValueError("invalid_known_paths")
    rows, remaining = [], MAX_BYTES
    for name in sorted(set(paths)):
        row = {"path": name, "kind": "unknown", "sha256": None,
               "bytes": None, "available": False}
        try:
            path = Path(name)
            if (not path.is_absolute() or str(path) != name
                    or path.resolve(strict=True) != path):
                raise ValueError("noncanonical_known_path")
            info = path.lstat()
            if stat.S_ISDIR(info.st_mode):
                row.update(kind="directory", available=True)
            elif stat.S_ISREG(info.st_mode):
                row.update(kind="file", bytes=info.st_size, available=True)
                if info.st_size <= remaining:
                    raw, _ = _read(Path("/"), _path(name[1:]))
                    if len(raw) <= remaining:
                        remaining -= len(raw)
                        row.update(sha256=sha256_hex(raw), bytes=len(raw))
            else:
                raise ValueError("nonregular_known_path")
        except Exception:
            row.update(sha256=None, bytes=None, available=False)
        rows.append(row)
    return rows


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 3:
        raise ValueError("expected_workspace_source_paths_known_paths")
    result = current_source(args[0], json.loads(args[1]))
    result["paths"] = known_paths(json.loads(args[2]))
    print(canonical_json(result).decode())


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print('{"error":"invalid_metadata_request"}')
        sys.exit(1)
