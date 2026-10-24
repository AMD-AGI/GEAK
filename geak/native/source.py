# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture exact source bytes and executable modes, and check retained patches."""

import contextlib
import os
import stat
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

from ._util import require


def _path(value):
    """Return ``value`` if it is a canonical relative POSIX path."""
    if (not isinstance(value, str) or not value or "\\" in value or "\0" in value
            or PurePosixPath(value).is_absolute() or ".." in value.split("/")
            or PurePosixPath(value).as_posix() != value or value == "."):
        raise ValueError("source/artifact path must be canonical and relative")
    return value


def _read(root, name):
    """Read a regular file without following symlinks, including parent directories.

    Returns ``(bytes, executable_bits)``. Raises if the file changed while read.
    """
    parts = _path(name).split("/")
    with contextlib.ExitStack() as stack:
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        stack.callback(os.close, fd)
        for part in parts[:-1]:
            fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            stack.callback(os.close, fd)
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        stream = stack.enter_context(os.fdopen(fd, "rb"))
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("source and evidence must be regular files")
        data = stream.read()
        after = os.fstat(fd)
        stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode")
        if any(getattr(before, k) != getattr(after, k) for k in stable) or len(data) != after.st_size:
            raise ValueError("file changed during capture")
        return data, stat.S_IMODE(after.st_mode) & 0o111


def _need(value, reason):
    require(value, reason, ValueError)


def _snapshot(root, paths):
    root = Path(root).absolute()
    _need(root.resolve(strict=True) == root, "canonical workspace required")
    return {name: _read(root, _path(name)) for name in paths}


def _git(root, patch, *options):
    # No inherited Git configuration or work-tree override may redirect apply.
    completed = subprocess.run(
        ["git", "apply", *options, "-"], cwd=root, input=patch,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "GIT_CONFIG_NOSYSTEM": "1",
             "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CEILING_DIRECTORIES": str(root.parent)})
    _need(completed.returncode == 0,
          "patch could not be checked/applied: " + completed.stderr.decode("utf-8", errors="replace").strip())
    return completed.stdout


def _patched(source, patch):
    """Apply the captured patch only to a disposable declared source closure."""
    with tempfile.TemporaryDirectory(prefix="geak-control-source-") as directory:
        root = Path(directory)
        for name, (raw, mode) in source.items():
            _need(".git" not in Path(name).parts, "Git metadata is outside the patchable source closure")
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            path.chmod(0o644 | mode)
        if patch:
            # Forward/reverse numstat names the destination/source of renames.
            # NUL records preserve quoted names, spaces and tabs without guessing
            # at the human-readable diff header grammar.
            for direction in ((), ("--reverse",)):
                stats = _git(root, patch, "--numstat", "-z", *direction)
                _need(stats.endswith(b"\0"), "patch has no declared file changes")
                for row in stats[:-1].split(b"\0"):
                    fields = row.split(b"\t", 2)
                    _need(len(fields) == 3, "invalid patch file inventory")
                    name = _path(fields[2].decode("utf-8"))
                    _need(name in source, "patch touches a file outside the source closure: " + name)
            _git(root, patch, "--whitespace=nowarn")
        files = set()
        for path in root.rglob("*"):
            _need(not path.is_symlink(), "patched source contains a symlink")
            if not path.is_dir():
                files.add(path.relative_to(root).as_posix())
        _need(files == set(source), "patched source closure differs")
        return _snapshot(root, source)
