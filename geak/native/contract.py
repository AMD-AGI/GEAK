# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""File, JSON and identity helpers for pinned native task contracts.

These helpers read bytes only; they do not import task code or manage search,
archives, GPU ownership or process lifecycles.
"""

import contextlib
import json
import math
import os
import re
import stat
from pathlib import Path, PurePosixPath

from ._util import require, sha256_hex


def _canonical(value):
    def check(item):
        if type(item) is dict:
            if not all(type(key) is str for key in item):
                raise ValueError("JSON keys must be strings")
            for child in item.values():
                check(child)
        elif type(item) is list:
            for child in item:
                check(child)
        else:
            if type(item) not in (str, int, float, bool, type(None)):
                raise ValueError("non-JSON value")
            if type(item) is float and not math.isfinite(item):
                raise ValueError("nonfinite JSON value")
    check(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _keys(value, names):
    require(type(value) is dict and set(value) == set(names), "unexpected contract fields", ValueError)


def _text(value):
    require(type(value) is str and bool(value.strip()) and "\x00" not in value,
          "expected nonempty text", ValueError)


def _sha(value):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value), "invalid SHA256", ValueError)


def _revision(value):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value),
          "revision must be a pinned commit or content digest", ValueError)


def _path(value):
    _text(value)
    path = PurePosixPath(value)
    require(not path.is_absolute() and str(path) == value and value != "."
          and not any(part in (".", "..") for part in path.parts) and "\\" not in value,
          "artifact path must be canonical and relative", ValueError)
    return value


def _pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "duplicate JSON key: " + key, ValueError)
        result[key] = value
    return result


def _decode(data):
    value = json.loads(data, object_pairs_hook=_pairs)
    _canonical(value)  # Reject NaN/Infinity as well as non-JSON values.
    return value


def _read(path):
    """Read one stable regular file without following any directory/file symlink."""
    path = Path(path).absolute()
    require(not any(part in (".", "..") for part in path.parts), "noncanonical input path", ValueError)
    try:
        with contextlib.ExitStack() as stack:
            fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            stack.callback(os.close, fd)
            for part in path.parts[1:-1]:
                fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                stack.callback(os.close, fd)
            fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
            stream = stack.enter_context(os.fdopen(fd, "rb"))
            before = os.fstat(fd)
            require(stat.S_ISREG(before.st_mode), "artifact must be a regular file", ValueError)
            data = stream.read()
            after = os.fstat(fd)
            stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode")
            require(all(getattr(before, key) == getattr(after, key) for key in stable)
                  and len(data) == after.st_size, "artifact changed during read", ValueError)
            return data
    except OSError as error:
        raise ValueError("artifact unavailable or unsafe: " + str(path)) from error


def _ids(value):
    require(type(value) is list and bool(value), "case domain must be a nonempty list", ValueError)
    require(all(type(item) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", item)
              for item in value), "invalid case ID", ValueError)
    require(len(value) == len(set(value)), "duplicate case ID", ValueError)


def job_digest(contract):
    """Hash the complete validated task without changing its stored identity."""
    from .external import validate_external_task
    return sha256_hex(_canonical(validate_external_task(contract)))
