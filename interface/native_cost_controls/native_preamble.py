# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recognize one measured Bash loader warning from exact runtime files."""

from __future__ import annotations

import hashlib
import stat
from pathlib import Path

BASH_LIBRARY_WARNING = (
    "/bin/bash: /opt/venv/lib/python3.12/site-packages/torch/lib/libtinfo.so.6: "
    "no version information available (required by /bin/bash)\n"
)
_ASSETS = (
    ("/bin/bash", "/usr/bin/bash", "bc5945feb8bd26203ebfafea5ce1878bb2e32cb8fb50ab7ae395cfb1e1aaaef1"),
    ("/opt/venv/lib/python3.12/site-packages/torch/lib/libtinfo.so.6",
     "/opt/venv/lib/python3.12/site-packages/torch/lib/libtinfo.so",
     "34f5f4b320ab193dee91aa706f563ed7f3e175af74f334e1535b10302e13615b"),
)


def qualified_bash_warning(native_shell):
    """Return a bound profile only for the measured shell and library bytes.

    The driver retains the complete raw native output. Its existing projector
    can remove one exact leading warning after this runtime profile qualifies.
    Missing, replaced, or unsupported runtime files qualify no warning.
    """
    if native_shell != "/bin/bash":
        return None
    files = {}
    try:
        for name, expected_path, expected_hash in _ASSETS:
            path = Path(name)
            target = path.resolve(strict=True)
            metadata = target.stat()
            if (str(target) != expected_path or not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_size > 4 * 1024 * 1024):
                return None
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
            if actual != expected_hash or path.resolve(strict=True) != target:
                return None
            files[name] = {"resolved_path": str(target), "sha256": actual}
    except (OSError, RuntimeError):
        return None
    return {"prefix": BASH_LIBRARY_WARNING, "native_shell": native_shell, "files": files}
