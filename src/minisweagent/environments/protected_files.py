"""Detect shell commands that would write to protected files.

Used to enforce read-only baseline files (e.g. kernel.py in optimization tasks).
"""

import re


def would_write_protected_file(command: str, protected_files: list[str]) -> str | None:
    """Return the first protected filename that the command would write to, or None.

    Heuristics: redirects (> file, >> file), tee, sed -i, cp/mv with file as destination.
    """
    if not protected_files:
        return None
    # Normalize: strip and work with a single line for simplicity (multi-line commands may redirect to file)
    cmd = command.strip()
    for name in protected_files:
        # Escape for regex (e.g. kernel.py -> kernel\.py)
        escaped = re.escape(name)
        # Redirect overwrite or append
        if re.search(r"[>]{1,2}\s*" + escaped + r"\b", cmd):
            return name
        # tee (writes to file)
        if re.search(r"\btee\s+.*" + escaped + r"\b", cmd):
            return name
        # sed -i in-place edit (GNU: sed -i 's/.../...' file, BSD: sed -i '' ...)
        if re.search(r"\bsed\s+.*-i\s*['\"]?.*" + escaped + r"\b", cmd):
            return name
        # cp/mv with file as destination (name at end of command or before ; & | newline)
        if re.search(r"\bcp\s+.*\s+" + escaped + r"\s*[;|&\n]?\s*$", cmd, re.MULTILINE | re.DOTALL):
            return name
        if re.search(r"\bmv\s+.*\s+" + escaped + r"\s*[;|&\n]?\s*$", cmd, re.MULTILINE | re.DOTALL):
            return name
    return None


def blocked_message(filename: str) -> str:
    """Standard message when a write to a protected file is blocked."""
    return (
        f"Blocked: modifying '{filename}' is not allowed. "
        "Keep it as the baseline reference; create new files (e.g. kernel_v2.py, kernel_v3.py) for optimizations."
    )
