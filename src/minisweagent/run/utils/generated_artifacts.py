"""Helpers for excluding generated helper artifacts from patches.

GEAK runs often materialize temporary harness helpers, standalone benchmark
binaries, and wrapper scripts at the worktree root. Those artifacts should not
be treated as source patches, and can later break ``git apply`` during round
evaluation when they leak into patch capture.
"""

from __future__ import annotations

import subprocess
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from typing import Any


_ROOT_GENERATED_DIRS = {
    "build",
    "build_harness",
    ".aiter_jit",
    "_eval_worktree",
}

_ROOT_GENERATED_FILES = {
    "run.sh",
    "run_harness.sh",
    "test_harness.py",
    "rocprim_version.hpp",
    "CMakeCache.txt",
    "cmake_install.cmake",
    "Makefile",
    "_geak_eval_cmd.sh",
}

_ROOT_GENERATED_GLOBS = (
    "_geak_test_cmd_*.sh",
    "test*_harness.py",
    "test*_harness.cpp",
    "test_*_focused.py",
    "*_standalone",
    "*_standalone.cpp",
    "*_test",
    "*_test.exe",
    "*.bak",
    "*.o",
    "*.obj",
    "*.out",
    "*.bin",
)


def _normalize_rel_path(rel_path: str) -> str:
    return PurePosixPath(str(rel_path).lstrip("./")).as_posix()


def is_generated_helper_artifact(rel_path: str) -> bool:
    """Return True when *rel_path* looks like a GEAK-generated helper artifact.

    Matching is intentionally conservative and targets root-level helper files
    and helper build directories, not normal source files inside the repo tree.
    """

    rel = _normalize_rel_path(rel_path)
    if not rel:
        return False

    path = PurePosixPath(rel)
    parts = path.parts
    if not parts:
        return False

    if parts[0] in _ROOT_GENERATED_DIRS:
        return True

    if len(parts) != 1:
        return False

    name = parts[0]
    if name in _ROOT_GENERATED_FILES:
        return True

    return any(fnmatch(name, pattern) for pattern in _ROOT_GENERATED_GLOBS)


def generated_helper_excludes(cwd: Path | None = None) -> list[str]:
    """Return git/diff exclude patterns for generated helper artifacts."""

    excludes = [
        "run.sh",
        "run_harness.sh",
        "build",
        "build_harness",
        ".aiter_jit",
        "_eval_worktree",
        "test_harness.py",
        "test_harness_*.py",
        "test_harness_*.cpp",
        "rocprim_version.hpp",
        "_geak_test_cmd_*.sh",
        "_geak_eval_cmd.sh",
    ]
    if cwd is not None and cwd.is_dir():
        for child in cwd.iterdir():
            if is_generated_helper_artifact(child.name):
                excludes.append(child.name)
    # Stable order makes debugging easier.
    return sorted(dict.fromkeys(excludes))


def _parse_git_diff_paths(header: str) -> tuple[str, str] | None:
    """Extract ``(a_path, b_path)`` from a ``diff --git`` header."""

    prefix = "diff --git a/"
    if not header.startswith(prefix):
        return None
    remainder = header[len(prefix):].rstrip("\n")
    separator = " b/"
    if separator not in remainder:
        return None
    a_path, b_path = remainder.split(separator, 1)
    return a_path, b_path


def strip_generated_helper_sections(patch_text: str) -> tuple[str, list[str]]:
    """Drop diff sections that touch generated helper artifacts.

    Returns ``(sanitized_patch_text, removed_paths)``. Only ``diff --git`` style
    sections are filtered; non-diff preamble is preserved.
    """

    if not patch_text.strip():
        return patch_text, []

    lines = patch_text.splitlines(keepends=True)
    preamble: list[str] = []
    sections: list[list[str]] = []
    current: list[str] | None = None

    for line in lines:
        if line.startswith("diff --git "):
            if current is not None:
                sections.append(current)
            current = [line]
            continue
        if current is None:
            preamble.append(line)
        else:
            current.append(line)

    if current is not None:
        sections.append(current)

    if not sections:
        return patch_text, []

    kept: list[str] = list(preamble)
    removed: list[str] = []
    for section in sections:
        parsed = _parse_git_diff_paths(section[0])
        if parsed is None:
            kept.extend(section)
            continue
        a_path, b_path = parsed
        if is_generated_helper_artifact(a_path) or is_generated_helper_artifact(b_path):
            removed.append(b_path or a_path)
            continue
        kept.extend(section)

    return "".join(kept), removed


def apply_patch_with_generated_helper_fallback(
    *,
    patch_text: str,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    """Apply a git patch, retrying after stripping generated helper sections.

    Returns ``(result, removed_paths)``. When the patch only contained generated
    helper artifacts, the empty sanitized patch is treated as a successful no-op.
    """

    result = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", "--binary", "-"],
        cwd=str(cwd),
        input=patch_text,
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode == 0:
        return result, []

    sanitized_patch, removed_paths = strip_generated_helper_sections(patch_text)
    if not removed_paths:
        return result, []

    if not sanitized_patch.strip():
        noop = subprocess.CompletedProcess(
            args=["git", "apply", "--whitespace=nowarn", "--binary", "-"],
            returncode=0,
            stdout=result.stdout,
            stderr=result.stderr,
        )
        return noop, removed_paths

    retry = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", "--binary", "-"],
        cwd=str(cwd),
        input=sanitized_patch,
        capture_output=True,
        text=True,
        env=env,
    )
    return retry, removed_paths
