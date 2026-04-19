"""Helpers for excluding generated helper artifacts from patches.

GEAK runs often materialize temporary harness helpers, standalone benchmark
binaries, and wrapper scripts at the worktree root. Those artifacts should not
be treated as source patches, and can later break ``git apply`` during round
evaluation when they leak into patch capture.
"""

from __future__ import annotations

import re
import subprocess
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath

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
    "baseline_metrics.json",
    "profile.json",
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
        "baseline_metrics.json",
        "profile.json",
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
    remainder = header[len(prefix) :].rstrip("\n")
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


# Conflict-marker regexes. We reject patches whose 3-way merge result contains
# any of the classic markers so we never silently apply corrupted content.
_CONFLICT_MARKER_RE = re.compile(rb"^(<{7} |={7}$|>{7} )", re.MULTILINE)


def _worktree_has_conflict_markers(cwd: Path) -> bool:
    """Return True if any file in ``cwd`` contains git conflict markers.

    Only inspects tracked-by-filesystem files (skips ``.git``) and treats any
    byte-level match as a conflict. Binary files usually don't match the
    markers, so false positives are rare.
    """

    for path in cwd.rglob("*"):
        try:
            if ".git" in path.relative_to(cwd).parts:
                continue
        except ValueError:
            continue
        if not path.is_file():
            continue
        try:
            with path.open("rb") as fh:
                data = fh.read(1024 * 1024)  # cap at 1MB per file for speed
        except OSError:
            continue
        if _CONFLICT_MARKER_RE.search(data):
            return True
    return False


def _register_object_alternates(cwd: Path, alternates: list[Path]) -> bool:
    """Append ``alternates`` to this repo's object store. Returns True if any
    new path was actually added. Best-effort; silently skips paths that can't
    be resolved or appended.
    """

    try:
        objects_dir_result = subprocess.run(
            ["git", "rev-parse", "--git-path", "objects/info/alternates"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

    alternates_path = Path(objects_dir_result.stdout.strip())
    if not alternates_path.is_absolute():
        alternates_path = cwd / alternates_path
    alternates_path.parent.mkdir(parents=True, exist_ok=True)
    existing = alternates_path.read_text() if alternates_path.exists() else ""

    new_lines: list[str] = []
    for alt in alternates:
        try:
            resolved = Path(alt).resolve(strict=False)
        except (OSError, RuntimeError):
            continue
        if not resolved.is_dir():
            continue
        line = str(resolved)
        if line in existing or line in new_lines:
            continue
        new_lines.append(line)

    if not new_lines:
        return False

    suffix = "" if existing.endswith("\n") or not existing else "\n"
    alternates_path.write_text(existing + suffix + "\n".join(new_lines) + "\n")
    return True


def _try_three_way_with_alternates(
    *,
    patch_text: str,
    cwd: Path,
    env: dict[str, str] | None,
    alternates: list[Path],
) -> subprocess.CompletedProcess[str] | None:
    """Fallback: register object alternates and attempt ``git apply --3way``.

    Only accepts the result if git reports success AND no conflict markers
    are produced in the working tree. Returns the successful CompletedProcess
    or None on any failure / conflict-marker detection.
    """

    if not alternates:
        return None
    if not _register_object_alternates(cwd, alternates):
        return None

    # Ensure any partial state from prior failed applies is reset before the
    # 3-way attempt. ``git apply`` without ``--index`` doesn't touch the index,
    # and failed applies are atomic on disk, so this is a safety net only.
    try:
        subprocess.run(
            ["git", "checkout", "--", "."],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
    except FileNotFoundError:
        return None

    result = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", "--binary", "--3way", "-"],
        cwd=str(cwd),
        input=patch_text,
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        return None
    if _worktree_has_conflict_markers(cwd):
        # Reject silently-conflicted result; caller will propagate the
        # original plain-apply failure.
        try:
            subprocess.run(
                ["git", "checkout", "--", "."],
                cwd=str(cwd),
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )
        except FileNotFoundError:
            pass
        return None
    return result


_DIFF_RUN_HEADER_RE = re.compile(
    r"^(---|\+\+\+)\s+([^\t\n]+)(\t[^\n]*)?$",
    re.MULTILINE,
)


def normalize_patch_paths(patch_text: str, target_basename: str = "kernel.py") -> str:
    """Convert ``diff -ruN`` style headers (absolute paths) into git-style.

    ``diff -ruN`` produces headers like::

        --- /home/user/repo/.../kernel.py    2026-04-17 19:10:02 +0000
        +++ kernel.py                        2026-04-19 01:21:00 +0000

    ``git apply`` expects::

        --- a/kernel.py
        +++ b/kernel.py

    This function rewrites any ``--- /abs/path/<basename>`` and
    ``+++ <basename>`` (or ``+++ /abs/path/<basename>``) headers into the
    git-style equivalent so the patch applies cleanly in any worktree
    where the target file lives at the same relative path.

    Returns the patch text unchanged if it already uses git-style headers
    (no absolute-path header found). Safe to call multiple times.
    """
    if not patch_text or "--- " not in patch_text:
        return patch_text

    needs_normalization = False
    for line in patch_text.splitlines()[:20]:  # only look at the head
        if line.startswith(("--- /", "+++ /")):
            needs_normalization = True
            break
        if line.startswith("--- ") and target_basename in line and " a/" not in line:
            needs_normalization = True
            break

    if not needs_normalization:
        return patch_text

    def _rewrite(match: re.Match[str]) -> str:
        prefix = match.group(1)  # --- or +++
        path = match.group(2).strip()
        # Extract just the basename (strip absolute path)
        basename = Path(path).name if path != "/dev/null" else path
        if basename == "/dev/null":
            return f"{prefix} /dev/null"
        side = "a" if prefix == "---" else "b"
        return f"{prefix} {side}/{basename}"

    normalized = _DIFF_RUN_HEADER_RE.sub(_rewrite, patch_text)
    return normalized


def apply_patch_with_generated_helper_fallback(
    *,
    patch_text: str,
    cwd: Path,
    env: dict[str, str] | None = None,
    object_alternates: list[Path] | None = None,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    """Apply a git patch, retrying after stripping generated helper sections.

    Returns ``(result, removed_paths)``. When the patch only contained generated
    helper artifacts, the empty sanitized patch is treated as a successful no-op.

    ``object_alternates`` is an optional list of ``.git/objects`` directories
    from sibling worktrees (e.g. the sub-agents that produced the patch). If
    the primary plain apply and the sanitized retry both fail, this function
    will register those alternates into the current repo's object store and
    attempt a ``git apply --3way`` to bridge patch-lineage mismatches (see
    commit history for the refk_identity R1 case). The 3-way result is only
    accepted if git reports success AND no conflict markers are produced.
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

    # NEW: try path-normalization for diff-ruN-style absolute-path headers
    # (e.g. "--- /home/user/repo/.../kernel.py" instead of "--- a/kernel.py").
    # Some sub-agent worktrees fall through the git-repo detection in
    # save_and_test._get_patch_content and use ``diff -ruN`` which produces
    # absolute paths that ``git apply`` cannot resolve.
    normalized = normalize_patch_paths(patch_text)
    if normalized != patch_text:
        norm_result = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", "--binary", "-"],
            cwd=str(cwd),
            input=normalized,
            capture_output=True,
            text=True,
            env=env,
        )
        if norm_result.returncode == 0:
            return norm_result, []

    sanitized_patch, removed_paths = strip_generated_helper_sections(patch_text)
    if not removed_paths:
        three_way = _try_three_way_with_alternates(
            patch_text=patch_text,
            cwd=cwd,
            env=env,
            alternates=object_alternates or [],
        )
        if three_way is not None:
            return three_way, []
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
    if retry.returncode == 0:
        return retry, removed_paths

    three_way = _try_three_way_with_alternates(
        patch_text=sanitized_patch,
        cwd=cwd,
        env=env,
        alternates=object_alternates or [],
    )
    if three_way is not None:
        return three_way, removed_paths
    return retry, removed_paths
