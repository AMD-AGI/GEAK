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
    "_geak_harness",
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
    "*.orig_backup",
    "*.baseline_*",
)

# Hipify / pip-install side-effect basename globs. These files are produced by
# building the worktree (typically ``pip install -e .`` running hipify on the
# CUDA sources), so they are NOT authored by the agent and MUST NOT be captured
# in the patch. Without this, two failure modes appear during round eval:
#   1. Patch files balloon to 6k+ lines of hipified C++ noise.
#   2. ``git apply`` fails with ``... already exists in working directory`` /
#      ``patch failed: pyproject.toml:1`` because the eval worktree's own
#      install regenerates the same files first.
# Patterns match against the path basename, so they apply at any depth.
_INSTALL_SIDE_EFFECT_BASENAME_GLOBS = (
    "*_hip.cuh",          # e.g. sgl-kernel/include/hip/hip_act_and_mul_hip.cuh
    "*_hip.h",            # e.g. sgl-kernel/include/utils_hip.h
    "*_hip.hpp",
    "*.hip",              # e.g. sgl-kernel/csrc/elementwise/activation.hip
    "pyproject.toml",     # rewritten by some pip install -e flows
    "setup.py.bak",
    "MANIFEST.in.bak",
)


def _normalize_rel_path(rel_path: str) -> str:
    return PurePosixPath(str(rel_path).lstrip("./")).as_posix()


def _matches_install_side_effect(name: str) -> bool:
    """Return True when *name* (a path basename) is a hipify / install artifact."""
    return any(fnmatch(name, pattern) for pattern in _INSTALL_SIDE_EFFECT_BASENAME_GLOBS)


def is_generated_helper_artifact(rel_path: str) -> bool:
    """Return True when *rel_path* looks like a GEAK-generated helper artifact.

    Matching covers two categories:

    1. Root-level helper files / dirs that GEAK itself materializes (harness
       scripts, build dirs, profiling dumps).
    2. Hipify / ``pip install -e .`` side-effect files (``*_hip.cuh``,
       ``*.hip``, ``pyproject.toml`` rewrites, ...). These can appear at any
       depth inside the worktree and must NEVER end up in a captured patch.
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

    # Install side-effects can appear anywhere in the tree (e.g. under
    # ``sgl-kernel/csrc/.../activation.hip``), so check the basename
    # regardless of depth.
    if _matches_install_side_effect(parts[-1]):
        return True

    if len(parts) != 1:
        return False

    name = parts[0]
    if name in _ROOT_GENERATED_FILES:
        return True

    return any(fnmatch(name, pattern) for pattern in _ROOT_GENERATED_GLOBS)


def install_side_effect_git_pathspecs() -> list[str]:
    """Return git pathspec strings (already prefixed with ``:(exclude,glob)``)
    that exclude hipify / install side-effect files at any depth.

    Each entry is ready to be passed as a single argv element to ``git diff``
    (e.g. ``git diff -- . :(exclude,glob)**/*.hip``).
    """

    return [f":(exclude,glob)**/{pattern}" for pattern in _INSTALL_SIDE_EFFECT_BASENAME_GLOBS]


def install_side_effect_diff_basenames() -> list[str]:
    """Return basename globs suitable for ``diff --exclude=<glob>``.

    GNU ``diff --exclude`` runs fnmatch against the basename of each candidate
    file, so a pattern like ``*.hip`` matches at any depth automatically.
    """

    return list(_INSTALL_SIDE_EFFECT_BASENAME_GLOBS)


def generated_helper_excludes(cwd: Path | None = None) -> list[str]:
    """Return basename-style exclude patterns for generated helper artifacts.

    These are fed into both ``git diff -- . :(exclude)<entry>`` (which matches
    against full path) and ``diff -ruN --exclude=<entry>`` (which matches
    against basename). Hipify / install side-effects are intentionally NOT
    returned here; callers should additionally consult
    :func:`install_side_effect_git_pathspecs` /
    :func:`install_side_effect_diff_basenames` because those need different
    formatting per backend (``:(exclude,glob)`` for git, plain basename for
    GNU diff).
    """

    excludes = [
        "run.sh",
        "run_harness.sh",
        "build",
        "build_harness",
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


def _section_is_binary(section_lines: list[str]) -> bool:
    """True when a diff section contains a GIT binary patch (never source code)."""
    return any("GIT binary patch" in line for line in section_lines[:10])


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
        if _section_is_binary(section):
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

    return _DIFF_RUN_HEADER_RE.sub(_rewrite, patch_text)


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
