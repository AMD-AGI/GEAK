"""Helpers for excluding generated helper artifacts from patches.

GEAK runs often materialize temporary harness helpers, standalone benchmark
binaries, and wrapper scripts at the worktree root. Those artifacts should not
be treated as source patches, and can later break ``git apply`` during round
evaluation when they leak into patch capture.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import tempfile
from collections.abc import Callable
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


def _strip_diff_sections(
    patch_text: str,
    predicate: Callable[[str], bool],
) -> tuple[str, list[str]]:
    """Generic helper: drop ``diff --git`` sections whose path satisfies *predicate*.

    Used by :func:`strip_jit_cache_sections` so JIT filtering stays behaviourally
    aligned with :func:`strip_generated_helper_sections` (only the predicate differs).

    Returns ``(sanitized_patch_text, removed_paths)``. Non-diff preamble and
    sections without parseable ``diff --git`` headers are preserved verbatim.
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
        if predicate(a_path) or predicate(b_path):
            removed.append(b_path or a_path)
            continue
        kept.extend(section)

    return "".join(kept), removed


# ---------------------------------------------------------------------------
# JIT runtime cache exclusion
# ---------------------------------------------------------------------------
#
# The patterns above target files the GEAK *infrastructure* writes at the
# worktree root. They do NOT cover JIT runtime cache files that the kernel
# under test writes inside the package tree (e.g. aiter flydsl_cache pkls).
# Those can ride along inside every patch and pollute final_report.json's
# ``optimized_codes`` with hundreds of 0-byte "added" entries.

_JIT_CACHE_PATH_SEGMENTS: frozenset[str] = frozenset(
    {
        "flydsl_cache",
        ".triton",
        "triton_cache",
        "torch_compile_cache",
        ".aiter_jit",
        "__pycache__",
    }
)

# Directory-segment *prefixes* that identify a JIT runtime cache even when the
# cache dir carries a suffix. Triton honours a per-process ``TRITON_CACHE_DIR``
# that a worktree-local harness can point at a dir such as ``.triton_cache_geak``
# (or ``triton_cache_slot0``); exact-equality matching against the segment set
# above silently misses those. That gap let ~680 KB of
# ``.triton_cache_geak/<hash>/fused_moe_kernel.{hsaco,amdgcn,llir,ttir,ttgir,
# source,json}`` ride along inside the reactor's per-round patch, which then
# failed to re-apply on a clean checkout and collapsed GEAK's internal A/B
# speedup to 1.0x (so the KEEP gate never fired hands-off).
#
# Kept toolchain-agnostic on purpose: the prefixes cover the cache dirs that do
# NOT literally contain "cache" (``.triton``, ``.aiter``/``.aiter_jit``,
# ``torchinductor_<user>``); the substrings below catch every ``*cache*`` dir
# (``triton_cache``, ``torch_compile_cache``, ``.cache``, ``__pycache__`` ...).
_JIT_CACHE_SEGMENT_PREFIXES: tuple[str, ...] = (
    ".triton",        # .triton, .triton_cache_geak, triton_cache_slotN
    ".aiter",         # .aiter, .aiter_jit, .aiter_cpp_itfs
    "torch_compile",  # torch_compile_cache
    "torchinductor",  # torchinductor_<user> (torch._inductor runtime cache)
    "flydsl_cache",
)

# Directory-segment *substrings*: any path component containing one of these is
# treated as a runtime/compile cache dir. ``cache`` catches the long tail of
# toolchain cache dirs (``triton_cache``, ``torch_compile_cache``, ``.cache``,
# ``__pycache__``, ``*_cache``) without enumerating each one. Applied to
# *directory* components only (never the leaf file), so source files such as
# ``cache_utils.py`` / ``triton_cache_config.py`` are never misclassified.
_JIT_CACHE_SEGMENT_SUBSTRINGS: tuple[str, ...] = ("cache",)

# Extensions that are *always* JIT/AOT compile outputs (Triton IR dumps and GPU
# code objects / object files), never hand-written source. Belt-and-suspenders
# so an unrecognised cache-dir name still cannot smuggle compiled blobs into a
# source patch.
_JIT_COMPILE_OUTPUT_SUFFIXES: frozenset[str] = frozenset(
    {
        ".hsaco",  # AMD GPU code object (compiled binary)
        ".amdgcn",  # AMDGCN assembly dump
        ".llir",  # LLVM IR dump
        ".ttir",  # Triton IR dump
        ".ttgir",  # Triton GPU IR dump
        ".cubin",  # NVIDIA CUDA binary
        ".ptx",  # NVIDIA PTX assembly
        ".spv",  # SPIR-V binary
        ".o",  # object file (compiled)
        ".so",  # shared object (compiled)
    }
)

_JIT_NESTED_BUILD_DIRS: frozenset[str] = frozenset({"build", "__pycache__"})


def _segment_is_jit_cache_dir(segment: str) -> bool:
    """True when a single path segment names a JIT runtime cache directory."""

    if segment in _JIT_CACHE_PATH_SEGMENTS:
        return True
    if any(segment.startswith(prefix) for prefix in _JIT_CACHE_SEGMENT_PREFIXES):
        return True
    return any(sub in segment for sub in _JIT_CACHE_SEGMENT_SUBSTRINGS)


def is_jit_cache_artifact(rel_path: str | None) -> bool:
    """Return True when *rel_path* is a JIT/compile cache artifact.

    A path is a cache artifact when any of its *directory* components names a JIT
    runtime cache (exact match in ``_JIT_CACHE_PATH_SEGMENTS``, a suffixed cache
    dir matched by ``_JIT_CACHE_SEGMENT_PREFIXES``, or a ``*cache*`` dir matched
    by ``_JIT_CACHE_SEGMENT_SUBSTRINGS``), when it sits under a
    ``jit/{build,__pycache__}`` tree, or when its extension is an unambiguous
    compile output (``_JIT_COMPILE_OUTPUT_SUFFIXES``). Prefix/substring matching
    is restricted to *directory* components so source files such as
    ``triton_cache_config.py`` / ``cache_utils.py`` are never misclassified.
    """

    if not rel_path:
        return False
    rel = str(rel_path).lstrip("/")
    if rel.startswith("./"):
        rel = rel[2:]
    if not rel:
        return False
    pure = PurePosixPath(rel)
    parts = pure.parts

    # Any directory component that is a JIT cache dir taints the whole path.
    for seg in parts[:-1]:
        if _segment_is_jit_cache_dir(seg):
            return True
    # The leaf only counts on an exact cache-dir name (e.g. a bare ".triton"
    # entry), to avoid flagging source files whose name merely starts with a
    # cache prefix.
    if parts and parts[-1] in _JIT_CACHE_PATH_SEGMENTS:
        return True
    for i in range(len(parts) - 1):
        if parts[i] == "jit" and parts[i + 1] in _JIT_NESTED_BUILD_DIRS:
            return True
    # Unambiguous compile outputs by extension (covers unknown cache dirs).
    if pure.suffix in _JIT_COMPILE_OUTPUT_SUFFIXES:
        return True
    return False


def strip_jit_cache_sections(patch_text: str) -> tuple[str, list[str]]:
    """Drop diff sections whose path lives inside a JIT runtime cache."""

    return _strip_diff_sections(patch_text, is_jit_cache_artifact)


def jit_cache_diff_basename_excludes() -> list[str]:
    """Return JIT-cache exclude patterns for ``diff -ruN`` / ``git diff``.

    Includes exact cache-dir basenames, prefix globs for suffixed cache dirs
    (e.g. ``.triton_cache_geak``), substring globs for any ``*cache*`` dir, and
    compile-output extension globs so the cache trees are skipped before the
    patch is even rendered.
    """

    patterns: set[str] = set(_JIT_CACHE_PATH_SEGMENTS)
    patterns.update(f"{prefix}*" for prefix in _JIT_CACHE_SEGMENT_PREFIXES)
    patterns.update(f"*{sub}*" for sub in _JIT_CACHE_SEGMENT_SUBSTRINGS)
    patterns.update(f"*{suffix}" for suffix in _JIT_COMPILE_OUTPUT_SUFFIXES)
    return sorted(patterns)


def jit_cache_env(
    work_dir: str | Path | None,
    *,
    base: str | Path | None = None,
) -> dict[str, str]:
    """Return env vars that relocate JIT/compile caches OUTSIDE *work_dir*.

    This is the PRIMARY, toolchain-agnostic half of the "no JIT artifacts in the
    round patch" fix. GEAK runs each optimization round inside a git worktree
    (*work_dir*) and captures the per-round patch with ``git diff`` *inside that
    worktree*. If a runtime compiler (Triton, torch inductor, ...) writes its
    cache under the worktree, the compiled blobs (``*.hsaco`` / IR dumps) get
    swept into the patch; re-applying that patch for the reactor's A/B then either
    aborts the atomic ``git apply`` (the blobs already exist in the freshly
    seeded worktree) or writes stale blobs that mask the patched kernel -- so the
    measured speedup collapses to 1.0x and a genuinely-good kernel (the
    fused_moe +1.68x) never clears the KEEP gate.

    Pointing the caches at a dedicated dir OUTSIDE the diffed tree keeps the
    captured patch source-only and lets the A/B compile the patched source fresh
    and see the real speedup. The location is:

    * **outside** *work_dir* (so ``git diff`` in the worktree never sees it);
    * **stable per worktree** (a slot reuses its own compile cache across rounds
      -- no needless recompiles; Triton is content-addressed by source so this is
      always correct);
    * **unique per worktree** (parallel slots never collide / contend).

    Sets ``TRITON_CACHE_DIR`` (Triton) and ``GEAK_JIT_CACHE_DIR`` (a generic root
    the sanitized harness and other toolchains -- aiter, inductor -- hang their
    caches off of; see the harness-sanitizer R5 rule). Returns ``{}`` when
    *work_dir* is falsy. ``base`` overrides the parent dir (test hook / explicit
    out-of-tree root); it defaults to ``$TMPDIR/geak_jit_cache``.
    """
    if not work_dir:
        return {}
    wt = Path(work_dir).resolve()
    if base is not None:
        root = Path(base)
    else:
        root = Path(tempfile.gettempdir()) / "geak_jit_cache"
    slot = hashlib.sha1(str(wt).encode("utf-8")).hexdigest()[:16]
    cache_root = root / slot
    return {
        "GEAK_JIT_CACHE_DIR": str(cache_root),
        "TRITON_CACHE_DIR": str(cache_root / "triton"),
    }


def _section_is_pure_addition(section_lines: list[str]) -> bool:
    """True when a diff section creates a brand-new file (``new file mode``)."""

    for line in section_lines[1:6]:
        if line.startswith("new file mode"):
            return True
        if line.startswith("@@"):
            break
    return False


def strip_already_present_additions(patch_text: str, base_dir: str | Path) -> tuple[str, list[str]]:
    """Drop ``diff --git`` sections that *add* a file already present in *base_dir*.

    ``create_worktree`` seeds every fresh worktree with the repo's untracked,
    non-ignored files (see ``task_file._copy_untracked_files``). When the
    captured patch *also* records those same files as new-file additions (because
    ``git add -N`` listed them at capture time), re-applying the patch onto such a
    worktree aborts atomically with ``error: <path>: already exists in working
    directory`` -- taking the legitimate source modification down with it and
    leaving GEAK's A/B at 1.0x. Such additions are no-ops (an identical file is
    already there), so dropping them is safe and lets the real edit apply.

    Only *pure additions* (``new file mode``) whose target already exists under
    *base_dir* are removed. Modifications, deletions, and additions of genuinely
    new (agent-authored) files are preserved verbatim.

    Returns ``(sanitized_patch_text, removed_paths)``.
    """

    if not patch_text.strip():
        return patch_text, []

    base = Path(base_dir)
    lines = patch_text.splitlines(keepends=True)
    preamble: list[str] = []
    sections: list[list[str]] = []
    current: list[str] | None = None
    for line in lines:
        if line.startswith("diff --git "):
            if current is not None:
                sections.append(current)
            current = [line]
        elif current is None:
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
        if parsed is not None and _section_is_pure_addition(section):
            a_path, b_path = parsed
            target = b_path or a_path
            try:
                present = bool(target) and (base / target).exists()
            except OSError:
                present = False
            if present:
                removed.append(target)
                continue
        kept.extend(section)
    return "".join(kept), removed


def sanitize_patch_for_apply(patch_text: str, cwd: str | Path) -> tuple[str, list[str]]:
    """Strip patch sections that can never carry an applicable source edit.

    Combines, in order: JIT/compile-cache stripping, removal of new-file
    additions already present in *cwd* (the destination worktree), and
    generated-helper stripping. Returns ``(sanitized_text, removed_paths)``.
    """

    removed_all: list[str] = []
    text, removed = strip_jit_cache_sections(patch_text)
    removed_all.extend(removed)
    text, removed = strip_already_present_additions(text, cwd)
    removed_all.extend(removed)
    text, removed = strip_generated_helper_sections(text)
    removed_all.extend(removed)
    return text, removed_all


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

    # Strip sections that can never carry an applicable source edit, then retry:
    #   (1) JIT/compile-cache artifacts (Triton TRITON_CACHE_DIR blobs, IR dumps);
    #   (2) new-file additions whose target already exists in this worktree --
    #       create_worktree seeds the worktree with the repo's untracked,
    #       non-ignored files, so re-adding them aborts the *atomic* apply with
    #       "already exists in working directory", killing the real edit too;
    #   (3) generated helper artifacts (legacy behaviour).
    # This is what lets the reactor's clean-checkout re-apply land the genuine
    # kernel edit so the A/B measures the true speedup instead of 1.0x.
    sanitized_patch, removed_paths = sanitize_patch_for_apply(patch_text, cwd)
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
