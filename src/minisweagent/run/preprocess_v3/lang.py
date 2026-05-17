"""Language detection for the v3 preprocess pipeline (pre-step 0b).

This module is a **thin v3-side adapter** over the canonical
:mod:`minisweagent.kernel_languages` package. It deliberately does not
redefine ``KernelLanguage`` — the existing frozen dataclass is the single
source of truth for language-specific metadata across the codebase, and
the v3 pipeline reuses it unchanged.

What this module adds on top of the canonical registry:

* :data:`FLYDSL` and :data:`UNKNOWN` sentinel ``KernelLanguage`` instances
  so the v3 detection signature can return a non-``None`` value in all
  cases (the canonical registry returns ``None`` for "no match" and has
  no FlyDSL entry yet).
* :func:`detect_language` — a file-level heuristic that combines the
  canonical extension + ``detect_hints`` regex scoring with a v3-specific
  FlyDSL fallback and a definitive ``UNKNOWN`` sentinel.
* :func:`detect_language_for_repo` — a repo-level walker that takes the
  majority vote across kernel-candidate files.

Both functions are intentionally **deterministic and offline** — no
network calls, no LLM calls. They run in pre-step 0b before any subagent
work is dispatched.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path

from minisweagent.kernel_languages import registry
from minisweagent.kernel_languages.base import KernelLanguage

logger = logging.getLogger(__name__)


FLYDSL: KernelLanguage = KernelLanguage(
    name="flydsl",
    file_extensions=frozenset({".fdsl"}),
    detect_hints=(
        r"^import\s+flydsl",
        r"\bfrom\s+flydsl\b",
        r"@flydsl\.",
    ),
    kb_namespace="flydsl",
)
"""Sentinel FlyDSL ``KernelLanguage``.

FlyDSL has no entry in :mod:`minisweagent.kernel_languages` yet (the full
language bundle — prompts, templates, tool_set — lands in a later PR). This
sentinel exists so :func:`detect_language` can return a real
:class:`KernelLanguage` for FlyDSL inputs today. It is **not** registered
into the canonical registry to avoid prematurely advertising a language
bundle that the rest of the codebase doesn't yet support.
"""


UNKNOWN: KernelLanguage = KernelLanguage(
    name="unknown",
    file_extensions=frozenset(),
    detect_hints=(),
    kb_namespace="unknown",
)
"""Sentinel "no signal" ``KernelLanguage``.

Returned by :func:`detect_language` and :func:`detect_language_for_repo`
when no other language matches. Callers should treat
``result.name == "unknown"`` as the explicit "no detection" condition
rather than relying on identity comparison (the registry may return its
own ``KernelLanguage`` instance for the same logical language).
"""


_FLYDSL_FILENAME_TOKEN = "flydsl"
_FLYDSL_EXTENSIONS = frozenset({".fdsl"})

_KERNEL_CANDIDATE_EXTENSIONS = frozenset({".py", ".cu", ".hip", ".cpp", ".cxx", ".fdsl"})

_SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".tox",
        ".nox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "node_modules",
        "build",
        "dist",
        "_build",
        ".eggs",
        ".ipynb_checkpoints",
        ".venv",
        "venv",
        "env",
    }
)


def _read_file_text(path: Path) -> str:
    """Best-effort UTF-8 read; returns ``""`` on any error or for non-files."""
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _score_hints(content: str, hints: tuple[str, ...]) -> int:
    """Return the count of ``hints`` regex patterns that match ``content``."""
    if not content or not hints:
        return 0
    score = 0
    for pat in hints:
        if re.search(pat, content, re.MULTILINE):
            score += 1
    return score


def detect_language(kernel_path: Path) -> KernelLanguage:
    """Detect the kernel language of a single file.

    Resolution order (offline, deterministic):

    1. **FlyDSL fast path** — file extension is ``.fdsl`` OR the filename
       contains ``flydsl`` (case-insensitive). This matches the PR plan's
       ``**/*flydsl*`` glob and wins immediately because FlyDSL has no
       overlap with the .py/.cu extensions claimed by Triton/HIP.

    2. **Content-signal scoring across registered languages** — for each
       language registered in :mod:`minisweagent.kernel_languages` whose
       ``file_extensions`` contains this file's suffix, count how many of
       its ``detect_hints`` regexes match the file content. The highest-
       scoring language with a **non-zero** score wins. This deliberately
       deviates from :func:`registry.detect_best`'s "ext alone is a
       signal" rule — a plain ``.py`` with no Triton-y content is
       ``UNKNOWN``, not silently misclassified as Triton.

    3. **FlyDSL content fallback** — any FlyDSL ``detect_hints`` match
       (``import flydsl`` etc.) returns :data:`FLYDSL`. Useful for files
       whose name doesn't include "flydsl" but whose content does.

    4. :data:`UNKNOWN` sentinel.

    Tie-breaks on equal score sort by ``KernelLanguage.name`` ascending,
    matching :func:`detect_language_for_repo`'s rule so the two functions
    stay consistent.

    Args:
        kernel_path: Path to a single kernel file. Missing files and
            unreadable files fall through to :data:`UNKNOWN` rather than
            raising — pre-step-0b must never crash the pipeline on a
            user-supplied bad path.

    Returns:
        A :class:`KernelLanguage` instance — never ``None``.
    """
    path = Path(kernel_path)
    suffix = path.suffix.lower()

    if suffix in _FLYDSL_EXTENSIONS or _FLYDSL_FILENAME_TOKEN in path.name.lower():
        return FLYDSL

    content = _read_file_text(path)

    scored: list[tuple[int, str, KernelLanguage]] = []
    for lang in registry.all():
        if suffix not in lang.file_extensions:
            continue
        score = _score_hints(content, lang.detect_hints)
        if score > 0:
            scored.append((score, lang.name, lang))

    if scored:
        scored.sort(key=lambda item: (-item[0], item[1]))
        return scored[0][2]

    if _score_hints(content, FLYDSL.detect_hints) > 0:
        return FLYDSL

    return UNKNOWN


def _iter_kernel_candidates(repo_root: Path) -> list[Path]:
    """Yield kernel-candidate files under ``repo_root``.

    A candidate is a file whose suffix is in
    :data:`_KERNEL_CANDIDATE_EXTENSIONS`, found by walking the tree while
    skipping the standard noise directories (``.git``, ``__pycache__``, …).
    Returns a deterministic, sorted list so detect_language_for_repo's
    majority vote is reproducible across platforms.
    """
    found: list[Path] = []
    if not repo_root.is_dir():
        return found

    for entry in sorted(repo_root.rglob("*")):
        if entry.is_dir():
            continue
        if any(part in _SKIP_DIRS for part in entry.parts):
            continue
        if entry.suffix.lower() in _KERNEL_CANDIDATE_EXTENSIONS:
            found.append(entry)
    return found


def detect_language_for_repo(repo_root: Path) -> KernelLanguage:
    """Detect the dominant kernel language of a repository.

    Walks ``repo_root`` collecting kernel-candidate files (see
    :data:`_KERNEL_CANDIDATE_EXTENSIONS`), runs :func:`detect_language`
    on each, and returns the language with the most votes.
    :data:`UNKNOWN` votes are ignored when tallying — they only "win" if
    no other language scored at all.

    Tie-break: when two non-unknown languages tie on vote count, the
    one whose ``name`` sorts first lexicographically wins. This makes
    the function fully deterministic.

    Args:
        repo_root: Path to the repository root. Non-existent paths
            return :data:`UNKNOWN`.

    Returns:
        A :class:`KernelLanguage` instance — never ``None``.
    """
    root = Path(repo_root)
    if not root.is_dir():
        return UNKNOWN

    counter: Counter[str] = Counter()
    by_name: dict[str, KernelLanguage] = {}

    for kernel_path in _iter_kernel_candidates(root):
        detected = detect_language(kernel_path)
        if detected.name == UNKNOWN.name:
            continue
        counter[detected.name] += 1
        by_name.setdefault(detected.name, detected)

    if not counter:
        return UNKNOWN

    # `most_common` is sort-stable but ties depend on insertion order; we
    # apply our own deterministic tiebreak.
    best_count = max(counter.values())
    tied = sorted(name for name, count in counter.items() if count == best_count)
    winning_name = tied[0]
    return by_name[winning_name]


__all__ = [
    "FLYDSL",
    "UNKNOWN",
    "KernelLanguage",
    "detect_language",
    "detect_language_for_repo",
    "registry",
]
