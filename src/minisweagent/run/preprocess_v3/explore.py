"""Step 1 — deterministic codebase-explore for the v3 preprocess pipeline.

Wraps the existing :mod:`minisweagent.run.preprocess.codebase_context`
``CODEBASE_CONTEXT.md`` generator under a clean v3 contract:

* The public surface is :func:`explore_codebase`, which returns a
  frozen :class:`CodebaseContext` dataclass carrying the rendered
  markdown text, the list of in-repo files discovered by the kernel
  dependency BFS, and the optional path to the written file.
* Detection inputs (``kernel_path`` + ``kernel_language``) are the
  v3 caller's responsibility; this module never re-runs language
  detection — that's pre-step-0b's job (see
  :mod:`minisweagent.run.preprocess_v3.lang`).
* The step is **strictly deterministic**: it reads files, walks
  imports / ``#include`` directives, and renders markdown. There are
  no LLM calls, no network access, and no side effects beyond
  optionally writing ``CODEBASE_CONTEXT.md``.

The legacy implementation under ``run/preprocess/`` does the heavy
lifting (directory tree pruning, AST-based Python import parsing,
regex-based C/C++ ``#include`` parsing, BFS dependency walk). Once
``commit-set-5`` lands the v3 namespace becomes the single source
of truth and this wrapper inlines those helpers; until then the
wrapper exists to keep the v3 boundary clean.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from minisweagent.kernel_languages.base import KernelLanguage

# TODO(commit-set-5): inline; old preprocess/ goes away
from minisweagent.run.preprocess.codebase_context import (
    _build_dependency_tree,
    _build_directory_tree,
    generate_codebase_context,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CodebaseContext:
    """Result of :func:`explore_codebase`.

    Attributes:
        text:
            The rendered ``CODEBASE_CONTEXT.md`` markdown content.
            Always non-empty: at minimum it contains the repository
            layout tree and the target kernel marker.
        files:
            In-repo files discovered by the kernel dependency BFS.
            The list is paths relative to ``repo_root`` (mirroring the
            legacy generator's own ``rel_file`` rendering) and always
            starts with the target kernel itself so callers can use
            ``files[0]`` as the kernel anchor.
        out_path:
            Filesystem path to the written ``CODEBASE_CONTEXT.md``
            when ``out_path`` was supplied to :func:`explore_codebase`,
            or ``None`` when the caller asked for in-memory rendering
            only.
        kernel_language:
            The :class:`KernelLanguage` the caller routed through. We
            keep a reference so downstream steps (commandment render,
            baseline metrics) don't have to thread the language
            separately when they already have the explore result.
    """

    text: str
    files: list[str] = field(default_factory=list)
    out_path: Path | None = None
    kernel_language: KernelLanguage | None = None


def explore_codebase(
    repo_root: Path,
    kernel_path: Path,
    kernel_language: KernelLanguage,
    *,
    out_path: Path | None = None,
) -> CodebaseContext:
    """Generate a deterministic codebase briefing for the orchestrator.

    Builds a pruned repository directory tree plus a kernel-rooted
    dependency tree (BFS over imports / includes, restricted to files
    inside *repo_root*) and renders both as ``CODEBASE_CONTEXT.md``.
    The exact rendering matches
    :func:`minisweagent.run.preprocess.codebase_context.generate_codebase_context`
    by construction — this function is a thin v3-side wrapper that
    formalises the return type and decouples the caller from the
    legacy module location.

    Args:
        repo_root:
            Root directory of the cloned repository (typically the
            ``baseline/`` tree produced by
            :func:`minisweagent.run.preprocess_v3.clone.split_repo_for_baseline_and_eval`).
        kernel_path:
            Path to the target kernel file. Must live inside
            ``repo_root``; the legacy generator falls back to absolute
            paths when this isn't true, but the v3 contract expects
            an in-tree kernel.
        kernel_language:
            :class:`KernelLanguage` resolved by pre-step-0b. Carried
            through onto :class:`CodebaseContext` so downstream code
            can route language-specific formatting without re-detecting.
            Not currently used by the legacy generator (it inspects
            file suffixes directly), but recorded for the v3 surface
            to remain stable.
        out_path:
            When supplied, the rendered markdown is written here and
            ``CodebaseContext.out_path`` reports back the resolved
            absolute path. When ``None``, the function only returns
            the in-memory render — useful for tests, dry-runs, and
            callers that want to post-process before committing to
            disk.

    Returns:
        A :class:`CodebaseContext` carrying the rendered markdown,
        the BFS-discovered in-repo dependency files (kernel first),
        and the path the file was written to (when ``out_path`` was
        given).

    Raises:
        FileNotFoundError: If ``repo_root`` is not a directory or
            ``kernel_path`` is not a regular file.
    """
    repo_root_resolved = Path(repo_root).resolve()
    kernel_path_resolved = Path(kernel_path).resolve()

    if not repo_root_resolved.is_dir():
        raise FileNotFoundError(f"explore_codebase: repo_root not a directory: {repo_root_resolved}")
    if not kernel_path_resolved.is_file():
        raise FileNotFoundError(f"explore_codebase: kernel_path not a file: {kernel_path_resolved}")

    # Compute the kernel's repo-relative path for the file list. The
    # legacy generator prints relative paths in the dependency table
    # and falls back to absolute on ValueError; we mirror the same
    # behaviour so CodebaseContext.files agrees with what the markdown
    # claims.
    try:
        rel_kernel = str(kernel_path_resolved.relative_to(repo_root_resolved))
    except ValueError:
        rel_kernel = str(kernel_path_resolved)

    deps = _build_dependency_tree(repo_root_resolved, kernel_path_resolved)
    files: list[str] = [rel_kernel]
    for dep in deps:
        dep_file = dep.get("file")
        if dep_file and dep_file not in files:
            files.append(dep_file)

    written_path: Path | None = None
    if out_path is not None:
        out_path_resolved = Path(out_path).resolve()
        # The legacy generator wants an output *directory* and writes
        # ``CODEBASE_CONTEXT.md`` inside it; the v3 contract takes a
        # *file* path so callers can pin the exact filename. We honor
        # the caller's exact path by routing through the directory
        # form and renaming when the target name differs.
        target_dir = out_path_resolved.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        legacy_path = generate_codebase_context(
            repo_root_resolved,
            kernel_path_resolved,
            target_dir,
        )
        if legacy_path != out_path_resolved:
            content = legacy_path.read_text(encoding="utf-8")
            out_path_resolved.write_text(content, encoding="utf-8")
            if legacy_path != out_path_resolved and legacy_path.exists():
                # Avoid leaving a stray ``CODEBASE_CONTEXT.md`` next
                # to the user-chosen filename when they differ.
                legacy_path.unlink()
        written_path = out_path_resolved
        text = written_path.read_text(encoding="utf-8")
    else:
        # In-memory render — replicate the legacy section assembly so
        # we don't touch disk. Keep this aligned with
        # ``generate_codebase_context`` so tests catch any drift.
        sections: list[str] = ["# Codebase Context\n"]
        tree = _build_directory_tree(repo_root_resolved, kernel_path_resolved)
        sections.append("## Repository Layout\n")
        sections.append(f"```\n{tree}\n```\n")
        sections.append("## Kernel Dependency Tree\n")
        sections.append(f"Target kernel: `{rel_kernel}`\n")

        by_depth: dict[int, list[dict]] = {}
        for dep in deps:
            by_depth.setdefault(dep["depth"], []).append(dep)

        for depth in sorted(by_depth):
            if depth == 1:
                sections.append("### Direct dependencies\n")
                sections.append("| File | Imports | Description |")
                sections.append("|------|---------|-------------|")
                for dep in by_depth[depth]:
                    names = ", ".join(f"`{n}`" for n in dep["names"]) if dep["names"] else "*module*"
                    sections.append(f"| `{dep['file']}` | {names} | {dep['description']} |")
            else:
                sections.append(f"\n### Transitive dependencies (depth {depth})\n")
                sections.append("Improving these may improve the target kernel's performance.\n")
                sections.append("| File | Imports | Used by | Description |")
                sections.append("|------|---------|---------|-------------|")
                for dep in by_depth[depth]:
                    names = ", ".join(f"`{n}`" for n in dep["names"]) if dep["names"] else "*module*"
                    sections.append(f"| `{dep['file']}` | {names} | `{dep['imported_by']}` | {dep['description']} |")
            sections.append("")

        if not deps:
            sections.append("No in-repo dependencies found.\n")

        text = "\n".join(sections)

    return CodebaseContext(
        text=text,
        files=files,
        out_path=written_path,
        kernel_language=kernel_language,
    )


__all__ = ["CodebaseContext", "explore_codebase"]
