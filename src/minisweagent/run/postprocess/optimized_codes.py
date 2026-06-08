"""Snapshot the post-patch state of all files touched by the best patch.

``final_report.json`` historically pointed at ``best_patch`` only, leaving
downstream consumers to figure out which files actually changed and what
their post-patch contents look like. This module reconstructs that view:

  * Replays the best patch onto a clean baseline checkout of ``repo_root``.
  * Records every added / modified / deleted / renamed path via
    ``git diff --name-status``.
  * Copies the post-patch contents of added and modified files into
    ``<output_dir>/optimized_codes/`` preserving their repo-relative
    layout, so the directory is a self-contained, immediately usable
    snapshot of "the optimized codebase".
  * Returns a manifest dict suitable for direct embedding under the
    ``optimized_codes`` key of ``final_report.json``.

The function is best-effort: any failure (missing patch, apply error, etc.)
is reported as a structured ``{"status": "skipped", "reason": ...}`` so the
caller can safely embed the result without branching.
"""

from __future__ import annotations

import difflib
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from minisweagent.run.postprocess.evaluation import (
    PatchApplyError,
    cleanup_eval_worktree,
    setup_eval_worktree,
)
from minisweagent.run.utils.generated_artifacts import is_jit_cache_artifact
from minisweagent.run.utils.git_safe_env import get_git_safe_env

logger = logging.getLogger(__name__)

DEFAULT_TARGET_DIR_NAME = "optimized_codes"

# Stop reconstructing if the snapshot would exceed this size. Protects against
# accidental copying of huge binary artifacts or build outputs that slipped
# into the patch.
_MAX_TOTAL_BYTES = 256 * 1024 * 1024  # 256 MiB


def _git_diff_name_status(eval_dir: Path) -> list[tuple[str, list[str]]]:
    """Return ``[(status_code, paths)]`` from ``git diff --name-status HEAD``.

    Rename / copy statuses (``R100``, ``C075`` etc.) yield two paths
    ``[old, new]``; every other status yields a single path.

    ``git apply`` leaves newly created files untracked, so a plain
    ``git diff HEAD`` misses them. We first mark untracked files with
    ``git add -N`` (intent-to-add) so they show up as additions in the
    diff without staging any content; this is reversible/no-op and uses
    rename-detection (``-M``) so renames are properly classified rather
    than reported as add+delete pairs.
    """
    git_env = get_git_safe_env(eval_dir.parent)
    subprocess.run(
        ["git", "add", "-N", "--", "."],
        cwd=str(eval_dir),
        capture_output=True,
        text=True,
        env=git_env,
        timeout=60,
    )
    result = subprocess.run(
        ["git", "diff", "--name-status", "-M", "-z", "HEAD"],
        cwd=str(eval_dir),
        capture_output=True,
        text=True,
        env=git_env,
        timeout=60,
    )
    if result.returncode != 0:
        logger.warning(
            "git diff --name-status failed in %s (rc=%s): %s",
            eval_dir,
            result.returncode,
            result.stderr.strip(),
        )
        return []

    # -z output: NUL-separated fields. For R/C entries, status and old/new
    # paths arrive as three consecutive fields; other statuses use two.
    tokens = result.stdout.split("\x00")
    entries: list[tuple[str, list[str]]] = []
    i = 0
    while i < len(tokens):
        status = tokens[i]
        if not status:
            i += 1
            continue
        i += 1
        if status[0] in ("R", "C"):
            if i + 1 >= len(tokens):
                break
            entries.append((status, [tokens[i], tokens[i + 1]]))
            i += 2
        else:
            if i >= len(tokens):
                break
            entries.append((status, [tokens[i]]))
            i += 1
    return entries


def _classify_entries(
    entries: list[tuple[str, list[str]]],
) -> tuple[list[str], list[str], list[str], list[dict[str, str]]]:
    """Bucket name-status entries into added / modified / deleted / renamed."""
    added: list[str] = []
    modified: list[str] = []
    deleted: list[str] = []
    renamed: list[dict[str, str]] = []
    for status, paths in entries:
        head = status[0]
        if head == "A":
            added.append(paths[0])
        elif head == "M":
            modified.append(paths[0])
        elif head == "D":
            deleted.append(paths[0])
        elif head in ("R", "C"):
            if len(paths) == 2:
                renamed.append({"from": paths[0], "to": paths[1]})
        elif head == "T":
            # Type change (e.g. file -> symlink). Treat as modified so the
            # post-patch artifact still gets snapshotted.
            modified.append(paths[0])
    return added, modified, deleted, renamed


def _drop_jit_cache_paths(
    added: list[str],
    modified: list[str],
    deleted: list[str],
    renamed: list[dict[str, str]],
) -> tuple[
    list[str],
    list[str],
    list[str],
    list[dict[str, str]],
    list[str],
]:
    """Filter JIT-cache paths out of the classified diff entries.

    Layer B of the "no JIT pkls in final_report.optimized_codes" fix:
    even if the upstream patch capture in ``save_and_test`` failed to
    strip a flydsl_cache pkl (e.g. brand-new JIT cache pattern we haven't
    enumerated yet), final_report.json still ends up clean here. The
    function returns the four scrubbed lists plus a sorted list of the
    paths that were dropped, so the manifest can expose what was filtered.
    """

    kept_added = [p for p in added if not is_jit_cache_artifact(p)]
    kept_modified = [p for p in modified if not is_jit_cache_artifact(p)]
    kept_deleted = [p for p in deleted if not is_jit_cache_artifact(p)]
    kept_renamed: list[dict[str, str]] = []
    dropped: set[str] = set()
    dropped.update(p for p in added if is_jit_cache_artifact(p))
    dropped.update(p for p in modified if is_jit_cache_artifact(p))
    dropped.update(p for p in deleted if is_jit_cache_artifact(p))
    for entry in renamed:
        to_path = entry.get("to", "")
        from_path = entry.get("from", "")
        if is_jit_cache_artifact(to_path) or is_jit_cache_artifact(from_path):
            if to_path:
                dropped.add(to_path)
            if from_path and from_path != to_path:
                dropped.add(from_path)
            continue
        kept_renamed.append(entry)
    return kept_added, kept_modified, kept_deleted, kept_renamed, sorted(dropped)


def _copy_snapshot(
    eval_dir: Path,
    target_dir: Path,
    rel_paths: list[str],
) -> tuple[list[str], int]:
    """Copy each ``rel_paths`` from ``eval_dir`` to ``target_dir`` preserving
    the relative layout. Returns ``(files_copied, total_bytes)``.
    """
    copied: list[str] = []
    total = 0
    for rel in rel_paths:
        # Reject anything that tries to escape the worktree via .. segments
        # or absolute paths (defense in depth; git should never produce them).
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            logger.warning("Skipping suspicious diff path %r", rel)
            continue

        src = eval_dir / rel_path
        if not src.is_file() and not src.is_symlink():
            logger.debug("Skipping non-file diff entry: %s", src)
            continue

        dst = target_dir / rel_path
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_symlink():
                link_target = os.readlink(src)
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                dst.symlink_to(link_target)
            else:
                shutil.copy2(src, dst)
                total += dst.stat().st_size
        except OSError as exc:
            logger.warning("Failed to snapshot %s -> %s: %s", src, dst, exc)
            continue

        copied.append(str(rel_path))
        if total > _MAX_TOTAL_BYTES:
            logger.warning(
                "optimized_codes snapshot exceeded size cap (%d bytes); truncating.",
                _MAX_TOTAL_BYTES,
            )
            break
    return copied, total


def _reconstruct_from_translation_snapshot(
    output_dir: Path,
    target_dir: Path,
) -> dict[str, Any] | None:
    """Rebuild the optimized-code view directly from translation snapshots.

    The worktree-based patch capture (``save_and_test`` -> ``git diff`` in
    ``GEAK_WORK_DIR``) cannot see the FlyDSL kernel under optimization,
    because that kernel lives in ``output_dir`` (e.g. ``kernel_foo_flydsl.py``)
    rather than inside the bootstrapped worktree (a copy of ``repo_root``).
    The resulting ``*.patch`` artifacts therefore capture only the
    auto-generated ``.gitignore`` and drop every real kernel change.

    To recover the true translation->optimization delta we diff the
    pre-optimization snapshot ``<stem>.translated.py`` (written by the
    translate stage) against the final in-place kernel ``<stem>.py``. When at
    least one such pair differs we emit a real unified patch
    (``output_dir/optimization.patch``) and snapshot the optimized kernels
    into ``target_dir``.

    Returns a manifest dict on success, or ``None`` to signal the caller to
    fall back to the legacy patch-replay path (no snapshots / no diffs).
    """
    snapshots = sorted(output_dir.glob("*.translated.py"))
    if not snapshots:
        return None

    patch_chunks: list[str] = []
    changed_files: list[str] = []
    unchanged_files: list[str] = []

    for snap in snapshots:
        stem = snap.name[: -len(".translated.py")]
        final = snap.with_name(f"{stem}.py")
        if not final.is_file():
            continue
        try:
            before = snap.read_text().splitlines(keepends=True)
            after = final.read_text().splitlines(keepends=True)
        except OSError as exc:
            logger.warning("snapshot reconstruction: could not read %s/%s: %s", snap, final, exc)
            continue
        if before == after:
            unchanged_files.append(final.name)
            continue
        diff = difflib.unified_diff(before, after, fromfile=f"a/{final.name}", tofile=f"b/{final.name}")
        chunk = "".join(diff)
        if not chunk.strip():
            unchanged_files.append(final.name)
            continue
        patch_chunks.append(f"diff --git a/{final.name} b/{final.name}\n{chunk}")
        changed_files.append(final.name)

    if not patch_chunks:
        return None

    patch_text = "\n".join(patch_chunks)
    if not patch_text.endswith("\n"):
        patch_text += "\n"
    patch_out: Path | None = output_dir / "optimization.patch"
    try:
        patch_out.write_text(patch_text)
    except OSError as exc:
        logger.warning("snapshot reconstruction: could not write %s: %s", patch_out, exc)
        patch_out = None

    if target_dir.exists():
        shutil.rmtree(target_dir, ignore_errors=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    copied: list[str] = []
    for name in changed_files:
        src = output_dir / name
        try:
            shutil.copy2(src, target_dir / name)
            total_bytes += (target_dir / name).stat().st_size
            copied.append(name)
        except OSError as exc:
            logger.warning("snapshot reconstruction: could not copy %s: %s", src, exc)

    logger.info(
        "collect_optimized_codes: reconstructed %d optimized kernel(s) from translation "
        "snapshot diff (%d unchanged) -> %s",
        len(copied),
        len(unchanged_files),
        target_dir,
    )
    return {
        "status": "complete",
        "source": "translation_snapshot_diff",
        "directory": str(target_dir),
        "patch": str(patch_out) if patch_out else None,
        "files": sorted(copied),
        "added": [],
        "modified": sorted(copied),
        "deleted": [],
        "renamed": [],
        "unchanged": sorted(unchanged_files),
        "total_bytes": total_bytes,
        "filtered_jit_cache": [],
    }


def collect_optimized_codes(
    repo_root: str | Path,
    patch_file: str | Path | None,
    output_dir: str | Path,
    *,
    target_dir_name: str = DEFAULT_TARGET_DIR_NAME,
) -> dict[str, Any]:
    """Materialize the post-patch state of files touched by *patch_file*.

    Args:
        repo_root: Repository the patch was generated against.
        patch_file: Best patch from the GEAK run, or ``None``.
        output_dir: Directory containing ``final_report.json``; the snapshot
            lands at ``output_dir/<target_dir_name>``.
        target_dir_name: Override the snapshot subdirectory name.

    Returns:
        A manifest dict ready to embed under ``final_report.json``'s
        ``optimized_codes`` key. Never raises.
    """
    output_dir = Path(output_dir).resolve()
    target_dir = output_dir / target_dir_name

    # Preferred path: rebuild the optimized view from the translation snapshot
    # (<stem>.translated.py vs <stem>.py). The worktree git-diff patch capture
    # cannot see the FlyDSL kernel (it lives in output_dir), so the *.patch
    # artifacts only contain .gitignore; diffing the actual kernel files
    # recovers the real translation->optimization delta.
    try:
        snapshot_manifest = _reconstruct_from_translation_snapshot(output_dir, target_dir)
    except Exception as exc:  # never let reconstruction abort the report
        logger.warning("snapshot reconstruction raised: %s", exc, exc_info=True)
        snapshot_manifest = None
    if snapshot_manifest is not None:
        return snapshot_manifest

    if not patch_file:
        return {"status": "skipped", "reason": "no_best_patch"}

    patch_path = Path(patch_file)
    if not patch_path.is_file():
        return {"status": "skipped", "reason": "patch_file_missing", "patch": str(patch_path)}
    if patch_path.stat().st_size == 0:
        return {"status": "skipped", "reason": "empty_patch", "patch": str(patch_path)}

    repo = Path(repo_root)
    if not repo.exists():
        return {"status": "skipped", "reason": "repo_root_missing", "repo_root": str(repo)}

    eval_dir: Path | None = None
    try:
        try:
            eval_dir = setup_eval_worktree(str(repo), str(patch_path), output_dir)
        except PatchApplyError as exc:
            logger.warning(
                "collect_optimized_codes: patch did not apply against %s: %s",
                repo,
                exc,
            )
            return {"status": "skipped", "reason": "patch_apply_failed", "error": str(exc)}

        entries = _git_diff_name_status(eval_dir)
        added, modified, deleted, renamed = _classify_entries(entries)

        added, modified, deleted, renamed, filtered_jit_cache = _drop_jit_cache_paths(added, modified, deleted, renamed)
        if filtered_jit_cache:
            logger.info(
                "collect_optimized_codes: dropped %d JIT-cache path(s) from snapshot (sample: %s)",
                len(filtered_jit_cache),
                ", ".join(filtered_jit_cache[:3]),
            )

        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        target_dir.mkdir(parents=True, exist_ok=True)

        snapshot_paths = sorted(set(added)) + sorted(set(modified)) + sorted(r["to"] for r in renamed)
        files_copied, total_bytes = _copy_snapshot(eval_dir, target_dir, snapshot_paths)

        manifest: dict[str, Any] = {
            "status": "complete",
            "directory": str(target_dir),
            "files": sorted(files_copied),
            "added": sorted(added),
            "modified": sorted(modified),
            "deleted": sorted(deleted),
            "renamed": sorted(renamed, key=lambda r: (r.get("from", ""), r.get("to", ""))),
            "total_bytes": total_bytes,
            "filtered_jit_cache": filtered_jit_cache,
        }
        logger.info(
            "collect_optimized_codes: snapshotted %d file(s) (%d added, %d modified, %d deleted, %d renamed, "
            "%d JIT-cache dropped) totalling %d bytes at %s",
            len(files_copied),
            len(added),
            len(modified),
            len(deleted),
            len(renamed),
            len(filtered_jit_cache),
            total_bytes,
            target_dir,
        )
        return manifest

    except Exception as exc:  # belt-and-suspenders -- never abort the report
        logger.warning("collect_optimized_codes failed: %s", exc, exc_info=True)
        return {"status": "skipped", "reason": "unexpected_error", "error": str(exc)}
    finally:
        if eval_dir is not None:
            try:
                cleanup_eval_worktree(str(repo), eval_dir)
            except Exception as exc:
                logger.debug("optimized_codes eval worktree cleanup failed: %s", exc)
