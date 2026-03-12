"""Canonical testcase / harness reuse across experiments.

The goal is apples-to-apples comparison: once a kernel has a validated harness,
later experiments should reuse the same testcase selection instead of letting
discovery or UnitTestAgent drift to a different harness.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def get_testcase_cache_dir() -> Path | None:
    """Return the shared testcase cache directory, if configured."""
    raw = os.environ.get("GEAK_TESTCASE_CACHE_DIR", "").strip()
    if not raw:
        return None
    path = Path(raw).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_testcase_cache_key(kernel_url: str, kernel_path: str | Path) -> str:
    """Build a stable cache key for a kernel."""
    kernel_path = Path(kernel_path)
    slug = "".join(ch if ch.isalnum() else "-" for ch in kernel_path.stem.lower()).strip("-") or "kernel"
    digest = hashlib.sha256(kernel_url.strip().encode("utf-8")).hexdigest()[:12]
    return f"{slug}-{digest}"


def get_testcase_cache_entry(cache_dir: Path, cache_key: str) -> Path:
    """Return the directory for a single kernel's cached testcase selection."""
    entry = cache_dir / cache_key
    entry.mkdir(parents=True, exist_ok=True)
    return entry


def _rewrite_paths(text: str, replacements: dict[str, str]) -> str:
    if not text:
        return text
    updated = text
    for old, new in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        if old and new and old != new:
            updated = updated.replace(old, new)
    return updated


def _read_manifest(entry_dir: Path) -> dict[str, Any] | None:
    path = entry_dir / "manifest.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def materialize_cached_harness(
    entry_dir: Path,
    *,
    repo_root: str | Path,
    output_dir: str | Path,
    kernel_path: str | Path,
) -> tuple[str, str, dict[str, Any]] | None:
    """Materialize a cached harness selection for the current experiment."""
    manifest = _read_manifest(entry_dir)
    if not manifest:
        return None

    repo_root = str(Path(repo_root).resolve())
    output_dir = Path(output_dir).resolve()
    kernel_path = str(Path(kernel_path).resolve())

    source_harness = str(manifest.get("source_harness_path", "")).strip()
    source_test_command = str(manifest.get("source_test_command", "")).strip()
    source_repo_root = str(manifest.get("source_repo_root", "")).strip()
    source_output_dir = str(manifest.get("source_output_dir", "")).strip()
    source_kernel_path = str(manifest.get("source_kernel_path", "")).strip()
    source_type = str(manifest.get("source_type", "")).strip()

    if not source_harness or not source_test_command or not source_type:
        return None

    replacements = {
        source_repo_root: repo_root,
        source_output_dir: str(output_dir),
        source_kernel_path: kernel_path,
    }

    if source_type == "repo_relative":
        relpath = str(manifest.get("relative_harness_path", "")).strip()
        if not relpath:
            return None
        harness_path = str((Path(repo_root) / relpath).resolve())
        if not Path(harness_path).is_file():
            return None
        test_command = _rewrite_paths(
            source_test_command,
            {**replacements, source_harness: harness_path},
        )
        return test_command, harness_path, manifest

    if source_type == "standalone":
        snapshot_name = str(manifest.get("snapshot_name", "")).strip()
        if not snapshot_name:
            return None
        snapshot_path = entry_dir / snapshot_name
        if not snapshot_path.is_file():
            return None
        target_name = str(manifest.get("materialized_name", "")).strip() or f"cached_{Path(source_harness).name}"
        harness_path = output_dir / target_name
        harness_path.parent.mkdir(parents=True, exist_ok=True)
        text = snapshot_path.read_text()
        text = _rewrite_paths(
            text,
            {
                **replacements,
                source_harness: str(harness_path),
            },
        )
        harness_path.write_text(text)
        test_command = _rewrite_paths(
            source_test_command,
            {
                **replacements,
                source_harness: str(harness_path),
            },
        )
        return test_command, str(harness_path), manifest

    return None


def save_cached_harness(
    entry_dir: Path,
    *,
    kernel_url: str,
    source: str,
    test_command: str,
    harness_path: str | Path,
    repo_root: str | Path,
    output_dir: str | Path,
    kernel_path: str | Path,
    harness_results: list[dict[str, Any]] | None = None,
) -> Path | None:
    """Persist the chosen validated harness selection for later experiments."""
    harness_path = Path(harness_path).resolve()
    if not harness_path.is_file():
        return None

    repo_root_path = Path(repo_root).resolve()
    output_dir_path = Path(output_dir).resolve()
    kernel_path = Path(kernel_path).resolve()
    entry_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "version": 1,
        "kernel_url": kernel_url,
        "source": source,
        "source_test_command": test_command,
        "source_harness_path": str(harness_path),
        "source_repo_root": str(repo_root_path),
        "source_output_dir": str(output_dir_path),
        "source_kernel_path": str(kernel_path),
        "harness_results": [
            {
                "mode": r.get("mode"),
                "success": bool(r.get("success")),
                "returncode": r.get("returncode"),
            }
            for r in (harness_results or [])
        ],
    }

    try:
        rel = harness_path.relative_to(repo_root_path)
    except ValueError:
        rel = None

    if rel is not None:
        manifest["source_type"] = "repo_relative"
        manifest["relative_harness_path"] = str(rel)
    else:
        manifest["source_type"] = "standalone"
        snapshot_name = f"snapshot{harness_path.suffix or '.py'}"
        materialized_name = f"cached_{harness_path.name}"
        (entry_dir / snapshot_name).write_text(harness_path.read_text())
        manifest["snapshot_name"] = snapshot_name
        manifest["materialized_name"] = materialized_name

    manifest_path = entry_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path
