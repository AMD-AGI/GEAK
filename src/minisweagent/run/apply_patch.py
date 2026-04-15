"""Apply the best patch from a completed optimization run to the original repo."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from minisweagent.run.utils.generated_artifacts import apply_patch_with_generated_helper_fallback

logger = logging.getLogger(__name__)


def apply_best_patch(report_path: Path, repo: Path) -> bool:
    """Read *final_report.json* and apply its ``best_patch`` to *repo*.

    Returns ``True`` when the patch was applied successfully, ``False``
    otherwise (missing patch, apply failure, etc.).  Errors are logged
    but never raised so the caller can treat this as a best-effort step.
    """
    if not report_path.is_file():
        logger.warning("apply_best_patch: report not found: %s", report_path)
        return False

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("apply_best_patch: failed to read report: %s", exc)
        return False

    status = report.get("status", "")
    if status in ("error", "complete_no_patch"):
        logger.warning("apply_best_patch: skipping — report status is '%s'.", status)
        return False

    best_patch = report.get("best_patch")
    if not best_patch:
        logger.warning("apply_best_patch: no best_patch in report.")
        return False

    patch_path = Path(best_patch)
    if not patch_path.is_file():
        logger.warning("apply_best_patch: patch file does not exist: %s", patch_path)
        return False
    if patch_path.stat().st_size == 0:
        logger.warning("apply_best_patch: patch file is empty: %s", patch_path)
        return False

    repo = repo.resolve()
    if not repo.is_dir():
        logger.warning("apply_best_patch: repo path does not exist: %s", repo)
        return False

    speedup = report.get("best_speedup")
    if isinstance(speedup, (int, float)) and speedup <= 1.0:
        logger.warning(
            "apply_best_patch: best_speedup is %.4f (no improvement); applying anyway.",
            speedup,
        )

    try:
        patch_text = patch_path.read_text(encoding="utf-8", errors="replace")
        result, removed_paths = apply_patch_with_generated_helper_fallback(
            patch_text=patch_text,
            cwd=repo,
        )
    except Exception as exc:
        logger.error("apply_best_patch: unexpected error: %s", exc)
        return False

    if removed_paths:
        logger.info(
            "apply_best_patch: stripped generated artifacts: %s",
            ", ".join(removed_paths),
        )

    if result.returncode != 0:
        logger.error(
            "apply_best_patch: git apply failed (rc=%d): %s",
            result.returncode,
            result.stderr,
        )
        return False

    logger.info("apply_best_patch: successfully applied %s to %s", patch_path.name, repo)
    return True
