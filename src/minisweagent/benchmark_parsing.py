"""Deterministic benchmark output parsing and patch selection.

Provides regex-based extraction of latency metrics from harness output
and a ``compute_best_patch()`` function that selects the best non-empty
patch by comparing benchmark numbers -- no LLM involved.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def parse_median_latency_ms(output: str) -> float | None:
    """Extract median latency (ms) from harness benchmark output."""
    m = re.search(
        r"(?:median\s+(?:latency|time)[\w\s]*|total\s+median\s+time)\s*:\s*([\d.]+(?:e[+-]?\d+)?)\s*ms",
        output,
        re.IGNORECASE,
    )
    return float(m.group(1)) if m else None


def parse_total_kernel_time_ms(output: str) -> float | None:
    """Extract TOTAL_KERNEL_TIME_MS or BENCHMARK_LATENCY_MS from harness benchmark output."""
    m = re.search(
        r"(?:TOTAL_KERNEL_TIME_MS|BENCHMARK_LATENCY_MS):\s*([\d.]+(?:e[+-]?\d+)?)",
        output,
    )
    return float(m.group(1)) if m else None


def parse_shape_count(output: str) -> int | None:
    """Extract shape count from harness benchmark output."""
    m = re.search(r"(\d+)\s+shapes", output, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _extract_latency(text: str) -> float | None:
    """Try all parsers in priority order and return the first match."""
    val = parse_total_kernel_time_ms(text)
    if val is not None:
        return val
    return parse_median_latency_ms(text)


def compute_best_patch(patch_dir: Path) -> dict[str, Any] | None:
    """Deterministically select the best non-empty patch from a task directory.

    Reads ``patch_0_test.txt`` as the baseline benchmark output, then
    compares every ``patch_N_test.txt`` (N >= 1) whose corresponding
    ``patch_N.patch`` is non-empty.  Returns a dict compatible with
    ``best_results.json``, or *None* if no valid candidate exists.

    The returned dict has the same schema the LLM-based SelectPatchAgent
    would produce, so downstream code (orchestrator, ParallelAgent) can
    consume it unchanged.
    """
    baseline_file = patch_dir / "patch_0_test.txt"
    if not baseline_file.exists():
        return None

    baseline_text = baseline_file.read_text()
    baseline_ms = _extract_latency(baseline_text)
    if baseline_ms is None or baseline_ms <= 0:
        return None

    best_speedup = 0.0
    best_patch_id: str | None = None
    best_patch_file: str | None = None
    best_test_file: str | None = None

    for test_file in sorted(patch_dir.glob("patch_*_test.txt")):
        name = test_file.stem.replace("_test", "")  # e.g. "patch_3"
        if name == "patch_0":
            continue

        patch_file = patch_dir / f"{name}.patch"
        if not patch_file.exists():
            continue
        if patch_file.stat().st_size == 0:
            continue

        candidate_text = test_file.read_text()
        candidate_ms = _extract_latency(candidate_text)
        if candidate_ms is None or candidate_ms <= 0:
            continue

        speedup = baseline_ms / candidate_ms
        if speedup > best_speedup:
            best_speedup = speedup
            best_patch_id = name
            best_patch_file = str(patch_file)
            best_test_file = str(test_file)

    if best_patch_id is None:
        return None

    return {
        "best_patch_id": best_patch_id,
        "best_patch_speedup": round(best_speedup, 6),
        "best_patch_file": best_patch_file,
        "best_patch_test_output": best_test_file,
        "llm_selection_analysis": (
            f"Deterministic selection: baseline={baseline_ms:.4f}ms, "
            f"best candidate latency from {best_patch_id}. "
            f"Speedup={best_speedup:.4f}x."
        ),
    }


def rewrite_best_results(patch_dir: Path) -> dict[str, Any] | None:
    """Overwrite ``best_results.json`` with deterministic selection if possible.

    Falls back to the existing ``best_results.json`` if deterministic
    parsing cannot extract metrics (e.g. non-standard harness output).
    Also rejects the existing result if it references an empty patch.

    Returns the final best_results dict, or None.
    """
    det = compute_best_patch(patch_dir)
    existing_path = patch_dir / "best_results.json"

    if det is not None:
        existing_path.write_text(json.dumps(det, indent=2))
        logger.info(
            "Deterministic best_results for %s: %s (%.4fx)",
            patch_dir.name,
            det["best_patch_id"],
            det["best_patch_speedup"],
        )
        return det

    # Fallback: validate the existing LLM-written result
    if existing_path.exists():
        try:
            existing = json.loads(existing_path.read_text())
            pf = existing.get("best_patch_file")
            if pf and Path(pf).exists() and Path(pf).stat().st_size == 0:
                existing["best_patch_speedup"] = 1.0
                existing["llm_selection_analysis"] = (
                    (existing.get("llm_selection_analysis") or "")
                    + " [Overridden: patch is empty (0 bytes), speedup clamped to 1.0]"
                )
                existing_path.write_text(json.dumps(existing, indent=2))
                logger.info(
                    "Clamped empty-patch speedup to 1.0 for %s", patch_dir.name
                )
            return existing
        except (json.JSONDecodeError, ValueError):
            pass

    return None
