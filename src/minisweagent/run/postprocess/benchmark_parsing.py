"""Deterministic benchmark output parsing and patch selection.

Provides regex-based extraction of latency metrics from harness output
and a ``compute_best_patch()`` function that selects the best non-empty
patch by comparing benchmark numbers -- no LLM involved.

Measurement methodology:
- Uses ``benchmark_baseline.txt`` (the canonical unmodified baseline benchmark)
- Prioritizes ``GEAK_RESULT_LATENCY_MS=<number>`` marker (standardized)
- Falls back to legacy parsers and universal latency keyword scanner
- Only reports speedups > 1.0 (genuine improvements over true baseline)
- Clamps LLM-inflated results to 1.0 when no real improvement exists
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from minisweagent.run import benchmark_output as _benchmark_output

logger = logging.getLogger(__name__)

compute_shape_speedups = _benchmark_output.compute_shape_speedups
extract_benchmark_config_lines = _benchmark_output.extract_benchmark_config_lines
extract_latency_ms = _benchmark_output.extract_latency_ms
extract_reported_speedup = _benchmark_output.extract_reported_speedup
parse_google_benchmark_ms = _benchmark_output.parse_google_benchmark_ms
parse_median_latency_ms = _benchmark_output.parse_median_latency_ms
parse_shape_count = _benchmark_output.parse_shape_count
parse_shape_latencies_ms = _benchmark_output.parse_shape_latencies_ms
parse_total_kernel_time_ms = _benchmark_output.parse_total_kernel_time_ms


def _find_original_baseline_ms(patch_dir: Path) -> float | None:
    """Walk up from patch_dir to find benchmark_baseline.txt (the canonical baseline).

    The preprocessing phase writes benchmark_baseline.txt at the kernel
    output root (e.g. patches/exp0/rope/benchmark_baseline.txt).  Task dirs
    are nested under results/round_N/strategy_name, so we walk upward.
    """
    d = patch_dir
    for _ in range(8):
        bl = d / "benchmark_baseline.txt"
        if bl.is_file():
            text = bl.read_text()
            lat = extract_latency_ms(text)
            if lat is not None and lat > 0:
                return lat
        parent = d.parent
        if parent == d:
            break
        d = parent
    return None


def compute_best_patch(patch_dir: Path) -> dict[str, Any] | None:
    """Deterministically select the best non-empty patch from a task directory.

    Uses ``benchmark_baseline.txt`` as the canonical (unmodified) baseline rather
    than ``patch_0_test.txt`` which is the agent's first attempt.  Only
    returns a result if a patch genuinely beats the true baseline (>1.0x).
    """
    original_bl = _find_original_baseline_ms(patch_dir)

    baseline_file = patch_dir / "patch_0_test.txt"
    baseline_text = ""
    baseline_shape_latencies: dict[str, float] = {}
    if original_bl is not None:
        baseline_ms = original_bl
        baseline_source = "benchmark_baseline.txt"
        logger.debug("compute_best_patch(%s): using benchmark_baseline.txt (%.4f ms).", patch_dir.name, original_bl)
        baseline_file_path = next(
            (p for p in [patch_dir, *patch_dir.parents] if (p / "benchmark_baseline.txt").is_file()), None
        )
        if baseline_file_path is not None:
            baseline_text = (baseline_file_path / "benchmark_baseline.txt").read_text()
            baseline_shape_latencies = parse_shape_latencies_ms(baseline_text)
    elif baseline_file.exists():
        baseline_text = baseline_file.read_text()
        baseline_ms = extract_latency_ms(baseline_text)
        baseline_source = "patch_0_test.txt (FALLBACK)"
        logger.debug("compute_best_patch(%s): fallback to patch_0_test.txt baseline.", patch_dir.name)
        baseline_shape_latencies = parse_shape_latencies_ms(baseline_text)
    else:
        logger.debug("compute_best_patch(%s): no baseline found; returning None.", patch_dir.name)
        return None

    if baseline_ms is None or baseline_ms <= 0:
        logger.debug("compute_best_patch(%s): invalid baseline_ms=%s; returning None.", patch_dir.name, baseline_ms)
        return None

    best_speedup = 0.0
    best_candidate_ms: float | None = None
    best_patch_id: str | None = None
    best_patch_file: str | None = None
    best_test_file: str | None = None
    best_patch_size: int = 0
    best_shape_speedups: dict[str, dict[str, float]] = {}
    best_candidate_shape_latencies: dict[str, float] = {}

    for test_file in sorted(patch_dir.glob("patch_*_test.txt")):
        name = test_file.stem.replace("_test", "")

        patch_file = patch_dir / f"{name}.patch"
        if not patch_file.exists():
            continue
        psz = patch_file.stat().st_size
        if psz == 0:
            continue

        candidate_text = test_file.read_text()
        candidate_ms = extract_latency_ms(candidate_text)
        if candidate_ms is None or candidate_ms <= 0:
            continue
        candidate_shape_latencies = parse_shape_latencies_ms(candidate_text)

        speedup = baseline_ms / candidate_ms
        if speedup > best_speedup:
            best_speedup = speedup
            best_candidate_ms = candidate_ms
            best_patch_id = name
            best_patch_file = str(patch_file)
            best_test_file = str(test_file)
            best_patch_size = psz
            best_candidate_shape_latencies = candidate_shape_latencies
            best_shape_speedups = compute_shape_speedups(baseline_shape_latencies, candidate_shape_latencies)

    if best_patch_id is None or best_speedup <= 1.0:
        logger.debug(
            "compute_best_patch(%s): no patch beat baseline (best_speedup=%.4f).",
            patch_dir.name,
            best_speedup,
        )
        return None

    return {
        "best_patch_id": best_patch_id,
        "best_patch_speedup": round(best_speedup, 6),
        "best_patch_file": best_patch_file,
        "best_patch_test_output": best_test_file,
        "best_patch_size_bytes": best_patch_size,
        "baseline_latency_ms": round(baseline_ms, 6),
        "candidate_latency_ms": round(best_candidate_ms, 6),
        "baseline_source": baseline_source,
        "baseline_shape_latency_ms": baseline_shape_latencies,
        "candidate_shape_latency_ms": best_candidate_shape_latencies,
        "per_shape_speedups": best_shape_speedups,
        "llm_selection_analysis": (
            f"Deterministic: baseline={baseline_ms:.4f}ms ({baseline_source}), "
            f"candidate={best_candidate_ms:.4f}ms from {best_patch_id}. "
            f"Speedup={best_speedup:.4f}x. Patch={best_patch_size}B."
        ),
    }


def rewrite_best_results(patch_dir: Path) -> dict[str, Any] | None:
    """Overwrite ``best_results.json`` with deterministic selection if possible.

    Uses the canonical baseline from benchmark_baseline.txt.  If no patch
    genuinely improves on the true baseline, clamps any LLM-reported
    speedup to 1.0x to prevent false positives.
    """
    det = compute_best_patch(patch_dir)
    existing_path = patch_dir / "best_results.json"
    original_bl = _find_original_baseline_ms(patch_dir)

    if det is not None:
        existing_path.write_text(json.dumps(det, indent=2))
        logger.info(
            "Deterministic best_results for %s: %s (%.4fx)",
            patch_dir.name,
            det["best_patch_id"],
            det["best_patch_speedup"],
        )
        return det

    if existing_path.exists():
        try:
            existing = json.loads(existing_path.read_text())
            pf = existing.get("best_patch_file")

            if pf and Path(pf).exists() and Path(pf).stat().st_size == 0:
                logger.warning("rewrite_best_results(%s): empty patch; clamping speedup to 1.0.", patch_dir.name)
                existing["best_patch_speedup"] = 1.0
                existing["llm_selection_analysis"] = (
                    existing.get("llm_selection_analysis") or ""
                ) + " [Overridden: patch is empty (0 bytes), speedup clamped to 1.0]"
                existing_path.write_text(json.dumps(existing, indent=2))
                return existing

            if original_bl is not None:
                logger.info(
                    "rewrite_best_results(%s): no patch beat true baseline %.4f ms; clamping to 1.0.",
                    patch_dir.name,
                    original_bl,
                )
                existing["best_patch_speedup"] = 1.0
                existing["baseline_latency_ms"] = original_bl
                existing["baseline_source"] = "benchmark_baseline.txt"
                existing["llm_selection_analysis"] = (
                    existing.get("llm_selection_analysis") or ""
                ) + f" [Clamped: no patch beat true baseline {original_bl:.4f}ms]"
                existing_path.write_text(json.dumps(existing, indent=2))
                return existing

            return existing
        except (json.JSONDecodeError, ValueError) as exc:
            logger.debug("rewrite_best_results(%s): failed to read existing best_results.json: %s", patch_dir.name, exc)

    return None
