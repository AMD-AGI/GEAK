# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.

"""Deterministic benchmark output parsing and patch selection.

Provides regex-based extraction of latency metrics from harness output
and a ``compute_best_patch()`` function that selects the best non-empty
patch by comparing benchmark numbers -- no LLM involved.

Measurement methodology:
- Uses ``benchmark_baseline.txt`` (true unmodified kernel) as the baseline
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

logger = logging.getLogger(__name__)


def parse_median_latency_ms(output: str) -> float | None:
    """Extract median latency (ms) from harness benchmark output."""
    m = re.search(
        r"(?:[Mm]edian\s+(?:latency|time)[\w\s]*|total\s+median\s+time)\s*:\s*([\d.]+(?:e[+-]?\d+)?)\s*ms",
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


def _parse_benchmark_metric(output: str) -> float | None:
    """Extract from BENCHMARK_METRIC:, median_latency_ms:, or Geomean (ms): lines."""
    for pat in (
        r"BENCHMARK_METRIC:\s*median_latency_ms=([\d.]+(?:e[+-]?\d+)?)",
        r"median_latency_ms:\s*([\d.]+(?:e[+-]?\d+)?)",
        r"Geomean\s*\(ms\)\s*:\s*([\d.]+(?:e[+-]?\d+)?)",
    ):
        m = re.search(pat, output, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


def parse_google_benchmark_ms(output: str) -> float | None:
    """Parse Google Benchmark format: <name> <iters> <latency> ms."""
    m = re.search(r"^\S+\s+\d+\s+([\d.]+(?:e[+-]?\d+)?)\s+ms", output, re.MULTILINE)
    return float(m.group(1)) if m else None


def parse_shape_count(output: str) -> int | None:
    """Extract shape count from harness benchmark output."""
    m = re.search(r"(\d+)\s+shapes", output, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _universal_latency_fallback(text: str) -> float | None:
    """Last-resort: find a number near latency-related keywords in the last
    30 lines of output. Handles formats like 'Overall Median: 0.052ms'."""
    keywords = {"median", "overall", "geomean", "latency", "total"}
    candidates: list[float] = []
    lines = text.strip().splitlines()
    for line in lines[-30:]:
        lower = line.lower()
        if not any(kw in lower for kw in keywords):
            continue
        for m in re.finditer(r"([\d.]+(?:e[+-]?\d+)?)\s*ms", line):
            val = float(m.group(1))
            if 0.0001 < val < 100000:
                candidates.append(val)
    return candidates[-1] if candidates else None


def _extract_latency(text: str) -> float | None:
    """Extract latency from benchmark output.

    Priority:
    1. GEAK_RESULT_LATENCY_MS=<number> (standardized marker, always correct)
    2. Legacy format parsers (TOTAL_KERNEL_TIME_MS, BENCHMARK_METRIC, etc.)
    3. Universal fallback: last number near latency keywords in output
    """
    m = re.search(r"GEAK_RESULT_LATENCY_MS=([\d.]+(?:e[+-]?\d+)?)", text)
    if m:
        return float(m.group(1))

    val = parse_total_kernel_time_ms(text)
    if val is not None:
        return val
    val = _parse_benchmark_metric(text)
    if val is not None:
        return val
    val = parse_median_latency_ms(text)
    if val is not None:
        return val
    val = parse_google_benchmark_ms(text)
    if val is not None:
        return val

    return _universal_latency_fallback(text)


def _find_original_baseline_ms(patch_dir: Path) -> float | None:
    """Walk up from patch_dir to find benchmark_baseline.txt (the true baseline).

    The preprocessing phase writes benchmark_baseline.txt at the kernel
    output root (e.g. patches/exp0/rope/benchmark_baseline.txt).  Task dirs
    are nested under results/round_N/strategy_name, so we walk upward.
    """
    d = patch_dir
    for _ in range(8):
        bl = d / "benchmark_baseline.txt"
        if bl.is_file():
            text = bl.read_text()
            lat = _extract_latency(text)
            if lat is not None and lat > 0:
                return lat
        parent = d.parent
        if parent == d:
            break
        d = parent
    return None


def compute_best_patch(patch_dir: Path) -> dict[str, Any] | None:
    """Deterministically select the best non-empty patch from a task directory.

    Uses ``benchmark_baseline.txt`` as the true (unmodified) baseline rather
    than ``patch_0_test.txt`` which is the agent's first attempt.  Only
    returns a result if a patch genuinely beats the true baseline (>1.0x).
    """
    original_bl = _find_original_baseline_ms(patch_dir)

    baseline_file = patch_dir / "patch_0_test.txt"
    if original_bl is not None:
        baseline_ms = original_bl
        baseline_source = "benchmark_baseline.txt"
    elif baseline_file.exists():
        baseline_text = baseline_file.read_text()
        baseline_ms = _extract_latency(baseline_text)
        baseline_source = "patch_0_test.txt (FALLBACK)"
    else:
        return None

    if baseline_ms is None or baseline_ms <= 0:
        return None

    best_speedup = 0.0
    best_candidate_ms: float | None = None
    best_patch_id: str | None = None
    best_patch_file: str | None = None
    best_test_file: str | None = None
    best_patch_size: int = 0

    for test_file in sorted(patch_dir.glob("patch_*_test.txt")):
        name = test_file.stem.replace("_test", "")

        patch_file = patch_dir / f"{name}.patch"
        if not patch_file.exists():
            continue
        psz = patch_file.stat().st_size
        if psz == 0:
            continue

        candidate_text = test_file.read_text()
        candidate_ms = _extract_latency(candidate_text)
        if candidate_ms is None or candidate_ms <= 0:
            continue

        speedup = baseline_ms / candidate_ms
        if speedup > best_speedup:
            best_speedup = speedup
            best_candidate_ms = candidate_ms
            best_patch_id = name
            best_patch_file = str(patch_file)
            best_test_file = str(test_file)
            best_patch_size = psz

    if best_patch_id is None or best_speedup <= 1.0:
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
        "llm_selection_analysis": (
            f"Deterministic: baseline={baseline_ms:.4f}ms ({baseline_source}), "
            f"candidate={best_candidate_ms:.4f}ms from {best_patch_id}. "
            f"Speedup={best_speedup:.4f}x. Patch={best_patch_size}B."
        ),
    }


def rewrite_best_results(patch_dir: Path) -> dict[str, Any] | None:
    """Overwrite ``best_results.json`` with deterministic selection if possible.

    Uses the true baseline from benchmark_baseline.txt.  If no patch
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
                existing["best_patch_speedup"] = 1.0
                existing["llm_selection_analysis"] = (
                    (existing.get("llm_selection_analysis") or "")
                    + " [Overridden: patch is empty (0 bytes), speedup clamped to 1.0]"
                )
                existing_path.write_text(json.dumps(existing, indent=2))
                return existing

            if original_bl is not None:
                existing["best_patch_speedup"] = 1.0
                existing["baseline_latency_ms"] = original_bl
                existing["baseline_source"] = "benchmark_baseline.txt"
                existing["llm_selection_analysis"] = (
                    (existing.get("llm_selection_analysis") or "")
                    + f" [Clamped: no patch beat true baseline {original_bl:.4f}ms]"
                )
                existing_path.write_text(json.dumps(existing, indent=2))
                return existing

            return existing
        except (json.JSONDecodeError, ValueError):
            pass

    return None
