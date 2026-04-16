"""Profiling fingerprint: numeric vector from GPU performance counters.

Kernels with similar fingerprints use hardware in similar ways and
respond to similar optimization strategies -- even across kernel categories.

Reference: Zigon & Song (ICCS 2020, DOI:10.1007/978-3-030-50371-0_7)
"""

from __future__ import annotations

import math
from typing import Any

CANONICAL_METRICS = [
    "max_mem_bw_pct",
    "compute_util_pct",
    "l1_coalescing_pct",
    "occupancy",
    "l2_hit",
    "valu_ratio",
    "vmem_ratio",
    "dependency_wait_pct",
    "active_cycle_pct",
    "ipc",
]

METRIC_WEIGHTS = {
    "max_mem_bw_pct": 3.0,
    "compute_util_pct": 3.0,
    "occupancy": 2.0,
    "l1_coalescing_pct": 2.0,
    "l2_hit": 1.5,
    "valu_ratio": 1.0,
    "vmem_ratio": 1.0,
    "dependency_wait_pct": 1.0,
    "active_cycle_pct": 1.0,
    "ipc": 0.5,
}

_DEFAULT_VALUE = 0.5
_NORM_RANGES: dict[str, tuple[float, float]] = {
    "max_mem_bw_pct": (0.0, 100.0),
    "compute_util_pct": (0.0, 100.0),
    "l1_coalescing_pct": (0.0, 100.0),
    "occupancy": (0.0, 100.0),
    "l2_hit": (0.0, 100.0),
    "valu_ratio": (0.0, 1.0),
    "vmem_ratio": (0.0, 1.0),
    "dependency_wait_pct": (0.0, 100.0),
    "active_cycle_pct": (0.0, 100.0),
    "ipc": (0.0, 10.0),
}


def build_fingerprint(metrics: dict[str, Any]) -> list[float]:
    """Build a normalized fingerprint vector from profiling metrics."""
    vec: list[float] = []
    for key in CANONICAL_METRICS:
        raw = metrics.get(key)
        if raw is None or not isinstance(raw, (int, float)):
            vec.append(_DEFAULT_VALUE)
            continue
        lo, hi = _NORM_RANGES.get(key, (0.0, 100.0))
        if hi <= lo:
            vec.append(_DEFAULT_VALUE)
        else:
            normalized = max(0.0, min(1.0, (float(raw) - lo) / (hi - lo)))
            vec.append(normalized)
    return vec


def fingerprint_similarity(a: list[float], b: list[float]) -> float:
    """Weighted cosine similarity between two fingerprint vectors."""
    if len(a) != len(b) or len(a) != len(CANONICAL_METRICS):
        return 0.0

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for i, key in enumerate(CANONICAL_METRICS):
        w = METRIC_WEIGHTS.get(key, 1.0)
        wa = a[i] * w
        wb = b[i] * w
        dot += wa * wb
        norm_a += wa * wa
        norm_b += wb * wb

    denom = math.sqrt(norm_a) * math.sqrt(norm_b)
    if denom < 1e-12:
        return 0.0
    return dot / denom


def bottleneck_bonus(a: str, b: str) -> float:
    """Bonus score when bottleneck types match."""
    if not a or not b:
        return 0.0
    return 0.3 if a.lower() == b.lower() else 0.0


def category_bonus(a: str, b: str) -> float:
    """Bonus when kernel categories match."""
    if not a or not b or a == "unknown" or b == "unknown":
        return 0.0
    return 0.15 if a.lower() == b.lower() else 0.0
