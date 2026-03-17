"""Profile-Metric Similarity Search (KernelBlaster-inspired).

Enables cross-kernel transfer by finding past optimizations with similar
profiling metric vectors, regardless of kernel category.

Key insight: a rope kernel and a gemm kernel with the same LDS bottleneck
pattern share optimization strategies, even though they're different categories.

Uses cosine similarity on normalized metric vectors for robust matching
that's insensitive to absolute metric values but sensitive to patterns.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

# Metrics used for similarity comparison (ordered)
SIMILARITY_METRICS = [
    "memory.hbm_bandwidth_utilization",
    "memory.coalescing_efficiency",
    "memory.l1_hit_rate",
    "memory.l2_hit_rate",
    "memory.lds_bank_conflicts",
    "memory.global_load_efficiency",
    "memory.global_store_efficiency",
    "memory.l2_bandwidth",
]


def extract_metric_vector(profiling_metrics: dict) -> list[float]:
    """Extract a normalized metric vector from profiling results."""
    vector = []
    for metric in SIMILARITY_METRICS:
        val = profiling_metrics.get(metric, 0.0)
        if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
            val = 0.0
        vector.append(float(val))
    return vector


def normalize_vector(v: list[float]) -> list[float]:
    """L2-normalize a vector."""
    norm = math.sqrt(sum(x * x for x in v))
    if norm < 1e-10:
        return [0.0] * len(v)
    return [x / norm for x in v]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_norm = normalize_vector(a)
    b_norm = normalize_vector(b)
    return sum(x * y for x, y in zip(a_norm, b_norm))


def find_similar_profiles(
    query_metrics: dict,
    stored_outcomes: list[dict],
    top_k: int = 5,
    min_similarity: float = 0.5,
) -> list[tuple[dict, float]]:
    """Find past outcomes with similar profiling metric patterns.

    Args:
        query_metrics: Current kernel's profiling metrics
        stored_outcomes: List of past outcome dicts (must have 'profiling_metrics' field)
        top_k: Number of results to return
        min_similarity: Minimum cosine similarity threshold

    Returns:
        List of (outcome, similarity_score) tuples, sorted by similarity
    """
    query_vec = extract_metric_vector(query_metrics)
    if all(v == 0 for v in query_vec):
        return []

    results = []
    for outcome in stored_outcomes:
        stored_metrics = outcome.get("profiling_metrics")
        if not stored_metrics:
            continue
        if isinstance(stored_metrics, str):
            try:
                stored_metrics = json.loads(stored_metrics)
            except (json.JSONDecodeError, TypeError):
                continue

        stored_vec = extract_metric_vector(stored_metrics)
        if all(v == 0 for v in stored_vec):
            continue

        sim = cosine_similarity(query_vec, stored_vec)
        if sim >= min_similarity:
            results.append((outcome, sim))

    results.sort(key=lambda x: -x[1])
    return results[:top_k]


def format_similar_profiles(
    matches: list[tuple[dict, float]],
    exclude_category: str | None = None,
) -> str:
    """Format similar profile matches for injection into agent prompt."""
    if not matches:
        return ""

    filtered = matches
    if exclude_category:
        filtered = [(o, s) for o, s in matches if o.get("kernel_category") != exclude_category]

    if not filtered:
        return ""

    lines = ["\nCross-kernel profile matches (similar hardware bottleneck patterns):"]
    for outcome, sim in filtered[:3]:
        cat = outcome.get("kernel_category", "?")
        sp = outcome.get("speedup_achieved", 0)
        strat = outcome.get("strategy_name", "?")
        bn = outcome.get("bottleneck_type", "?")
        success = "OK" if outcome.get("success") else "FAIL"
        lines.append(
            f"  - [{success}] {cat} ({bn}): {sp:.2f}x speedup via {strat} "
            f"(profile similarity: {sim:.0%})"
        )
    return "\n".join(lines)
