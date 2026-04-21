"""Helpers for deterministic summaries over selected profiled kernels."""

from __future__ import annotations

from typing import Any

SINGLE_BOTTLENECK_THRESHOLD_PCT = 80.0
SECONDARY_GUIDANCE_THRESHOLD_PCT = 25.0


def _kernel_duration_us(kernel: dict[str, Any]) -> float:
    try:
        return float(kernel.get("duration_us", kernel.get("metrics", {}).get("duration_us", 0)) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def normalize_bottleneck_label(value: Any) -> str:
    """Normalize bottleneck labels to a stable canonical vocabulary."""
    text = str(value or "").strip().lower()
    if not text:
        return "unknown"
    if "latency" in text:
        return "latency"
    if "memory" in text:
        return "memory"
    if "compute" in text:
        return "compute"
    if "lds" in text:
        return "lds"
    if "balanced" in text:
        return "balanced"
    if text == "unknown":
        return "unknown"
    return "unknown"


def build_selected_kernel_summary(selected: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a deterministic bottleneck summary for selected kernels.

    The summary is intentionally independent of the incoming kernel ordering so
    downstream consumers can rely on a stable shape even when ``top_kernels``
    preserves an LLM-provided relevance order.
    """

    if not selected:
        return {
            "mode": "single",
            "primary_bottleneck": "unknown",
            "primary_bottleneck_pct_of_selected": 0.0,
            "single_threshold_pct": SINGLE_BOTTLENECK_THRESHOLD_PCT,
            "selected_duration_us": 0.0,
            "bottleneck_mix": [],
        }

    total_duration_us = sum(_kernel_duration_us(k) for k in selected)
    use_duration_weights = total_duration_us > 0
    total_weight = total_duration_us if use_duration_weights else float(len(selected))

    buckets: dict[str, dict[str, Any]] = {}
    for kernel in selected:
        name = str(kernel.get("name", "?"))
        duration_us = round(_kernel_duration_us(kernel), 3)
        bottleneck = normalize_bottleneck_label(kernel.get("bottleneck"))
        bucket = buckets.setdefault(
            bottleneck,
            {
                "bottleneck": bottleneck,
                "duration_us": 0.0,
                "weight": 0.0,
                "kernels": [],
            },
        )
        bucket["duration_us"] += duration_us
        kernel_weight = duration_us if use_duration_weights else 1.0
        bucket["weight"] += kernel_weight
        bucket["kernels"].append(
            {
                "name": name,
                "duration_us": duration_us,
                "pct_of_selected": round(100.0 * kernel_weight / total_weight, 1),
            }
        )

    bottleneck_mix = []
    for bucket in buckets.values():
        kernels = sorted(bucket["kernels"], key=lambda item: (-item["duration_us"], item["name"]))
        pct_of_selected = round(100.0 * bucket["weight"] / total_weight, 1)
        bottleneck_mix.append(
            {
                "bottleneck": bucket["bottleneck"],
                "duration_us": round(bucket["duration_us"], 3),
                "pct_of_selected": pct_of_selected,
                "kernels": kernels,
            }
        )

    bottleneck_mix.sort(key=lambda item: (-item["pct_of_selected"], item["bottleneck"]))
    primary = bottleneck_mix[0]["bottleneck"]
    primary_pct = bottleneck_mix[0]["pct_of_selected"]
    mode = "single" if len(bottleneck_mix) == 1 or primary_pct >= SINGLE_BOTTLENECK_THRESHOLD_PCT else "mixed"

    return {
        "mode": mode,
        "primary_bottleneck": primary,
        "primary_bottleneck_pct_of_selected": primary_pct,
        "single_threshold_pct": SINGLE_BOTTLENECK_THRESHOLD_PCT,
        "selected_duration_us": round(total_duration_us, 3),
        "bottleneck_mix": bottleneck_mix,
    }


def get_selected_kernel_summary(profiling_metrics: dict[str, Any] | None) -> dict[str, Any]:
    """Return an existing selected-kernel summary or build one from top_kernels."""
    if not isinstance(profiling_metrics, dict):
        return build_selected_kernel_summary([])

    summary = profiling_metrics.get("selected_kernel_summary")
    if isinstance(summary, dict) and summary.get("bottleneck_mix") is not None:
        return summary

    top_kernels = profiling_metrics.get("top_kernels")
    if isinstance(top_kernels, list):
        return build_selected_kernel_summary(top_kernels)
    return build_selected_kernel_summary([])


def derive_primary_bottleneck(profiling_metrics: dict[str, Any] | None) -> str:
    """Derive the canonical scalar bottleneck from profiling metrics."""
    if not isinstance(profiling_metrics, dict):
        return "unknown"

    summary = get_selected_kernel_summary(profiling_metrics)
    primary = normalize_bottleneck_label(summary.get("primary_bottleneck"))
    if primary != "unknown":
        return primary

    primary = normalize_bottleneck_label(profiling_metrics.get("primary_bottleneck"))
    if primary != "unknown":
        return primary

    top_level = normalize_bottleneck_label(profiling_metrics.get("bottleneck"))
    if top_level != "unknown":
        return top_level

    top_kernels = profiling_metrics.get("top_kernels")
    if isinstance(top_kernels, list) and top_kernels:
        return build_selected_kernel_summary(top_kernels).get("primary_bottleneck", "unknown")

    return "unknown"


def guidance_bottlenecks(
    profiling_metrics: dict[str, Any] | None,
    *,
    secondary_threshold_pct: float = SECONDARY_GUIDANCE_THRESHOLD_PCT,
    max_families: int = 2,
) -> list[str]:
    """Return the bottleneck families that should drive optimization guidance."""
    summary = get_selected_kernel_summary(profiling_metrics)
    mix = summary.get("bottleneck_mix", [])
    if not isinstance(mix, list) or not mix:
        primary = normalize_bottleneck_label(summary.get("primary_bottleneck"))
        return [] if primary == "unknown" else [primary]

    selected: list[str] = []
    for idx, entry in enumerate(mix):
        if len(selected) >= max_families:
            break
        bottleneck = normalize_bottleneck_label(entry.get("bottleneck"))
        pct = float(entry.get("pct_of_selected", 0.0) or 0.0)
        if bottleneck == "unknown":
            continue
        if idx == 0 or pct >= secondary_threshold_pct:
            selected.append(bottleneck)
    return selected


def format_bottleneck_summary(profiling_metrics: dict[str, Any] | None) -> str:
    """Format a human-readable summary of the selected bottleneck mix."""
    summary = get_selected_kernel_summary(profiling_metrics)
    primary = normalize_bottleneck_label(summary.get("primary_bottleneck"))
    primary_pct = float(summary.get("primary_bottleneck_pct_of_selected", 0.0) or 0.0)
    if primary == "unknown":
        return "Primary bottleneck: unknown"

    if summary.get("mode") == "mixed":
        mix = summary.get("bottleneck_mix", [])
        if isinstance(mix, list) and mix:
            parts = []
            for entry in mix[:2]:
                bottleneck = normalize_bottleneck_label(entry.get("bottleneck"))
                if bottleneck == "unknown":
                    continue
                pct = float(entry.get("pct_of_selected", 0.0) or 0.0)
                parts.append(f"{bottleneck} {pct:.1f}%")
            if parts:
                return f"Mixed bottlenecks: {', '.join(parts)} (primary: {primary})"
        return f"Mixed bottlenecks (primary: {primary})"

    return f"Primary bottleneck: {primary} ({primary_pct:.1f}% of selected kernel time)"
