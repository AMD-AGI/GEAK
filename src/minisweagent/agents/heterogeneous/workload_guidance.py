"""Backend-specific workload guidance for task generation.

Pure functions that build "Prefer First / Consider Next / Deprioritize"
strategy blocks based on kernel backend type and profiling bottleneck.
Injected into the task generator's LLM prompt to guide strategy selection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from minisweagent.run.utils.selected_kernel_summary import (
    derive_primary_bottleneck,
    format_bottleneck_summary,
    guidance_bottlenecks,
)

_HIP_SEARCH_HINT_PATTERNS = (
    "binary_search",
    "lower_bound",
    "upper_bound",
    "search_n",
    "device_search",
    "haystack",
    "needle",
)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_optional_float(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "unknown"
    return f"{value:.1f}{suffix}"


def _normalized_bottleneck(baseline_metrics: dict[str, Any]) -> str:
    return derive_primary_bottleneck(baseline_metrics)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def _is_hip_like_kernel(kernel: dict[str, Any]) -> bool:
    path = str(kernel.get("file_path", "")).lower()
    ext = Path(path).suffix.lower()
    kernel_type = str(kernel.get("kernel_type", "")).lower()
    if kernel_type in {"triton", "ck", "asm"}:
        return False
    return kernel_type == "hip" or (
        ext in {".hpp", ".h", ".cpp", ".cu", ".hip"} and any(token in path for token in ("rocprim", "hip", "rocm"))
    )


def _is_triton_like_kernel(kernel: dict[str, Any]) -> bool:
    path = str(kernel.get("file_path", "")).lower()
    kernel_type = str(kernel.get("kernel_type", "")).lower()
    if kernel_type == "triton":
        return True
    return "triton" in path and path.endswith(".py")


def _detect_backend(kernel: dict[str, Any]) -> str:
    if _is_triton_like_kernel(kernel):
        return "triton"
    if _is_hip_like_kernel(kernel):
        return "hip"
    return "generic"


def _is_search_like_workload(kernel: dict[str, Any], baseline_metrics: dict[str, Any]) -> bool:
    evidence_chunks: list[str] = [
        str(kernel.get("kernel_name", "")),
        str(kernel.get("file_path", "")),
        str(baseline_metrics.get("kernel_name", "")),
    ]
    for top in baseline_metrics.get("top_kernels", []) or []:
        evidence_chunks.append(str(top.get("name", "")))
    haystack = " ".join(evidence_chunks).lower()
    return any(pat in haystack for pat in _HIP_SEARCH_HINT_PATTERNS)


def _profiling_summary_lines(baseline_metrics: dict[str, Any]) -> list[str]:
    top_kernels = baseline_metrics.get("top_kernels", [])
    if not top_kernels:
        return ["Profiling summary: no kernel data available."]

    lines = ["Profiling summary (per-kernel):"]
    for k in top_kernels:
        km = k.get("metrics", {}) or {}
        hbm_util = _safe_float(km.get("memory.hbm_bandwidth_utilization"))
        l2_hit = _safe_float(km.get("memory.l2_hit_rate"))
        hbm_read = _safe_float(km.get("memory.hbm_read_bandwidth"))
        hbm_write = _safe_float(km.get("memory.hbm_write_bandwidth"))
        l1_hit = _safe_float(km.get("memory.l1_hit_rate"))
        coalescing = _safe_float(km.get("memory.coalescing_efficiency"))

        line = (
            f"- {k.get('name', '?')}: "
            f"bottleneck={k.get('bottleneck', '?')}"
            f"; duration={_format_optional_float(_safe_float(k.get('duration_us')), ' us')}"
            f" ({k.get('pct_of_selected', '?')}%)"
            f"; HBM util={_format_optional_float(hbm_util, '%')}"
            f"; L2 hit={_format_optional_float(l2_hit, '%')}"
        )
        extras = []
        if hbm_read is not None:
            extras.append(f"HBM read BW={hbm_read:.1f}")
        if hbm_write is not None:
            extras.append(f"HBM write BW={hbm_write:.1f}")
        if l1_hit is not None:
            extras.append(f"L1 hit={l1_hit:.1f}%")
        if coalescing is not None:
            extras.append(f"coalescing={coalescing:.1f}%")
        if extras:
            line += f"; {'; '.join(extras)}"
        lines.append(line)

        for obs in k.get("observations", []):
            lines.append(f"  - {obs}")

    return lines


def _build_triton_guidance(kernel: dict[str, Any], baseline_metrics: dict[str, Any]) -> str:
    bottlenecks = guidance_bottlenecks(baseline_metrics)
    primary_bottleneck = _normalized_bottleneck(baseline_metrics)

    prefer_first = [
        "Algorithmic kernel-body rewrites that change the reduction tree, tiling scheme, decomposition, or math formulation.",
        "Operation fusion or launch-count reduction when adjacent work can be merged into the Triton kernel body.",
    ]
    consider_next = [
        "Shape-specialized kernel variants when different input regimes clearly want different algorithms or tile structures.",
        "Kernel-body memory-layout and live-range cleanup that directly supports the hottest profiled path.",
    ]
    deprioritize = [
        "@triton.autotune-only config sweeps.",
        "Pure num_warps / num_stages / BLOCK_* parameter search without a kernel-body change.",
        "Python dispatch, import-routing, or wrapper-only edits unless profiling clearly shows the wrapper dominates.",
    ]

    if not bottlenecks or primary_bottleneck in {"balanced", "unknown"}:
        prefer_first.extend(
            [
                "Profiling-driven kernel-body simplifications on the hottest sub-kernels instead of generic parameter sweeps.",
                "Common kernel optimization strategies such as fusion, shape-specialized variants, and memory/computation reordering.",
            ]
        )
    for bottleneck in bottlenecks:
        if bottleneck == "memory":
            prefer_first.extend(
                [
                    "Memory-access rewrites inside the kernel body: better blocking, fewer redundant loads/stores, and higher SRAM/L2 reuse.",
                    "Masking, pointer-arithmetic, or load/store simplifications that reduce HBM traffic on the hottest path.",
                ]
            )
            consider_next.append(
                "Vectorized or blocked load/store patterns when they are part of a broader kernel-body memory-traffic reduction plan."
            )
        elif bottleneck == "compute":
            prefer_first.extend(
                [
                    "Instruction-count reduction and control-flow simplification inside hot loops.",
                    "MFMA / tl.dot-friendly reformulations, cheaper math primitives, or algorithmic approximations when correct.",
                ]
            )
            consider_next.append(
                "Register-pressure and live-range reductions that let the compiler schedule the kernel body more efficiently."
            )
        elif bottleneck == "latency":
            prefer_first.extend(
                [
                    "Fuse adjacent short kernels so each launch performs materially more work.",
                    "Increase work per program or use persistent / multi-tile kernel patterns that amortize launch overhead.",
                ]
            )
            consider_next.append(
                "Shape-specialized kernel variants for small vs large shapes so short kernels are not forced into one-size-fits-all code."
            )
        elif bottleneck == "lds":
            prefer_first.extend(
                [
                    "LDS-bank-conflict reduction and staged-access restructuring inside the kernel body.",
                    "Move transient data from LDS to registers when it reduces LDS pressure without hurting occupancy too much.",
                ]
            )

    prefer_first = _dedupe(prefer_first)
    consider_next = _dedupe(consider_next)
    deprioritize = _dedupe(deprioritize)

    lines = [
        "Triton backend detected. Prefer profiling-driven kernel-body strategies over autotune or wrapper work.",
        format_bottleneck_summary(baseline_metrics),
        *_profiling_summary_lines(baseline_metrics),
        "Planning policy:",
        "- Fill most task slots with 'Prefer First' families below.",
        "- Only add autotune / launch / wrapper tasks after at least 3 preferred-family tasks exist.",
        "- Leave GPUs idle if the remaining ideas are only low-priority wrapper work.",
        "Prefer First:",
        *[f"- {item}" for item in prefer_first],
        "Consider Next:",
        *[f"- {item}" for item in consider_next],
        "Deprioritize Until Later:",
        *[f"- {item}" for item in deprioritize],
    ]
    return "\n".join(lines)


def _build_hip_guidance(kernel: dict[str, Any], baseline_metrics: dict[str, Any]) -> str:
    top_kernels = baseline_metrics.get("top_kernels", [])
    bottlenecks = guidance_bottlenecks(baseline_metrics)
    bottleneck = _normalized_bottleneck(baseline_metrics)
    hbm_utils = [_safe_float((k.get("metrics", {}) or {}).get("memory.hbm_bandwidth_utilization")) for k in top_kernels]
    hbm_utils_valid = [h for h in hbm_utils if h is not None]
    max_hbm_util = max(hbm_utils_valid) if hbm_utils_valid else None
    bandwidth_deprioritized = bottleneck == "latency" and (max_hbm_util is None or max_hbm_util < 10.0)
    is_search_like = _is_search_like_workload(kernel, baseline_metrics)

    prefer_first = [
        "Algorithmic HIP kernel-body rewrites that change the search / reduction / tiling structure.",
        "Common kernel optimizations driven by the hottest profiled path, not by generic occupancy or launch heuristics.",
    ]
    consider_next = [
        "Kernel-body memory-layout, register-pressure, or LDS-usage cleanup that directly helps the profiled bottleneck.",
        "Size-specialized kernel variants when one generic implementation is serving multiple very different workload regimes.",
    ]
    deprioritize = [
        "Launch-config or occupancy-only tuning.",
        "Wrapper / dispatch / copy-path edits unless profiling shows they dominate total time.",
    ]

    if not bottlenecks or bottleneck in {"balanced", "unknown"}:
        prefer_first.extend(
            [
                "Fusion, algorithmic simplification, and memory/computation reordering based on the hottest profiled sub-kernels.",
                "Operation-specific or size-specific kernel variants when the profile suggests one implementation is serving mismatched regimes.",
            ]
        )
    for family in bottlenecks:
        if family == "memory":
            prefer_first.extend(
                [
                    "Coalescing, vectorized access, or LDS staging when they directly raise effective bandwidth on the hot path.",
                    "Global-memory traffic reduction by fusing steps or recomputing cheap values instead of reloading them.",
                ]
            )
            consider_next.append(
                "Wavefront-level memory-access reordering or bank-conflict reduction when it is supported by the profile."
            )
        elif family == "compute":
            prefer_first.extend(
                [
                    "Instruction-count reduction, branch simplification, and cheaper per-thread math in the hottest loops.",
                    "Wave intrinsics, MFMA-friendly decomposition, or unrolled inner loops when they reduce compute bottlenecks.",
                ]
            )
        elif family == "latency":
            prefer_first.extend(
                [
                    "Branchless/control-flow simplification that reduces serialized decision cost in short kernels.",
                    "Operation-specific specialization so the hot path does not pay for generic functionality it does not need.",
                    "Wavefront-cooperative or persistent-work patterns that amortize per-launch or per-query overhead.",
                ]
            )
            if is_search_like:
                prefer_first.extend(
                    [
                        "Size-specialized kernel variants for separate small / medium / huge haystack paths.",
                        "Wavefront-cooperative upper-level search or coarse-index narrowing when preprocessing can be amortized.",
                    ]
                )
            if bandwidth_deprioritized:
                deprioritize.insert(0, "Bandwidth-maximization or generic vectorization ideas as the main strategy.")
                deprioritize.insert(1, "Items-per-thread or throughput-only tuning without a latency-reduction hypothesis.")
        elif family == "lds":
            prefer_first.extend(
                [
                    "LDS-bank-conflict reduction and staged-access redesign inside the kernel body.",
                    "Register-vs-LDS tradeoff changes that lower LDS pressure on the hot path.",
                ]
            )

    prefer_first = _dedupe(prefer_first)
    consider_next = _dedupe(consider_next)
    deprioritize = _dedupe(deprioritize)

    lines = [
        "HIP backend detected. Prefer profiling-driven kernel-body strategies over launch tuning or wrapper work.",
        format_bottleneck_summary(baseline_metrics),
        *_profiling_summary_lines(baseline_metrics),
        "Planning policy:",
        "- Fill most task slots with 'Prefer First' families below.",
        "- Only add launch / dispatch / wrapper tasks after at least 3 preferred-family tasks exist.",
        "- Leave GPUs idle if the remaining ideas are only low-priority wrapper work.",
        "Prefer First:",
        *[f"- {item}" for item in prefer_first],
        "Consider Next:",
        *[f"- {item}" for item in consider_next],
        "Deprioritize Until Later:",
        *[f"- {item}" for item in deprioritize],
    ]

    if is_search_like and bottleneck == "latency":
        l2_hits = [_safe_float((k.get("metrics", {}) or {}).get("memory.l2_hit_rate")) for k in top_kernels]
        l2_hit_valid = [h for h in l2_hits if h is not None]
        min_l2 = min(l2_hit_valid) if l2_hit_valid else None
        lines.extend(
            [
                "Search / pointer-chasing classifier:",
                (
                    f"- Evidence: bottleneck={bottleneck}; max HBM utilization={_format_optional_float(max_hbm_util, '%')}; "
                    f"min L2 hit rate={_format_optional_float(min_l2, '%')}"
                ),
                "- Treat this as latency-bound search work, so branchlessness, specialization, and cooperative search matter more than throughput tuning.",
            ]
        )

    return "\n".join(lines)


def _build_workload_guidance(kernel: dict[str, Any], baseline_metrics: dict[str, Any]) -> str:
    """Return backend/workload-specific guidance for task planning."""
    backend = _detect_backend(kernel)
    if backend == "triton":
        return _build_triton_guidance(kernel, baseline_metrics)
    if backend == "hip":
        return _build_hip_guidance(kernel, baseline_metrics)
    if not baseline_metrics:
        return ""
    lines = [
        "Backend-specific classifier unavailable, but profiling guidance is still mandatory.",
        *_profiling_summary_lines(baseline_metrics),
        "Prefer First:",
        "- Algorithmic kernel-body rewrites, fusion, and common kernel optimizations suggested by the hottest profiled path.",
        "Deprioritize Until Later:",
        "- Autotune-only, launch-only, and dispatch-only work unless profiling strongly implicates them.",
    ]
    return "\n".join(lines)
