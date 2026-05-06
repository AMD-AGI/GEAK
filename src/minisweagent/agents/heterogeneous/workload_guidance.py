"""Backend-specific workload guidance for task generation.

Pure functions that build "Prefer First / Consider Next / Deprioritize"
strategy blocks based on kernel backend type and profiling bottleneck.
Injected into the task generator's LLM prompt to guide strategy selection.

Language detection goes through the ``kernel_languages.registry`` —
literal equality comparisons against language names are forbidden in
this module (enforced by ``scripts/refactor_ci/check_language_leaks.py``).
When the registry cannot identify a language, we fall back to the path
heuristics (``.hip``/``.cu`` + ROCm tokens → HIP-like; Python file
whose path mentions Triton → Triton-like) so the guidance still fires
in partial-discovery scenarios.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from minisweagent.run.utils.gpu_arch import detect_gpu_arch, is_wmma_capable

from minisweagent.kernel_languages import registry

_HIP_SEARCH_HINT_PATTERNS = (
    "binary_search",
    "lower_bound",
    "upper_bound",
    "search_n",
    "device_search",
    "haystack",
    "needle",
)

# Names carried around by heterogeneous/task_generator and the discovery
# dict that should NOT be treated as HIP-like even though they may land
# in C++-extension territory.  Kept as a data tuple instead of inline
# checks so the set is obvious and extensible.
_NON_HIP_KERNEL_NAMES: tuple[str, ...] = ("ck", "asm")


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
    text = str(baseline_metrics.get("bottleneck", "unknown")).lower().strip()
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
    return "unknown"


def _resolved_language_name(kernel: dict[str, Any]) -> str:
    """Return the canonical KernelLanguage name for ``kernel`` or ``""``.

    Order of resolution (all via ``kernel_languages.registry`` — no
    literal equality checks):
      1. ``kernel["kernel_type"]`` passed through
         ``registry.detect_best_by_name`` — handles aliases (``rocm``
         → ``hip`` etc).
      2. Fall back to ``registry.detect_best`` on the kernel file path
         if the file exists.
      3. ``""`` when both routes return None.
    """
    raw_type = str(kernel.get("kernel_type", "")).strip().lower()
    if raw_type in _NON_HIP_KERNEL_NAMES:
        # Legacy guard: these are NOT canonical KernelLanguage names and
        # we shouldn't route them through either branch of the guidance.
        return raw_type

    if raw_type:
        lang = registry.detect_best_by_name(raw_type)
        if lang is not None:
            return lang.name

    file_path = str(kernel.get("file_path", "")).strip()
    if file_path:
        lang = registry.detect_best(Path(file_path))
        if lang is not None:
            return lang.name

    return ""


def _is_hip_like_kernel(kernel: dict[str, Any]) -> bool:
    name = _resolved_language_name(kernel)
    if name == "hip":
        return True
    if name in _NON_HIP_KERNEL_NAMES or name == "triton":
        return False

    # Last-resort path heuristic for partial-discovery cases: HIP-ish
    # file extensions containing a ROCm / HIP token.
    path = str(kernel.get("file_path", "")).lower()
    ext = Path(path).suffix.lower()
    return ext in {".hpp", ".h", ".cpp", ".cu", ".hip"} and any(
        token in path for token in ("rocprim", "hip", "rocm")
    )


def _is_triton_like_kernel(kernel: dict[str, Any]) -> bool:
    name = _resolved_language_name(kernel)
    if name == "triton":
        return True
    if name and name != "":
        return False

    # Partial-discovery fallback: Python file whose path mentions
    # "triton".  Only hit when the registry couldn't decide.
    path = str(kernel.get("file_path", "")).lower()
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
    metrics = baseline_metrics.get("metrics", {}) or {}
    duration_us = _safe_float(baseline_metrics.get("duration_us"))
    hbm_util = _safe_float(metrics.get("memory.hbm_bandwidth_utilization"))
    l2_hit = _safe_float(metrics.get("memory.l2_hit_rate"))
    bottleneck = _normalized_bottleneck(baseline_metrics)
    return [
        "Profiling summary:",
        (
            f"- Bottleneck: {bottleneck}"
            f"; kernel duration: {_format_optional_float(duration_us, ' us')}"
            f"; HBM utilization: {_format_optional_float(hbm_util, '%')}"
            f"; L2 hit rate: {_format_optional_float(l2_hit, '%')}"
        ),
    ]


def _build_triton_guidance(kernel: dict[str, Any], baseline_metrics: dict[str, Any]) -> str:
    bottleneck = _normalized_bottleneck(baseline_metrics)

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
        _matrix_label = "WMMA" if is_wmma_capable(detect_gpu_arch()) else "MFMA"
        prefer_first.extend(
            [
                "Instruction-count reduction and control-flow simplification inside hot loops.",
                f"{_matrix_label} / tl.dot-friendly reformulations, cheaper math primitives, or algorithmic approximations when correct.",
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
    else:
        prefer_first.extend(
            [
                "Profiling-driven kernel-body simplifications on the hottest sub-kernels instead of generic parameter sweeps.",
                "Common kernel optimization strategies such as fusion, shape-specialized variants, and memory/computation reordering.",
            ]
        )

    lines = [
        "Triton backend detected. Prefer profiling-driven kernel-body strategies over autotune or wrapper work.",
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
    metrics = baseline_metrics.get("metrics", {}) or {}
    bottleneck = _normalized_bottleneck(baseline_metrics)
    hbm_util = _safe_float(metrics.get("memory.hbm_bandwidth_utilization"))
    bandwidth_deprioritized = bottleneck == "latency" and (hbm_util is None or hbm_util < 10.0)
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

    if bottleneck == "memory":
        prefer_first.extend(
            [
                "Coalescing, vectorized access, or LDS staging when they directly raise effective bandwidth on the hot path.",
                "Global-memory traffic reduction by fusing steps or recomputing cheap values instead of reloading them.",
            ]
        )
        consider_next.append(
            "Wavefront-level memory-access reordering or bank-conflict reduction when it is supported by the profile."
        )
    elif bottleneck == "compute":
        _matrix_label = "WMMA" if is_wmma_capable(detect_gpu_arch()) else "MFMA"
        prefer_first.extend(
            [
                "Instruction-count reduction, branch simplification, and cheaper per-thread math in the hottest loops.",
                f"Wave intrinsics, {_matrix_label}-friendly decomposition, or unrolled inner loops when they reduce compute bottlenecks.",
            ]
        )
    elif bottleneck == "latency":
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
    elif bottleneck == "lds":
        prefer_first.extend(
            [
                "LDS-bank-conflict reduction and staged-access redesign inside the kernel body.",
                "Register-vs-LDS tradeoff changes that lower LDS pressure on the hot path.",
            ]
        )
    else:
        prefer_first.extend(
            [
                "Fusion, algorithmic simplification, and memory/computation reordering based on the hottest profiled sub-kernels.",
                "Operation-specific or size-specific kernel variants when the profile suggests one implementation is serving mismatched regimes.",
            ]
        )

    lines = [
        "HIP backend detected. Prefer profiling-driven kernel-body strategies over launch tuning or wrapper work.",
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
        l2_hit = _safe_float(metrics.get("memory.l2_hit_rate"))
        lines.extend(
            [
                "Search / pointer-chasing classifier:",
                (
                    f"- Evidence: bottleneck={bottleneck}; HBM utilization={_format_optional_float(hbm_util, '%')}; "
                    f"L2 hit rate={_format_optional_float(l2_hit, '%')}"
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
