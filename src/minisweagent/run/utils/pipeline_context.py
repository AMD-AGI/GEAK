"""Shared profiling guidance and pipeline-context helpers."""

from __future__ import annotations

import importlib
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from minisweagent.run.utils.metrix_profile import build_metrix_profile_kwargs
from minisweagent.run.utils.selected_kernel_summary import (
    derive_primary_bottleneck,
    format_bottleneck_summary,
    guidance_bottlenecks,
)

logger = logging.getLogger(__name__)

_BOTTLENECK_GUIDANCE: dict[str, str] = {
    "balanced": (
        "## Optimization Guidance (bottleneck: balanced)\n"
        '"Balanced" means no single resource is saturated. Actionable kernel-body approaches:\n'
        "1. INCREASE ARITHMETIC INTENSITY: Fuse adjacent operations into the kernel loop "
        "so more compute happens per memory access.\n"
        "2. REDUCE MEMORY TRAFFIC: Cache intermediate results in registers or LDS "
        "instead of reading/writing global memory.\n"
        "3. IMPROVE PARALLELISM: Restructure loops to expose more independent work per "
        "wavefront; consider split-K or multi-pass approaches.\n"
        "4. ALTERNATIVE ALGORITHMS: Try a fundamentally different algorithm for the same "
        "computation (different reduction tree, different scan, tiled vs non-tiled, etc.).\n"
        "5. COMPILER GUIDANCE: Restructure Triton/HIP code to help the compiler generate "
        "better ISA -- avoid tl.where in hot loops, use tl.constexpr aggressively, "
        "minimize live variables across tl.dot calls.\n"
    ),
    "memory-bound": (
        "## Optimization Guidance (bottleneck: memory-bound)\n"
        "The kernel is limited by memory bandwidth. Focus on kernel-body changes:\n"
        "1. VECTORIZED LOADS: Use float4/float2 vector loads to maximize HBM throughput.\n"
        "2. COALESCED ACCESS: Ensure adjacent threads access adjacent memory addresses.\n"
        "3. LDS STAGING: Stage global memory reads through LDS to improve access patterns.\n"
        "4. REDUCE DATA MOVEMENT: Recompute values instead of storing and reloading them.\n"
        "5. OPERATION FUSION: Fuse the memory-bound kernel with adjacent elementwise ops "
        "to amortize memory access cost over more computation.\n"
        "6. TILING / BLOCKING: Increase tile sizes to improve data reuse from L2 cache.\n"
    ),
    "compute-bound": (
        "## Optimization Guidance (bottleneck: compute-bound)\n"
        "The kernel is limited by arithmetic throughput. Focus on kernel-body changes:\n"
        "1. REDUCE INSTRUCTION COUNT: Simplify expressions, use hardware intrinsics "
        "(tl.math.rsqrt, fma), eliminate redundant computations.\n"
        "2. USE MFMA INSTRUCTIONS: On AMD GPUs, restructure computation to use Matrix "
        "Fused Multiply-Add for dense linear algebra.\n"
        "3. STRENGTH REDUCTION: Replace expensive ops (div, mod, pow) with cheaper "
        "equivalents (shifts, masks, lookup tables).\n"
        "4. LOOP UNROLLING: Manually unroll inner loops to help the compiler schedule "
        "instructions more aggressively.\n"
        "5. ALGORITHM CHANGE: Switch to an algorithm with lower computational complexity "
        "(e.g., O(n log n) vs O(n^2), approximate methods).\n"
    ),
    "latency-bound": (
        "## Optimization Guidance (bottleneck: latency-bound)\n"
        "The kernel is too short to saturate any resource. Focus on kernel-body changes:\n"
        "1. INCREASE WORK PER KERNEL: Process more elements per thread or per block "
        "to amortize kernel launch overhead.\n"
        "2. FUSE KERNELS: Merge this kernel with adjacent ones to eliminate launch gaps.\n"
        "3. PERSISTENT KERNEL: Convert to a persistent kernel pattern that stays resident "
        "and processes multiple tiles without relaunching.\n"
        "4. INCREASE BLOCK SIZE: Use larger thread blocks to improve GPU occupancy for "
        "this short-running kernel.\n"
    ),
    "lds-bound": (
        "## Optimization Guidance (bottleneck: lds-bound)\n"
        "The kernel is limited by LDS (Local Data Share) bandwidth or capacity.\n"
        "1. REDUCE LDS BANK CONFLICTS: Pad shared memory arrays to avoid stride-32 "
        "access patterns (on AMD: 32 banks, 4 bytes each).\n"
        "2. REDUCE LDS USAGE: Move data from LDS to registers where possible to free "
        "LDS capacity and improve occupancy.\n"
        "3. OPTIMIZE LDS ACCESS PATTERN: Restructure loops so that LDS reads/writes "
        "are coalesced within each wavefront.\n"
        "4. SPLIT COMPUTATION: Break the kernel into phases that use LDS at different "
        "times to reduce peak LDS pressure.\n"
    ),
}

_SEARCH_WORKLOAD_HINTS = (
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


def _search_workload_guidance(metrics: dict) -> list[str]:
    """Add narrower guidance for latency-bound HIP search workloads."""
    evidence_chunks = [str(metrics.get("kernel_name", ""))]
    for top in metrics.get("top_kernels", []) or []:
        evidence_chunks.append(str(top.get("name", "")))
    haystack = " ".join(evidence_chunks).lower()
    if not any(hint in haystack for hint in _SEARCH_WORKLOAD_HINTS):
        return []

    bottleneck = derive_primary_bottleneck(metrics)
    if "latency" not in bottleneck:
        return []

    top_kernels = metrics.get("top_kernels", [])
    all_hbm = [_safe_float((k.get("metrics", {}) or {}).get("memory.hbm_bandwidth_utilization")) for k in top_kernels]
    all_l2 = [_safe_float((k.get("metrics", {}) or {}).get("memory.l2_hit_rate")) for k in top_kernels]
    hbm_util = max((h for h in all_hbm if h is not None), default=None)
    l2_hit = min((h for h in all_l2 if h is not None), default=None)
    if hbm_util is not None and hbm_util >= 10.0:
        return []

    hbm_text = f"{hbm_util:.1f}%" if hbm_util is not None else "unknown"
    l2_text = f"{l2_hit:.1f}%" if l2_hit is not None else "unknown"
    return [
        "## Workload Guidance (HIP search / pointer-chasing)",
        (
            "Profiler evidence suggests a latency-bound search workload: "
            f"HBM utilization={hbm_text}, L2 hit rate={l2_text}."
        ),
        "Prioritize branchless search logic, operation-specific specialization, and size-specialized variants.",
        "Also consider wavefront-cooperative upper-level search and amortized pivot-table narrowing when correctness rules allow it.",
        "Deprioritize generic vectorization or bandwidth-maximization ideas unless later profiling shows memory throughput is actually the limiter.",
        "",
    ]


def _guidance_block_lines(family: str) -> list[str]:
    aliases = {
        "memory": "memory-bound",
        "compute": "compute-bound",
        "latency": "latency-bound",
        "lds": "lds-bound",
        "balanced": "balanced",
    }
    family = aliases.get(family, family)
    family = family if family in _BOTTLENECK_GUIDANCE else "balanced"
    return _BOTTLENECK_GUIDANCE[family].strip().splitlines()


def build_bottleneck_guidance(metrics: dict) -> list[str]:
    """Return actionable optimization guidance lines based on bottleneck mix."""
    families = guidance_bottlenecks(metrics)
    primary = derive_primary_bottleneck(metrics)
    if not families:
        families = [primary if primary in _BOTTLENECK_GUIDANCE else "balanced"]

    if len(families) == 1:
        lines = _guidance_block_lines(families[0])
        lines.extend(_search_workload_guidance(metrics))
        lines.append("")
        return lines

    lines = [
        "## Optimization Guidance (mixed bottlenecks)",
        "Prioritize kernel-body changes that address the dominant families below first.",
        "",
    ]
    for family in families:
        block = _guidance_block_lines(family)
        lines.append(f"### Focus Area: {family}-bound")
        lines.extend(block[1:])
        lines.append("")
    lines.extend(_search_workload_guidance(metrics))
    lines.append("")
    return lines


def format_gpu_info(gpu_info: dict) -> list[str]:
    """Format GPU info dict into context lines for agent prompts."""
    if not gpu_info:
        return []
    arch = gpu_info.get("architecture", gpu_info.get("gfx_version", "unknown"))
    name = gpu_info.get("name", gpu_info.get("model", "AMD GPU"))
    cus = gpu_info.get("compute_units", "?")
    hbm_bw = gpu_info.get("peak_hbm_bandwidth_gbps", gpu_info.get("hbm_bandwidth", "?"))
    lds_per_cu = gpu_info.get("lds_per_cu_kb", 64)
    vgprs = gpu_info.get("vgprs_per_cu", 512)
    return [
        f"## GPU Architecture: {name} ({arch})",
        f"- Architecture: {arch}",
        f"- Compute Units: {cus}",
        f"- Peak HBM bandwidth: {hbm_bw} GB/s",
        f"- LDS per CU: {lds_per_cu} KB (32 banks on gfx9xx)",
        f"- VGPRs per CU: {vgprs}",
        "- Wavefront size: 64 (AMD default), some kernels can use 32",
        "- MFMA (Matrix Fused Multiply-Add) instructions available for dense math",
        "- Use these specs to guide your kernel optimizations (tile sizes, occupancy, LDS usage).",
        "",
    ]


def gpu_arch_context(profiling_path: str) -> list[str]:
    """Extract GPU architecture info from profile.json and format it."""
    try:
        data = json.loads(Path(profiling_path).read_text())
    except Exception:
        logger.debug("Could not read or parse profiling JSON at %s", profiling_path, exc_info=True)
        return []

    results = data.get("results", [])
    if not results:
        return []

    gpu_info = results[0].get("gpu_info", {}) if isinstance(results[0], dict) else {}
    if not gpu_info:
        for result in results:
            if isinstance(result, dict) and result.get("gpu_info"):
                gpu_info = result["gpu_info"]
                break

    return format_gpu_info(gpu_info)


def inject_pipeline_context(
    task_body: str,
    config: dict,
    *,
    commandment_text: str | None = None,
    baseline_metrics: dict | None = None,
    profiling_path: str | None = None,
    kernel_path: str | None = None,
    repo_root: str | None = None,
    test_command: str | None = None,
    codebase_context: str | None = None,
    benchmark_baseline: str | None = None,
) -> tuple[str, dict]:
    """Prepend pipeline context to *task_body* and augment *config*."""
    cfg = dict(config)
    ctx: list[str] = [
        "## Pipeline Context (auto-injected from task metadata)",
        "",
    ]

    if kernel_path:
        ctx.append(f"KERNEL FILE TO EDIT: {kernel_path}")
    if repo_root:
        ctx.append(f"REPO ROOT: {repo_root}")
    if test_command:
        ctx.append(f"TEST COMMAND: {test_command}")
    ctx.append("")

    ctx.append(
        "IMPORTANT: Only edit files within your REPO ROOT directory. "
        "Do NOT search or modify files outside of it. "
        "The KERNEL FILE TO EDIT path above is the exact file you should optimize."
    )
    ctx.append("")

    if commandment_text:
        ctx.append("## COMMANDMENT (evaluation contract -- you MUST follow these rules)")
        ctx.append(commandment_text.strip())
        ctx.append("")

    if baseline_metrics:
        dur = baseline_metrics.get("benchmark_duration_us", baseline_metrics.get("duration_us", "unknown"))
        ctx.append("## Baseline Performance (your optimization must improve on these)")
        ctx.append(f"Duration: {dur} us")
        ctx.append(format_bottleneck_summary(baseline_metrics))
        top = baseline_metrics.get("top_kernels", [])
        if top:
            ctx.append("Profiled kernels (selected as relevant to the optimization target):")
            for kernel in top[:5]:
                km = kernel.get("metrics", {}) or {}
                bn_tag = f" [{kernel['bottleneck']}]" if kernel.get("bottleneck") else ""
                hbm = km.get("memory.hbm_bandwidth_utilization")
                l2 = km.get("memory.l2_hit_rate")
                line = (
                    f"  - {kernel.get('name', '?')}: "
                    f"{kernel.get('duration_us', '?')} us "
                    f"({kernel.get('pct_of_selected', '?')}%){bn_tag}"
                )
                if hbm is not None:
                    line += f"; HBM util={hbm:.1f}%"
                if l2 is not None:
                    line += f"; L2 hit={l2:.1f}%"
                ctx.append(line)
                for obs in kernel.get("observations", []):
                    ctx.append(f"    - {obs}")
        ctx.append("")
        ctx.extend(build_bottleneck_guidance(baseline_metrics))

    gpu_lines = format_gpu_info(baseline_metrics.get("gpu_info", {})) if baseline_metrics else []
    if not gpu_lines and profiling_path and Path(profiling_path).exists():
        gpu_lines = gpu_arch_context(profiling_path)
    ctx.extend(gpu_lines)

    if benchmark_baseline:
        ctx.append("## Benchmark Baseline (compare your save_and_test output against this)")
        ctx.append(
            "This is the original kernel's canonical benchmark output from the same full benchmark contract used for patch testing."
        )
        ctx.append("Your save_and_test output includes canonical benchmark results -- compare against these numbers.")
        ctx.append(f"```\n{benchmark_baseline.strip()}\n```")
        ctx.append("")

    if codebase_context:
        ctx.append("## Codebase Context (kernel dependency tree)")
        ctx.append(
            "The dependency tree below shows in-repo files the target kernel "
            "imports. Every listed dependency is a potential optimization "
            "target -- improving any of them can reduce overall latency."
        )
        ctx.append(codebase_context.strip())
        ctx.append("")
        cfg["codebase_context"] = codebase_context.strip()

    ctx.append(
        "IMPORTANT: Baseline profiling and performance metrics are already "
        "established and provided above. Do NOT run save_and_test for a "
        "baseline run. Start optimizing immediately."
    )
    ctx.append("")

    try:
        integration = importlib.import_module("minisweagent.memory.integration")
        assemble_memory_context = getattr(integration, "assemble_memory_context", None)
        if assemble_memory_context is not None:
            _bm = baseline_metrics or {}
            _mem_ctx = assemble_memory_context(
                kernel_path=kernel_path,
                profiling_metrics=_bm,
            )
            if _mem_ctx:
                ctx.append("## Optimization Memory (from past kernel optimization runs)")
                ctx.append(_mem_ctx.strip())
                ctx.append("")
    except Exception:
        logger.debug("Could not assemble optimization memory context", exc_info=True)

    enriched = "\n".join(ctx) + "\n" + task_body
    return enriched, cfg


def run_baseline_profile(
    test_command: str,
    *,
    gpu_id: int = 0,
    ensure_mcp_importable: Callable[[], None],
    extract_harness_path: Callable[[str], str],
) -> dict:
    """Profile the test harness via profiler-mcp (includes warmup)."""
    ensure_mcp_importable()
    profile_server = importlib.import_module("profiler_mcp.server")
    profile_kernel = profile_server.profile_kernel

    harness = extract_harness_path(test_command)
    profile_cmd = f"python {harness} --profile"

    _profile_fn = getattr(profile_kernel, "fn", profile_kernel)
    return _profile_fn(**build_metrix_profile_kwargs(profile_cmd, gpu_id, quick=False))
