"""Format retrieved experiences into a concise, actionable context block.

Key design principle: show enough for the orchestrator to REASON about
whether each strategy applies to the CURRENT kernel, not blindly copy.
Keeps context under ~15K chars to avoid drowning the kernel's own code.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from minisweagent.memory.cross_session.schemas import ExperienceRecord, StrategySkill


_MAX_CONTEXT_CHARS = 30_000

def format_landscape_context(
    experiences: list[ExperienceRecord],
    skills: list[StrategySkill],
    query_category: str = "",
    query_bottleneck: str = "",
    query_language: str = "",
) -> str:
    """Format experiences into a concise context block with reasoning guidance."""
    if not experiences and not skills:
        return ""

    lang = query_language or _guess_language(experiences)
    parts: list[str] = []
    n = len(experiences)
    parts.append(f"### Cross-Session Memory (from {n} similar kernel{'s' if n != 1 else ''})")
    parts.append("")
    parts.append(_build_reasoning_guidance(lang))
    parts.append("")

    budget = _MAX_CONTEXT_CHARS
    for exp in experiences:
        exp_dict = exp.to_dict() if hasattr(exp, "to_dict") else {}
        chunk = _format_single_experience(exp, exp_dict)
        if budget - len(chunk) < 0 and parts:
            parts.append(f"\n*(context budget reached — {n - experiences.index(exp)} more experiences omitted)*")
            break
        budget -= len(chunk)
        parts.append(chunk)

    return "\n".join(parts)


def _guess_language(experiences: list[ExperienceRecord]) -> str:
    for exp in experiences:
        lang = getattr(exp, "kernel_language", "")
        if lang:
            return lang
    return ""


def _build_reasoning_guidance(lang: str) -> str:
    if lang == "hip":
        return (
            "**IMPORTANT**: These are from SIMILAR but NOT identical HIP kernels. "
            "Before adopting any strategy below, compare the code diff against "
            "YOUR kernel's actual architecture. Only use strategies where the "
            "underlying HIP patterns match (same __global__ kernel structure, "
            "same data access patterns, same bottleneck). For example, LDS tiling "
            "with SoA layout transfers well between spatial search kernels that "
            "iterate over point clouds, but a warp-parallel scan for KNN may not "
            "help a gather/scatter kernel with different access patterns."
        )
    return (
        "**IMPORTANT**: These are from SIMILAR but NOT identical kernels. "
        "Before adopting any strategy below, compare the code diff against "
        "YOUR kernel's actual architecture. Only use strategies where the "
        "underlying Triton patterns match (same tl.dot loop, same data types, "
        "same bottleneck). If the KB kernel uses split-K for GEMM but your "
        "kernel's bottleneck is in quantization/dequantization, skip the GEMM "
        "strategies and look for quant-related patterns instead."
    )


def _format_single_experience(exp: ExperienceRecord, exp_dict: dict) -> str:
    """Format a single experience with all rich fields.

    Strategies are the single source of truth -- every strategy has a
    measured speedup AND the full code diff.  We split them into
    'what worked' (speedup > 1.0) and 'what regressed' (speedup < 1.0)
    so the agent sees both the code to copy and the code to avoid.
    """
    parts: list[str] = []
    kn = exp.kernel_name or "unknown"
    sp = exp.best_speedup

    parts.append(f"## {kn} ({sp:.2f}x speedup, {exp.bottleneck_type}-bound)")

    key_insight = exp.key_insight if exp.key_insight else exp_dict.get("key_insight", "")
    if key_insight:
        parts.append(f"*{key_insight}*")

    profiling = exp_dict.get("profiling_insight", "")
    if profiling:
        parts.append(f"\n**Profiling insight:** {profiling}")

    baseline_bm = exp_dict.get("baseline_benchmark", "")
    if baseline_bm:
        parts.append(f"\n**Baseline per-shape benchmark:**\n```\n{baseline_bm[:1500]}\n```")

    round_insights = exp_dict.get("round_insights", [])
    if round_insights:
        parts.append("\n**Round-by-round results:**")
        for ri in round_insights:
            parts.append(f"  {ri}")

    strategies = exp_dict.get("strategies", [])
    if strategies:
        improved = sorted(
            [s for s in strategies if s.get("speedup", 0) > 1.0],
            key=lambda s: -s["speedup"],
        )
        regressed = sorted(
            [s for s in strategies if 0 < s.get("speedup", 0) < 1.0],
            key=lambda s: s["speedup"],
        )

        if improved:
            parts.append(f"\n**Strategies that WORKED ({len(improved)} total, showing top 3 with code):**")
            for s in improved[:3]:
                parts.append(f"\n### {s['round']}/{s['task']} — {s['speedup']}x")
                code = s.get("after_code", "")
                if code:
                    parts.append(f"```diff\n{code[:4000]}\n```")
            if len(improved) > 3:
                parts.append(f"\n*Also worked:* " + ", ".join(
                    f"{s['task']}={s['speedup']}x" for s in improved[3:]
                ))

        if regressed:
            parts.append(f"\n**Strategies that REGRESSED ({len(regressed)} total, showing worst 2 with code):**")
            for s in regressed[:2]:
                parts.append(f"\n### AVOID: {s['round']}/{s['task']} — {s['speedup']}x (regression)")
                code = s.get("after_code", "")
                if code:
                    parts.append(f"```diff\n{code[:3000]}\n```")
            if len(regressed) > 2:
                parts.append(f"\n*Also regressed:* " + ", ".join(
                    f"{s['task']}={s['speedup']}x" for s in regressed[2:]
                ))
    else:
        code_section = _best_code_insights_legacy(exp)
        if code_section:
            parts.append(code_section)

    parts.append(f"\n**Trajectory:** {exp.trajectory_sketch}" if exp.trajectory_sketch else "")

    return "\n".join(p for p in parts if p)


def _best_code_insights_legacy(exp: ExperienceRecord) -> str:
    """Fallback for experiences that only have patch_content (no strategies list)."""
    if not exp.patch_content and not exp.code_changes_summary:
        return ""

    parts = [f"**Proven code patterns from {exp.kernel_name} ({exp.best_speedup:.2f}x speedup):**"]

    if exp.code_changes_summary:
        parts.append(f"  Summary: {exp.code_changes_summary}")

    if exp.patch_content:
        parts.append(f"```diff\n{exp.patch_content[:4000]}\n```")

    return "\n".join(parts)
