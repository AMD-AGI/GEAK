"""Format retrieved experiences into a concise, actionable context block.

Key design principle: show enough for the orchestrator to REASON about
whether each strategy applies to the CURRENT kernel, not blindly copy.
Keeps context under ~15K chars to avoid drowning the kernel's own code.
"""

from __future__ import annotations

import re

from minisweagent.memory.cross_session.schemas import ExperienceRecord, StrategySkill

_MAX_CONTEXT_FULL = 40_000  # raised from 20k so the 20-40kB winning patches aren't truncated
_MAX_CONTEXT_COMPACT = 8_000  # raised from 4k to surface more strategy names with their speedups

_MAX_BEST_PATCH_CHARS = 8_000  # raised from 4k — full diffs for top-3 winners fit directly
_MAX_REGRESSION_PATCH_CHARS = 2_000  # raised from 1.5k — dead-ends still useful as code-level anti-patterns
_TOP_IMPROVED_STRATEGIES = 5  # raised from 3 — agent sees more of the working space
_TOP_REGRESSED_STRATEGIES = 3  # raised from 2 — more anti-patterns to avoid
_MAX_BASELINE_BENCHMARK_CHARS = 2_000  # raised from 1.5k

# Single-source-of-truth speedup threshold for classifying strategies as
# WORKED vs MARGINAL vs REGRESSED. Mirrors the KB inclusion threshold
# (``min_store_speedup`` in config.py / ``GEAK_MEMORY_MIN_SPEEDUP`` env var
# / ``min_speedup_threshold`` in knowledge_base.json). Default 1.10x.
import os as _os

_SPEEDUP_THRESHOLD = float(_os.environ.get("GEAK_MEMORY_MIN_SPEEDUP", "1.10"))

# Regexes for extracting high-signal optimization parameters from a winning
# Triton patch. Each captured value is surfaced to the LLM as a short,
# actionable ``key params`` list so it can apply the specific value directly
# rather than having to parse a 5k-char diff.
_PARAM_PATTERNS = [
    ("num_warps", re.compile(r"num_warps\s*=\s*(\d+)")),
    ("num_stages", re.compile(r"num_stages\s*=\s*(\d+)")),
    ("BLOCK_SIZE", re.compile(r"BLOCK_(?:T|M|N|K|D|D_HALF)\s*=\s*(\d+)")),
    ("XBLOCK", re.compile(r"XBLOCK\s*[:=]\s*(\d+)")),
    ("dtype_cast", re.compile(r"\.to\(\s*tl\.(float32|float16|bfloat16|int32|int64|uint32|uint64)\s*\)")),
    ("bitcast", re.compile(r"bitcast\s*=\s*True|tl\.(int32|uint32)\s*,\s*bitcast")),
    ("tl_arange_dtype", re.compile(r"tl\.arange\([^)]*\)\.to\(\s*tl\.(int32|int64)\s*\)")),
    (
        "prealloc_output",
        re.compile(r"torch\.empty\(|torch\.empty_like\(|output_tensor\s*\.copy_|prealloc", re.IGNORECASE),
    ),
    ("tl_constexpr_dtype", re.compile(r"tl\.constexpr\s*=\s*tl\.(int32|int64|float32|float16)")),
    ("cuda_graph", re.compile(r"torch\.cuda\.CUDAGraph|cudagraph|cuda_graph", re.IGNORECASE)),
    ("hip_extension", re.compile(r"torch\.utils\.cpp_extension\.load|load_inline\(|hipLaunchKernelGGL")),
]


def _extract_key_params(patch_text: str, max_params: int = 8) -> list[str]:
    """Extract high-signal optimization parameters from a patch's ``+`` lines only.

    We restrict matching to the lines added by the patch (``+`` prefix) so
    the extracted values reflect what the winning change introduced, not
    what the original code already had. Duplicates are deduped, and at
    most ``max_params`` distinct params are returned.
    """
    if not patch_text:
        return []

    added_lines = [line[1:] for line in patch_text.splitlines() if line.startswith("+") and not line.startswith("+++")]
    if not added_lines:
        return []
    added_text = "\n".join(added_lines)

    found: list[str] = []
    seen: set[str] = set()
    for _label, pat in _PARAM_PATTERNS:
        for m in pat.finditer(added_text):
            val = m.group(0).strip()
            if val in seen:
                continue
            seen.add(val)
            found.append(val)
            if len(found) >= max_params:
                return found
    return found


def _code_fingerprint(code: str) -> str:
    """Short structural fingerprint the agent can compare across entries.

    Returns ``"{byte_length}B, sha256=abcdef12..."``. Lets the agent judge
    identity by comparing fingerprints rather than requiring us to inject
    a special-case banner.
    """
    import hashlib

    if not code:
        return "absent"
    return f"{len(code)}B, sha256={hashlib.sha256(code.encode()).hexdigest()[:12]}"


def format_landscape_context(
    experiences: list[ExperienceRecord],
    skills: list[StrategySkill] | None = None,
    query_bottleneck: str = "",
    compact: bool = False,
    target_code: str = "",
    per_entry_code_sim: list[float] | None = None,
) -> str:
    """Format cross-session experiences into an "added context" block.

    Two forms:
      * compact=True  — strategy names + speedups only (no code diffs).
      * compact=False — full evidence per entry: baseline kernel_structure,
        strategies per round with diffs, profiler insights, dead-ends.

    ``target_code`` is the raw source of the kernel being optimized right
    now. It is used to emit the agent's own fingerprint at the top of the
    context so the agent can compare it to each KB entry's stored code
    fingerprint and decide verbatim vs translate.

    ``per_entry_code_sim`` is a parallel list to ``experiences`` giving
    the Jaccard similarity between ``target_code`` and each entry's
    stored ``original_kernel_code``. Surfaced per entry so the agent
    weighs KB hints by their actual overlap -- 100% means verbatim, 25%
    means partial same-family, 10% means distant cross-family reference
    only. When every entry is <25%, a top-of-context hint is emitted so
    the agent doesn't anchor on a weak match.
    """
    if not experiences and not skills:
        return ""

    if per_entry_code_sim is None or len(per_entry_code_sim) != len(experiences):
        per_entry_code_sim = [0.0] * len(experiences)

    lang = _guess_language(experiences)
    parts: list[str] = []
    n = len(experiences)
    parts.append(f"### Cross-Session Memory (from {n} similar kernel{'s' if n != 1 else ''})")
    parts.append("")

    # Provide the agent's own kernel fingerprint so it can cross-reference
    # against each KB entry's ``Code fingerprint`` line shown below. When a
    # KB entry's fingerprint matches this value byte-for-byte, the stored
    # patch is guaranteed to apply and is expected to reproduce the
    # measured speedup -- the agent reads that off the evidence directly,
    # no prescriptive banner required.
    if target_code:
        parts.append(f"**Your kernel fingerprint**: {_code_fingerprint(target_code)}")
        parts.append("")

    parts.append(_build_reasoning_guidance(lang))
    parts.append("")

    budget = _MAX_CONTEXT_COMPACT if compact else _MAX_CONTEXT_FULL
    for idx, exp in enumerate(experiences):
        exp_dict = exp.to_dict() if hasattr(exp, "to_dict") else {}
        cs = per_entry_code_sim[idx] if idx < len(per_entry_code_sim) else 0.0
        chunk = _format_compact(exp, exp_dict) if compact else _format_single_experience(exp, exp_dict, code_sim=cs)
        if budget - len(chunk) < 0 and parts:
            parts.append(f"\n*(budget reached — {n - idx} more omitted)*")
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
    """Informed-decision framing — agent cross-references KB with current state.

    The previous prescriptive version ("Decide", "skip GEMM strategies", "make
    it your first priority") biased the agent into KB-anchored thinking even
    when fundamentally different optimizations were reachable.

    The new framing makes explicit what the agent should examine in the KB
    entries below (strategies tried, baseline code of the KB kernel, code
    diffs, what worked vs what didn't, profiler insights, round trajectory)
    and cross-reference against (a) the current kernel.py source it sees,
    (b) the current profile.json, (c) the current bottleneck — then decide
    based on that informed comparison.
    """
    return (
        "*Below: evidence from past optimization runs. Each entry reports "
        "its hardware, baseline→best latency, bottleneck, code similarity "
        "to your current kernel (Jaccard line-overlap %), code_fingerprint, "
        "key_insight, extracted key params, profiler insight, round-by-round "
        "results, winning diffs, and regression diffs. You also have your "
        "current kernel's full source and profiler metrics from the main "
        "task. Use both inputs to form your own plan -- the KB informs "
        "your decision, it does not make it for you.*"
    )


def _format_compact(exp: ExperienceRecord, exp_dict: dict) -> str:
    """Lightweight format: key insight + strategy names with speedups, no code diffs.

    For agents that get the full task + memory in a single message and can't
    do multi-step reasoning about which strategies to adopt.
    """
    parts: list[str] = []
    kn = exp.kernel_name or "unknown"
    sp = exp.best_speedup
    parts.append(f"## {kn} ({sp:.2f}x, {exp.bottleneck_type}-bound)")

    key_insight = exp.key_insight or exp_dict.get("key_insight", "")
    if key_insight:
        parts.append(f"*{key_insight}*")

    strategies = exp_dict.get("strategies", [])
    if strategies:
        improved = [s for s in strategies if s.get("speedup", 0) >= _SPEEDUP_THRESHOLD]
        improved.sort(key=lambda s: -s["speedup"])
        if improved:
            parts.append("Worked: " + ", ".join(f"{s['task']}={s['speedup']}x" for s in improved[:5]))
        regressed = [s for s in strategies if 0 < s.get("speedup", 0) < 1.0]
        if regressed:
            parts.append("Avoid: " + ", ".join(f"{s['task']}={s['speedup']}x" for s in regressed[:3]))

    return "\n".join(p for p in parts if p)


def _format_single_experience(exp: ExperienceRecord, exp_dict: dict, code_sim: float = 0.0) -> str:
    """Format a single experience with all rich fields.

    Strategies are the single source of truth -- every strategy has a
    measured speedup AND the full code diff.  We split them into
    'what worked' (speedup >= _SPEEDUP_THRESHOLD = 1.10x) and 'what regressed' (speedup < 1.0)
    so the agent sees both the code to copy and the code to avoid.

    We also surface a short ``key params`` list extracted from the best
    strategy's diff so the LLM can directly apply the specific numeric /
    dtype values that made the winning patch work, without having to parse
    the full 5k-char diff.

    ``code_sim`` is the Jaccard line-overlap between this entry's stored
    ``original_kernel_code`` and the target kernel's current source. It
    gets printed alongside the evidence so the agent can judge how much
    weight to give this entry: 100% = apply verbatim, 25-60% = adapt
    techniques, <25% = distant cross-family reference only.
    """
    parts: list[str] = []
    kn = exp.kernel_name or "unknown"
    sp = exp.best_speedup

    parts.append(f"## {kn} ({sp:.2f}x speedup, {exp.bottleneck_type}-bound)")

    # Generic evidence header — everything the agent needs to judge whether
    # this entry applies. No special-case banner for byte-identical code:
    # the agent compares the fingerprints and other dimensions below against
    # its own context and makes the call.
    hw = getattr(exp, "hardware", "") or exp_dict.get("hardware", "")
    category = exp_dict.get("kernel_category", "") or ""
    language = exp_dict.get("kernel_language", "") or ""
    baseline_ms = getattr(exp, "baseline_latency_ms", 0) or 0
    best_ms = getattr(exp, "best_latency_ms", 0) or 0
    code_fp = _code_fingerprint(exp_dict.get("original_kernel_code", "") or "")
    best_strat = exp_dict.get("best_strategy", "") or ""
    change_cat = exp_dict.get("best_change_category", "") or ""
    kstruct = exp_dict.get("kernel_structure", "") or ""
    timestamp = exp_dict.get("timestamp", "") or ""

    parts.append(
        f"**Context**: hardware=`{hw or 'unknown'}` | language=`{language}` | "
        f"category=`{category}` | recorded={timestamp[:10] if timestamp else '?'}"
    )
    parts.append(f"**Code similarity to your kernel**: {code_sim * 100.0:.1f}% (Jaccard over source lines)")
    parts.append(
        f"**Performance**: baseline={baseline_ms:.4f}ms → best={best_ms:.4f}ms ({sp:.4f}x, {exp.bottleneck_type}-bound)"
    )
    parts.append(f"**Code fingerprint**: `{code_fp}`  ← compare byte-for-byte to your own fingerprint above")
    if kstruct:
        parts.append(f"**Kernel structure**: {kstruct}")
    if best_strat:
        parts.append(f"**Best strategy**: `{best_strat}`" + (f" (change_category={change_cat})" if change_cat else ""))

    key_insight = exp.key_insight if exp.key_insight else exp_dict.get("key_insight", "")
    if key_insight:
        parts.append(f"*Technique: {key_insight}*")

    change_summary = exp_dict.get("code_changes_summary", "") or ""
    if change_summary and len(change_summary) > 20:
        parts.append(f"*What the winning diff actually does: {change_summary[:600]}*")

    # Surface the specific optimization parameters from the best strategy's
    # diff as a short actionable list. This gives the LLM the concrete
    # values (num_warps, BLOCK_T, dtype casts, etc.) it should try, rather
    # than requiring it to parse a 5k-char diff to extract them.
    strategies_list = exp_dict.get("strategies", [])
    best_patch_text = ""
    if strategies_list:
        improved_list = sorted(
            [s for s in strategies_list if s.get("speedup", 0) >= _SPEEDUP_THRESHOLD],
            key=lambda s: -s["speedup"],
        )
        if improved_list:
            best_patch_text = improved_list[0].get("after_code", "") or ""
    if not best_patch_text:
        best_patch_text = exp_dict.get("patch_content", "") or exp.patch_content or ""
    key_params = _extract_key_params(best_patch_text) if best_patch_text else []
    if key_params:
        parts.append("\n**Key params from winning patch:**")
        for kp in key_params:
            parts.append(f"  - {kp}")

    profiling = exp_dict.get("profiling_insight", "")
    if profiling:
        parts.append(f"\n**Profiling insight:** {profiling}")

    prof_metrics = exp_dict.get("profiling_metrics", {}) or {}
    if prof_metrics and isinstance(prof_metrics, dict):
        keep = {
            k: v
            for k, v in prof_metrics.items()
            if k
            in (
                "bottleneck_type",
                "baseline_latency_ms",
                "best_latency_ms",
                "speedup",
                "benchmark_iterations",
                "shapes_count",
                "warmup_iterations",
                "note",
            )
        }
        if keep:
            parts.append("**Profiling metrics (numeric):** " + ", ".join(f"{k}={v}" for k, v in keep.items()))

    baseline_bm = exp_dict.get("baseline_benchmark", "")
    if baseline_bm:
        parts.append(f"\n**Baseline per-shape benchmark:**\n```\n{baseline_bm[:_MAX_BASELINE_BENCHMARK_CHARS]}\n```")

    round_insights = exp_dict.get("round_insights", [])
    if round_insights:
        parts.append("\n**Round-by-round results:**")
        for ri in round_insights:
            parts.append(f"  {ri}")

    strategies = exp_dict.get("strategies", [])
    if strategies:
        improved = sorted(
            [s for s in strategies if s.get("speedup", 0) >= _SPEEDUP_THRESHOLD],
            key=lambda s: -s["speedup"],
        )
        regressed = sorted(
            [s for s in strategies if 0 < s.get("speedup", 0) < 1.0],
            key=lambda s: s["speedup"],
        )

        if improved:
            parts.append(
                f"\n**Strategies that WORKED ({len(improved)} total, showing top "
                f"{_TOP_IMPROVED_STRATEGIES} with code):**"
            )
            for s in improved[:_TOP_IMPROVED_STRATEGIES]:
                parts.append(f"\n### {s['round']}/{s['task']} — {s['speedup']}x")
                code = s.get("after_code", "")
                if code:
                    parts.append(f"```diff\n{code[:_MAX_BEST_PATCH_CHARS]}\n```")
            if len(improved) > _TOP_IMPROVED_STRATEGIES:
                parts.append(
                    "\n*Also worked:* "
                    + ", ".join(f"{s['task']}={s['speedup']}x" for s in improved[_TOP_IMPROVED_STRATEGIES:])
                )

        if regressed:
            parts.append(
                f"\n**Strategies that REGRESSED ({len(regressed)} total, showing worst "
                f"{_TOP_REGRESSED_STRATEGIES} with code):**"
            )
            for s in regressed[:_TOP_REGRESSED_STRATEGIES]:
                parts.append(f"\n### AVOID: {s['round']}/{s['task']} — {s['speedup']}x (regression)")
                code = s.get("after_code", "")
                if code:
                    parts.append(f"```diff\n{code[:_MAX_REGRESSION_PATCH_CHARS]}\n```")
            if len(regressed) > _TOP_REGRESSED_STRATEGIES:
                parts.append(
                    "\n*Also regressed:* "
                    + ", ".join(f"{s['task']}={s['speedup']}x" for s in regressed[_TOP_REGRESSED_STRATEGIES:])
                )
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
