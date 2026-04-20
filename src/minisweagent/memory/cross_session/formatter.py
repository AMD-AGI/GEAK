"""Format retrieved experiences into a concise, actionable context block.

Key design principle: show enough for the orchestrator to REASON about
whether each strategy applies to the CURRENT kernel, not blindly copy.
Keeps context under ~15K chars to avoid drowning the kernel's own code.
"""

from __future__ import annotations

import re
from pathlib import Path

from minisweagent.memory.cross_session.schemas import ExperienceRecord, StrategySkill

_MAX_CONTEXT_FULL = 20_000  # was 60_000 — reduced to avoid prompt dilution
_MAX_CONTEXT_COMPACT = 4_000  # was 8_000 — same reason

_MAX_BEST_PATCH_CHARS = 4_000  # was 8_000 — show enough to reason, not enough to dominate
_MAX_REGRESSION_PATCH_CHARS = 1_500  # was 3_500
_TOP_IMPROVED_STRATEGIES = 3  # was 5
_TOP_REGRESSED_STRATEGIES = 2  # was 3 — dead-ends are reference, not focus
_MAX_BASELINE_BENCHMARK_CHARS = 1_500  # was 3_500

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
    ("prealloc_output", re.compile(r"torch\.empty\(|torch\.empty_like\(|output_tensor\s*\.copy_|prealloc", re.IGNORECASE)),
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


def _build_adaptation_note(exp, target_kernel_path: str) -> str:
    """Produce a per-experience adaptation note based on CODE comparison.

    Code identity is the source of truth — names are organizational labels
    that don't reflect semantic equivalence (e.g. ``fused_qkv_rope`` and
    ``fused_qkv_MLA`` may share a name suffix but have completely different
    code).

    Compares the experience's stored ``original_kernel_code`` against the
    current ``target_kernel_path`` content:
      * Codes match exactly → "verified for this exact code; apply verbatim"
      * Codes differ (or unavailable) → "related kernel; adapt the technique"
    """
    if not target_kernel_path:
        return ""
    kb_code = (getattr(exp, "original_kernel_code", "") or "").strip()
    cur_code = ""
    try:
        cur_code = Path(target_kernel_path).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        cur_code = ""
    if kb_code and cur_code and kb_code == cur_code:
        return (
            "**SAME CODE — VERIFIED PATCH** — the patch above was measured on the "
            "EXACT same source you are optimizing now. Apply it directly via "
            "`str_replace` / `write` / `git apply` to reproduce the verified speedup."
        )
    return (
        "**Adaptation note:** the patch above was measured on a related kernel "
        "with different source. Apply the *technique* (key params above), but "
        "translate signatures / variable names / control flow to fit your kernel."
    )


def format_landscape_context(
    experiences: list[ExperienceRecord],
    skills: list[StrategySkill] | None = None,
    query_category: str = "",
    query_bottleneck: str = "",
    query_language: str = "",
    compact: bool = False,
    target_kernel_path: str = "",
) -> str:
    """Format cross-session experiences into an "added context" block.

    Two forms:
      * compact=True  — strategy names + speedups only (no code diffs).
      * compact=False — full evidence per entry: baseline kernel_structure,
        strategies per round with diffs, profiler insights, dead-ends.

    ``target_kernel_path`` enables an exact-code-match check that only
    emits an imperative reference note when the stored baseline code is
    byte-identical to the current kernel.py (i.e. the diff will apply
    verbatim). For all other cases the entries are shown below as
    reference; the agent forms its own plan and uses them for informed
    cross-reference, not as directives to follow.
    """
    if not experiences and not skills:
        return ""

    lang = query_language or _guess_language(experiences)
    parts: list[str] = []
    n = len(experiences)
    parts.append(f"### Cross-Session Memory (from {n} similar kernel{'s' if n != 1 else ''})")
    parts.append("")

    top_hint = _build_top_hint(experiences, target_kernel_path) if experiences else ""
    if top_hint:
        parts.append(top_hint)
        parts.append("")

    parts.append(_build_reasoning_guidance(lang))
    parts.append("")

    budget = _MAX_CONTEXT_COMPACT if compact else _MAX_CONTEXT_FULL
    for exp in experiences:
        exp_dict = exp.to_dict() if hasattr(exp, "to_dict") else {}
        chunk = _format_compact(exp, exp_dict) if compact else _format_single_experience(
            exp, exp_dict, target_kernel_path=target_kernel_path
        )
        if budget - len(chunk) < 0 and parts:
            parts.append(f"\n*(budget reached — {n - experiences.index(exp)} more omitted)*")
            break
        budget -= len(chunk)
        parts.append(chunk)

    return "\n".join(parts)


def _classify_kb_match(top, target_kernel_path: str) -> tuple[str, str]:
    """Classify the KB entry purely by CODE comparison.

    Names are organizational labels and don't reflect semantic equivalence
    (e.g. ``fused_qkv_rope`` and ``fused_qkv_MLA`` may share name tokens
    but have completely different code; the same name might also refer to
    different versions of the same kernel as it evolves). This function
    only looks at code content:

      * KB entry's stored ``original_kernel_code`` matches the current
        ``target_kernel_path`` content exactly → SAME CODE, patch applies
        verbatim. The verified speedup is reproducible.
      * Otherwise → RELATED KERNEL — present the technique, agent adapts.

    Returns ``(tag, hint)`` where ``tag`` is a short bracketed annotation
    inserted into the FIRST-MOVE label, and ``hint`` is the verb hint
    used in the decision prompt.
    """
    kb_code = (getattr(top, "original_kernel_code", "") or "").strip()
    cur_code = ""
    if target_kernel_path:
        try:
            cur_code = Path(target_kernel_path).read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            cur_code = ""
    if kb_code and cur_code and kb_code == cur_code:
        return " [SAME CODE — diff applies verbatim]", "apply verbatim"
    return "", "adapt the technique to your kernel (KB entry was measured on different code)"


def _build_top_hint(experiences: list[ExperienceRecord], target_kernel_path: str) -> str:
    """Emit a concise reference label only when an EXACT code match exists.

    Design (passive RAG style): the KB is REFERENCE material, not a directive.
    Only emit a "top hit" hint when the stored ``original_kernel_code`` is
    byte-identical to the current kernel — i.e. the diff is GUARANTEED to
    apply verbatim. For all other cases the diffs are still shown below as
    optional reference; the agent decides applicability without prompting.

    This avoids biasing the agent toward incremental KB-style strategies
    when fundamentally different (and potentially better) optimizations
    are reachable from scratch on the actual current kernel.
    """
    if not experiences:
        return ""
    if not target_kernel_path:
        return ""
    try:
        cur_code = Path(target_kernel_path).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""

    # Find the highest-speedup experience whose ``original_kernel_code`` is
    # byte-identical to the current kernel — i.e. whose stored diff WILL
    # apply verbatim (guaranteed). We scan across all retrieved experiences
    # rather than only ``experiences[0]`` because the retriever's ranking
    # is text-similarity driven and may float a lower-speedup entry to the
    # top even when a higher-speedup entry on the IDENTICAL source exists.
    best_match = None
    best_sp = 0.0
    for exp in experiences:
        if not getattr(exp, "success", False):
            continue
        sp = float(getattr(exp, "best_speedup", 0.0) or 0.0)
        if sp <= 1.03:
            continue
        kb_code = (getattr(exp, "original_kernel_code", "") or "").strip()
        if kb_code and kb_code == cur_code and sp > best_sp:
            best_match, best_sp = exp, sp

    if best_match is None:
        return ""

    return (
        f"**EXACT-CODE MATCH FOUND**: The KB contains a patch that was measured on "
        f"byte-identical source to your current kernel.py — strategy "
        f"`{best_match.best_strategy}` verified **{best_sp:.2f}x** speedup. "
        f"The diff is reproduced verbatim below. Because the baseline code is "
        f"IDENTICAL, applying this diff directly (via `str_replace` / `write` / "
        f"`git apply`) is expected to reproduce the measured speedup exactly.\n\n"
        f"This does not prescribe behavior: if your analysis of the current "
        f"kernel + profile suggests a fundamentally different optimization with "
        f"higher potential, pursue that instead. Treat the KB patch as a strong, "
        f"free starting-point candidate — one option among your own ideas."
    )


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
        "*Below is ADDED CONTEXT from past optimization runs (similar — not "
        "identical — kernels). Each entry includes: the baseline kernel.py "
        "of that past run, the strategies tried per round, the actual code "
        "diffs (winning + regressions), measured speedups, dead-ends to "
        "avoid, profiler insights, and the round-by-round trajectory.*\n\n"
        "*Examine each entry: compare its baseline code to YOUR current "
        "kernel.py, its bottleneck to YOUR profile output, its diffs to "
        "what would actually apply here. Then make an informed decision: "
        "adopt verbatim if applicable, adapt the technique if partially "
        "applicable, or set aside and proceed with your own analysis. The "
        "KB does not prescribe — it informs.*"
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


def _format_single_experience(exp: ExperienceRecord, exp_dict: dict, target_kernel_path: str = "") -> str:
    """Format a single experience with all rich fields.

    Strategies are the single source of truth -- every strategy has a
    measured speedup AND the full code diff.  We split them into
    'what worked' (speedup >= _SPEEDUP_THRESHOLD = 1.10x) and 'what regressed' (speedup < 1.0)
    so the agent sees both the code to copy and the code to avoid.

    We also surface a short ``key params`` list extracted from the best
    strategy's diff so the LLM can directly apply the specific numeric /
    dtype values that made the winning patch work, without having to parse
    the full 5k-char diff.
    """
    parts: list[str] = []
    kn = exp.kernel_name or "unknown"
    sp = exp.best_speedup

    parts.append(f"## {kn} ({sp:.2f}x speedup, {exp.bottleneck_type}-bound)")

    key_insight = exp.key_insight if exp.key_insight else exp_dict.get("key_insight", "")
    if key_insight:
        parts.append(f"*{key_insight}*")

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
        parts.append("\n**Key params from winning patch (apply these values directly):**")
        for kp in key_params:
            parts.append(f"  - {kp}")

    adaptation = _build_adaptation_note(exp, target_kernel_path)
    if adaptation:
        parts.append(f"\n{adaptation}")

    profiling = exp_dict.get("profiling_insight", "")
    if profiling:
        parts.append(f"\n**Profiling insight:** {profiling}")

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
