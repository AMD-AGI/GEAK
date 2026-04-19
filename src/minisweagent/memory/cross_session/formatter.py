"""Format retrieved experiences into a concise, actionable context block.

Key design principle: show enough for the orchestrator to REASON about
whether each strategy applies to the CURRENT kernel, not blindly copy.
Keeps context under ~15K chars to avoid drowning the kernel's own code.
"""

from __future__ import annotations

import re
from pathlib import Path

from minisweagent.memory.cross_session.schemas import ExperienceRecord, StrategySkill

_MAX_CONTEXT_FULL = 60_000
_MAX_CONTEXT_COMPACT = 8_000

_MAX_BEST_PATCH_CHARS = 8_000
_MAX_REGRESSION_PATCH_CHARS = 3_500
_TOP_IMPROVED_STRATEGIES = 5
_TOP_REGRESSED_STRATEGIES = 3
_MAX_BASELINE_BENCHMARK_CHARS = 3_500

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


def _build_adaptation_note(kb_kernel_name: str, target_kernel_path: str) -> str:
    """Produce a tailored note based on whether KB entry is same/different kernel.

    For SAME-kernel KB entries (e.g. previously-verified fused_qkv_rope
    patches retrieved when optimizing fused_qkv_rope), emit a strong
    "directly applicable" message that overrides the default conservative
    "translate techniques" guidance in the agent prompt. The patch was
    measured on this exact file in a prior run and can be applied verbatim
    via ``str_replace`` / ``write`` / ``git apply``.

    For DIFFERENT-kernel KB entries, keep the "translate techniques"
    guidance — the LLM otherwise has to figure out that e.g. a monkey-patch
    for ``aiter.ops.triton.fused_qk_rope_cat_and_cache_mla`` needs to be
    translated into direct edits in a standalone ``kernel.py``.
    """
    if not kb_kernel_name or not target_kernel_path:
        return ""
    tgt_name = target_kernel_path.rsplit("/", 2)[-2] if "/" in target_kernel_path else ""
    if not tgt_name:
        return ""
    if tgt_name.lower() == kb_kernel_name.lower():
        return (
            f"**SAME-KERNEL VERIFIED PATCH** — the patch above was measured on the "
            f"EXACT kernel you are optimizing (`{tgt_name}`) in a prior run. "
            f"Apply it DIRECTLY via `str_replace` / `write` / `git apply` to reproduce "
            f"the verified speedup. Do NOT 'translate' or rewrite — copy the diff "
            f"verbatim, then iterate from there if you want to push further."
        )
    return (
        f"**Adaptation note:** KB patch was for `{kb_kernel_name}` (different kernel). "
        f"Your target is `{tgt_name}` — apply the same *techniques* (the `key params` above), "
        f"but translate the *structure* to your kernel's architecture rather than copying the patch verbatim."
    )


def format_landscape_context(
    experiences: list[ExperienceRecord],
    skills: list[StrategySkill],
    query_category: str = "",
    query_bottleneck: str = "",
    query_language: str = "",
    compact: bool = False,
    target_kernel_path: str = "",
    rag_snippets: list[dict] | None = None,
) -> str:
    """Format experiences into a context block with reasoning guidance.

    compact=True produces a lightweight summary (strategy names + speedups,
    no code diffs) suitable for homogeneous agents that can't do multi-step
    strategy reasoning.

    target_kernel_path, when provided, is used to emit a short adaptation
    note when the KB seed kernel differs from the target kernel, so the
    LLM knows to translate techniques rather than copy the patch verbatim.

    rag_snippets, when provided, injects a short "Domain KB" section of
    external AMD/ROCm/Triton reference knowledge retrieved from the
    rag-mcp markdown corpus (BM25-ranked against the target kernel's
    source). Complements the experience-based memory above with domain
    grounding.
    """
    if not experiences and not skills and not rag_snippets:
        return ""

    lang = query_language or _guess_language(experiences)
    parts: list[str] = []
    n = len(experiences)
    parts.append(f"### Cross-Session Memory (from {n} similar kernel{'s' if n != 1 else ''})")
    parts.append("")

    # Lead with a concrete, imperative directive based on the single best KB
    # entry so the LLM sees an actionable first-move before any caveats.
    top_hint = _build_top_hint(experiences, target_kernel_path) if experiences else ""
    if top_hint:
        parts.append(top_hint)
        parts.append("")

    # Domain-KB block (complementary signal: AMD aiter reports, Triton-on-ROCm
    # guides, kernel-family case studies). Kept compact: top 2 snippets.
    if rag_snippets:
        parts.append(_build_rag_block(rag_snippets, target_kernel_path))
        parts.append("")

    parts.append(_build_reasoning_guidance(lang))
    parts.append("")

    budget = _MAX_CONTEXT_COMPACT if compact else _MAX_CONTEXT_FULL
    for exp in experiences:
        exp_dict = exp.to_dict() if hasattr(exp, "to_dict") else {}
        if compact:
            chunk = _format_compact(exp, exp_dict)
        else:
            chunk = _format_single_experience(exp, exp_dict, target_kernel_path=target_kernel_path)
        if budget - len(chunk) < 0 and parts:
            parts.append(f"\n*(budget reached — {n - experiences.index(exp)} more omitted)*")
            break
        budget -= len(chunk)
        parts.append(chunk)

    return "\n".join(parts)


def _build_rag_block(snippets: list[dict], target_kernel_path: str) -> str:
    """Format the Domain-KB block from rag_hook.query_rag() results.

    Each snippet contributes ~1200-1500 chars. Top-2 default keeps the
    block bounded (~3000 chars total).
    """
    if not snippets:
        return ""
    lines: list[str] = []
    lines.append(
        "### Domain KB (AMD aiter / Triton-on-ROCm — complementary reference):"
    )
    lines.append(
        "*These snippets are retrieved from the AMD GPU knowledge base "
        "(rag-mcp) based on operation overlap with your kernel. Use them "
        "for hardware-specific tuning hints (warp size, HBM bandwidth, "
        "MFMA intrinsics, etc.) NOT as patches to copy. They are NOT from "
        "successful runs — they are reference material.*"
    )
    lines.append("")
    for idx, snip in enumerate(snippets, 1):
        title = snip.get("title", "")[:140]
        layer = snip.get("layer", "unknown")
        path = snip.get("path", "")
        body = snip.get("body", "")
        lines.append(f"**[KB-{idx}] {title}**  (from `{path}`, {layer})")
        lines.append("")
        lines.append(body.strip())
        lines.append("")
    return "\n".join(lines)


def _classify_kb_match(top, target_kernel_path: str) -> tuple[str, str]:
    """Classify how confidently the KB entry matches the current target kernel.

    Returns ``(tag, hint)`` where:
      - ``tag`` is a short bracketed annotation (e.g. ``[SAME CODE — diff
        applies verbatim]``) inserted into the FIRST-MOVE label.
      - ``hint`` is a short verb hint (``apply verbatim`` / ``adapt to
        your kernel``) used in the decision prompt.

    Identity is determined by CODE, not by name. Names like
    ``fused_qkv_rope`` and ``fused_qkv_MLA`` are organizational labels;
    actual semantics live in the source. We compare the KB entry's
    ``original_kernel_code`` to the current ``kernel.py`` content:

      * Exact match → SAME CODE → patch applies verbatim
      * KB entry has stored code, doesn't match → DIFFERENT CODE → diff
        likely won't apply cleanly; technique may still transfer
      * KB entry has no stored code → fall back to name comparison as
        a weaker prior (logged as `[name-only match]`)
    """
    target_name = ""
    if target_kernel_path:
        tail = target_kernel_path.rstrip("/").rsplit("/", 1)[-1]
        if tail.endswith(".py"):
            parts_ = target_kernel_path.rstrip("/").split("/")
            target_name = parts_[-2] if len(parts_) >= 2 else tail
        else:
            target_name = tail
    name_match = bool(
        target_name and top.kernel_name and target_name.lower() == top.kernel_name.lower()
    )

    kb_code = (getattr(top, "original_kernel_code", "") or "").strip()
    cur_code = ""
    if target_kernel_path:
        try:
            cur_code = Path(target_kernel_path).read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            cur_code = ""

    if kb_code and cur_code:
        if kb_code == cur_code:
            return " [SAME CODE — diff applies verbatim]", "apply verbatim"
        if name_match:
            return (
                " [SAME KERNEL NAME, DIFFERENT CODE — review diff for applicability]",
                "review diff against your current code; apply if compatible, else adapt",
            )
        return "", "adapt to your kernel (code differs from KB entry's source)"

    if name_match:
        return " [name-only match — no code stored, treat as related kernel]", "adapt to your kernel"
    return "", "adapt to your kernel"


def _build_top_hint(experiences: list[ExperienceRecord], target_kernel_path: str) -> str:
    """Emit a concise top-hit label and a single decision prompt.

    Design principle: present the data (verified speedup, code-identity tag,
    key params, full diff below); let the agent decide based on the diff
    + the current kernel.py + the profiling output whether the strategy
    is applicable. If yes, apply as first priority. If no, the agent is
    free to optimize from its own analysis without penalty.

    Identity is determined by CODE comparison, not name matching, since
    names like ``fused_qkv_rope`` and ``fused_qkv_MLA`` are organizational
    labels and the actual semantics live in the source.
    """
    if not experiences:
        return ""
    top = experiences[0]
    speedup = getattr(top, "best_speedup", 0.0) or 0.0
    if not top.success or speedup <= 1.03:
        return ""

    patch_text = getattr(top, "best_patch", "") or ""
    params = _extract_key_params(patch_text, max_params=3) if patch_text else []
    params_str = "; ".join(params) if params else "see diff below"

    match_tag, application_hint = _classify_kb_match(top, target_kernel_path)

    return (
        f"**KB top hit**: `{top.kernel_name}` verified **{speedup:.2f}x**"
        f"{match_tag}. Key params: {params_str}. Full diff below.\n\n"
        f"**Decide**: based on (a) the diff below, (b) your current kernel.py "
        f"code, and (c) the profiling output — does this strategy apply to "
        f"the current kernel? If YES, make it your first priority "
        f"({application_hint}). If NO, optimize from your own analysis "
        f"directly without trying it."
    )


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
        improved = [s for s in strategies if s.get("speedup", 0) > 1.0]
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
    'what worked' (speedup > 1.0) and 'what regressed' (speedup < 1.0)
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
            [s for s in strategies_list if s.get("speedup", 0) > 1.0],
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

    adaptation = _build_adaptation_note(kn, target_kernel_path)
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
            [s for s in strategies if s.get("speedup", 0) > 1.0],
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
