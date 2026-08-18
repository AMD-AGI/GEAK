#!/usr/bin/env python3
"""Phase 2.2: rank Fusion 2.1 candidates into a Top-K by benefit vs difficulty.

Deterministic post-process of the Phase 2.1 artifacts (fusion_candidates.json +
fusion_candidate_validation.json + the semantic table). It does NOT re-derive
facts; it consumes the harness-computed savings and adds:

  * a difficulty tier per candidate (A/B/C1/C2/C3) mapped to the 3.1 routing;
  * grouping of candidates that share one implementation into a "recipe"
    (build once, reuse across patterns) — benefit aggregates across patterns
    within a phase, effort is counted once;
  * per-phase ranking (prefill and decode are different forwards; never summed),
    scored by benefit-per-effort, with mutual-exclusion groups flagged.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict


# Difficulty tiers, ordered by build cost, mapped to the 3.1 routing.
TIER_WEIGHT = {"A": 1.0, "B": 3.0, "C1": 6.0, "C2": 10.0, "C3": 20.0}
TIER_ROUTE = {
    "A": "ConfigSweep (flag/env)",
    "B": "HeadKernel direct_light/code_patch (wire existing API)",
    "C1": "kernel_workflow author — single helper, same language",
    "C2": "kernel_workflow author — single helper, cross language",
    "C3": "kernel_workflow author — main-body / algorithmic rewrite",
}
# implementation_class -> base tier. new_helper_kernel is refined to C1/C2 by the
# region language; main_kernel_or_algorithmic is always C3.
IMPL_BASE_TIER = {
    "existing_flag_or_env": "A",
    "existing_api_integrated": "B",  # integrating an existing kernel = code = B
    "existing_api_needs_adapter": "B",
    "reference_path_port": "B",
    "new_helper_kernel": "C",
    "main_kernel_or_algorithmic": "C3",
}
PROVIDER_LANG = {
    "aiter": "hip", "ck": "hip", "hipblaslt": "hip",
    "triton": "triton", "torch_native": "torch", "unknown": "unknown"}
# GEAK authors most fluently in Triton; a fusion whose region is already Triton
# is a same-language helper (C1), otherwise it is cross-language (C2).
NATIVE_AUTHOR_LANG = "triton"
BLOCKED_READINESS = {"blocked_shape", "blocked_evidence"}


# Readable labels for the action table.
FAMILY_LABEL = {
    "collective_norm": "AllReduce + RMSNorm",
    "collective_norm_quant": "AllReduce + RMSNorm + Quant",
    "norm_quant": "RMSNorm + Quant",
    "activation_quant": "激活(SiLU) + Quant",
    "quant_gemm_prologue": "Quant + GEMM prologue",
    "gemm_epilogue_layout": "GEMM epilogue + layout/cast",
    "layout_norm": "layout-copy + RMSNorm",
    "mla_head_prep": "MLA head-prep (RoPE/kv-cache)",
    "attn_prologue_layout": "Attention prologue layout",
    "moe_router_helpers": "MoE router/topk/sort",
    "router_topk": "MoE router + topk",
}
# Three tiers by realization cost (the authoritative definition):
#   A = env var / flag only (no code)      -> existing_flag_or_env  (现成算子=有)
#   B = integrate an existing kernel (code) -> existing_api_* / port (现成算子=有)
#   C = author a new kernel                 -> new_helper_kernel / main_algo
#                                                                   (现成算子=无)
# 现成算子(exact) is binary: 有 for A/B (a ready kernel exists), 无 for C. It no
# longer sub-shades B into 接入 vs 接入+适配 — either the kernel exists or it does
# not.
def _action_verb(impl_class):
    if impl_class == "existing_flag_or_env":
        return "开启"          # flip a flag/env (verify it engages in trace)
    if impl_class in ("existing_api_integrated", "existing_api_needs_adapter",
                      "reference_path_port"):
        return "接入"          # wire in the existing kernel
    return "实现"              # author a new kernel (C)


def _short_pattern(pattern_id):
    text = str(pattern_id or "").replace("P_MLA_", "").replace("P_", "")
    text = text.replace("_ATTENTION", "").replace("ATTENTION", "")
    return text.title().replace("_", " ") or "?"


def _family_label(family):
    return FAMILY_LABEL.get(family, str(family or "?").replace("_", " "))


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def _row_provider_index(table):
    index = {}
    for item in table.get("tables", []):
        for row in item.get("rows", []):
            index[row.get("row_id")] = row.get("provider")
    return index


def _region_lang(candidate, row_provider):
    rows = candidate.get("removable_row_ids") or [
        m.get("row_id") for m in candidate.get("members", [])]
    langs = [PROVIDER_LANG.get(row_provider.get(rid), "unknown")
             for rid in rows if rid in row_provider]
    langs = [lang for lang in langs if lang and lang != "unknown"]
    if not langs:
        return "unknown"
    return Counter(langs).most_common(1)[0][0]


def _tier(candidate, row_provider):
    base = IMPL_BASE_TIER.get(candidate.get("implementation_class"), "C")
    if base != "C":
        return base
    lang = _region_lang(candidate, row_provider)
    return "C1" if lang == NATIVE_AUTHOR_LANG else "C2"


_TIER_GROUP = {
    "existing_flag_or_env": "cfg", "existing_api_integrated": "api",
    "existing_api_needs_adapter": "api", "reference_path_port": "api",
    "new_helper_kernel": "author", "main_kernel_or_algorithmic": "author"}


def _recipe_key(candidate):
    """Candidates that ONE implementation would satisfy share this key.

    Keyed by family + API + implementation tier-group so a code-patch (B) and an
    author-track (C) candidate that merely cite the same API are NOT merged into
    one recipe (which would conflate their benefit and blur the tier/verb).
    """
    family = candidate.get("family", "?")
    apis = candidate.get("existing_apis") or []
    api = apis[0].get("name") if apis else None
    group = _TIER_GROUP.get(candidate.get("implementation_class"), "author")
    return "%s :: %s :: %s" % (family, api or ("new:" + family), group)


def _is_actionable(candidate, savings, tier, guard_blocked_ids):
    """Can this fusion be applied at the traced shape (this stage)?

    Actionable = there is savings, the readiness is not blocked, and a size guard
    does not block the fused path at this shape. 现成算子(exact) does NOT gate
    actionability anymore — A/B always have a kernel, and whether a fused path
    engages at this shape is a size-guard fact, not an exact fact. The one dead
    case is a collective whose fused kernel a size guard blocks entirely at this
    shape (recorded in collective_guard_checks: verdict=exceeds).
    C (author-track) is actionable-by-authoring; it is separated from A/B by the
    tier filter, not here.
    """
    if savings.get("estimate_us") is None:
        return False
    if candidate.get("readiness") in BLOCKED_READINESS:
        return False
    if candidate.get("candidate_id") in guard_blocked_ids:
        return False  # fused path blocked by a size guard at this shape
    return True


def rank(candidates_path, validation_path, semantic_table_path, top_k,
         tiers="A,B"):
    payload = _load(candidates_path)
    validation = _load(validation_path)
    table = _load(semantic_table_path)
    row_provider = _row_provider_index(table)

    metrics = validation.get("metrics", {})
    savings_by_id = {
        s["candidate_id"]: s for s in metrics.get("candidate_savings", [])}
    phase_total = metrics.get("phase_total_forward_us", {}) or {}
    # Collective candidates whose fused path a size guard blocks at this shape
    # (harness-computed, deterministic). These are 现成算子=有 but not applicable
    # here, so they drop off the actionable board.
    guard_blocked_ids = {
        c["candidate_id"] for c in metrics.get("collective_guard_checks", [])
        if c.get("verdict") == "exceeds"}

    candidates = {c["candidate_id"]: c for c in payload.get("candidates", [])}

    # Annotate each candidate with its tier + language + savings.
    annotated = {}
    for cid, cand in candidates.items():
        sv = savings_by_id.get(cid, {})
        tier = _tier(cand, row_provider)
        apis = cand.get("existing_apis") or []
        annotated[cid] = {
            "candidate_id": cid,
            "phase": cand.get("phase"),
            "pattern_id": cand.get("pattern_id"),
            "family": cand.get("family"),
            "tier": tier,
            "region_lang": _region_lang(cand, row_provider),
            "recipe_key": _recipe_key(cand),
            "readiness": cand.get("readiness"),
            "exact_kernel_status": cand.get("exact_kernel_status"),
            "implementation_class": cand.get("implementation_class"),
            "removable_row_ids": cand.get("removable_row_ids") or [],
            "estimate_us": sv.get("estimate_us"),
            "stack_estimate_us": sv.get("stack_estimate_us"),
            "ceiling_count": sv.get("ceiling_count"),
            "basis": sv.get("basis"),
            "live_call_seam": (cand.get("live_call_seam") or "").strip(),
            "api_name": apis[0].get("name") if apis else None,
            "actionable": _is_actionable(cand, sv, tier, guard_blocked_ids),
        }

    # Group candidates into recipes; benefit aggregates across patterns within a
    # phase (additive), effort is counted once. Prefill and decode stay separate.
    recipes = {}
    for cid, info in annotated.items():
        key = info["recipe_key"]
        recipe = recipes.setdefault(key, {
            "recipe_key": key,
            "family": info["family"],
            "tiers": Counter(),
            "occurrences": [],
            "phase_benefit_us": defaultdict(float),
            "phase_full_us": defaultdict(float),
            "phase_tiers": defaultdict(Counter),
            "phase_actionable_occ": defaultdict(list),
            "phases": set(),
            "patterns": set(),
        })
        recipe["tiers"][info["tier"]] += 1
        recipe["occurrences"].append(info)
        recipe["phases"].add(info["phase"])
        recipe["patterns"].add((info["phase"], info["pattern_id"]))
        stack = float(info.get("stack_estimate_us") or 0.0)
        recipe["phase_full_us"][info["phase"]] += stack
        if info["actionable"]:
            recipe["phase_benefit_us"][info["phase"]] += stack
            recipe["phase_tiers"][info["phase"]][info["tier"]] += 1
            recipe["phase_actionable_occ"][info["phase"]].append(info)

    # Finalize. Tier and score are computed PER PHASE from the occurrences that
    # actually deliver that phase's benefit — not a global max over all phases.
    # A recipe can be A/ConfigSweep in decode (a flag engages it) yet blocked in
    # prefill (guard); the decode board must score it as A, not be dragged to B
    # by the prefill occurrence. Per phase we take the cheapest (min-effort) tier
    # among that phase's actionable occurrences.
    tier_rank = {"A": 0, "B": 1, "C1": 2, "C2": 3, "C3": 4}
    ranked_recipes = []
    for recipe in recipes.values():
        global_tier = max(recipe["tiers"], key=lambda t: tier_rank.get(t, 9))
        per_phase = {}
        for phase in sorted(recipe["phases"]):
            benefit = round(recipe["phase_benefit_us"][phase], 3)
            forward = float(phase_total.get(phase, 0.0) or 0.0)
            pct = round(benefit / forward * 100.0, 4) if forward else None
            phase_tiers = recipe["phase_tiers"].get(phase)
            if phase_tiers:
                phase_tier = min(phase_tiers, key=lambda t: tier_rank.get(t, 9))
            else:
                phase_tier = global_tier  # no actionable occ this phase
            weight = TIER_WEIGHT.get(phase_tier, 10.0)
            occ = recipe["phase_actionable_occ"].get(phase, [])
            # coverage per phase: "pattern×layers" for the actionable occurrences
            cover = []
            for o in sorted(occ, key=lambda x: x.get("pattern_id") or ""):
                cover.append("%s×%s" % (
                    _short_pattern(o.get("pattern_id")),
                    o.get("ceiling_count") or "?"))
            # activation handle: flag for A, API for B
            seams = [o.get("live_call_seam") for o in occ
                     if o.get("live_call_seam")]
            apis = [o.get("api_name") for o in occ if o.get("api_name")]
            handle = (seams[0] if phase_tier == "A" and seams
                      else (apis[0] if apis else (seams[0] if seams else "")))
            per_phase[phase] = {
                "actionable_us": benefit,
                "full_us": round(recipe["phase_full_us"][phase], 3),
                "forward_pct": pct,
                "tier": phase_tier,
                "route": TIER_ROUTE.get(phase_tier, "?"),
                "score": round(pct / weight, 5) if pct is not None else None,
                "coverage": ", ".join(dict.fromkeys(cover)),
                "handle": handle,
                "exact": "有" if any(
                    o.get("exact_kernel_status") == "yes" for o in occ)
                else "无",
                "impl_class": Counter(
                    o.get("implementation_class") for o in occ).most_common(
                        1)[0][0] if occ else None,
            }
        ranked_recipes.append({
            "recipe_key": recipe["recipe_key"],
            "family": recipe["family"],
            "tier": global_tier,
            "route": TIER_ROUTE.get(global_tier, "?"),
            "occurrence_count": len(recipe["occurrences"]),
            "patterns": sorted("%s/%s" % (p, q) for p, q in recipe["patterns"]),
            "per_phase": per_phase,
            "occurrences": recipe["occurrences"],
        })

    # Per-phase mutual exclusion: recipes whose occurrences in the same
    # (phase, pattern) share a removable row cannot both be applied there.
    exclusion = defaultdict(set)
    by_cell_row = defaultdict(lambda: defaultdict(set))
    for recipe in ranked_recipes:
        for occ in recipe["occurrences"]:
            cell = (occ["phase"], occ["pattern_id"])
            for rid in occ["removable_row_ids"]:
                by_cell_row[cell][rid].add(recipe["recipe_key"])
    for cell, rows in by_cell_row.items():
        for keys in rows.values():
            if len(keys) > 1:
                for k in keys:
                    exclusion[k] |= (keys - {k})
    for recipe in ranked_recipes:
        recipe["mutually_exclusive_with"] = sorted(
            exclusion.get(recipe["recipe_key"], set()))

    # ONE merged Top-K action list. Each row is an actionable (recipe, phase):
    # the concrete thing to integrate, with a 阶段 column. Ordered by difficulty
    # (A before B before C) then by whole-forward benefit within a tier. This
    # stage only surfaces A and B (C author-track is summarized, not ranked).
    tier_rank = {"A": 0, "B": 1, "C1": 2, "C2": 3, "C3": 4}
    show_tiers = {t.strip() for t in (tiers or "A,B").split(",") if t.strip()}
    actions = []
    deferred_author = []
    for recipe in ranked_recipes:
        for phase, pp in recipe["per_phase"].items():
            if (pp.get("actionable_us") or 0.0) <= 0:
                continue
            # The removable-row set IS the fusion identity: two candidates that
            # remove the same rows are the same fusion (only the backend kernel
            # variant differs); different removable rows = a different fusion
            # (e.g. AR+norm vs AR+norm+quant) and must stay a separate row.
            removable_key = frozenset(
                rid for occ in recipe["occurrences"]
                if occ.get("phase") == phase
                for rid in (occ.get("removable_row_ids") or []))
            row = {
                "phase": phase,
                "tier": pp["tier"],
                "route": pp["route"],
                "action": "%s %s %s 融合" % (
                    phase.capitalize(),
                    _action_verb(pp.get("impl_class")),
                    _family_label(recipe["family"])),
                "coverage": pp.get("coverage", ""),
                "handle": pp.get("handle", ""),
                "forward_us": pp["actionable_us"],
                "forward_pct": pp["forward_pct"],
                "exact": pp.get("exact", "无"),
                "recipe_key": recipe["recipe_key"],
                "removable_key": removable_key,
                "mutually_exclusive_with": recipe["mutually_exclusive_with"],
            }
            if pp["tier"] in show_tiers:
                actions.append(row)
            else:
                deferred_author.append(row)
    actions.sort(key=lambda a: (
        tier_rank.get(a["tier"], 9), -(a["forward_pct"] or 0.0)))
    # Fold ONLY true duplicates: same phase + same removable-row set = the same
    # fusion realized by different backend kernel variants → one row. Different
    # removable sets (AR+norm vs AR+norm+quant) or different tiers stay SEPARATE
    # rows even if mutually exclusive — a partial cheap option (A) and a fuller
    # costlier option (B) are a tradeoff the reader/3.2 should see, not one we
    # silently drop. Mutual exclusion is annotated, never collapsed away.
    collapsed = []
    seen = {}
    for a in actions:
        # fold only true dupes: same phase + same removable set + same tier
        # (same fusion, different backend kernel name). Different tier stays
        # separate so an A(partial) and B(fuller) option are both visible.
        key = (a["phase"], a["removable_key"], a["tier"])
        if a["removable_key"] and key in seen:
            seen[key]["variant_count"] += 1
            continue
        a["variant_count"] = 1
        seen[key] = a
        collapsed.append(a)
    actions = collapsed
    if top_k:
        actions = actions[:top_k]
    for a in actions + deferred_author:  # drop internal unserializable key
        a.pop("removable_key", None)

    result = {
        "schema_version": 2,
        "phase": "rank_topk",
        "top_k": top_k,
        "shown_tiers": sorted(show_tiers),
        "tier_weights": TIER_WEIGHT,
        "phase_total_forward_us": phase_total,
        "recipe_count": len(ranked_recipes),
        "topk_actions": actions,
        "deferred_author_count": len(deferred_author),
        "deferred_author": deferred_author,
        "recipes": [
            {k: v for k, v in r.items() if k != "occurrences"}
            for r in ranked_recipes],
    }
    return result, actions, ranked_recipes


def _esc(value):
    return str(value or "").replace("|", "\\|")


def tier_rank_c(tier):
    return {"C1": 0, "C2": 1, "C3": 2}.get(tier, 3)


def render_markdown(result, actions):
    lines = ["# Kernel Fusion Top-K (Phase 2.2)", ""]
    fwd = result["phase_total_forward_us"]
    lines.append(
        "一张合并 Top-K（prefill/decode 用「阶段」列区分）。按**实现难度 A→B** 排序"
        "（先摘低垂果实），同实现难度内按整-forward 收益占比排。C（需自写 kernel）见文末"
        "同格式表，暂缓。**收益均为 roofline 工程估算，落地前以 benchmark 确认。**")
    lines.append("")
    lines.append("实现难度 = 落地工作量：**A** = 配置开启（翻 flag / 确认已启用，零代码）；"
                 "**B** = 接入现成算子（改代码把已存在的 fused kernel 接进来）；"
                 "**C** = 没有现成算子，需自写 kernel。")
    lines.append("**现成算子** = 有没有可用的现成 fused kernel：**有** = A/B（kernel 已存在）；"
                 "**无** = C（需 author）。它只表示算子在不在，不表示接起来轻重。")
    lines.append("整-forward 占比分母：prefill ≈ %.0f µs / decode ≈ %.0f µs（各自 forward，不混算）。"
                 % (fwd.get("prefill", 0.0), fwd.get("decode", 0.0)))
    lines.append("互斥（✳）的融合方案（同批算子、每处只落一个）都列出、标注供你/3.2 选，不替你择优。")
    lines.append("")
    _HEADER = (
        "| 排名 | 实现难度 | 阶段 | 优先行动（集成什么） | 覆盖范围 | "
        "对应 Kernel / API（怎么开）| 预期整-forward 收益 | 现成算子 | 互斥 |")
    _SEP = "|---:|:--:|:--:|---|---|---|---:|:--:|:--:|"

    def _render_rows(rows):
        for i, a in enumerate(rows, 1):
            pct = ("%.2f%%" % a["forward_pct"]) if a[
                "forward_pct"] is not None else "n/a"
            handle = a["handle"] or "-"
            if a.get("variant_count", 1) > 1:
                handle += "（%d 个等价 kernel 变体）" % a["variant_count"]
            mex = "✳" if a.get("mutually_exclusive_with") else "-"
            lines.append(
                "| %d | **%s** | %s | %s | %s | `%s` | %.0f µs（%s）| %s | %s |"
                % (i, a["tier"], a["phase"].capitalize(),
                   _esc(a["action"]), _esc(a["coverage"] or "-"),
                   _esc(handle), a["forward_us"], pct, a["exact"], mex))

    lines.append(_HEADER)
    lines.append(_SEP)
    _render_rows(actions)
    lines.append("")
    if any(a.get("mutually_exclusive_with") for a in actions):
        lines.append(
            "> 互斥（✳）：同一位置的这些融合共享同批算子，**每处只落一个**——"
            "它们是「便宜但部分（如 AR+Norm，A）」vs「更完整但需集成（如 AR+Norm+Quant，B）」"
            "的取舍，都列出供你/3.2 选，不替你择优。")
        lines.append("")
    if result.get("deferred_author_count"):
        # C 类：与 A/B 完全相同的列，只是现成算子=无、暂缓。
        lines.append(
            "## C 类（无现成算子，需自写 kernel，本阶段暂缓，共 %d 项）"
            % result["deferred_author_count"])
        lines.append("")
        lines.append(_HEADER)
        lines.append(_SEP)
        _render_rows(sorted(
            result["deferred_author"],
            key=lambda x: (tier_rank_c(x["tier"]), -(x["forward_pct"] or 0.0))))
        lines.append("")
    lines.append(
        "说明：收益为 roofline 工程估算（融合消除的访存往返+launch），"
        "非实测、以 benchmark 为准；只计当前可落地(actionable)的层，guard 挡住/需自写的不计入。"
        "每条候选证据与实现难点见同目录 `FUSION_CANDIDATES.md` / `FUSION_TOPK_NOTES.md`。")
    lines.append("")
    return "\n".join(lines) + "\n"


def run(candidates_path, validation_path, semantic_table_path,
        out_md, out_json, top_k, tiers="A,B"):
    result, actions, _ = rank(
        candidates_path, validation_path, semantic_table_path, top_k, tiers)
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    with open(out_md, "w") as fh:
        fh.write(render_markdown(result, actions))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--semantic-table", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument(
        "--tiers", default="A,B",
        help="difficulty tiers to include in the ranked table (default A,B; "
             "C author-track is summarized separately)")
    args = parser.parse_args()
    result = run(
        args.candidates, args.validation, args.semantic_table,
        args.out_md, args.out_json, args.top_k, args.tiers)
    print(json.dumps({
        "recipe_count": result["recipe_count"],
        "topk_actions": len(result["topk_actions"]),
        "deferred_author_count": result["deferred_author_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
