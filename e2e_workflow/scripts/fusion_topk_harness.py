#!/usr/bin/env python3
"""Phase 2.2: rank Fusion 2.1 candidates into a Top-K by benefit vs difficulty.

Deterministic post-process of the Phase 2.1 artifacts (fusion_candidates.json +
fusion_candidate_validation.json + the semantic table). It does NOT re-derive
facts; it consumes the harness-computed savings and adds:

  * a difficulty tier per candidate (A/B/C1/C2/C3) mapped to the 3.1 routing;
  * grouping of candidates that share one implementation into a "recipe"
    (build once, reuse across patterns) — benefit aggregates across patterns
    within a phase, effort is counted once;
  * workload-aware ranking: retain each phase's local benefit, then weight one
    prefill forward plus ``OSL - 1`` decode forwards for the requested workload;
  * benefit-per-effort scoring, with mutual-exclusion groups flagged.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict


# Difficulty tiers, ordered by build cost, mapped to the 3.1 routing.
TIER_WEIGHT = {"A": 1.0, "B": 3.0, "C1": 6.0, "C2": 10.0, "C3": 20.0}
TIER_ROUTE = {
    "A": "KernelFusion apply-back (flag/env)",
    "B": "KernelFusion apply-back (wire existing API)",
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


def _nonnegative_int(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _workload_model(workload, phase_total):
    """Build the request-level phase weights used by the merged Top-K.

    The prefill trace already represents the requested ISL shape, so ISL must
    not multiply its measured forward time again. The first output token is
    produced by prefill; the remaining output tokens each require one decode
    forward, hence ``decode_forwards = max(OSL - 1, 0)``. Concurrency is also
    already represented by the captured serving shape and is metadata here.

    Older standalone callers may omit workload. They keep the legacy phase-
    local ranking instead of silently inventing an OSL.
    """
    workload = workload if isinstance(workload, dict) else {}
    isl = _nonnegative_int(workload.get("isl"))
    osl = _nonnegative_int(workload.get("osl"))
    conc = _nonnegative_int(workload.get("conc"))
    available = isl is not None and osl is not None
    if not available:
        return {
            "available": False,
            "isl": isl,
            "osl": osl,
            "conc": conc,
            "prefill_forwards": None,
            "decode_forwards": None,
            "ranking_metric": "phase_forward_pct",
            "reason": "isl/osl not supplied; preserved phase-local ranking",
        }, {}, None

    phase_weights = {
        "prefill": 1,
        "decode": max(osl - 1, 0),
    }
    total = sum(
        float(phase_total.get(phase, 0.0) or 0.0) * weight
        for phase, weight in phase_weights.items())
    return {
        "available": True,
        "isl": isl,
        "osl": osl,
        "conc": conc,
        "prefill_forwards": phase_weights["prefill"],
        "decode_forwards": phase_weights["decode"],
        "ranking_metric": "workload_forward_pct",
    }, phase_weights, round(total, 3)


def _ranking_pct(row):
    workload_pct = row.get("workload_forward_pct")
    if workload_pct is not None:
        return workload_pct
    return row.get("forward_pct") or 0.0


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


def _tier(candidate, row_provider, catalog_match=None):
    base = IMPL_BASE_TIER.get(candidate.get("implementation_class"), "C")
    if base != "C":
        return base
    # The candidate declared author-track, but the deterministic catalog says a
    # real installed kernel covers its op-set — so it is an existing_api (B), not
    # something to author. Floor it at B so it lands on the A/B board instead of
    # being dropped to deferred_author on a false "no kernel" claim.
    if catalog_match and catalog_match.get("match"):
        return "B"
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
         tiers="A,B", workload=None):
    payload = _load(candidates_path)
    validation = _load(validation_path)
    table = _load(semantic_table_path)
    row_provider = _row_provider_index(table)

    metrics = validation.get("metrics", {})
    # per-candidate catalog match (from candidate_harness --catalog); floors a
    # falsely-author-track candidate's tier at B so it is not dropped.
    catalog_match = metrics.get("candidate_catalog_match", {}) or {}
    savings_by_id = {
        s["candidate_id"]: s for s in metrics.get("candidate_savings", [])}
    phase_total = metrics.get("phase_total_forward_us", {}) or {}
    workload_info, phase_weights, workload_total = _workload_model(
        workload, phase_total)
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
        tier = _tier(cand, row_provider, catalog_match.get(cid))
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
            "member_row_ids": ([
                member.get("row_id") for member in cand.get("members", [])
                if member.get("row_id")]
                or list(cand.get("removable_row_ids") or [])),
            "removable_row_ids": cand.get("removable_row_ids") or [],
            "estimate_us": sv.get("estimate_us"),
            "stack_estimate_us": sv.get("stack_estimate_us"),
            "ceiling_count": sv.get("ceiling_count"),
            "basis": sv.get("basis"),
            "live_call_seam": (cand.get("live_call_seam") or "").strip(),
            "api_name": apis[0].get("name") if apis else None,
            "actionable": _is_actionable(cand, sv, tier, guard_blocked_ids),
        }

    # Candidate-level degradation is discovered before recipe aggregation.
    # A narrow occurrence belongs below the smallest strict member-set
    # superset in the SAME phase/pattern.  Using member rows expresses the
    # actual fusion span; using removable rows made the relation depend on the
    # arbitrary "heaviest row is anchor" choice and mixed independent QK
    # norm+quant sites into the collective ladder.
    for cid, info in annotated.items():
        members = set(info.get("member_row_ids") or [])
        parents = []
        if members:
            for other_id, other in annotated.items():
                if other_id == cid:
                    continue
                if (other.get("phase"), other.get("pattern_id")) != (
                        info.get("phase"), info.get("pattern_id")):
                    continue
                other_members = set(other.get("member_row_ids") or [])
                if members < other_members:
                    parents.append((len(other_members),
                                    -(float(other.get("stack_estimate_us") or 0.0)),
                                    other_id))
        parent_id = min(parents)[2] if parents else None
        info["parent_candidate_id"] = parent_id
        info["parent_recipe_key"] = (
            annotated[parent_id]["recipe_key"] if parent_id else None)

    def _cohort_key(info):
        """Split one API recipe only where its degradation context differs.

        Occurrences of the same combination stay one unit-side test.  A
        norm+quant occurrence below AR+norm+quant and an unrelated QK
        norm+quant occurrence do not: only the former may be covered by the
        wider collective test.
        """
        parent = info.get("parent_recipe_key")
        return "%s :: %s" % (
            info["recipe_key"],
            "child-of=" + parent if parent else "standalone")

    # Group candidates into recipes; benefit aggregates across patterns within a
    # phase (additive), effort is counted once. Prefill and decode stay separate.
    recipes = {}
    for cid, info in annotated.items():
        key = _cohort_key(info)
        recipe = recipes.setdefault(key, {
            "recipe_key": key,
            "base_recipe_key": info["recipe_key"],
            "parent_recipe_key": info.get("parent_recipe_key"),
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
            workload_weight = phase_weights.get(phase)
            workload_savings = (
                round(benefit * workload_weight, 3)
                if workload_weight is not None else None)
            workload_pct = (
                round(workload_savings / workload_total * 100.0, 4)
                if workload_savings is not None and workload_total else None)
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
                "workload_weight": workload_weight,
                "workload_savings_us": workload_savings,
                "workload_forward_pct": workload_pct,
                "tier": phase_tier,
                "route": TIER_ROUTE.get(phase_tier, "?"),
                "score": round(
                    (workload_pct if workload_pct is not None else pct) / weight,
                    5) if (workload_pct is not None or pct is not None) else None,
                "coverage": ", ".join(dict.fromkeys(cover)),
                "handle": handle,
                "exact": "有" if any(
                    o.get("exact_kernel_status") == "yes" for o in occ)
                else "无",
                "impl_class": Counter(
                    o.get("implementation_class") for o in occ).most_common(
                        1)[0][0] if occ else None,
                # The concrete candidates this row stands for. Phase 3.0/3.1
                # account against these ids, so the row is an assignment rather
                # than a suggestion.
                "candidate_ids": sorted(o["candidate_id"] for o in occ),
            }
        ranked_recipes.append({
            "recipe_key": recipe["recipe_key"],
            "base_recipe_key": recipe["base_recipe_key"],
            "parent_recipe_key": recipe["parent_recipe_key"],
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
                "workload_weight": pp["workload_weight"],
                "workload_savings_us": pp["workload_savings_us"],
                "workload_forward_pct": pp["workload_forward_pct"],
                "exact": pp.get("exact", "无"),
                "recipe_key": recipe["recipe_key"],
                "removable_key": removable_key,
                "candidate_ids": pp.get("candidate_ids") or [],
                "mutually_exclusive_with": recipe["mutually_exclusive_with"],
            }
            if pp["tier"] in show_tiers:
                actions.append(row)
            else:
                deferred_author.append(row)
    actions.sort(key=lambda a: (
        tier_rank.get(a["tier"], 9), -_ranking_pct(a)))
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
            # A folded variant is the SAME fusion by another backend name --
            # its candidates still have to be accounted for downstream.
            seen[key]["candidate_ids"] = sorted(
                set(seen[key]["candidate_ids"]) | set(a["candidate_ids"]))
            continue
        a["variant_count"] = 1
        seen[key] = a
        collapsed.append(a)
    actions = collapsed
    truncated = []
    if top_k and len(actions) > top_k:
        # A silent [:K] makes the board read as "this is the whole surface".
        # It is not: on DSR1 2026-08-26 the fusion phase produced 42 candidates
        # and the reader saw a table with no denominator on it.
        truncated = actions[top_k:]
        actions = actions[:top_k]
    # ---- subsumption ladder ---------------------------------------------
    # A fusion whose removable-row set is a STRICT SUBSET of another surviving
    # row's set is a rung BELOW it on the same ladder: AR+norm sits inside
    # AR+norm+quant. Both rows stay on the board (the tradeoff is real and the
    # reader must see it), but they must not each pay for their own unit-side
    # microbench slot. On DSR1 2026-09-03 e01(AR+norm, 4 cands) and e02(
    # AR+norm+quant, 4 cands) burned 8 of the 10 unit-side slots proving
    # overlapping halves of one ladder, and e04 -- the oproj flatten-quant
    # fusion carrying a ★★★ prior with two measured e2e confirms -- never got
    # a slot at all. Unit-side, the superset's microbench exercises every row
    # the subset would have; the subset inherits its coverage. (This is sound
    # ONLY unit-side. At apply-back a superset that wins does NOT prove the
    # subset is worse -- 2026-09-03 measured the AR+norm+quant superset at
    # -0.45% while AR+norm delivered +1.65% -- so the integrator descends the
    # ladder rung by rung instead of pruning it.)
    #
    # Link to the MINIMAL strict superset so the ladder descends one rung at a
    # time; ties break deterministically on (set size, benefit, board order).
    by_index = list(enumerate(actions))
    for i, a in by_index:
        ka = a["removable_key"]
        if not ka:
            a["_subsumed_by_index"] = None
            continue
        supersets = [
            (len(b["removable_key"]), -_ranking_pct(b), j)
            for j, b in by_index
            if j != i and b["phase"] == a["phase"]
            and b["removable_key"] and ka < b["removable_key"]]
        a["_subsumed_by_index"] = min(supersets)[2] if supersets else None
    # A chain a < b < c must not let b claim coverage from c while a claims it
    # from b and nothing anchors the chain: resolve each row to the TOP of its
    # ladder for cost accounting, and keep the one-rung link for descent.
    for i, a in by_index:
        seen_chain = {i}
        top = a["_subsumed_by_index"]
        while top is not None and actions[top]["_subsumed_by_index"] is not None:
            if top in seen_chain:  # defensive: strict subset can't cycle
                break
            seen_chain.add(top)
            top = actions[top]["_subsumed_by_index"]
        a["_ladder_top_index"] = top

    for a in actions + truncated + deferred_author:  # drop unserializable key
        a.pop("removable_key", None)

    # ---- the binding execution list -------------------------------------
    # Top-K used to describe itself as an ADVISORY routing prior, and Phase 3
    # treated it that way: 42 candidates in, 16 verdicts out, no row anywhere
    # saying what happened to the other 26. The board now emits the DENOMINATOR
    # the later phases are accounted against.
    exec_list = []
    for index, a in enumerate(actions, 1):
        candidate_ids = a["candidate_ids"]
        # One implementation is benched once.  Use the occurrence with the
        # largest measured stack opportunity as the representative instead of
        # relying on candidate-id sort order; this exercises the shape/site
        # that matters most for the ranked benefit.
        representative = max(
            candidate_ids,
            key=lambda cid: (
                float(annotated[cid].get("stack_estimate_us") or 0.0),
                float(annotated[cid].get("estimate_us") or 0.0),
                cid),
        ) if candidate_ids else None
        exec_list.append({
            "exec_id": "e%02d" % index,
            "rank": index,
            "phase": a["phase"],
            "tier": a["tier"],
            "action": a["action"],
            "handle": a["handle"],
            "candidate_ids": candidate_ids,
            # One recipe/cohort is one concrete unit-side combination.  The
            # representative is measured once; sibling occurrences are kept in
            # the denominator and inherit the representative's auditable result.
            "unit_representative_candidate_id": representative,
            "unit_equivalent_candidate_ids": [
                cid for cid in candidate_ids if cid != representative],
            "forward_us": a["forward_us"],
            "forward_pct": a["forward_pct"],
            "workload_weight": a["workload_weight"],
            "workload_savings_us": a["workload_savings_us"],
            "workload_forward_pct": a["workload_forward_pct"],
            "exclusive_group": None,
            # Every entry must end in one of these downstream. "not mentioned"
            # is not an outcome.
            "required_disposition": [
                "applied", "blocked", "deferred_with_reason"],
        })
    exec_ids = [e["exec_id"] for e in exec_list]
    for entry, a in zip(exec_list, actions):
        sub_idx = a.pop("_subsumed_by_index", None)
        top_idx = a.pop("_ladder_top_index", None)
        entry["subsumed_by"] = exec_ids[sub_idx] if sub_idx is not None else None
        entry["ladder_top"] = exec_ids[top_idx] if top_idx is not None else None
        entry["subsumes"] = []
        # Unit-side budget is charged to LADDER TOPS only: a subsumed rung is
        # covered by its top's microbench and must be recorded downstream as
        # subsumed_pass (covered) or promoted to its own slot if the top FAILS
        # -- never silently dropped and never laundered as a waiver.
        entry["unit_cost"] = 0 if top_idx is not None else 1
    by_exec = {e["exec_id"]: e for e in exec_list}
    for entry in exec_list:
        if entry["subsumed_by"]:
            by_exec[entry["subsumed_by"]]["subsumes"].append(entry["exec_id"])

    # Mutual exclusion becomes an explicit DECIDE-ONE group instead of a ✳ in a
    # column. The reader still chooses; what changes is that not choosing is now
    # visibly an open decision rather than a silently dropped row.
    exec_by_recipe = defaultdict(list)
    for entry, a in zip(exec_list, actions):
        exec_by_recipe[(a["phase"], a["recipe_key"])].append(entry["exec_id"])

    # Exclusion is a pairwise CONFLICT GRAPH, not an equivalence class. Two
    # entries conflict when they consume the same row at the same position; that
    # relation is emphatically not transitive -- A can conflict with B and B
    # with C while A and C land together happily. Collapsing a component into
    # "choose exactly one" would quietly forbid a legal combination, which is
    # the same failure as dropping a candidate, just dressed as caution. So the
    # component is reported with its actual edges, and "choose 1" is claimed
    # only when every pair inside it really does conflict.
    conflicts = set()
    for entry, a in zip(exec_list, actions):
        for other_key in a.get("mutually_exclusive_with") or []:
            for other_id in exec_by_recipe.get((a["phase"], other_key), []):
                if other_id != entry["exec_id"]:
                    conflicts.add(
                        tuple(sorted((entry["exec_id"], other_id))))
    parent = {e["exec_id"]: e["exec_id"] for e in exec_list}

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for left, right in conflicts:
        ra, rb = _find(left), _find(right)
        if ra != rb:
            parent[rb] = ra
    groups = defaultdict(list)
    for entry in exec_list:
        groups[_find(entry["exec_id"])].append(entry)
    exclusive_groups = []
    for gindex, (_root, members) in enumerate(
            sorted((k, v) for k, v in groups.items() if len(v) > 1), 1):
        group_id = "x%02d" % gindex
        ids = [m["exec_id"] for m in members]
        edges = sorted(
            pair for pair in conflicts
            if pair[0] in set(ids) and pair[1] in set(ids))
        is_clique = len(edges) == len(ids) * (len(ids) - 1) // 2
        for entry in members:
            entry["exclusive_group"] = group_id
            entry["conflicts_with"] = sorted(
                other for pair in conflicts if entry["exec_id"] in pair
                for other in pair if other != entry["exec_id"])
        exclusive_groups.append({
            "group_id": group_id,
            "phase": members[0]["phase"],
            "choose": 1 if is_clique else "compatible_subset",
            "conflict_edges": [list(pair) for pair in edges],
            "reason": (
                "these fusions consume the same rows at the same position; "
                "exactly one can land there" if is_clique else
                "these fusions overlap PAIRWISE (see conflict_edges); pick any "
                "set with no conflicting pair -- not necessarily just one"),
            "member_exec_ids": ids,
            "members": [
                {"exec_id": m["exec_id"], "action": m["action"],
                 "tier": m["tier"], "forward_pct": m["forward_pct"],
                 "workload_forward_pct": m["workload_forward_pct"],
                 "conflicts_with": m.get("conflicts_with", []),
                 "candidate_ids": m["candidate_ids"]}
                for m in members],
        })

    # Catalog-backed family coverage: a family whose region a real installed
    # kernel covers must reach the A/B board (>=1 exec) or be an explicit author
    # deferral — never silently vanish because it was mis-tiered upstream. This
    # is the denominator the DSR1 miss lacked (mla/kv-write/V-absorb dropped to
    # deferred_author on a false "no kernel" and never surfaced anywhere).
    fam_of = {cid: cand.get("family") for cid, cand in candidates.items()}
    catalog_backed = {cid for cid, m in catalog_match.items() if m.get("match")}
    catalog_families = {fam_of.get(cid) for cid in catalog_backed}
    exec_families = {fam_of.get(cid) for a in actions for cid in a["candidate_ids"]}
    deferred_families = {
        fam_of.get(cid) for a in deferred_author for cid in a["candidate_ids"]}
    family_coverage = {
        "catalog_backed_families": sorted(f for f in catalog_families if f),
        "on_board_families": sorted(f for f in exec_families if f),
        "author_deferred_families": sorted(f for f in deferred_families if f),
        "uncovered_catalog_families": sorted(
            f for f in catalog_families
            if f and f not in exec_families and f not in deferred_families),
    }

    all_candidate_ids = sorted(candidates)
    listed_ids = sorted({
        cid for a in actions for cid in a["candidate_ids"]})
    result = {
        "schema_version": 3,
        "phase": "rank_topk",
        "top_k": top_k,
        # Coverage travels WITH the board. A Top-K built on a phase whose shapes
        # were never resolved, or on a table with uncovered fusible regions, is
        # a ranking of an incomplete surface -- the reader has to be able to see
        # that on the same page as the ranking.
        "phase_coverage": validation.get("phase_coverage"),
        "region_coverage": validation.get("region_coverage"),
        "candidate_total": len(all_candidate_ids),
        "candidates_on_board": len(listed_ids),
        "truncated_count": len(truncated),
        "truncated_actions": truncated,
        "execution_list": exec_list,
        "exclusive_groups": exclusive_groups,
        "family_coverage": family_coverage,
        "shown_tiers": sorted(show_tiers),
        "tier_weights": TIER_WEIGHT,
        "phase_total_forward_us": phase_total,
        "workload": workload_info,
        "workload_total_forward_us": workload_total,
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


def _render_gain(row):
    """Render the metric that actually determines merged-board order."""
    workload_pct = row.get("workload_forward_pct")
    workload_us = row.get("workload_savings_us")
    if workload_pct is not None and workload_us is not None:
        local = ("%.2f%%" % row["forward_pct"]
                 if row.get("forward_pct") is not None else "n/a")
        return "%.0f µs（%.2f%%；单次 %.0f µs / phase %s）" % (
            workload_us, workload_pct, row.get("forward_us") or 0.0, local)
    return ("%.0f µs（%.2f%%）" %
            (row.get("forward_us") or 0.0, row["forward_pct"])) \
        if row.get("forward_pct") is not None else "n/a"


def tier_rank_c(tier):
    return {"C1": 0, "C2": 1, "C3": 2}.get(tier, 3)


def _render_coverage(result):
    """The denominator and the evidence class, on the same page as the ranking.

    A Top-K with no denominator on it reads as "this is the whole surface".
    On DSR1 2026-08-26 it was 12 rows out of 42 candidates, on a table whose
    decode shapes had never been resolved, and nothing on the page said so.
    """
    lines = ["## 覆盖与证据（先看这里，再看排名）", ""]
    pc = result.get("phase_coverage") or {}
    rc = result.get("region_coverage") or {}
    if pc.get("available"):
        stats = pc.get("shape_resolution_by_phase") or {}
        for phase in sorted(stats):
            stat = stats[phase] or {}
            lines.append(
                "- **%s** shape 解析 %s/%s（%.0f%%）"
                % (phase, stat.get("resolved"), stat.get("rows"),
                   100.0 * float(stat.get("resolved_fraction") or 0.0)))
        if pc.get("decode_evidence"):
            lines.append("- decode 证据等级：`%s`%s" % (
                pc["decode_evidence"],
                "（需 graph-construction shape capture）"
                if pc.get("decode_requires_graph_capture") else ""))
        for problem in pc.get("problems") or []:
            lines.append("- ⚠️ %s" % problem)
        if pc.get("waiver"):
            lines.append("- ⚠️ 覆盖问题被显式豁免：%s" % pc["waiver"])
    else:
        lines.append(
            "- ⚠️ 上游未提供 `phase_coverage`：**无法确认 decode 是否被分析过**。")
    if rc:
        lines.append(
            "- 可融合区间（fusible region）：%s 个，已覆盖 %s / 已延后 %s / "
            "**未覆盖 %s**"
            % (rc.get("regions_total"), rc.get("covered"),
               rc.get("deferred"), rc.get("uncovered")))
    total = result.get("candidate_total")
    if total:
        lines.append(
            "- 候选 %s 条，进入本板 %s 条；因 `--top-k %s` 截断 %s 条"
            % (total, result.get("candidates_on_board"), result.get("top_k"),
               result.get("truncated_count", 0)))
    lines.append("")
    for a in result.get("truncated_actions") or []:
        lines.append("  - 截断：%s（%s，%.2f%%）" % (
            a["action"], a["tier"], _ranking_pct(a)))
    if result.get("truncated_actions"):
        lines.append("")
    return lines


def _render_execution_list(result):
    """The board as an assignment list, not a suggestion list.

    UnitSide and KernelFusion apply-back are accounted against these
    entries: every one must end `applied`, `blocked`, or `deferred` WITH a
    reason. Silence is not a disposition.
    """
    exec_list = result.get("execution_list") or []
    if not exec_list:
        return []
    lines = ["## 执行清单（Phase 3.0/3.1 按此逐条交代）", ""]
    lines.append(
        "下面每一条都必须在 3.0 单侧 / 3.1 apply-back 里有明确结论："
        "**已落地 / 被挡（原因）/ 延后（原因）**。没提到 = 覆盖漏洞，不是「跳过」。")
    lines.append("")
    lines.append("| exec | 阶段 | 难度 | 动作 | 候选 ID | 单侧代表 | 收益 | 互斥组 | 阶梯 |")
    lines.append("|:--|:--:|:--:|---|---|---|---:|:--:|:--|")
    for entry in exec_list:
        if entry.get("subsumed_by"):
            ladder = "⊂ `%s`（单侧由其覆盖）" % entry["subsumed_by"]
        elif entry.get("subsumes"):
            ladder = "⊃ %s（单侧代测）" % ", ".join(
                "`%s`" % c for c in entry["subsumes"])
        else:
            ladder = "-"
        representative = entry.get("unit_representative_candidate_id")
        equivalent_count = len(entry.get("unit_equivalent_candidate_ids") or [])
        unit_text = ("`%s`（+%d 等价实例）" %
                     (representative, equivalent_count)
                     if representative else "-")
        lines.append("| `%s` | %s | **%s** | %s | %s | %s | %s | %s | %s |" % (
            entry["exec_id"], entry["phase"].capitalize(), entry["tier"],
            _esc(entry["action"]),
            ", ".join("`%s`" % c for c in entry["candidate_ids"]) or "-",
            unit_text,
            ("%.2f%%" % _ranking_pct(entry))
            if (entry.get("workload_forward_pct") is not None or
                entry.get("forward_pct") is not None) else "n/a",
            entry.get("exclusive_group") or "-", ladder))
    lines.append("")
    if any(e.get("subsumed_by") for e in exec_list):
        lines.append(
            "> **阶梯（⊂/⊃）**：可去除行集合是严格子集的行 = 同一阶梯上更低的一级"
            "（AR+norm 内含于 AR+norm+quant）。**单侧**只给阶梯顶端行扣预算，"
            "顶端 microbench 已经跑过子集的每一行，子集记 `subsumed_pass`；"
            "顶端单侧**失败**时子集必须补一个自己的单侧名额，不得当作已覆盖。")
        lines.append(
            "> **apply-back 不做这种剪枝**：超集赢不代表子集更差"
            "（2026-09-03 实测 AR+norm+quant −0.45%，而 AR+norm +1.65%）。"
            "3.1 从阶梯顶端往下逐级测，失败/落在噪声带内就降一级；"
            "带 ★★★ 先验且有实测 e2e 的子集一律单独跑一条边际 A/B。")
        lines.append("")
    for group in result.get("exclusive_groups") or []:
        if group.get("choose") == 1:
            lines.append(
                "> **互斥组 `%s`（%s）：以下 %d 条只能落一条，必须显式择一**"
                % (group["group_id"], group["phase"], len(group["members"])))
        else:
            lines.append(
                "> **重叠组 `%s`（%s）：以下 %d 条两两部分冲突，选一组互不冲突的**"
                % (group["group_id"], group["phase"], len(group["members"])))
        for member in group["members"]:
            conflict = member.get("conflicts_with") or []
            lines.append("> - `%s` %s（%s，%s）%s" % (
                member["exec_id"], _esc(member["action"]), member["tier"],
                ("%.2f%%" % _ranking_pct(member))
                if (member.get("workload_forward_pct") is not None or
                    member.get("forward_pct") is not None) else "n/a",
                ("　与 %s 冲突" % ", ".join("`%s`" % c for c in conflict))
                if conflict else ""))
        lines.append(">")
        lines.append("> 未做选择 = 未决事项，不等于这批融合不存在。")
        lines.append("")
    return lines


def render_markdown(result, actions):
    lines = ["# Kernel Fusion Top-K (Phase 2.2)", ""]
    fwd = result["phase_total_forward_us"]
    workload = result.get("workload") or {}
    lines.append(
        "一张合并 Top-K（prefill/decode 用「阶段」列区分）。按**实现难度 A→B** 排序"
        "（先摘低垂果实），同实现难度内按真实 workload 的累计 forward 收益占比排。"
        "C（需自写 kernel）见文末"
        "同格式表，暂缓。**收益均为 roofline 工程估算，落地前以 benchmark 确认。**")
    lines.append("")
    lines.append("实现难度 = 落地工作量：**A** = 配置开启（翻 flag / 确认已启用，零代码）；"
                 "**B** = 接入现成算子（改代码把已存在的 fused kernel 接进来）；"
                 "**C** = 没有现成算子，需自写 kernel。")
    lines.append("**现成算子** = 有没有可用的现成 fused kernel：**有** = A/B（kernel 已存在）；"
                 "**无** = C（需 author）。它只表示算子在不在，不表示接起来轻重。")
    lines.append("单次 forward：prefill ≈ %.0f µs / decode ≈ %.0f µs。"
                 % (fwd.get("prefill", 0.0), fwd.get("decode", 0.0)))
    if workload.get("available"):
        lines.append(
            "真实 workload：ISL=%s / OSL=%s / conc=%s；prefill forward × %s，"
            "decode forward × %s（OSL−1），累计 forward 分母 ≈ %.0f µs。"
            "ISL/conc 已体现在采集到的 phase shape 中，不重复相乘。"
            % (workload.get("isl"), workload.get("osl"),
               workload.get("conc"), workload.get("prefill_forwards"),
               workload.get("decode_forwards"),
               result.get("workload_total_forward_us") or 0.0))
    else:
        lines.append(
            "⚠️ 未提供完整 ISL/OSL，无法做 request-level 加权；本次兼容旧行为，"
            "同难度内按各 phase 自身的 forward 占比排序。")
    lines.append("互斥（✳）的融合方案（同批算子、每处只落一个）都列出、标注供你/3.2 选，不替你择优。")
    lines.append("")
    lines.append(
        "**本板是执行清单，不是建议清单。** 文末「执行清单」里的每一条，"
        "3.0 单侧 / 3.1 apply-back 都必须给出明确处置（已落地 / 被挡 / 延后+原因）；"
        "互斥组内选哪一条仍然由你/3.2 决定，但**不做选择本身是一个未决事项**，"
        "不能当成这批融合不存在。")
    lines.append("")
    lines.extend(_render_coverage(result))
    _HEADER = (
        "| 排名 | 实现难度 | 阶段 | 优先行动（集成什么） | 覆盖范围 | "
        "对应 Kernel / API（怎么开）| 预期 workload forward 收益 | 现成算子 | 互斥 |")
    _SEP = "|---:|:--:|:--:|---|---|---|---:|:--:|:--:|"

    def _render_rows(rows):
        for i, a in enumerate(rows, 1):
            handle = a["handle"] or "-"
            if a.get("variant_count", 1) > 1:
                handle += "（%d 个等价 kernel 变体）" % a["variant_count"]
            mex = "✳" if a.get("mutually_exclusive_with") else "-"
            lines.append(
                "| %d | **%s** | %s | %s | %s | `%s` | %s | %s | %s |"
                % (i, a["tier"], a["phase"].capitalize(),
                   _esc(a["action"]), _esc(a["coverage"] or "-"),
                   _esc(handle), _render_gain(a), a["exact"], mex))

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
            key=lambda x: (tier_rank_c(x["tier"]), -_ranking_pct(x))))
        lines.append("")
    lines.extend(_render_execution_list(result))
    lines.append(
        "说明：收益为 roofline 工程估算（融合消除的访存往返+launch），"
        "非实测、以 benchmark 为准；只计当前可落地(actionable)的层，guard 挡住/需自写的不计入。"
        "每条候选证据与实现难点见同目录 `FUSION_CANDIDATES.md` / `FUSION_TOPK_NOTES.md`。")
    lines.append("")
    return "\n".join(lines) + "\n"


def run(candidates_path, validation_path, semantic_table_path,
        out_md, out_json, top_k, tiers="A,B", workload=None):
    result, actions, _ = rank(
        candidates_path, validation_path, semantic_table_path, top_k, tiers,
        workload)
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
    parser.add_argument("--isl", type=int)
    parser.add_argument("--osl", type=int)
    parser.add_argument("--conc", type=int)
    parser.add_argument(
        "--tiers", default="A,B",
        help="difficulty tiers to include in the ranked table (default A,B; "
             "C author-track is summarized separately)")
    args = parser.parse_args()
    workload = {"isl": args.isl, "osl": args.osl, "conc": args.conc}
    result = run(
        args.candidates, args.validation, args.semantic_table,
        args.out_md, args.out_json, args.top_k, args.tiers, workload)
    print(json.dumps({
        "recipe_count": result["recipe_count"],
        "topk_actions": len(result["topk_actions"]),
        "deferred_author_count": result["deferred_author_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
