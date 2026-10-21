#!/usr/bin/env python3
"""Select kernels whose whole-workload GPU-time share exceeds a threshold.

The share is NOT a capture window's raw device time. It is
`sum over phases of phase_share[phase] * share_of_phase_pct[kernel][phase]`,
where `phase_share` is the PREFILL/DECODE split estimated from the serving log
and the measured sampling points. A single short capture can establish what a
phase executes; it cannot establish how much of the workload that phase is.

This computes scope only. It neither proves editability nor launches GPU work.
Input is the kernel-extract ranking artifact, on one common device-time basis.
"""

import argparse
import json
import math
from pathlib import Path

# A phase measured at one point has no spread to fit a cost model against, so its
# mass is whatever that one point happened to execute. Two is the minimum that can
# disagree; plan_sampling_points.py defaults to far more than two.
MIN_POINTS_PER_PHASE = 2
WINDOW_BASES = {"single_window", "capture_window", "trace_window"}

# A fusion group may be summed only when its members are DISJOINT device-kernel rows.
# Launch adjacency establishes that: two kernels that ran one after the other occupied
# different device time. Name similarity does not -- a shared callable name is exactly
# how a parent op and its own child kernel look, and summing those double-counts.
ADJACENCY_EVIDENCE_KINDS = {
    "observed_consecutive_launches",   # measured launch order on one stream
    "observed_dispatch_chain",         # one seam dispatching the stages in order
}


def _percent(value, field):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 100):
        raise ValueError(f"{field} must be a finite number in [0, 100]")
    return value


def _phase_share(document, phases, min_points, allow_window_denominator):
    """Validate the log-weighted phase split and return it as fractions summing to 1.

    Refusing here rather than downstream is deliberate: a missing split used to fall
    back to whatever `selection_pct_gpu_time` the ranking already carried, which on a
    5-second window made one long prefill batch outrank the whole decode phase.
    """
    share = document.get("phase_share")
    if not isinstance(share, dict) or not share:
        raise ValueError(
            "phase_share is missing. Run observed_workload.py -> plan_sampling_points.py "
            "-> serving_probe.py (once per point) -> coverage_ledger.py and record the "
            "log-weighted PREFILL/DECODE split, or pass --allow-window-denominator to "
            "declare an explicitly non-comparable window-only selection.")

    basis = share.get("basis")
    if not isinstance(basis, str) or not basis.strip():
        raise ValueError("phase_share.basis must name how the split was established")
    if basis in WINDOW_BASES and not allow_window_denominator:
        raise ValueError(
            f"phase_share.basis={basis!r} is a single capture window. Phase mass must come "
            "from the serving log's instance counts weighted by measured per-point cost. "
            "Pass --allow-window-denominator only to record a declared-approximate scope.")

    if not allow_window_denominator:
        # Counting a point that serving_probe.py could not verify is how a split gets
        # built from captures of the wrong phase: four DECODE points whose traces held
        # no decode step span at all still produced a decode mass, and the resulting
        # 49/51 split outranked a workload that is really ~12/88.
        points = share.get("points")
        if not isinstance(points, list) or not points:
            raise ValueError(
                "phase_share.points must list every sampling point with its "
                "serving_probe.py verdict and whether it was counted")
        counted = {p: 0 for p in phases}
        seen = set()
        for i, rec in enumerate(points):
            if not isinstance(rec, dict):
                raise ValueError(f"phase_share.points[{i}] must be an object")
            pid = rec.get("point_id")
            if not isinstance(pid, str) or not pid.strip() or pid in seen:
                raise ValueError("phase_share.points need unique, nonempty point_id values")
            seen.add(pid)
            verified, is_counted = rec.get("verified"), bool(rec.get("counted"))
            if verified not in (True, False, "partial"):
                raise ValueError(
                    f"phase_share.points[{pid}].verified must be true, false or "
                    '"partial" — serving_probe.py\'s own verdict, not a summary')
            if not is_counted:
                continue
            if verified is False:
                raise ValueError(
                    f"phase_share.points[{pid}] is counted but serving_probe.py could not "
                    "verify it. An unverified capture may hold another phase entirely; "
                    "exclude it and say so, or recapture the point.")
            if verified == "partial" and not rec.get("relabelled_to_achieved_shape"):
                raise ValueError(
                    f"phase_share.points[{pid}] verified only partially — the capture is "
                    "real but stands for a different shape than planned. Count it only "
                    "after relabelling it to the achieved shape and re-deriving the "
                    "instances it represents, then set relabelled_to_achieved_shape.")
            phase = rec.get("phase")
            if phase in counted:
                counted[phase] += 1
        for phase in phases:
            # A phase with no observed activity contributes no mass and needs no points.
            if share.get(phase) and counted[phase] < min_points:
                raise ValueError(
                    f"phase {phase} carries mass but only {counted[phase]} verified "
                    f"sampling point(s) were counted; at least {min_points} are required. "
                    "Its shape variance and cost model are unobserved at one point.")
        measured = share.get("points_measured")
        if isinstance(measured, dict):
            for phase in phases:
                if measured.get(phase) is not None and measured[phase] != counted[phase]:
                    raise ValueError(
                        f"phase_share.points_measured.{phase}={measured[phase]} disagrees "
                        f"with {counted[phase]} counted point(s) in phase_share.points")

    fractions, total = {}, 0.0
    for phase in phases:
        value = share.get(phase)
        if value is None:
            raise ValueError(f"phase_share.{phase} is missing; absent is not zero")
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(value) or value < 0:
            raise ValueError(f"phase_share.{phase} must be a finite nonnegative number")
        fractions[phase] = float(value)
        total += float(value)
    if total <= 0:
        raise ValueError("phase_share must assign positive mass to at least one phase")
    # Accept either fractions or percentages; normalize so the two spellings agree.
    return {phase: value / total for phase, value in fractions.items()}, share


def _fusion_groups(document, by_id, group_pct_gt):
    """Evaluate declared fusion groups and return their verdicts.

    A chain of consecutive stages is a fusion candidate as a whole: rope-then-cache-write,
    norm-then-quant, score-then-top-k, gemm1-then-gemm2. Judging each stage alone against
    the threshold hides the chain, because the fusable mass is the sum and no single stage
    carries it. So a group whose combined share clears the threshold RECALLS every member,
    including stages far below it on their own.

    Summing is only sound for disjoint rows, which is why adjacency evidence is required
    and name-based grouping is rejected: a parent op and its child kernel share a name and
    overlap completely, and adding them inflates the group by the child's whole cost.
    """
    groups = document.get("fusion_groups", [])
    if groups in (None, []):
        return []
    if not isinstance(groups, list):
        raise ValueError("fusion_groups must be a list when present")

    verdicts, seen_ids = [], set()
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError("each fusion group must be an object")
        gid = group.get("id")
        if not isinstance(gid, str) or not gid.strip() or gid in seen_ids:
            raise ValueError("fusion groups need unique, nonempty id values")
        seen_ids.add(gid)

        members = group.get("members")
        if not isinstance(members, list) or len(members) < 2:
            raise ValueError(f"fusion group {gid!r} needs at least two members; a single "
                             "kernel is not a chain")
        if len(set(members)) != len(members):
            raise ValueError(f"fusion group {gid!r} repeats a member, which double-counts it")
        missing = [m for m in members if m not in by_id]
        if missing:
            raise ValueError(f"fusion group {gid!r} names rows absent from the ranking: "
                             f"{', '.join(map(str, missing))}")

        evidence = group.get("adjacency_evidence")
        kind = evidence.get("kind") if isinstance(evidence, dict) else None
        if kind not in ADJACENCY_EVIDENCE_KINDS:
            raise ValueError(
                f"fusion group {gid!r} needs adjacency_evidence.kind in "
                f"{sorted(ADJACENCY_EVIDENCE_KINDS)}; a grouping inferred from shared "
                "callable or kernel names is not evidence that the members are disjoint")

        group_pct = sum(by_id[m]["overall_pct"] for m in members)
        verdicts.append({
            "id": gid, "members": list(members), "group_overall_pct": group_pct,
            "member_overall_pct": {m: by_id[m]["overall_pct"] for m in members},
            "recalled": group_pct > group_pct_gt,
            "adjacency_evidence": evidence,
            "rationale": group.get("rationale", ""),
            "reason": "fusion_group_share_above_threshold" if group_pct > group_pct_gt
            else "fusion_group_share_not_above_threshold",
        })
    return verdicts


def select_targets(document, phases=("PREFILL", "DECODE"), overall_pct_gt=3,
                   min_points=MIN_POINTS_PER_PHASE, allow_window_denominator=False,
                   top_k=None, group_pct_gt=None, ut_exclude_kinds=("collective",)):
    _percent(overall_pct_gt, "overall_pct_gt")
    if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int)
                              or top_k < 1):
        raise ValueError("top_k must be a positive integer when given")
    if not phases or len(set(phases)) != len(phases):
        raise ValueError("phases must be nonempty and unique")
    ranking = document.get("ranking")
    if not isinstance(ranking, list) or not ranking:
        raise ValueError("ranking must be a nonempty list; missing evidence is not an empty target set")

    ut_exclude = {str(k).strip().lower() for k in (ut_exclude_kinds or ())}
    weights, share_doc = _phase_share(document, phases, min_points,
                                      allow_window_denominator)

    rows, seen = [], set()
    for row in ranking:
        identity = row.get("id", row.get("name"))
        if not isinstance(identity, str) or not identity.strip() or identity in seen:
            raise ValueError("ranking rows need unique, nonempty id or name values")
        seen.add(identity)
        shares = row.get("share_of_phase_pct")
        if not isinstance(shares, dict) or not shares:
            raise ValueError(f"{identity}.share_of_phase_pct is missing")
        for phase, share in shares.items():
            _percent(share, f"{identity}.share_of_phase_pct.{phase}")
        # The whole-workload share is reconstructed here, on the log-weighted split.
        # A ranking's own selection_pct_gpu_time is kept for disclosure only; it is
        # a window figure and is never what the threshold is applied to.
        overall = sum(weights[p] * shares.get(p, 0.0) for p in phases)
        reported = row.get("selection_pct_gpu_time")
        if reported is not None:
            _percent(reported, f"{identity}.selection_pct_gpu_time")
        kind = row.get("entity_kind")
        if kind is not None and (not isinstance(kind, str) or not kind.strip()):
            raise ValueError(f"{identity}.entity_kind must be a nonempty string when given")
        rows.append({"id": identity, "name": row.get("name", identity),
                     "overall_pct": overall, "phase_shares": shares,
                     "entity_kind": (kind or "gpu_kernel").strip().lower(),
                     "window_pct_reported": reported})

    per_phase = {}
    for phase in phases:
        ordered = sorted((r for r in rows if r["phase_shares"].get(phase, 0) > 0),
                         key=lambda r: (-r["phase_shares"][phase], r["id"]))
        for rank, row in enumerate(ordered, 1):
            row.setdefault("phase_ranks", {})[phase] = rank
        per_phase[phase] = {
            "observed_kernel_count": len(ordered),
            "phase_share_of_workload": weights[phase],
            # Ranks are inventory, not a gate: a phase leader can be below threshold
            # and a rank-20 row can be above it.
            "ranked": [{"id": r["id"], "name": r["name"], "phase_rank": r["phase_ranks"][phase],
                        "phase_pct": r["phase_shares"][phase], "overall_pct": r["overall_pct"],
                        "selected": r["overall_pct"] > overall_pct_gt} for r in ordered],
        }

    by_id = {r["id"]: r for r in rows}
    group_threshold = overall_pct_gt if group_pct_gt is None else _percent(
        group_pct_gt, "group_pct_gt")
    groups = _fusion_groups(document, by_id, group_threshold)

    # Recall every member of a group whose combined share clears the threshold, so the
    # chain stays whole for a later fusion. Membership is recorded per row; the group's
    # own mass is NOT added anywhere, so a member counts once no matter how many groups
    # name it.
    recalled_by = {}
    for group in groups:
        if not group["recalled"]:
            continue
        for member in group["members"]:
            recalled_by.setdefault(member, []).append(group["id"])
    for identity, group_ids in recalled_by.items():
        by_id[identity]["fusion_groups"] = group_ids

    individually = {r["id"] for r in rows if r["overall_pct"] > overall_pct_gt}
    qualified = sorted((r for r in rows
                        if r["id"] in individually or r["id"] in recalled_by),
                       key=lambda r: (-r["overall_pct"], r["id"]))
    capped = qualified[:top_k] if top_k is not None else qualified
    dropped_by_cap = [{"id": r["id"], "name": r["name"], "overall_pct": r["overall_pct"],
                       "reason": "above_threshold_but_outside_top_k"}
                      for r in qualified[top_k:]] if top_k is not None else []

    # A communication operator holds real device time, so it stays selected and stays in
    # the denominator -- but it is not optimized here, so no UT is authored for it. That
    # is a scope decision, not a deferral: `deferred` means pending work and invites a
    # retry that is never coming.
    def _ut_scope(r):
        if r["entity_kind"] in ut_exclude:
            return ("excluded",
                    "%s is a communication operator: selected for mass accounting, "
                    "out of optimization scope, no UT authored" % r["entity_kind"])
        return ("ut", "")

    targets = []
    for r in capped:
        scope, why = _ut_scope(r)
        targets.append({
            "id": r["id"], "name": r["name"], "overall_pct": r["overall_pct"],
            "entity_kind": r["entity_kind"],
            "selected_phases": [p for p in phases if r["phase_shares"].get(p, 0) > 0],
            "phase_ranks": r.get("phase_ranks", {}),
            "phase_pct": {p: r["phase_shares"][p] for p in phases
                          if r["phase_shares"].get(p, 0) > 0},
            "window_pct_reported": r["window_pct_reported"],
            "fusion_groups": r.get("fusion_groups", []),
            "ut_scope": scope,
            **({"ut_excluded_because": why} if why else {}),
            "reason": "overall_share_above_threshold" if r["id"] in individually
            else "recalled_by_fusion_group"})

    # capped holds distinct rows, so this is already deduplicated across groups.
    selected_mass = sum(r["overall_pct"] for r in capped)
    return {
        "scope": "extraction_target_selection_only",
        "policy": {"phases": list(phases), "overall_pct_gt": overall_pct_gt,
                   "combination": "overall_share_threshold_on_log_weighted_phase_split",
                   "rank_is_inventory_only": True, "top_k_cap": top_k,
                   "min_points_per_phase": min_points,
                   "fusion_group_pct_gt": group_threshold,
                   "ut_excluded_entity_kinds": sorted(ut_exclude),
                   "window_denominator_allowed": bool(allow_window_denominator)},
        "phase_share": {"weights": weights, "declared": share_doc},
        "denominators": document.get("denominators", {}),
        "per_phase": per_phase,
        "fusion_groups": groups,
        "targets": targets,
        "target_count": len(targets),
        "recalled_count": sum(1 for t in targets
                              if t["reason"] == "recalled_by_fusion_group"),
        "ut_target_count": sum(1 for t in targets if t["ut_scope"] == "ut"),
        "ut_excluded": [{"id": t["id"], "name": t["name"],
                         "entity_kind": t["entity_kind"],
                         "overall_pct": t["overall_pct"],
                         "why": t["ut_excluded_because"]}
                        for t in targets if t["ut_scope"] == "excluded"],
        "selected_overall_pct": selected_mass,
        "dropped_by_cap": dropped_by_cap,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ranking", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--phases", nargs="+", default=["PREFILL", "DECODE"])
    parser.add_argument("--overall-pct-gt", type=float, default=3)
    parser.add_argument("--min-points-per-phase", type=int, default=MIN_POINTS_PER_PHASE)
    parser.add_argument("--allow-window-denominator", action="store_true",
                        help="record a declared-approximate selection from one capture "
                             "window; the output is stamped non-comparable")
    parser.add_argument("--top-k", type=int, default=None,
                        help="optional cap on how many above-threshold targets are "
                             "returned; the remainder stays in dropped_by_cap")
    parser.add_argument("--ut-exclude-kind", action="append", default=None,
                        metavar="ENTITY_KIND",
                        help="entity_kind selected for mass accounting but not given "
                             "a UT (repeatable; defaults to 'collective')")
    parser.add_argument("--fusion-group-pct-gt", type=float, default=None,
                        help="threshold a declared fusion group's combined share must "
                             "exceed to recall its members (defaults to --overall-pct-gt)")
    args = parser.parse_args()
    if args.output and args.output.resolve() == args.ranking.resolve():
        parser.error("output must not overwrite the source ranking")
    try:
        result = select_targets(json.loads(args.ranking.read_text()), args.phases,
                                args.overall_pct_gt, args.min_points_per_phase,
                                args.allow_window_denominator, args.top_k,
                                args.fusion_group_pct_gt,
                                tuple(args.ut_exclude_kind)
                                if args.ut_exclude_kind else ("collective",))
    except (ValueError, TypeError, AttributeError) as error:
        parser.error(str(error))
    result["source_ranking_json"] = str(args.ranking.resolve())
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
