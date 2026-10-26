#!/usr/bin/env python3
"""Select phase Top-K operators (or kernels) whose whole-workload share exceeds a threshold.

This computes scope only. It neither proves editability nor launches GPU work.
Input is the kernel-extract ranking artifact, on one common device-time basis.

Three modes:

  (default)        per kernel name, as the ranking lists them
  --candidates     the attribution set: every kernel above --candidate-min-pct overall or
                   in a phase's top --candidate-top. kernel_shapes.py / source_map.py
                   resolve their launch stacks so selection can see operators
  --source-map     per operator: kernels are summed under source_map.py's op key (library
                   entry @ innermost nn.Module) and the same Top-K / threshold rule applies
                   to the sums. One operator's mass is often split over kernels none of
                   which clears 3% alone -- GLM-5.2's fused MoE ran gemm1 (9.9%), two
                   gemm2 variants (2.3% + 2.1%) and a sort/quant kernel (1.8%) -- and by
                   kernel name the stage-2 and sort kernels dropped out of their own op.
                   Every member of a selected operator becomes a target row (`op`,
                   `selected_via`). An operator in the same nn.Module class with the same
                   family, launched only in phases the selected one is absent from, is
                   its other phase branch (e.g. the indexer's prefill score next to its
                   decode score) and is selected with it (`phase_branch_of`); "same
                   family" = some family's kernel patterns match members of both.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _percent(value, field):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 100):
        raise ValueError(f"{field} must be a finite number in [0, 100]")
    return value


def select_targets(document, phases=("PREFILL", "DECODE"), top_k=5, overall_pct_gt=3):
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    _percent(overall_pct_gt, "overall_pct_gt")
    if not phases or len(set(phases)) != len(phases):
        raise ValueError("phases must be nonempty and unique")
    ranking = document.get("ranking")
    if not isinstance(ranking, list) or not ranking:
        raise ValueError("ranking must be a nonempty list; missing evidence is not an empty target set")

    rows, seen = [], set()
    for row in ranking:
        identity = row.get("id", row.get("name"))
        if not isinstance(identity, str) or not identity.strip() or identity in seen:
            raise ValueError("ranking rows need unique, nonempty id or name values")
        seen.add(identity)
        total = _percent(row.get("selection_pct_gpu_time"), f"{identity}.selection_pct_gpu_time")
        shares = row.get("share_of_phase_pct")
        if not isinstance(shares, dict) or not shares:
            raise ValueError(f"{identity}.share_of_phase_pct is missing")
        for phase, share in shares.items():
            _percent(share, f"{identity}.share_of_phase_pct.{phase}")
        rows.append({"id": identity, "name": row.get("name", identity),
                     "overall_pct": total, "phase_shares": shares})

    selected, per_phase = {}, {}
    for phase in phases:
        ordered = sorted((r for r in rows if r["phase_shares"].get(phase, 0) > 0),
                         key=lambda r: (-r["phase_shares"][phase], r["id"]))
        top_rows = []
        # Rank first, then filter. Filtering before [:top_k] would promote rank 6.
        for rank, row in enumerate(ordered[:top_k], 1):
            keep = row["overall_pct"] > overall_pct_gt
            top_rows.append({"id": row["id"], "name": row["name"], "phase_rank": rank,
                             "phase_pct": row["phase_shares"][phase],
                             "overall_pct": row["overall_pct"], "selected": keep,
                             "reason": "phase_top_k_and_overall_threshold" if keep
                             else "overall_share_not_above_threshold"})
            if keep:
                target = selected.setdefault(row["id"], {
                    "id": row["id"], "name": row["name"], "overall_pct": row["overall_pct"],
                    "selected_phases": [], "phase_ranks": {},
                })
                target["selected_phases"].append(phase)
                target["phase_ranks"][phase] = rank
        per_phase[phase] = {"observed_kernel_count": len(ordered), "top_k": top_rows}

    return {
        "scope": "extraction_target_selection_only",
        "policy": {"phases": list(phases), "top_k_per_phase": top_k,
                   "overall_pct_gt": overall_pct_gt,
                   "combination": "phase_top_k_and_overall_threshold", "backfill": False},
        "denominators": document.get("denominators", {}),
        "per_phase": per_phase, "targets": list(selected.values()),
        "target_count": len(selected),
    }


def candidates(document, phases=("PREFILL", "DECODE"), top=30, min_pct=0.1):
    """Kernels whose launch stacks selection by operator needs."""
    rows = document.get("ranking") or []
    keep = {}
    for r in rows:
        if r.get("kind") == "memory":
            continue
        if (r.get("selection_pct_gpu_time") or 0) >= min_pct:
            keep[r["id"]] = r
    for ph in phases:
        ordered = sorted((r for r in rows if (r.get("share_of_phase_pct") or {}).get(ph, 0) > 0
                          and r.get("kind") != "memory"),
                         key=lambda r: (-r["share_of_phase_pct"][ph], r["id"]))
        for r in ordered[:top]:
            keep[r["id"]] = r
    out = sorted(keep.values(), key=lambda r: -(r.get("selection_pct_gpu_time") or 0))
    return {"scope": "attribution_candidates",
            "policy": {"phases": list(phases), "top_per_phase": top, "min_overall_pct": min_pct},
            "targets": [{"id": r["id"], "name": r.get("name", r["id"]),
                         "overall_pct": r.get("selection_pct_gpu_time")} for r in out],
            "target_count": len(out),
            "covered_pct": round(sum(r.get("selection_pct_gpu_time") or 0 for r in out), 3)}


def _families_of(kernels):
    try:
        from draft_task_decls import families, families_hit
    except ImportError:
        return set()
    return families_hit(kernels, families())


def select_ops(document, source_map, phases=("PREFILL", "DECODE"), top_k=5, overall_pct_gt=3):
    sm = source_map.get("kernels") or {}
    ops = {}
    for r in document.get("ranking") or []:
        k = r.get("id", r.get("name"))
        op = (sm.get(k) or {}).get("op") if r.get("kind") != "memory" else None
        key = op["key"] if op else "kernel:" + k
        o = ops.setdefault(key, {"id": key, "name": key, "members": [], "op": op,
                                 "selection_pct_gpu_time": 0.0, "share_of_phase_pct": {}})
        o["members"].append({"id": k, "overall_pct": r.get("selection_pct_gpu_time") or 0,
                             "share_of_phase_pct": r.get("share_of_phase_pct") or {},
                             "kind": r.get("kind"),
                             "op_keys_seen": (sm.get(k) or {}).get("op_keys_seen")})
        o["selection_pct_gpu_time"] += r.get("selection_pct_gpu_time") or 0
        for ph, v in (r.get("share_of_phase_pct") or {}).items():
            o["share_of_phase_pct"][ph] = o["share_of_phase_pct"].get(ph, 0) + v
    for o in ops.values():
        o["selection_pct_gpu_time"] = min(100.0, round(o["selection_pct_gpu_time"], 6))
        o["share_of_phase_pct"] = {ph: min(100.0, round(v, 6))
                                   for ph, v in o["share_of_phase_pct"].items()}
        if not o["share_of_phase_pct"]:
            o["share_of_phase_pct"] = {phases[0]: 0.0}
    res = select_targets({"ranking": list(ops.values()),
                          "denominators": document.get("denominators", {})},
                         phases, top_k, overall_pct_gt)
    chosen = {t["id"]: t for t in res["targets"]}

    # the other phase branch of a selected operator
    for key in list(chosen):
        o = ops[key]
        cls = (o["op"] or {}).get("module_class")
        if not cls:
            continue
        live = {ph for ph, v in o["share_of_phase_pct"].items() if v > 0}
        fam = _families_of([m["id"] for m in o["members"]])
        for k2, o2 in ops.items():
            if k2 in chosen or (o2["op"] or {}).get("module_class") != cls:
                continue
            live2 = {ph for ph, v in o2["share_of_phase_pct"].items() if v > 0}
            if not live2 or live2 & live or not fam:
                continue
            if not fam & _families_of([m["id"] for m in o2["members"]]):
                continue
            chosen[k2] = {"id": k2, "name": k2, "overall_pct": o2["selection_pct_gpu_time"],
                          "selected_phases": sorted(live2), "phase_ranks": {},
                          "phase_branch_of": key}

    targets, op_rows = [], []
    for key, t in chosen.items():
        o = ops[key]
        members = sorted(o["members"], key=lambda m: -m["overall_pct"])
        op_rows.append({"id": key, "overall_pct": o["selection_pct_gpu_time"],
                        "share_of_phase_pct": o["share_of_phase_pct"],
                        "selected_phases": t["selected_phases"], "phase_ranks": t["phase_ranks"],
                        "phase_branch_of": t.get("phase_branch_of"),
                        "entry": ((o["op"] or {}).get("entry") or {}).get("target_callable"),
                        "module_class": (o["op"] or {}).get("module_class"),
                        "members": [m["id"] for m in members]})
        for m in members:
            targets.append({"id": m["id"], "name": m["id"], "overall_pct": m["overall_pct"],
                            "op": key, "op_overall_pct": o["selection_pct_gpu_time"],
                            "selected_phases": t["selected_phases"],
                            "phase_ranks": t["phase_ranks"],
                            "selected_via": ("phase_branch_of:" + t["phase_branch_of"])
                            if t.get("phase_branch_of") else "op",
                            "op_keys_seen": m.get("op_keys_seen")})
    res["policy"]["unit"] = "operator"
    res["policy"]["operator_key"] = "source_map op.key (library entry @ innermost nn.Module)"
    res["policy"]["phase_branches"] = "same nn.Module class and family, disjoint phases"
    res["ops"] = sorted(op_rows, key=lambda r: -r["overall_pct"])
    res["targets"], res["target_count"] = targets, len(targets)
    res["op_count"] = len(op_rows)
    res["selected_pct"] = round(sum(r["overall_pct"] for r in op_rows), 3)
    res["op_ranking"] = [{"id": o["id"], "overall_pct": o["selection_pct_gpu_time"],
                          "share_of_phase_pct": o["share_of_phase_pct"],
                          "members": len(o["members"])}
                         for o in sorted(ops.values(), key=lambda o: -o["selection_pct_gpu_time"])[:40]]
    return res


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ranking", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--phases", nargs="+", default=["PREFILL", "DECODE"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--overall-pct-gt", type=float, default=3)
    parser.add_argument("--source-map", type=Path,
                        help="source_map.py output: select by operator instead of kernel name")
    parser.add_argument("--candidates", action="store_true",
                        help="write the attribution set (kernels whose stacks to resolve)")
    parser.add_argument("--candidate-top", type=int, default=30)
    parser.add_argument("--candidate-min-pct", type=float, default=0.1)
    args = parser.parse_args()
    if args.output and args.output.resolve() == args.ranking.resolve():
        parser.error("output must not overwrite the source ranking")
    try:
        doc = json.loads(args.ranking.read_text())
        if args.candidates:
            result = candidates(doc, args.phases, args.candidate_top, args.candidate_min_pct)
        elif args.source_map:
            result = select_ops(doc, json.loads(args.source_map.read_text()), args.phases,
                                args.top_k, args.overall_pct_gt)
        else:
            result = select_targets(doc, args.phases, args.top_k, args.overall_pct_gt)
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
