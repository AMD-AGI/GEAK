#!/usr/bin/env python3
"""Select phase Top-K kernels whose whole-workload share exceeds a threshold.

This computes scope only. It neither proves editability nor launches GPU work.
Input is the kernel-extract ranking artifact, on one common device-time basis.
"""

import argparse
import json
import math
from pathlib import Path


def _percent(value, field):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 100):
        raise ValueError(f"{field} must be a finite number in [0, 100]")
    return value


def _triton_base(row):
    """Jit base of a Triton specialization name (``base`` + ``_`` + constexpr suffix), else None.

    Same rule as kernel_selection._specialized_from: the suffix starts at the first all-caps
    segment and carries a digit-only segment, so ``base_v2`` stays its own kernel."""
    if "triton" not in (row.get("classification"), row.get("backend_guess")):
        return None
    segments = row["name"].split("_")
    for i in range(1, len(segments)):
        if segments[i].isupper():
            base = "_".join(segments[:i])
            rest = segments[i:]
            return base if base.strip("_") and any(s.isdigit() for s in rest) else None
    return None


def _merge_specializations(rows):
    """One row per Triton jit kernel: autotune variants are one source, so they share one Top-K slot."""
    merged, by_base = [], {}
    for row in rows:
        base = row.pop("base")
        if base is None or base not in by_base:
            merged.append(row)
            if base is not None:
                by_base[base] = row
            continue
        into = by_base[base]
        into["overall_pct"] = min(100, into["overall_pct"] + row["overall_pct"])
        for phase, share in row["phase_shares"].items():
            into["phase_shares"][phase] = min(100, into["phase_shares"].get(phase, 0) + share)
        into.setdefault("specializations", [into["id"]]).append(row["id"])
    return merged


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
                     "overall_pct": total, "phase_shares": dict(shares),
                     "base": _triton_base({**row, "name": row.get("name", identity)})})
    rows.sort(key=lambda r: -r["overall_pct"])
    rows = _merge_specializations(rows)

    selected, per_phase = {}, {}
    for phase in phases:
        ordered = sorted((r for r in rows if r["phase_shares"].get(phase, 0) > 0),
                         key=lambda r: (-r["phase_shares"][phase], r["id"]))
        top_rows = []
        # Rank first, then filter. Filtering before [:top_k] would promote rank 6.
        for rank, row in enumerate(ordered[:top_k], 1):
            keep = row["overall_pct"] > overall_pct_gt
            variants = {"specializations": row["specializations"]} if "specializations" in row else {}
            top_rows.append({"id": row["id"], "name": row["name"], "phase_rank": rank,
                             "phase_pct": row["phase_shares"][phase],
                             "overall_pct": row["overall_pct"], "selected": keep,
                             "reason": "phase_top_k_and_overall_threshold" if keep
                             else "overall_share_not_above_threshold", **variants})
            if keep:
                target = selected.setdefault(row["id"], {
                    "id": row["id"], "name": row["name"], "overall_pct": row["overall_pct"],
                    "selected_phases": [], "phase_ranks": {}, **variants,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ranking", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--phases", nargs="+", default=["PREFILL", "DECODE"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--overall-pct-gt", type=float, default=3)
    args = parser.parse_args()
    if args.output and args.output.resolve() == args.ranking.resolve():
        parser.error("output must not overwrite the source ranking")
    try:
        result = select_targets(json.loads(args.ranking.read_text()), args.phases,
                                args.top_k, args.overall_pct_gt)
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
