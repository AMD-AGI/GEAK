#!/usr/bin/env python3
"""Tier-B route report builder for the mega MoE analysis line.

Deterministic, stdlib-only. NEW infra -- NOT part of the SHA-pinned
`moe_bottleneck` skill bundle (SKILL.md / analyze.py / validate_output.py are
untouched). This builder shapes the per-rank/paired latencies the mega
benchmark *already* produces into the minimal `geak-megamoe-analysis-v1` report
that `analyze.py --report` accepts, so `analyze.py` can emit route-causality
findings (which stage-category dominates the fused-vs-baseline delta) at
`analysis_status=awaiting_measurement`, reduced confidence, with NO rocprofv3
measurement tracks. That heavier collection is Tier C, deliberately out of scope.

The output matches `_report(with_tracks=False)` in
tests/test_moe_bottleneck_analysis.py:32-60 (the legacy route form -- no
`metric` key -> analyze.py's legacy branch at analyze.py:219). analyze.py only
requires, per route_comparison: `tokens_per_rank`, `e2e_rank_max_delta_ms`,
`e2e_rank_max_delta_pct`, and a `profile_category_delta.<cat>` carrying
`rank_max_delta_ms` + `rank_max_delta_pct`. It reads `profile_confidence` /
`timing_confidence` off the comparison regardless of branch (analyze.py:145-156).

Granularity limit, surfaced honestly in the report: the bench harness lumps
stage2+combine into a single `stage2_combine` metric unless the candidate's
`component_timings` splits them. When lumped, we emit `stage1` from the split
metric, attribute the lumped delta to `stage2` (combine=0), stamp
`profile_confidence="low"` / `timing_confidence="low"` and
`incomplete_reasons=["stage2_combine lumped by bench harness"]`. Downstream
findings then carry the reduced-confidence label instead of overclaiming a
combine-vs-stage2 split the instrument never resolved.

CLI:
  python3 build_mega_route_report.py \
    --paired <paired_readings.json>            # [{guard,base,cand}, ...]
    --baseline-per-case <baseline.json>        # [{name, latency_ms, ...cats}, ...]
    [--component-timings <cand_components.json>]# optional per-category candidate ms
    [--tokens-per-rank <int>]                  # default: inferred from guard names
    [--baseline-stages <stages.json>]          # measured frozen per-stage ms per guard
    [--component-breakdown <rows.json>]        # Verify rows {component,busy_ms,wait_ms,...}
    [--breakdown-guard <guard>]                # guard for rows that carry no `guard`
    --output <report.json>

The three bracketed measurement flags are optional; without them the output is
byte-identical to the Tier-B form above. With them, candidate per-stage values
come from the fused kernel's own per-stage busy time (Verify
`component_breakdown`), baseline per-stage values from the measured frozen
kernels, each category carries its baseline/candidate ms, and the fused
candidate's in-kernel wait is surfaced as its own `fused_wait` category.

Exit 0 on success; prints GEAK_MEGA_ROUTE_REPORT=<path>.
"""

import argparse
import json
import sys
from statistics import median

SCHEMA_VERSION = "geak-megamoe-analysis-v1"

# Canonical analyze.py route categories.
CANON = ("stage1", "stage2", "combine")

# Tolerant key aliases -> canonical category. `component_timings` items are
# free-form (kernel_mega.js MEGA_ANALYSIS_SCHEMA additionalProperties:true),
# and BASELINE_PER_CASE rows are also open objects, so we probe by alias.
ALIASES = {
    "stage1": "stage1",
    "stage1_dispatch_gemm1": "stage1",
    "dispatch_gemm1": "stage1",
    "gemm1": "stage1",
    "stage2": "stage2",
    "gemm2": "stage2",
    "combine": "combine",
    "scatter": "combine",
    "p2p": "combine",
    # Lumped metric the bench reduces over ranks (build_live_evidence.py:104-109).
    "stage2_combine": "stage2_combine",
    "stage2combine": "stage2_combine",
}


def _num(x):
    """Best-effort float or None (never raises on dirty input)."""
    try:
        if x is None:
            return None
        v = float(x)
        # Reject NaN/inf so analyze.py's _finite() never trips.
        if v != v or v in (float("inf"), float("-inf")):
            return None
        return v
    except (TypeError, ValueError):
        return None


def _categories_from(obj):
    """Extract {canonical_or_lumped: ms} from an open dict by alias probing.

    Returns a dict possibly containing 'stage1','stage2','combine','stage2_combine'.
    Only the first alias hit per target wins (deterministic by ALIASES order).
    """
    out = {}
    if not isinstance(obj, dict):
        return out
    lowered = {str(k).strip().lower(): v for k, v in obj.items()}
    for alias, target in ALIASES.items():
        if target in out:
            continue
        if alias in lowered:
            v = _num(lowered[alias])
            if v is not None:
                out[target] = v
    return out


def _reduce_pairs(readings):
    """guard -> (median_base, median_cand) over its {base,cand} readings."""
    by_guard = {}
    for r in readings or []:
        if not isinstance(r, dict):
            continue
        g = str(r.get("guard", "")).strip()
        b = _num(r.get("base"))
        c = _num(r.get("cand"))
        if not g or b is None or c is None:
            continue
        by_guard.setdefault(g, {"base": [], "cand": []})
        by_guard[g]["base"].append(b)
        by_guard[g]["cand"].append(c)
    reduced = {}
    for g, d in by_guard.items():
        if d["base"] and d["cand"]:
            reduced[g] = (median(d["base"]), median(d["cand"]))
    return reduced


def _tokens_from_guard(guard, default):
    """Infer tokens_per_rank from a guard name like '8192_uniform' or 't8192'."""
    import re

    m = re.search(r"(\d{2,6})", str(guard))
    if m:
        return int(m.group(1))
    return default


def _baseline_index(rows):
    """name -> categories dict, from BASELINE_PER_CASE rows."""
    idx = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or row.get("case_id") or "").strip()
        if not name:
            continue
        idx[name] = _categories_from(row)
    return idx


def _match_baseline(guard, base_idx):
    """Find the baseline category dict whose case name best matches a guard."""
    if guard in base_idx:
        return base_idx[guard]
    gl = str(guard).lower()
    for name, cats in base_idx.items():
        nl = name.lower()
        if nl in gl or gl in nl:
            return cats
    return {}


def _breakdown_index(rows, default_guard):
    """guard -> categories from Verify component_breakdown rows.

    A row's `busy_ms` is the stage's own work time; `wait_ms` rows are summed
    into `_wait`. Rows without `guard` belong to `default_guard` (dropped when
    that is not given). `_methods` records the measurement methods seen.
    """
    idx = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        guard = str(row.get("guard") or default_guard or "").strip()
        comp = ALIASES.get(str(row.get("component", "")).strip().lower())
        if not guard or not comp:
            continue
        cats = idx.setdefault(guard, {"_wait": 0.0, "_methods": set()})
        busy = _num(row.get("busy_ms"))
        if busy is not None and comp not in cats:
            cats[comp] = busy
        wait = _num(row.get("wait_ms"))
        if wait is not None:
            cats["_wait"] += wait
        cats["_methods"].add(str(row.get("method", "")).strip().lower())
    for cats in idx.values():
        # Pair a split side with a lumped side by summing the split halves.
        if "stage2_combine" not in cats and "stage2" in cats and "combine" in cats:
            cats["stage2_combine"] = cats["stage2"] + cats["combine"]
    return idx


def _category_delta(base_cats, cand_cats, annotate=False):
    """Build profile_category_delta + confidence flags.

    Returns (profile_category_delta, lumped_bool). Deltas are candidate-minus-
    baseline per category. When only a lumped stage2_combine is available it is
    folded onto stage2 with combine=0 and lumped=True (low confidence upstream).
    """
    delta = {}
    lumped = False

    def _emit(cat, b, c):
        if b is None or c is None:
            return
        d = c - b
        pct = (d / b * 100.0) if b not in (0, 0.0) else 0.0
        delta[cat] = {"rank_max_delta_ms": d, "rank_max_delta_pct": pct}
        if annotate:
            delta[cat].update({"baseline_rank_max": b, "candidate_rank_max": c})

    # stage1 is always split when present.
    _emit("stage1", base_cats.get("stage1"), cand_cats.get("stage1"))

    have_split = ("stage2" in base_cats and "stage2" in cand_cats) or (
        "combine" in base_cats and "combine" in cand_cats
    )
    if have_split:
        _emit("stage2", base_cats.get("stage2"), cand_cats.get("stage2"))
        _emit("combine", base_cats.get("combine"), cand_cats.get("combine"))
    else:
        # Fold the lumped stage2_combine onto stage2; combine unresolved -> 0.
        b = base_cats.get("stage2_combine")
        c = cand_cats.get("stage2_combine")
        if b is not None and c is not None:
            _emit("stage2", b, c)
            delta["combine"] = {"rank_max_delta_ms": 0.0, "rank_max_delta_pct": 0.0}
            lumped = True
    return delta, lumped


def build_report(paired, baseline_rows, component_timings, tokens_default,
                 baseline_stages=None, breakdown=None, breakdown_guard=None):
    reduced = _reduce_pairs(paired)
    base_idx = _baseline_index(baseline_rows)
    cand_idx = _baseline_index(component_timings) if component_timings else {}
    measured = baseline_stages is not None or breakdown is not None
    for name, cats in _baseline_index(baseline_stages).items():
        if "stage2_combine" not in cats and "stage2" in cats and "combine" in cats:
            cats["stage2_combine"] = cats["stage2"] + cats["combine"]
        base_idx.setdefault(name, {}).update(cats)
    bd_idx = _breakdown_index(breakdown, breakdown_guard)

    cases = [{"case_id": name} for name in base_idx] or [
        {"case_id": g} for g in reduced
    ]

    route_comparisons = []
    for guard, (b_e2e, c_e2e) in reduced.items():
        e2e_delta = c_e2e - b_e2e
        e2e_pct = (e2e_delta / b_e2e * 100.0) if b_e2e else 0.0
        base_cats = _match_baseline(guard, base_idx)
        cand_cats = bd_idx.get(guard) or _match_baseline(guard, cand_idx)
        pcd, lumped = _category_delta(base_cats, cand_cats, annotate=measured)
        rc = {
            "tokens_per_rank": _tokens_from_guard(guard, tokens_default),
            "e2e_rank_max_delta_ms": e2e_delta,
            "e2e_rank_max_delta_pct": e2e_pct,
            "profile_category_delta": pcd,
        }
        # Tier B never has rocprof tracks -> confidence is capped low, and lower
        # still (with an explicit reason) when stage2/combine could not be split.
        rc["timing_confidence"] = "low"
        rc["profile_confidence"] = "low"
        if lumped:
            rc["incomplete_reasons"] = ["stage2_combine lumped by bench harness"]
        if measured:
            wait = cand_cats.get("_wait") if guard in bd_idx else None
            if wait:
                pcd["fused_wait"] = {
                    "rank_max_delta_ms": wait,
                    "rank_max_delta_pct": (wait / b_e2e * 100.0) if b_e2e else 0.0,
                    "baseline_rank_max": 0.0, "candidate_rank_max": wait,
                }
                rc["category_notes"] = {"fused_wait": (
                    "in-kernel cross-stage wait of the fused candidate (per-workgroup "
                    "average, rank-max); the serialized baseline kernels have no in-kernel "
                    "cross-stage wait, so baseline=0 and pct is relative to baseline e2e")}
            if not pcd:
                rc.setdefault("incomplete_reasons", []).append(
                    "no per-stage measurement for this guard")
            elif (guard in bd_idx and cand_cats["_methods"] == {"timestamp"}
                    and not lumped and "stage1" in pcd):
                rc["profile_confidence"] = "medium"
        route_comparisons.append(rc)

    return {
        "schema_version": SCHEMA_VERSION,
        "cases": cases,
        "route_comparisons": route_comparisons,
    }


def _load(path):
    if not path:
        return None
    with open(path, "r") as f:
        return json.load(f)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build a Tier-B mega route report.")
    ap.add_argument("--paired", required=True, help="paired_readings JSON [{guard,base,cand}]")
    ap.add_argument("--baseline-per-case", required=True, help="BASELINE_PER_CASE JSON")
    ap.add_argument("--component-timings", default=None, help="optional candidate per-category JSON")
    ap.add_argument("--tokens-per-rank", type=int, default=8192)
    ap.add_argument("--baseline-stages", default=None, help="measured baseline per-stage JSON")
    ap.add_argument("--component-breakdown", default=None, help="Verify component_breakdown JSON")
    ap.add_argument("--breakdown-guard", default=None, help="guard for rows without `guard`")
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)

    paired = _load(args.paired)
    baseline_rows = _load(args.baseline_per_case)
    component_timings = _load(args.component_timings)
    baseline_stages = _load(args.baseline_stages)
    breakdown = _load(args.component_breakdown)
    if isinstance(baseline_stages, dict):
        baseline_stages = baseline_stages.get("baseline_stages")
    if isinstance(breakdown, dict):  # accept reduce_stage_meter.py output directly
        breakdown = breakdown.get("component_breakdown")

    if not isinstance(paired, list) or not isinstance(baseline_rows, list):
        # Degrade, never fail: emit a schema-valid empty report so analyze.py
        # still runs and reports awaiting_measurement.
        report = {"schema_version": SCHEMA_VERSION, "cases": [], "route_comparisons": []}
    else:
        report = build_report(paired, baseline_rows, component_timings, args.tokens_per_rank,
                              baseline_stages, breakdown, args.breakdown_guard)

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(f"GEAK_MEGA_ROUTE_REPORT={args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
