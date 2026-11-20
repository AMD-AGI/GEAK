#!/usr/bin/env python3
"""Measured sampling points + UT cases  ->  covered and UNCOVERED device-time mass.

Step four, and the one the report is graded on. It converts per-point kernel
measurements into whole-leg device-time mass using the observed instance counts, then
states what fraction of that mass a unit test actually exercises -- and what fraction
it does not, with the same prominence.

    mass(kernel) = SUM over sampling points of
                     per_instance_device_us(kernel, point) x instances_represented(point)

The instance counts come from the serving log of the measured leg, so this is the
leg's mass, not the probe's.

The refusal that makes the number mean something
------------------------------------------------
If a phase was measured at ONE point, every kernel's mass in that phase is
`one_measurement x count`. A unit test written at that same point then "covers" 100%
of it -- not because it covers the workload, but because the denominator and the
numerator are the same single measurement. That number is a tautology, and reporting it
as coverage is the most flattering mistake available here.

So: a phase measured at a single point gets NO mass-coverage percentage. It gets the
instance-share the UT shapes cover (a real, weaker statement) and an explicit note that
mass coverage is unavailable until a second point is measured. Two phases can therefore
report two different kinds of number in the same run, and they are labelled as such.

Collective spin is occupied time, not kernel work
-------------------------------------------------
On unbalanced shapes, ranks that arrive early spin inside a collective and the profiler
bills that spin as device time -- enough to rank an all-reduce as the single largest
"kernel" in the workload, sending optimization after a symptom of rank skew. With
per-rank costs available, each collective is split
    transfer = min over ranks      (the laggard pays transfer only)
    wait     = mean - min          (the skew)
`wait` is booked to `__collective_wait__`, which STAYS in the denominator -- it is real
occupied time -- but is marked not optimizable so it cannot be selected as a target.
Total mass is unchanged by the split.

Stdlib only.
"""
import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict

COLLECTIVE_PAT = re.compile(r"cross_device_reduce|quickreduce|allreduce|nccl|rccl", re.I)
WAIT_KERNEL = "__collective_wait__"
TAIL_KERNEL = "__untracked_tail__"


def load_census(path, phase=None):
    """Per-kernel device microseconds for ONE instance of a point's shape.

    Accepted shapes (first that matches):
      {"per_kernel": {name: us, ...}}
      {"per_kernel": {name: {"total_us": us, ...}}}
      {"kernels":  [{"name":.., "total_us":..}, ...]}
      {"per_rank": {rank: {name: us}}}        -- enables the collective split
    `instances` (how many step instances the census covers) is divided out if present.
    """
    d = json.load(open(path))
    per_rank = d.get("per_rank")
    pk = d.get("per_kernel")
    if pk is None and isinstance(d.get("kernels"), list):
        pk = {k["name"]: k.get("total_us", 0) for k in d["kernels"]}
    if "top_kernels" in d:
        raise SystemExit("coverage_ledger: Top-N summaries are not a census; use point_census.py on the raw trace")
    if pk is None and per_rank:
        agg = defaultdict(float)
        for _r, kv in per_rank.items():
            for k, v in kv.items():
                agg[k] += _us(v)
        n = max(1, len(per_rank))
        pk = {k: v / n for k, v in agg.items()}
    if pk is None:
        raise SystemExit(
            "coverage_ledger: %s has none of per_kernel / kernels / per_rank. The census "
            "format is documented in load_census(); a silently empty census would read "
            "as a kernel-free workload." % path)
    pk = {k: _us(v) for k, v in pk.items()}
    inst = d.get("instances") or d.get("step_instances") or 1
    if inst and inst != 1:
        pk = {k: v / inst for k, v in pk.items()}
        # main() rebuilds the census from per_rank when it is present, so it has to be
        # reduced to one instance as well; otherwise a window total is multiplied by the
        # observed instance count as if it were one instance's cost.
        if per_rank:
            per_rank = {r: {k: _us(v) / inst for k, v in kv.items()}
                        for r, kv in per_rank.items()}
    return pk, per_rank, d


def _us(v):
    if isinstance(v, dict):
        return float(v.get("total_us", v.get("us", 0)) or 0)
    return float(v or 0)


def split_collectives(per_rank):
    """-> {kernel: us} with collective spin moved to __collective_wait__."""
    if not per_rank:
        return None
    names = set()
    for kv in per_rank.values():
        names.update(kv)
    out, wait = {}, 0.0
    nranks = len(per_rank)
    for name in names:
        vals = [_us(per_rank[r].get(name, 0)) for r in per_rank]
        mean = sum(vals) / nranks
        if COLLECTIVE_PAT.search(name):
            lo = min(vals)
            out[name] = lo
            wait += mean - lo
        else:
            out[name] = mean
    if wait > 0:
        # Add, never assign: a census that already split its collectives per launch
        # (point_census.py) carries its own __collective_wait__ row, which reaches this
        # point as an ordinary rank mean. Its rows are identical across ranks, so the
        # residual here is float noise, and assigning it erased the census's wait.
        out[WAIT_KERNEL] = out.get(WAIT_KERNEL, 0.0) + wait
    return out


def ut_covers(case, phase, shape, tol):
    if (case.get("phase") or "").upper() != phase:
        return False
    s = case.get("shape")
    if s in (None, 0) or not shape:
        return False
    try:
        return abs(math.log(float(s) / float(shape))) <= math.log(tol)
    except (ValueError, ZeroDivisionError):
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--points", required=True, help="sampling_points.json or sampling_points_final.json")
    ap.add_argument("--census", action="append", default=[], metavar="POINT_ID=PATH",
                    help="per-point kernel census; repeat per measured point")
    ap.add_argument("--ut-cases", default=None,
                    help='JSON list of UT cases: [{"sig":..,"phase":"PREFILL",'
                         '"shape":4761,"effective_calls":N}]')
    ap.add_argument("--shape-tolerance", type=float, default=1.25,
                    help="a UT case covers a point within this multiplicative factor")
    ap.add_argument("--target", type=float, default=90.0,
                    help="coverage target per material phase, for the verdict line")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    plan = json.load(open(args.points))
    by_id = {p["point_id"]: p for p in plan["points"]}

    censuses = {}
    for spec in args.census:
        if "=" not in spec:
            raise SystemExit("coverage_ledger: --census wants POINT_ID=PATH, got %r" % spec)
        pid, path = spec.split("=", 1)
        if pid not in by_id:
            raise SystemExit("coverage_ledger: census for unknown point %r. Planned: %s"
                             % (pid, ", ".join(by_id)))
        pk, per_rank, raw = load_census(path, by_id[pid]["phase"])
        split = split_collectives(per_rank)
        censuses[pid] = {"per_kernel": split or pk, "path": path,
                         "collective_split_applied": bool(split),
                         "ranks": len(per_rank) if per_rank else None}

    missing = [p for p in by_id if p not in censuses]

    ut_cases = json.load(open(args.ut_cases)) if args.ut_cases else []
    if isinstance(ut_cases, dict):
        ut_cases = ut_cases.get("cases", [])

    # ---- mass per kernel per phase -------------------------------------
    phases = defaultdict(lambda: {"points": [], "mass": defaultdict(float),
                                  "covered": defaultdict(float), "instances": 0})
    for pid, c in censuses.items():
        p = by_id[pid]
        phase = p["phase"]
        inst = p["represented"]["instances"]
        shape = p.get("new_tokens") if phase == "PREFILL" else p.get("batch_size")
        covered = any(ut_covers(u, phase, shape, args.shape_tolerance) for u in ut_cases)
        e = phases[phase]
        e["points"].append({"point_id": pid, "shape": shape, "instances": inst,
                            "ut_covered": covered,
                            "collective_wait_mass_us": c["per_kernel"].get(WAIT_KERNEL, 0.0) * inst,
                            "max_shape_ratio": p["represented"]["max_shape_ratio"]})
        e["instances"] += inst
        for k, us in c["per_kernel"].items():
            e["mass"][k] += us * inst
            if covered:
                e["covered"][k] += us * inst

    report = {"points_plan": args.points, "shape_tolerance": args.shape_tolerance,
              "ut_cases_supplied": len(ut_cases), "phases": {}, "warnings": []}

    if missing:
        report["warnings"].append(
            "%d planned point(s) have no census and contributed NO mass: %s. Their "
            "represented instances are absent from every denominator below, so these "
            "numbers describe the points that were measured, not the whole leg."
            % (len(missing), ", ".join(sorted(missing))))

    for phase, e in phases.items():
        total = sum(e["mass"].values())
        cov = sum(e["covered"].values())
        npts = len(e["points"])
        single = npts < 2
        kernels = sorted(({"kernel": k, "mass_us": round(v, 1),
                           "phase_mass_share_pct": round(100.0 * v / total, 4) if total else 0,
                           "covered_mass_us": round(e["covered"].get(k, 0.0), 1),
                           "optimizable": k not in (WAIT_KERNEL, TAIL_KERNEL)}
                          for k, v in e["mass"].items()),
                         key=lambda d: -d["mass_us"])
        ph = {
            "points_measured": npts,
            "instances_represented": e["instances"],
            "total_mass_us": round(total, 1),
            "kernels": kernels,
            "points": e["points"],
            "uncovered_points": [p for p in e["points"] if not p["ut_covered"]],
        }
        if single:
            inst_cov = sum(p["instances"] for p in e["points"] if p["ut_covered"])
            ph["mass_coverage_pct"] = None
            ph["mass_coverage_refused_because"] = (
                "this phase was measured at ONE sampling point, so every kernel's mass "
                "here is one measurement times a count. A UT written at that point "
                "covers 100%% of it by construction, which says nothing about the "
                "workload. Measure a second point to make a mass percentage meaningful.")
            ph["instance_coverage_pct"] = round(
                100.0 * inst_cov / e["instances"], 3) if e["instances"] else 0.0
            ph["instance_coverage_meaning"] = (
                "share of observed phase instances whose shape a UT case exercises. A "
                "real but weaker statement than mass coverage.")
        else:
            ph["mass_coverage_pct"] = round(100.0 * cov / total, 3) if total else 0.0
            ph["uncovered_mass_us"] = round(total - cov, 1)
            ph["uncovered_mass_pct"] = round(100.0 * (total - cov) / total, 3) if total else 0.0
            ph["meets_target"] = ph["mass_coverage_pct"] >= args.target
        tail = e["mass"].get(TAIL_KERNEL, 0.0)
        if tail and total:
            report["warnings"].append(
                "%s: %.1f%% of the device mass is below the census's top-N cut and is "
                "booked to %s -- it is in the denominator but has no kernel identity, so "
                "it can neither be targeted nor claimed as covered. Re-run the census "
                "with a larger --top to name it."
                % (phase, 100.0 * tail / total, TAIL_KERNEL))
        report["phases"][phase] = ph

    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1)

    print("=== coverage ledger (%d UT cases, tolerance %.2fx)"
          % (len(ut_cases), args.shape_tolerance))
    for phase, ph in report["phases"].items():
        if ph.get("mass_coverage_pct") is None:
            print("  %-8s %d point(s): mass coverage NOT REPORTABLE (single point); "
                  "instance coverage %.1f%%"
                  % (phase, ph["points_measured"], ph["instance_coverage_pct"]))
        else:
            print("  %-8s %d points: covered %.2f%% of %.0f ms device mass -- "
                  "UNCOVERED %.2f%% %s"
                  % (phase, ph["points_measured"], ph["mass_coverage_pct"],
                     ph["total_mass_us"] / 1000.0, ph["uncovered_mass_pct"],
                     "(meets %.0f%% target)" % args.target if ph["meets_target"]
                     else "(BELOW the %.0f%% target)" % args.target))
        for up in ph["uncovered_points"]:
            print("      uncovered point %s (shape %s, %d instances)"
                  % (up["point_id"], up["shape"], up["instances"]))
    for w in report["warnings"]:
        print("  !! %s" % w)
    print("wrote %s" % args.out)


if __name__ == "__main__":
    sys.exit(main())
