#!/usr/bin/env python3
"""coverage_ledger.py output -> kernel_ranking.json on one whole-workload denominator.

Step five, between the ledger and `select_targets.py`. Every run used to hand-write this
(the last two wrote 91- and 294-line versions with different denominators), so it is a
script now.

    python3 build_ranking.py --coverage coverage.json --out kernel_ranking.json \
        [--eager-captured DECODE] [--probe-result evidence/probe/*.json]

Denominator
-----------
mass(kernel) = per-instance census x observed instance count, summed over the points of
each phase (the ledger computes it). The denominator is the sum over phases. A kernel's
`selection_pct_gpu_time` is its share of that; `share_of_phase_pct` its share inside one
phase. Nested scopes, not additive.

Rows that are not kernels stay in the denominator but out of `ranking`, so they can
neither take a Top-5 slot nor be selected: `__untracked_tail__` (mass below a census
cut) and `__collective_wait__` (a collective's peer spin, split out by point_census).
They are listed under `non_selectable`.

`--eager-captured PHASE`: that phase is deployed under a CUDA graph but was captured
eagerly, because graph replay hides its kernels from the profiler. Kernel durations
carry over; the collective wait does not -- with launches spread out by Python every
rank idles inside the next all-reduce (measured: 92% of an eager decode step, nine times
the wall clock of the leg once multiplied by the observed iteration count). That phase's
`__collective_wait__` is therefore taken OUT of the denominator and reported under
`excluded_from_denominator` with this reason. Collective transfer stays in.

Stdlib only.
"""
import argparse
import glob
import json
import os
import sys

WAIT = "__collective_wait__"
TAIL = "__untracked_tail__"
NON_KERNEL = {WAIT, TAIL}


def kind(name):
    n = name.lower()
    if name in NON_KERNEL:
        return "non_kernel_accounting"
    if n.startswith("memcpy") or n.startswith("memset"):
        return "memory"
    if any(t in n for t in ("cross_device_reduce", "allreduce", "all_reduce", "quickreduce",
                            "allgather", "all_gather", "reduce_scatter", "nccl", "rccl")):
        return "collective"
    return "compute"


def probe_points(paths):
    """serving_probe.py results -> phase_share.points records."""
    out = []
    for p in paths:
        try:
            d = json.load(open(p))
        except (OSError, ValueError):
            continue
        if "point_id" not in d or "status" not in d:
            continue
        out.append({"point_id": d["point_id"], "phase": (d.get("point") or {}).get("phase"),
                    "serving_probe_status": d["status"],
                    "trace_verified": d.get("trace_verified"), "result": os.path.abspath(p)})
    return out


def build(cov, eager_captured=(), probes=(), eager_points=()):
    phases = cov.get("phases") or {}
    if not phases:
        raise SystemExit("build_ranking: the ledger has no phases; no census was given to it")
    eager_points = set(eager_points)
    known = {p["point_id"] for e in phases.values() for p in e.get("points", [])}
    if eager_points - known:
        raise ValueError("eager points absent from ledger: %s" % sorted(eager_points - known))
    excluded, phase_total = {}, {}
    for ph, e in phases.items():
        total = e["total_mass_us"]
        phase_wait = sum(k["mass_us"] for k in e["kernels"] if k["kernel"] == WAIT)
        selected = [p for p in e.get("points", []) if p["point_id"] in eager_points]
        if ph in eager_captured:
            wait = phase_wait
        else:
            if any("collective_wait_mass_us" not in p for p in selected):
                raise ValueError("point wait mass missing; regenerate coverage_ledger")
            wait = min(phase_wait, sum(p["collective_wait_mass_us"] for p in selected))
        wait = round(wait, 1)
        if wait:
            excluded[ph] = {"kernel": WAIT, "mass_us": round(wait, 1),
                            "point_ids": ([p["point_id"] for p in e.get("points", [])]
                                          if ph in eager_captured else
                                          [p["point_id"] for p in selected]),
                            "pct_of_captured_phase": round(100 * wait / total, 3),
                            "why": "these points were captured eagerly, deployed under a CUDA "
                                   "graph: their peer spin is an artifact of eager launch"}
            total -= wait
        phase_total[ph] = total
    grand = sum(phase_total.values())
    if grand <= 0:
        raise SystemExit("build_ranking: zero total mass")

    rows, nonsel = {}, {}
    for ph, e in phases.items():
        for k in e["kernels"]:
            name = k["kernel"]
            mass = k["mass_us"]
            if name == WAIT and ph in excluded:
                mass = max(0.0, mass - excluded[ph]["mass_us"])
            if mass == 0:
                continue
            share = 100.0 * mass / phase_total[ph] if phase_total[ph] else 0.0
            dest = nonsel if name in NON_KERNEL else rows
            r = dest.setdefault(name, {"id": name, "name": name, "kind": kind(name),
                                       "mass_us": 0.0, "share_of_phase_pct": {}})
            r["mass_us"] += mass
            r["share_of_phase_pct"][ph] = round(share, 4)
    for r in list(rows.values()) + list(nonsel.values()):
        for ph in phases:
            # Every phase was censused, so absence from one is a measured zero there.
            r["share_of_phase_pct"].setdefault(ph, 0.0)
        r["selection_pct_gpu_time"] = round(100.0 * r["mass_us"] / grand, 4)
        r["mass_us"] = round(r["mass_us"], 1)
    ranking = sorted(rows.values(), key=lambda r: (-r["mass_us"], r["id"]))

    counted = {p["point_id"] for e in phases.values() for p in e.get("points", [])}
    pts = []
    for p in probes:
        p = dict(p, counted=p["point_id"] in counted)
        pts.append(p)
    return {
        "contract": "kernel_extract/kernel_ranking/v1",
        "denominators": {
            "basis": "observed instance count x per-instance census, summed over phases",
            "total_device_us": round(grand, 1),
            "phase_total_us": {p: round(v, 1) for p, v in phase_total.items()},
            "excluded_from_denominator": excluded,
            "coverage_ledger_points": cov.get("points_plan"),
        },
        "phase_share": {
            "basis": "observed_instances_x_measured_points",
            **{p: round(100.0 * v / grand, 3) for p, v in phase_total.items()},
            "points": pts,
        },
        "warnings": cov.get("warnings", []),
        "ranking": ranking,
        "non_selectable": sorted(nonsel.values(), key=lambda r: -r["mass_us"]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--coverage", required=True, help="coverage_ledger.py --out")
    ap.add_argument("--eager-captured", nargs="*", default=[],
                    help="phases deployed under a CUDA graph but captured eagerly")
    ap.add_argument("--eager-point", action="append", default=[],
                    help="individual eager-captured point; retain graph points' wait mass")
    ap.add_argument("--probe-result", nargs="*", default=[],
                    help="serving_probe.py --out files (globs allowed), for phase_share.points")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    paths = sorted({p for g in a.probe_result for p in (glob.glob(g) or [g])})
    out = build(json.load(open(a.coverage)), set(a.eager_captured), probe_points(paths),
                a.eager_point)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    d = out["denominators"]
    print("%d kernels, denominator %.3f s, phase split %s"
          % (len(out["ranking"]), d["total_device_us"] / 1e6,
             {p: v for p, v in out["phase_share"].items() if p not in ("basis", "points")}))
    for ph, ex in d["excluded_from_denominator"].items():
        print("  %s: %.1f%% of captured mass was eager collective wait, excluded"
              % (ph, ex["pct_of_captured_phase"]))
    for r in out["ranking"][:15]:
        print("  %7.3f%%  %-10s %s" % (r["selection_pct_gpu_time"], r["kind"], r["name"][:80]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
