#!/usr/bin/env python3
"""Observed workload + a capture-time budget -> the sampling points that cover the most
of it, in (shape x context), within that budget.

    python3 plan_points_budget.py --observed evidence/observed_workload.json \
        --time-budget-s 1200 [--phase-weight PREFILL=0.2,DECODE=0.8 | --prior-ranking
        kernel_ranking.json] --out sampling_points.json

Why two dimensions
------------------
A prefill batch's cost is set by the tokens it extends (the M every GEMM and MoE kernel
sees) AND by the context those tokens attend over: on an agentic replay ~96% of prompt
tokens are prefix-cache hits, so a 3k-token extend sits on a 30k-490k-token context, and
attention, the indexer and the KV gather scale with that context. Decode is the same
with batch size for the first axis. Covering only the first axis looked like 99% with 12
prefill points and was 59% in both; every point here is a (shape, context) pair and an
observed instance is covered only when both are within tolerance.

How points are chosen
---------------------
Greedy on covered mass per second: each step adds the candidate (an observed shape and
context) that covers the most still-uncovered, phase-weighted mass per estimated capture
second, until the budget is spent or no candidate adds --min-gain-pct. Mass is prefill
NEW tokens and decode iterations, each as a share of its own phase, weighted by
`--phase-weight` (or the phase split of a prior kernel_ranking.json): they are different
units, and which phase dominates device time is a measurement, not a guess. A point's
cost is a fixed per-window overhead plus building its context (the prefix it warms, or
the decode prompts it holds), at the rates measured on GLM-5.2 TP8.

Prefill points are single sequences (serving_probe.py drives one request per batch):
`new_tokens` is the extend, `cached_tokens` the prefix it sits on. Decode points hold
`batch_size` requests at `context_tokens` each. Every observed instance is assigned to
its nearest point, so `represented` partitions each phase exactly.
Stdlib only.
"""
import argparse
import json
import math
import sys

# Measured on GLM-5.2-MXFP4 TP8 (sglang 0.5.20, probe defaults after the 1022 runs).
WINDOW_OVERHEAD_S = {"PREFILL": 17.0, "DECODE": 12.0}   # 1022c: prefill median 20 s/point
CONTEXT_TOKENS_PER_S = 25000.0


def rows_of(obs):
    """-> {phase: [(shape, context, mass, instances)]} from observed_workload.py."""
    pre = []
    for r in obs.get("raw_prefill_rows", []):
        n = r.get("new_token") or 0
        if n <= 0:
            continue
        seqs = max(1, r.get("new_seq") or 1)
        ctx = ((r.get("cached_token") or 0) + n) / seqs
        pre.append((n, max(1.0, ctx), float(n), 1))
    iv = obs.get("decode_log_interval") or 1
    dec = []
    for r in obs.get("raw_decode_rows", []):
        bs = r.get("running") or 0
        if bs <= 0 or not r.get("token"):
            continue
        it = r.get("iterations") or iv
        dec.append((bs, max(1.0, r["token"] / bs), float(it), it))
    return {"PREFILL": pre, "DECODE": dec}


def dist(a, b, tol, tol_ctx):
    """Normalised 2-D distance; <= 1 means both axes are within tolerance."""
    return max(abs(math.log(a[0] / b[0])) / math.log(tol),
               abs(math.log(a[1] / b[1])) / math.log(tol_ctx))


def cost_s(phase, shape, ctx):
    if phase == "PREFILL":
        return WINDOW_OVERHEAD_S[phase] + ctx / CONTEXT_TOKENS_PER_S
    # Decode prompts are shared across points by request index (serving_probe.py), so a
    # point pays roughly for the context of the requests it adds, bounded by its own.
    return WINDOW_OVERHEAD_S[phase] + shape * ctx / CONTEXT_TOKENS_PER_S / 4.0


def plan(rows, weights, budget_s, tol, tol_ctx, min_gain_pct):
    cands, totals = [], {}
    for phase, rs in rows.items():
        totals[phase] = sum(r[2] for r in rs) or 1.0
        seen = set()
        for r in rs:
            key = (phase, r[0], round(r[1], -2))
            if key in seen:
                continue
            seen.add(key)
            c = (r[0], round(r[1], -2) or 1.0)
            cover = [i for i, x in enumerate(rs) if dist(x, c, tol, tol_ctx) <= 1.0]
            cands.append({"phase": phase, "shape": c[0], "context": c[1], "cover": cover,
                          "cost_s": cost_s(phase, c[0], c[1])})
    left = {p: [True] * len(rs) for p, rs in rows.items()}
    chosen, spent = [], 0.0
    while True:
        best, best_rate, best_gain = None, 0.0, 0.0
        for c in cands:
            if c.get("taken") or spent + c["cost_s"] > budget_s:
                continue
            rs, lf = rows[c["phase"]], left[c["phase"]]
            gain = sum(rs[i][2] for i in c["cover"] if lf[i]) / totals[c["phase"]]
            gain *= weights.get(c["phase"], 0.0)
            if gain / c["cost_s"] > best_rate:
                best, best_rate, best_gain = c, gain / c["cost_s"], gain
        if best is None or 100.0 * best_gain < min_gain_pct:
            break
        best["taken"] = True
        best["gain_pct"] = round(100.0 * best_gain, 3)
        chosen.append(best)
        spent += best["cost_s"]
        for i in best["cover"]:
            left[best["phase"]][i] = False
    return chosen, spent, left, totals


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--observed", required=True)
    ap.add_argument("--time-budget-s", type=float, default=1200.0,
                    help="capture time for all points (the 30-minute profile leaves ~20 "
                         "minutes after server start, evidence window and post-processing)")
    ap.add_argument("--shape-tolerance", type=float, default=1.25)
    ap.add_argument("--context-tolerance", type=float, default=1.5)
    ap.add_argument("--phase-weight", default=None, help="e.g. PREFILL=0.2,DECODE=0.8")
    ap.add_argument("--prior-ranking", default=None,
                    help="kernel_ranking.json of an earlier round: its phase_share is the weight")
    ap.add_argument("--min-gain-pct", type=float, default=0.05,
                    help="stop when the best candidate adds less weighted coverage than this")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    obs = json.load(open(a.observed))
    rows = rows_of(obs)
    if a.phase_weight:
        weights = {k.strip().upper(): float(v) for k, v in
                   (p.split("=") for p in a.phase_weight.split(","))}
        basis = "explicit --phase-weight"
    elif a.prior_ranking:
        ps = json.load(open(a.prior_ranking)).get("phase_share") or {}
        weights = {p: float(ps[p]) / 100.0 for p in rows if p in ps}
        basis = "phase_share of %s" % a.prior_ranking
    else:
        weights = {p: 0.5 for p in rows}
        basis = ("EVEN: no device-time split known yet; pass --prior-ranking or "
                 "--phase-weight after a first census")
    chosen, spent, left, totals = plan(rows, weights, a.time_budget_s, a.shape_tolerance,
                                       a.context_tolerance, a.min_gain_pct)

    points, summary = [], {}
    for phase, rs in rows.items():
        mine = [c for c in chosen if c["phase"] == phase]
        acc = {id(c): {"instances": 0, "mass": 0.0, "max": 1.0} for c in mine}
        for r in rs:
            if not mine:
                break
            c = min(mine, key=lambda c: dist(r, (c["shape"], c["context"]),
                                             a.shape_tolerance, a.context_tolerance))
            e = acc[id(c)]
            e["instances"] += r[3]
            e["mass"] += r[2]
            e["max"] = max(e["max"], math.exp(abs(math.log(r[0] / c["shape"]))),
                           math.exp(abs(math.log(r[1] / c["context"]))))
        for c in sorted(mine, key=lambda c: (c["shape"], c["context"])):
            e = acc[id(c)]
            ctx = int(c["context"])
            pt = {"phase": phase, "grounds": "coverage_greedy",
                  "est_capture_s": round(c["cost_s"], 1),
                  "marginal_weighted_coverage_pct": c["gain_pct"],
                  "represented": {"instances": e["instances"], "proxy_mass": round(e["mass"], 1),
                                  "phase_proxy_mass_share_pct":
                                      round(100.0 * e["mass"] / totals[phase], 3),
                                  "max_shape_ratio": round(e["max"], 3)}}
            if phase == "PREFILL":
                pt.update(point_id="pr_%d_c%dk" % (c["shape"], round(ctx / 1000)),
                          new_tokens=int(c["shape"]), cached_tokens=max(0, ctx - int(c["shape"])),
                          running=1, context_tokens=ctx)
            else:
                pt.update(point_id="de_%d_c%dk" % (c["shape"], round(ctx / 1000)),
                          batch_size=int(c["shape"]), context_tokens=ctx)
            points.append(pt)
        out_mass = sum(r[2] for r, l in zip(rs, left[phase]) if l)
        summary[phase] = {"points": len(mine), "weight": weights.get(phase, 0.0),
                          "mass_outside_tolerance_pct": round(100.0 * out_mass / totals[phase], 2)}
    doc = {"contract": "kernel_extract/sampling_points/v2", "planner": "coverage_greedy",
           "observed_workload": a.observed, "time_budget_s": a.time_budget_s,
           "est_capture_s": round(spent, 1), "shape_tolerance": a.shape_tolerance,
           "context_tolerance": a.context_tolerance, "phase_weight": weights,
           "phase_weight_basis": basis, "phase_summary": summary,
           "cost_model": {"window_overhead_s": WINDOW_OVERHEAD_S,
                          "context_tokens_per_s": CONTEXT_TOKENS_PER_S},
           "points": points}
    with open(a.out, "w") as fh:
        json.dump(doc, fh, indent=1)
    print("%d points, est. %.0f s of %.0f s budget (weights %s)"
          % (len(points), spent, a.time_budget_s, weights))
    for ph, s in summary.items():
        print("  %-8s %3d points, %5.2f%% of phase mass outside %.2fx shape / %.2fx context"
              % (ph, s["points"], s["mass_outside_tolerance_pct"], a.shape_tolerance,
                 a.context_tolerance))
    print("->", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
