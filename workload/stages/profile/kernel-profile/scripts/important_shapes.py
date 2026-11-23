#!/usr/bin/env python3
"""Observed shapes worth a UT case, ranked by estimated device time, not by proxy.

    python3 important_shapes.py --plan sampling_plan.json --coverage coverage.json \
        [--compile-variants evidence/compile_variants.json] \
        --out evidence/important_shapes.json [--target-pct 90] [--max-cells N]

Every observed cell of the plan (captured or not) gets an estimated time: its instances x
the measured per-instance device time of the nearest captured point of its phase (log
distance in shape and context; prefill scaled by new tokens). A phase with no measured
point falls back to the proxy, and says so.

Prefill shapes are nearly all unique (824 distinct new-token counts in 1341 batches of an
agentic replay), so prefill cells are merged into bands first: new tokens within a factor
1.25, never across a power of two (Triton block sizes and autotune keys, aiter padded-M
tiers), the prefill chunk, or a token count at which the server compiled another variant
of one of its kernels (compile_variants.json); context within a factor 2. A band is
represented by its heaviest observed cell. Decode cells stay as planned (exact batch).

Per phase, cells are taken heaviest first until `--target-pct` of the phase's estimated
time or the plan's `important_cells` (`--max-cells`), plus the smallest/largest shape and
context cells (boundaries: correctness only). Each cell carries its launch as observed in
the serving log: graph or eager, and for graph the capture size the scheduler pads to.

`calibration` lists each measured point's time per proxy unit (prefill new token, decode
iteration): a wide range is how much the proxy alone would have misranked. Stdlib only.
"""
import argparse
import bisect
import json
import math
import sys


def nearest(points, shape, context):
    return min(points, key=lambda p: abs(math.log(shape / p["shape"]))
               + abs(math.log(context / p["context"])))


def capture_bs(shape, graph_bs):
    return next((b for b in graph_bs if b >= shape), None)


def variant_cuts(variants):
    """Token counts at which an eagerly launched kernel first compiled a variant, for every
    kernel that compiled more than one: its first tensor argument's leading dimension. A
    spurious cut (a weight dimension) only makes bands finer."""
    rows = {}
    for v in (variants or {}).get("variants", []):
        if v.get("in_graph_capture"):
            continue
        for f in (v.get("frames") or [])[:1]:
            t = next((x for x in (f.get("args") or {}).values()
                      if isinstance(x, dict) and x.get("__tensor__") and x.get("shape")), None)
            if t:
                rows.setdefault(v.get("fn"), set()).add(int(t["shape"][0]))
    return sorted({r for s in rows.values() if len(s) > 1 for r in s})


def band(c, chunk, cuts):
    n, ctx = c["partition"][1], c["point"]["context_tokens"]
    return (c["partition"][2], n == chunk, math.ceil(math.log2(n)), math.floor(math.log(n, 1.25)),
            math.floor(math.log2(ctx)), bisect.bisect_right(cuts, n))


def merge(rows, key):
    """Groups of cells under one key, represented by the heaviest."""
    groups = {}
    for c in rows:
        groups.setdefault(key(c), []).append(c)
    out = []
    for members in groups.values():
        g = dict(max(members, key=lambda c: c["est"]))
        for k in ("est", "instances", "proxy_mass"):
            g[k] = sum(c[k] for c in members)
        shapes = [c["partition"][1] for c in members]
        ctxs = [c["point"]["context_tokens"] for c in members]
        g.update(members=len(members), lo=min(shapes), hi=max(shapes),
                 ctx_lo=min(ctxs), ctx_hi=max(ctxs))
        out.append(g)
    return out


def rank(plan, coverage, target_pct=90.0, max_cells=None, variants=None):
    max_cells = max_cells or plan.get("important_cells") or 24
    cells = plan.get("captures", []) + plan.get("reused", []) + plan.get("unselected", [])
    graph_bs = (plan.get("boundaries") or {}).get("graph_bs") or []
    chunk, cuts = (plan.get("boundaries") or {}).get("prefill_chunk"), variant_cuts(variants)
    out, calibration = [], {}
    for phase in sorted({c["partition"][0] for c in cells}):
        rows = [c for c in cells if c["partition"][0] == phase]
        pts = [p for p in (coverage.get("phases", {}).get(phase) or {}).get("points", [])
               if p.get("us_per_instance") and p.get("shape") and p.get("context")]
        for c in rows:
            pt, shape = c["point"], c["partition"][1]
            if pts:
                q = nearest(pts, shape, pt["context_tokens"])
                scale = shape / q["shape"] if phase == "PREFILL" else 1.0
                c["est"] = c["instances"] * q["us_per_instance"] * scale
            else:
                c["est"] = c["proxy_mass"]
        rows = merge(rows, (lambda c: band(c, chunk, cuts)) if phase == "PREFILL"
                     else (lambda c: c["point"]["point_id"]))
        total = sum(c["est"] for c in rows) or 1.0
        proxy = sum(c["proxy_mass"] for c in rows) or 1.0
        chosen, acc = {}, 0.0
        for c in sorted(rows, key=lambda c: -c["est"]):
            if acc >= target_pct / 100 * total or len(chosen) >= max_cells:
                break
            chosen[c["point"]["point_id"]] = ["time"]
            acc += c["est"]
        for lo, hi in (("lo", "hi"), ("ctx_lo", "ctx_hi")):
            for c in (min(rows, key=lambda c: c[lo]), max(rows, key=lambda c: c[hi])):
                chosen.setdefault(c["point"]["point_id"], []).append("boundary")
        for c in rows:
            pt = c["point"]
            why = chosen.get(pt["point_id"])
            if not why:
                continue
            graph = bool(c["partition"][3])
            launch = {"mode": "graph" if graph else "eager"}
            if graph:
                launch["capture_bs"] = capture_bs(c["partition"][1], graph_bs)
            out.append({"id": pt["point_id"], "phase": phase, "shape": c["partition"][1],
                        "sequences": c["partition"][2], "context": pt["context_tokens"],
                        "cached_tokens": pt.get("cached_tokens"), "instances": c["instances"],
                        "time_share_pct": round(100 * c["est"] / total, 4),
                        "proxy_share_pct": round(100 * c["proxy_mass"] / proxy, 4),
                        "why": sorted(set(why)), "launch": launch})
            if c["members"] > 1:
                out[-1].update(members=c["members"], shape_range=[c["lo"], c["hi"]],
                               context_range=[c["ctx_lo"], c["ctx_hi"]])
        calibration[phase] = {
            "basis": "measured" if pts else "proxy (no measured point)",
            "time_covered_pct": round(100 * acc / total, 3),
            "capped": acc < target_pct / 100 * total,
            "points": [{"point_id": p["point_id"], "shape": p["shape"], "context": p["context"],
                        "us_per_proxy_unit": round(p["us_per_instance"]
                                                   / (p["shape"] if phase == "PREFILL" else 1), 3)}
                       for p in pts]}
    if not graph_bs and any(c["launch"].get("capture_bs") is None and c["launch"]["mode"] == "graph"
                            for c in out):
        calibration["note"] = ("graph capture sizes unknown (no server_info): capture_bs is "
                               "null; the UT must use the padded size the server replays")
    return {"contract": "kernel_profile/important_shapes/v1", "target_pct": target_pct,
            "max_cells": max_cells,
            "graph_bs": graph_bs, "cells": out, "calibration": calibration}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plan", required=True, help="sampling_plan.json")
    ap.add_argument("--coverage", required=True, help="coverage.json from coverage_ledger.py")
    ap.add_argument("--compile-variants", help="the server's compile_variants.json (prefill band cuts)")
    ap.add_argument("--target-pct", type=float, default=90.0)
    ap.add_argument("--max-cells", type=int, default=None,
                    help="per phase; default the plan's important_cells (plan_points.py "
                         "--important-cells, 24). Every cell becomes a correctness case per stream "
                         "of each task: 66 prefill cells made 330 attention cases and an "
                         "unfinishable correctness gate (`capped` says the target was not met)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    variants = json.load(open(a.compile_variants)) if a.compile_variants else None
    doc = rank(json.load(open(a.plan)), json.load(open(a.coverage)), a.target_pct, a.max_cells, variants)
    with open(a.out, "w") as f:
        json.dump(doc, f, indent=1)
    for phase, c in sorted(doc["calibration"].items()):
        if isinstance(c, dict):
            n = sum(x["phase"] == phase for x in doc["cells"])
            print("%s: %d important cell(s), %.1f%% of estimated time (%s)%s"
                  % (phase, n, c["time_covered_pct"], c["basis"],
                     "; capped at %d cells" % doc["max_cells"] if c["capped"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
