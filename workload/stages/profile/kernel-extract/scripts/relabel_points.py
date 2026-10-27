#!/usr/bin/env python3
"""Relabel every sampling point to the shape its probe ACHIEVED and re-partition the
observed workload over the achieved shapes.

A probe rarely executes exactly the planned shape: the scheduler admits a multi-sequence
batch one request at a time, a cached prefix re-extends its page-size residue, a decode
batch pads to a graph size. A `partial` point is still a real measurement, but of the
shape it achieved, so before the ledger counts it:

  * new_tokens / batch_size  := the census's achieved shape (planned kept as planned_shape)
  * represented.instances    := every observed instance (one prefill batch row; one decode
                                iteration = log row x decode_log_interval) assigned to the
                                nearest counted point in log-shape, so the points partition
                                each phase: nothing dropped, nothing counted twice
  * unrepresented_bands      := instances farther than --tolerance from every point, grouped
                                into bands (they still count toward their nearest point,
                                and plan_ut_cases.py turns each band into a bucket case)

A point without a census is kept with `counted: false` and represents nothing.

    python3 relabel_points.py --observed evidence/observed_workload.json \
        --points sampling_points.json --census-dir evidence/census \
        --probe-dir evidence/probe --out sampling_points_final.json
Stdlib only.
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict


def observed_rows(obs, phase):
    """-> [(shape, instances, context)] from observed_workload.py's raw rows. Context is
    per sequence: a prefill batch's cached + new tokens over its sequences, a decode
    row's tokens over its running requests."""
    if phase == "PREFILL":
        return [(r["new_token"], 1,
                 max(1.0, ((r.get("cached_token") or 0) + r["new_token"]) / max(1, r.get("new_seq") or 1)))
                for r in obs.get("raw_prefill_rows", []) if r.get("new_token")]
    iv = obs.get("decode_log_interval") or 1
    return [(r["running"], r.get("iterations") or iv,
             max(1.0, (r.get("token") or r["running"]) / r["running"]))
            for r in obs.get("raw_decode_rows", []) if r.get("running")]


def point_context(p):
    if p["phase"] == "PREFILL":
        if p.get("context_tokens"):
            return float(p["context_tokens"])
        n = max(1, int(p.get("running") or 1))
        return max(1.0, ((p.get("cached_tokens") or 0) + (p.get("new_tokens") or 0)) / n)
    return float(p.get("context_tokens") or 1.0)


def bands_2d(rows, tol, tol_ctx, phase, live):
    """Outside rows (shape, n, nearest, ctx) grouped on a grid of one tolerance per axis."""
    import math as _m
    cells = defaultdict(list)
    for shape, n, near, ctx in rows:
        cells[(int(_m.log(shape) // _m.log(tol)),
               int(_m.log(ctx) // _m.log(tol_ctx)))].append((shape, n, near, ctx))
    out = []
    for rs in cells.values():
        tot = sum(n for _s, n, _p, _c in rs)
        shape = int(round(_m.exp(sum(n * _m.log(s) for s, n, _p, _c in rs) / tot)))
        ctx = int(round(_m.exp(sum(n * _m.log(c) for _s, n, _p, c in rs) / tot)))
        near = sorted(live, key=lambda p: max(
            abs(_m.log(shape / p["achieved_shape"])) / _m.log(tol),
            abs(_m.log(ctx / p["achieved_context"])) / _m.log(tol_ctx)))
        out.append({"lo": min(s for s, *_ in rs), "hi": max(s for s, *_ in rs),
                    "shape": shape, "context": ctx, "instances": tot,
                    "nearest_point": near[0]["point_id"],
                    "bracketing_points_2d": [p["point_id"] for p in near[:2]],
                    "proxy_mass": sum((s if phase == "PREFILL" else 1) * n for s, n, _p, _c in rs)})
    return out


def bands(rows, tol, phase):
    """Group (shape, n, nearest point) rows into contiguous bands no wider than tol.

    `proxy_mass` ranks them: new tokens for prefill (cost grows with the batch's tokens),
    iterations for decode (one iteration is one forward whatever the batch)."""
    out = []
    for shape, n, near in sorted(rows):
        b = out[-1] if out else None
        if b and shape <= b["lo"] * tol and near == b["nearest_point"]:
            b["hi"] = shape
            b["instances"] += n
            b["_shapes"].append((shape, n))
        else:
            out.append({"lo": shape, "hi": shape, "instances": n, "nearest_point": near,
                        "_shapes": [(shape, n)]})
    for b in out:
        # instance-weighted geometric centre: the case shape that stands for the band
        tot = sum(n for _s, n in b["_shapes"])
        b["proxy_mass"] = sum((s if phase == "PREFILL" else 1) * n for s, n in b["_shapes"])
        b["shape"] = int(round(math.exp(sum(n * math.log(s) for s, n in b.pop("_shapes")) / tot)))
    return out


def relabel(obs, plan, census_dir, probe_dir, tol, tol_ctx=None):
    """tol_ctx (from a v2 plan's context_tolerance) makes the assignment two-dimensional:
    an instance is near a point only when its shape AND its context are."""
    pts = []
    for p in plan["points"]:
        p = dict(p)
        cpath = os.path.join(census_dir, p["point_id"] + ".json")
        if not os.path.exists(cpath):
            p.update(counted=False, counted_reason="no census: no usable window was captured")
            p["represented"] = dict(p.get("represented") or {}, instances=0)
            pts.append(p)
            continue
        c = json.load(open(cpath))
        key = "new_tokens" if p["phase"] == "PREFILL" else "batch_size"
        p["planned_shape"] = p.get(key)
        p["achieved_shape"] = c["achieved_shape"]
        p[key] = c["achieved_shape"]
        if p["planned_shape"]:
            p["achieved_over_planned"] = round(c["achieved_shape"] / p["planned_shape"], 4)
        p["achieved_context"] = point_context(p)
        p["census"] = os.path.abspath(cpath)
        p["census_instances"] = c["instances"]
        for tag in ("_r2", ""):
            jp = os.path.join(probe_dir, p["point_id"] + tag + ".json")
            if os.path.exists(jp):
                j = json.load(open(jp))
                p["probe_status"], p["probe_result"] = j.get("status"), os.path.abspath(jp)
                break
        p["counted"] = True
        pts.append(p)

    summary = {}
    for phase in ("PREFILL", "DECODE"):
        live = [p for p in pts if p["phase"] == phase and p.get("counted")]
        rows = observed_rows(obs, phase)
        if not live:
            if rows:
                summary[phase] = {"total_instances": sum(r[1] for r in rows),
                                  "note": "no counted point: this phase has no measured cost"}
            continue
        acc = {p["point_id"]: {"instances": 0, "max_ratio": 1.0} for p in live}
        far, total = [], 0
        lt, lc = math.log(tol), math.log(tol_ctx) if tol_ctx else None
        for shape, n, ctx in rows:
            if lc:
                d = lambda p: max(abs(math.log(shape / p["achieved_shape"])) / lt,  # noqa: E731
                                  abs(math.log(ctx / p["achieved_context"])) / lc)
            else:
                d = lambda p: abs(math.log(shape / p["achieved_shape"])) / lt  # noqa: E731
            best = min(live, key=d)
            ratio = math.exp(abs(math.log(shape / best["achieved_shape"])))
            if lc:
                ratio = max(ratio, math.exp(abs(math.log(ctx / best["achieved_context"]))))
            e = acc[best["point_id"]]
            e["instances"] += n
            e["max_ratio"] = max(e["max_ratio"], ratio)
            total += n
            if d(best) > 1.0:
                far.append((shape, n, best["point_id"], ctx))
        for p in live:
            e = acc[p["point_id"]]
            p["represented"] = {
                "instances": e["instances"],
                "instance_share_of_phase_pct": round(100.0 * e["instances"] / total, 4)
                if total else 0.0,
                "max_shape_ratio": round(e["max_ratio"], 3),
                "basis": "one prefill batch row" if phase == "PREFILL"
                         else "one decode iteration (log rows x decode_log_interval)"}
        ub = (bands_2d(far, tol, tol_ctx, phase, live) if lc
              else bands([(s, n, pid) for s, n, pid, _c in far], tol, phase))
        mass = sum((s if phase == "PREFILL" else 1) * n for s, n, _c in rows)
        by_shape = sorted(live, key=lambda p: p["achieved_shape"])
        for b in ub:
            below = [p["point_id"] for p in by_shape if p["achieved_shape"] <= b["shape"]]
            above = [p["point_id"] for p in by_shape if p["achieved_shape"] > b["shape"]]
            b["bracketing_points"] = b.pop("bracketing_points_2d", None) or (
                ([below[-1]] if below else []) + ([above[0]] if above else []))
            b["instance_share_of_phase_pct"] = round(100.0 * b["instances"] / total, 4)
            b["proxy_mass_share_of_phase_pct"] = round(100.0 * b["proxy_mass"] / mass, 4)
        out_n = sum(n for _s, n, _p, _c in far)
        summary[phase] = {
            "points": [p["point_id"] for p in live],
            "total_instances": total,
            "instances_outside_tolerance": out_n,
            "instances_outside_tolerance_pct": round(100.0 * out_n / total, 3) if total else 0.0,
            "proxy_mass_outside_tolerance_pct": round(
                100.0 * sum(b["proxy_mass"] for b in ub) / mass, 3) if mass else 0.0,
            "unrepresented_bands": sorted(ub, key=lambda b: -b["proxy_mass"]),
        }
    return {"contract": "kernel_extract/sampling_points_final/v1",
            "derived_from": {"points": plan.get("_path"), "census_dir": os.path.abspath(census_dir)},
            "shape_tolerance": tol, "context_tolerance": tol_ctx,
            "phase_summary": summary, "points": pts}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--observed", required=True)
    ap.add_argument("--points", required=True)
    ap.add_argument("--census-dir", required=True)
    ap.add_argument("--probe-dir", required=True)
    ap.add_argument("--tolerance", type=float, default=1.25)
    ap.add_argument("--context-tolerance", type=float, default=None,
                    help="also require the context within this factor (default: the plan's "
                         "context_tolerance; one-dimensional when neither is set)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    plan = json.load(open(a.points))
    plan["_path"] = os.path.abspath(a.points)
    out = relabel(json.load(open(a.observed)), plan, a.census_dir, a.probe_dir, a.tolerance,
                  a.context_tolerance or plan.get("context_tolerance"))
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    for p in out["points"]:
        if p.get("counted") and p.get("achieved_over_planned", 1) != 1:
            print("  %s relabelled %s -> %s" % (p["point_id"], p["planned_shape"],
                                                p["achieved_shape"]))
    for ph, s in out["phase_summary"].items():
        if "instances_outside_tolerance_pct" in s:
            print("%s: %d instances, %.2f%% (%.2f%% of proxy mass) outside tolerance of "
                  "every point, %d band(s)"
                  % (ph, s["total_instances"], s["instances_outside_tolerance_pct"],
                     s["proxy_mass_outside_tolerance_pct"], len(s["unrepresented_bands"])))
    print("->", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
