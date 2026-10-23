#!/usr/bin/env python3
"""Each task's UT case plan, from the measured points and the bands they do not cover.

A task's manifest spans the served distribution, not one point in it. Three kinds of
case, each labelled:

  point     one per counted sampling point of a phase the task owns where its kernels
            launched. effective_calls = launches per instance of the task's kernels at
            that point (kernel_shapes.py) x the instances the point represents.
            coverage_ledger.py credits a point's census mass only through such a case.
  bucket    an unrepresented band from relabel_points.py (observed weight, no point
            within tolerance) whose share of its phase is at least --min-band-share, up
            to --max-buckets per phase. Its launches per instance are the mean of its
            bracketing points (`cost_basis: interpolated`). Its instances are SUBTRACTED
            from the point that represented them, so no instance is weighted twice.
  envelope  a dispatch boundary / graph-size probe the task declares itself
            (task_decls `envelope_cases`), effective_calls 0: tested, never weighted.

    python3 plan_ut_cases.py --points sampling_points_final.json --shapes-dir evidence/shapes \
        --decls task_decls.json --out evidence/ut_case_plan.json
Stdlib only.
"""
import argparse
import json
import os
import sys
from collections import defaultdict


def lpi(shapes, kernels):
    """Launches per instance of `kernels` in one shape manifest (None if no manifest)."""
    if shapes is None:
        return None
    return round(sum(shapes["kernels"][k]["launches_per_instance"]
                     for k in kernels if k in shapes["kernels"]), 4)


def context_of(p):
    """Per-request context of a point: decode's context_tokens, prefill's cached + new
    tokens per sequence. What attention / indexer / cache axes scale with."""
    if p["phase"] == "DECODE":
        return p.get("context_tokens")
    if p.get("achieved_context"):
        return int(round(p["achieved_context"]))
    n = max(1, int(p.get("running") or 1))
    return int(round(((p.get("cached_tokens") or 0) + (p.get("new_tokens") or 0)) / n))


def plan(points_final, shapes_dir, tasks, min_band_share=0.5, max_buckets=6):
    pts = {p["point_id"]: p for p in points_final["points"] if p.get("counted")}
    shapes = {}
    for pid in pts:
        f = os.path.join(shapes_dir, pid + ".json")
        shapes[pid] = json.load(open(f)) if os.path.exists(f) else None
    summary = points_final.get("phase_summary", {})
    out = {}
    for t in tasks:
        label, kernels = t["label"], t["kernels"]
        phases = t.get("phases") or ["PREFILL", "DECODE"]
        cases, residual = [], {}
        for phase in phases:
            owned = [p for p in pts.values() if p["phase"] == phase
                     and (lpi(shapes[p["point_id"]], kernels) or 0) > 0]
            if not owned:
                continue
            owned_ids = {p["point_id"] for p in owned}
            bands = [b for b in summary.get(phase, {}).get("unrepresented_bands", [])
                     if b.get("proxy_mass_share_of_phase_pct", 0) >= min_band_share
                     and b["nearest_point"] in owned_ids][:max_buckets]
            moved = defaultdict(int)
            for b in bands:
                moved[b["nearest_point"]] += b["instances"]
            for p in sorted(owned, key=lambda p: p["achieved_shape"]):
                n = lpi(shapes[p["point_id"]], kernels)
                inst = p["represented"]["instances"] - moved[p["point_id"]]
                cases.append({
                    "case_id": "%s__%s" % (label, p["point_id"]), "kind": "point",
                    "point_id": p["point_id"], "phase": phase, "shape": p["achieved_shape"],
                    "context": context_of(p),
                    "launches_per_instance": n, "instances_represented": inst,
                    "effective_calls": int(round(n * inst)), "cost_basis": "measured",
                    "census": p.get("census"),
                    "shape_manifest": os.path.abspath(os.path.join(shapes_dir,
                                                                   p["point_id"] + ".json"))})
            for b in bands:
                ref = [r for r in b["bracketing_points"] if r in owned_ids]
                ns = [lpi(shapes[r], kernels) for r in ref]
                n = round(sum(ns) / len(ns), 4) if ns else 0.0
                ctx = ([b["context"]] if b.get("context") else
                       [context_of(pts[r]) for r in ref if context_of(pts[r])])
                cases.append({
                    "case_id": "%s__%s_bucket_%d%s" % (label, phase.lower(), b["shape"],
                                                       "_c%dk" % round(b["context"] / 1000)
                                                       if b.get("context") else ""),
                    "kind": "bucket", "phase": phase, "shape": b["shape"],
                    "range": [b["lo"], b["hi"]], "cost_reference_points": ref,
                    "context": int(round(sum(ctx) / len(ctx))) if ctx else None,
                    "launches_per_instance": n, "instances_represented": b["instances"],
                    "effective_calls": int(round(n * b["instances"])),
                    "cost_basis": "interpolated",
                    "why": "%.2f%% of %s proxy mass sits outside every point's tolerance here"
                           % (b["proxy_mass_share_of_phase_pct"], phase)})
            left = summary.get(phase, {}).get("proxy_mass_outside_tolerance_pct", 0.0) - sum(
                b["proxy_mass_share_of_phase_pct"] for b in bands)
            residual[phase] = round(max(0.0, left), 3)
        for e in t.get("envelope_cases") or []:
            c = dict(e, kind="envelope", effective_calls=0, instances_represented=0,
                     cost_basis="not_weighted")
            c.setdefault("case_id", "%s__%s_envelope_%s" % (label, e["phase"].lower(), e["shape"]))
            cases.append(c)
        out[label] = {"cases": cases,
                      "proxy_mass_outside_every_case_pct": residual}
    return {"contract": "kernel_extract/ut_case_plan/v1",
            "shape_tolerance": points_final.get("shape_tolerance"),
            "rules": {"min_band_share_pct": min_band_share, "max_buckets_per_phase": max_buckets},
            "tasks": out}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--points", required=True, help="relabel_points.py output")
    ap.add_argument("--shapes-dir", required=True)
    ap.add_argument("--decls", required=True, help="task_decls.json")
    ap.add_argument("--min-band-share", type=float, default=0.5)
    ap.add_argument("--max-buckets", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    decls = json.load(open(a.decls))
    out = plan(json.load(open(a.points)), a.shapes_dir, decls["tasks"],
               a.min_band_share, a.max_buckets)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    for label, t in out["tasks"].items():
        kinds = defaultdict(int)
        for c in t["cases"]:
            kinds[c["kind"]] += 1
        print("  %-36s %s  outside every case: %s" % (
            label, " ".join("%s:%d" % kv for kv in sorted(kinds.items())),
            t["proxy_mass_outside_every_case_pct"]))
    print("->", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
