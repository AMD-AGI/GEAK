#!/usr/bin/env python3
"""Each task's UT case plan, from the measured points and the bands they do not cover.

A task's manifest spans the served distribution, not one point in it. Three kinds of
case, each labelled:

  point     one per counted sampling point of a phase the task owns where its kernels or
            companions launched. effective_calls = complete op invocations per instance
            x the instances the point represents. A task may declare invocation_anchors:
            {"kernels": ["one_leaf_per_call"], "why": "source-backed exactly-once reason"}.
            Multiple anchors require mutually_exclusive=true: they are alternative leaves,
            exactly one per call, whose observed counts are summed (never max). Without
            anchors, the legacy sum of all selected/companion leaf launches is retained
            and reported explicitly as invocation_count_basis.
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
import math
import os
import sys
from collections import defaultdict


def lpi(shapes, kernels):
    """Launches per instance of `kernels` in one shape manifest (None if no manifest)."""
    if shapes is None:
        return None
    return round(sum(shapes["kernels"][k]["launches_per_instance"]
                     for k in kernels if k in shapes["kernels"]), 4)


def invocation_basis(task):
    """Validate an explicit exactly-once anchor contract, or report legacy counting."""
    members = list(task.get("kernels") or []) + [c["kernel"] for c in task.get("companions") or []
                                               if c.get("kernel")]
    if "invocation_anchors" not in task:
        return {"kind": "legacy_leaf_launch_sum", "kernels": members}
    anchor = task["invocation_anchors"]
    prefix = "%s: invocation_anchors" % task.get("label", "task")
    if not isinstance(anchor, dict):
        raise ValueError(prefix + " must be an object with kernels and why")
    kernels = anchor.get("kernels")
    if (not isinstance(kernels, list) or not kernels
            or any(not isinstance(k, str) or not k for k in kernels)
            or len(set(kernels)) != len(kernels)):
        raise ValueError(prefix + ".kernels must be a nonempty list of distinct kernel identities")
    why = anchor.get("why")
    if not isinstance(why, str) or not why.strip():
        raise ValueError(prefix + " needs a source-backed exactly-once why")
    outside = set(kernels) - set(members)
    if outside:
        raise ValueError(prefix + " must belong to selected kernels or companions: " + ", ".join(sorted(outside)))
    if len(kernels) > 1 and anchor.get("mutually_exclusive") is not True:
        raise ValueError(prefix + " with multiple kernels requires mutually_exclusive=true and source-backed why")
    return dict(anchor, kind="anchor_launch_sum")


def invocation_lpi(shapes, basis, label, point_id):
    """One count per full call. Missing anchors cannot silently erase an owned point."""
    if basis["kind"] == "legacy_leaf_launch_sum":
        return lpi(shapes, basis["kernels"])
    counts = []
    for kernel in basis["kernels"]:
        item = (shapes or {}).get("kernels", {}).get(kernel)
        if item is None:
            continue
        n = item.get("launches_per_instance")
        if isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n) or n < 0:
            raise ValueError("%s/%s: invalid invocation anchor count for %s" % (label, point_id, kernel))
        counts.append(n)
    n = round(sum(counts), 4)
    if n <= 0:
        raise ValueError("%s/%s: task kernels or companions launched, but no declared invocation anchor "
                         "launched; cannot infer complete op calls" % (label, point_id))
    return n


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
    """Single-config: as below. Multi-config (`points_final["configs"]`, written by the profile
    slot's multi_config.py merge): each config is planned on its own points and its own
    phase_summary, and every case carries that config's `config_id` -- the UT normalizes call
    mass per config before pooling (harness_lib._config_weights), so the configs never need a
    common instance scale here."""
    configs = points_final.get("configs")
    if not configs:
        return _plan_one(points_final, shapes_dir, tasks, min_band_share, max_buckets)
    out = {"contract": "kernel_extract/ut_case_plan/v1",
           "shape_tolerance": points_final.get("shape_tolerance"),
           "rules": {"min_band_share_pct": min_band_share, "max_buckets_per_phase": max_buckets,
                     "per_config": True},
           "configs": sorted(configs), "tasks": {}}
    for cid in sorted(configs):
        sub = {"points": [p for p in points_final["points"] if p.get("config_id") == cid],
               "phase_summary": configs[cid].get("phase_summary") or {},
               "shape_tolerance": points_final.get("shape_tolerance")}
        one = _plan_one(sub, shapes_dir, tasks, min_band_share, max_buckets)
        for label, t in one["tasks"].items():
            dest = out["tasks"].setdefault(label, {"cases": [], "proxy_mass_outside_every_case_pct": {},
                "invocation_anchors": t["invocation_anchors"],
                "invocation_count_basis": t["invocation_count_basis"]})
            for c in t["cases"]:
                if c["kind"] == "envelope":
                    # declared by the task, not measured: test it once, never weight it
                    if cid == sorted(configs)[0]:
                        dest["cases"].append(c)
                    continue
                c["config_id"] = cid
                if c["kind"] != "point":   # point ids are already config-prefixed by the merge
                    c["case_id"] = c["case_id"].replace(label + "__", "%s__%s__" % (label, cid), 1)
                dest["cases"].append(c)
            dest["proxy_mass_outside_every_case_pct"][cid] = t["proxy_mass_outside_every_case_pct"]
    return out


def _plan_one(points_final, shapes_dir, tasks, min_band_share=0.5, max_buckets=6):
    pts = {p["point_id"]: p for p in points_final["points"] if p.get("counted")}
    shapes = {}
    for pid in pts:
        f = os.path.join(shapes_dir, pid + ".json")
        shapes[pid] = json.load(open(f)) if os.path.exists(f) else None
    summary = points_final.get("phase_summary", {})
    out = {}
    for t in tasks:
        # Companions are the op's other dispatch branches: a point where only a companion
        # launched (DeepSeek-V4's mHC at prefill >= 8568 tokens runs its large-M branch) is
        # still the task's, or the UT never reaches the branch it is required to cover.
        label = t["label"]
        kernels = list(t["kernels"]) + [c["kernel"] for c in t.get("companions") or []
                                        if c.get("kernel")]
        basis = invocation_basis(t)
        # A shared dependency can run for unrelated operators. With explicit
        # anchors, only selected leaves or exactly-once anchors establish ownership;
        # a companion anchor can still identify another dispatch branch/phase.
        anchored_members = list(t["kernels"]) + basis["kernels"]
        phases = t.get("phases") or ["PREFILL", "DECODE"]
        cases, residual = [], {}
        for phase in phases:
            owned = [p for p in pts.values() if p["phase"] == phase
                     and (any(k in (shapes[p["point_id"]] or {}).get("kernels", {}) for k in anchored_members)
                          if basis["kind"] == "anchor_launch_sum"
                          else (lpi(shapes[p["point_id"]], kernels) or 0) > 0)]
            if not owned:
                continue
            counts = {p["point_id"]: invocation_lpi(shapes[p["point_id"]], basis, label, p["point_id"])
                      for p in owned}
            owned_ids = {p["point_id"] for p in owned}
            bands = [b for b in summary.get(phase, {}).get("unrepresented_bands", [])
                     if b.get("proxy_mass_share_of_phase_pct", 0) >= min_band_share
                     and b["nearest_point"] in owned_ids][:max_buckets]
            moved = defaultdict(int)
            for b in bands:
                moved[b["nearest_point"]] += b["instances"]
            for p in sorted(owned, key=lambda p: p["achieved_shape"]):
                n = counts[p["point_id"]]
                inst = p["represented"]["instances"] - moved[p["point_id"]]
                cases.append({
                    "case_id": "%s__%s" % (label, p["point_id"]), "kind": "point",
                    "point_id": p["point_id"], "phase": phase, "shape": p["achieved_shape"],
                    "context": context_of(p),
                    # Keep the old field as an alias for downstream case transformers.
                    "launches_per_instance": n, "invocations_per_instance": n,
                    "kernel_launches_per_instance": lpi(shapes[p["point_id"]], kernels),
                    "invocation_count_basis": basis, "instances_represented": inst,
                    "effective_calls": int(round(n * inst)), "cost_basis": "measured",
                    "census": p.get("census"),
                    "shape_manifest": os.path.abspath(os.path.join(shapes_dir,
                                                                   p["point_id"] + ".json"))})
            for b in bands:
                ref = [r for r in b["bracketing_points"] if r in owned_ids]
                ns = [counts[r] for r in ref]
                n = round(sum(ns) / len(ns), 4) if ns else 0.0
                leaf_ns = [lpi(shapes[r], kernels) for r in ref]
                ctx = ([b["context"]] if b.get("context") else
                       [context_of(pts[r]) for r in ref if context_of(pts[r])])
                cases.append({
                    "case_id": "%s__%s_bucket_%d%s" % (label, phase.lower(), b["shape"],
                                                       "_c%dk" % round(b["context"] / 1000)
                                                       if b.get("context") else ""),
                    "kind": "bucket", "phase": phase, "shape": b["shape"],
                    "range": [b["lo"], b["hi"]], "cost_reference_points": ref,
                    "context": int(round(sum(ctx) / len(ctx))) if ctx else None,
                    "launches_per_instance": n, "invocations_per_instance": n,
                    "kernel_launches_per_instance": round(sum(leaf_ns) / len(leaf_ns), 4) if leaf_ns else 0.0,
                    "invocation_count_basis": basis, "instances_represented": b["instances"],
                    "effective_calls": int(round(n * b["instances"])),
                    "cost_basis": "interpolated",
                    "why": "%.2f%% of %s proxy mass sits outside every point's tolerance here"
                           % (b["proxy_mass_share_of_phase_pct"], phase)})
            left = summary.get(phase, {}).get("proxy_mass_outside_tolerance_pct", 0.0) - sum(
                b["proxy_mass_share_of_phase_pct"] for b in bands)
            residual[phase] = round(max(0.0, left), 3)
        for e in t.get("envelope_cases") or []:
            c = dict(e, kind="envelope", effective_calls=0, instances_represented=0,
                     cost_basis="not_weighted", invocation_count_basis={"kind": "unweighted_envelope"})
            c.setdefault("case_id", "%s__%s_envelope_%s" % (label, e["phase"].lower(), e["shape"]))
            cases.append(c)
        out[label] = {"cases": cases,
                      "invocation_anchors": t.get("invocation_anchors"),
                      "invocation_count_basis": basis,
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
