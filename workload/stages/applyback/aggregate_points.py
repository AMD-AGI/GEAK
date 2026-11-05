#!/usr/bin/env python3
"""Per-point A/Bs -> one ab.json + result.json for a multi-config baseline.

The baseline stage marked a few representative configs (adopt-ci-artifacts: each group's
lowest and highest concurrency). Applyback measures every one of them in `$OUT/points/<pid>/`; the slot's finalize.py
judges each against its own baseline and then calls this. This folds them:

  verdict  regression  if any point regressed beyond its noise band
           gain        if some point gained and none regressed
           no_change   if every point stayed inside its band
           incomplete  if a point produced no A/B (its own result.json says why)

A gain at one concurrency is not allowed to hide a regression at another: the point of
measuring the curve's ends is to see both. `e2e_delta_pct` at the top is the geometric
mean over points, labelled as such; the per-point deltas are the result.
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "_shared"))
sys.path.insert(0, HERE)
from emit_result import emit, read_json  # noqa: E402


def main():
    out = os.environ["OUT"]
    pids = sys.argv[1:]
    points, missing = {}, {}
    for pid in pids:
        d = os.path.join(out, "points", pid)
        ab = read_json(os.path.join(d, "ab.json"))
        res = read_json(os.path.join(d, "result.json"))
        if ab and res.get("status") == "ok":
            points[pid] = ab
        else:
            missing[pid] = res.get("notes") or "no ab.json"
    verdicts = {p: ab["verdict"] for p, ab in points.items()}
    if missing:
        verdict = "incomplete"
    elif "regression" in verdicts.values():
        verdict = "regression"
    elif "gain" in verdicts.values():
        verdict = "gain"
    else:
        verdict = "no_change"
    ratios = [ab["e2e_ratio"] for ab in points.values() if ab.get("e2e_ratio")]
    geo = math.exp(sum(math.log(r) for r in ratios) / len(ratios)) if ratios else None
    first = points.get(pids[0]) if pids else None
    ab = {
        "multi_config": True,
        "reference": "each point's own InferenceX CI number (baseline adopt-ci-artifacts)",
        "verdict": verdict,
        "points": points,
        "points_without_ab": missing,
        "e2e_delta_pct": round((geo - 1) * 100, 4) if geo else None,
        "e2e_delta_pct_basis": "geometric mean of the per-point ratios",
        # single-point readers (report.py) see the first representative point here
        "baseline": (first or {}).get("baseline"),
        "candidate": (first or {}).get("candidate"),
        "op_level": (first or {}).get("op_level"),
        "op_speedup_max": (first or {}).get("op_speedup_max"),
        "kernels_applied": (first or {}).get("kernels_applied"),
    }
    with open(os.path.join(out, "ab.json"), "w") as f:
        json.dump(ab, f, indent=1)
    per = "; ".join("%s %+.3f%% (%s)" % (p, a["e2e_delta_pct"], a["verdict"])
                    for p, a in points.items())
    notes = "%d representative points: %s." % (len(pids), per or "none measured")
    if missing:
        notes += " Without an A/B: %s." % "; ".join("%s: %s" % kv for kv in missing.items())
    metrics = {"points": len(pids), "points_measured": len(points), "verdict": verdict}
    if geo:
        metrics["e2e_delta_pct_geomean"] = ab["e2e_delta_pct"]
    for p, a in points.items():
        metrics["e2e_delta_pct:" + p] = a["e2e_delta_pct"]
    status = "fail" if missing and not points else ("blocked" if missing else "ok")
    return emit(status, artifacts={"ab": "ab.json", "applied": "applied.json", "overlay": "overlay"},
                metrics=metrics, notes=notes)


if __name__ == "__main__":
    sys.exit(main())
