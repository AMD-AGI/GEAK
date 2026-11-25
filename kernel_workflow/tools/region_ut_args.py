#!/usr/bin/env python3
"""mode=mega args from a region UT task dir (args.region_ut): what the UT fixes, so a caller passes only the UT.

A region UT (e.g. a KernelFusion L3 MegaKernel region) is a GEAK task dir: meta.json, cases.py, unittest.py
and kernel_src/ (knowledge/mega_region_ut.md). Its own unittest.py is the benchmark; its case sigs are the
guards; its tolerances are the accuracy threshold; its bound modules are what a candidate may replace.
kernel_lane.js fills every arg the caller left unset from this JSON (an arg the caller passed wins).
The heaviest case (served weight) is the target; every other case is a regression guard with a 2% noise
floor, so an implementation may keep the stock path there (a declared refusal). The UT's graph-replay
stress (meta.graph_replays) is the liveness evidence the finalist gate counts.

Usage: region_ut_args.py <task_dir>   -> prints one JSON object (and exits 1 with {"error"} on a bad UT)
"""
import json
import os
import sys


def region_ut_args(task_dir):
    task = os.path.abspath(task_dir)
    with open(os.path.join(task, "meta.json")) as fh:
        meta = json.load(fh)
    cases = [c for c in meta.get("cases") or [] if isinstance(c, dict) and c.get("sig")]
    if not cases:
        raise ValueError("meta.json has no cases with a sig")
    if not os.path.isfile(os.path.join(task, "unittest.py")):
        raise ValueError("the task dir has no unittest.py")
    binds = meta.get("candidate_bind") or []
    binds = binds if isinstance(binds, list) else [binds]
    modules = [b.get("module") or b.get("impl_module") for b in binds if isinstance(b, dict)]
    if not [m for m in modules if m]:
        raise ValueError("meta.json binds no module (candidate_bind)")
    sigs = [c["sig"] for c in cases]
    tols = [float(c.get("tol") or meta.get("tol") or 0) for c in cases]
    rel = meta.get("correct_metric") == "rel_l2"
    weights = [float(c.get("weight") or 0) for c in cases]
    target = sigs[weights.index(max(weights))] if any(weights) else sigs[0]
    others = [s for s in sigs if s != target]
    replays = int(meta.get("graph_replays") or 0)
    return {
        "benchmark_harness": os.path.join(task, "unittest.py"),
        "candidate_import_modules": [m for m in modules if m],
        "target_guards": [target],
        "regression_guards": others,
        "regression_floor_by_guard": {s: 0.98 for s in others},
        "required_replays": max(30, replays),
        "required_accuracy_cases": sigs,
        "accuracy_metric": "relL2" if rel else "elementwise",
        "accuracy_threshold": max(tols) if any(tols) else 0.02,
        # The UT replays every case under one captured graph itself.
        "require_graph_capture": "false",
        "positive_control": {
            "name": "synthetic slowdown inside the seam", "guard": target,
            "how": "a calibrated device-side delay inside a kernel_src/ copy of the seam, measured with the "
                   "UT's own unittest.py (knowledge/mega_region_ut.md)",
            "expected_pct_lo": -9.0, "expected_pct_hi": -4.0, "magnitude": "constructed",
            "abort_on_fail": False, "required_control_pairs": 5, "required_null_pairs": 8,
        },
        "region_ut_world_size": int(meta.get("world_size") or 1),
        **({} if replays >= 30 else {"region_ut_warning": "meta.graph_replays < 30: the UT gives no liveness "
                                                           "evidence, so no finalist can pass mode=mega's gate"}),
    }


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(json.dumps({"error": "usage: region_ut_args.py <task_dir>"}))
        return 1
    try:
        print(json.dumps(region_ut_args(argv[0])))
        return 0
    except (OSError, ValueError, TypeError) as e:
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
