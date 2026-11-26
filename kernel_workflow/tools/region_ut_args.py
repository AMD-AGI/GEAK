#!/usr/bin/env python3
"""mode=mega args from a region UT task dir (args.region_ut): what the UT fixes, so a caller passes only the UT.

A region UT is the task dir KernelFusion builds for a fusion region (e2e_workflow fusion_ut_contract.md):
meta.json (`cases` with served `weight`, `candidate_bind`, `acceptance` {tol, noise_bound, min_speedup}),
unittest.py, kernel_src/, run/self_noise.json. kernel_lane.js fills every arg the caller left unset from
this JSON (an arg the caller passed wins); knowledge/mega_region_ut.md says how each role uses the UT.

- benchmark: the UT's own unittest.py; accuracy: its CORRECTNESS gate at acceptance.tol (the UT's units).
- guards: the heaviest served case is the target; every other case is a regression guard whose floor is
  widened by the UT's measured self-noise for that case (run/self_noise.json, 2x the |1 - ratio|), never
  above 0.98, so a noisy small case cannot veto a candidate on noise alone.
- liveness: graph-regime cases are captured once and replayed `--iters` times per leg in the UT's served
  mode without a hang; that count is the replay evidence (required_replays).

Usage: region_ut_args.py <task_dir>   -> prints one JSON object (and exits 1 with {"error"} on a bad UT)
"""
import json
import os
import re
import sys


def _ut_iters(unittest_py):
    """The UT's default timed replays per case (`--iters`), or None when it declares none."""
    with open(unittest_py) as fh:
        m = re.search(r"""add_argument\(\s*["']--iters["'][^)]*default\s*=\s*(\d+)""", fh.read())
    return int(m.group(1)) if m else None


def region_ut_args(task_dir):
    task = os.path.abspath(task_dir)
    with open(os.path.join(task, "meta.json")) as fh:
        meta = json.load(fh)
    cases = [c for c in meta.get("cases") or [] if isinstance(c, dict) and c.get("sig")]
    if not cases:
        raise ValueError("meta.json has no cases with a sig")
    ut = os.path.join(task, "unittest.py")
    if not os.path.isfile(ut):
        raise ValueError("the task dir has no unittest.py")
    bind = meta.get("candidate_bind")
    binds = bind if isinstance(bind, list) else [bind]
    modules = [b.get("impl_module") or b.get("module") for b in binds if isinstance(b, dict)]
    if not [m for m in modules if m]:
        raise ValueError("meta.json binds no module (candidate_bind)")
    acc = meta.get("acceptance") if isinstance(meta.get("acceptance"), dict) else {}
    if acc.get("tol") is None:
        raise ValueError("meta.json has no acceptance.tol (not a fusion-contract UT)")

    sigs = [c["sig"] for c in cases]
    weights = [float(c.get("weight") or c.get("effective_calls") or 0) for c in cases]
    target = sigs[weights.index(max(weights))] if any(weights) else sigs[0]
    others = [s for s in sigs if s != target]
    noise = {}
    try:
        with open(os.path.join(task, "run", "self_noise.json")) as fh:
            noise = {c["sig"]: abs(1 - float(c["ratio"])) for c in json.load(fh).get("per_case") or []
                     if c.get("sig") and c.get("ratio")}
    except (OSError, ValueError, TypeError, KeyError):
        pass
    floor = {s: round(min(0.98, 1 - 2 * noise.get(s, 0.0)), 3) for s in others}

    graph = any(c.get("regime") == "graph" for c in cases)
    iters = _ut_iters(ut) if graph else None
    out = {
        "benchmark_harness": ut,
        "candidate_import_modules": [m for m in modules if m],
        "target_guards": [target],
        "regression_guards": others,
        "regression_floor_by_guard": floor,
        "required_replays": max(30, iters or 0),
        "required_accuracy_cases": sigs,
        "accuracy_metric": "ut_max_err",
        "accuracy_threshold": float(acc["tol"]),
        # The UT captures and replays its graph cases itself.
        "require_graph_capture": "false",
        "positive_control": {
            "name": "synthetic slowdown inside the seam", "guard": target,
            "how": "a calibrated device-side delay inside a kernel_src/ copy of the seam, measured with the "
                   "UT's own unittest.py (knowledge/mega_region_ut.md)",
            "expected_pct_lo": -9.0, "expected_pct_hi": -4.0, "magnitude": "constructed",
            "abort_on_fail": False, "required_control_pairs": 5, "required_null_pairs": 8,
        },
    }
    if not graph or not iters or iters < 30:
        out["region_ut_warning"] = ("the UT replays no graph case --iters >= 30 times: it gives no liveness "
                                    "evidence, so no finalist can pass mode=mega's gate")
    return out


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
