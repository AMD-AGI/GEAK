#!/usr/bin/env python3
"""A starting contract for the region UT adapter (knowledge/mega_region_ut.md) on the UT layouts GEAK produces.

The adapter agent owns the contract: it starts from this output when the layout is known, corrects it from
the UT, and derives every field itself for any other layout ({"error"} here).

A region UT is a GEAK UT task dir (unittest.py, cases.py, meta.json, kernel_src/, run/):
- kernel_ut: the e2e kernel-extraction UT (e2e_workflow kernel-ut skill): meta.workload.cases (sig, regime,
  effective_calls), meta.tol;
- acceptance: a UT that also declares meta.cases (served `weight`) and meta.acceptance {tol, noise_bound,
  min_speedup}, and times every case in unittest.py.

- benchmark: the UT's own unittest.py; accuracy: its correctness gate at acceptance.tol, else meta.tol.
- guards: the heaviest case (weight, else effective_calls) is the target; every other case is a regression
  guard whose floor is widened by the UT's measured self-noise for that case (run/self_noise.json, 2x the
  |1 - ratio|), never above 0.98, so a noisy small case cannot veto a candidate on noise alone.
- liveness: a UT with a `--iters` option (an acceptance UT runner) replays each graph-regime case that many
  times per leg; otherwise it checks one refreshed-input graph replay per graph case. That count is the
  replay evidence.

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
    workload = meta.get("workload") if isinstance(meta.get("workload"), dict) else {}
    cases = [c for c in meta.get("cases") or workload.get("cases") or [] if isinstance(c, dict) and c.get("sig")]
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
    tol = acc.get("tol") if acc.get("tol") is not None else meta.get("tol")
    if tol is None:
        raise ValueError("meta.json has no acceptance.tol or tol")
    fusion = bool(acc) and bool(meta.get("cases"))

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

    phases = meta.get("phase_regimes") if isinstance(meta.get("phase_regimes"), dict) else {}
    graph = any(c.get("regime") == "graph" or (phases.get(c.get("regime")) or {}).get("cuda_graph") is True
                for c in cases)
    replays = (_ut_iters(ut) or 1) if graph else None
    out = {
        "benchmark_harness": ut,
        "candidate_import_modules": [m for m in modules if m],
        "target_guards": [target],
        "regression_guards": others,
        "regression_floor_by_guard": floor,
        "required_replays": replays or 30,
        "required_accuracy_cases": sigs,
        "accuracy_metric": "ut_max_err",
        "accuracy_threshold": float(tol),
        # The UT captures and replays its graph cases itself.
        "require_graph_capture": "false",
        "positive_control": {
            "name": "synthetic slowdown inside the seam", "guard": target,
            "how": "a calibrated device-side delay inside a kernel_src/ copy of the seam, measured with the "
                   "UT's own unittest.py (knowledge/mega_region_ut.md)",
            "expected_pct_lo": -9.0, "expected_pct_hi": -4.0, "magnitude": "constructed",
            "abort_on_fail": False, "required_control_pairs": 5, "required_null_pairs": 8,
        },
        "region_ut_format": "acceptance" if fusion else "kernel_ut",
    }
    if not replays:
        out["region_ut_warning"] = ("the UT replays no captured graph: it gives no liveness evidence, so no "
                                    "finalist can pass mode=mega's gate")
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
