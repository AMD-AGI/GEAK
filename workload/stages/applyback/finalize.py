#!/usr/bin/env python3
"""Validate kernel integration, then use Evaluate's shared serving comparison.

This remains the standard recipe's applyback finalizer. Direct configuration/runtime
experiments use the system integration skill and Evaluate without inventing kernel
artifacts. Kernel source-identity and shadow checks remain mandatory in this adapter.
"""
import importlib.util
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(os.path.dirname(HERE), "evaluate", "finalize.py")
spec = importlib.util.spec_from_file_location("_workload_evaluate", path)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def check_identity(out, run_root):
    """One hash check: what the overlay carries is what cold verification accepted.

    `applied.json` records each installed kernel with the hashes the overlay actually
    holds. `optimized.json` records what two cold runs measured. Equal or the stage is a
    fail -- a kernel whose bytes drifted between the two has no evidence behind it.
    """
    applied = evaluation.read_json(os.path.join(out, "applied.json"))
    rows = applied.get("applied")
    if not isinstance(rows, list):
        return applied, "applied.json records no installed kernels"
    optimized = evaluation.read_json(os.path.join(run_root, "optimize", "out", "optimized.json"))
    verified = {r.get("kernel"): r for r in (optimized.get("kernels") or [])
                if isinstance(r, dict)}
    for row in rows:
        name = row.get("kernel")
        source = verified.get(name)
        if not source:
            return applied, ("%s was installed but optimize never reported it" % name)
        expected = source.get("kernel_src_hashes")
        if not isinstance(expected, dict) or not expected:
            return applied, ("%s has no kernel_src_hashes from cold verification; recover "
                             "the recorded source identity rather than hashing current "
                             "work in progress" % name)
        got = row.get("kernel_src_hashes")
        if got != expected:
            return applied, ("%s: the overlay's source does not match the cold-verified "
                             "kernel_src_hashes; the bytes that were measured are not the "
                             "bytes that were installed" % name)
        meta = source.get("task_meta_sha256")
        if meta and row.get("task_meta_sha256") not in (None, meta):
            return applied, ("%s: task metadata differs from the cold-verified "
                             "task_meta_sha256, so the binding may point elsewhere" % name)
    return applied, None


def main():
    out = os.environ.get("OUT")
    if not out:
        print("run this inside the environment run.py prepares", file=sys.stderr)
        return 2
    run_root = os.environ.get("RUN_ROOT") or os.path.dirname(os.path.dirname(out))
    params = evaluation.read_json(os.environ["PARAMS"]) if os.environ.get("PARAMS") else {}

    # A multi-config baseline (adopt-ci-artifacts): each point was measured in
    # $OUT/points/<pid>/ and is judged here against its own baseline, then folded
    # (aggregate_points.py): a gain at one concurrency cannot hide a regression at another.
    # Which points must have been measured is points.py's answer, the same list the session
    # launched from: every config a UT case names (params.points, default "covered").
    top = evaluation.read_json(os.path.join(run_root, "baseline", "out", "baseline.json"))
    if top.get("multi_config") and top.get("points") \
            and not os.environ.get("APPLYBACK_POINT"):
        import points as point_selection
        sel = point_selection.select(top, point_selection.tasks_dir_of(run_root),
                                     params.get("points") or "covered")
        if sel["unknown"]:
            return evaluation.emit("blocked", notes="params.points names %s, which baseline.json does not "
                        "have" % sel["unknown"])
        wanted = set(sel["points"]) | set(sel["unmeasured_covered"])
        pids = [p for p in point_selection.measure_order(top) if p in wanted]
        for pid in pids:
            pout = os.path.join(out, "points", pid)
            os.makedirs(pout, exist_ok=True)
            tasks = ", ".join(sel["ut_configs"].get(pid, [])) or "none"
            why = None
            if pid in sel["unmeasured_covered"]:
                why = ("a UT priced this config (tasks: %s) but params.points=%r left it out "
                       "of the end-to-end check" % (tasks, sel["spec"]))
            elif not os.path.exists(os.path.join(pout, "e2e", "bench_summary.json")):
                why = ("selected for measurement (UT tasks: %s) but never measured: "
                       "points/%s/e2e/bench_summary.json does not exist" % (tasks, pid))
            if why:
                stale = os.path.join(pout, "ab.json")
                if os.path.lexists(stale):
                    os.replace(stale, stale + ".before-unmeasured.%d" % time.time_ns())
                with open(os.path.join(pout, "result.json"), "w") as fh:
                    json.dump({"status": "blocked", "notes": why}, fh, indent=1)
                continue
            env = dict(os.environ, OUT=pout, RUN_ROOT=run_root, APPLYBACK_POINT=pid,
                       APPLYBACK_BASELINE=os.path.join(run_root, "baseline", "out", "points",
                                                       pid, "baseline.json"))
            subprocess.call([sys.executable, os.path.abspath(__file__)], env=env)
        sys.argv = [sys.argv[0]] + pids
        os.environ["APPLYBACK_POINT_SELECTION"] = json.dumps(sel)
        import aggregate_points
        return aggregate_points.main()

    def fail(reason, status="fail"):
        return evaluation.reject(out, reason, status)

    # --- The overlay: was anything installed, did it shadow, is it what was verified.
    applied, problem = check_identity(out, run_root)
    if not applied:
        return fail("there is no applied.json, so nothing records what was installed",
                    status="blocked")
    if not applied.get("applied"):
        return fail("there is no kernel to apply back: the optimize stage produced nothing "
                    "that passed cold re-verification. Every stage ran; the conclusion is "
                    "that no usable optimization has been found yet.", status="blocked")
    if problem:
        return fail(problem)

    shadow = evaluation.read_json(os.path.join(out, "shadow_check.json"))
    shadowed = shadow.get("all_shadowed")
    if shadowed is None:
        shadowed = applied.get("all_shadowed")
    if shadowed is not True:
        return fail("the overlay did not shadow every target (shadow_check.json). Without "
                    "that proof, 'no gain' and 'the patch was never loaded' are the same "
                    "result, so no end-to-end number is admitted.")

    proof_cfg = params.get("serving_source_proof") or {}
    if proof_cfg:
        proof = evaluation.read_json(os.path.join(out, proof_cfg.get("file") or
                                       "serving_source_proof.json"))
        ranks = proof.get("ranks")
        want = proof_cfg.get("world_size")
        have = len(ranks) if isinstance(ranks, (list, dict)) else 0
        if want and have != want:
            return fail("serving source proof covers %d of %d ranks; a local callable "
                        "check is preflight, not evidence of what the serving workers "
                        "loaded" % (have, want))

    return evaluation.main(applied)


if __name__ == "__main__":
    sys.exit(main())
