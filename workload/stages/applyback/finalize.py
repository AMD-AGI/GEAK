#!/usr/bin/env python3
"""Validate kernel integration, then use Evaluate's shared serving comparison.

This remains the standard recipe's applyback finalizer. Direct configuration/runtime
experiments use the system integration skill and Evaluate without inventing kernel
artifacts. Kernel source-identity and shadow checks remain mandatory in this adapter.
"""
import importlib.util
import os
import sys

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
