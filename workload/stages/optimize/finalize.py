#!/usr/bin/env python3
"""The optimize slot's finalize adapter: worker and verification artifacts ->
`$OUT/result.json`.

It belongs to the **slot**, not to an implementation — geak-kernel-workflow and
subagent-raw produce identically shaped artifacts (which is precisely what makes them
comparable), so this mapping is written once and survives a change of implementation.

One hard rule: **only cold-re-verified numbers are reported**. The best_speedup a worker
measured during optimization goes into result.json as a `claimed` field for comparison,
but `kept` and every externally usable speedup come from verified.json. That is how a
warm-PASS / cold-FAIL kernel was stopped in the previous campaign.

And one more thing stated in the notes rather than hidden: operator speedup is **not**
end-to-end gain. This stage has no authority to claim an end-to-end number; that belongs
to applyback.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))          # stages/optimize
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "scripts"))
from emit_result import emit, read_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheduler-rc", type=int, default=0)
    args = ap.parse_args()

    out = os.environ.get("OUT")
    if not out:
        print("run this inside the environment prepared by stage.py prepare",
              file=sys.stderr)
        return 2
    work = os.path.join(out, "work")
    sdir = os.path.join(work, "state")

    # --- What the workers reported during the run (for comparison only).
    claimed = {}
    for f in sorted(os.listdir(sdir)) if os.path.isdir(sdir) else []:
        if f.endswith(".status.json"):
            claimed[f[: -len(".status.json")]] = read_json(os.path.join(sdir, f))
    if not claimed:
        return emit("fail", notes="there are no worker status files at all — the scheduler "
                                  "never dispatched anything; see %s/scheduler.log" % work)

    blocked = [k for k, v in claimed.items() if v.get("status") == "blocked"]
    if len(blocked) == len(claimed):
        return emit("blocked",
                    notes="all %d tasks were judged blocked by the runner (unmet "
                          "precondition, exit code 78). Usually a missing credential or "
                          "binary; see %s/logs/*/attempt_*.log. Configure "
                          "runner_env.env_file and rerun — credit resumes."
                          % (len(claimed), work))

    # --- Cold re-verification results: the only numbers that may be quoted.
    vpath = os.path.join(out, "verified.json")
    ver = read_json(vpath)
    vrows = {r["kernel"]: r for r in ver.get("kernels", [])}
    if not vrows:
        return emit("fail", notes="there is no verified.json — cold re-verification did not "
                                  "run, and no speedup may be used before it passes. See "
                                  "%s/verify.out" % out)

    kernels, total_credit = [], 0
    for k in sorted(claimed):
        c, v = claimed[k], vrows.get(k, {})
        total_credit += int(c.get("credited_seconds") or 0)
        kernels.append({
            "kernel": k,
            "verdict": v.get("verdict", "not_verified"),
            "verified_speedup": v.get("speedup"),
            "claimed_speedup": c.get("best_speedup"),
            "attempts": c.get("attempts", 0),
            "credited_seconds": c.get("credited_seconds", 0),
            "worker_status": c.get("status"),
            "best_kernel_src": os.path.relpath(
                os.path.join(work, "runs", k, "best_kernel_src"), out)
            if os.path.exists(os.path.join(work, "runs", k, "best_kernel_src")) else None,
            "notes": v.get("notes", []),
        })

    kept = [r for r in kernels if r["verdict"] == "pass"]
    kept.sort(key=lambda r: -(r["verified_speedup"] or 0))
    by_verdict = {}
    for r in kernels:
        by_verdict[r["verdict"]] = by_verdict.get(r["verdict"], 0) + 1

    summary = {
        "skill": os.environ.get("STAGE_SKILL"),
        "runner": os.path.basename(os.environ.get("RUNNER", "")),
        "scheduler_rc": args.scheduler_rc,
        "tasks_total": len(kernels),
        "verdict_counts": by_verdict,
        "gpu_hours": round(total_credit / 3600.0, 2),
        "kept": [{"kernel": r["kernel"], "speedup": r["verified_speedup"]} for r in kept],
        "kernels": kernels,
    }
    with open(os.path.join(out, "optimized.json"), "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)

    notes = [
        "Only kernels that pass cold re-verification (verdict=pass) may enter applyback; "
        "claimed_speedup is what the worker reported during optimization and is for "
        "comparison only.",
        "Operator speedup is not end-to-end gain — this stage claims no end-to-end number; "
        "that belongs to applyback. Historically MoE sorting measured 1.80x on the "
        "microbenchmark for +0.2356% whole-machine, and split4 measured 1.342x for "
        "-0.0872%.",
    ]
    if blocked:
        notes.append("%d tasks were judged blocked: %s" % (len(blocked), ", ".join(blocked)))
    if args.scheduler_rc not in (0, 130):
        notes.append("the scheduler exited abnormally (rc=%d); results already kept remain "
                     "valid and a rerun resumes from credit." % args.scheduler_rc)

    # Keeping no kernel at all is **not** a stage failure: the stage ran, the numbers are
    # real, and the conclusion happens to be "no win". Marking it fail would send someone
    # to repair something that is not broken. applyback discovers on its own that there is
    # nothing to install.
    if not kept:
        notes.insert(0, "no kernel passed cold re-verification (%s) — the stage completed "
                        "normally, but there is nothing to apply back." % by_verdict)
    return emit("ok",
                artifacts={"optimized": "optimized.json",
                           "verified": os.path.relpath(vpath, out)},
                metrics={"tasks_total": len(kernels), "kept": len(kept),
                         "gpu_hours": summary["gpu_hours"],
                         "best_speedup": kept[0]["verified_speedup"] if kept else None},
                notes=" ".join(notes))


if __name__ == "__main__":
    sys.exit(main())
