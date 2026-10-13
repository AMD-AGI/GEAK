#!/usr/bin/env python3
"""The profile slot's finalize adapter: map whatever the skill left in $OUT into
result.json.

It belongs to the **slot**, not to any implementation. Swapping in another profile skill
requires no edit here, as long as the new skill produces the files named in the table in
README.md. It only reads $OUT and writes nothing but result.json.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/scripts")
from emit_result import emit, read_json  # noqa: E402

REQUIRED = ["kernel_workload.json", "kernel_ranking.json", "selection_targets.json"]
TASK_FILES = ["unittest.py", "meta.json", "cases.py"]


def main():
    out = os.environ.get("OUT")
    if not out:
        print("OUT is not set; run stage.py prepare profile first", file=sys.stderr)
        return 2

    missing = [f for f in REQUIRED if not os.path.exists(os.path.join(out, f))]
    if missing:
        return emit(status="fail", artifacts={},
                    notes="required artifacts are missing: %s. The skill's §2 (census and "
                          "target selection) did not finish; complete it and rerun this "
                          "finalize." % missing)

    sel = read_json(os.path.join(out, "selection_targets.json"))
    targets = sel.get("targets") or []
    selected_pct = sum(t.get("overall_pct") or 0.0 for t in targets)

    census = read_json(os.path.join(out, "kernel_ranking.json"))
    censused = len(census.get("ranking") or [])

    tasks_dir = os.path.join(out, "tasks")
    built, broken = [], []
    if os.path.isdir(tasks_dir):
        for name in sorted(os.listdir(tasks_dir)):
            td = os.path.join(tasks_dir, name)
            if not os.path.isdir(td):
                continue
            lack = [f for f in TASK_FILES if not os.path.exists(os.path.join(td, f))]
            if not os.path.isdir(os.path.join(td, "kernel_src")):
                lack.append("kernel_src/")
            (broken if lack else built).append((name, lack))

    artifacts = {k.split(".")[0]: k for k in REQUIRED}
    for opt in ("task_plan.json", "REPORT.md", "results.json"):
        if os.path.exists(os.path.join(out, opt)):
            artifacts[opt.split(".")[0].lower()] = opt
    if built:
        artifacts["tasks"] = "tasks"

    metrics = {
        "kernels_censused": censused,
        "targets_selected": len(targets),
        "tasks_built": len(built),
        "selected_pct_gpu_time": round(selected_pct, 4),
    }

    notes = []
    status = "ok"
    if broken:
        status = "blocked"
        notes.append("these task directories are incomplete and the optimize stage cannot "
                     "run on them: " +
                     "; ".join("%s is missing %s" % (n, l) for n, l in broken))
    if len(built) < len(targets):
        status = "blocked"
        notes.append("%d identities were selected but only %d tasks were built. Every "
                     "selected identity needs either a task or a stated blocker; complete "
                     "it and rerun finalize." % (len(targets), len(built)))
    if not targets:
        status = "fail"
        notes.append("targets in selection_targets.json is empty. Missing evidence is not "
                     "the same as an empty target set.")

    return emit(status=status, artifacts=artifacts, metrics=metrics,
                notes="; ".join(notes))


if __name__ == "__main__":
    sys.exit(main())
