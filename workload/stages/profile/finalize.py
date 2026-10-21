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


def coverage(out, targets):
    """(uncovered, covered_by_task, blocked) by identity, per task_plan.json.

    Returns `None` when the skill declared no plan, so the caller can fall back."""
    plan = read_json(os.path.join(out, "task_plan.json"), default={})
    if not plan.get("tasks") and not plan.get("blocked_identities"):
        return None
    owned, blocked = set(), set()
    for t in plan.get("tasks") or []:
        owned.update(t.get("selected_identities") or [])
    for b in plan.get("blocked_identities") or []:
        if b.get("kernel") and (b.get("blocker") or "").strip():
            blocked.add(b["kernel"])
    ids = [t.get("id") or t.get("name") for t in targets]
    return ([i for i in ids if i not in owned and i not in blocked],
            [i for i in ids if i in owned], [i for i in ids if i in blocked])
TASK_FILES = ["unittest.py", "meta.json", "cases.py"]


def ut_verdict(task_dir):
    """(passed, why) from the task's own receipt, run/ut_result.json: exit_code 0 is a UT whose
    every gate ran and passed. A task without one has not been run on a GPU."""
    path = os.path.join(task_dir, "run", "ut_result.json")
    if not os.path.exists(path):
        return False, "no run/ut_result.json"
    receipt = read_json(path)
    if receipt.get("exit_code") != 0:
        return False, "exit_code %r (%s)" % (receipt.get("exit_code"), receipt.get("verdict", "?"))
    return True, ""


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

    unpassed = []
    for name, _ in built:
        passed, why = ut_verdict(os.path.join(tasks_dir, name))
        if not passed:
            unpassed.append("%s: %s" % (name, why))

    metrics = {
        "kernels_censused": censused,
        "targets_selected": len(targets),
        "tasks_built": len(built),
        "tasks_ut_passed": len(built) - len(unpassed),
        "selected_pct_gpu_time": round(selected_pct, 4),
    }

    notes = []
    status = "ok"
    if unpassed:
        status = "blocked"
        notes.append("these tasks have no passing UT receipt (run/ut_result.json with exit_code "
                     "0): %s. Run each task's unittest.py on a GPU, or move the identities it "
                     "owns to task_plan.json blocked_identities with the reason"
                     % "; ".join(unpassed))
    if broken:
        status = "blocked"
        notes.append("these task directories are incomplete and the optimize stage cannot "
                     "run on them: " +
                     "; ".join("%s is missing %s" % (n, l) for n, l in broken))
    cov = coverage(out, targets)
    if cov is not None:
        uncovered, owned, blocked_ids = cov
        metrics["identities_owned_by_tasks"] = len(owned)
        metrics["identities_blocked"] = len(blocked_ids)
        if uncovered:
            status = "blocked"
            notes.append("these selected identities have neither a task nor a stated "
                         "blocker in task_plan.json: %s" % uncovered)
    elif len(built) < len(targets):
        status = "blocked"
        notes.append("%d identities were selected but only %d tasks were built, and no "
                     "task_plan.json declares how they map. Every selected identity needs "
                     "either a task or a stated blocker; complete it and rerun finalize."
                     % (len(targets), len(built)))
    if not targets:
        status = "fail"
        notes.append("targets in selection_targets.json is empty. Missing evidence is not "
                     "the same as an empty target set.")

    return emit(status=status, artifacts=artifacts, metrics=metrics,
                notes="; ".join(notes))


if __name__ == "__main__":
    sys.exit(main())
