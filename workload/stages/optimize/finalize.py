#!/usr/bin/env python3
"""The optimize slot's verdict: account for every extracted task, judge no measurement.

It belongs to the slot, not to any implementation (`stages/CONTRACT.md` section 3.1), and
the driver runs it after every session. What it checks is *accounting*: that the wave
covered every task profile delivered, that capacity deferrals name a task and a reason,
and that the campaign budget was actually spent. **It never re-derives whether a kernel is
correct or faster.** A previous `verify.py` here did re-derive that, parsed three output
dialects of its own, and recorded a task that crashed before comparing any output as a
correctness failure. The task's own unit test is the authority on the kernel; this file is
the authority on the delivery.

Checked here rather than left to the session because an agent that optimized six of ten
tasks will report on six.

    python3 stages/optimize/finalize.py          # inside the environment run.py prepares
"""

import argparse
import json
import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "_shared"))
from emit_result import emit, read_json  # noqa: E402


def valid_limit(limit):
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("task_limit must be a positive integer")
    return limit


def check_limit(tasks, limit):
    """How many tasks may run at once -- a campaign capacity policy, not a discovery cap.

    `task_limit` is the number of optimization tasks this campaign can hold in flight,
    one per GPU. It bounds *scheduling*. It used to be applied to extraction instead,
    which made a complete ten-task profile delivery structurally impossible: the same
    skill run by hand delivered ten tasks and the slot would have called that a defect.
    Extraction stops where the evidence stops; the remainder is scheduled in a later wave
    and recorded as capacity-deferred."""
    valid_limit(limit)
    if len(tasks) > limit:
        raise ValueError("%d tasks are scheduled in one wave, exceeding task_limit=%d "
                         "(one task per GPU). Carry the remainder as capacity_deferred "
                         "and schedule it after; do not delete an extracted task: %s"
                         % (len(tasks), limit, ", ".join(tasks)))


def check_accounting(tasks, selected, deferred=None):
    """Account for the whole delivery, independently of instantaneous GPU capacity."""
    deferrals = {}
    for row in deferred or []:
        name = row.get("kernel") if isinstance(row, dict) else row
        if not isinstance(name, str) or not name:
            raise ValueError("every capacity_deferred row must name a task")
        if not isinstance(row, dict) or not str(row.get("reason") or "").strip():
            raise ValueError("capacity-deferred task %s needs a stated reason: a deferral "
                             "is a disposition that keeps its identity and its evidence, "
                             "not an omission" % name)
        if name in deferrals:
            raise ValueError("Optimization must account for every extracted task exactly "
                             "once; duplicate capacity-deferred task: " + name)
        deferrals[name] = row
    missing = sorted(set(tasks) - set(selected) - set(deferrals))
    unexpected = sorted((set(selected) | set(deferrals)) - set(tasks))
    overlap = sorted(set(selected) & set(deferrals))
    if missing or unexpected or overlap or len(selected) != len(set(selected)):
        raise ValueError("Optimization must account for every extracted task exactly "
                         "once, as optimized or as capacity-deferred with a reason; "
                         "unaccounted=%s, not extracted=%s, both optimized and deferred="
                         "%s. max_tasks/exclude_tasks must not silently narrow the "
                         "profile delivery." % (missing, unexpected, overlap))
    return deferrals


def check_dispatch(tasks, selected, limit, deferred=None):
    """Every extracted task is in this wave or deferred with a reason."""
    check_limit(selected, limit)
    deferrals = check_accounting(tasks, selected, deferred)
    unused = min(len(tasks), limit) - len(selected)
    if deferrals and unused > 0:
        raise ValueError("%d tasks were deferred for capacity while %d of the %d slots "
                         "are unused: %s. Capacity defers what does not fit, in task "
                         "order; it is not a reason to optimize fewer tasks than the "
                         "campaign can hold." % (len(deferrals), unused, limit,
                                                 sorted(deferrals)))


def check_report(tasks, report, limit, required_seconds=None, work_root=None):
    rows = report.get("kernels")
    if not isinstance(rows, list) or any(not isinstance(r, dict) or
                                        not isinstance(r.get("kernel"), str) for r in rows):
        raise ValueError("optimized.json needs one named kernels row per extracted task")
    deferred = report.get("capacity_deferred") or []
    valid_limit(limit)
    names = [r["kernel"] for r in rows]
    check_accounting(tasks, names, deferred)
    if report.get("tasks_total") != len(tasks):
        raise ValueError("tasks_total must count all extracted tasks, not just kept kernels")
    kept = report.get("kept")
    if not isinstance(kept, list) or any(not isinstance(r, dict) or
                                        r.get("kernel") not in names for r in kept):
        raise ValueError("kept must name only tasks present in the complete report")
    if len({r["kernel"] for r in kept}) != len(kept):
        raise ValueError("kept contains duplicate task names")
    attempts = [r.get("attempts") for r in rows]
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in attempts):
        raise ValueError("Each task must record its nonnegative attempt count")
    unattempted = ([r["kernel"] for r in rows if r["attempts"] == 0]
                   + [r["kernel"] for r in deferred])
    if any(not r.get("notes") for r in rows if r["attempts"] == 0):
        raise ValueError("Unattempted tasks need a stated blocker or interruption in notes")
    if required_seconds is not None:
        if deferred:
            raise ValueError("Capacity-deferred tasks have not completed the required %ds "
                             "optimization budget: %s"
                             % (required_seconds, ", ".join(r["kernel"] for r in deferred)))
        for row in rows:
            credit = row.get('credited_seconds', 0)
            if isinstance(credit, bool) or not isinstance(credit, int) or credit < required_seconds:
                raise ValueError("%s has not completed the required %ds optimization budget"
                                 % (row['kernel'], required_seconds))
            if work_root is not None:
                state = json.loads((Path(work_root) / 'state' / (row['kernel']+'.status.json')).read_text())
                if state.get('status') != 'done' or state.get('credited_seconds', 0) < required_seconds:
                    raise ValueError("worker state does not prove the full budget for " + row['kernel'])
    return {"tasks_extracted": len(tasks), "tasks_reported": len(rows),
            "tasks_attempted": sum(n > 0 for n in attempts), "tasks_kept": len(kept),
            "unattempted_tasks": unattempted,
            "all_tasks_attempted": not unattempted,
            "tasks_capacity_deferred": len(deferred),
            "capacity_deferred": sorted(r["kernel"] for r in deferred)}


def main():
    """Judge `$OUT` and write `result.json`; the paths come from the environment.

    The flags remain for reading a delivery by hand. Without `$OUT` it prints its findings
    and writes nothing, which is how the optimization session can ask in advance what it
    will be held to."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", type=Path,
                        help="extracted tasks; default $TASKS_DIR or $RUN_ROOT/profile/out/tasks")
    parser.add_argument("--optimized", type=Path, help="default $OUT/optimized.json")
    parser.add_argument("--task-limit", type=int, help="default the stage's task_limit")
    args = parser.parse_args()
    out = os.environ.get("OUT")
    params = read_json(os.environ["PARAMS"]) if os.environ.get("PARAMS") else {}
    task_root = os.environ.get("TASKS_DIR")
    if not task_root and os.environ.get("RUN_ROOT"):
        task_root = os.path.join(os.environ["RUN_ROOT"], "profile", "out", "tasks")
    tasks_dir = args.tasks_dir or (Path(task_root) if task_root else None)
    optimized = args.optimized or (Path(out) / "optimized.json" if out else None)

    try:
        limit = args.task_limit if args.task_limit is not None else params.get("task_limit", 8)
        valid_limit(limit)
        if tasks_dir is None or not tasks_dir.is_dir():
            raise ValueError("no extracted task directory to account against (%s); the "
                             "profile stage owns it" % (tasks_dir or "$RUN_ROOT/profile/out/tasks"))
        tasks = sorted(p.name for p in tasks_dir.iterdir() if p.is_dir())
        if not tasks:
            raise ValueError("No extracted task directories were found")
        result = {"tasks_extracted": len(tasks), "task_limit": limit,
                  "tasks_within_capacity": min(len(tasks), limit),
                  "tasks_over_capacity": max(0, len(tasks) - limit)}
        if optimized is None or not optimized.exists():
            raise ValueError("the stage left no optimized.json; nothing accounts for the "
                             "%d extracted tasks" % len(tasks))
        required = params.get("time_budget_sec", 172800) if params.get("require_full_budget") else None
        result.update(check_report(tasks, json.loads(optimized.read_text()), limit,
                                   required_seconds=required,
                                   work_root=optimized.parent / "work"))
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        if not out:
            parser.exit(1, "Task scope: %s\n" % exc)
        # An absent delivery is a precondition nobody satisfied; a delivery that does not
        # add up is work that was attempted and did not hold (CONTRACT section 3).
        missing = isinstance(exc, OSError) or "no extracted task directory" in str(exc) \
            or "left no optimized.json" in str(exc)
        return emit("blocked" if missing else "fail", notes="Task scope: %s" % exc)

    if not out:
        print(json.dumps(result, indent=2))
        return 0
    artifacts = {"optimized": os.path.relpath(str(optimized), out)}
    for name in ("REPORT.md", "work"):
        if os.path.exists(os.path.join(out, name)):
            artifacts[name.split(".")[0].lower()] = name
    notes = ("%d of %d extracted tasks optimized, %d kept, %d capacity-deferred"
             % (result["tasks_attempted"], result["tasks_extracted"],
                result["tasks_kept"], result["tasks_capacity_deferred"]))
    if result["unattempted_tasks"]:
        notes += "; unattempted: " + ", ".join(result["unattempted_tasks"])
    return emit("ok", artifacts=artifacts, metrics=result, notes=notes)


if __name__ == "__main__":
    raise SystemExit(main())
