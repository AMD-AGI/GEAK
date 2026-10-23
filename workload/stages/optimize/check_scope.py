#!/usr/bin/env python3
"""Check task names and campaign capacity, without judging any kernel measurement."""

import argparse
import json
import os
from pathlib import Path


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
                         "are unused: %s. Capacity defers what does not fit, in roofline "
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--optimized", type=Path)
    parser.add_argument("--task-limit", type=int)
    args = parser.parse_args()
    try:
        params = json.loads(Path(os.environ["PARAMS"]).read_text()) if os.environ.get("PARAMS") else {}
        limit = args.task_limit if args.task_limit is not None else params.get("task_limit", 8)
        tasks = sorted(p.name for p in args.tasks_dir.iterdir() if p.is_dir())
        if not tasks:
            raise ValueError("No extracted task directories were found")
        valid_limit(limit)
        result = {"tasks_extracted": len(tasks), "tasks": tasks, "task_limit": limit,
                  "tasks_within_capacity": min(len(tasks), limit),
                  "tasks_over_capacity": max(0, len(tasks) - limit)}
        if args.optimized:
            required = params.get('time_budget_sec', 172800) if params.get('require_full_budget') else None
            result.update(check_report(tasks, json.loads(args.optimized.read_text()), limit,
                                       required_seconds=required, work_root=args.optimized.parent/'work'))
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        parser.exit(1, "Task scope: %s\n" % exc)


if __name__ == "__main__":
    raise SystemExit(main())
