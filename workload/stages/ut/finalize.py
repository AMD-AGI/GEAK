#!/usr/bin/env python3
"""The UT slot's verdict on what the skill left in `$OUT`.

The UT stage writes and qualifies the tasks its request names (`ut_request.json`, from
whoever decided the task boundaries); it reads no profile evidence. This checks that every
requested task exists and carries its own receipts, and that every selected identity in
the request is owned by a task directory or carries a stated blocker:

  * **Nothing but receipts counts.** "The gates were never run" and "every gate is green"
    must not produce the same `result.json`: each task needs `run/ut_result.json` (its own
    verdict over its gates) and `run/RETURN.json` (the delivery gate, written only when its
    report is ok).
  * **Capacity is not extraction.** `task_limit` is an optimize-side scheduling policy; a
    complete delivery is never truncated to fit it.

It reads `$OUT` and writes nothing but `result.json`.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "_shared"))
from emit_result import emit, read_json  # noqa: E402
from task_plan import selected_identities, task_records  # noqa: E402

TASK_FILES = ["unittest.py", "meta.json", "cases.py"]
UT_RESULT = os.path.join("run", "ut_result.json")
DELIVERY_RETURN = os.path.join("run", "RETURN.json")
# Percentage points. Shares are reported to five decimals; this is a rounding allowance,
# not a tolerance for a different number.
EPSILON = 0.01

SEVERITY = {"ok": 0, "blocked": 1, "fail": 2}


def worse(a, b):
    return a if SEVERITY[a] >= SEVERITY[b] else b


def stage_params(out):
    params = read_json(os.environ.get("PARAMS") or os.path.join(out, "params.json"),
                       default={})
    if params:
        return params
    config = read_json(os.environ.get("CONFIG", ""), default={})
    return config.get("stages", {}).get(os.environ.get("STAGE", "ut"), {}).get("params", {}) or {}


def has_task_dir(out, name):
    return (isinstance(name, str) and name and name == os.path.basename(name)
            and os.path.isdir(os.path.join(out, "tasks", name, "kernel_src")))


def disposition(out, targets):
    """Who owns each selected identity, and which ones carry a stated blocker.

    `results.json` is the skill's final delivery record (SKILL.md §5) and outranks
    `task_plan.json`, which is a plan: in the hand-driven delivery the plan still named
    one task that the record had since split into a delivered task and a blocked
    identity. The plan is the fallback for a run that has not reached §5 yet.

    An identity named to a task directory that does not exist stays unaccounted for. A
    name in a plan is not a delivery."""
    ids = [t.get("id") or t.get("name") for t in targets]
    results = read_json(os.path.join(out, "results.json"), default={})
    rows = results.get("identities")
    if isinstance(rows, list) and rows:
        named, blocked = {}, set()
        for row in rows:
            ident = row.get("id") or row.get("device_kernel")
            task = row.get("task")
            if not ident:
                continue
            if isinstance(task, str) and task:
                named[ident] = task
            elif (str(row.get("why") or row.get("blocker") or "")).strip():
                blocked.add(ident)
        if all(i in named or i in blocked for i in ids):
            owned = {i: t for i, t in named.items() if has_task_dir(out, t)}
            declared = [r.get("task") for r in results.get("tasks") or []
                        if isinstance(r.get("task"), str)]
            return {"owned": owned, "blocked": blocked, "source": "results.json",
                    "declared_tasks": sorted(set(declared) or set(named.values()))}

    plan = read_json(os.path.join(out, "task_plan.json"), default={})
    owned, blocked, declared = {}, set(), []
    for name, identities in task_records(plan):
        declared.append(name)
        if not has_task_dir(out, name):
            continue
        for ident in identities:
            if ident in owned and owned[ident] != name:
                raise ValueError("selected identity %s is assigned to multiple task "
                                 "owners: %s, %s" % (ident, owned[ident], name))
            owned[ident] = name
    for row in plan.get("blocked_identities") or []:
        if row.get("kernel") and (row.get("blocker") or "").strip():
            blocked.add(row["kernel"])
        elif row.get("id") and (row.get("blocker") or row.get("why") or "").strip():
            blocked.add(row["id"])
    for row in plan.get("blocked") or []:
        if (row.get("reason") or "").strip():
            blocked.update(selected_identities(row))
    return {"owned": owned, "blocked": blocked, "source": "task_plan.json",
            "declared_tasks": sorted({n for n in declared if isinstance(n, str)})}


def task_evidence(out, name):
    """What `tasks/<name>/` is missing: structure, UT verdict, delivery gate.

    Both receipts are required. `run/ut_result.json` is the task's own verdict over its
    gates, and `run/RETURN.json` is written by `kernel_extraction_delivery.mjs` only when
    its report is `ok`, so an absent one means the delivery gate did not pass here --
    which is not the same claim as the gates passing, and neither substitutes for the
    other."""
    td = os.path.join(out, "tasks", name)
    lack = [f for f in TASK_FILES if not os.path.exists(os.path.join(td, f))]
    if not os.path.isdir(os.path.join(td, "kernel_src")):
        lack.append("kernel_src/")

    ut = None
    if not os.path.exists(os.path.join(td, UT_RESULT)):
        ut = ("%s has no %s, so nothing shows its gates were run. A task whose UT was "
              "never executed and a task whose UT is green must not read the same."
              % (name, UT_RESULT))
    else:
        doc = read_json(os.path.join(td, UT_RESULT), default={})
        failed = doc.get("failed_gates")
        if doc.get("ok") is not True or (failed not in (None, [], {})):
            ut = ("%s/%s reports ok=%r with failed_gates=%r" % (name, UT_RESULT,
                                                                doc.get("ok"), failed))

    gate = None
    if not os.path.exists(os.path.join(td, DELIVERY_RETURN)):
        gate = ("%s has no %s; the delivery gate writes that file only when its report "
                "is ok, so its absence means the gate did not pass for this task"
                % (name, DELIVERY_RETURN))
    else:
        doc = read_json(os.path.join(td, DELIVERY_RETURN), default={})
        short = doc.get("short_name")
        if not doc:
            gate = "%s/%s is empty or is not a JSON object" % (name, DELIVERY_RETURN)
        elif isinstance(short, str) and short and short != name:
            gate = ("%s/%s names short_name=%r, so it is another task's delivery record"
                    % (name, DELIVERY_RETURN, short))
    return {"name": name, "lack": lack, "ut": ut, "gate": gate}


def reported_coverage(out):
    """(declared, achieved) percentages of whole-run GPU time, as the skill measured them.

    Declared coverage is what the delivered tasks own. Achieved coverage is what their
    captured operands actually launched. They are different numbers -- one delivery
    declared 42.92% and timed 22.54% of it, because a task's captured decode operands
    launched neither of the two kernels it owned -- and the slot derives neither from the
    other."""
    results = read_json(os.path.join(out, "results.json"), default={})
    cov = results.get("coverage") or {}
    declared = cov.get("delivered_declared_pct_gpu_time")
    achieved = cov.get("delivered_achieved_pct_gpu_time")
    rows = [r for r in results.get("identities") or []
            if r.get("status") == "delivered"]
    if achieved is None and rows and all("achieved_pct_gpu_time" in r for r in rows):
        achieved = sum(r.get("achieved_pct_gpu_time") or 0.0 for r in rows)
    if declared is None and rows and all("declared_pct_gpu_time" in r for r in rows):
        declared = sum(r.get("declared_pct_gpu_time") or 0.0 for r in rows)
    numeric = []
    for value in (declared, achieved):
        numeric.append(float(value)
                       if isinstance(value, (int, float)) and not isinstance(value, bool)
                       else None)
    return numeric[0], numeric[1]


def main():
    out = os.environ.get("OUT")
    if not out:
        print("OUT is not set; run run.py prepare ut first", file=sys.stderr)
        return 2
    params = stage_params(out)
    if not os.path.exists(os.path.join(out, "task_plan.json")):
        return emit(status="blocked", artifacts={},
                    notes="no task_plan.json: the stage installs its tasks from a ut_request.json "
                          "(scripts/scaffold_tasks.py); no request was installed. Task boundaries "
                          "are decided upstream, not here.")
    selection = read_json(os.path.join(out, "selection_targets.json"), default={})
    plan = read_json(os.path.join(out, "task_plan.json"), default={})
    targets = selection.get("targets") or [
        {"id": i} for _n, ids in task_records(plan) for i in ids] + [
        {"id": b.get("kernel")} for b in plan.get("blocked_identities") or [] if b.get("kernel")]
    shares = {t.get("id") or t.get("name"): float(t.get("overall_pct") or 0.0)
              for t in targets}
    status, notes = "ok", []

    tasks_dir = os.path.join(out, "tasks")
    names = sorted(n for n in (os.listdir(tasks_dir) if os.path.isdir(tasks_dir) else [])
                   if os.path.isdir(os.path.join(tasks_dir, n)))
    if not names:
        status = worse(status, "blocked")
        notes.append("no task directory was generated; accounted-for blockers are not a UT "
                     "delivery")
    evidence = [task_evidence(out, n) for n in names]
    broken = [e for e in evidence if e["lack"]]
    ut_green = [e for e in evidence if not e["lack"] and not e["ut"]]
    gate_green = [e for e in evidence if not e["lack"] and not e["gate"]]
    qualified = [e for e in evidence if not e["lack"] and not e["ut"] and not e["gate"]]
    minimum = params.get("minimum_qualified_tasks", 0)
    if type(minimum) is not int or minimum < 0:
        return emit(status="fail", artifacts={},
                    notes="minimum_qualified_tasks must be a nonnegative integer")
    if len(qualified) < minimum:
        status = worse(status, "blocked")
        notes.append("configured minimum_qualified_tasks=%d, observed fully qualified tasks=%d; "
                     "complete the supported nonoverlapping delivery before optimization"
                     % (minimum, len(qualified)))

    if broken:
        status = worse(status, "blocked")
        notes.append("these task directories are incomplete and the optimize stage cannot "
                     "run on them: " +
                     "; ".join("%s is missing %s" % (e["name"], e["lack"]) for e in broken))
    ut_problems = [e["ut"] for e in evidence if e["ut"] and not e["lack"]]
    if ut_problems:
        status = worse(status, "blocked")
        notes.append("GPU UT evidence is missing or not green: " + "; ".join(ut_problems))
    gate_problems = [e["gate"] for e in evidence if e["gate"] and not e["lack"]]
    if gate_problems:
        status = worse(status, "blocked")
        notes.append("delivery-gate evidence is missing or inconsistent: "
                     + "; ".join(gate_problems))

    try:
        owner = disposition(out, targets)
    except ValueError as exc:
        status = worse(status, "blocked")
        notes.append(str(exc))
        owner = {"owned": {}, "blocked": set(), "declared_tasks": [], "source": "none"}
    owned_ids = [i for i in shares if i in owner["owned"]]
    blocked_ids = [i for i in shares if i not in owner["owned"] and i in owner["blocked"]]
    unaccounted = [i for i in shares if i not in owner["owned"] and i not in owner["blocked"]]
    if unaccounted:
        status = worse(status, "blocked")
        notes.append("these selected identities have neither an existing task directory "
                     "nor a stated blocker in %s: %s. A blocked identity is recorded, "
                     "never renormalized away." % (owner["source"], unaccounted))

    undelivered = [n for n in owner["declared_tasks"] if n not in names]
    if undelivered:
        status = worse(status, "blocked")
        notes.append("%s declares these tasks, which have no directory under tasks/: %s"
                     % (owner["source"], undelivered))

    owned_pct = sum(shares[i] for i in owned_ids)
    blocked_pct = sum(shares[i] for i in blocked_ids)
    declared_pct, achieved_pct = reported_coverage(out)
    if achieved_pct is None:
        notes.append("UT acceptance uses the frozen case and source-delivery receipts; "
                     "whole-serving coverage is not reported here and is not inferred from "
                     "a UT pass.")
    else:
        if declared_pct is None:
            declared_pct = owned_pct
        if achieved_pct > declared_pct + EPSILON:
            status = worse(status, "blocked")
            notes.append("achieved coverage %.5f%% exceeds declared coverage %.5f%%; "
                         "a delivery cannot time more GPU share than it owns."
                         % (achieved_pct, declared_pct))
        if declared_pct > owned_pct + EPSILON:
            status = worse(status, "blocked")
            notes.append("results.json declares %.5f%% of GPU time covered, but the "
                         "identities with an existing task directory hold %.5f%% in "
                         "selection_targets.json." % (declared_pct, owned_pct))

    # Capacity is an optimize-side scheduling policy, not an extraction cap: a complete
    # delivery is never truncated to fit eight GPUs, and the remainder is carried with its
    # reason rather than deleted.
    config = read_json(os.environ.get("CONFIG", ""), default={})
    limit = config.get("stages", {}).get("optimize", {}).get("params", {}).get("task_limit", 8)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        limit = 8
    deferred = max(0, len(names) - limit)
    if deferred:
        notes.append("%d tasks were extracted and the campaign's optimize capacity is "
                     "task_limit=%d, so %d will be scheduled in a later wave; the optimize "
                     "stage picks which by roofline order and records the rest as "
                     "capacity_deferred. This is not a defect in the extraction."
                     % (len(names), limit, deferred))

    artifacts = {"task_plan": "task_plan.json"}
    for opt in ("selection_targets.json", "REPORT.md", "results.json"):
        if os.path.exists(os.path.join(out, opt)):
            artifacts[opt.split(".")[0].lower()] = opt
    if names:
        artifacts["tasks"] = "tasks"

    metrics = {
        "tasks_built": len(names),
        "identities_owned_by_tasks": len(owned_ids),
        "identities_blocked": len(blocked_ids),
        "owned_pct_gpu_time": round(owned_pct, 5),
        "blocked_pct_gpu_time": round(blocked_pct, 5),
        "declared_pct_gpu_time": None if declared_pct is None else round(declared_pct, 5),
        "achieved_pct_gpu_time": None if achieved_pct is None else round(achieved_pct, 5),
        "tasks_within_capacity": min(len(names), limit),
        "tasks_capacity_deferred": deferred,
        "tasks_ut_passed_all_gates": len(ut_green),
        "tasks_delivery_gate_passed": len(gate_green),
        "minimum_qualified_tasks": minimum,
        "tasks_fully_qualified": len(qualified),
        "required_ut_validation": "correctness",
        # Four outcomes, reported separately. A subset of passing tasks is not full
        # acceptance, and none of these four may stand in for another.
        "acceptance_inventory_completion": not unaccounted,
        "acceptance_task_completion": not broken and not undelivered and bool(names),
        "acceptance_gpu_ut_status": bool(names) and len(ut_green) == len(names),
        "acceptance_delivery_consistency": bool(names) and len(gate_green) == len(names),
    }
    return emit(status=status, artifacts=artifacts, metrics=metrics, notes="; ".join(notes))


if __name__ == "__main__":
    sys.exit(main())
