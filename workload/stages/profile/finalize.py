#!/usr/bin/env python3
"""The profile slot's verdict on what the skill left in `$OUT`.

It belongs to the **slot**, not to any implementation. Swapping in another profile skill
requires no edit here, as long as the new skill produces the files named in the table in
README.md. It only reads `$OUT` and writes nothing but `result.json`.

It used to be a format converter: count the files, map them into `artifacts`, say `ok`.
That was not enough, and a side-by-side comparison of one pipeline run against the same
skill driven by hand showed exactly where the slack was:

  * **The configured scope was never checked against what ran.** A run whose `$PARAMS`
    asked for phase Top 10 and overall share >0.5% was finalized `ok` while
    `selection_targets.json` recorded the skill's own SKILL.md default, Top 5 and >3%.
    Nothing compared the two numbers, so a quarter of the candidate space silently
    disappeared and the stage reported success.
  * **Nothing read the per-task UT evidence.** "The gates were never run" and "every gate
    is green" produced identical `result.json` files.
  * **Capacity was enforced against extraction.** `task_limit` is an eight-GPU campaign
    scheduling policy; applied here it made a complete ten-task delivery structurally
    impossible, which is a narrower delivery, not a smaller one.

So the rule this file implements is: what the configuration asked for must be what ran,
and every claim is carried by a file somebody can open. `$PARAMS` is the executable
policy; a skill's own documented default never overrides it (stages/CONTRACT.md).
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)) + "/scripts")
sys.path.insert(0, HERE)
from emit_result import emit, read_json  # noqa: E402
from select_candidates import selector_args  # noqa: E402
from task_plan import selected_identities, task_records  # noqa: E402

REQUIRED = ["kernel_ranking.json", "selection_targets.json"]
INVENTORY = ("selection_scope", "candidate_inventory", "coverage_summary")
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
    """What was actually asked of this stage.

    `$PARAMS` is the merged parameter file `stage.py prepare` wrote; `$CONFIG` is the
    fallback so this script can be run against a finished `$OUT` from anywhere."""
    params = read_json(os.environ.get("PARAMS") or os.path.join(out, "params.json"),
                       default={})
    if params:
        return params
    config = read_json(os.environ.get("CONFIG", ""), default={})
    stage = os.environ.get("STAGE", "profile")
    return config.get("stages", {}).get(stage, {}).get("params", {}) or {}


def configured_scope(params):
    """(top_k, threshold, phases) exactly as `select_candidates.py` would pass them.

    Reusing `selector_args` is the point: the slot verifies the same arguments it would
    have sent, so parameter validation cannot drift between the two sides."""
    args = selector_args(params)
    return int(args[1]), float(args[3]), list(args[5:])


def _scope_from_args(args):
    """(top_k, threshold, phases) read back out of a recorded selector argv."""
    if not isinstance(args, list) or any(not isinstance(a, str) for a in args):
        raise ValueError("selector arguments must be a list of strings")
    values = {}
    for flag in ("--top-k", "--overall-pct-gt", "--phases"):
        if args.count(flag) != 1:
            raise ValueError("selector arguments must record %s exactly once" % flag)
        start = args.index(flag) + 1
        end = next((i for i in range(start, len(args)) if args[i].startswith("--")),
                   len(args))
        values[flag] = args[start:end]
        if not values[flag] or (flag != "--phases" and len(values[flag]) != 1):
            raise ValueError("selector arguments have invalid values for %s" % flag)
    try:
        top_k = int(values["--top-k"][0])
        threshold = float(values["--overall-pct-gt"][0])
    except (TypeError, ValueError):
        raise ValueError("selector arguments must record numeric selection cutoffs") from None
    # Reuse the slot's validation, including finite thresholds and unique phase names.
    return configured_scope({"selection_top_k": top_k,
                             "selection_overall_pct_gt": threshold,
                             "selection_phases": values["--phases"]})


def _describe(top_k, threshold, phases):
    return "phase top %s, overall share > %s%%%s" % (
        top_k, threshold, (", phases %s" % (phases,)) if phases else "")


def scope_findings(selection, ranking, scope):
    """Did the configured selection scope actually reach the selector?

    Three witnesses, in decreasing authority: the `policy` block the selector itself
    wrote into its output; the `selector.args` the skill recorded in its
    `selection_scope`; and the `selection_invocation` receipt `select_candidates.py`
    stamps on its own output. The receipt is optional -- a skill driven directly does not
    go through that adapter -- but a receipt that disagrees is a finding, not a detail."""
    top_k, threshold, phases = scope
    want = _describe(top_k, threshold, phases)
    findings = []

    policy = selection.get("policy")
    if not isinstance(policy, dict):
        return [("blocked",
                 "selection_targets.json carries no policy block, so nothing shows that "
                 "the configured scope (%s) reached the selector. The selector records "
                 "the policy it applied; a target list without one is unattributable."
                 % want)]
    got = (policy.get("top_k_per_phase"), policy.get("overall_pct_gt"),
           policy.get("phases"))
    missing = [key for key in ("top_k_per_phase", "overall_pct_gt", "phases")
               if key not in policy]
    if missing:
        findings.append(("blocked", "selection_targets.json.policy is missing %s; "
                         "the configured scope is %s and every selection parameter "
                         "needs a receipt." % (missing, want)))
    else:
        try:
            configured_scope({"selection_top_k": got[0],
                              "selection_overall_pct_gt": got[1],
                              "selection_phases": got[2]})
        except ValueError as exc:
            findings.append(("fail", "selection_targets.json.policy records invalid "
                             "%s; configured %s: %s" % (_describe(*got), want, exc)))
    if not missing and got != scope:
        findings.append((
            "fail",
            "the selector ran at %s but this stage was configured for %s. $PARAMS is the "
            "executable policy and the skill's SKILL.md default does not override it: "
            "rerun the selection through stages/profile/select_candidates.py at the "
            "configured scope, or change the configuration and say why."
            % (_describe(*got), want)))

    witnesses = []
    declared_scope = ranking.get("selection_scope")
    selector = declared_scope.get("selector") if isinstance(declared_scope, dict) else None
    if isinstance(selector, dict) and "args" in selector:
        witnesses.append(("kernel_ranking.json.selection_scope.selector.args", selector["args"]))
    if "selection_invocation" in selection:
        receipt = selection["selection_invocation"]
        witnesses.append(("selection_targets.json.selection_invocation.arguments",
                          receipt.get("arguments") if isinstance(receipt, dict) else None))
    for source, arguments in witnesses:
        try:
            was = _scope_from_args(arguments)
        except ValueError as exc:
            findings.append(("fail", "%s cannot establish the configured %s: %s; "
                             "recorded arguments=%r" % (source, want, exc, arguments)))
            continue
        if was != scope:
            findings.append(("fail", "%s records %s, not the configured %s. The "
                             "ranking and target list must agree about what was selected."
                             % (source, _describe(*was), want)))
    return findings


def denominator(out, ranking):
    """(milliseconds, source) for the share denominator, or (None, None).

    Two shapes are accepted because both are real: a skill that writes a separate
    `kernel_workload.json` census, and a skill that reuses a prior census and carries its
    denominators inside the ranking. What the slot requires is that the number every
    share is a fraction of can be named, not that a particular file exists."""
    workload = read_json(os.path.join(out, "kernel_workload.json"), default={})
    for key in ("total_gpu_time_ms", "total_device_ms", "total_ms"):
        value = workload.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            return float(value), "kernel_workload.json:%s" % key
    denominators = ranking.get("denominators") or {}
    for key in ("total_device_ms", "total_gpu_time_ms", "total_ms"):
        value = denominators.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            return float(value), "kernel_ranking.json:denominators.%s" % key
    return None, None


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
        print("OUT is not set; run stage.py prepare profile first", file=sys.stderr)
        return 2

    missing = [f for f in REQUIRED if not os.path.exists(os.path.join(out, f))]
    if missing:
        return emit(status="fail", artifacts={},
                    notes="required artifacts are missing: %s. The skill's §2 (census and "
                          "target selection) did not finish; complete it and rerun this "
                          "finalize." % missing)

    selection = read_json(os.path.join(out, "selection_targets.json"))
    ranking = read_json(os.path.join(out, "kernel_ranking.json"))
    targets = selection.get("targets") or []
    shares = {t.get("id") or t.get("name"): float(t.get("overall_pct") or 0.0)
              for t in targets}

    status, notes = "ok", []
    params = stage_params(out)
    try:
        scope = configured_scope(params)
    except ValueError as exc:
        return emit(status="fail", artifacts={},
                    notes="this stage's configured selection scope is itself invalid, so "
                          "nothing can be verified against it: %s" % exc)
    for severity, note in scope_findings(selection, ranking, scope):
        status = worse(status, severity)
        notes.append(note)

    census_ms, census_source = denominator(out, ranking)
    if census_ms is None:
        status = worse(status, "blocked")
        notes.append("no census denominator can be resolved: neither "
                     "kernel_workload.json nor kernel_ranking.json.denominators carries a "
                     "positive total device time. Every share in this delivery is a "
                     "fraction of a number that must be nameable.")

    lacking = [key for key in INVENTORY if not ranking.get(key)]
    if lacking:
        status = worse(status, "blocked")
        notes.append("kernel_ranking.json lacks the required inventory sections: %s. "
                     "Account for unselected and deferred work before claiming extraction "
                     "complete." % list(lacking))

    tasks_dir = os.path.join(out, "tasks")
    names = sorted(n for n in (os.listdir(tasks_dir) if os.path.isdir(tasks_dir) else [])
                   if os.path.isdir(os.path.join(tasks_dir, n)))
    evidence = [task_evidence(out, n) for n in names]
    broken = [e for e in evidence if e["lack"]]
    ut_green = [e for e in evidence if not e["lack"] and not e["ut"]]
    gate_green = [e for e in evidence if not e["lack"] and not e["gate"]]

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
        status = worse(status, "blocked")
        notes.append("results.json reports no achieved coverage "
                     "(coverage.delivered_achieved_pct_gpu_time, or per-identity "
                     "achieved_pct_gpu_time). Owning a boundary is not the same as timing "
                     "it: a task can own a kernel its captured operands never launch, and "
                     "without that number the delivery cannot be told apart from one that "
                     "covered everything it declared.")
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

    if not targets:
        status = worse(status, "fail")
        notes.append("targets in selection_targets.json is empty. Missing evidence is not "
                     "the same as an empty target set.")

    artifacts = {k.split(".")[0]: k for k in REQUIRED}
    for opt in ("kernel_workload.json", "task_plan.json", "REPORT.md", "results.json"):
        if os.path.exists(os.path.join(out, opt)):
            artifacts[opt.split(".")[0].lower()] = opt
    if names:
        artifacts["tasks"] = "tasks"

    metrics = {
        "kernels_censused": len(ranking.get("ranking") or []),
        "targets_selected": len(targets),
        "tasks_built": len(names),
        "selected_pct_gpu_time": round(sum(shares.values()), 5),
        "census_denominator_ms": census_ms,
        "census_denominator_source": census_source,
        "selection_top_k": scope[0],
        "selection_overall_pct_gt": scope[1],
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
        # SKILL.md §5: four outcomes, reported separately. A subset of passing tasks is
        # not full acceptance, and none of these four may stand in for another.
        "acceptance_inventory_completion": not unaccounted and not lacking,
        "acceptance_task_completion": not broken and not undelivered and bool(names),
        "acceptance_gpu_ut_status": bool(names) and len(ut_green) == len(names),
        "acceptance_delivery_consistency": bool(names) and len(gate_green) == len(names),
    }

    return emit(status=status, artifacts=artifacts, metrics=metrics,
                notes="; ".join(notes))


if __name__ == "__main__":
    sys.exit(main())
