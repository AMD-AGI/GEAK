#!/usr/bin/env python3
"""The profile slot's verdict on what the skill left in `$OUT`.

It belongs to the **slot**, not to any implementation. Swapping in another profile skill
requires no edit here, as long as the new skill produces the files named in the table in
README.md. It reads `$OUT` and writes `kernels.json` (kernels.py) and `result.json`.
Task boundaries and UTs are not profile deliveries; the UT slot judges those.

It used to be a format converter: count the files, map them into `artifacts`, say `ok`.
That was not enough, and a side-by-side comparison of one pipeline run against the same
skill driven by hand showed exactly where the slack was:

  * **The configured scope was never checked against what ran.** A run whose `$PARAMS`
    asked for phase Top 10 and overall share >0.5% was finalized `ok` while
    `selection_targets.json` recorded the skill's own SKILL.md default, Top 5 and >3%.
    Nothing compared the two numbers, so a quarter of the candidate space silently
    disappeared and the stage reported success.

So the rule this file implements is: what the configuration asked for must be what ran,
and every claim is carried by a file somebody can open. `$PARAMS` is the executable
policy; a skill's own documented default never overrides it (stages/CONTRACT.md).
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "_shared"))
sys.path.insert(0, HERE)
from emit_result import emit, read_json  # noqa: E402
from select_candidates import selector_args  # noqa: E402

REQUIRED = ["kernel_ranking.json", "selection_targets.json"]
INVENTORY = ("selection_scope", "candidate_inventory", "coverage_summary")

SEVERITY = {"ok": 0, "blocked": 1, "fail": 2}


def worse(a, b):
    return a if SEVERITY[a] >= SEVERITY[b] else b


def stage_params(out):
    """What was actually asked of this stage.

    `$PARAMS` is the merged parameter file `run.py prepare` wrote; `$CONFIG` is the
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


def main():
    out = os.environ.get("OUT")
    if not out:
        print("OUT is not set; run run.py prepare profile first", file=sys.stderr)
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

    if not targets:
        status = worse(status, "fail")
        notes.append("targets in selection_targets.json is empty. Missing evidence is not "
                     "the same as an empty target set.")

    kernels = None
    if status != "fail":
        from kernels import build, write
        try:
            kernels = build(out, params)
            write(os.path.join(out, "kernels.json"), kernels)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            status = worse(status, "blocked")
            notes.append("kernels.json cannot be exported: %s" % exc)

    artifacts = {k.split(".")[0]: k for k in REQUIRED}
    for opt in ("kernels.json", "kernel_workload.json", "REPORT.md"):
        if os.path.exists(os.path.join(out, opt)):
            artifacts[opt.split(".")[0].lower()] = opt

    metrics = {
        "kernels_censused": len(ranking.get("ranking") or []),
        "targets_selected": len(targets),
        "selected_pct_gpu_time": round(sum(shares.values()), 5),
        "census_denominator_ms": census_ms,
        "census_denominator_source": census_source,
        "selection_top_k": scope[0],
        "selection_overall_pct_gt": scope[1],
        "points_measured": len(kernels["points"]) if kernels else None,
        "kernels_with_gaps": sum(bool(k["gaps"]) for k in kernels["kernels"]) if kernels else None,
        "acceptance_inventory_completion": not lacking,
    }

    return emit(status=status, artifacts=artifacts, metrics=metrics,
                notes="; ".join(notes))


if __name__ == "__main__":
    sys.exit(main())
