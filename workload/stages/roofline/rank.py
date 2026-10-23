#!/usr/bin/env python3
"""The roofline slot's finalize adapter: fold, compute expected_e2e_gain_pct, order the work.

It belongs to the slot, not to any one implementation. It deliberately does **not** write
its own folding sum — there is exactly one correct way to fold many shapes into a single
number, the implementation skill ships it (`scripts/fold.py`), and this file shells out to
it. Swapping in a different roofline implementation only requires that it also provide
`scripts/fold.py` and `rows.json`; this file does not change.

Usage (inside the environment prepared by stage.py prepare):
    python3 stages/roofline/rank.py [--roof-key roofline_us] [--rows rows.json]
"""

import argparse
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))          # stages/roofline
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "scripts"))
from emit_result import emit, read_json  # noqa: E402
from task_plan import task_records  # noqa: E402


def positive(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def case_weight(row):
    value = row.get("case_weight")
    if value is None:
        return 1.0
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value) or value < 0):
        return None
    return value


def regime_weights(rows):
    """Spread each kernel's time across regimes by measured_us. Time domain, same basis as the fold."""
    per = {}
    for r in rows:
        k, g = r.get("kernel"), r.get("regime")
        us = r.get("measured_us")
        if k is None or g is None or not positive(us):
            continue
        w = case_weight(r)
        if w is None or w == 0:
            continue
        per.setdefault(k, {}).setdefault(g, 0.0)
        per[k][g] += w * us
    out = {}
    for k, regs in per.items():
        total = sum(regs.values())
        out[k] = {g: (v / total if total else 0.0) for g, v in regs.items()}
    return out


def bound_by_time(rows):
    """The bound label holding the majority of the time for each (kernel, regime). A
    model's classification, not proof of the measured bottleneck."""
    acc = {}
    for r in rows:
        b = r.get("bound")
        us = r.get("measured_us")
        w = case_weight(r)
        if not b or not positive(us) or w is None or w == 0:
            continue
        key = (r.get("kernel"), r.get("regime"))
        acc.setdefault(key, {}).setdefault(b, 0.0)
        acc[key][b] += w * us
    return {k: max(v.items(), key=lambda kv: kv[1])[0] for k, v in acc.items()}


def task_dir_index(plan):
    """task_plan.json -> {selected identity: task directory name}.

    Empty when the upstream skill declared no plan, which leaves the caller on its
    name-matching fallback."""
    idx = {}
    for name, identities in task_records(plan):
        if not name:
            continue
        for k in identities:
            idx.setdefault(k, name)
    return idx


def target_index(sel):
    """selection_targets.json -> {possible name: target}. Both id and name are registered
    so the rows have something to match on."""
    idx = {}
    for t in sel.get("targets") or []:
        for key in (t.get("id"), t.get("name")):
            if key:
                idx.setdefault(key, t)
    return idx


def task_analysis(task, rows, roof_key="roofline_us"):
    """Separate completed unsupported modeling from missing task/case inputs."""
    meta = read_json(os.path.join(task, "meta.json"), {})
    workload = meta.get("workload") or {}
    cases = workload.get("cases") if isinstance(workload, dict) else None
    problems = []
    if not os.path.isdir(os.path.join(task, "kernel_src")) or not isinstance(cases, list) or not cases:
        problems.append("task source or workload case inventory is missing")
    cases = cases if isinstance(cases, list) else []
    expected = {c.get("sig") for c in cases or [] if positive(c.get("effective_calls"))}
    present = {r.get("case") for r in rows if positive(case_weight(r))}
    missing = sorted(expected - present)
    if missing:
        problems.append("positive-mass cases have no analysis row: " + ", ".join(missing))
    if not rows:
        problems.append("task has no analysis rows")
    active = [r for r in rows if case_weight(r) is None or case_weight(r) > 0]
    unsupported = []
    inadmissible = []
    suspect = []
    for r in active:
        label = str(r.get("case"))
        if case_weight(r) is None:
            problems.append("invalid case weight: " + label)
        provenance = r.get("provenance") or {}
        if provenance.get("qualification_refresh_required") is True:
            problems.append("task qualification needs refresh: " + label)
        modeled = positive(r.get("measured_us")) and positive(r.get(roof_key))
        if not modeled:
            unsupported.append(label)
            if (not positive(r.get(roof_key)) and not r.get("reason")
                    and r.get("status") not in ("unsupported", "native_local_bytes_only")):
                problems.append("case has neither a model result nor an unsupported explanation: " + label)
        coverage = r.get("coverage")
        suspect_case = r.get("suspect") is True or (modeled and r[roof_key] > r["measured_us"])
        if suspect_case:
            suspect.append(label)
        if (r.get("ranking_admissible") is False or not modeled or suspect_case
                or (coverage is not None and (not positive(coverage) or coverage != 1))):
            inadmissible.append(label)
    complete = not problems
    admissible = complete and bool(active) and not inadmissible
    return {"processed": complete,
            "ranking_admissible": admissible,
            "status": ("incomplete_inputs" if not complete else
                       "completed" if admissible else "completed_unsupported"),
            "expected_positive_cases": len(expected), "missing_positive_cases": missing,
            "unsupported_positive_rows": unsupported, "inadmissible_positive_rows": inadmissible,
            "suspect_positive_rows": suspect,
            "problems": problems}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", default="rows.json", help="per-case table, relative to $OUT")
    ap.add_argument("--roof-key", default=None,
                    help="name of the per-case roof column; defaults to params.roof_key, then roofline_us")
    args = ap.parse_args(argv)

    out = os.environ.get("OUT")
    inp = os.environ.get("IN")
    skill_root = os.environ.get("SKILL_ROOT")
    if not out or not skill_root:
        print("OUT/SKILL_ROOT missing; run stage.py prepare roofline first", file=sys.stderr)
        return 2
    params = read_json(os.environ.get("PARAMS", ""), {})
    roof_key = args.roof_key or params.get("roof_key") or "roofline_us"

    rows_path = os.path.join(out, args.rows)
    if not os.path.exists(rows_path):
        return emit(status="fail", artifacts={},
                    notes="%s is missing. A roofline implementation must produce a "
                          "per-case rows.json (kernel/regime/case/measured_us/%s); see "
                          "this stage's README." % (args.rows, roof_key))

    fold = os.path.join(skill_root, "scripts", "fold.py")
    if not os.path.exists(fold):
        return emit(status="fail", artifacts={},
                    notes="implementation %s has no scripts/fold.py. The folding algorithm "
                          "must come from the implementation itself; this adapter does not "
                          "reimplement the sum." % os.path.basename(skill_root))

    folded_path = os.path.join(out, "folded.json")
    proc = subprocess.run(
        ["python3", fold, "--rows", rows_path, "--roof-key", roof_key,
         "--out", folded_path, "--label", "roof"],
        capture_output=True, text=True)
    with open(os.path.join(out, "fold.log"), "w") as f:
        f.write(proc.stdout + proc.stderr)
    if proc.returncode != 0 or not os.path.exists(folded_path):
        return emit(status="fail", artifacts={},
                    notes="fold.py exited %d, see fold.log." % proc.returncode)

    folded = read_json(folded_path)
    rows = (read_json(rows_path).get("rows") or [])
    weights = regime_weights(rows)
    bounds = bound_by_time(rows)

    sel = read_json(os.path.join(inp or "", "selection_targets.json"))
    plan = read_json(os.path.join(inp or "", "task_plan.json"), {})
    plan_idx = task_dir_index(plan)
    members = {name: identities for name, identities in task_records(plan) if name}
    idx = target_index(sel)
    if not idx:
        return emit(status="fail", artifacts={},
                    notes="upstream %s/selection_targets.json has no targets, so there is "
                          "no GPU share to read and no expected e2e gain to rank by." % inp)

    tasks_root = os.path.join(inp or "", "tasks")
    task_order = list(members)
    if os.path.isdir(tasks_root):
        task_order += [name for name in sorted(os.listdir(tasks_root))
                       if name not in task_order and os.path.isdir(os.path.join(tasks_root, name, "kernel_src"))]

    def owner(kernel):
        target = idx.get(kernel) or {}
        candidates = [plan_idx.get(kernel), plan_idx.get(target.get("id"))]
        candidates += [kernel, target.get("id"), target.get("name")]
        return next((name for name in candidates if name in task_order), None)

    rows_by_task = {name: [] for name in task_order}
    for row in rows:
        name = owner(row.get("kernel"))
        if name:
            rows_by_task[name].append(row)
    analyses = {name: task_analysis(os.path.join(tasks_root, name), group, roof_key)
                for name, group in rows_by_task.items()}

    def target(kernel):
        if kernel in idx:
            return idx[kernel]
        # Task-keyed rows model the declared complete boundary. Read its owned
        # selected identities once; never multiply a whole-task fold across leaves.
        targets = {(idx[k].get("id") or idx[k].get("name") or k): idx[k]
                   for k in members.get(kernel, []) if k in idx}
        shares = [t.get("overall_pct") for t in targets.values()]
        if targets and all(positive(value) for value in shares):
            return {"id": kernel, "overall_pct": sum(shares)}
        return None

    def new_record(kernel):
        t, name = target(kernel), owner(kernel)
        return {"kernel": kernel, "id": (t or {}).get("id", kernel),
                "pct_gpu_time": (t or {}).get("overall_pct"), "in_selection": t is not None,
                "task": name,
                "task_dir": os.path.join("tasks", name) if name and os.path.isdir(os.path.join(tasks_root, name)) else None,
                "analysis": analyses.get(name, {"processed": False, "ranking_admissible": False,
                                                 "status": "incomplete_inputs", "problems": ["no task owner"]}),
                "regimes": [], "expected_e2e_gain_pct": 0.0, "_any_scored": False}

    per_kernel = {}
    for key, agg in folded.items():
        kernel, _, regime = key.rpartition("/")   # a kernel name may contain /, a regime never does
        rec = per_kernel.setdefault(kernel, new_record(kernel))

        suspect = bool(agg.get("suspect"))
        sp = agg.get("speedup")
        frac = weights[kernel].get(regime, 0.0) if kernel in weights else None
        pct = rec["pct_gpu_time"]
        gain = None
        admissible = rec["analysis"]["ranking_admissible"] and not suspect
        if admissible and positive(sp) and positive(pct) and frac is not None:
            # Expected e2e gain: Amdahl. Spread the overall share across regimes by time,
            # then score each regime.
            gain = pct * frac * (1.0 - 1.0 / sp)
            rec["expected_e2e_gain_pct"] += gain
            rec["_any_scored"] = True

        rec["regimes"].append({
            "regime": regime,
            "roofline_pct": agg.get("pct") if admissible else None,
            "attainable_speedup": sp if admissible else None,
            "bound_type": bounds.get((kernel, regime)) if admissible else None,
            "suspect": suspect,
            "coverage": agg.get("coverage"),
            "cases": agg.get("n"),
            "time_share_within_kernel": round(frac, 6) if frac is not None else None,
            "expected_e2e_gain_pct": None if gain is None else round(gain, 6),
            "fold_scope": "complete_positive_mass" if admissible else "supported_subset_diagnostic",
        })

    represented = {r["task"] for r in per_kernel.values()}
    for name in task_order:
        if name not in represented:
            per_kernel[name] = new_record(name)

    kernels = []
    for rec in per_kernel.values():
        if not rec.pop("_any_scored"):
            rec["expected_e2e_gain_pct"] = None
        else:
            rec["expected_e2e_gain_pct"] = round(rec["expected_e2e_gain_pct"], 6)
        # A single value for the top-level view: the regime holding the most time. The
        # breakdown always stays in regimes.
        top = max(rec["regimes"], key=lambda r: r["time_share_within_kernel"] or 0.0, default={})
        rec["roofline_pct"] = top.get("roofline_pct")
        rec["attainable_speedup"] = top.get("attainable_speedup")
        rec["bound_type"] = top.get("bound_type")
        rec["suspect"] = (any(r["suspect"] for r in rec["regimes"])
                          or bool(rec["analysis"].get("suspect_positive_rows")))
        kernels.append(rec)

    kernels.sort(key=lambda r: (r["expected_e2e_gain_pct"] is None,
                                -(r["expected_e2e_gain_pct"] or 0.0),
                                task_order.index(r["task"]) if r["task"] in task_order else len(task_order),
                                r["kernel"]))
    ranked = [k for k in kernels if k["expected_e2e_gain_pct"] is not None]

    doc = {
        "rank_key": "expected_e2e_gain_pct",
        "fallback_execution_order": "profile task_plan order, then remaining task directories; not predicted gain",
        "formula": "pct_gpu_time * time_share_within_kernel * (1 - 1/attainable_speedup), summed over regimes",
        "roofline_mode": params.get("roofline_mode", "unknown"),
        "roof_key": roof_key,
        "fold": "performed by the implementation's own scripts/fold.py (time domain; numerator and denominator summed separately, divided once)",
        "caveats": [
            "bound_type is a model's classification, not proof of the measured bottleneck.",
            "suspect=true means efficiency > 1, which falsifies the denominator; it is not clamped to 100% and no speedup may be inferred from it.",
            "TL percentages and rocprof percentages are not the same quantity and must not be divided into one another.",
            "The expected gain here is an upper bound. An operator win with no e2e movement has already happened twice in this scenario; only applyback may claim an e2e gain.",
        ],
        "kernel_count": len(kernels),
        "task_count": len(task_order),
        "ranked_count": len(ranked),
        "sum_expected_e2e_gain_pct": (round(sum(k["expected_e2e_gain_pct"] for k in ranked), 4)
                                      if len(ranked) == len(kernels) else None),
        "kernels": kernels,
    }
    with open(os.path.join(out, "roofline.json"), "w") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")

    notes = []
    status = "ok"
    unscored = [k["kernel"] for k in kernels if k["expected_e2e_gain_pct"] is None]
    if unscored:
        notes.append("no expected gain could be computed for these kernels (suspect, or "
                     "attainable missing); they are listed but not ranked: %s"
                     % ", ".join(unscored))
    missing_task = [k["kernel"] for k in kernels if not k["task_dir"]]
    if missing_task:
        status = "blocked"
        notes.append("these analysis entries have no matching UT directory under the "
                     "upstream tasks/, so the optimize stage cannot claim them: %s. Fill in "
                     "the profile stage's task directories first." % ", ".join(missing_task))
    incomplete = [name for name, analysis in analyses.items() if not analysis["processed"]]
    if incomplete or not kernels:
        status = "blocked"
        notes.append("missing or unprocessed task inputs: " + ", ".join(incomplete))
    elif not ranked:
        notes.append("all task analyses completed with unsupported whole-task bounds; retained execution order follows task_plan, not predicted gain")

    return emit(status=status,
                artifacts={"roofline": "roofline.json", "folded": "folded.json",
                           "fold_log": "fold.log", "rows": args.rows},
                metrics={"kernel_count": len(kernels), "ranked_count": len(ranked),
                         "sum_expected_e2e_gain_pct": doc["sum_expected_e2e_gain_pct"]},
                notes="; ".join(notes))


if __name__ == "__main__":
    sys.exit(main())
