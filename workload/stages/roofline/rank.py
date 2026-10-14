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
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))          # stages/roofline
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "scripts"))
from emit_result import emit, read_json  # noqa: E402


def regime_weights(rows):
    """Spread each kernel's time across regimes by measured_us. Time domain, same basis as the fold."""
    per = {}
    for r in rows:
        k, g = r.get("kernel"), r.get("regime")
        us = r.get("measured_us")
        if k is None or g is None or not isinstance(us, (int, float)):
            continue
        w = r.get("case_weight")
        w = w if isinstance(w, (int, float)) and w > 0 else 1.0
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
        if not b or not isinstance(us, (int, float)):
            continue
        key = (r.get("kernel"), r.get("regime"))
        acc.setdefault(key, {}).setdefault(b, 0.0)
        acc[key][b] += us
    return {k: max(v.items(), key=lambda kv: kv[1])[0] for k, v in acc.items()}


def task_dir_index(plan):
    """task_plan.json -> {selected identity: task directory name}.

    Empty when the upstream skill declared no plan, which leaves the caller on its
    name-matching fallback."""
    idx = {}
    for t in (plan or {}).get("tasks") or []:
        name = t.get("name") or t.get("id")
        if not name:
            continue
        for k in t.get("selected_identities") or []:
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
    plan_idx = task_dir_index(read_json(os.path.join(inp or "", "task_plan.json"), {}))
    idx = target_index(sel)
    if not idx:
        return emit(status="fail", artifacts={},
                    notes="upstream %s/selection_targets.json has no targets, so there is "
                          "no GPU share to read and no expected e2e gain to rank by." % inp)

    tasks_root = os.path.join(inp or "", "tasks")
    per_kernel = {}
    for key, agg in folded.items():
        kernel, _, regime = key.rpartition("/")   # a kernel name may contain /, a regime never does
        t = idx.get(kernel)
        rec = per_kernel.setdefault(kernel, {
            "kernel": kernel,
            "id": (t or {}).get("id", kernel),
            "pct_gpu_time": (t or {}).get("overall_pct"),
            "in_selection": t is not None,
            "task_dir": None,
            "regimes": [],
            "expected_e2e_gain_pct": 0.0,
            "_any_scored": False,
        })
        if rec["task_dir"] is None:
            # The declared owner first: a task directory is named after the math boundary it
            # covers, which is only incidentally a kernel name.
            cands = [plan_idx.get(kernel)]
            if t:
                cands += [t.get("id"), t.get("name")]
            cands.append(kernel)
            for cand in cands:
                if cand and os.path.isdir(os.path.join(tasks_root, cand)):
                    rec["task_dir"] = os.path.join("tasks", cand)
                    break

        suspect = bool(agg.get("suspect"))
        sp = agg.get("speedup")
        frac = weights.get(kernel, {}).get(regime)
        if frac is None:
            frac = 1.0 / max(1, len([k for k in folded if k.startswith(kernel + "/")]))
        pct = rec["pct_gpu_time"]
        gain = None
        if (not suspect and isinstance(sp, (int, float)) and sp > 0
                and isinstance(pct, (int, float))):
            # Expected e2e gain: Amdahl. Spread the overall share across regimes by time,
            # then score each regime.
            gain = pct * frac * (1.0 - 1.0 / sp)
            rec["expected_e2e_gain_pct"] += gain
            rec["_any_scored"] = True

        rec["regimes"].append({
            "regime": regime,
            "roofline_pct": agg.get("pct"),
            "attainable_speedup": None if suspect else sp,
            "bound_type": bounds.get((kernel, regime)),
            "suspect": suspect,
            "coverage": agg.get("coverage"),
            "cases": agg.get("n"),
            "time_share_within_kernel": round(frac, 6),
            "expected_e2e_gain_pct": None if gain is None else round(gain, 6),
        })

    kernels = []
    for rec in per_kernel.values():
        if not rec.pop("_any_scored"):
            rec["expected_e2e_gain_pct"] = None
        else:
            rec["expected_e2e_gain_pct"] = round(rec["expected_e2e_gain_pct"], 6)
        # A single value for the top-level view: the regime holding the most time. The
        # breakdown always stays in regimes.
        top = max(rec["regimes"], key=lambda r: r["time_share_within_kernel"])
        rec["roofline_pct"] = top["roofline_pct"]
        rec["attainable_speedup"] = top["attainable_speedup"]
        rec["bound_type"] = top["bound_type"]
        rec["suspect"] = any(r["suspect"] for r in rec["regimes"])
        kernels.append(rec)

    kernels.sort(key=lambda r: (r["expected_e2e_gain_pct"] is None,
                                -(r["expected_e2e_gain_pct"] or 0.0), r["kernel"]))
    ranked = [k for k in kernels if k["expected_e2e_gain_pct"] is not None]

    doc = {
        "rank_key": "expected_e2e_gain_pct",
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
        "ranked_count": len(ranked),
        "sum_expected_e2e_gain_pct": round(sum(k["expected_e2e_gain_pct"] for k in ranked), 4),
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
    missing_task = [k["kernel"] for k in ranked if not k["task_dir"]]
    if missing_task:
        status = "blocked"
        notes.append("these kernels are ranked but have no matching UT directory under the "
                     "upstream tasks/, so the optimize stage cannot claim them: %s. Fill in "
                     "the profile stage's task directories first." % ", ".join(missing_task))
    if not ranked:
        status = "blocked"
        notes.append("no kernel produced an expected gain, so no optimization order can be derived.")

    return emit(status=status,
                artifacts={"roofline": "roofline.json", "folded": "folded.json",
                           "fold_log": "fold.log", "rows": args.rows},
                metrics={"kernel_count": len(kernels), "ranked_count": len(ranked),
                         "sum_expected_e2e_gain_pct": doc["sum_expected_e2e_gain_pct"]},
                notes="; ".join(notes))


if __name__ == "__main__":
    sys.exit(main())
