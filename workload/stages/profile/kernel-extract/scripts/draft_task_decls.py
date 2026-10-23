#!/usr/bin/env python3
"""Draft task_decls.json -- the one file in the profile stage an agent edits.

Everything around a task boundary is evidence the earlier scripts already collected;
what is left is judgement. This drafts that judgement from the evidence, says on every
decision what it was based on, and marks each one `needs_review`. The agent reviews,
edits, empties `needs_review`, and `build_task_plan.py` does the rest.

    python3 draft_task_decls.py --eval-dir "$EVAL_DIR" --out "$EVAL_DIR/task_decls.json"

reads   selection_targets.json  kernel_ranking.json  sampling_points_final.json
        evidence/shapes/*.json  evidence/source_map.json  evidence/adjacency.json

How the draft decides (each is a proposal, not a verdict):

  grouping     selected kernels joined by consecutive launches on one stream
               (adjacency.py) that ALSO share a serving-stack frame (source_map.py)
               narrower than the one every selected kernel passes through (a layer's
               forward is not an operator); anything else is a task of its own
  excluded     collective and memory rows: a one-GPU UT cannot exercise a collective
               (references/collective-ut.md) and a device copy has no editable source
  target       the innermost frame every kernel of the task passes through
  ut_family    the family whose `match` patterns hit the most kernels of the task (and
               its launching ops); `custom` when none does
  phases       the phases at whose counted points the task's kernels launched
  joint_phases one row per distinct set of the task's kernels launched together at a
               phase's points; two rows in one phase are dispatch branches of one op,
               and a branch whose kernel was not selected is listed as a companion
  case_axes    per phase, each var axis as an expression in `shape` (the point's
               new_tokens / batch size) and `context` (its per-request context). The
               draft sets only the family's first var axis = shape, which is usually
               wrong for speculative decode (M = batch x draft tokens) -- review it.
Stdlib only.
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from adjacency import chains  # noqa: E402
from source_map import common_seam  # noqa: E402

FAMILIES_DIR = os.path.join(os.path.dirname(HERE), "ut_contracts", "families")
EXCLUDE_KINDS = {
    "collective": "collective: a one-GPU UT cannot exercise it; it needs the multi-rank "
                  "harness of references/collective-ut.md",
    "memory": "device copy/memset: no kernel source to extract or edit",
}


def load(path, default=None):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return default


def families():
    out = {}
    for f in sorted(glob.glob(os.path.join(FAMILIES_DIR, "*.json"))):
        d = json.load(open(f))
        m = d.get("match") or {}
        out[d["family"]] = {"kernel": [re.compile(p) for p in m.get("kernel", [])],
                            "op": [re.compile(p) for p in m.get("op", [])],
                            "var_axis": next((n for n, a in d["axes"].items() if a.get("var")),
                                             None)}
    return out


def guess_family(kernels, ops, fams):
    scores = []
    for name, f in fams.items():
        k = sum(1 for n in kernels if any(p.search(n) for p in f["kernel"]))
        o = sum(1 for n in ops if any(p.search(n) for p in f["op"]))
        if k or o:
            scores.append((k, o, name))
    scores.sort(reverse=True)
    if not scores:
        return "custom", "no family's match patterns hit these kernels or their launching ops"
    k, o, name = scores[0]
    alt = ", ".join("%s(%d/%d)" % (n, kk, oo) for kk, oo, n in scores[1:3])
    return name, "%d/%d kernels and %d launching op(s) match %s%s" % (
        k, len(kernels), o, name, "; runners-up: " + alt if alt else "")


def label_for(kernels, seam, fam):
    if seam and seam.get("target_callable"):
        base = seam["target_callable"].split(":")[-1].split(".")[-1]
    else:
        base = re.sub(r"\W+", "_", kernels[0].split("(")[0].split("<")[0])
    base = re.sub(r"_+", "_", base.strip("_")).lower()[:40] or "task"
    return "%s__%s" % (fam if fam != "custom" else "op", base)


def draft(eval_dir):
    ev = lambda *p: os.path.join(eval_dir, *p)  # noqa: E731
    sel = load(ev("selection_targets.json"))
    rank = load(ev("kernel_ranking.json"), {})
    pts = load(ev("sampling_points_final.json"), {}).get("points", [])
    sm = load(ev("evidence", "source_map.json"), {"kernels": {}})
    adj = load(ev("evidence", "adjacency.json"), {"pairs": []})
    if not sel:
        raise SystemExit("draft_task_decls: no selection_targets.json in %s" % eval_dir)
    kind = {r["name"]: r.get("kind") for r in rank.get("ranking", [])}
    targets = [t["id"] for t in sel.get("targets", [])]
    phase_of = {p["point_id"]: p["phase"] for p in pts if p.get("counted")}
    launched = defaultdict(set)          # kernel -> point ids where it launched
    for f in glob.glob(ev("evidence", "shapes", "*.json")):
        d = json.load(open(f))
        for k in d.get("kernels", {}):
            launched[k].add(d.get("point_id") or os.path.basename(f)[:-5])
    fams = families()

    excluded, rest = [], []
    for t in targets:
        if kind.get(t) in EXCLUDE_KINDS:
            excluded.append({"kernels": [t], "kind": kind[t], "reason": EXCLUDE_KINDS[kind[t]],
                             "needs_review": ["confirm the exclusion; a selected identity "
                                              "excluded here is reported as blocked"]})
        else:
            rest.append(t)

    # A frame every selected kernel passes through (the decoder layer, the model forward)
    # is not an operator boundary: two kernels that merely launch back to back across
    # two operators of one layer share it. Only a narrower common frame links them.
    framed = [k for k in rest if (sm["kernels"].get(k) or {}).get("frames")]
    broad = common_seam(sm, framed) if len(framed) > 2 else None

    def linked(a, b):
        seam = common_seam(sm, [a, b])
        return seam is not None and (broad is None
                                     or seam["target_callable"] != broad["target_callable"])
    groups = [g for g in chains(adj, linked=linked) if set(g) <= set(rest)]
    grouped = {k for g in groups for k in g}
    groups += [[k] for k in rest if k not in grouped]
    # Kernels that never launch together but enter through the same callable are
    # branches of one operation (prefill vs decode, or a batch-size dispatch): one task.
    by_seam, merged = {}, []
    for g in groups:
        seam = common_seam(sm, g) if len(g) > 1 else (sm["kernels"].get(g[0]) or {}).get("launcher")
        key = (seam or {}).get("target_callable")
        if key and key in by_seam:
            by_seam[key].extend(g)
        else:
            merged.append(g)
            if key:
                by_seam[key] = g
    groups = merged

    tasks, labels = [], set()
    for g in groups:
        seam = common_seam(sm, g) if len(g) > 1 else (sm["kernels"].get(g[0]) or {}).get("launcher")
        if seam is None and len(g) > 1:
            seam = (sm["kernels"].get(g[0]) or {}).get("launcher")
        ops = sorted({o for k in g for o in (sm["kernels"].get(k) or {}).get("launching_ops", [])})
        fam, fam_why = guess_family(g, ops + ([seam["target_callable"]] if seam else []), fams)
        pts_g = sorted({p for k in g for p in launched.get(k, ())}, key=lambda p: (phase_of.get(p, ""), p))
        phases = sorted({phase_of[p] for p in pts_g if p in phase_of})
        review = []
        joint = []
        if len(g) > 1:
            for ph in phases:
                by_set = defaultdict(list)
                for p in pts_g:
                    if phase_of.get(p) == ph:
                        by_set[tuple(sorted(k for k in g if p in launched.get(k, ())))].append(p)
                for i, (ks, ps) in enumerate(sorted(by_set.items(), key=lambda kv: kv[1])):
                    joint.append({"regime": "%s_%d" % (ph.lower(), i), "phase": ph,
                                  "required_kernels": list(ks), "points": ps})
            if all(len(j["required_kernels"]) == 1 for j in joint):
                review.append("every row holds one kernel: these are mutually exclusive "
                              "branches of one op (dispatch_ut), not a chain")
            elif len({tuple(j["required_kernels"]) for j in joint}) > len(phases):
                review.append("some phase has more than one joint row: these are dispatch "
                              "branches of one op; a branch kernel that was not selected "
                              "belongs in companions")
        var_axis = fams.get(fam, {}).get("var_axis")
        review += [
            "grouping: %s" % ("consecutive launches + common frame %s" % seam["frame"]
                              if len(g) > 1 and seam else "no consecutive-launch partner with "
                              "a common frame; a task of its own"),
            "ut_family: %s" % fam_why,
            "target_callable: %s" % ((seam or {}).get("frame") or "no serving-stack frame "
                                     "(trace taken without with_stack?)"),
            "case_axes: %s = shape is a placeholder; derive each var axis from the host op "
            "dims in evidence/shapes as an expression in shape/context (e.g. batch x draft "
            "tokens under speculation); build_task_plan.py --check lists what is missing"
            % var_axis,
            "axes/attrs: pin the family's const axes and attrs from evidence/shapes host_op",
            "companions: dispatch-branch siblings and mandatory chain members below the "
            "threshold (look for kernels launched under the same op at other points)",
        ]
        label, n = label_for(g, seam, fam), 2
        while label in labels:
            label, n = "%s_%d" % (label_for(g, seam, fam), n), n + 1
        labels.add(label)
        tasks.append({
            "label": label,
            "kernels": g, "companions": [], "phases": phases,
            "ut_family": fam, "ut_family_why": fam_why if fam == "custom" else None,
            "target_callable": (seam or {}).get("target_callable"),
            "target_file": (seam or {}).get("path"), "target_sha256": (seam or {}).get("sha256"),
            "launching_ops": ops,
            "joint_phases": joint,
            "axes": {}, "attrs": {},
            "case_axes": {ph: {var_axis: "shape"} for ph in phases} if var_axis else {},
            "phase_regimes": {ph: {"cuda_graph": ph == "DECODE"} for ph in phases},
            "envelope_cases": [],
            "needs_review": review,
        })
    return {"contract": "kernel_extract/task_decls/v1",
            "drafted_from": {k: ev(*v) for k, v in {
                "selection": ("selection_targets.json",), "source_map": ("evidence", "source_map.json"),
                "adjacency": ("evidence", "adjacency.json")}.items()},
            "instructions": "Review every task and exclusion, edit what is wrong, then empty "
                            "its needs_review list. build_task_plan.py refuses a declaration "
                            "with anything left in needs_review.",
            "tasks": tasks, "excluded": excluded, "blocked": []}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true", help="overwrite an existing declaration")
    a = ap.parse_args()
    if os.path.exists(a.out) and not a.force:
        raise SystemExit("draft_task_decls: %s exists (it may hold reviewed decisions); "
                         "pass --force to overwrite" % a.out)
    out = draft(a.eval_dir)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    for t in out["tasks"]:
        print("  %-44s %-18s %d kernel(s) %s  %s" % (t["label"], t["ut_family"], len(t["kernels"]),
                                                    "/".join(t["phases"]), t["target_callable"]))
    for x in out["excluded"]:
        print("  excluded %-36s %s" % (x["kernels"][0][:36], x["kind"]))
    print("-> %s  (review it, then build_task_plan.py)" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
