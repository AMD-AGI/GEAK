#!/usr/bin/env python3
"""Which CI points a multi-config applyback measures, and in what order.

One list, read by the session (to decide what to launch) and by finalize.py (to decide
what must have been measured), so the two can never disagree.

**A config the UT covered is a config applyback measures.** Every config named by a UT
case (`tasks/<k>/meta.json` -> `workload.cases[].config_id`) priced the kernels that were
kept; leaving it out of the end-to-end check means a regression there is never seen.
That is the default (`params.points: "covered"`). The two older spellings stay:

  "covered"         every baseline point that some UT case names (default)
  "all"             every baseline point
  "representative"  each group's two ends only -- a screening run, not a verdict:
                    covered points it leaves out are reported as unmeasured
  [pid, ...]        exactly these

The order is the baseline's `measure_order` when it has one (each group's low, middle and
high concurrency first, then the rest ascending) and is otherwise recomputed by the same
rule, so a baseline written before `measure_order` existed is not silently narrowed.

    points.py [--run-root DIR] [--tasks-dir DIR] [--points SPEC]
        prints one point id per line, in measuring order; --json for the full record
"""

import argparse
import glob
import json
import os
import sys


def read_json(path, default=None):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def ut_configs(tasks_dir):
    """{config_id: [task, ...]} for every config a UT case names."""
    seen = {}
    for meta in sorted(glob.glob(os.path.join(tasks_dir or "", "*", "meta.json"))):
        doc = read_json(meta, {}) or {}
        cases = ((doc.get("workload") or {}).get("cases")) or []
        task = os.path.basename(os.path.dirname(meta))
        for case in cases:
            cid = case.get("config_id") if isinstance(case, dict) else None
            if cid:
                seen.setdefault(cid, [])
                if task not in seen[cid]:
                    seen[cid].append(task)
    return seen


def measure_order(top):
    """The baseline's measure_order, or the same rule applied to its points and groups."""
    if top.get("measure_order"):
        return list(top["measure_order"])
    conc = {p["point_id"]: (p.get("config") or {}).get("concurrency", 0)
            for p in top.get("points") or []}
    groups = top.get("groups") or {"all": list(conc)}
    first, rest = [], []
    for _, ids in sorted(groups.items()):
        ids = sorted(ids, key=lambda i: conc.get(i, 0))
        pick = list(dict.fromkeys([ids[0], ids[len(ids) // 2], ids[-1]])) if len(ids) > 2 else ids
        first += pick
        rest += [i for i in ids if i not in pick]
    return first + rest


def select(top, tasks_dir, spec="covered"):
    """The points to measure, in order, plus what was left out and why."""
    order = measure_order(top)
    known = set(order)
    covered = ut_configs(tasks_dir)
    if isinstance(spec, list):
        chosen = [p for p in order if p in spec] + [p for p in spec if p not in known]
    elif spec == "all":
        chosen = order
    elif spec == "representative":
        rep = list(top.get("representative") or [])
        chosen = [p for p in order if p in rep]
    else:  # "covered" (default); with no UT cases to read, every point
        chosen = [p for p in order if p in covered] if covered else order
    required = [p for p in order if p in covered] if covered else list(chosen)
    return {
        "spec": spec,
        "points": chosen,
        "required": required,
        "unmeasured_covered": [p for p in required if p not in chosen],
        "ut_configs": covered,
        "ut_configs_without_baseline": sorted(set(covered) - known),
        "unknown": [p for p in chosen if p not in known],
    }


def tasks_dir_of(run_root):
    return os.environ.get("TASKS_DIR") or os.path.join(run_root, "profile", "out", "tasks")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-root", default=os.environ.get("RUN_ROOT"))
    ap.add_argument("--tasks-dir")
    ap.add_argument("--points", help='"covered", "all", "representative" or a JSON list')
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if not a.run_root:
        ap.error("--run-root (or RUN_ROOT) is required")
    spec = a.points
    if spec is None:
        params = read_json(os.environ.get("PARAMS") or "", {}) or {}
        spec = params.get("points") or "covered"
    elif spec.startswith("["):
        spec = json.loads(spec)
    top = read_json(os.path.join(a.run_root, "baseline", "out", "baseline.json"), {}) or {}
    if not top.get("multi_config"):
        print("baseline.json is not multi_config; there are no points to select",
              file=sys.stderr)
        return 2
    sel = select(top, a.tasks_dir or tasks_dir_of(a.run_root), spec)
    if a.json:
        print(json.dumps(sel, indent=1))
    else:
        print("\n".join(sel["points"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
