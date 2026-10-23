#!/usr/bin/env python3
"""Launch-order evidence between the selected kernels, for grouping them into tasks.

Two kernels belong in one task when they are stages of one operation, and the timeline
is what shows it: on one stream, how often is a launch of A followed by a launch of B,
and how many other kernels sit in between? `intervening_kernels` 0 on the same stream
is "consecutive launches" -- the evidence a fusion group or a joint UT needs. A shared
callable name is not: a parent op and its own child kernel share one and overlap
completely.

    python3 adjacency.py --trace-dir <point dir> [--trace-dir ...] \
        --targets selection_targets.json [--kernel NAME ...] --out evidence/adjacency.json

For every ordered pair (A, B) of targets this counts, over all given windows, the
launches of A whose NEXT target launch on the same stream is B, with the median number
of non-target kernels between them, as a share of A's launches and of B's (a stage that
only one dispatch branch runs is preceded by its producer every time, though the
producer is followed by it only on that branch). Only TP-0 of each directory is read.
Stdlib only.
"""
import argparse
import json
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace_scan import scan, DEVICE_CATS  # noqa: E402
from kernel_shapes import pick_rank_file, targets_from  # noqa: E402

CONSECUTIVE_SHARE = 0.9


def transitions(path, wanted):
    """{(A, B): [intervening counts]} and launches per target, for one trace."""
    s = scan(path, DEVICE_CATS)
    by_stream = defaultdict(list)
    for cat, name, ts, _dur, _pid, tid, _a in s.events:
        by_stream[tid].append(name)
    pairs, launches = defaultdict(list), defaultdict(int)
    for seq in by_stream.values():
        last, gap = None, 0
        for name in seq:
            if name in wanted:
                launches[name] += 1
                if last is not None:
                    pairs[(last, name)].append(gap)
                last, gap = name, 0
            elif last is not None:
                gap += 1
    return pairs, launches


def adjacency(trace_dirs, wanted):
    pairs, launches, sources = defaultdict(list), defaultdict(int), []
    for d in trace_dirs:
        path = pick_rank_file(d)
        sources.append(path)
        p, l = transitions(path, wanted)
        for k, v in p.items():
            pairs[k].extend(v)
        for k, v in l.items():
            launches[k] += v
    rows = []
    for (a, b), gaps in pairs.items():
        if a == b:
            continue
        share = len(gaps) / launches[a] if launches[a] else 0.0
        # Either direction is a chain: a dispatch-branch stage (a gemm2 variant that runs
        # only at small batch) is preceded by its producer every time, while the producer
        # is followed by it only on that branch.
        share_to = len(gaps) / launches[b] if launches[b] else 0.0
        zero = sum(1 for g in gaps if g == 0) / len(gaps)
        best = max(share, share_to)
        rows.append({"from": a, "to": b, "followed": len(gaps), "launches_from": launches[a],
                     "launches_to": launches[b],
                     "share_of_from_launches": round(share, 4),
                     "share_of_to_launches": round(share_to, 4),
                     "intervening_kernels_median": statistics.median(gaps),
                     "zero_intervening_share": round(zero, 4), "same_stream": True,
                     "kind": ("observed_consecutive_launches"
                              if best >= CONSECUTIVE_SHARE and zero >= CONSECUTIVE_SHARE
                              else "observed_order" if best >= CONSECUTIVE_SHARE else "weak")})
    rows.sort(key=lambda r: (-max(r["share_of_from_launches"], r["share_of_to_launches"]),
                             r["from"], r["to"]))
    return {"contract": "kernel_extract/adjacency/v1", "traces": sources,
            "rule": "next target launch on the same stream; consecutive = >=%.0f%% of A's "
                    "launches followed by B, or of B's launches preceded by A, with no kernel "
                    "between" % (100 * CONSECUTIVE_SHARE),
            "launches": dict(launches), "pairs": rows}


def chains(adj, min_kind=("observed_consecutive_launches",), linked=None):
    """Connected groups of targets joined by consecutive-launch pairs (draft_task_decls.py).

    `observed_order` is not enough: a whole decoder layer is one ordered sequence, so
    joining on it puts every target in one group. `linked(a, b)` can veto a pair (the
    draft requires a common serving-stack seam as well)."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for r in adj["pairs"]:
        if r["kind"] in min_kind and (linked is None or linked(r["from"], r["to"])):
            parent[find(r["from"])] = find(r["to"])
    groups = defaultdict(set)
    for k in parent:
        groups[find(k)].add(k)
    return [sorted(g) for g in groups.values() if len(g) > 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trace-dir", action="append", required=True)
    ap.add_argument("--targets", help="selection_targets.json")
    ap.add_argument("--kernel", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    wanted = set(a.kernel) | (targets_from(a.targets) if a.targets else set())
    if len(wanted) < 2:
        raise SystemExit("adjacency: need at least two kernels")
    out = adjacency(a.trace_dir, wanted)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    for r in out["pairs"]:
        if r["kind"] != "weak":
            print("  %-32s %-40s -> %-40s %5.1f%% gap~%s"
                  % (r["kind"], r["from"][:40], r["to"][:40],
                     100 * r["share_of_from_launches"], r["intervening_kernels_median"]))
    print("-> %s (%d chains)" % (a.out, len(chains(out))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
