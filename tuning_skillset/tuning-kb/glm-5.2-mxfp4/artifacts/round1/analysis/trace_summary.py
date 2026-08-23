#!/usr/bin/env python3
"""Summarise a torch-profiler trace from an SGLang rank into GPU-kernel time shares.

Written for this experiment because TraceLens could not produce a usable profile of the
concurrency-64 workload (see reference/tracelens/). Reads one rank's .trace.json.gz,
keeps only the GPU-side kernel events, and ranks them by total device time.

    python3 trace_summary.py /tmp/prof_conc64/conc64-TP-0.trace.json.gz [--top N]
"""
import argparse
import gzip
import json
import re
from collections import defaultdict


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)


def norm(name):
    """Collapse template/launch-config noise so the same kernel aggregates."""
    n = re.sub(r"<[^<>]*>", "<>", name)
    n = re.sub(r"\b\d{3,}\b", "N", n)
    return n[:110]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--json-out")
    args = ap.parse_args()

    data = load(args.trace)
    events = data["traceEvents"]

    # torch profiler tags device-side work with cat "kernel" (compute) and
    # "gpu_memcpy"/"gpu_memset". Everything else is host-side and must not be
    # summed into a GPU budget.
    gpu_cats = {"kernel", "gpu_memcpy", "gpu_memset"}
    tot = defaultdict(float)
    cnt = defaultdict(int)
    cat_of = {}
    total = 0.0
    for e in events:
        if e.get("cat") in gpu_cats and e.get("ph") == "X":
            k = norm(e["name"])
            d = e.get("dur", 0.0)
            tot[k] += d
            cnt[k] += 1
            cat_of[k] = e["cat"]
            total += d

    rows = sorted(tot.items(), key=lambda kv: -kv[1])
    print(f"trace: {args.trace}")
    print(f"total GPU event time: {total/1000:.1f} ms across {sum(cnt.values())} events")
    print(f"{'%GPU':>7} {'ms':>10} {'calls':>7} {'us/call':>9}  kernel")
    print("-" * 130)
    out = []
    for k, v in rows[: args.top]:
        pct = 100.0 * v / total if total else 0.0
        print(f"{pct:7.2f} {v/1000:10.2f} {cnt[k]:7d} {v/cnt[k]:9.2f}  {k}")
        out.append(
            {"kernel": k, "pct": pct, "ms": v / 1000, "calls": cnt[k],
             "us_per_call": v / cnt[k], "cat": cat_of[k]}
        )

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"total_gpu_ms": total / 1000, "kernels": out}, f, indent=2)


if __name__ == "__main__":
    main()
