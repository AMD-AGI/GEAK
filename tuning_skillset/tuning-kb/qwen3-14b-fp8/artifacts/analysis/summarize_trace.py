#!/usr/bin/env python3
"""Aggregate a SGLang/torch GPU trace into per-kernel totals for one decode step.

Reports both the summed kernel busy time and the wall span, so a gap between them
(launch bubbles, host stalls) is visible rather than hidden inside a percentage.
"""
import gzip
import json
import re
import sys
from collections import defaultdict


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)


def main():
    path = sys.argv[1]
    nsteps = int(sys.argv[2]) if len(sys.argv) > 2 else None
    tr = load(path)
    evs = [e for e in tr["traceEvents"] if e.get("ph") == "X"]
    kern = [e for e in evs if e.get("cat") in ("kernel", "gpu_op", "Kernel")]
    if not kern:
        cats = sorted({e.get("cat") for e in evs})
        print("no kernel events; categories present:", cats)
        return

    lo = min(e["ts"] for e in kern)
    hi = max(e["ts"] + e.get("dur", 0) for e in kern)
    span_us = hi - lo
    busy = defaultdict(float)
    count = defaultdict(int)
    for e in kern:
        busy[e["name"]] += e.get("dur", 0)
        count[e["name"]] += 1

    total = sum(busy.values())

    # Infer the step count from the attention kernel if not given: it runs once per
    # layer per step, and the model has 40 layers.
    if nsteps is None:
        pa = [n for n in count if "QKV_mfma16" in n]
        nsteps = max(1, round(count[pa[0]] / 40)) if pa else 1

    print(f"trace: {path}")
    print(f"kernel events: {len(kern)}   distinct: {len(busy)}   inferred decode steps: {nsteps}")
    print(f"GPU busy total: {total/1e3:.2f} ms   wall span: {span_us/1e3:.2f} ms   "
          f"idle in span: {(span_us-total)/1e3:.2f} ms ({100*(span_us-total)/span_us:.1f}%)")
    print(f"per step: busy {total/nsteps/1e3:.3f} ms   wall {span_us/nsteps/1e3:.3f} ms")
    print()
    print(f"{'us/step':>10} {'%':>6} {'calls/step':>11}  kernel")
    print("-" * 100)
    for name, us in sorted(busy.items(), key=lambda kv: -kv[1]):
        short = re.sub(r"<.*>", "<...>", name)[:78]
        print(f"{us/nsteps:10.1f} {100*us/total:6.2f} {count[name]/nsteps:11.1f}  {short}")

    # Roll up into the categories that matter for this experiment.
    groups = {
        "attention (paged decode)": lambda n: "QKV_mfma16" in n or "ll4mi_reduce" in n,
        "GEMM (fp8 blockscale)": lambda n: "gemm" in n.lower() or "Gemm" in n or "CK" in n,
        "kv cache write": lambda n: "cache" in n.lower() and "gemm" not in n.lower(),
        "norm / quant": lambda n: "norm" in n.lower() or "quant" in n.lower(),
        "rope": lambda n: "rope" in n.lower() or "Rope" in n,
        "activation (silu/mul)": lambda n: "silu" in n.lower() or "mul_and" in n.lower(),
    }
    print()
    print("rollup:")
    claimed = set()
    for label, pred in groups.items():
        tot = sum(us for n, us in busy.items() if pred(n) and n not in claimed)
        for n in busy:
            if pred(n):
                claimed.add(n)
        if tot:
            print(f"  {label:28s} {tot/nsteps/1e3:7.3f} ms/step  {100*tot/total:5.1f}%")
    other = sum(us for n, us in busy.items() if n not in claimed)
    print(f"  {'other':28s} {other/nsteps/1e3:7.3f} ms/step  {100*other/total:5.1f}%")


if __name__ == "__main__":
    main()
