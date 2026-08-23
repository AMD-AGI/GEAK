#!/usr/bin/env python3
"""Second-pass trace analysis: wall span, step markers, and the prefill/decode split
of the bimodal DSA kernel.

`main_kernel` (tilelang DSA sparse attention) serves both prefill and decode, so its
aggregate share says nothing about which regime to attack. Splitting its duration
histogram separates the two clusters.

    python3 trace_detail.py /tmp/prof_conc64/conc64-TP-0.trace.json.gz
"""
import argparse
import gzip
import json
from collections import defaultdict


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--split-kernel", default="main_kernel")
    args = ap.parse_args()

    ev = load(args.trace)["traceEvents"]

    gpu = [e for e in ev if e.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}
           and e.get("ph") == "X"]
    t0 = min(e["ts"] for e in gpu)
    t1 = max(e["ts"] + e.get("dur", 0) for e in gpu)
    span = (t1 - t0) / 1000.0
    busy = sum(e.get("dur", 0) for e in gpu) / 1000.0
    print(f"GPU wall span      : {span:.1f} ms")
    print(f"summed kernel time : {busy:.1f} ms  ({100*busy/span:.1f}% of span -- "
          f">100% means concurrent streams)")

    steps = [e for e in ev if e.get("name", "").startswith("ProfilerStep")]
    print(f"ProfilerStep events: {len(steps)}")
    if steps:
        sd = sorted(e.get("dur", 0) / 1000.0 for e in steps)
        print(f"  step duration ms : min {sd[0]:.1f} med {sd[len(sd)//2]:.1f} max {sd[-1]:.1f}")

    # Bimodal split of the target kernel.
    ks = sorted(e.get("dur", 0) for e in gpu if e["name"] == args.split_kernel)
    if ks:
        tot = sum(ks)
        print(f"\n{args.split_kernel}: {len(ks)} calls, {tot/1000:.1f} ms total")
        # Largest gap in the sorted durations separates the two regimes.
        gap_i, gap = 0, 0
        for i in range(1, len(ks)):
            if ks[i] - ks[i - 1] > gap:
                gap, gap_i = ks[i] - ks[i - 1], i
        lo, hi = ks[:gap_i], ks[gap_i:]
        for label, grp in (("cheap cluster (decode)", lo), ("costly cluster (prefill)", hi)):
            if not grp:
                continue
            s = sum(grp)
            print(f"  {label:26s}: {len(grp):5d} calls  {s/1000:8.1f} ms "
                  f"({100*s/tot:5.1f}% of kernel)  {s/len(grp):8.1f} us/call  "
                  f"range {grp[0]:.1f}-{grp[-1]:.1f} us")

    # Roll device time up into functional groups so the budget is readable.
    groups = {
        "DSA attention (main_kernel)": lambda n: n == "main_kernel",
        "all-reduce (quickreduce)": lambda n: "quickreduce" in n or "cross_device_reduce" in n,
        "MoE expert GEMM": lambda n: "mfma_moe" in n,
        "MoE reduction/sort/quant": lambda n: ("moe_reduction" in n or "moe_sort" in n
                                               or "moe_sorting" in n or "grouped_topk" in n
                                               or "scaled_quant" in n
                                               or "shared_experts" in n),
        "dense GEMM (hipBLASLt/aiter)": lambda n: (n.startswith("Cijk_") or "hgemm_" in n
                                                   or "bf16gemm" in n),
        "DSA indexer": lambda n: ("mqa_logits" in n or "topk_transform" in n
                                  or "hadamard" in n or "act_quant" in n),
        "norm/rope/cache": lambda n: ("rmsnorm" in n or "rope" in n or "sbhd_cached" in n),
    }
    agg = defaultdict(float)
    total = 0.0
    for e in gpu:
        d = e.get("dur", 0)
        total += d
        for g, pred in groups.items():
            if pred(e["name"]):
                agg[g] += d
                break
        else:
            agg["other"] += d
    print("\nfunctional breakdown of GPU device time")
    print("-" * 55)
    for g, v in sorted(agg.items(), key=lambda kv: -kv[1]):
        print(f"  {100*v/total:6.2f}%  {v/1000:9.1f} ms  {g}")


if __name__ == "__main__":
    main()
