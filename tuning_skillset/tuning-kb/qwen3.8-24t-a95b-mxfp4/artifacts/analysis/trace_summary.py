#!/usr/bin/env python3
"""Summarise a torch-profiler trace: GPU kernel time by name, grouped.

Usage: trace_summary.py <trace.json.gz> [--top N] [--group]
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


def kernels(trace):
    """Yield (name, dur_us) for GPU kernel events."""
    for e in trace.get("traceEvents", []):
        if e.get("ph") != "X":
            continue
        cat = e.get("cat", "")
        if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            yield e["name"], e.get("dur", 0), cat


def short(name):
    """Collapse a mangled/templated kernel name to something readable."""
    n = name
    n = re.sub(r"<.*>", "", n)          # template args
    n = re.sub(r"\(.*\)", "", n)        # signature
    n = n.replace("void ", "").strip()
    n = n.split("::")[-1]
    return n[:78] if n else name[:78]


def bucket(name):
    n = name.lower()
    if "moe1" in n or "moe2" in n:
        return "MoE GEMM (aiter asm mxfp4)"
    if "moe_sort" in n or "moe_sorting" in n or "topkgating" in n or "moe_align" in n:
        return "MoE routing/sort"
    if "paged_attention" in n or "fmha" in n or "attn_fwd" in n or "mha_" in n:
        return "full attention"
    if "gated_delta" in n or "chunk_" in n or "causal_conv" in n or "recurrent" in n:
        return "linear attention (GDN)"
    if "cijk" in n or "gemm" in n or "hgemm" in n:
        return "dense GEMM"
    if "reduce" in n and ("cross_device" in n or "all" in n or "quick" in n):
        return "collective"
    if "sample" in n or "argmax" in n or "softmax_kernel" in n:
        return "sampling"
    if "rmsnorm" in n or "norm" in n:
        return "norm"
    if "rope" in n or "rotary" in n:
        return "rope"
    if "memcpy" in n or "memset" in n or "Memcpy" in name or "Memset" in name:
        return "memcpy/memset"
    if "elementwise" in n or "act_and_mul" in n or "cat_" in n or "copy" in n or "index" in n:
        return "elementwise/copy"
    if "quant" in n or "scaled" in n:
        return "quant"
    return "other"


def main():
    path = sys.argv[1]
    topn = 30
    if "--top" in sys.argv:
        topn = int(sys.argv[sys.argv.index("--top") + 1])
    tr = load(path)
    by_name = defaultdict(float)
    by_count = defaultdict(int)
    total = 0.0
    for name, dur, cat in kernels(tr):
        key = short(name) if cat == "kernel" else cat
        by_name[key] += dur
        by_count[key] += 1
        total += dur

    print(f"# {path}")
    print(f"# total GPU kernel time: {total/1000:.3f} ms over {sum(by_count.values())} launches")
    print()
    by_bucket = defaultdict(float)
    for k, v in by_name.items():
        by_bucket[bucket(k)] += v
    print("## by bucket")
    for k, v in sorted(by_bucket.items(), key=lambda x: -x[1]):
        print(f"  {100*v/total:6.2f}%  {v/1000:9.3f} ms  {k}")
    print()
    print(f"## top {topn} kernels")
    for k, v in sorted(by_name.items(), key=lambda x: -x[1])[:topn]:
        print(f"  {100*v/total:6.2f}%  {v/1000:9.3f} ms  n={by_count[k]:6d}  {k}")


if __name__ == "__main__":
    main()
