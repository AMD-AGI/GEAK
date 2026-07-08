#!/usr/bin/env python3
"""Count DECODE forward steps captured in the newest torch profiler trace under a directory.

Used by bench_e2e.sh's representativeness gate: a profiling window that under-captures decode biases
BOTH head selection (raw %GPU) and the decode weight-share. The gate needs a cheap decode-step count so
it can enlarge the window and re-capture when the count is below N = max(30, 5*ceil(OSL/CONC)).

Decode-step proxy = the launch count of the busiest COMPUTE kernel whose input shapes are HIDDEN.
Shape-hidden launches are graph-replayed launches, i.e. the decode deployment context (prefill runs
eager with real Input Dims; see attribute_weights.py / harness_lib.deployment_graph_mode). The busiest
such compute kernel (MoE/attn/GEMM) fires ~once per decode forward step, so its hidden-launch count is a
good lower-bound estimate of the decode steps in the window. Prints a single integer to stdout (0 on any
error, so the caller degrades to "re-capture / low-confidence" rather than crashing the bench).
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_SKIP = ("memcpy", "memset", "copy", "cast", "elementwise", "fill", "index_", "cat_")


def _newest_trace(d):
    pats = ("*.pt.trace.json*", "*.trace.json*", "*.json.gz")
    files = []
    for p in pats:
        files += glob.glob(os.path.join(d, "**", p), recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    return max(files, key=os.path.getmtime) if files else ""


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    try:
        from parse_profile import parse_torch_trace  # reuse the exact trace reader
    except Exception:
        print(0)
        return
    tr = _newest_trace(d)
    if not tr:
        print(0)
        return
    try:
        agg, _total_us, _launches = parse_torch_trace(tr)
    except Exception:
        print(0)
        return
    # Rank compute kernels by GPU time; among the busiest few (the real per-forward compute heads, not a
    # high-multiplicity pointwise), take the max shape-hidden launch count. This is decode LAUNCHES (>=
    # decode forward steps, since a per-layer kernel fires once per layer) — a conservative floor that
    # reliably catches the pathological under-capture (hidden ~= 0) and guarantees >= N decode latency
    # samples for a stable decode weight-share.
    try:
        heads = sorted(
            ((info.get("total_us", 0.0), name, info) for name, info in agg.items()
             if not any(s in name.lower() for s in _SKIP)),
            reverse=True,
        )[:5]
        best = 0
        for _t, _name, info in heads:
            by_case = info.get("by_case") or {}
            hidden = sum(c.get("count", 0) for (sig, _dt), c in by_case.items() if not sig)
            if hidden > best:
                best = hidden
    except Exception:
        print(0)
        return
    print(int(best))


if __name__ == "__main__":
    main()
