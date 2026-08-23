#!/usr/bin/env python3
"""Split a profiled window into prefill and decode regimes and budget each separately.

`trace_summary.py` ranks kernels over the whole window, which answers "what is expensive"
but not "what is expensive in the half of the run I am looking at". At ISL 8192 / OSL 1024
the scheduler alternates chunked-prefill passes with decode batches, and the two regimes
have completely different kernel mixes -- so a single ranked table silently reweights the
budget by whatever the capture window happened to contain.

Segmentation uses the DSA attention kernel as a clock. It is named `main_kernel` in both
regimes and runs once per layer, but its cost is strongly bimodal (milliseconds for a
16384-token prefill chunk, tens of microseconds for a 64-row decode step), so the largest
gap in its sorted durations separates them. Every GPU event is then attributed to the
regime of the nearest preceding `main_kernel`.

Reported per regime: contiguous wall span (which includes bubbles, and so is what e2e
actually pays), summed device time, and the top kernels.

    python3 trace_regime.py /tmp/prof_conc64/conc64-TP-0.trace.json.gz
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
    n = re.sub(r"<[^<>]*>", "<>", name)
    n = re.sub(r"\b\d{3,}\b", "N", n)
    return n[:90]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--clock", default="main_kernel")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    ev = load(args.trace)["traceEvents"]
    gpu = sorted(
        (e for e in ev if e.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}
         and e.get("ph") == "X"),
        key=lambda e: e["ts"],
    )

    # Bimodal split of the clock kernel -> per-call regime label.
    ks = sorted(e.get("dur", 0) for e in gpu if e["name"] == args.clock)
    gap_i, gap = 0, 0
    for i in range(1, len(ks)):
        if ks[i] - ks[i - 1] > gap:
            gap, gap_i = ks[i] - ks[i - 1], i
    thresh = (ks[gap_i - 1] + ks[gap_i]) / 2
    print(f"{args.clock}: {len(ks)} calls, prefill/decode threshold {thresh:.1f} us "
          f"({len([k for k in ks if k >= thresh])} prefill, "
          f"{len([k for k in ks if k < thresh])} decode)")

    # Attribute every GPU event to the regime of the nearest preceding clock call.
    regime = None
    tagged = []
    for e in gpu:
        if e["name"] == args.clock:
            regime = "prefill" if e.get("dur", 0) >= thresh else "decode"
        if regime is not None:
            tagged.append((regime, e))

    # Contiguous wall span per regime: sum the durations of maximal same-regime runs.
    spans = defaultdict(float)
    run_lo = run_hi = None
    run_reg = None
    for reg, e in tagged:
        if reg != run_reg:
            if run_reg is not None:
                spans[run_reg] += run_hi - run_lo
            run_reg, run_lo, run_hi = reg, e["ts"], e["ts"] + e.get("dur", 0)
        else:
            run_hi = max(run_hi, e["ts"] + e.get("dur", 0))
    if run_reg is not None:
        spans[run_reg] += run_hi - run_lo

    dev = defaultdict(float)
    per = {"prefill": defaultdict(float), "decode": defaultdict(float)}
    cnt = {"prefill": defaultdict(int), "decode": defaultdict(int)}
    for reg, e in tagged:
        d = e.get("dur", 0)
        dev[reg] += d
        per[reg][norm(e["name"])] += d
        cnt[reg][norm(e["name"])] += 1

    tot_span = sum(spans.values())
    print(f"\nregime wall span (bubbles included -- this is what e2e pays)")
    print("-" * 66)
    for reg in ("prefill", "decode"):
        print(f"  {reg:8s} {spans[reg]/1000:9.1f} ms  {100*spans[reg]/tot_span:5.1f}% of window"
              f"   device-busy {dev[reg]/1000:8.1f} ms "
              f"({100*dev[reg]/spans[reg] if spans[reg] else 0:5.1f}% occupancy)")

    for reg in ("prefill", "decode"):
        print(f"\ntop kernels in {reg}  (share of the {reg} wall span)")
        print("-" * 90)
        for k, v in sorted(per[reg].items(), key=lambda kv: -kv[1])[: args.top]:
            print(f"  {100*v/spans[reg]:6.2f}%  {v/1000:8.1f} ms {cnt[reg][k]:6d} calls "
                  f"{v/cnt[reg][k]:8.1f} us  {k}")


if __name__ == "__main__":
    main()
