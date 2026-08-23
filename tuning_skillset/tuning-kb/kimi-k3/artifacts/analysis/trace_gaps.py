#!/usr/bin/env python3
"""Is the decode step GPU-bound?  Union-busy vs wall span for a captured trace.

The kernel table says 36.18 ms of device time in the window, but a step at
concurrency 64 lands near 52 ms of wall clock, so either the window is more than
one step or the GPU is idle for a third of it.  This answers that from the trace
itself: per device stream, the union of kernel intervals against the span they
cover, plus the biggest gaps and which host op sits in them.

    python3 analysis/trace_gaps.py analysis/profiles/candC_s3/candC_s3-TP-0.trace.json.gz
"""
import gzip
import json
import sys
from collections import defaultdict


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)


def union(iv):
    iv = sorted(iv)
    tot, cur_s, cur_e = 0.0, None, None
    merged = []
    for s, e in iv:
        if cur_s is None:
            cur_s, cur_e = s, e
        elif s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            tot += cur_e - cur_s
            cur_s, cur_e = s, e
    if cur_s is not None:
        merged.append((cur_s, cur_e))
        tot += cur_e - cur_s
    return tot, merged


def main():
    path = sys.argv[1]
    tr = load(path)
    ev = tr["traceEvents"]

    kern = defaultdict(list)  # (pid, tid) -> intervals
    host = []
    launches = []
    for e in ev:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat", "")
        if cat in ("kernel", "gpu_op", "Kernel", "gpu_memcpy", "gpu_memset"):
            kern[(e["pid"], e["tid"])].append((e["ts"], e["ts"] + e.get("dur", 0.0)))
        elif cat in ("cpu_op", "user_annotation", "cuda_runtime", "hip_runtime"):
            host.append(e)
            if "GraphLaunch" in e.get("name", ""):
                launches.append(e)

    print(f"{len(kern)} device streams")
    all_iv = []
    for k, iv in sorted(kern.items(), key=lambda kv: -len(kv[1])):
        busy, merged = union(iv)
        span = merged[-1][1] - merged[0][0]
        all_iv += iv
        print(f"  stream {k}: {len(iv):>5} kernels  busy {busy/1e3:8.2f} ms  "
              f"span {span/1e3:8.2f} ms  -> {100*busy/span:5.1f}% busy")

    busy, merged = union(all_iv)
    span = merged[-1][1] - merged[0][0]
    print(f"\nall streams: busy {busy/1e3:.2f} ms over a {span/1e3:.2f} ms span "
          f"-> {100*busy/span:.1f}% busy, {(span-busy)/1e3:.2f} ms idle")

    gaps = []
    for (s0, e0), (s1, e1) in zip(merged, merged[1:]):
        gaps.append((s1 - e0, e0, s1))
    gaps.sort(reverse=True)
    print(f"\n{len(gaps)} gaps; top 12:")
    for g, e0, s1 in gaps[:12]:
        # host ops covering the gap, longest first
        cov = [h for h in host
               if h["ts"] <= s1 and h["ts"] + h.get("dur", 0) >= e0]
        cov.sort(key=lambda h: -h.get("dur", 0))
        names = ", ".join(f"{h['name'][:44]}({h.get('dur',0)/1e3:.1f}ms)"
                          for h in cov[:3])
        print(f"  {g/1e3:7.2f} ms idle  <- {names or 'no host op covers it'}")

    print(f"\n{len(launches)} graph launches on the host")
    if launches:
        launches.sort(key=lambda h: h["ts"])
        print(f"  first {launches[0]['ts']/1e3:.2f} ms, "
              f"last {launches[-1]['ts']/1e3:.2f} ms")


if __name__ == "__main__":
    main()
