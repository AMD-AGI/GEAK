#!/usr/bin/env python3
"""Device-time (not wall-time) measurement of aiter's topk_softmax at the shapes
the frozen workload actually hits.

bench_topk.py timed with cuda events around a python loop and got a flat ~19 us
at every M from 64 to 16384 -- that flatness is python/launch overhead, not the
kernel. This uses torch.profiler and reads the summed device duration of the
topkGatingSoftmax kernel itself, which is what the server-side profile ranks on.
"""
import sys, torch
from torch.profiler import profile, ProfilerActivity
from aiter.ops.moe_op import topk_softmax

dev = f"cuda:{sys.argv[1] if len(sys.argv) > 1 else 0}"
torch.cuda.set_device(dev)
E, K = 512, 10
print(f"{'tokens':>7} {'us/call':>9} {'calls':>6}  kernel")
for M in (64, 128, 256, 512, 8192, 16384):
    g = torch.randn(M, E, dtype=torch.bfloat16, device=dev)
    w = torch.empty(M, K, dtype=torch.float32, device=dev)
    i = torch.empty(M, K, dtype=torch.int32, device=dev)
    t = torch.empty(M, K, dtype=torch.int32, device=dev)
    for _ in range(50):
        topk_softmax(w, i, t, g, True)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(200):
            topk_softmax(w, i, t, g, True)
        torch.cuda.synchronize()
    for ev in p.key_averages():
        if "topkGating" in ev.key:
            us = ev.self_device_time_total / ev.count
            print(f"{M:7d} {us:9.3f} {ev.count:6d}  {ev.key[:70]}")
