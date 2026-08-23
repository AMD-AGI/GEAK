#!/usr/bin/env python3
"""Calibrate what a trivially cheap kernel costs on this device, so a 10 us topk-softmax
can be judged in absolute terms rather than against nothing.

The question in FINDINGS.md attempt #4 was whether 10.57 us for a 64x512 topk-softmax is
"slow". 64 rows x 512 experts is 32k elements -- nothing. This times a torch.softmax over
exactly that shape using torch.profiler device time (the same method as prof_topk.py,
because a wall-clock loop around a python wrapper measures ~19 us of dispatch and tells you
nothing).

    python3 analysis/topk/calib_elementwise.py

Result: 4.40 us for a pure elementwise pass over the same tensor. So the launch-and-touch
floor is ~4.4 us and the stock topk kernel was spending ~6 us on top of it -- enough to be
worth attacking, but bounded: even a perfect kernel could only return ~6 us per call. The
patched kernel lands at 8.40 us, taking about a third of that gap.
"""
import torch
from torch.profiler import profile, ProfilerActivity
torch.cuda.set_device("cuda:3")
x = torch.randn(64, 512, dtype=torch.bfloat16, device="cuda:3")
for _ in range(50): y = torch.softmax(x.float(), -1)
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as p:
    for _ in range(200): y = torch.softmax(x.float(), -1)
    torch.cuda.synchronize()
for ev in p.key_averages():
    if ev.self_device_time_total > 0 and ev.count >= 100:
        print(f"{ev.self_device_time_total/ev.count:8.3f} us x{ev.count}  {ev.key[:70]}")
