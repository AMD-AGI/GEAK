#!/usr/bin/env python3
"""Measure the achievable HBM bandwidth of one MI355X, to have a roofline to judge
memory-bound kernels against.

The spec figure for gfx950 is ~8 TB/s. That is the number you cannot exceed, not the
number a real kernel can hit. This does a large device-to-device copy (read+write) and a
large reduction (read-only) at 1 GB and 2 GB, both far past any cache, and reports both.

    python3 analysis/hw/hbm_bandwidth.py

Result on crsuse2-m2m-287 (MI355X, ROCm 7.2): ~5.2 TB/s read-only. That is the constant
used in FINDINGS.md to score the paged-attention kernel -- it moves its KV working set at
3.77 TB/s, i.e. 72% of achievable, which is what makes it the top open thread rather than
a finished one. A kernel already at 5 TB/s would not be worth touching.
"""
import torch
torch.cuda.set_device("cuda:3")
for gb in (1, 2):
    n = int(gb*1e9//2)
    x = torch.empty(n, dtype=torch.bfloat16, device="cuda:3").normal_()
    y = torch.empty_like(x)
    for _ in range(5): y.copy_(x)
    torch.cuda.synchronize()
    e0,e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record()
    for _ in range(20): y.copy_(x)
    e1.record(); torch.cuda.synchronize()
    ms = e0.elapsed_time(e1)/20
    print(f"copy {gb}GB: {ms*1000:8.1f} us  {2*n*2/ms/1e6:7.0f} GB/s (r+w)")
    # pure read
    for _ in range(5): s = x.sum()
    torch.cuda.synchronize(); e0.record()
    for _ in range(20): s = x.sum()
    e1.record(); torch.cuda.synchronize()
    ms = e0.elapsed_time(e1)/20
    print(f"read {gb}GB: {ms*1000:8.1f} us  {n*2/ms/1e6:7.0f} GB/s (r)")
