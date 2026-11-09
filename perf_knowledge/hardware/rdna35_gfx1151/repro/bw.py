#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read-read-write streaming bandwidth (torch.add of two buffers into a third) vs working set.
Host-timed, so the small (LLC-sized) cases measure launch overhead, not the cache."""
import sys

if any(a in ("-h", "--help") for a in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

import time

import torch

torch.cuda.init()
print("arch:", torch.cuda.get_device_properties(0).gcnArchName, "| L3 = 32 MB")
for mb in (8, 64, 256, 1024):
    n = mb * 1024 * 1024 // 4
    x = torch.rand(n, device='cuda'); y = torch.rand(n, device='cuda')
    for _ in range(3): o = x + y          # warm
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    R = 20
    for _ in range(R): o = x + y
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / R
    byts = 3 * n * 4                       # 2 reads + 1 write
    tag = "<= L3 (32MB), may be cache-served" if mb*3 <= 32 else "> L3, DRAM-bound"
    print("  buf=%5d MB  traffic=%6.0f MB  %7.3f ms  -> %6.1f GB/s   %s"
          % (mb, byts/1e6, dt*1e3, byts/dt/1e9, tag))
    del x, y, o; torch.cuda.empty_cache()
