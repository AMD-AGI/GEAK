#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Drift under 4 minutes of sustained memory-bound load (256 MB torch.add)."""
import sys

if any(a in ("-h", "--help") for a in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

import statistics
import time

import torch

torch.cuda.init()
n = 256*1024*1024//4
x = torch.rand(n, device='cuda'); y = torch.rand(n, device='cuda')
def bw():
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(40): torch.add(x, y)
    torch.cuda.synchronize()
    return 3*n*4/((time.perf_counter()-t0)/40)/1e9
for _ in range(5): bw()
s=[]; t0=time.time()
while time.time()-t0 < 240:          # 4 minutes of sustained load
    v=bw(); s.append((time.time()-t0, v))
first=[v for t,v in s if t<30]; last=[v for t,v in s if t>210]
print("samples=%d over %.0fs" % (len(s), s[-1][0]))
print("first 30s: mean=%.1f" % statistics.mean(first))
print("last  30s: mean=%.1f" % statistics.mean(last))
print("drift = %+.2f%%" % (100*(statistics.mean(last)-statistics.mean(first))/statistics.mean(first)))
allv=[v for _,v in s]
print("overall mean=%.1f sd=%.2f CV=%.2f%% min=%.1f max=%.1f" %
      (statistics.mean(allv), statistics.stdev(allv), 100*statistics.stdev(allv)/statistics.mean(allv), min(allv), max(allv)))
