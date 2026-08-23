#!/usr/bin/env python3
"""Isolate aiter's topkGatingSoftmax at the decode and prefill routing shapes.

Decode profile: vllm::moe::topkGatingSoftmax<bf16,VPT=32,EXPERTS=512,WARPS_PER_CTA=2,...>
costs 10.33 us for a 64x512 top-10 routing problem (3.52% of decode device time).
aiter's launcher hard-codes WARPS_PER_TB=2 for EXPERTS==512 and picks
BYTES_PER_LDG=64 -> VPT=32 -> THREADS_PER_ROW=16 -> ROWS_PER_WARP=4 ->
ROWS_PER_CTA=8, so 64 tokens launch 8 workgroups on a 256-CU GPU.
"""
import sys, torch
from aiter.ops.moe_op import topk_softmax

dev = f"cuda:{sys.argv[1] if len(sys.argv) > 1 else 0}"
E, K = 512, 10
print(f"{'tokens':>7} {'us':>9} {'blocks(now)':>12}")
for M in (64, 128, 256, 8192, 16384):
    g = torch.randn(M, E, dtype=torch.bfloat16, device=dev)
    w = torch.empty(M, K, dtype=torch.float32, device=dev)
    i = torch.empty(M, K, dtype=torch.int32, device=dev)
    tei = torch.empty(M, K, dtype=torch.int32, device=dev)
    for _ in range(20):
        topk_softmax(w, i, tei, g, True)
    torch.cuda.synchronize(dev)
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    N = 200
    e0.record()
    for _ in range(N):
        topk_softmax(w, i, tei, g, True)
    e1.record()
    torch.cuda.synchronize(dev)
    us = e0.elapsed_time(e1) * 1000 / N
    rows_per_cta = 8            # WARPS_PER_TB=2 * ROWS_PER_WARP=4
    print(f"{M:7d} {us:9.3f} {(M + rows_per_cta - 1)//rows_per_cta:12d}")
