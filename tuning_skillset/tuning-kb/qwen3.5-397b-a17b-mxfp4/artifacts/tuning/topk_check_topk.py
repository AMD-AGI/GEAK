#!/usr/bin/env python3
"""Correctness check for aiter's topk_softmax against a torch reference.

Run before/after the topk_softmax_kernels.cu LDG-width change. The routing
weights feed straight into the MoE, so a wrong index or weight here is a wrong
answer, not a slow one.
"""
import sys, torch
from aiter.ops.moe_op import topk_softmax

dev = f"cuda:{sys.argv[1] if len(sys.argv) > 1 else 0}"
torch.cuda.set_device(dev)
torch.manual_seed(0)
E, K = 512, 10
bad = 0
for M in (1, 3, 64, 65, 128, 255, 256, 512, 8192, 16384):
    for dt in (torch.bfloat16, torch.float16, torch.float32):
        g = torch.randn(M, E, dtype=dt, device=dev)
        w = torch.empty(M, K, dtype=torch.float32, device=dev)
        i = torch.empty(M, K, dtype=torch.int32, device=dev)
        t = torch.empty(M, K, dtype=torch.int32, device=dev)
        topk_softmax(w, i, t, g, True)
        gf = g.float()
        rv, ri = torch.topk(gf, K, dim=-1)
        rw = torch.softmax(rv, dim=-1)
        di = (i.long() != ri).sum().item()
        dw = (w - rw).abs().max().item()
        ok = di == 0 and dw < 2e-3
        bad += not ok
        print(f"M={M:6d} {str(dt).split('.')[-1]:>9}  idx_mismatch={di:6d}  max_w_err={dw:.2e}  {'ok' if ok else 'FAIL'}")
print("ALL OK" if bad == 0 else f"{bad} FAILURES")
