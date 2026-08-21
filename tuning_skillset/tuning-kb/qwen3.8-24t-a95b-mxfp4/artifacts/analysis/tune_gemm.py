#!/usr/bin/env python3
"""Search hipBLASLt / asm / triton / opus solutions for this model's bf16 GEMM shapes.

Reports the best per shape against what aiter's shipped config currently picks.
"""
import json, re, sys, time
from collections import defaultdict
import torch, torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
import aiter
from aiter import hipb_mm, hipb_findallsols, hipb_create_extension, gemm_a16w16_asm
from aiter.tuned_gemm import tgemm, get_GEMM_A16W16_config

dev = "cuda:0"; dt = torch.bfloat16
torch.cuda.set_device(dev); hipb_create_extension()

def kernel_us(fn, iters=50):
    for _ in range(15): fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(iters): fn()
        torch.cuda.synchronize()
    p.export_chrome_trace("/tmp/t.json")
    tot = sum(e["dur"] for e in json.load(open("/tmp/t.json"))["traceEvents"]
              if e.get("ph") == "X" and e.get("cat") == "kernel")
    return tot / iters

def ok(y, ref):
    return torch.isfinite(y).all() and (y.float() - ref).abs().max().item() < 0.6 * max(1.0, ref.abs().max().item()) * 1e-2

def tune(name, M, N, K):
    x = torch.randn(M, K, device=dev, dtype=dt) / 8
    w = torch.randn(N, K, device=dev, dtype=dt) / 8
    ref = F.linear(x, w).float()
    cur_cfg = get_GEMM_A16W16_config(M=M, N=N, K=K, bias=False, dtype=str(dt), otype=str(dt))
    cur = kernel_us(lambda: tgemm.mm(x, w, None, otype=dt))
    cands = []
    try:
        sols = hipb_findallsols(x, w.t(), None, dt, None, None, None, False)
    except Exception as e:
        sols = []
    for s in sols:
        try:
            y = hipb_mm(x, w.t(), s, None, dt, None, None, None, False)
            if not ok(y, ref): continue
            cands.append(("hipblaslt", s, kernel_us(lambda: hipb_mm(x, w.t(), s, None, dt, None, None, None, False))))
        except Exception:
            pass
    # triton
    try:
        from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16 as tri
        y = tri(x, w, bias=None, dtype=dt)
        if ok(y, ref): cands.append(("triton", -1, kernel_us(lambda: tri(x, w, bias=None, dtype=dt))))
    except Exception as e: pass
    # asm, various splitK
    for sk in (None, 0, 1, 2, 4, 8):
        try:
            o = torch.empty(M, N, dtype=dt, device=dev)
            y = gemm_a16w16_asm(x, w, o, None, sk, None, False)
            if y is None or not ok(y, ref): continue
            cands.append((f"asm(splitK={sk})", -1, kernel_us(lambda: gemm_a16w16_asm(x, w, torch.empty(M, N, dtype=dt, device=dev), None, sk, None, False))))
        except Exception: pass
    cands.sort(key=lambda c: c[2])
    best = cands[0] if cands else ("none", -1, float("inf"))
    print(f"{name:16s} M{M:<6d} N{N:<6d} K{K:<5d} cur={cur_cfg['libtype']:>9s}/{cur_cfg.get('solidx')} {cur:8.2f}us | best={best[0]:>12s}/{best[1]} {best[2]:8.2f}us | {cur/best[2] if best[2] else 0:5.2f}x  ({len(cands)} cands)")
    for c in cands[:4]:
        print(f"      {c[0]:>12s}/{c[1]:<8} {c[2]:8.2f}us")
    del x, w, ref; torch.cuda.empty_cache()
    return name, M, N, K, cur, best

SHAPES = [("in_proj",4608,8192),("out_proj",8192,2048),("router_gate",512,8192),
          ("shared_gate_up",512,8192),("shared_down",8192,256),("in_proj_ba",32,8192)]
Ms = [int(a) for a in sys.argv[1:]] or [64]
for M in Ms:
    for nm, N, K in SHAPES:
        tune(nm, M, N, K)
