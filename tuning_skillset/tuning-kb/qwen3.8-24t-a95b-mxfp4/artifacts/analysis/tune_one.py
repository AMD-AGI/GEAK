#!/usr/bin/env python3
"""Time one GEMM shape across hipBLASLt solutions + asm/triton. Crash-resumable.

usage: tune_one.py OUT.jsonl M N K [start_idx]
Each candidate result is flushed immediately, so a GPU fault only loses one candidate.
"""
import json, os, sys
import torch, torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
import aiter
from aiter import hipb_mm, hipb_findallsols, hipb_create_extension, gemm_a16w16_asm
from aiter.tuned_gemm import tgemm, get_GEMM_A16W16_config

out_path, M, N, K = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
start = int(sys.argv[5]) if len(sys.argv) > 5 else 0
dev = "cuda:0"; dt = torch.bfloat16
torch.cuda.set_device(dev); hipb_create_extension()
f = open(out_path, "a")
def emit(**kw):
    kw.update(M=M, N=N, K=K); f.write(json.dumps(kw) + "\n"); f.flush(); os.fsync(f.fileno())

x = torch.randn(M, K, device=dev, dtype=dt) / 8
w = torch.randn(N, K, device=dev, dtype=dt) / 8
ref = F.linear(x, w).float()
tol = 2e-2 * max(1.0, ref.abs().max().item())

def kernel_us(fn, iters=40):
    iters = max(3, min(iters, int(40 * 64 / max(1, M))) ) if M > 512 else iters
    for _ in range(10): fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(iters): fn()
        torch.cuda.synchronize()
    p.export_chrome_trace(f"/tmp/t{os.getpid()}.json")
    return sum(e["dur"] for e in json.load(open(f"/tmp/t{os.getpid()}.json"))["traceEvents"]
               if e.get("ph") == "X" and e.get("cat") == "kernel") / iters

def check(y):
    return y is not None and torch.isfinite(y).all().item() and (y.float() - ref).abs().max().item() < tol

sols = list(hipb_findallsols(x, w.t(), bias=None, out_dtype=dt))
extra = [("triton", -1), ("torch", -1)] + [("asm", sk) for sk in (0, 1, 2, 4, 8)]
allc = [("hipblaslt", s) for s in sols] + extra
if start == 0:
    cfg = get_GEMM_A16W16_config(M=M, N=N, K=K, bias=False, dtype=str(dt), otype=str(dt))
    emit(kind="current", lib=cfg["libtype"], sol=cfg.get("solidx"), us=kernel_us(lambda: tgemm.mm(x, w, None, otype=dt)))
    emit(kind="ncands", n=len(allc))

for i in range(start, len(allc)):
    lib, s = allc[i]
    emit(kind="probe", idx=i, lib=lib, sol=s)          # marker: if we fault, driver resumes at i+1
    try:
        if lib == "hipblaslt":
            fn = lambda: hipb_mm(x, w.t(), s, None, dt, None, None, None, False)
        elif lib == "triton":
            from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16 as tri
            fn = lambda: tri(x, w, bias=None, dtype=dt)
        elif lib == "torch":
            fn = lambda: F.linear(x, w)
        else:
            fn = lambda: gemm_a16w16_asm(x, w, torch.empty(M, N, dtype=dt, device=dev), None, s, None, False)
        y = fn(); torch.cuda.synchronize()
        if not check(y):
            emit(kind="bad", idx=i, lib=lib, sol=s); continue
        emit(kind="ok", idx=i, lib=lib, sol=s, us=kernel_us(fn))
    except Exception as e:
        emit(kind="err", idx=i, lib=lib, sol=s, msg=str(e)[:120])
emit(kind="done")
