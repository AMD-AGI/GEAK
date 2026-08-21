#!/usr/bin/env python3
"""Verify the installed tuned-GEMM rows: lookup hits, numerics match, timing holds.

Runs against the real tgemm.mm entry point that sglang's UnquantizedLinearMethod
uses, so what is checked here is what the server will do.

The search that produced the rows timed each candidate once, in its own process.
That is fine for ranking 2000 candidates but too loose to accept a 5% claim, so
this pass interleaves tuned-vs-torch and takes medians over REPS rounds. A row
that cannot beat torch by BEAT here does not belong in the table.
"""
import csv, json, os, statistics, sys
import torch, torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
from aiter.tuned_gemm import tgemm, get_GEMM_A16W16_config

CSV = os.environ.get(
    "CFG", "/sgl-workspace/aiter/aiter/configs/model_configs/qwen3_8_2_4t_a95b_bf16_tuned_gemm.csv"
)
REPS = 5
BEAT = 1.03
dev, dt = "cuda:0", torch.bfloat16
torch.cuda.set_device(dev)


def kernel_us(fn, iters):
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    p.export_chrome_trace("/tmp/vcfg.json")
    ev = json.load(open("/tmp/vcfg.json"))["traceEvents"]
    return sum(e["dur"] for e in ev if e.get("ph") == "X" and e.get("cat") == "kernel") / iters


fails = []
print(f"{'M':>6} {'N':>6} {'K':>6}  {'libtype':10} {'sol':>8} {'search':>8} "
      f"{'tuned':>8} {'torch':>9} {'speedup':>8}  {'err':>9}  verdict")
for row in csv.DictReader(open(CSV)):
    M, N, K = int(row["M"]), int(row["N"]), int(row["K"])
    cfg = get_GEMM_A16W16_config(M=M, N=N, K=K, bias=False, dtype=str(dt), otype=str(dt))
    x = torch.randn(M, K, device=dev, dtype=dt) / 8
    w = torch.randn(N, K, device=dev, dtype=dt) / 8
    ref = F.linear(x, w).float()
    tuned_fn = lambda: tgemm.mm(x, w, None, otype=dt)
    torch_fn = lambda: F.linear(x, w)
    err = (tuned_fn().float() - ref).abs().max().item() / max(1.0, ref.abs().max().item())
    iters = max(3, min(40, int(40 * 64 / M))) if M > 512 else 40
    for _ in range(10):
        tuned_fn(); torch_fn()
    a, b = [], []
    for _ in range(REPS):                      # interleaved, so drift hits both arms
        a.append(kernel_us(tuned_fn, iters))
        b.append(kernel_us(torch_fn, iters))
    tuned, tor = statistics.median(a), statistics.median(b)
    ok_lib = cfg["libtype"] == row["libtype"]
    ok_err = err < 2e-2
    ok_fast = tuned * BEAT < tor
    bad = [t for t, ok in (("LIB", ok_lib), ("ERR", ok_err), ("SLOW", ok_fast)) if not ok]
    if bad:
        fails.append((M, N, K))
    print(f"{M:6d} {N:6d} {K:6d}  {cfg['libtype']:10} {str(cfg.get('solidx')):>8} "
          f"{float(row['us']):8.2f} {tuned:8.2f} {tor:9.2f} {tor/tuned:8.3f}x  {err:9.2e}  "
          f"{'keep' if not bad else 'DROP ' + ' '.join(bad)}")
    del x, w, ref
    torch.cuda.empty_cache()

print(f"\n{len(fails)} failing rows: {fails}")
sys.exit(1 if fails else 0)
