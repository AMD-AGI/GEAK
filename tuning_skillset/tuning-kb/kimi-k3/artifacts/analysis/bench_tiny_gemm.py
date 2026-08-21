#!/usr/bin/env python3
"""Where should the K3 tiny-GEMM dispatch thresholds sit on gfx950?

`sglang/kernels/ops/kimi_k3/__init__.py` routes two decode GEMVs to purpose-built
kernels, but only below a hard token limit:

    _K3_N_GEMM_DISPATCH_MAP = {(144, 7168): 16, (896, 7168): 8}   # tiny_n_gemm
    _K3_K_GEMM_DISPATCH_MAP = {(1536, 128): 12}                   # tiny_k_gemm

Above the limit it falls through to `torch.nn.functional.linear`.  The frozen
workload decodes at bs 63, so both fall through, and in the arm-C profile they
are hipBLASLt kernels that run at a fraction of a TB/s:

    Cijk_..MT16x16x1024   69 calls  11.9 us  2.28%   [f_a|b], N=144  K=7168
    Cijk_..MT32x16x64     69 calls   5.7 us  1.08%   f_b,     N=1536 K=128

`max_m` and `split_n` are template parameters of the JIT kernels, so the limit
is not a property of the kernel -- it is a tuning constant.  This sweeps m
against both callees to find where each kernel actually stops winning here.

    python3 analysis/bench_tiny_gemm.py
    python3 analysis/bench_tiny_gemm.py --m 32 63 64 96
"""
import argparse
import os
import statistics
import sys

import torch

sys.path.insert(0, "/sgl-workspace/sglang/python")
sys.path.insert(0, "/work/tuning_skillset/benchmark")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph_harness import cuda_graph_bench  # noqa: E402

from sglang.kernels.ops.gemm.tiny_gemm import (  # noqa: E402
    _default_k_split_n,
    _default_split_n,
    tiny_k_gemm_bf16,
    tiny_n_gemm_bf16,
)

_SPIN = None


def warm_clocks():
    """Idle sclk on this part sits at ~94 MHz; a few ms of replay does not move
    it.  Spin a big GEMM before every timing so all candidates see one clock."""
    global _SPIN
    if _SPIN is None:
        _SPIN = tuple(
            torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
            for _ in range(2)
        )
    a, b = _SPIN
    for _ in range(400):
        a @ b
    torch.cuda.synchronize()


def bench(fn, out, ref):
    warm_clocks()
    r = cuda_graph_bench(
        fn,
        warmup=20,
        iters=60,
        dirty=lambda: out.zero_(),
        verify=lambda: torch.allclose(out, ref, atol=2e-2, rtol=2e-2),
    )
    assert r["mode"] == "cudagraph", r["mode"]
    return statistics.median(r["times_ms"]) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, nargs="*",
                    default=[8, 16, 24, 32, 48, 63, 64, 96, 128])
    args = ap.parse_args()

    torch.manual_seed(0)
    for kind, N, K, shipped in (("n", 144, 7168, 16), ("k", 1536, 128, 12)):
        w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / K**0.5
        print(f"\ntiny_{kind}_gemm  N={N} K={K}  (shipped limit m<={shipped})")
        print(f"{'m':>5} {'torch us':>9} {'tiny us':>9} {'speedup':>8} "
              f"{'split_n':>8} {'max|err|':>9}")
        for m in args.m:
            x = torch.randn(m, K, dtype=torch.bfloat16, device="cuda")
            ref = torch.nn.functional.linear(x, w)
            out = torch.empty_like(ref)

            t_torch = bench(
                lambda: torch.mm(x, w.t(), out=out), out, ref
            )
            # max_m is the kernel's compile-time token bound; build one that
            # covers this m rather than reusing the shipped 16/12.
            try:
                if kind == "n":
                    sp = _default_split_n(N, K, m, x.device)
                    fn = lambda: tiny_n_gemm_bf16(x, w, out, max_m=m, split_n=sp)
                else:
                    sp = _default_k_split_n(N, K)
                    fn = lambda: tiny_k_gemm_bf16(x, w, out, max_m=m)
                fn()
                err = (out.float() - ref.float()).abs().max().item()
                t_tiny = bench(fn, out, ref)
            except Exception as e:
                print(f"{m:>5} {t_torch:>9.2f}   n/a  ({type(e).__name__}: "
                      f"{str(e)[:60]})")
                continue
            print(f"{m:>5} {t_torch:>9.2f} {t_tiny:>9.2f} {t_torch / t_tiny:>7.3f}x "
                  f"{sp:>8} {err:>9.2e}")


if __name__ == "__main__":
    main()
