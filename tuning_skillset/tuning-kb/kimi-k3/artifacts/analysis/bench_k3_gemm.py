#!/usr/bin/env python3
"""What do the three `_k3_bf16_gemm` call sites cost, and is torch the right callee?

`kimi_k3.py::_k3_bf16_gemm` deliberately calls `torch.nn.functional.linear` /
`torch.mm` instead of going through `UnquantizedLinearMethod`, because the fused
MoE front, the deferred shared-down GEMM and the merged KDA b/f/a GEMV operate on
raw merged weight views rather than on a Linear module.  On this stack
`SGLANG_USE_AITER=1` means every *other* bf16 linear is dispatched by
`aiter.tuned_gemm`, so these three are the only dense GEMMs still landing on
torch -> hipBLASLt.  In the arm-C decode profile they are the `Cijk_*` kernels:

    MT160x64x128    92 calls  31.3 us   7.86%   fused front
    MT64x32x256     92 calls   8.5 us   2.17%   shared down
    MT16x16x1024    69 calls  11.9 us   2.28%   merged b/f/a
    MT32x16x64      69 calls   5.7 us   1.08%   (f_b, tiny)

This times each shape three ways -- torch (what runs today), `tgemm.mm` (aiter's
tuned dispatch, i.e. what routing the helper through aiter would give), and each
aiter libtype that accepts the shape -- so the prize is sized before anything is
changed.  Timing is graph-captured per `tuning-core/graph_captured_benchmarking.md`.

    python3 analysis/bench_k3_gemm.py
    python3 analysis/bench_k3_gemm.py --m 1 8 63 64
"""
import argparse
import os
import statistics
import sys

import torch

sys.path.insert(0, "/work/tuning_skillset/benchmark")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph_harness import cuda_graph_bench  # noqa: E402

H = 7168  # config.hidden_size

# (label, N, K) at TP=8, from kimi_k3.py:
#   front  = shared gate_up (2*2*3072/8) + router gate (896) + latent down (3584)
#   shared = shared down, K = 2*3072/8
#   bfa    = f_a_proj (head_dim 128) + b_proj (HV 12) padded to a multiple of 8
SHAPES = [
    ("front  [gate_up|gate|latent]", 1536 + 896 + 3584, H),
    ("shared-down", H, 768),
    ("bfa    [f_a|b]", 144, H),
]


_SPIN = None


def warm_clocks(seconds=1.0):
    """Hold the GPU busy so sclk leaves its 94 MHz idle floor.

    A graph-captured replay of one 40 us GEMM is ~2 ms of work in total, which
    is far too short to move the clocks: measured cold, every shape here reads
    ~35% slow, and the first candidate in a list reads slower than the last.
    Spin a big GEMM first, and again between candidates, so all of them are
    timed at the same (boosted) clock.
    """
    global _SPIN
    if _SPIN is None:
        a = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
        _SPIN = (a, torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda"))
    a, b = _SPIN
    ev = torch.cuda.Event(enable_timing=True)
    ev2 = torch.cuda.Event(enable_timing=True)
    ev.record()
    for _ in range(400):
        a @ b
    ev2.record()
    ev2.synchronize()


def bench(fn, dirty=None, verify=None):
    warm_clocks()
    r = cuda_graph_bench(fn, warmup=20, iters=60, dirty=dirty, verify=verify)
    assert r["mode"] == "cudagraph", r["mode"]
    return statistics.median(r["times_ms"]) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, nargs="*", default=[63])
    args = ap.parse_args()

    from aiter.tuned_gemm import (
        get_GEMM_A16W16_config,
        is_skinny_default_shape,
        solMap,
        tgemm,
    )
    from aiter.ops.gemm_op_common import get_padded_m

    torch.manual_seed(0)
    for label, N, K in SHAPES:
        w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / K**0.5
        for M in args.m:
            x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
            ref = torch.nn.functional.linear(x, w)
            cfg = get_GEMM_A16W16_config(
                M, N, K, False, "torch.bfloat16", "torch.bfloat16", False, False
            )
            print(
                f"\n{label}  M={M} N={N} K={K}   "
                f"weight {N * K * 2 / 1e6:.1f} MB   padded_M="
                f"{[get_padded_m(M, N, K, gl) for gl in (0, 1)]}   "
                f"aiter picks {cfg['libtype']}"
                f"{' (skinny-default)' if is_skinny_default_shape(M, N, K, torch.bfloat16) else ''}"
            )

            t_torch = bench(lambda: torch.nn.functional.linear(x, w))
            print(f"    {'torch F.linear':<22} {t_torch:7.2f} us  (what runs today)")

            out = tgemm.mm(x, w, None, otype=torch.bfloat16)
            err = (out.float() - ref.float()).abs().max().item()
            t_aiter = bench(lambda: tgemm.mm(x, w, None, otype=torch.bfloat16))
            print(
                f"    {'tgemm.mm (' + cfg['libtype'] + ')':<22} {t_aiter:7.2f} us  "
                f"{t_torch / t_aiter:5.3f}x   max|err| {err:.2e}"
            )

            # Only the solidx-free backends can be probed blind; asm / opus /
            # flydsl need a candidate kernel id, which is what the aiter tuner
            # (csrc/gemm_a16w16/gemm_a16w16_tune.py) searches for.
            for name in ("triton", "skinny"):
                fn = solMap[name]
                try:
                    o = fn(x, w, cfg.get("solidx", 0), None, torch.bfloat16,
                           config=cfg)
                    e_ = (o.float() - ref.float()).abs().max().item()
                    t = bench(lambda: fn(x, w, cfg.get("solidx", 0), None,
                                         torch.bfloat16, config=cfg))
                except Exception as exc:
                    print(f"    {name:<22} n/a  ({type(exc).__name__}: "
                          f"{str(exc)[:60]})")
                    continue
                print(f"    {name:<22} {t:7.2f} us  {t_torch / t:5.3f}x   "
                      f"max|err| {e_:.2e}")


if __name__ == "__main__":
    main()
