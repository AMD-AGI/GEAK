#!/usr/bin/env python3
"""Interleaved A/B of the two ways to get the bpreshuffle FP8 scale layout.

A (baseline): per-1x128 quant writing scales row-major, then
   ``scale.t().contiguous().t()`` -- a real transposing copy kernel.
B (candidate): ask the quant kernel for ``shuffle_scale=True`` and then only
   reinterpret the strides, no copy.

Both produce byte-identical scale bytes on these shapes (verified separately).
Rule 6b: interleave the arms, one timed replay each per round, median of rounds.
"""
import statistics, sys, time
import torch
sys.path.insert(0, "/sgl-workspace/aiter")
import aiter
from aiter.ops.quant import get_hip_quant

quant = get_hip_quant(aiter.QuantType.per_1x128)

SHAPES = [(64, 5120), (64, 17408), (8192, 5120), (8192, 17408)]
ROUNDS, ITERS = 9, 50


def timed(fn, iters):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    torch.cuda.synchronize()
    return g


print(f"{'shape':>14} {'A copy us':>10} {'B view us':>10} {'quantA':>8} {'quantB':>8} {'delta':>9}  equal")
print("-" * 78)
for M, K in SHAPES:
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

    def arm_a():
        y, s0 = quant(x, quant_dtype=aiter.dtypes.fp8, transpose_scale=False)
        return y, s0.t().contiguous().t()

    def arm_b():
        y, s1 = quant(x, quant_dtype=aiter.dtypes.fp8, transpose_scale=True)
        return y, torch.as_strided(s1, s1.shape, (1, s1.shape[0]))

    def quant_a():
        quant(x, quant_dtype=aiter.dtypes.fp8, transpose_scale=False)

    def quant_b():
        quant(x, quant_dtype=aiter.dtypes.fp8, transpose_scale=True)

    ya, sa = arm_a(); yb, sb = arm_b()
    ok = torch.equal(ya, yb) and torch.equal(sa, sb)

    graphs = {"a": timed(arm_a, ITERS), "b": timed(arm_b, ITERS),
              "qa": timed(quant_a, ITERS), "qb": timed(quant_b, ITERS)}
    samples = {k: [] for k in graphs}
    for _ in range(ROUNDS):
        for k, g in graphs.items():
            torch.cuda.synchronize()
            t0 = time.perf_counter(); g.replay(); torch.cuda.synchronize()
            samples[k].append((time.perf_counter() - t0) / ITERS * 1e6)
    m = {k: statistics.median(v) for k, v in samples.items()}
    print(f"{f'{M}x{K}':>14} {m['a']:10.2f} {m['b']:10.2f} {m['qa']:8.2f} {m['qb']:8.2f} "
          f"{(m['b']/m['a']-1)*100:+8.2f}%  {ok}")
