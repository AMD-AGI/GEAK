#!/usr/bin/env python3
"""Sweep the launch geometry of chunk_gated_delta_rule_fwd_kernel_h_blockdim64 at this
model's real prefill shape.

Why this kernel: it is #2 in the prefill profile at 7.70% of device time (477.79 us x 1440
calls = 688 ms of 8940 ms across 4 ranks), and prefill is 36% of the benchmark's wall clock.
It is a Triton kernel in SGLang's own tree, and SGLang deliberately gives it a *single*
hardcoded autotune config -- multi-config autotune corrupts the state pool, because the
kernel writes the final recurrent state back into initial_state in place, so triton's
benchmark phase would run it many times over live state. The source comment says the env
knobs exist "while allowing model/hardware-local validation of the selected tile". That
validation is what this script is.

Shipped default is BV=32, num_warps=4, num_stages=2. At this model's TP=4 shard the grid is
    (cdiv(V, BV), N * H) = (cdiv(128, 32), N * 16) = (4, 32) for a 2-sequence chunk
i.e. 128 workgroups on a 256-CU device. Half the device has no work.

Shapes (Qwen3.5-397B-A17B at TP=4): Hg=4 key heads, H=16 value heads, K=V=128,
BT=CHUNK_SIZE=64, varlen with cu_seqlens, initial_state pool in float32 (mamba_ssm_dtype).
chunked_prefill_size is 16384 and ISL is 8192, so the production call is N=2 x T=8192.

Method: torch.profiler device time (self_device_time_total / count). A wall-clock loop
around the python wrapper measures dispatch (~19 us floor on this stack) and would rank
every config identically -- that mistake is recorded in analysis/topk/bench_topk.py.

    python3 analysis/gdn/bench_chunk_h.py                 # default sweep, cuda:3
    python3 analysis/gdn/bench_chunk_h.py --dev cuda:0 --seqs 1,2,4
"""
import argparse
import itertools
import sys

sys.path.insert(0, "/sgl-workspace/sglang/python")

import torch
import triton
from torch.profiler import ProfilerActivity, profile

P = argparse.ArgumentParser()
P.add_argument("--dev", default="cuda:3")
P.add_argument("--seqs", default="2", help="comma list of sequence counts per chunk")
P.add_argument("--isl", type=int, default=8192)
P.add_argument("--bv", default="16,32,64,128")
P.add_argument("--warps", default="1,2,4,8,16")
P.add_argument("--stages", default="1,2,3,4")
P.add_argument("--iters", type=int, default=30)
P.add_argument("--warmup", type=int, default=8)
A = P.parse_args()

from sglang.kernels.ops.attention.fla import chunk_delta_h as CD

KERNEL = CD.chunk_gated_delta_rule_fwd_kernel_h_blockdim64

# Qwen3.5-397B-A17B-MXFP4, TP=4
HG, H, K, V = 4, 16, 128, 128
DEV = A.dev
torch.cuda.set_device(DEV)
torch.manual_seed(0)


def make(nseq, isl):
    T = nseq * isl
    k = torch.randn(1, T, HG, K, dtype=torch.bfloat16, device=DEV) * 0.1
    w = torch.randn(1, T, H, K, dtype=torch.bfloat16, device=DEV) * 0.1
    u = torch.randn(1, T, H, V, dtype=torch.bfloat16, device=DEV) * 0.1
    # g is chunk_local_cumsum output: non-positive, float32
    g = -torch.rand(1, T, H, dtype=torch.float32, device=DEV) * 0.01
    state = torch.randn(nseq + 2, H, V, K, dtype=torch.float32, device=DEV) * 0.01
    idx = torch.arange(nseq, dtype=torch.int32, device=DEV)
    cu = torch.tensor([i * isl for i in range(nseq + 1)], dtype=torch.int32, device=DEV)
    return dict(k=k, w=w, u=u, g=g, initial_state=state, initial_state_indices=idx,
                cu_seqlens=cu)


def run(args, cfg):
    """One timed measurement of the kernel under triton.Config cfg."""
    KERNEL.configs = [cfg]
    KERNEL.cache.clear()
    for _ in range(A.warmup):
        CD.chunk_gated_delta_rule_fwd_h(**args)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(A.iters):
            CD.chunk_gated_delta_rule_fwd_h(**args)
        torch.cuda.synchronize()
    for ev in prof.key_averages():
        if "chunk_gated_delta_rule_fwd_kernel_h" in ev.key and ev.count:
            return ev.self_device_time_total / ev.count
    return float("nan")


BVS = [int(x) for x in A.bv.split(",")]
WARPS = [int(x) for x in A.warps.split(",")]
STAGES = [int(x) for x in A.stages.split(",")]

for nseq in [int(x) for x in A.seqs.split(",")]:
    args = make(nseq, A.isl)
    ntok = nseq * A.isl
    print(f"\n=== N={nseq} seqs x {A.isl} = {ntok} tokens, Hg={HG} H={H} K={K} V={V} ===")
    print(f"{'BV':>4} {'warps':>6} {'stages':>7} {'grid':>10} {'us':>10}   note")
    base = None
    rows = []
    for bv, nw, ns in itertools.product(BVS, WARPS, STAGES):
        cfg = triton.Config({"BV": bv}, num_warps=nw, num_stages=ns)
        grid = (triton.cdiv(V, bv)) * nseq * H
        try:
            us = run(args, cfg)
        except Exception as e:  # OOM on shared memory, bad tile, unsupported warps
            print(f"{bv:4d} {nw:6d} {ns:7d} {grid:10d} {'-':>10}   {type(e).__name__}: "
                  f"{str(e).splitlines()[0][:60]}")
            continue
        note = ""
        if (bv, nw, ns) == (CD.GDN_CHUNK_H_BV, CD.GDN_CHUNK_H_NUM_WARPS,
                            CD.GDN_CHUNK_H_NUM_STAGES):
            note = "<-- shipped default"
            base = us
        rows.append((us, bv, nw, ns, grid))
        print(f"{bv:4d} {nw:6d} {ns:7d} {grid:10d} {us:10.2f}   {note}")
    rows.sort()
    print(f"\n  best 6 at N={nseq}:")
    for us, bv, nw, ns, grid in rows[:6]:
        sp = f"{base/us:.3f}x" if base else "-"
        print(f"    BV={bv:<4} warps={nw:<3} stages={ns:<2} grid={grid:<5} "
              f"{us:9.2f} us   {sp} vs default")
