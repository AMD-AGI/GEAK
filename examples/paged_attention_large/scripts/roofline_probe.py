#!/usr/bin/env python3
"""Probe how high paged_attention can push HBM bandwidth / GFLOP/s.

Builds *consistent* large decode inputs directly (bypassing the synthetic
build_inputs, which caps seq_lens to ~32), giving each sequence its own KV
blocks so the kernel must actually stream KV from HBM. Times with CUDA events
and reports achieved bandwidth and compute vs the empirical MI300 peaks.

  python3 scripts/roofline_probe.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import task_runner as T  # noqa: E402  (module-level: chdir + path setup)

import torch  # noqa: E402

# Empirical MI300 peaks (from rocprof-compute roofline.csv on this box).
PEAK_HBM_GBPS = 4170.53
PEAK_BF16_MFMA_GFLOPS = 510391.03

H = 16          # query heads
KVH = 1         # kv heads (GQA ratio 16)
D = 128         # head size
X = 8           # kv-cache packing factor (head_size/x = 16)
BLK = 16        # block size
PART = 256      # PARTITION_SIZE
DTYPE = torch.bfloat16


def build(S: int, L: int):
    dev = "cuda"
    nblk_per_seq = math.ceil(L / BLK)
    num_blocks = S * nblk_per_seq            # disjoint blocks -> real HBM traffic
    P = math.ceil(L / PART)

    q = torch.randn(S, H, D, dtype=DTYPE, device=dev)
    key_cache = torch.randn(num_blocks, KVH, D // X, BLK, X, dtype=DTYPE, device=dev)
    value_cache = torch.randn(num_blocks, KVH, D, BLK, dtype=DTYPE, device=dev)
    out = torch.empty(S, H, D, dtype=DTYPE, device=dev)
    exp_sums = torch.empty(S, H, P, dtype=torch.float32, device=dev)
    max_logits = torch.empty(S, H, P, dtype=torch.float32, device=dev)
    tmp_out = torch.empty(S, H, P, D, dtype=DTYPE, device=dev)
    # each seq i owns blocks [i*nblk_per_seq, (i+1)*nblk_per_seq)
    block_tables = (
        torch.arange(num_blocks, dtype=torch.int32, device=dev).reshape(S, nblk_per_seq)
    )
    seq_lens = torch.full((S,), L, dtype=torch.int32, device=dev)
    query_start_loc = torch.arange(S + 1, dtype=torch.int32, device=dev)  # 1 token/seq
    k_scale = torch.ones(1, dtype=torch.float32, device=dev)
    v_scale = torch.ones(1, dtype=torch.float32, device=dev)

    args = (out, exp_sums, max_logits, tmp_out, q, key_cache, value_cache,
            KVH, 1.0 / math.sqrt(D), block_tables, seq_lens, query_start_loc,
            BLK, L, None, "auto", k_scale, v_scale, None, "f16")
    # bytes that MUST come from HBM (KV is >L2 by construction): K + V, bf16
    kv_bytes = S * L * KVH * D * 2 * 2
    # attention flops: QK^T (2*L*D) + softmax*V (2*L*D), per (seq, query head)
    flops = S * H * (2 * L * D + 2 * L * D)
    alloc_gb = (key_cache.nbytes + value_cache.nbytes) / 1e9
    return args, kv_bytes, flops, alloc_gb, num_blocks


def bench(op, args, iters=30, warmup=10):
    for _ in range(warmup):
        op(*args)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record(); op(*args); ends[i].record()
    torch.cuda.synchronize()
    return sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / iters  # ms


def profile_one(op, S: int, L: int, iters: int):
    """Run one config in a steady loop (no timing) for an external profiler."""
    args, kv_bytes, flops, alloc_gb, num_blocks = build(S, L)
    print(f"[profile] S={S} L={L} KValloc={alloc_gb:.1f}GB num_blocks={num_blocks} iters={iters}",
          flush=True)
    for _ in range(iters):
        op(*args)
    torch.cuda.synchronize()


def main():
    op = T._load_op()

    # Single-config profiling mode (for rocprof-compute): set PROBE_S / PROBE_L.
    if os.environ.get("PROBE_S"):
        S = int(os.environ["PROBE_S"])
        L = int(os.environ.get("PROBE_L", "2048"))
        iters = int(os.environ.get("PROBE_ITERS", "20"))
        profile_one(op, S, L, iters)
        return

    configs = [
        (1024, 1024), (1024, 4096),
        (4096, 2048), (8192, 2048),
        (2048, 8192), (8192, 8192),
        (16384, 4096),
    ]
    print(f"{'S(seqs)':>8} {'ctx_L':>6} {'KValloc_GB':>10} {'ms':>8} "
          f"{'HBM_GB/s':>9} {'HBM%':>6} {'GFLOP/s':>9} {'BF16%':>6}")
    for S, L in configs:
        try:
            args, kv_bytes, flops, alloc_gb, _ = build(S, L)
            if alloc_gb > 80:
                print(f"{S:>8} {L:>6} {alloc_gb:>10.1f}  (skip: alloc too large)")
                continue
            ms = bench(op, args)
            t = ms / 1e3
            bw = kv_bytes / t / 1e9
            gf = flops / t / 1e9
            print(f"{S:>8} {L:>6} {alloc_gb:>10.1f} {ms:>8.3f} "
                  f"{bw:>9.1f} {100*bw/PEAK_HBM_GBPS:>5.1f}% "
                  f"{gf:>9.1f} {100*gf/PEAK_BF16_MFMA_GFLOPS:>5.2f}%")
            del args
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"{S:>8} {L:>6}  ERROR: {str(e)[:80]}")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
