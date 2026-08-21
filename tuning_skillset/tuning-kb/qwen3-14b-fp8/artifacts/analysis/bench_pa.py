#!/usr/bin/env python3
"""Standalone microbenchmark for the aiter decode paged-attention kernel at the
exact shape this experiment's workload produces.

Qwen3-14B-FP8, TP=1:  40 q heads / 8 kv heads / head_dim 128, bf16 KV, page_size 1.
Decode batch 64, context ~8192..9216 (ISL 8192 + up to OSL 1024).

Usage:
    QKV_VERSION=GOLDEN       python3 bench_pa.py
    QKV_VERSION=EXPERIMENTAL python3 bench_pa.py

The kernel is JIT-built per (version, shape) tuple, so the two versions land in
different build directories and cannot alias each other.
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, "/sgl-workspace/aiter")
from csrc.cpp_itfs.pa.pa_ragged import paged_attention_ragged  # noqa: E402

PARTITION_SIZE = 256


def build(bs, ctx, num_q_heads, num_kv_heads, head_dim, layout, max_context_len, device):
    """Allocate a KV pool and page table shaped like SGLang's at page_size 1."""
    total_tokens = bs * ctx
    # Pool is deliberately larger than the live set so `layout` can scatter into it,
    # mirroring a real allocator that has been running for a while.
    pool_tokens = total_tokens * 2

    k_cache = torch.randn(pool_tokens, 1, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    v_cache = torch.randn(pool_tokens, 1, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    q = torch.randn(bs, num_q_heads, head_dim, dtype=torch.bfloat16, device=device)

    kv_indptr = torch.arange(0, (bs + 1) * ctx, ctx, dtype=torch.int32, device=device)
    if layout == "contig":
        # Each sequence owns one contiguous run of slots.
        kv_indices = torch.arange(total_tokens, dtype=torch.int32, device=device)
    elif layout == "interleave":
        # Sequences round-robin over slots: what a page_size-1 free list produces when
        # 64 sequences are decoding in lockstep and each appends one token per step.
        idx = torch.arange(total_tokens, device=device)
        seq, pos = idx // ctx, idx % ctx
        kv_indices = (pos * bs + seq).to(torch.int32)
    elif layout == "shuffle":
        kv_indices = torch.randperm(pool_tokens, device=device)[:total_tokens].to(torch.int32)
    else:
        raise ValueError(layout)

    kv_last_page_lens = torch.ones(bs, dtype=torch.int32, device=device)
    max_num_partitions = (max_context_len + PARTITION_SIZE - 1) // PARTITION_SIZE

    out = torch.empty(bs, num_q_heads, head_dim, dtype=torch.bfloat16, device=device)
    nbytes_f32 = 4
    workspace = torch.empty(
        (bs * num_q_heads * max_num_partitions * head_dim) * nbytes_f32
        + 2 * (bs * num_q_heads * max_num_partitions) * 4,
        dtype=torch.uint8,
        device=device,
    )
    k_scale = v_scale = torch.tensor([1.0], dtype=torch.float32, device=device)
    return dict(
        out=out, workspace=workspace, q=q, k_cache=k_cache, v_cache=v_cache,
        kv_indptr=kv_indptr, kv_indices=kv_indices, kv_last_page_lens=kv_last_page_lens,
        max_num_partitions=max_num_partitions, k_scale=k_scale, v_scale=v_scale,
    )


def run_once(t, scale):
    paged_attention_ragged(
        t["out"], t["workspace"], t["q"], t["k_cache"], t["v_cache"], scale,
        t["kv_indptr"], t["kv_indices"], t["kv_last_page_lens"],
        1,                       # block_size (page_size 1)
        t["max_num_partitions"],
        None, "auto", "NHD", 0.0,
        t["k_scale"], t["v_scale"], None, PARTITION_SIZE,
    )


def timeit(t, scale, iters=50, warmup=10):
    for _ in range(warmup):
        run_once(t, scale)
    torch.cuda.synchronize()
    # CUDA-graph the measured region: the live server runs this kernel inside a
    # captured decode graph, so launch overhead should not be in the number.
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            run_once(t, scale)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        for _ in range(iters):
            run_once(t, scale)
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        g.replay()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / iters)
    return best * 1e3  # ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--ctx", type=int, default=8704)
    p.add_argument("--layout", default="contig", choices=["contig", "interleave", "shuffle"])
    p.add_argument("--max-context-len", type=int, default=11264)
    p.add_argument("--check", action="store_true", help="dump out tensor checksum for A/B numerics")
    a = p.parse_args()

    dev = "cuda"
    torch.manual_seed(0)
    num_q_heads, num_kv_heads, head_dim = 40, 8, 128
    scale = 1.0 / (head_dim ** 0.5)

    t = build(a.bs, a.ctx, num_q_heads, num_kv_heads, head_dim, a.layout, a.max_context_len, dev)
    ms = timeit(t, scale)

    # One decode step of ONE layer reads K and V for every live token.
    kv_bytes = a.bs * a.ctx * 2 * num_kv_heads * head_dim * 2
    gbps = kv_bytes / (ms * 1e-3) / 1e9
    ver = os.getenv("QKV_VERSION", "GOLDEN")
    print(
        f"version={ver:12s} layout={a.layout:10s} bs={a.bs} ctx={a.ctx}  "
        f"{ms*1e3:8.1f} us   {gbps:7.1f} GB/s   "
        f"40 layers -> {ms*40:6.2f} ms/decode-step"
    )
    if a.check:
        o = t["out"].float()
        print(f"  out: sum={o.sum().item():.6f} absmax={o.abs().max().item():.6f} "
              f"mean={o.mean().item():.8f}")


if __name__ == "__main__":
    main()
