#!/usr/bin/env python3
"""Interleaved sweep of `partition_size` for the aiter decode paged-attention kernel.

SGLang hard-codes `_AITER_PARTITION_SIZE_ROCM = 256` in
`python/sglang/srt/layers/attention/aiter_backend.py:127`. It is a plain module
constant — no env var, no server flag — so it is a source-level knob, and nothing
in the repo records why 256.

partition_size sets how many KV tokens one workgroup covers, so it trades grid
width (parallelism, and the size of the second-pass reduce) against per-workgroup
work and softmax-rescale count. At bs=64 / ctx~8700 / 8 KV heads the grid is
64*8*ceil(ctx/P) workgroups over 256 CUs, i.e. 68 waves at P=256 and 17 at P=1024.

Measurement follows tuning-core/measurement.md Rule 6b: the candidates are
**interleaved** — one timed replay of each per round, N rounds, median across
rounds — rather than run to completion one after another, because a back-to-back
sweep on gfx950 measures clock drift as if it were a kernel difference.

Usage:
    python3 bench_pa_partition.py --sizes 128,256,512,1024 --layout interleave
"""
import argparse
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, "/sgl-workspace/aiter")
from csrc.cpp_itfs.pa.pa_ragged import paged_attention_ragged  # noqa: E402


def build(bs, ctx, num_q_heads, num_kv_heads, head_dim, layout, max_context_len,
          partition_size, device):
    total_tokens = bs * ctx
    pool_tokens = total_tokens * 2

    g = torch.Generator(device=device).manual_seed(0)
    k_cache = torch.randn(pool_tokens, 1, num_kv_heads, head_dim, dtype=torch.bfloat16,
                          device=device, generator=g)
    v_cache = torch.randn(pool_tokens, 1, num_kv_heads, head_dim, dtype=torch.bfloat16,
                          device=device, generator=g)
    q = torch.randn(bs, num_q_heads, head_dim, dtype=torch.bfloat16, device=device,
                    generator=g)

    kv_indptr = torch.arange(0, (bs + 1) * ctx, ctx, dtype=torch.int32, device=device)
    if layout == "contig":
        kv_indices = torch.arange(total_tokens, dtype=torch.int32, device=device)
    elif layout == "interleave":
        idx = torch.arange(total_tokens, device=device)
        seq, pos = idx // ctx, idx % ctx
        kv_indices = (pos * bs + seq).to(torch.int32)
    elif layout == "shuffle":
        kv_indices = torch.randperm(pool_tokens, device=device, generator=g)[
            :total_tokens
        ].to(torch.int32)
    else:
        raise ValueError(layout)

    kv_last_page_lens = torch.ones(bs, dtype=torch.int32, device=device)
    max_num_partitions = (max_context_len + partition_size - 1) // partition_size

    out = torch.empty(bs, num_q_heads, head_dim, dtype=torch.bfloat16, device=device)
    workspace = torch.empty(
        (bs * num_q_heads * max_num_partitions * head_dim) * 4
        + 2 * (bs * num_q_heads * max_num_partitions) * 4,
        dtype=torch.uint8,
        device=device,
    )
    k_scale = v_scale = torch.tensor([1.0], dtype=torch.float32, device=device)
    return dict(
        out=out, workspace=workspace, q=q, k_cache=k_cache, v_cache=v_cache,
        kv_indptr=kv_indptr, kv_indices=kv_indices, kv_last_page_lens=kv_last_page_lens,
        max_num_partitions=max_num_partitions, k_scale=k_scale, v_scale=v_scale,
        partition_size=partition_size,
    )


def run_once(t, scale):
    paged_attention_ragged(
        t["out"], t["workspace"], t["q"], t["k_cache"], t["v_cache"], scale,
        t["kv_indptr"], t["kv_indices"], t["kv_last_page_lens"],
        1, t["max_num_partitions"],
        None, "auto", "NHD", 0.0,
        t["k_scale"], t["v_scale"], None, t["partition_size"],
    )


def capture(t, scale, iters):
    for _ in range(10):
        run_once(t, scale)
    torch.cuda.synchronize()
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
    return g


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--ctx", type=int, default=8704)
    p.add_argument("--layout", default="interleave",
                   choices=["contig", "interleave", "shuffle"])
    p.add_argument("--max-context-len", type=int, default=11264)
    p.add_argument("--sizes", default="128,256,512,1024")
    p.add_argument("--rounds", type=int, default=9)
    p.add_argument("--iters", type=int, default=30)
    a = p.parse_args()

    dev = "cuda"
    num_q_heads, num_kv_heads, head_dim = 40, 8, 128
    scale = 1.0 / (head_dim ** 0.5)
    sizes = [int(s) for s in a.sizes.split(",")]

    # One 4.5 GB KV pool shared by every arm: only the workspace and the
    # partition_size differ, so the arms see byte-identical inputs and the
    # server's own allocation is not crowded out.
    shared = build(a.bs, a.ctx, num_q_heads, num_kv_heads, head_dim, a.layout,
                   a.max_context_len, sizes[0], dev)
    arms = {}
    for ps in sizes:
        t = dict(shared)
        t["partition_size"] = ps
        t["max_num_partitions"] = (a.max_context_len + ps - 1) // ps
        t["out"] = torch.empty_like(shared["out"])
        t["workspace"] = torch.empty(
            (a.bs * num_q_heads * t["max_num_partitions"] * head_dim) * 4
            + 2 * (a.bs * num_q_heads * t["max_num_partitions"]) * 4,
            dtype=torch.uint8, device=dev,
        )
        arms[ps] = (t, capture(t, scale, a.iters))
        # Reference output for the numerics check: all partition sizes must agree.
        arms[ps][1].replay()
        torch.cuda.synchronize()
        t["ref"] = t["out"].clone()

    # Interleaved rounds: rotate through every candidate once per round.
    samples = {ps: [] for ps in sizes}
    for _ in range(a.rounds):
        for ps in sizes:
            _, g = arms[ps]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            g.replay()
            torch.cuda.synchronize()
            samples[ps].append((time.perf_counter() - t0) / a.iters)

    kv_bytes = a.bs * a.ctx * 2 * num_kv_heads * head_dim * 2
    base = statistics.median(samples[256]) if 256 in samples else None
    ver = os.getenv("QKV_VERSION", "default(EXPERIMENTAL where supported)")
    print(f"pa_ragged partition-size sweep   version={ver}")
    print(f"bs={a.bs} ctx={a.ctx} layout={a.layout} rounds={a.rounds} iters={a.iters}")
    print()
    print(f"{'P':>6} {'grid wgs':>9} {'us':>9} {'spread':>8} {'TB/s':>7} "
          f"{'40L ms/step':>12} {'vs P=256':>9}  maxdiff")
    print("-" * 80)
    ref256 = arms[256][0]["ref"] if 256 in arms else None
    for ps in sizes:
        v = samples[ps]
        med = statistics.median(v)
        spread = (max(v) - min(v)) / med * 100
        us = med * 1e6
        tbs = kv_bytes / med / 1e12
        rel = (med / base - 1) * 100 if base else 0.0
        wgs = a.bs * num_kv_heads * ((a.ctx + ps - 1) // ps)
        diff = ((arms[ps][0]["ref"] - ref256).abs().max().item()
                if ref256 is not None else float("nan"))
        print(f"{ps:6d} {wgs:9d} {us:9.1f} {spread:7.2f}% {tbs:7.3f} "
              f"{med*40*1e3:12.3f} {rel:+8.2f}%  {diff:.4g}")


if __name__ == "__main__":
    main()
