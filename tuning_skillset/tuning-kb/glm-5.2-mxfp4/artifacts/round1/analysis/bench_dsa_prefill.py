#!/usr/bin/env python3
"""Isolated bench of the DSA sparse-MLA prefill op at the shape this workload runs.

Shape taken from the concurrency-64 profile (analysis/profile_conc64_tp0.json):
    q_nope [16384, 8, 512] bf16   q_rope [16384, 8, 64] bf16
    kv     [3283392, 1, 576] fp8_e4m3   indices [16384, 1, 2048] int32
i.e. tp_q_head_num=8 (64 heads / TP8), v_head_dim=512, tail 64, index_topk=2048.

The live path is: cat(q_nope, q_rope) -> .to(fp8) -> tilelang_sparse_fwd. All three
of those are charged to the candidate here, because all three are what the candidate
replaces.

Candidates are timed INTERLEAVED (tuning-core/measurement.md Rule 6b): on gfx950 a
back-to-back A/B drifts far more than the effect being measured.
"""
import argparse
import os
import statistics
import sys

import torch

sys.path.insert(0, "/sgl-workspace/sglang/python")


def make_inputs(seq, heads, d_v, d_tail, topk, n_slots, device, isl=8192, page=64, seed=0):
    """Build inputs matching a real chunked-prefill step of this workload.

    A 16384-token chunk is two ISL-8192 requests (--chunked-prefill-size 16384). Each
    request's KV lives in `page`-sized pages scattered through the pool (radix cache is
    disabled, so allocation is fragmented), and DSA's indexer emits, per query at prefix
    length L:

      L <= topk : the whole prefix, in order, then -1 padding  (naive_paged_transform)
      L >  topk : topk scattered slots, in selection order      (radix_topk, no padding)

    Both branches are reproduced here, because the padding structure is precisely what the
    block-skip candidate exploits and a uniformly-random index row would hide it.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    dim = d_v + d_tail
    q_nope = torch.randn(seq, heads, d_v, generator=g, device=device, dtype=torch.bfloat16) * 0.1
    q_rope = torch.randn(seq, heads, d_tail, generator=g, device=device, dtype=torch.bfloat16) * 0.1
    kv = (torch.randn(n_slots, 1, dim, generator=g, device=device, dtype=torch.bfloat16) * 0.1).to(
        torch.float8_e4m3fn
    )

    n_req = seq // isl
    npage_req = isl // page
    idx = torch.empty(seq, topk, dtype=torch.int32, device=device)
    ar_topk = torch.arange(topk, device=device)
    for r in range(n_req):
        # Scattered page table -> per-position cache slot, exactly page_to_slot().
        pages = torch.randperm(n_slots // page, generator=g, device=device)[:npage_req]
        slot = (pages.to(torch.int64) * page).repeat_interleave(page) + (
            torch.arange(isl, device=device) % page
        )
        pos = torch.arange(isl, device=device)
        L = (pos + 1).unsqueeze(1)  # causal prefix length

        # long-prefix branch: topk scattered picks from [0, L)
        pick = (torch.rand(isl, topk, generator=g, device=device) * L).long().clamp(max=isl - 1)
        rows = slot[pick].to(torch.int32)
        # short-prefix branch: the in-order prefix, then -1
        short = torch.where(
            ar_topk.unsqueeze(0) < L,
            slot[ar_topk.clamp(max=isl - 1)].unsqueeze(0).expand(isl, topk),
            torch.full((1, 1), -1, dtype=torch.int64, device=device),
        ).to(torch.int32)
        idx[r * isl : (r + 1) * isl] = torch.where(L <= topk, short, rows)

    return q_nope, q_rope, kv, idx.unsqueeze(1).contiguous()


def time_interleaved(cands, rounds, iters, warmup=3):
    """Rotate through candidates once per round; report the median across rounds."""
    per = {k: [] for k in cands}
    for k, fn in cands.items():
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()
    for _ in range(rounds):
        for k, fn in cands.items():
            st = torch.cuda.Event(enable_timing=True)
            en = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            st.record()
            for _ in range(iters):
                fn()
            en.record()
            torch.cuda.synchronize()
            per[k].append(st.elapsed_time(en) / iters)
    return {k: (statistics.median(v), min(v), max(v)) for k, v in per.items()}


def rel_err(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return ((a - b).norm() / b.norm().clamp(min=1e-9)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=16384)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--pages", type=int, default=1 << 20)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()

    dev = "cuda"
    d_v, d_tail, topk = 512, 64, 2048
    sm_scale = (d_v + d_tail) ** -0.5

    q_nope, q_rope, kv, idx = make_inputs(
        args.seq, args.heads, d_v, d_tail, topk, args.pages, dev
    )
    print(f"seq={args.seq} heads={args.heads} d_v={d_v} tail={d_tail} topk={topk} "
          f"pages={args.pages}")
    print(f"q_nope {tuple(q_nope.shape)} {q_nope.dtype} | kv {tuple(kv.shape)} {kv.dtype} "
          f"| idx {tuple(idx.shape)} {idx.dtype}")

    from sglang.kernels.ops.attention.utils import concat_mla_absorb_q_general
    from sglang.kernels.ops.attention.dsa.tilelang_kernel import tilelang_sparse_fwd
    from sglang.kernels.ops.attention.dsa.triton_sparse_mla import triton_sparse_mla_fwd

    def live_tilelang():
        q_all = concat_mla_absorb_q_general(q_nope, q_rope)
        return tilelang_sparse_fwd(
            q=q_all, kv=kv, indices=idx, sm_scale=sm_scale, d_v=d_v
        )

    def triton_stock():
        return triton_sparse_mla_fwd(
            q_nope=q_nope, q_rope=q_rope, kv=kv, indices=idx,
            sm_scale=sm_scale, d_v=d_v,
        )

    cands = {"live (cat+cast+tilelang)": live_tilelang, "triton_sparse_mla (stock)": triton_stock}

    # Candidate B: build the fp8 q in one fused kernel instead of cat(bf16) -> .to(fp8).
    # The helper already exists and the decode/flashmla path already uses it; only this
    # tilelang prefill branch still pays the unfused pair.
    from sglang.kernels.ops.kvcache.cache_ops import concat_and_cast_q_fp8_pad

    heads, dim = q_nope.shape[1], d_v + d_tail

    def fused_q_fp8():
        out = q_nope.new_empty((q_nope.shape[0], heads, dim), dtype=kv.dtype)
        concat_and_cast_q_fp8_pad(out, q_nope, q_rope, heads)
        return out

    def tl_fusedq():
        return tilelang_sparse_fwd(
            q=fused_q_fp8(), kv=kv, indices=idx, sm_scale=sm_scale, d_v=d_v
        )

    cands["tilelang + fused q-prep (cand B)"] = tl_fusedq

    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from tl_dsa_blockskip import tilelang_sparse_fwd_skip
    except Exception as e:
        print(f"(no blockskip variant: {e})")
    else:
        # Both guard modes are carried here. Neither is free at decode (0.927x for the
        # any-valid reduce, 0.961x for the cheap first-slot test), so the guard is
        # dispatched prefill-only regardless -- which means the only question left is
        # which one is faster on the shape where it actually runs.
        def mk_skip(qf, mode):
            def f():
                return tilelang_sparse_fwd_skip(
                    q=qf(), kv=kv, indices=idx, sm_scale=sm_scale, d_v=d_v,
                    skip_mode=mode,
                )
            return f

        def q_bf16():
            return concat_mla_absorb_q_general(q_nope, q_rope)

        cands["tilelang + block-skip/reduce (A)"] = mk_skip(q_bf16, "reduce")
        cands["tilelang + block-skip/first  (A)"] = mk_skip(q_bf16, "first")
        cands["tilelang + A(reduce) + B"] = mk_skip(fused_q_fp8, "reduce")
        cands["tilelang + A(first)  + B"] = mk_skip(fused_q_fp8, "first")

    try:
        from sglang.kernels.ops.attention.dsa.triton_sparse_mla_h8 import (
            triton_sparse_mla_fwd_h8,
        )
    except Exception as e:
        print(f"(no h8 variant yet: {e})")
    else:
        cands["triton_sparse_mla_h8 (authored)"] = lambda: triton_sparse_mla_fwd_h8(
            q_nope=q_nope, q_rope=q_rope, kv=kv, indices=idx,
            sm_scale=sm_scale, d_v=d_v,
        )

    ref = live_tilelang().float()
    print("\ncorrectness vs live tilelang path (relative L2):")
    for k, fn in cands.items():
        if k.startswith("live"):
            continue
        print(f"  {k:34s} {rel_err(fn().float(), ref):.5f}")

    print(f"\ninterleaved timing, {args.rounds} rounds x {args.iters} iters")
    res = time_interleaved(cands, args.rounds, args.iters)
    base = res["live (cat+cast+tilelang)"][0]
    print(f"{'candidate':36s} {'median ms':>10} {'min':>9} {'max':>9} {'vs live':>9}")
    for k, (med, lo, hi) in res.items():
        print(f"{k:36s} {med:10.3f} {lo:9.3f} {hi:9.3f} {base/med:8.3f}x")


if __name__ == "__main__":
    main()
