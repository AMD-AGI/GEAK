#!/usr/bin/env python3
"""Sweep the split-K factor of the DSA sparse-MLA kernel at the DECODE shape.

The pure-decode profile (analysis/profile_decode_tp0.json) puts `main_kernel` at 13.41% of
decode GPU time, moving 64 x 2048 x 576 B = 75.5 MB in ~34 us = 2.2 TB/s, which is ~28% of
what this part can do. The reason is visible in the launch geometry rather than the kernel
body. `tilelang_sparse_fwd` picks the split via:

    _pick_inner_iter(seq, ni, cu=256, block_per_cu=2)
        max_it = seq * ni / (cu * block_per_cu)
        -> largest power-of-two divisor of ni that is <= max_it

At decode, seq=64 and ni = topk/block_I = 32, so max_it = 64*32/512 = 4 and inner_iter=4,
giving n_groups=8 and a grid of 64*8 = 512 blocks against 256 CUs x 2 = 512 slots: exactly
one wave, by construction. The docstring says this "avoids under-utilization while
minimizing the number of partial groups", and one wave does technically occupy every CU --
but a single wave has nothing left to schedule while a block stalls on its indexed gather,
so the memory system never gets enough requests in flight. Smaller inner_iter means more
groups, more blocks, more waves, and more overlap, paid for with a wider combine pass.

This sweep drives inner_iter over every power-of-two divisor of ni and times the full
partial+combine pair, interleaved (Rule 6b), against the stock end-to-end call.

    HIP_VISIBLE_DEVICES=0 SGLANG_USE_AITER=1 python3 sweep_dsa_decode.py
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, "/sgl-workspace/sglang/python")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_dsa_prefill import rel_err, time_interleaved  # noqa: E402


def make_decode_inputs(bs, heads, d_v, d_tail, topk, ctx, n_slots, device, page=64, seed=0):
    """Decode step: one query per sequence, each attending to topk keys of its own context.

    At OSL 1024 on ISL 8192 every sequence is far longer than index_topk=2048, so the
    indexer's radix path fills all topk slots and there is no -1 padding here -- unlike
    prefill. Slots are scattered because the pages are.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    dim = d_v + d_tail
    q_nope = torch.randn(bs, heads, d_v, generator=g, device=device, dtype=torch.bfloat16) * 0.1
    q_rope = torch.randn(bs, heads, d_tail, generator=g, device=device, dtype=torch.bfloat16) * 0.1
    kv = (torch.randn(n_slots, 1, dim, generator=g, device=device, dtype=torch.bfloat16) * 0.1).to(
        torch.float8_e4m3fn
    )
    idx = torch.empty(bs, topk, dtype=torch.int32, device=device)
    npage = ctx // page
    for b in range(bs):
        pages = torch.randperm(n_slots // page, generator=g, device=device)[:npage]
        slot = (pages.to(torch.int64) * page).repeat_interleave(page) + (
            torch.arange(ctx, device=device) % page
        )
        pick = torch.randperm(ctx, generator=g, device=device)[:topk]
        idx[b] = slot[pick].to(torch.int32)
    return q_nope, q_rope, kv, idx.unsqueeze(1).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=9216)
    ap.add_argument("--pages", type=int, default=1 << 20)
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--block-i", type=int, default=64)
    args = ap.parse_args()

    dev = "cuda"
    d_v, d_tail, topk = 512, 64, 2048
    sm_scale = (d_v + d_tail) ** -0.5
    q_nope, q_rope, kv, idx = make_decode_inputs(
        args.bs, args.heads, d_v, d_tail, topk, args.ctx, args.pages, dev
    )
    print(f"decode shape: bs={args.bs} heads={args.heads} topk={topk} ctx={args.ctx}")
    print(f"gather per call: {args.bs * topk * (d_v + d_tail) / 1e6:.1f} MB")

    from sglang.kernels.ops.attention.dsa.tilelang_kernel import (
        _pick_inner_iter,
        sparse_mla_fwd_decode_combine,
        sparse_mla_fwd_decode_partial_fp8,
        tilelang_sparse_fwd,
    )
    from sglang.kernels.ops.attention.utils import concat_mla_absorb_q_general

    BI = args.block_i
    ni = topk // BI
    stock_ii = _pick_inner_iter(args.bs, ni, 256, 2)
    print(f"block_I={BI} ni={ni}  stock inner_iter={stock_ii} "
          f"-> n_groups={ni // stock_ii}, grid={args.bs * (ni // stock_ii)} blocks\n")

    q_all_fp8 = concat_mla_absorb_q_general(q_nope, q_rope).to(kv.dtype)

    def live():
        q_all = concat_mla_absorb_q_general(q_nope, q_rope)
        return tilelang_sparse_fwd(q=q_all, kv=kv, indices=idx, sm_scale=sm_scale, d_v=d_v)

    def mk(ii):
        part = sparse_mla_fwd_decode_partial_fp8(
            args.heads, d_v, d_tail, topk, sm_scale=sm_scale,
            block_I=BI, inner_iter=ii, threads=256,
        )
        ng = ni // ii
        comb = sparse_mla_fwd_decode_combine(
            args.heads, d_v, ng * BI, head_per_block=4, block_I=BI, threads=256,
        )

        def f():
            po, pl = part(q_all_fp8.unsqueeze(0), kv.unsqueeze(0), idx.unsqueeze(0))
            return comb(po, pl)

        return f

    # Reference is the STOCK split with q already fp8, not `live()`. live() re-runs the
    # concat and the bf16->fp8 cast on every call as eager ops; at this shape the pair
    # costs more in host dispatch than the attention costs on device, which would show up
    # as a fake 1.3x for every candidate. In the server this path is inside a HIP graph.
    ref = mk(stock_ii)().float()

    def graphed(f, iters):
        """Capture `iters` back-to-back calls into a HIP graph.

        Without this the harness measures host dispatch, not the kernel: a decode call is
        ~38 us of device work behind two eager launches, so the CPU, not the GPU, sets the
        pace and every candidate collapses to the same number.
        """
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(5):
                f()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(iters):
                f()
        return g

    # The block-skip kernel (candidate A) is a prefill optimisation, but it is the same
    # factory that serves decode, so it has to be shown not to cost anything here. At
    # decode every sequence is far longer than topk, so no index block is ever all-padding
    # and the guard never fires -- `skip_empty=True` below measures the price of asking,
    # and `skip_empty=False` checks that the parse-time switch really does hand decode
    # back the stock kernel rather than a lightly-penalised one.
    from tl_dsa_blockskip import sparse_mla_fwd_decode_partial_fp8_skip

    def mk_skip(ii, skip_empty=True, skip_mode="reduce"):
        part = sparse_mla_fwd_decode_partial_fp8_skip(
            args.heads, d_v, d_tail, topk, sm_scale=sm_scale,
            block_I=BI, inner_iter=ii, threads=256, skip_empty=skip_empty,
            skip_mode=skip_mode,
        )
        ng = ni // ii
        comb = sparse_mla_fwd_decode_combine(
            args.heads, d_v, ng * BI, head_per_block=4, block_I=BI, threads=256,
        )

        def f():
            po, pl = part(q_all_fp8.unsqueeze(0), kv.unsqueeze(0), idx.unsqueeze(0))
            return comb(po, pl)

        return f

    cands, errs = {}, {}
    iis = [i for i in (32, 16, 8, 4, 2, 1) if ni % i == 0]
    for ii in iis:
        ng = ni // ii
        tag = "  <-- stock" if ii == stock_ii else ""
        name = f"inner_iter={ii:<2d} n_groups={ng:<2d} grid={args.bs*ng:<5d}{tag}"
        try:
            f = mk(ii)
            out = f()
            torch.cuda.synchronize()
            g = graphed(f, args.iters)
        except Exception as e:
            print(f"  SKIP {name}: {type(e).__name__}: {str(e)[:100]}")
            continue
        # Split-K changes where the online-softmax max is taken, which changes the fp8
        # quantisation of the probabilities, so a shift here is expected rather than a bug
        # -- it is reported and then gated on gsm8k, not thresholded away.
        errs[name] = rel_err(out.float(), ref)
        cands[name] = g.replay

    for se, sm, label in ((True, "reduce", "guard ON  reduce (any-valid AllReduce)"),
                          (True, "first", "guard ON  first  (suffix invariant)"),
                          (False, "reduce", "guard OFF        (== stock source)")):
        sname = f"block-skip src, {label}"
        sf = mk_skip(stock_ii, skip_empty=se, skip_mode=sm)
        sout = sf()
        torch.cuda.synchronize()
        errs[sname] = rel_err(sout.float(), ref)
        cands[sname] = graphed(sf, args.iters).replay

    print(f"\ninterleaved HIP-graph timing, {args.rounds} rounds x {args.iters} calls/graph\n")
    res = time_interleaved(cands, args.rounds, 1, warmup=3)
    base = [v[0] for k, v in res.items() if "<-- stock" in k][0]
    gb = args.bs * topk * (d_v + d_tail) / 1e9
    print(f"{'config':46s} {'med us':>9} {'min':>8} {'max':>8} {'vs stock':>9} {'TB/s':>7} {'relL2':>9}")
    for k, (med, lo, hi) in sorted(res.items(), key=lambda kv: kv[1][0]):
        per = med / args.iters
        print(f"{k:46s} {per*1000:9.2f} {lo*1000/args.iters:8.2f} {hi*1000/args.iters:8.2f} "
              f"{base/med:8.3f}x {gb/(per/1000):7.2f} {errs.get(k, 0.0):9.2g}")


if __name__ == "__main__":
    main()
