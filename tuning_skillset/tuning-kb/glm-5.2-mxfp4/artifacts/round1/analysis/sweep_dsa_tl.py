#!/usr/bin/env python3
"""Sweep the DSA sparse-MLA fp8 prefill kernel's launch geometry at this workload's shape.

`tilelang_sparse_fwd` hardcodes `block_I, threads, block_per_cu = 64, 256, 2` for every
gfx95 fp8 call, and `sparse_mla_fwd_decode_partial_fp8` hardcodes `h_per_block = 16`.
None of those were chosen for tp_q_head_num=8: at 8 heads the 16-row head tile is half
padding, and the fp32 output accumulators (4 x [h_per_block, 128] per block) are the
kernel's dominant register cost, so h_per_block drives occupancy and therefore how well
the indexed KV gather is latency-hidden.

Everything here is timed INTERLEAVED against the live configuration in the same process
(tuning-core Rule 6b) rather than as a table of independent absolute times, because on
gfx950 sequential timings drift far more than the differences being resolved.

    HIP_VISIBLE_DEVICES=0 SGLANG_USE_AITER=1 python3 sweep_dsa_tl.py
"""
import argparse
import itertools
import os
import statistics
import sys

import torch

sys.path.insert(0, "/sgl-workspace/sglang/python")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_dsa_prefill import make_inputs, rel_err, time_interleaved  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=16384)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--pages", type=int, default=1 << 20)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--block-i", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--threads", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--hpb", type=int, nargs="+", default=[8, 16])
    args = ap.parse_args()

    dev = "cuda"
    d_v, d_tail, topk = 512, 64, 2048
    sm_scale = (d_v + d_tail) ** -0.5
    q_nope, q_rope, kv, idx = make_inputs(
        args.seq, args.heads, d_v, d_tail, topk, args.pages, dev
    )

    from sglang.kernels.ops.attention.dsa.tilelang_kernel import tilelang_sparse_fwd
    from sglang.kernels.ops.attention.utils import concat_mla_absorb_q_general

    from tl_dsa_blockskip import tilelang_sparse_fwd_skip

    def live():
        q_all = concat_mla_absorb_q_general(q_nope, q_rope)
        return tilelang_sparse_fwd(q=q_all, kv=kv, indices=idx, sm_scale=sm_scale, d_v=d_v)

    ref = live().float()
    cands = {"live (stock bI=64 thr=256 hpb=16)": live}
    errs = {}

    # Compile everything up front so a compile failure is reported as a skipped config
    # rather than poisoning the interleaved timing loop.
    for bi, thr, hpb in itertools.product(args.block_i, args.threads, args.hpb):
        name = f"skip bI={bi:<3d} thr={thr:<3d} hpb={hpb}"

        def mk(bi=bi, thr=thr, hpb=hpb):
            def f():
                q_all = concat_mla_absorb_q_general(q_nope, q_rope)
                return tilelang_sparse_fwd_skip(
                    q=q_all, kv=kv, indices=idx, sm_scale=sm_scale, d_v=d_v,
                    block_I=bi, threads=thr, h_per_block=hpb,
                )
            return f

        f = mk()
        try:
            out = f()
            torch.cuda.synchronize()
        except Exception as e:
            print(f"  SKIP {name}: {type(e).__name__}: {str(e)[:110]}")
            continue
        e = rel_err(out.float(), ref)
        if e > 5e-3:
            print(f"  WRONG {name}: rel-L2 {e:.4g}")
            continue
        errs[name] = e
        cands[name] = f

    print(f"\n{len(cands)-1} configs compiled and numerically clean")
    print(f"interleaved timing, {args.rounds} rounds x {args.iters} iters\n")
    res = time_interleaved(cands, args.rounds, args.iters)
    base = res["live (stock bI=64 thr=256 hpb=16)"][0]
    print(f"{'config':36s} {'median ms':>10} {'min':>9} {'max':>9} {'vs live':>9} {'relL2':>9}")
    for k, (med, lo, hi) in sorted(res.items(), key=lambda kv: kv[1][0]):
        print(f"{k:36s} {med:10.3f} {lo:9.3f} {hi:9.3f} {base/med:8.3f}x "
              f"{errs.get(k, 0.0):9.2g}")


if __name__ == "__main__":
    main()
