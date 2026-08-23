#!/usr/bin/env python3
"""Graph-captured benchmark for the Kimi-K3 attention-residual aggregation pair.

`sglang/srt/layers/attn_residual.py` dispatches on hardware capability:
`_use_fast()` requires `torch.cuda.get_device_capability()[0] >= 10`, i.e.
NVIDIA SM100+, so on gfx950 every aggregation point takes the Triton
`_score_kernel` + `_combine_kernel` fallback.  My own decode profile of the
patched server (`analysis/profiles/cand_s2/kernel_table.txt`) puts that pair at
4.18% + 2.30% = 6.48% of decode GPU time -- 186 calls of each per step, i.e. two
aggregation points on each of the 93 layers.

Timing is graph-captured (`tuning-core/graph_captured_benchmarking.md`): these
kernels are ~8 us and ~5 us in the server's decode graph, close enough to eager
launch overhead that eager timing would rank candidates by launch cost.  Every
result asserts `mode == "cudagraph"`; an eager fallback is reported, never
quoted.

    python3 analysis/bench_attn_res.py                 # stock only
    python3 analysis/bench_attn_res.py --cand          # stock vs candidate
    python3 analysis/bench_attn_res.py --cand --sweep  # + config search
"""
import argparse
import itertools
import os
import statistics
import sys

import torch

sys.path.insert(0, "/sgl-workspace/sglang/python")
sys.path.insert(0, "/work/tuning_skillset/benchmark")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph_harness import cuda_graph_bench  # noqa: E402

from sglang.srt.layers.attn_residual import (  # noqa: E402
    _BLOCK_H,
    _MAX_ROWS,
    _combine_kernel,
    _score_kernel,
)

H = 7168
BLOCK_NUM = 8  # cdiv(93, attn_res_block_size=12)


def mix_stock(prefix, bank, nvb, cw, scores, out):
    T = prefix.shape[0]
    _score_kernel[(T, nvb + 1)](
        prefix, bank, cw, scores, nvb, 1e-6,
        prefix.stride(0), bank.stride(0), bank.stride(1), scores.stride(0),
        H=H, BLOCK_H=_BLOCK_H, num_warps=8,
    )
    _combine_kernel[(T, H // _BLOCK_H)](
        prefix, bank, scores, out, nvb,
        prefix.stride(0), bank.stride(0), bank.stride(1),
        scores.stride(0), out.stride(0),
        BLOCK_H=_BLOCK_H, MAX_ROWS=_MAX_ROWS, num_warps=4,
    )
    return out


def make(T, device="cuda"):
    torch.manual_seed(0)
    prefix = torch.randn(T, H, dtype=torch.bfloat16, device=device)
    bank = torch.randn(T, BLOCK_NUM, H, dtype=torch.bfloat16, device=device)
    cw = torch.randn(H, dtype=torch.float32, device=device)
    scores = torch.empty(T, _MAX_ROWS, dtype=torch.float32, device=device)
    out = torch.empty_like(prefix)
    return prefix, bank, cw, scores, out


def reference(prefix, bank, nvb, cw):
    """fp64 oracle for the mixture, straight from the kernel semantics."""
    rows = torch.cat([bank[:, :nvb, :], prefix.unsqueeze(1)], dim=1).double()
    sumsq = (rows * rows).sum(-1)
    rrms = 1.0 / torch.sqrt(sumsq / H + 1e-6)
    s = (rows * cw.double()).sum(-1) * rrms
    p = torch.softmax(s, dim=-1)
    return (p.unsqueeze(-1) * rows).sum(1)


def row_bytes(T, nvb):
    """Row traffic only: score reads every row once, combine reads them again."""
    return 2 * T * (nvb + 1) * H * 2


def timed(step, out, ref, tol, reps=1):
    """Time `reps` back-to-back invocations inside one graph, report per-call us.

    At the decode shape a single call is ~10 us and a two-kernel graph replay
    costs about that much on its own, so `reps=1` measures the harness: the
    stock pair came out 17.9 us at nvb=1 and 20.2 us at nvb=7, i.e. flat against
    an 8x change in bytes.  In the server these kernels sit inside a decode
    graph of several hundred, where that fixed cost is pipelined away, so
    replaying a run of them is the closer analogue."""
    def multi():
        for _ in range(reps):
            step()

    r = cuda_graph_bench(
        multi, warmup=15, iters=40,
        dirty=lambda: out.zero_(),
        verify=lambda: torch.allclose(out.double(), ref, atol=tol, rtol=tol),
    )
    return r, statistics.median(r["times_ms"]) * 1e3 / reps  # us per call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--tokens", type=int, nargs="*", default=[64, 2048, 16384])
    ap.add_argument("--nvb", type=int, nargs="*", default=[1, 4, 7])
    ap.add_argument("--reps", type=int, default=32,
                    help="invocations captured per graph; amortizes replay cost")
    args = ap.parse_args()

    cand_mod = None
    if args.cand:
        import attn_res_cand as cand_mod

    props = torch.cuda.get_device_properties(0)
    print(f"device {props.name}  CU={props.multi_processor_count}  "
          f"capability={torch.cuda.get_device_capability()}  "
          f"-> _use_fast "
          f"{'ON' if torch.cuda.get_device_capability()[0] >= 10 else 'OFF (triton fallback)'}")
    print()

    hdr = f"{'T':>6} {'nvb':>4} {'MB':>7} {'stock us':>9} {'GB/s':>7}"
    if cand_mod:
        hdr += f" {'cand us':>9} {'GB/s':>7} {'speedup':>8} {'max|err|':>9}"
    print(hdr)

    tot_s = tot_c = 0.0
    for T, nvb in itertools.product(args.tokens, args.nvb):
        prefix, bank, cw, scores, out = make(T)
        ref = reference(prefix, bank, nvb, cw)
        mb = row_bytes(T, nvb) / 1e6
        tol = 3e-2

        rs, us_s = timed(lambda: mix_stock(prefix, bank, nvb, cw, scores, out),
                         out, ref, tol, args.reps)
        assert rs["mode"] == "cudagraph", f"stock: {rs['mode']}"
        line = f"{T:>6} {nvb:>4} {mb:>7.1f} {us_s:>9.2f} {mb/us_s*1e3:>7.0f}"
        tot_s += us_s

        if cand_mod:
            out_c = torch.empty_like(prefix)
            sc_c = torch.empty(T, _MAX_ROWS, dtype=torch.float32, device="cuda")
            rc, us_c = timed(
                lambda: cand_mod.mix_cand(prefix, bank, nvb, cw, sc_c, out_c),
                out_c, ref, tol, args.reps)
            assert rc["mode"] == "cudagraph", f"cand: {rc['mode']}"
            err = (out_c.double() - ref).abs().max().item()
            line += (f" {us_c:>9.2f} {mb/us_c*1e3:>7.0f} {us_s/us_c:>7.3f}x"
                     f" {err:>9.2e}")
            tot_c += us_c
        print(line)

    if cand_mod:
        print(f"\ntotal stock {tot_s:.1f} us   cand {tot_c:.1f} us"
              f"   -> {tot_s/tot_c:.3f}x")

    if args.sweep and cand_mod:
        # The two kernels are independent launches, so sweep them separately:
        # 8 + 8 configs instead of the 64 a joint product would cost.  BLOCK_H
        # must divide H = 7168 = 2^10 * 7, so 2048 is not a legal tile.
        for T in args.tokens:
            print(f"\n--- config sweep, T={T} (sum over nvb {args.nvb}) ---")
            prefix, bank, cw, scores, out = make(T)
            refs = {n: reference(prefix, bank, n, cw) for n in args.nvb}
            sc_c = torch.empty(T, _MAX_ROWS, dtype=torch.float32, device="cuda")
            out_c = torch.empty_like(prefix)

            def total_for(cfg, which):
                tot = 0.0
                for n in args.nvb:
                    # Time one kernel; the other still has to run so the guard
                    # has a correct output to verify against.
                    def step(n=n):
                        cand_mod.mix_cand(prefix, bank, n, cw, sc_c, out_c, cfg)
                    r, us = timed(step, out_c, refs[n], 3e-2, args.reps)
                    if r["mode"] != "cudagraph":
                        return None
                    tot += us
                return tot

            base = dict(cand_mod.CFG)
            for which, keys, grid in (
                ("score", ("score_block_h", "score_warps"),
                 list(itertools.product([256, 512, 1024], [1, 2, 4, 8]))),
                ("combine", ("combine_block_h", "combine_warps"),
                 list(itertools.product([256, 512, 1024], [1, 2, 4, 8]))),
            ):
                best = None
                for vals in grid:
                    cfg = dict(base)
                    cfg.update(dict(zip(keys, vals)))
                    try:
                        tot = total_for(cfg, which)
                    except Exception as e:
                        print(f"  {which} {dict(zip(keys, vals))} -> "
                              f"{type(e).__name__}: {str(e)[:70]}")
                        continue
                    if tot is None:
                        continue
                    mark = ""
                    if best is None or tot < best[0]:
                        best = (tot, dict(zip(keys, vals)))
                        mark = "  <-"
                    print(f"  {which:<8} {dict(zip(keys, vals))}  {tot:>8.2f} us{mark}")
                if best:
                    base.update(best[1])
                    print(f"  best {which}: {best[1]}  {best[0]:.2f} us")
            print(f"  tuned CFG for T={T}: {base}")


if __name__ == "__main__":
    main()
