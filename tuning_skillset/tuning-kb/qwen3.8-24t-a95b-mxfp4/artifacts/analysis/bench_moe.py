#!/usr/bin/env python3
"""Price the MoE 2-stage config knobs on this model's real shape.

    python3 /work/analysis/bench_moe.py            # decode tier
    python3 /work/analysis/bench_moe.py 8192       # a prefill tier

Needs one GPU and the server DOWN (it allocates ~6 GB of weights).

Why this exists: the live server logs, at every token tier,

    [fused_moe] using 2stage default for ('gfx950', 256, 64, 8192, 2048, 64, 9,
        'ActivationType.Silu', 'torch.bfloat16', 'torch.float4_e2m1fn_x2',
        'torch.float4_e2m1fn_x2', 'QuantType.per_1x32', True, False)

i.e. aiter has no tuned fmoe row for this model and falls back to
`get_block_size_M` / `get_ksplit` / `use_nt` heuristics.  moe1+moe2 are 33.1% of
decode, so if the heuristics are leaving anything on the table it is the largest
single lever left.  This script asks that question directly: hold everything else
fixed, override the three heuristic outputs, and see whether any other setting is
faster than the one the heuristic picked.

It also reports achieved HBM bandwidth against the weight bytes the routing
actually touches, which is what decides whether tuning can help at all.

Measurement follows tuning-core/measurement.md Rule 6b: candidates are interleaved
round by round rather than run back to back, and compared on medians.
"""
import os
import statistics
import sys

import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, fused_topk
from aiter.ops.shuffle import shuffle_weight
from aiter.utility import fp4_utils

# The model, at TP=8 / EP=8: 512 routed experts -> 64 local, hidden 8192,
# inter 2048, 9 routed slots + 1 always-masked fake EP slot.
MODEL_DIM = 8192
INTER_DIM = 2048
E_GLOBAL = 512
EP = 8
TOPK = 9

REPS = 7
ITERS = 30
WARMUP = 5


def build(token):
    ep_id = EP - 1
    local_E = E_GLOBAL // EP

    # one always-masked fake-expert slot appended to topk_ids, which is what makes
    # fused_moe treat this as EP and strip 1 from topk when building its config key
    expert_mask = torch.zeros((E_GLOBAL + 1,), dtype=dtypes.i32, device="cuda")
    expert_mask[ep_id * local_E : (ep_id + 1) * local_E] = 1
    fake_expertid = E_GLOBAL

    dtype = dtypes.bf16
    inp = torch.randn((token, MODEL_DIM), dtype=dtype, device="cuda") / 10
    score = torch.randn((token, E_GLOBAL), dtype=dtype, device="cuda")

    topk_ids = torch.empty((token, TOPK + 1), dtype=dtypes.i32, device="cuda")
    topk_weights = torch.empty((token, TOPK + 1), dtype=dtypes.fp32, device="cuda")
    ns_ids, s_ids = topk_ids.split([TOPK, 1], dim=1)
    ns_w, s_w = topk_weights.split([TOPK, 1], dim=1)
    s_ids[:] = fake_expertid
    s_w[:] = 0.0
    fused_topk(inp, score, TOPK, True, ns_ids, ns_w)

    w1 = torch.randn((local_E, INTER_DIM * 2, MODEL_DIM), dtype=dtype, device="cuda") / 10
    w2 = torch.randn((local_E, MODEL_DIM, INTER_DIM), dtype=dtype, device="cuda") / 10

    torch_quant = aiter.get_torch_quant(QuantType.per_1x32)
    w1_qt, w1_scale = torch_quant(w1, quant_dtype=dtypes.fp4x2)
    w1_qt = w1_qt.view(local_E, INTER_DIM * 2, MODEL_DIM // 2)
    w2_qt, w2_scale = torch_quant(w2, quant_dtype=dtypes.fp4x2)
    w2_qt = w2_qt.view(local_E, MODEL_DIM, INTER_DIM // 2)
    del w1, w2
    torch.cuda.empty_cache()

    # a4w4 mxfp4 layout: (16,16) weight shuffle + e8m0 scale shuffle, gate/up separated
    w1_a = shuffle_weight(w1_qt, layout=(16, 16))
    w2_a = shuffle_weight(w2_qt, layout=(16, 16))
    w1_s = fp4_utils.e8m0_shuffle(w1_scale)
    w2_s = fp4_utils.e8m0_shuffle(w2_scale)
    w1_a.is_shuffled = True
    w2_a.is_shuffled = True
    del w1_qt, w2_qt
    torch.cuda.empty_cache()

    # how many of the 64 local experts this routing actually touches -- the
    # denominator for the bandwidth number below
    local = topk_ids[:, :TOPK]
    lo, hi = ep_id * local_E, (ep_id + 1) * local_E
    hits = local[(local >= lo) & (local < hi)]
    touched = int(torch.unique(hits).numel())

    return dict(
        inp=inp, w1=w1_a, w2=w2_a, w1s=w1_s, w2s=w2_s,
        topk_weights=topk_weights, topk_ids=topk_ids, expert_mask=expert_mask,
        touched=touched, local_E=local_E, npairs=int(hits.numel()),
    )


def call(b):
    return fused_moe(
        b["inp"], b["w1"], b["w2"], b["topk_weights"], b["topk_ids"],
        expert_mask=b["expert_mask"],
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        w1_scale=b["w1s"], w2_scale=b["w2s"],
    )


def timeit(fn, iters=ITERS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters      # us


def main():
    token = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    torch.cuda.set_device(0)
    import aiter.fused_moe as fm

    b = build(token)

    # what the heuristics pick, before anything is overridden
    d_bm = fm.get_block_size_M(token, TOPK, b["local_E"], INTER_DIM)
    d_ks = fm.get_ksplit(token, TOPK, b["local_E"], INTER_DIM, MODEL_DIM)
    d_nt = fm.use_nt(token, TOPK, b["local_E"])
    print(f"# token={token}  local_experts={b['local_E']}  touched={b['touched']}  "
          f"pairs={b['npairs']}")
    print(f"# heuristic default: block_m={d_bm} ksplit={d_ks} use_nt={d_nt}")

    ref = call(b).clone()

    # weight bytes the routing actually reads: fp4 payload + e8m0 scale (1 per 32)
    per_expert = (2 * INTER_DIM * MODEL_DIM + MODEL_DIM * INTER_DIM) * (0.5 + 1 / 32)
    wbytes = per_expert * b["touched"]

    cands = []
    for bm in (16, 32, 64, 128):
        for ks in (0, 2, 4):
            for nt in (True, False):
                cands.append((bm, ks, nt))

    def make(bm, ks, nt):
        def run():
            fm.get_block_size_M = lambda *a, **k: bm
            fm.get_ksplit = lambda *a, **k: ks
            fm.use_nt = lambda *a, **k: nt
            return call(b)
        return run

    # one cheap correctness+viability probe per candidate before spending time on it
    live = []
    for bm, ks, nt in cands:
        fn = make(bm, ks, nt)
        try:
            out = fn()
            torch.cuda.synchronize()
            err = (out.float() - ref.float()).abs().max().item() / (
                ref.float().abs().max().item() + 1e-6)
            if err > 5e-2:
                print(f"#   skip block_m={bm} ksplit={ks} nt={nt}: err={err:.2e}")
                continue
            live.append(((bm, ks, nt), fn, err))
        except Exception as e:
            print(f"#   skip block_m={bm} ksplit={ks} nt={nt}: {type(e).__name__} "
                  f"{str(e)[:90]}")
    print(f"# {len(live)}/{len(cands)} candidates viable")

    times = {k: [] for k, _, _ in live}
    for r in range(REPS):                       # interleaved -- drift hits every arm
        for k, fn, _ in live:
            times[k].append(timeit(fn))

    rows = []
    for (k, _, err) in live:
        med = statistics.median(times[k])
        rows.append((med, k, err, min(times[k]), max(times[k])))
    rows.sort()

    print(f"\n# {'block_m':>7} {'ksplit':>6} {'nt':>5} {'us':>9} {'spread':>7} "
          f"{'TB/s':>7} {'vs default':>11}  err")
    dflt = next((m for m, k, *_ in rows if k == (d_bm, d_ks, d_nt)), None)
    for med, k, err, lo, hi in rows:
        tbs = wbytes / (med * 1e-6) / 1e12
        rel = f"{dflt / med:.3f}x" if dflt else "-"
        tag = "  <- default" if k == (d_bm, d_ks, d_nt) else ""
        print(f"  {k[0]:>7} {k[1]:>6} {str(k[2]):>5} {med:9.2f} "
              f"{(hi - lo) / med * 100:6.2f}% {tbs:7.2f} {rel:>11}  {err:.1e}{tag}")

    if dflt:
        best = rows[0]
        print(f"\n# best {best[1]} = {best[0]:.2f} us vs default {dflt:.2f} us "
              f"= {dflt / best[0]:.4f}x")


if __name__ == "__main__":
    main()
