#!/usr/bin/env python3
"""Race the FlyDSL mxfp4 MoE kernels against the heuristic pick, on this model's shape.

    python3 /work/analysis/tune_flydsl_moe.py [token] [--full]

Needs one GPU and the server DOWN.

`analysis/bench_moe.py` established that the fmoe CSV knobs (block_m, ksplit,
use_nt) do nothing here: this model's MoE does not go through the CK 2-stage path
at all.  `fused_moe` routes a4w4 mxfp4 to FlyDSL and, having no tuned row, logs

    [fused_moe] no tuned FlyDSL config for (...), using heuristic FlyDSL fallback
      (kn1='flydsl_moe1_afp4_wfp4_bf16_t32x128x256_w2',
       kn2='flydsl_moe2_afp4_wfp4_bf16_t32x128x256_atomic_bnt2')

That pair comes from a token-count if-ladder, not from a measurement, and there are
272 stage-1 and 257 stage-2 compiled variants to choose from.  moe1+moe2 are 33.1%
of decode, so this is the largest untuned surface left in the stack.

Method: inject a synthetic tuned row into `fused_moe.cfg_2stages` (the same dict a
tuned CSV would populate) so the real dispatch path runs the candidate kernel, then
time the whole `fused_moe` call.  Two phases -- stage 1 with the heuristic stage 2
held fixed, then stage 2 against the best stage 1 -- because the two kernels run
sequentially and do not interact beyond the shared block_m.

Every survivor is re-timed interleaved (tuning-core/measurement.md Rule 6b): the
coarse pass exists only to rank, never to accept.
"""
import re
import statistics
import sys

import torch

import aiter.fused_moe as fm
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.ops.flydsl.moe_kernels import _KERNEL_PARAMS

sys.path.insert(0, "/work/analysis")
from bench_moe import INTER_DIM, MODEL_DIM, TOPK, build, call  # noqa: E402

COARSE_ITERS = 20
COARSE_WARMUP = 3
FINE_REPS = 7
FINE_ITERS = 30
TOPN = 6            # survivors carried from the coarse pass into the interleaved pass


def keys_for(token):
    return (
        get_gfx_runtime(), get_cu_num(), token, MODEL_DIM, INTER_DIM, 64, TOPK,
        "ActivationType.Silu", "torch.bfloat16", "torch.float4_e2m1fn_x2",
        "torch.float4_e2m1fn_x2", "QuantType.per_1x32", True, False,
    )


def set_cfg(keys, kn1, kn2, block_m, ksplit=-1):
    """Plant the row a tuned CSV would have produced, and drop the memoized dispatch."""
    cfg = dict(kernelName1=kn1, kernelName2=kn2, block_m=block_m, ksplit=ksplit,
               run_1stage=False)
    fm.cfg_2stages = ({keys: cfg}, {})
    fm.get_2stage_cfgs.cache_clear()


def clear_cfg():
    fm.cfg_2stages = ({}, {})
    fm.get_2stage_cfgs.cache_clear()


def tile_m_of(name):
    m = re.search(r"_t(\d+)x\d+x\d+", name)
    return int(m.group(1)) if m else None


def timeit(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000.0 / iters


def probe(b, ref, keys, kn1, kn2, bm):
    """One coarse timing + a correctness check. Returns (us, err) or None if unusable."""
    try:
        set_cfg(keys, kn1, kn2, bm)
        out = call(b)
        torch.cuda.synchronize()
        err = (out.float() - ref.float()).abs().max().item() / (
            ref.float().abs().max().item() + 1e-6)
        if not (err < 5e-2):
            return None, err
        return timeit(lambda: call(b), COARSE_ITERS, COARSE_WARMUP), err
    except Exception:
        return None, None


def fine(b, keys, cands, label):
    """Interleaved medians over the survivors -- the only numbers that get believed."""
    times = {c: [] for c in cands}
    for _ in range(FINE_REPS):
        for c in cands:
            kn1, kn2, bm = c
            set_cfg(keys, kn1, kn2, bm)
            times[c].append(timeit(lambda: call(b), FINE_ITERS, 2))
    rows = sorted((statistics.median(v), k, min(v), max(v)) for k, v in times.items())
    print(f"\n# {label}: interleaved, {FINE_REPS} rounds x {FINE_ITERS} iters")
    for med, k, lo, hi in rows:
        print(f"  {med:9.2f} us  spread {(hi - lo) / med * 100:5.2f}%  "
              f"bm={k[2]:>3}  {k[0]}  |  {k[1]}")
    return rows


def main():
    token = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    full = "--full" in sys.argv
    torch.cuda.set_device(0)

    b = build(token)
    keys = keys_for(token)

    # the heuristic pair, taken from the live dispatch rather than re-derived
    clear_cfg()
    ref = call(b).clone()
    md = fm.get_2stage_cfgs.cache_info()
    del md
    d_kn1, d_kn2 = _heuristic_pair(token)
    d_bm = tile_m_of(d_kn1)
    print(f"# token={token} touched={b['touched']}/64 pairs={b['npairs']}")
    print(f"# heuristic: {d_kn1}  |  {d_kn2}  (block_m={d_bm})")

    s1_all = sorted(n for n in _KERNEL_PARAMS if "moe1_afp4_wfp4_bf16" in n)
    s2_all = sorted(n for n in _KERNEL_PARAMS if "moe2_afp4_wfp4_bf16" in n)
    if not full:
        # tile_m other than the heuristic's is still raced, but only the x128 tile_n
        # family, which is the one the heuristic ladder ever selects
        s1_all = [n for n in s1_all if re.search(r"_t\d+x128x256", n)]
        s2_all = [n for n in s2_all if re.search(r"_t\d+x128x(128|256)", n)]
    print(f"# racing {len(s1_all)} stage-1 and {len(s2_all)} stage-2 variants")

    # ---- phase A: stage 1, heuristic stage 2 held fixed (matched on tile_m) ----
    a = []
    for i, kn1 in enumerate(s1_all):
        bm = tile_m_of(kn1)
        kn2 = _match_stage2(bm, s2_all, d_kn2)
        if kn2 is None:
            continue
        us, err = probe(b, ref, keys, kn1, kn2, bm)
        if us is None:
            print(f"#   [{i+1}/{len(s1_all)}] skip {kn1} err={err}")
            continue
        a.append((us, kn1, kn2, bm))
        print(f"#   [{i+1}/{len(s1_all)}] {us:8.2f} us  {kn1}", flush=True)
    a.sort()
    if not a:
        print("no viable stage-1 candidate"); return
    survivors = [(kn1, kn2, bm) for _, kn1, kn2, bm in a[:TOPN]]
    base = (d_kn1, _match_stage2(d_bm, s2_all, d_kn2), d_bm)
    if base not in survivors:
        survivors.append(base)
    rows1 = fine(b, keys, survivors, "stage 1")
    best_kn1, best_kn2_a, best_bm = rows1[0][1]

    # ---- phase B: stage 2, best stage 1 held fixed ----
    c = []
    for i, kn2 in enumerate(s2_all):
        if tile_m_of(kn2) != best_bm:
            continue
        us, err = probe(b, ref, keys, best_kn1, kn2, best_bm)
        if us is None:
            continue
        c.append((us, kn2))
        print(f"#   s2 {us:8.2f} us  {kn2}", flush=True)
    c.sort()
    survivors2 = [(best_kn1, kn2, best_bm) for _, kn2 in c[:TOPN]]
    if (best_kn1, best_kn2_a, best_bm) not in survivors2:
        survivors2.append((best_kn1, best_kn2_a, best_bm))
    if base not in survivors2:
        survivors2.append(base)
    rows2 = fine(b, keys, survivors2, "stage 2")

    # ---- verdict against the heuristic, measured in the same interleaved pass ----
    best = rows2[0]
    dflt = next((m for m, k, *_ in rows2 if k == base), None)
    print(f"\n# best   {best[0]:.2f} us  {best[1][0]}  |  {best[1][1]}")
    if dflt:
        print(f"# default {dflt:.2f} us  -> {dflt / best[0]:.4f}x")


def _heuristic_pair(token):
    """Reproduce fused_moe.py's FlyDSL if-ladder (the fallback it logs)."""
    from aiter.ops.flydsl.moe_kernels import (
        flydsl_kernel_name,
        get_flydsl_kernel_params,
        pick_flydsl_stage2_tile_k,
    )
    s2tk = pick_flydsl_stage2_tile_k(INTER_DIM)
    if token < 2048:
        tm, s1s, s2s = 32, "_w2", "_bnt2"
    elif token < 4096:
        tm, s1s, s2s = 64, "_w3_bnt0", ""
    elif token < 16384:
        tm, s1s, s2s = 128, "_w2_bnt0", ""
    else:
        tm, s1s, s2s = 64, "_w4_bnt0", ""
    b1 = flydsl_kernel_name(1, "fp4", "fp4", "bf16", tm, 128, 256)
    b2 = flydsl_kernel_name(2, "fp4", "fp4", "bf16", tm, 128, s2tk, "atomic")
    kn1, kn2 = f"{b1}{s1s}", f"{b2}{s2s}"
    if get_flydsl_kernel_params(kn1) is None:
        kn1 = b1
    if get_flydsl_kernel_params(kn2) is None:
        kn2 = b2
    return kn1, kn2


def _match_stage2(bm, s2_all, d_kn2):
    """The heuristic stage-2 kernel retargeted to tile_m `bm`, if it exists."""
    want = re.sub(r"_t\d+x", f"_t{bm}x", d_kn2)
    if want in s2_all:
        return want
    for n in s2_all:
        if tile_m_of(n) == bm and n.endswith("_atomic"):
            return n
    return None


if __name__ == "__main__":
    main()
