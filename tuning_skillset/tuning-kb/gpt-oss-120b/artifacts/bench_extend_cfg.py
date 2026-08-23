#!/usr/bin/env python3
"""Tune the launch config of SGLang's Triton extend (prefill) kernel for
gpt-oss-120b's head_dim 64 on gfx950.

`_get_block_sizes_for_extend_attention` (extend_attention.py:38) has exactly one
gfx950 tuning, and it is for 128 < Lq <= 256:

    BLOCK_M, BLOCK_N = (128, 64); num_warps = 8
    # "a larger query tile halves KV bytes streamed per call ... Measured on
    #  MI350X head_dim 256: -36% kernel time, 28% -> 44% MFU"

head_dim 64 -- what gpt-oss-120b uses -- falls through to the generic AMD default
(64, 64) / 4 warps. The same argument applies here and the measured traffic says
so: at BLOCK_M=64 each of the 128 M-blocks re-reads its whole KV prefix, giving
8.66 GB of K/V reads per call against only 8.4 MB of unique KV, and the kernel
runs at 3.28 TB/s -- i.e. it is bound by that redundant traffic, not by FLOPs
(208 TFLOPS, ~8% of the bf16 peak).

Scored on the real per-forward-pass mix: 18 full-attention layers + 18
sliding-window(128) layers, since one config has to serve both.

Rule 6b: candidates are interleaved one round each, median across rounds.
Every candidate is checked against the shipped default's output first.
"""
import argparse
import itertools
import json
import statistics

import torch
import triton

import sglang.kernels.ops.attention.extend_attention as E

BS = 2
EXT = 8192
QH, KVH, HD = 32, 4, 64
N_FULL, N_SWA = 18, 18
WINDOW = 128
DEV = "cuda"
DT = torch.bfloat16


def build(seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    n = BS * EXT
    mk = lambda *s: torch.randn(*s, device=DEV, dtype=DT, generator=g)
    return dict(
        q=mk(n, QH, HD), k=mk(n, KVH, HD), v=mk(n, KVH, HD),
        o=torch.empty(n, QH, HD, device=DEV, dtype=DT),
        kbuf=mk(n, KVH, HD), vbuf=mk(n, KVH, HD),
        qo_indptr=torch.arange(BS + 1, device=DEV, dtype=torch.int32) * EXT,
        kv_indptr=torch.zeros(BS + 1, device=DEV, dtype=torch.int32),
        kv_indices=torch.empty(0, device=DEV, dtype=torch.int32),
        sinks=torch.randn(QH, device=DEV, dtype=torch.float32, generator=g),
    )


def call(t, w):
    E.extend_attention_fwd(
        t["q"], t["k"], t["v"], t["o"], t["kbuf"], t["vbuf"],
        t["qo_indptr"], t["kv_indptr"], t["kv_indices"],
        None, True, None, EXT, 1.0, 1.0,
        sm_scale=HD ** -0.5, logit_cap=0.0,
        sliding_window_size=w, sinks=t["sinks"],
        window_kv_offsets=None, page_size=1,
    )


# ---------------------------------------------------------------- overrides
_ORIG_BLOCKS = E._get_block_sizes_for_extend_attention
_ORIG_KERNEL = E._fwd_kernel


class _KernelProxy:
    """Wraps the triton.jit kernel so extra launch kwargs (num_stages,
    waves_per_eu) can be overridden without editing the launcher."""

    def __init__(self, inner, extra):
        self.inner, self.extra = inner, extra

    def __getitem__(self, grid):
        run = self.inner[grid]

        def _launch(*a, **kw):
            kw.update(self.extra)
            return run(*a, **kw)

        return _launch

    def __getattr__(self, n):
        return getattr(self.inner, n)


def set_cfg(cfg):
    if cfg is None:
        E._get_block_sizes_for_extend_attention = _ORIG_BLOCKS
        E._fwd_kernel = _ORIG_KERNEL
        return

    def _blocks(Lq, Lv):
        dm, dpe, dv, bm, bn, nw = _ORIG_BLOCKS(Lq, Lv)
        return (dm, dpe, dv, cfg.get("BLOCK_M", bm),
                cfg.get("BLOCK_N", bn), cfg.get("num_warps", nw))

    E._get_block_sizes_for_extend_attention = _blocks
    extra = {k: cfg[k] for k in ("num_stages", "waves_per_eu",
                                 "matrix_instr_nonkdim", "kpack") if k in cfg}
    E._fwd_kernel = _KernelProxy(_ORIG_KERNEL, extra) if extra else _ORIG_KERNEL


def out_for(cfg, t, w):
    set_cfg(cfg)
    try:
        t["o"].zero_()
        call(t, w)
        torch.cuda.synchronize()
        return t["o"].clone()
    finally:
        set_cfg(None)


def time_for(cfg, t, w, iters):
    set_cfg(cfg)
    try:
        return triton.testing.do_bench(lambda: call(t, w), warmup=5, rep=iters,
                                       return_mode="median")
    finally:
        set_cfg(None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--out", default="/work/analysis/extend_cfg_sweep.json")
    args = ap.parse_args()

    t = build()
    dm, dpe, dv, bm, bn, nw = _ORIG_BLOCKS(HD, HD)
    print(f"shipped default: BLOCK_M={bm} BLOCK_N={bn} num_warps={nw} "
          f"num_stages=1 waves_per_eu=1 matrix_instr_nonkdim=16 kpack=2")

    cands = []
    for BM, BN, W_, ST, WPE in itertools.product(
            [64, 128, 256], [32, 64, 128], [2, 4, 8], [1, 2], [1, 2]):
        if BM * BN * W_ > 128 * 128 * 8:      # LDS / register blowup guard
            continue
        if BM == 64 and BN == 64 and W_ == 4 and ST == 1 and WPE == 1:
            continue                          # that's the default arm
        cands.append(dict(BLOCK_M=BM, BLOCK_N=BN, num_warps=W_,
                          num_stages=ST, waves_per_eu=WPE))
    print(f"{len(cands)} candidates (+ shipped default)")

    # ---- correctness gate before any timing counts ------------------------
    refs = {w: out_for(None, t, w) for w in (-1, WINDOW)}
    good = []
    for c in cands:
        try:
            ok = True
            for w in (-1, WINDOW):
                o = out_for(c, t, w)
                r = refs[w]
                err = (o.float() - r.float()).abs().max().item()
                den = max(r.float().abs().max().item(), 1e-6)
                ok &= (err / den) < 0.02
            if ok:
                good.append(c)
        except Exception as ex:
            print(f"  FAIL {c}: {type(ex).__name__}: {str(ex)[:90]}")
    print(f"{len(good)} pass the correctness gate")

    arms = [("default", None)] + [(json.dumps(c, sort_keys=True), c) for c in good]
    samples = {n: [] for n, _ in arms}
    for r in range(args.rounds):
        for name, c in arms:                   # interleaved
            full = time_for(c, t, -1, args.iters)
            swa = time_for(c, t, WINDOW, args.iters)
            samples[name].append((full, swa))
        print(f"  round {r+1}/{args.rounds}", flush=True)

    res = []
    for name, _ in arms:
        f = statistics.median(s[0] for s in samples[name])
        s = statistics.median(s[1] for s in samples[name])
        res.append(dict(name=name, full_us=f * 1000, swa_us=s * 1000,
                        pass_ms=(N_FULL * f + N_SWA * s)))
    d = [r for r in res if r["name"] == "default"][0]
    for r in res:
        r["speedup"] = d["pass_ms"] / r["pass_ms"]
    res.sort(key=lambda r: r["pass_ms"])

    print(f"\ndefault: full {d['full_us']:.1f} us  swa {d['swa_us']:.1f} us  "
          f"-> {d['pass_ms']:.2f} ms per prefill pass (18 full + 18 swa)")
    print(f"{'pass_ms':>9} {'x':>7} {'full_us':>9} {'swa_us':>8}  config")
    for r in res[:15]:
        print(f"{r['pass_ms']:9.2f} {r['speedup']:7.3f} {r['full_us']:9.1f} "
              f"{r['swa_us']:8.1f}  {r['name']}")

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
