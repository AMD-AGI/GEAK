#!/usr/bin/env python3
"""Interleaved sweep of aiter's Triton `unified_attention` 3D decode kernel at the
exact shape gpt-oss-120b hits under this bundle's workload.

Shape comes from the reference trace (reference/tracelens, TP-1 rank), decode
steady state:

    kernel_unified_attention_3d_num_query_heads_32_num_queries_per_kv_8
    _BLOCK_SIZE_64_TILE_SIZE_64_HEAD_SIZE_64_NUM_SEGMENTS_PER_SEQ_16
    -> 18 calls/step x 100.0 us = 1800 us/step = 21.2% of decode

i.e. bs=64, 32 local q heads, 4 local kv heads, head_dim 64, page_size 64,
bf16 KV, attention sinks on, full-attention (no sliding window) layers.

Measurement discipline (tuning-core/measurement.md):
  * Rule 6b -- candidates are INTERLEAVED, one round each, median across rounds.
    Back-to-back timing on gfx950 manufactures and hides wins.
  * Rule 5  -- KV footprint is ~700 MB, far above the 256 MB MALL, and the page
    table is shuffled so each replay walks a different physical order.
  * Rule 1/2 -- do_bench does sync + warmup.
  * Correctness -- every candidate is checked against the shipped default's
    output before its timing is allowed to count.
"""
import argparse
import itertools
import json
import os
import statistics
import sys

import torch
import triton

import aiter.ops.triton.attention.unified_attention as UA
from aiter.ops.triton._triton_kernels.attention.unified_attention import (
    kernel_unified_attention_3d,
    reduce_segments,
)

# ---------------------------------------------------------------- problem shape
BS = int(os.environ.get("BS", 64))
NUM_Q_HEADS = 32          # 64 // tp2
NUM_KV_HEADS = 4          # 8 // tp2
HEAD_SIZE = 64
PAGE = 64
CONTEXT = 11264           # --context-length; max_seqlen_k = pages * page_size
MAX_PAGES = CONTEXT // PAGE          # 176
SEQLEN = int(os.environ.get("SEQLEN", 8704))   # ISL 8192 + mid-decode
SCALE = HEAD_SIZE ** -0.5
DEV = "cuda"


def build_inputs(seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = torch.randn(BS, NUM_Q_HEADS, HEAD_SIZE, device=DEV, dtype=torch.bfloat16, generator=g)
    # one private page range per sequence, so the footprint is bs*seqlen and no
    # two sequences share cache lines (matches --disable-radix-cache).
    n_pages = BS * MAX_PAGES
    k = torch.randn(n_pages, PAGE, NUM_KV_HEADS, HEAD_SIZE, device=DEV,
                    dtype=torch.bfloat16, generator=g)
    v = torch.randn(n_pages, PAGE, NUM_KV_HEADS, HEAD_SIZE, device=DEV,
                    dtype=torch.bfloat16, generator=g)
    out = torch.empty(BS, NUM_Q_HEADS, HEAD_SIZE, device=DEV, dtype=torch.bfloat16)
    # shuffled page table: physical pages are not contiguous per sequence in a
    # real server either, and a contiguous table would flatter the TLB.
    perm = torch.randperm(n_pages, device=DEV, generator=g).int()
    block_table = perm.view(BS, MAX_PAGES).contiguous()
    cu_seqlens_q = torch.arange(BS + 1, device=DEV, dtype=torch.int32)
    seqused_k = torch.full((BS,), SEQLEN, device=DEV, dtype=torch.int32)
    sinks = torch.randn(NUM_Q_HEADS, device=DEV, dtype=torch.float32, generator=g)
    return dict(q=q, k=k, v=v, out=out, block_table=block_table,
                cu_seqlens_q=cu_seqlens_q, seqused_k=seqused_k, sinks=sinks)


def call(t):
    UA.unified_attention(
        q=t["q"], k=t["k"], v=t["v"], out=t["out"],
        cu_seqlens_q=t["cu_seqlens_q"], max_seqlen_q=1,
        seqused_k=t["seqused_k"], max_seqlen_k=CONTEXT,
        softmax_scale=SCALE, causal=True, window_size=(-1, -1),
        block_table=t["block_table"], softcap=0,
        q_descale=None, k_descale=None, v_descale=None,
        sinks=t["sinks"],
    )


# ---------------------------------------------------------------- config override
_ORIG_SELECT_3D = UA.select_3d_config


def make_override(cfg):
    """Patch select_3d_config to return `cfg`, keeping everything else the
    shipped default (so we tune exactly the knobs named and nothing else)."""
    def _sel(*a, **kw):
        attn, red = _ORIG_SELECT_3D(*a, **kw)
        if cfg is None:
            return attn, red
        attn = dict(attn)
        red = dict(red)
        for k in ("TILE_SIZE", "NUM_SEGMENTS_PER_SEQ", "num_warps", "waves_per_eu",
                  "num_stages"):
            if k in cfg:
                attn[k] = cfg[k]
        # the reduce kernel must agree on the split count and tile size
        red["NUM_SEGMENTS_PER_SEQ"] = attn["NUM_SEGMENTS_PER_SEQ"]
        red["TILE_SIZE"] = attn["TILE_SIZE"]
        if "reduce_num_warps" in cfg:
            red["num_warps"] = cfg["reduce_num_warps"]
        return attn, red
    return _sel


def default_config():
    attn, red = _ORIG_SELECT_3D(
        HEAD_SIZE, PAGE, CONTEXT, UA.get_num_sms() * 4,
        BS * NUM_KV_HEADS, torch.bfloat16, torch.bfloat16,
    )
    return attn, red


def time_cfg(cfg, t, iters):
    UA.select_3d_config = make_override(cfg)
    try:
        return triton.testing.do_bench(lambda: call(t), warmup=10, rep=iters,
                                       return_mode="median")
    finally:
        UA.select_3d_config = _ORIG_SELECT_3D


def run_cfg_once(cfg, t):
    UA.select_3d_config = make_override(cfg)
    try:
        t["out"].zero_()
        call(t)
        torch.cuda.synchronize()
        return t["out"].clone()
    finally:
        UA.select_3d_config = _ORIG_SELECT_3D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--out", default="/work/analysis/unified_attn_sweep.json")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    t = build_inputs()
    attn_def, red_def = default_config()
    print(f"shape bs={BS} qh={NUM_Q_HEADS} kvh={NUM_KV_HEADS} hd={HEAD_SIZE} "
          f"page={PAGE} seqlen={SEQLEN} max_k={CONTEXT}")
    print(f"shipped default attn cfg: {attn_def}")
    print(f"shipped default reduce  : {red_def}")
    kv_bytes = BS * SEQLEN * NUM_KV_HEADS * HEAD_SIZE * 2 * 2
    print(f"KV bytes touched per call: {kv_bytes/1e6:.1f} MB")

    # ---- candidate space -------------------------------------------------
    base = dict(NUM_SEGMENTS_PER_SEQ=attn_def["NUM_SEGMENTS_PER_SEQ"],
                num_warps=attn_def["num_warps"],
                waves_per_eu=attn_def["waves_per_eu"],
                num_stages=attn_def["num_stages"])
    # TILE_SIZE is the axis select_3d_config never explores: it hard-clamps to
    # min(64, next_pow2(block_size)) = 64 at page size 64, and only ever raises it
    # through the gfx12-only gluon gather path. But kernel_unified_attention_3d
    # itself has a generic TILE_SIZE != BLOCK_SIZE branch (the `seq_offset //
    # BLOCK_SIZE` gather), so 128/256 are reachable on gfx950 too.
    tiles = [64, 128, 256]
    segs = [4, 8, 16, 32, 64, 128]
    warps = [1, 2, 4, 8]
    wpe = [1, 2]
    stages = [1, 2, 3]
    if args.quick:
        tiles, segs, warps, wpe, stages = [64, 128], [8, 16], [2, 4], [1, 2], [2]

    cands = []
    seen = set()
    for ts, s, w, e, st in itertools.product(tiles, segs, warps, wpe, stages):
        # a segment must cover at least one tile, else the split is degenerate
        if s * ts > 2 * CONTEXT:
            continue
        # prune: warps*segments*prgms is the total wave count; skip absurd ones
        if s * BS * NUM_KV_HEADS * w > 256 * 4 * 32:
            continue
        c = dict(TILE_SIZE=ts, NUM_SEGMENTS_PER_SEQ=s, num_warps=w,
                 waves_per_eu=e, num_stages=st)
        key = tuple(sorted(c.items()))
        if key in seen:
            continue
        seen.add(key)
        cands.append(c)
    print(f"{len(cands)} candidates (+ shipped default), {args.rounds} interleaved rounds")

    # ---- correctness gate, before any timing counts ----------------------
    ref = run_cfg_once(None, t)
    ok = {}
    for c in cands:
        try:
            o = run_cfg_once(c, t)
            err = (o.float() - ref.float()).abs().max().item()
            den = ref.float().abs().max().item()
            ok[tuple(sorted(c.items()))] = (err / max(den, 1e-6)) < 0.02
        except Exception as ex:
            print(f"  FAIL {c}: {type(ex).__name__}: {str(ex)[:100]}")
            ok[tuple(sorted(c.items()))] = False
    cands = [c for c in cands if ok[tuple(sorted(c.items()))]]
    print(f"{len(cands)} candidates pass the correctness gate")

    # ---- interleaved timing (Rule 6b) ------------------------------------
    arms = [("default", None)] + [(json.dumps(c, sort_keys=True), c) for c in cands]
    samples = {name: [] for name, _ in arms}
    for r in range(args.rounds):
        for name, c in arms:
            samples[name].append(time_cfg(c, t, args.iters))
        print(f"  round {r+1}/{args.rounds} done", flush=True)

    res = []
    for name, _ in arms:
        v = samples[name]
        res.append(dict(name=name, median=statistics.median(v), min=min(v),
                        max=max(v), spread=(max(v) - min(v)) / statistics.median(v),
                        samples=v))
    dflt = [r for r in res if r["name"] == "default"][0]
    for r in res:
        r["speedup_vs_default"] = dflt["median"] / r["median"]
    res.sort(key=lambda r: r["median"])

    print(f"\ndefault median {dflt['median']*1000:.2f} us  "
          f"(self-spread {100*dflt['spread']:.2f}%)")
    print(f"{'us':>8} {'x':>7} {'spread%':>8}  config")
    for r in res[:20]:
        print(f"{r['median']*1000:8.2f} {r['speedup_vs_default']:7.3f} "
              f"{100*r['spread']:8.2f}  {r['name']}")

    with open(args.out, "w") as f:
        json.dump(dict(shape=dict(bs=BS, seqlen=SEQLEN, page=PAGE,
                                  qh=NUM_Q_HEADS, kvh=NUM_KV_HEADS, hd=HEAD_SIZE),
                       default_cfg=attn_def, results=res), f, indent=2)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
