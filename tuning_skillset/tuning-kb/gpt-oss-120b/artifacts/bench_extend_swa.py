#!/usr/bin/env python3
"""Does SGLang's Triton extend (prefill) kernel actually exploit the sliding window?

The stage-split profile (analysis/prof_stage) says no: all 36 layers of a prefill
forward pass cost the same ~1440 us (1-seq chunk) / ~2800 us (2-seq chunk), even
though gpt-oss-120b alternates 18 sliding-window(128) layers with 18 full-attention
layers, and the decode path gets this right (unified_attention_2d 8.24 us vs
unified_attention_3d 95.01 us, 11x).

`_fwd_kernel` does carry a SLIDING_WINDOW_SIZE mask and a SKIP_TILE guard
(extend_attention.py:550-561), but the loop bound is still the causal diagonal
(`for start_n in range(0, cur_block_m_end, BLOCK_N)`), so every tile from 0 to the
diagonal is visited and its mask reduced even when the window makes it dead.

This measures the shape the server actually runs, at both window settings.

Shape from the EXTEND trace + launch_server.sh:
  chunked-prefill 16384, ISL 8192 -> batches of 2 whole requests
  TP=2 -> 32 local q heads, 4 local kv heads, head_dim 64, bf16, causal, sinks on
"""
import argparse
import statistics

import torch
import triton

from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd

BS = 2
EXT = 8192
QH, KVH, HD = 32, 4, 64
DEV = "cuda"
DT = torch.bfloat16


def build(seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    n = BS * EXT
    q = torch.randn(n, QH, HD, device=DEV, dtype=DT, generator=g)
    k = torch.randn(n, KVH, HD, device=DEV, dtype=DT, generator=g)
    v = torch.randn(n, KVH, HD, device=DEV, dtype=DT, generator=g)
    o = torch.empty(n, QH, HD, device=DEV, dtype=DT)
    # no prefix: with --disable-radix-cache and chunked-prefill 16384 >= ISL 8192,
    # each request is prefilled whole, so kv_indptr is all zeros.
    qo_indptr = torch.arange(BS + 1, device=DEV, dtype=torch.int32) * EXT
    kv_indptr = torch.zeros(BS + 1, device=DEV, dtype=torch.int32)
    kv_indices = torch.empty(0, device=DEV, dtype=torch.int32)
    kbuf = torch.randn(n, KVH, HD, device=DEV, dtype=DT, generator=g)
    vbuf = torch.randn(n, KVH, HD, device=DEV, dtype=DT, generator=g)
    sinks = torch.randn(QH, device=DEV, dtype=torch.float32, generator=g)
    return dict(q=q, k=k, v=v, o=o, qo_indptr=qo_indptr, kv_indptr=kv_indptr,
                kv_indices=kv_indices, kbuf=kbuf, vbuf=vbuf, sinks=sinks)


def call(t, w):
    extend_attention_fwd(
        t["q"], t["k"], t["v"], t["o"], t["kbuf"], t["vbuf"],
        t["qo_indptr"], t["kv_indptr"], t["kv_indices"],
        None,            # custom_mask
        True,            # is_causal
        None,            # mask_indptr
        EXT,             # max_len_extend
        1.0, 1.0,
        sm_scale=HD ** -0.5,
        logit_cap=0.0,
        sliding_window_size=w,
        sinks=t["sinks"],
        window_kv_offsets=None,
        page_size=1,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    t = build()
    arms = [("full (sliding_window=-1)", -1), ("swa   (sliding_window=128)", 128)]

    # correctness: the two arms are different maths, so only check each is finite
    for name, w in arms:
        t["o"].zero_()
        call(t, w)
        torch.cuda.synchronize()
        assert torch.isfinite(t["o"]).all(), f"{name} produced non-finite output"

    samples = {n: [] for n, _ in arms}
    for r in range(args.rounds):
        for name, w in arms:                       # interleaved, Rule 6b
            samples[name].append(
                triton.testing.do_bench(lambda: call(t, w), warmup=5,
                                        rep=args.iters, return_mode="median"))

    print(f"bs={BS} extend={EXT} qh={QH} kvh={KVH} hd={HD} causal, sinks on")
    base = None
    for name, _ in arms:
        v = samples[name]
        med = statistics.median(v)
        if base is None:
            base = med
        print(f"  {name}: {med*1000:8.1f} us   (min {min(v)*1000:.1f} "
              f"max {max(v)*1000:.1f})   {base/med:.2f}x vs full")

    ideal = base * (2 * 128 / EXT)
    print(f"\nif the window were exploited, the swa arm should cost roughly "
          f"{ideal*1000:.1f} us (2*W/N of the causal triangle), not "
          f"{statistics.median(samples[arms[1][0]])*1000:.1f} us")


if __name__ == "__main__":
    main()
