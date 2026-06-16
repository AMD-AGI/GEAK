---
title: attention_prefill_fmha on TileLang — SOTA card
kind: sota_card
operator: attention_prefill_fmha
backend: tilelang
gens: [gfx90a, gfx942]
dtypes: [fp16, bf16]
regimes: [prefill]
status: competitive
updated: 2026-06-08
sources:
  - https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-tilelang-kernel/README.html
  - https://github.com/tile-ai/tilelang
  - https://arxiv.org/abs/2511.08083
---

# attention_prefill_fmha × TileLang

## TL;DR
TileLang is the **best editability-per-FLOP** prefill attention on MI300X (CDNA3): FlashAttention forward
in ~80 lines of Python that runs **~1.53× faster than Triton** and **~2.7× PyTorch** (vendor/ROCm-blog).
It is the **default AMD attention backend in recent sglang images**. Choose it for fast iteration on a
custom attention variant where you want near-CK perf without writing CK. Treat **CDNA4/MI350 perf as
unproven** (HipKittens notes it under-uses CDNA4-peak idioms) and use aiter asm for the absolute ceiling.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| TileLang FA fwd | `tile-ai/tilelang` FA example | gfx90a/942; fp16/bf16 | **1.53× Triton, 2.7× PyTorch** @ MI300X, ROCm 7.0.1 / TileLang 0.1.7 (vendor) | editable prefill attn, CDNA3 |
| TileLang single attn kernel | HipKittens measurement | gfx942 | **257 TFLOPs** (well below peak; good for iteration not ceiling) | relative reference |

## Config space / knobs
FA optimal config on MI300X: **`block_M=128, block_N=32, threads=512`** (8 warps), `num_stages`
(pipeline depth), `num_split_q`, `qk_coalesced_width`/`v_coalesced_width` (vectorized loads),
`GemmWarpPolicy=FullRow`, `enable_rasterization=True`. Autotune over `itertools.product` (~108 candidates,
~1 s). The compiler maps tiles onto MFMA ("Auto MatrixCore") and emits the AMD LDS swizzle automatically.
See [../tuning.md](../tuning.md) and `languages/tilelang/`.

## Numerics / parity
fp32 online-softmax accumulate; fp16/bf16 storage. Greedy temp=0 parity vs a reference attention; bf16
tie-flips benign. fp8 attention is **WIP** in TileLang. See [../numerics.md](../numerics.md).

## Integration (rebind seam)
sglang: `--attention-backend tilelang` (the AMD default in recent images). The ~80-line Python kernel is
itself the edit seam — modify the tile program, re-autotune, e2e-gate.

## Pitfalls & anti-patterns
- **CDNA3-validated only** — MI350/CDNA4 perf unproven; HipKittens: lacks abstractions for flexible tile
  sizing under register pressure, thread-block scheduling, and cache-aware grid ordering; under-uses
  32×32×16 MFMA / `buffer_load_dwordx4` / XCD swizzle vs peak asm.
- Depends on CUTLASS/CK backend calls for some paths.
- Not the absolute ceiling — for that use aiter asm.

## How to verify
`pip install tilelang` (validated ROCm 7.0.1 / PyTorch 2.9.0 / TileLang 0.1.7); run the FA example,
compare latency vs Triton/PyTorch at identical `(B,H,sq,sk,d)`; check the autotuned config; greedy temp=0
parity vs reference.

## Alternatives / cross-links
[ck_tile.md](ck.md) · [triton.md](triton.md) · [asm.md](asm.md) · `languages/tilelang/` ·
`languages/tilelang/vs_triton.md` · [[../overview.md]].

## Sources
- TileLang FA on MI300X (1.53× Triton, 2.7× PyTorch, block_M=128/block_N=32/threads=512, ROCm 7.0.1): https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-tilelang-kernel/README.html
- tile-ai/tilelang (FA2 fwd MI300X #1406, AMD support): https://github.com/tile-ai/tilelang
- HipKittens (257 TFLOPs, CDNA-constraint gaps): https://arxiv.org/abs/2511.08083
