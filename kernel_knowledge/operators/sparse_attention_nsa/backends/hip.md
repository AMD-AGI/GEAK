---
title: sparse_attention_nsa on HIP/C++ — SOTA card
kind: sota_card
operator: sparse_attention_nsa
backend: hip
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
status: experimental
updated: 2026-06-08
sources:
  - https://fergusfinn.com/blog/deepseek-v4-flash-mi300x/
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0
---

# sparse_attention_nsa × HIP/C++

## TL;DR
There is **no public hand-written HIP NSA attention kernel** as of 2026-06. HIP's role in the NSA stack
on AMD is the **glue and gather layer**, not the attention matmul: the DeepSeek-V4 MI300X bring-up wrote
**ROCm-specific dispatch helpers** in the framework that route to AITER where a kernel exists and fall
through to Triton where it doesn't (paged MQA logits, sparse MLA prefill/decode). HIP/C++ is the right
tool if you need a custom **block-gather**, **fnuz quantise-and-insert KV**, or **capture-safe metadata
builder** that Triton can't express cleanly. For the attention itself, use the Triton path
([triton.md](triton.md)).

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| ROCm dispatch helper (HIP/Python glue) | DeepSeek-V4 MI300X bring-up | gfx942 | enables the path; +8.6% from static capture-safe metadata | gfx942 fallback routing, ragged metadata |
| fnuz quantise-and-insert KV helper | bring-up (fused quantise + paged insert) | gfx942; fp8 fnuz | correctness (cache byte match) | fp8 sliding-window/sparse KV cache |
| custom block-gather (HIP) | author | gfx942/950 | none published | non-contiguous selected-block gather |

> **Primarily Triton-portable; runs on MI300X via [[triton_amd]]. No hand-tuned CK/asm/HIP NSA attention
> kernel known as of 2026-06.** The portable path is Triton + HIP glue for gather/quant/metadata.

## Config space / knobs
- HIP wave64 model: block = multiple of 64, grid ≥1024 wgs, `__launch_bounds__(…,minWavesPerEU)` to cap
  VGPRs, `__restrict__` for wide `global_load_dwordx4`. See [[hip_cpp]].
- `-munsafe-fp-atomics` for atomic accumulate in a gather/scatter.
- fnuz fp8 byte layout for the quantise-and-insert helper (gfx942).

## Numerics / parity
Gather/quant glue must preserve the fnuz dialect (off-by-one bias → 2× error). Capture-safe static
metadata avoids index corruption under HIP graphs. See [numerics.md](../numerics.md).

## Integration (rebind seam)
HIP helpers are registered as custom ops / called from the framework attention backend; the seam is the
dispatcher (route gfx942 → Triton where AITER is broken). Verify which path runs with `AITER_LOG_MORE=1`.

## Pitfalls & anti-patterns
- Reinventing the attention matmul in HIP is rarely worth it vs Triton/CK — keep HIP to gather/quant/glue.
- Host→device scalar writes under HIP-graph capture → hangs/corruption; use static buffers.
- Wrong fnuz packing → silent 2× error in the indexer ranking.

## How to verify
Selection-overlap + greedy parity (see [numerics.md](../numerics.md)); `rocprofv3` to confirm the gather/
quant helpers aren't the bottleneck; check capture-safe metadata under `hipGraph` capture.

## Alternatives / cross-links
[overview.md](../overview.md) · [triton.md](triton.md) · [tilelang.md](tilelang.md) · languages: [[hip_cpp]] ·
[[triton_amd]] · ops: [[gather_scatter]] · [[paged_kv_copy]] · [[kv_cache_quant]].

## Sources
- ROCm helper / fnuz quantise-and-insert / capture-safe metadata: https://fergusfinn.com/blog/deepseek-v4-flash-mi300x/
- HIP wave64 / launch_bounds / fp-atomics: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
- aiter on-box (no HIP NSA attn kernel; Triton sparse-MLA only): `ROCm/aiter@a6bb49937` (on-box).
