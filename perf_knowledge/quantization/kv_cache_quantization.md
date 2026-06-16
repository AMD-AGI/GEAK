---
title: KV-cache quantization — fp8/int8 KV, per-head scale, paged & shuffled layout
kind: technique
gens: [gfx942, gfx950]
dtypes: [fp8_e4m3_fnuz, fp8_e4m3, int8]
regimes: [decode, both]
status: sota
updated: 2026-06-08
sources:
  - https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/
  - https://rocm.blogs.amd.com/artificial-intelligence/vllm-optimize/README.html
  - https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/gpu_dev_optimize/fp8_quantization_quark_vllm.html
  - https://rocm.blogs.amd.com/software-tools-optimization/vllm-omni/README.html
---

# KV-cache quantization

> **TL;DR.** The KV cache is a *separate quantization axis* from weights/activations: it is
> **memory-bound and grows with sequence × batch**, so FP8 KV roughly **doubles the context or batch you
> can hold** for ~free. On ROCm the KV format is **FP8 E4M3** (`--kv-cache-dtype fp8`, ROCm 6.2+);
> INT8 KV also exists. Scales can be per-tensor or **per-head**; the cache is stored in **paged** blocks
> and often **shuffled** for the attention kernel's layout. Kernel-level detail:
> [[operators/kv_cache_quant]], [[operators/paged_kv_copy]], [[operators/attention_decode_paged]].

## Why KV cache is its own problem
- KV memory = `2 · layers · heads · head_dim · seq · batch · bytes`. At long context it dominates HBM and
  caps batch size. FP8 (1 byte vs 2) **halves it** → ~2× the sequence length or batch at iso-memory
  (AMD: ROCm 6.2 on Instinct). This is a **decode/serving** lever, distinct from GEMM compute quant.
- It is read every decode step by attention, so the **dequant must be cheap** and fused into the
  attention kernel ([[operators/attention_decode_paged]], [[operators/mla_attention]]).

## Format choices on ROCm
| KV dtype | ROCm support | notes |
|---|---|---|
| **fp8 (e4m3)** | ✓ (Instinct, ROCm 6.2+) | the ROCm KV format; `--kv-cache-dtype fp8` |
| fp8_e5m2 | CUDA-only in vLLM | not the ROCm path |
| int8 | ✓ (Quark) | integer KV with scale/zero-point |
| auto (bf16/fp16) | ✓ | no quant (baseline) |
On ROCm, FP8 KV uses **E4M3** — and on CDNA3 that means the **FNUZ** dialect under the hood
([[fnuz_vs_ocp.md]]). RDNA3/Radeon only supports e5m2 in some Triton paths (a known crash with prefix
caching) — **not** an Instinct issue, but a portability caveat.

## Scale granularity: per-tensor vs per-head
- **Default (no calibration)**: scales = 1.0 — lowest accuracy.
- **On-the-fly (random-token)**: scales estimated from one warmup batch, then fixed.
- **Calibrated (recommended)**: scales from a dataset via Quark / llm-compressor. This unlocks
  **per-attention-head** scales (`STRATEGY="attn_head"` vs `"tensor"`), giving each head its own range —
  the KV analog of per-token activation quant ([[scaling_strategies.md]]).
- **Backend caveat**: per-head KV quant in the upstream recipe requires the **Flash Attention** backend;
  ROCm uses its own attention backends (AITER MHA/MLA, Triton, asm paged attention), so **verify per-head
  support** for your stack ([[operators/gqa_mqa_attention]], [[operators/mla_attention]]).

## Paged & shuffled layout
- The KV cache is stored in **paged blocks** (PagedAttention): fixed-size blocks indexed by a block
  table, not one contiguous tensor. The quantized cache keeps the same paging; quant scales live
  alongside the blocks ([[operators/paged_kv_copy]]).
- **Shuffled KV cache**: ROCm/AITER reorders the paged FP8 KV layout to match the attention kernel's
  access pattern (asm Paged Attention), which is part of the "optimized KV cache + assembly Paged
  Attention" throughput work. Wrong layout = correctness/perf loss — treat like the MXFP scale shuffle
  ([[block_scaling_mxfp.md]], [[operators/layout_shuffle]]).

## How to enable (serving)
- vLLM: `vllm serve <model> --kv-cache-dtype fp8` (dynamic), or load a Quark `*-FP8-KV` checkpoint:
  `LLM(model=..., kv_cache_dtype='fp8', quantization='quark')`. AMD ships pre-quantized
  `amd/...-FP8-KV` models. Quark export adds KV via `--kv_cache_dtype fp8`
  ([[calibration_and_quark.md]], [[deployment_recipes.md]]).
- **Perf note (AMD)**: *skipping* KV quant is often faster (no dequant in the hot attention loop);
  quantize KV when you need the **memory/context** headroom, then check the accuracy gate
  ([[accuracy_evaluation.md]]).

## Pitfalls
- **FP8 KV + prefix caching** crashes on RDNA3 (e4m3-vs-e5m2); not an Instinct bug, but verify on your
  stack.
- **Default scale=1.0 KV** silently loses accuracy — calibrate for long-context.
- **Per-head expecting Flash-Attn backend** on a ROCm stack that uses AITER/asm attention — confirm.
- **Assuming KV quant always speeds up** — it can slow decode (extra dequant); its win is memory.
- **Wrong shuffled-KV layout** — correctness/perf loss.

## Verify
- Memory: confirm KV bytes halve and max context/batch ~doubles.
- Accuracy: long-context task (e.g. long-output GSM8K / retrieval) before vs after; per-head should beat
  per-tensor on long sequences ([[accuracy_evaluation.md]]).

## Sources
- vLLM quantized KV cache (e4m3 on ROCm, per-head `attn_head` strategy, calibration): https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/
- AMD ROCm 6.2 FP8 KV (~2× seq/batch, `--kv-cache-dtype fp8`): https://rocm.blogs.amd.com/artificial-intelligence/vllm-optimize/README.html
- Quark FP8-KV checkpoint + `quantization='quark'`: https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/gpu_dev_optimize/fp8_quantization_quark_vllm.html
- Optimized KV cache + asm Paged Attention (AITER): https://rocm.blogs.amd.com/software-tools-optimization/vllm-omni/README.html
