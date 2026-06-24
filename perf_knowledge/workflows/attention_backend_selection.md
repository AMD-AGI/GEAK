---
title: Attention backend selection — decision tree for sglang/vLLM on MI300X/MI350X
kind: workflow
operator: attention_prefill_fmha
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode, both]
status: sota
updated: 2026-06-08
sources:
  - https://vllm.ai/blog/2026-02-27-rocm-attention-backend
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - GEAK/e2e_workflow/knowledge/gemm_tuning/aiter_gemm_tuning.md
---

# Attention backend selection

## TL;DR
The attention backend is a **server flag** (`--attention-backend ...`) — the cheapest e2e
lever (no source). Pick by `(phase, attention type, head-dim, KV dtype, gen)`, then A/B it
on the live server. Defaults are usually best, but the validated counter-example matters:
on **Qwen3.5-27B (gated-delta hybrid) / sglang / gfx942, `--attention-backend triton` won
+4.96% e2e** and stacked with the GEMM tune. Always gate the swap; never assume.

## Decision tree

```
What attention type does the model use?
├─ MLA (DeepSeek-R1, Kimi) ─────────────────────────────────────────────┐
│   default/recommended: ROCM_AITER_MLA (AITER MHA prefill + asm decode) │
│   gfx942 (MI300X/MI325X): ROCM_AITER_TRITON_MLA ~2-3% higher TPS       │
│   gfx950 (MI355X): ROCM_AITER_MLA matches/beats Triton (asm MHA        │
│                    prefill), best TTFT → keep the default              │
│   editable / Radeon baseline: TRITON_MLA (slower: ~1.3-1.5× on these)  │
│   see operators/mla_attention/                                          │
│
├─ MHA / GQA / MQA (Llama, Qwen, Mistral) ──────────────────────────────┐
│   default/recommended: ROCM_AITER_FA (3-path prefill/extend/decode)   │
│       + VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1 (shuffled KV for pa_fwd_asm)│
│   uniform workload, simpler: ROCM_AITER_UNIFIED_ATTN (single kernel,   │
│       within ~5% of AITER_FA; the 3-path edge shows on mixed/prefix)   │
│   editable surface / Radeon: TRITON_ATTN (default fallback baseline)   │
│   legacy: ROCM_ATTN (2-path; SLOW when KV head size unsupported →      │
│       2.7-4.4× worse, falls back to Triton decode) — avoid             │
│   see operators/gqa_mqa_attention/, operators/attention_prefill_fmha/  │
│
└─ Hybrid / gated-delta (Qwen3.5, Mamba-style) ─────────────────────────┐
    the attention path is a small share; the gated-delta Triton kernels  │
    dominate the editable mass. On sglang/gfx942 a TRITON attention      │
    backend swap was the e2e win (+4.96%, validated) — A/B it.           │
    see operators/linear_attention_gated_delta/                          │
```

## By phase
- **Prefill** = compute-bound (FMHA). CK-Tile FMHA is the fastest general default; AITER FA
  prefill rides asm/CK. Triton FA is editable (exposes a kernel surface); TileLang ≈1.5×
  Triton on CDNA3. `VLLM_USE_TRITON_FLASH_ATTN=0` selects CK in vLLM.
  See [`../operators/attention_prefill_fmha/`](../operators/attention_prefill_fmha/).
- **Decode** = memory-bound (paged). AITER's shared **assembly decode kernel** (`pa_fwd_asm`)
  is the SOTA path (MLA decode operates on the absorbed 576-dim latent). Use the shuffled KV
  layout to avoid conversion overhead.
  See [`../operators/attention_decode_paged/`](../operators/attention_decode_paged/),
  [`../operators/paged_kv_copy/`](../operators/paged_kv_copy/).
- **Chunked prefill / extend** = partially-cached requests; AITER FA's extend path handles
  100K+ contexts via ~32K chunks with LSE merging.
  See [`../operators/chunked_prefill/`](../operators/chunked_prefill/).

## head-dim and fp8 KV
- **Unsupported KV head sizes** are the classic ROCM_ATTN failure — it silently falls back to
  Triton decode and craters TPS. If a model has odd head dims, prefer AITER_FA / Triton
  unified paths that support them.
- **fp8 KV cache** → accuracy-gate, not byte-parity. AITER decode supports fp8 KV; the gain
  is bandwidth (decode is memory-bound). See [`../operators/kv_cache_quant/`](../operators/kv_cache_quant/),
  [`../quantization/`](../quantization/).

## gfx942 vs gfx950 ranking (from the vLLM ROCm blog, ISL=10K/OSL=1K, 64 conc)
- **MHA (Qwen3-235B), TPS rel. to AITER_FA=1.00:** AITER_FA best on all; UNIFIED 0.95–1.05×;
  TRITON_ATTN 1.08–1.30×; ROCM_ATTN 3.6–4.4× (broken head-dim fallback).
- **MLA (DeepSeek-R1), rel. to AITER_MLA=1.00:** AITER_MLA best on all; AITER_TRITON_MLA
  0.98–1.03× (slightly better on gfx942); TRITON_MLA 1.33–1.52×.
- **gfx942** (MI300X 192GB / MI325X 256GB): AITER_TRITON_MLA ~2–3% higher; **gfx950** (MI355X
  288GB): AITER_MLA matches/beats Triton (asm MHA prefill) and gets best TTFT.

## How to apply + gate
```bash
# vLLM (auto-select is usually right):
export VLLM_ROCM_USE_AITER=1
vllm serve <model> --tensor-parallel-size <tp> [--attention-backend ROCM_AITER_FA]
# sglang: --attention-backend triton | aiter | ... (A/B the swap)
```
This is a `winner_kind=flag` change in the e2e flow — candidate flags = current + this swap.
**Gate** it on the same-session tight A/B (`delta% > 0.5% AND cand_min > ref_max`) and parity
(a cross-backend bf16 argmax flip is real → check ≥10 prompts). See
[`optimize_e2e_model.md`](optimize_e2e_model.md) and
[`integrating_a_new_kernel.md`](integrating_a_new_kernel.md). **Stack** the attention flag and
the GEMM tune — they're orthogonal and compound (validated).

## Pitfalls
- FlashInfer is **not** available on AMD — don't reach for it.
- Forcing `--attention-backend` still needs `VLLM_ROCM_USE_AITER=1` for the AITER kernels.
- Don't assume the default — the gated-delta Qwen3.5 case where Triton won is real; A/B.
- ROCM_ATTN on a model with unsupported KV head dims = silent Triton fallback = big regression.

## Cross-links
- Prefill FMHA backends: [`../operators/attention_prefill_fmha/backends/`](../operators/attention_prefill_fmha/backends/)
- MLA: [`../operators/mla_attention/`](../operators/mla_attention/) · aiter MLA: [`../backends/aiter/attn_mla.md`](../backends/aiter/attn_mla.md)
- Decode paged: [`../operators/attention_decode_paged/`](../operators/attention_decode_paged/)
- Choosing backends overall: [`choosing_a_backend.md`](choosing_a_backend.md)
- Routing priors: [`../index/decision_trees.md`](../index/decision_trees.md)

## Sources
- 7 ROCm attention backends, names/flags, per-gen ranking, MLA/MHA routing: https://vllm.ai/blog/2026-02-27-rocm-attention-backend
- CK default FA / `VLLM_USE_TRITON_FLASH_ATTN=0`, FlashInfer N/A on AMD: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- Qwen3.5-27B `--attention-backend triton` +4.96% (gated-delta hybrid, sglang/gfx942): `GEAK/e2e_workflow/knowledge/gemm_tuning/aiter_gemm_tuning.md`.
