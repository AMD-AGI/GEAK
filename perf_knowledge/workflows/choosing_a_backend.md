---
title: Choosing a backend — per-operator-family recommendation guide
kind: workflow
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp4_e2m1, int8]
regimes: [prefill, decode, training, both]
status: sota
updated: 2026-06-08
sources:
  - GEAK/perf_knowledge/index/sota_matrix.md
  - GEAK/perf_knowledge/index/decision_trees.md
  - GEAK/e2e_workflow/roles/op_benchmarker.md
---

# Choosing a backend

## TL;DR
This is the one-page prior that summarizes the per-operator `overview` SOTA tables (and
[`../index/sota_matrix.md`](../index/sota_matrix.md)) into a "what to try first" guide.
These are **priors for ORDERING**, never a verdict — the operator's SOTA cards + a
same-session A/B on the immutable oracle are the judge
([`optimize_single_kernel.md`](optimize_single_kernel.md)). Past results never justify
*not* trying; they only order the attempts.

## GEMM family
| operator | try first → then | notes |
|---|---|---|
| **dense_gemm** | aiter DB tune (engages live path) → FlyDSL author → hipBLASLt → asm | Triton usually loses to tuned hipBLASLt; need to beat the library → hand-asm or FlyDSL. [`gemm_tuning_workflow.md`](gemm_tuning_workflow.md) |
| **scaled_quant_gemm** (fp8/int8) | aiter/CK scaled-GEMM → FlyDSL author (preferred for fp8/A4W4/mxfp4) | gfx942 fp8 FNUZ; gfx950 block-scaled MXFP. accuracy-gate. [`../operators/scaled_quant_gemm/`](../operators/scaled_quant_gemm/) |
| **skinny_gemv_decode** (M≤256) | aiter skinny/wvSplitK → split-K Triton/FlyDSL (fill 304 CUs) → fuse epilogue | [`../operators/skinny_gemv_decode/`](../operators/skinny_gemv_decode/) |
| **grouped_gemm_moe** | aiter fused_moe / FlyDSL / CK / asm | auto-selects quant kernel. [`../operators/grouped_gemm_moe/`](../operators/grouped_gemm_moe/) |
| **batched_gemm / splitk_streamk_gemm** | aiter / hipBLASLt → split-K Triton/FlyDSL | [`../operators/splitk_streamk_gemm/`](../operators/splitk_streamk_gemm/) |

## Attention family
| operator | try first → then | notes |
|---|---|---|
| **attention_prefill_fmha** | CK-Tile FMHA (fastest general) → AITER FA → Triton FA (editable) → TileLang (≈1.5× Triton on CDNA3) → asm | `VLLM_USE_TRITON_FLASH_ATTN=0` selects CK. FlashInfer N/A on AMD. [`../operators/attention_prefill_fmha/`](../operators/attention_prefill_fmha/) |
| **attention_decode_paged** | AITER asm decode (`pa_fwd_asm`) → CK → Triton | memory-bound; shuffled KV layout. [`../operators/attention_decode_paged/`](../operators/attention_decode_paged/) |
| **mla_attention** | AITER MLA (asm decode) → TileLang (≈95% of asm) → Triton MLA | gfx942: AITER_TRITON_MLA ~2-3% better; gfx950: AITER_MLA. [`attention_backend_selection.md`](attention_backend_selection.md) |
| **gqa_mqa / sliding_window / chunked_prefill** | AITER FA (3-path) → Triton unified | unsupported KV head-dim → avoid ROCM_ATTN. [`../operators/gqa_mqa_attention/`](../operators/gqa_mqa_attention/) |
| **linear_attention_gated_delta** | **Triton** (SOTA, editable) → HIP | the gated-delta Qwen3.5 path; ensure varlen serving survives. [`../operators/linear_attention_gated_delta/`](../operators/linear_attention_gated_delta/) |

## Editable custom kernels (norms / rope / act / elementwise)
| operator family | try first | notes |
|---|---|---|
| **rmsnorm / fused_add_rmsnorm / layernorm** | aiter fused → Triton (editable) | fuse residual-add. small Amdahl mass each → **stack**. [`../operators/rmsnorm/`](../operators/rmsnorm/) |
| **rope / mrope** | aiter fused rope+kvcache → Triton | [`../operators/rope/`](../operators/rope/) |
| **act_and_mul (silu/gelu)** | aiter → Triton; fuse into GEMM epilogue | [`../operators/act_and_mul_silu_gelu/`](../operators/act_and_mul_silu_gelu/), [`../operators/gemm_epilogue_fused/`](../operators/gemm_epilogue_fused/) |
| **softmax / sampling / argmax_topk** | Triton (editable) → aiter | [`../operators/sampling_topk_topp/`](../operators/sampling_topk_topp/) |
| **quant/dequant (fp8/fp4/int8), kv_cache_quant** | aiter scaled-quant → fused_norm_quant → Triton | accuracy-gate. [`../quantization/`](../quantization/) |

## Collectives / MoE dispatch
| operator | try first | notes |
|---|---|---|
| **allreduce** | QuickReduce (up to 3× on MI300X, no code change) → RCCL/aiter | [`../operators/allreduce/`](../operators/allreduce/), [`../backends/mori_rccl/`](../backends/mori_rccl/) |
| **moe_dispatch_combine / all_to_all** | aiter / MoRI | [`../operators/moe_dispatch_combine/`](../operators/moe_dispatch_combine/) |
| **allgather / reduce_scatter** | RCCL/aiter → MoRI | [`../operators/allgather/`](../operators/allgather/) |

## The cross-family heuristics (from the decision trees)
- **Serving stack routes through aiter (sglang/vLLM)?** → tune the aiter DB FIRST (it's the
  only lever that engages the live path); also author a Triton/FlyDSL impl and e2e-gate the
  best. ⚠ TunableOp / `HIPBLASLT_TUNING_FILE` do **not** engage the aiter path.
- **Raw torch/F.linear (no aiter)?** → hipBLASLt offline tune or TunableOp.
- **Beat the library matmul itself?** → hand-asm or FlyDSL; pure Triton usually loses.
- **Editable custom kernels** → Triton first (fastest to iterate).
- **Small Amdahl mass (<0.5% each)?** → STACK the cluster, gate combined.
- **Quantized GEMM** → gfx942: fp8 FNUZ via aiter/CK; gfx950: MXFP8/MXFP6/MXFP4 block-scaled
  (FP6 at FP4 rate). Always accuracy-gate, never byte-parity.

## gfx942 vs gfx950 quick note
- **gfx942 (CDNA3, MI300X/MI325X)**: fp8 FNUZ; AITER_TRITON_MLA slightly ahead for MLA.
- **gfx950 (CDNA4, MI350X)**: block-scaled MXFP8/MXFP6/MXFP4 MFMA (FP6 at FP4 rate);
  AITER_MLA (asm prefill) leads. See [`../hardware/cdna3_mi300/`](../hardware/cdna3_mi300/),
  [`../hardware/cdna4_mi350/`](../hardware/cdna4_mi350/).

## How to use this guide
1. Find the operator family above → get the ordered candidate list.
2. Run the Tier-A bake-off ([`optimize_single_kernel.md`](optimize_single_kernel.md)) — the
   A/B on the immutable oracle is the judge.
3. For a head op, also tune AND author; let the e2e gate pick best of {tuned, authored}.

## Sources
- Status badges + per-family rankings: `GEAK/perf_knowledge/index/sota_matrix.md` (each cell links the evidence card).
- Routing heuristics: `GEAK/perf_knowledge/index/decision_trees.md`.
- "try every backend, order by prior, never skip authoring": `GEAK/e2e_workflow/roles/op_benchmarker.md`.
