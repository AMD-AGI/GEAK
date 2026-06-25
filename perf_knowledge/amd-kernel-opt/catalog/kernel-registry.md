---
type: Reference
title: LLM-inference kernel registry
description: Every LLM-inference kernel referenced in this knowledge base, grouped by operator domain, with best measured speedup and a link to its case study where one exists.
tags: [registry, catalog, llm-inference, gfx942]
resource: /catalog/kernel_speedups_llm_inference.csv
timestamp: 2026-06-22T00:00:00Z
---

# Scope
The 91 LLM-inference kernels referenced across the optimization campaigns (3D/vision/generic-primitive ops excluded). Best speedup = the best correctness-passing run per kernel from the arena `task_result.yaml` corpus. Numbers here are the raw evaluation-run values; for the audited authoritative per-kernel campaign numbers see `/catalog/kernel_speedups.md`. Full data: [kernel_speedups_llm_inference.csv](/catalog/kernel_speedups_llm_inference.csv).

Legend: **bold** = has a deep [case study](/cases/index.md). ✗ = correctness FAIL (no valid speedup).

## Attention (18)
- **[hip_paged_attention_decode](/cases/paged-attention-decode.md)** — 4.60×  ·  `hip2hip/hip2hip-extracted-v2/hip_paged_attention_decode`
- **[paged_attention_decode](/cases/paged-attention-decode.md)** — 3.40×  ·  `campaign20/baseline/paged_attention_decode`
- **[hip_paged_attention_decode](/cases/paged-attention-decode.md)** — 1.60×  ·  `hip2hip/vllm-v2-cursor/hip_paged_attention_decode`
- **[paged_attention_decode](/cases/paged-attention-decode.md)** — 1.35×  ·  `hip2hip/hip2hip-extracted-v2-L2/paged_attention_decode`
- **[paged_attention_ragged](/cases/paged-attention-ragged.md)** — 1.29×  ·  `hip2hip/hip_kernel_extend_full/chushi/paged_attention_ragged`
- attention — 1.13×  ·  `hip2hip/hip_kernel_extend_full/puyuan_lv12/level2/attention`
- paged_attention_large — 1.08×  ·  `hip2hip/hip_kernel_extend_full/chushi/paged_attention_large`
- **[paged_attention_ragged](/cases/paged-attention-ragged.md)** — 1.07×  ·  `campaign20/baseline/paged_attention_ragged`
- 11_concat_and_cache_mla — 1.06×  ·  `hip2hip/hip2hip_extraction_0519/11_concat_and_cache_mla`
- paged_attention — 1.04×  ·  `hip2hip/hip_kernel_extend_full/chushi/paged_attention`
- attention_ragged — 1.03×  ·  `hip2hip/hip_kernel_extend_full/puyuan_lv12/level2/attention_ragged`
- **[kernel_unified_attention_2d](/cases/unified-attention-2d.md)** — 1.03×  ·  `campaign20/baseline/kernel_unified_attention_2d`
- **[hip_paged_attention_decode](/cases/paged-attention-decode.md)** — 1.01×  ·  `hip2hip/hip2hip-extracted/hip_paged_attention_decode`
- 01_paged_attention — 0.98×  ·  `hip2hip/hip2hip_extraction_0519/01_paged_attention`
- concat_and_cache_mla — 0.96×  ·  `hip2hip/hip_kernel_extend_full/chushi/concat_and_cache_mla`
- mhc_kernels — 0.95×  ·  `hip2hip/hip_kernel_extend_full/puyuan_lv12/level2/mhc_kernels`
- aiter_flash_attn_varlen_func — 0.00× ✗  ·  `hip2hip/hip2hip_extracted_chushi/aiter_flash_attn_varlen_func`
- pa_prefill — 0.00× ✗  ·  `hip2hip/hip_kernel_extend_full/puyuan_lv3/pa_prefill`

## MoE / routing (20)
- moe_align_block_size_kernels — 5.96×  ·  `hip2hip/hip_kernel_extend_full/puyuan_lv12/level1/moe_align_block_size_kernels`
- hip_grouped_topk — 4.38×  ·  `hip2hip/hip2hip-extracted-v2/hip_grouped_topk`
- hip_topk_softmax — 3.33×  ·  `hip2hip/hip2hip-extracted-v2/hip_topk_softmax`
- 03_moe_sum — 1.23×  ·  `hip2hip/hip2hip_extraction_0519/03_moe_sum`
- hip_grouped_topk — 1.23×  ·  `hip2hip/vllm-v2-cursor/hip_grouped_topk`
- **[moe_gemm_fp8_blockscale](/cases/moe-gemm-fp8-blockscale.md)** — 1.20×  ·  `campaign20/baseline/moe_gemm_fp8_blockscale`
- moe_sum — 1.18×  ·  `hip2hip/hip_kernel_extend_full/chushi/moe_sum`
- topk_softmax — 1.11×  ·  `hip2hip/hip_kernel_extend_full/chushi/topk_softmax`
- 08_topk_softmax — 1.10×  ·  `hip2hip/hip2hip_extraction_0519/08_topk_softmax`
- hip_topk_softmax — 1.06×  ·  `hip2hip/hip2hip-extracted/hip_topk_softmax`
- hip_grouped_topk — 1.01×  ·  `hip2hip/hip2hip-extracted/hip_grouped_topk`
- **[moe_stage1](/cases/moe-stage1.md)** — 1.00×  ·  `campaign20/baseline/moe_stage1`
- **[fused_moe_kernel_gptq_awq](/cases/fused-moe-gptq-awq.md)** — 1.00×  ·  `campaign20/baseline/fused_moe_kernel_gptq_awq`
- **[fused_moe_kernel](/cases/fused-moe-fp8-blockscale.md)** — 1.00×  ·  `campaign20/baseline/fused_moe_kernel`
- hip_topk_softmax — 1.00×  ·  `hip2hip/vllm-v2-cursor/hip_topk_softmax`
- topk_softmax_kernels_group — 1.00×  ·  `hip2hip/hip_kernel_extend_full/puyuan_lv12/level2/topk_softmax_kernels_group`
- **[moe_stage2](/cases/moe-stage2.md)** — 0.99×  ·  `campaign20/baseline/moe_stage2`
- topk_per_row_kernels — 0.94×  ·  `hip2hip/hip_kernel_extend_full/puyuan_lv12/level2/topk_per_row_kernels`
- **[fused_moe_int4_w4a16](/cases/fused-moe-int4-w4a16.md)** — 0.00× ✗  ·  `campaign20/baseline/fused_moe_int4_w4a16`
- **[_topk_forward](/cases/topk-forward.md)** — 0.00× ✗  ·  `campaign20/baseline/_topk_forward`

## GEMM (12)
- hip_skinny_gemm — 53.52×  ·  `hip2hip/hip2hip-extracted-v2/hip_skinny_gemm`
- **[_gemm_a8w8_blockscale_kernel](/cases/gemm-a8w8-blockscale.md)** — 1.30×  ·  `campaign20/baseline/_gemm_a8w8_blockscale_kernel`
- _w8a8_triton_block_scaled_mm — 1.08×  ·  `campaign20/baseline/_w8a8_triton_block_scaled_mm`
- hip_skinny_gemm — 1.05×  ·  `hip2hip/hip2hip-extracted/hip_skinny_gemm`
- skinny_gemm — 1.02×  ·  `hip2hip/hip2hip-extracted-v2-L2/skinny_gemm`
- wvSplitK — 1.01×  ·  `hip2hip/hip_kernel_extend_full/chushi/wvSplitK`
- wvSplitK — 1.01×  ·  `campaign20/baseline/wvSplitK`
- hip_skinny_gemm — 1.01×  ·  `hip2hip/vllm-v2-cursor/hip_skinny_gemm`
- **[gemm_a8w8_blockscale](/cases/gemm-a8w8-blockscale.md)** — 1.00×  ·  `campaign20/baseline/gemm_a8w8_blockscale`
- **[_gemm_a16_w16_kernel](/cases/gemm-a16-w16.md)** — 1.00×  ·  `campaign20/baseline/_gemm_a16_w16_kernel`
- 05_wvSplitK — 1.00×  ·  `hip2hip/hip2hip_extraction_0519/05_wvSplitK`
- vllm_wvsplitk — 0.00× ✗  ·  `hip2hip/hip2hip_extracted_chushi/vllm_wvsplitk`

## KV-cache (8)
- hip_reshape_and_cache — 1.05×  ·  `hip2hip/vllm-v2-cursor/hip_reshape_and_cache`
- hip_reshape_and_cache — 1.05×  ·  `hip2hip/hip2hip-extracted/hip_reshape_and_cache`
- hip_reshape_and_cache — 1.04×  ·  `hip2hip/hip2hip-extracted-v2/hip_reshape_and_cache`
- reshape_and_cache_flash — 1.03×  ·  `hip2hip/hip_kernel_extend_full/chushi/reshape_and_cache_flash`
- 09_reshape_and_cache_flash — 1.03×  ·  `hip2hip/hip2hip_extraction_0519/09_reshape_and_cache_flash`
- 06_reshape_and_cache — 1.01×  ·  `hip2hip/hip2hip_extraction_0519/06_reshape_and_cache`
- **[write_req_to_token_pool_triton](/cases/write-req-to-token-pool.md)** — 1.01×  ·  `campaign20/baseline/write_req_to_token_pool_triton`
- reshape_and_cache — 1.00×  ·  `hip2hip/hip_kernel_extend_full/chushi/reshape_and_cache`

## Activation (7)
- hip_silu_and_mul — 1.95×  ·  `hip2hip/vllm-v2-cursor/hip_silu_and_mul`
- vllm_silu_and_mul — 1.18×  ·  `hip2hip/hip2hip_extracted_chushi/vllm_silu_and_mul`
- silu_and_mul — 1.18×  ·  `hip2hip/hip_kernel_extend_full/chushi/silu_and_mul`
- silu — 1.16×  ·  `hip2hip/others/silu`
- 10_silu_and_mul — 1.15×  ·  `hip2hip/hip2hip_extraction_0519/10_silu_and_mul`
- hip_silu_and_mul — 1.07×  ·  `hip2hip/hip2hip-extracted-v2/hip_silu_and_mul`
- hip_silu_and_mul — 1.06×  ·  `hip2hip/hip2hip-extracted/hip_silu_and_mul`

## Normalization (15)
- rms_norm — 1.15×  ·  `hip2hip/hip_kernel_extend_full/chushi/rms_norm`
- fused_qk_norm — 1.10×  ·  `hip2hip/hip_kernel_extend_full/puyuan_lv12/level1/fused_qk_norm`
- 04_rms_norm — 1.03×  ·  `hip2hip/hip2hip_extraction_0519/04_rms_norm`
- 02_fused_add_rms_norm — 1.02×  ·  `hip2hip/hip2hip_extraction_0519/02_fused_add_rms_norm`
- hip_fused_add_rmsnorm — 1.01×  ·  `hip2hip/vllm-v2-cursor/hip_fused_add_rmsnorm`
- fused_add_rms_norm — 1.01×  ·  `hip2hip/hip_kernel_extend_full/chushi/fused_add_rms_norm`
- hip_fused_add_rmsnorm — 1.00×  ·  `hip2hip/hip2hip-extracted-v2/hip_fused_add_rmsnorm`
- groupnorm — 1.00×  ·  `hip2hip/hip_kernel_extend_full/puyuan_lv12/level2/groupnorm`
- gated_rmsnorm_quant_kernels — 0.99×  ·  `hip2hip/hip_kernel_extend_full/puyuan_lv12/level1/gated_rmsnorm_quant_kernels`
- aiter_rms_norm — 0.99×  ·  `hip2hip/hip2hip_extracted_chushi/aiter_rms_norm`
- rmsnorm_quant_kernels — 0.99×  ·  `hip2hip/hip_kernel_extend_full/puyuan_lv12/level1/rmsnorm_quant_kernels`
- hip_fused_add_rmsnorm — 0.98×  ·  `hip2hip/hip2hip-extracted/hip_fused_add_rmsnorm`
- aiter_rmsnorm2d_fwd_with_add — 0.97×  ·  `hip2hip/hip2hip_extracted_chushi/aiter_rmsnorm2d_fwd_with_add`
- fused_qk_rmsnorm_group_quant — 0.00× ✗  ·  `hip2hip/hip_kernel_extend_full/puyuan_lv12/level1/fused_qk_rmsnorm_group_quant`
- fused_qk_norm_rope_cache_quant — 0.00×  ·  `hip2hip/hip_kernel_extend_full/puyuan_lv12/level2/fused_qk_norm_rope_cache_quant`

## Positional embedding (6)
- hip_rotary_embedding — 1.24×  ·  `hip2hip/hip2hip-extracted-v2/hip_rotary_embedding`
- hip_rotary_embedding — 1.06×  ·  `hip2hip/vllm-v2-cursor/hip_rotary_embedding`
- hip_rotary_embedding — 1.03×  ·  `hip2hip/hip2hip-extracted/hip_rotary_embedding`
- 07_rotary_embedding — 1.01×  ·  `hip2hip/hip2hip_extraction_0519/07_rotary_embedding`
- vllm_rotary_embedding — 1.01×  ·  `hip2hip/hip2hip_extracted_chushi/vllm_rotary_embedding`
- rotary_embedding — 0.00× ✗  ·  `hip2hip/hip_kernel_extend_full/chushi/rotary_embedding`

## Quantization (1)
- **[_per_token_group_quant_fp8](/cases/per-token-group-quant-fp8.md)** — 1.01×  ·  `campaign20/baseline/_per_token_group_quant_fp8`

## SSM / linear (2)
- causal_conv1d_update — 1.00×  ·  `hip2hip/hip_kernel_extend_full/puyuan_lv12/level1/causal_conv1d_update`
- fused_split_gdr_update — 1.00×  ·  `hip2hip/hip_kernel_extend_full/puyuan_lv12/level1/fused_split_gdr_update`

## Linear attention (1)
- **[chunk_scaled_dot_kkt_fwd_kernel](/cases/chunk-scaled-dot-kkt.md)** — 0.99×  ·  `campaign20/baseline/chunk_scaled_dot_kkt_fwd_kernel`

## Other LLM (1)
- **[_fwd_grouped_kernel_stage1](/cases/fwd-grouped-stage1.md)** — 1.00×  ·  `campaign20/baseline/_fwd_grouped_kernel_stage1`

## At-ceiling kernels (documented ~1.00×, no headroom)

These returned ~1.0× **on purpose** — and that is the reference value: each has a *diagnosed*
reason there is no headroom, so a future attempt should stop here rather than re-attack it.
A bare 1.0× with no diagnosis (the many near-1.0× rows above) carries no such signal.
Source: `head_kernels/campaign20/FINAL_REPORT.md` (Director-validated).

| kernel | result | why it is at ceiling | signal that says "stop" |
|---|---|---|---|
| `fused_moe_int4_w4a16` | 1.00× | starting file is already the prior winner (load-once-unpack + scale/zp dedup, ~5.2× over stock); body AMDGCN-proven neutral — see [case](/cases/fused-moe-int4-w4a16.md) | the input kernel is itself an accepted winner |
| `moe_gemm_fp8_blockscale` | 1.005× | baseline is a hand-tuned aiter **ASM** kernel; in-place levers exhausted (a *backend swap* to 2-stage CK is a different task → 1.19×, see [case](/cases/moe-gemm-fp8-blockscale.md)) | vendor hand-tuned ASM baseline |
| `moe_stage1` | 1.00× (campaign20 regime) | ~96% MFMA peak; only remaining lever is in an off-limits `.cu`. A host-side V3→V1 pipeline lever found 1.08–1.11× in a *less-constrained* task — see [case](/cases/moe-stage1.md) | profiler ≈ peak MFMA + editable surface exhausted |
| `_w8a8_triton_block_scaled_mm` | 1.00× (0.994) | every throughput lever breaks the correctness/numerics gate → [numerics-gate anti-pattern](/anti-patterns/numerics-gate-violation.md) | faster variants all fail correctness |
| `wvSplitK` | 1.00× | ~67% launch-floor/HBM-bound with no editable host wrapper → [launch-bound anti-pattern](/anti-patterns/launch-bound-body-opts-invisible.md) | launch/HBM floor + no host surface to change |
| `paged_attention_large` | BLOCKED | launch-floor-bound, low headroom; benchmark gate times out before a stable measurement | cannot even establish a stable baseline |

See also the [bottleneck-first methodology](/methodology/bottleneck-first-classification.md) for
how to recognise a ceiling before spending attempts on it.

## Other referenced (outside the arena task_result corpus)
- **[MLA prefill flash-attn (head_dim_qk=192)](/cases/mla-prefill.md)** — 1.21× · `spare_kernels/k04_fmha_prefill`
- **[paged_attention vLLM single-pass](/cases/paged-attention-vllm-singlepass.md)** — 1.18× · `task_specific_skills/paged_attention_vllm`

## End-to-end serving (MiniMax-M2.5, 4×MI300X TP=4)
- +37.1% / +34.0% / +24.6% / +13.2% output tok/s across runs — decode `block_m 64→16` + CK bpreshuffle GEMM. See [block_m routing](/patterns/block-m-routing-sparsity.md), [backend dispatch](/patterns/backend-dispatch-swap.md), `/catalog/kernel_speedups.md` §A.

_Total: 91 kernels listed; 22 have a deep case study._