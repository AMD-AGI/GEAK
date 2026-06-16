---
title: RMSNorm + fp8-quant fusion — the e2e win from quantizing the norm output in-kernel
kind: case_study
operator: fused_norm_quant
backend: aiter
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3_fnuz, int8]
regimes: [prefill, decode, both]
status: sota
updated: 2026-06-08
sources:
  - https://github.com/sgl-project/sglang/issues/18466
  - /sgl-workspace/aiter/aiter/ops/rmsnorm.py
  - /sgl-workspace/aiter/aiter/ops/gated_rmsnorm_fp8_group_quant.py
---

# RMSNorm + fp8 dynamic-quant fusion

> The headline e2e number here is **vendor-reported** (SGLang issue #18466), labelled inline. We
> have not re-measured this fusion in the `e2e_workflow` eval dirs; the *mechanism* and the
> *gate* are the validated, transferable parts.

## Context
`fused_norm_quant` fuses **(fused-add) RMSNorm/LayerNorm + dynamic fp8/int8 quant** into one
kernel, so the norm output is written **already quantized**. Two wins from one fusion:
1. it **removes a whole standalone quant pass** (the quant is "free" while `y` is still in fp32
   registers — just a per-row/per-group max-reduction), and
2. it **halves the downstream GEMM input traffic** (qkv / up-gate / down read fp8 = ½ bytes, or
   fp4 = ¼) — and that downstream GEMM is the Amdahl head.

So although the norm itself is only **1–4% of GPU time**, the fusion's real value lands on the
**GEMM that consumes the quantized output**. Operator overview:
[`../../operators/fused_norm_quant/overview.md`](../../operators/fused_norm_quant/overview.md).
This is the [`../../operators/rmsnorm/`](../../operators/rmsnorm/) +
[`../../operators/quant_dequant_fp8/`](../../operators/quant_dequant_fp8/) seam.

## Baseline
- **Unfused path:** RMSNorm writes bf16 → a separate dynamic-quant kernel reads bf16, writes fp8
  + scale → the GEMM reads fp8. Two passes over the activation, plus the norm output written at
  full bf16 width.
- Reference for the gate: bf16-input GEMM (or the unfused norm→quant→GEMM chain), same model.

## What works (the fusion + the kernel entrypoints)
- **per-token dynamic fp8:** `scale[m] = max(|y[m,:]|)/fp8_max`, `y_q = round(y/scale)` — fnuz fp8
  on gfx942. aiter: `rmsnorm2d_fwd_with_dynamicquant` / `_with_add_dynamicquant`.
- **group fp8** (`gated_rmsnorm_fp8_group_quant`, HIP): scale per group of 128 (head_dim=128).
- **smoothquant int8:** per-channel smoothing × per-token scale (`_smoothquant`).
- **mxfp4:** block-32 e8m0 scale (gfx950).
- Norm math in fp32, scale in fp32, RNE rounding. Backends: aiter (CK/asm + HIP group-quant +
  flydsl), Triton (`_quant_rms_norm_kernel` family), HIP.

## What didn't / the traps (kept honestly)
- **The norm alone is tiny (1–4% GPU)** — fusing the *norm* for its own sake won't clear an e2e
  gate. The win only materializes because the **downstream GEMM** reads half the bytes. If the
  GEMM doesn't actually consume the quantized output (still bf16 GEMM), the fusion buys little.
- **fnuz fp8 off-by-2× trap on gfx942** — the group/per-token scale must match the consumer
  GEMM's dequant exactly, or you get silent magnitude error.
- **It's a task-accuracy decision, not byte parity** — gate on a small eval (gsm8k), not on
  bit-exactness. fp8 dynamic quant changes the numbers.
- This is the same fp8-accuracy caution that **rejected `--quantization fp8` on Qwen3.5-27B** —
  see [`../by_model/llama_fp8_serving.md`](../by_model/llama_fp8_serving.md).

## Final result (numbers, vendor-reported)
| metric | value | source / label |
|---|---|---|
| RMSNorm + FP8 dynamic-quant fusion, e2e latency | **1–6%** | **vendor** — SGLang #18466 (Qwen3, MI300X) |
| same, throughput | **1–2%** | **vendor** — SGLang #18466 |

The fusion is the **sibling lever** to the `act_and_mul` output-quant on the MLP down-proj side,
and the QK-norm+RoPE+quant variant (`fused_qk_rmsnorm_group_quant`,
`fused_qk_norm_rope_cache_quant`) is the attention-entry form. See
[`../../operators/fused_norm_quant/fusion.md`](../../operators/fused_norm_quant/fusion.md).

## Lessons
1. **Fuse the norm for the GEMM, not for the norm.** The 1–4% norm is not the prize; the halved
   GEMM input traffic is. Route the decision by where the quantized bytes get *consumed*.
2. **Match the scale granularity to the consumer GEMM's dequant** (per-token vs group-128) — a
   mismatch is a silent accuracy bug, worse on gfx942 fnuz fp8.
3. **Gate on task accuracy** (gsm8k), never byte parity — dynamic fp8 quant is lossy by design.
4. **The win is small and stacks** — like the editable-kernel cluster, treat it as a
   carry-forward lever that compounds with the GEMM-side fp8 path, not a standalone headliner.

## Cross-links
- Operator: [`../../operators/fused_norm_quant/overview.md`](../../operators/fused_norm_quant/overview.md) · numerics: [`../../operators/fused_norm_quant/numerics.md`](../../operators/fused_norm_quant/numerics.md) · aiter card: [`../../operators/fused_norm_quant/backends/aiter.md`](../../operators/fused_norm_quant/backends/aiter.md)
- Seam: [`../../operators/rmsnorm/`](../../operators/rmsnorm/) · [`../../operators/quant_dequant_fp8/`](../../operators/quant_dequant_fp8/) · [`../../operators/fused_add_rmsnorm/`](../../operators/fused_add_rmsnorm/)
- Fusion strategy: [`../../optimization/kernel_fusion_strategy.md`](../../optimization/kernel_fusion_strategy.md)
- fp8 serving recipe: [`../by_model/llama_fp8_serving.md`](../by_model/llama_fp8_serving.md) · quant: [`../../quantization/`](../../quantization/)
- The GEMM that consumes the output: [`gemm_aiter_db_tuning.md`](gemm_aiter_db_tuning.md) · [`../../operators/scaled_quant_gemm/`](../../operators/scaled_quant_gemm/)

## Sources
- 1–6% e2e latency / 1–2% throughput, RMSNorm+FP8 dynamic-quant fusion on Qwen3 MI300X (vendor): https://github.com/sgl-project/sglang/issues/18466.
- aiter norm+quant entrypoints: `/sgl-workspace/aiter/aiter/ops/rmsnorm.py`, `/sgl-workspace/aiter/aiter/ops/gated_rmsnorm_fp8_group_quant.py`.

<!-- MANIFEST: RMSNorm+fp8 dynamic-quant fusion — vendor 1–6% e2e latency / 1–2% throughput (sglang #18466, Qwen3 MI300X); win lands on halved downstream GEMM traffic, gated on task accuracy not byte parity. -->
