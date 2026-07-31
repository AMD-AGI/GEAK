---
title: dense_gemm — numerics
kind: quant
operator: dense_gemm
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e5m2_fnuz, mxfp4]
regimes: [prefill, decode, training]
updated: 2026-06-08
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html
  - https://blog.vllm.ai/2025/02/24/ptpc-fp8-rocm.html
  - ROCm/aiter@HEAD:gradlib/gradlib/gemm_tuner.py
---

# dense_gemm — numerics

## TL;DR
bf16/fp16 GEMM with **fp32 accumulate** is the parity baseline; swapping library *solutions*
(hipBLASLt/CK/asm) is same-math → byte-comparable within `err_ratio<0.05`. Quantized variants
(fp8/mxfp4) change the math and must be gated on **task accuracy**, never byte parity.

## Accumulation & dtype contract
- In: bf16/fp16. Accumulate: **fp32** (MFMA accumulators are fp32). Out: bf16 (sglang `nn.Linear`).
- fp8 (E4M3FNUZ on gfx942 / E4M3FN on gfx950): A,B fp8 + per-tensor, per-token/per-channel, **or
  per-`[128,128]` block** scales, fp32 accumulate, bf16 out. CDNA4 adds native fp4/fp6 dense + block-scaled
  MFMA. **a8w8 block-scale has a scale-*representation* caveat on CDNA4 — see the block-scale section below.**

## Parity bands
- **Same-dtype solution swap (the aiter tune)**: parity-safe. gradlib gates each candidate on
  `err_ratio < 0.05` vs a reference GEMM; accepted swaps move no task metric.
- **fp8 quant**: not byte-comparable. PTPC-FP8 (per-token-activation, per-channel-weight) on vLLM/ROCm
  recovers near-bf16 accuracy and is the recommended fp8 recipe; per-tensor fp8 is faster but lossier.
- **mxfp4 (CDNA4)**: per-32-element E8M0 block scale. Naive MXFP4 loses accuracy; online-rotation +
  SmoothQuant gets near-lossless (W4A16/W4A8). Gate on downstream eval, not numeric tolerance.

## fp8 a8w8 block-scale (per-`[128,128]`): arbitrary fp32 ≠ E8M0
A distinct fp8 regime from the per-tensor / per-token rows above: **a8w8 block-scale** carries one
**arbitrary fp32** scale per `[128,128]` tile of A and of B (CK `gemm_a8w8_blockscale`,
`weight_block_size=[128,128]`; fp32 accumulate, bf16 out). Porting it onto a **block-scaled-MFMA** path on
gfx950 hits a *representational* trap:
- **HW scale is E8M0; CK's scale is not.** CDNA4's block-scaled MFMA (`mfma_scale_*_f8f6f4`) takes an
  **E8M0** per-operand scale — power-of-two exponent only, no mantissa ([[quantization/block_scaling_mxfp.md]]).
  CK's per-block scale is **arbitrary fp32**. Folding it into the E8M0 operand rounds to the nearest power
  of two → **silently loses precision → parity fails**. No knob recovers it; it is representational.
- **Correct math = software fp32 post-MFMA scale (two-level accumulate).** Keep the MFMA **unscaled fp8**
  (pin any HW E8M0 scale to `1.0`), promote the fp8 partials to fp32 **after** the MFMA, ×the per-block
  fp32 scale, accumulate. Reproduces CK **bit-comparably (err=0, cos=1.0)**, and is **irreducible** — the
  scale changes every 128 along K, so one promote per K-block is mandatory (can't hoist out of the loop).
- **Gate** vs an fp32 `dequant→matmul→bf16` oracle at `rtol=atol=1e-2`; an E8M0-rounded path fails it.
  This card is the arch/backend-agnostic *why*; the gfx950 **CK→FlyDSL kernel-selection recipe** that
  applies it (which software-scale core per shape + XCD / scheduling perf levers) is the gated expert skill
  `flydsl_fp8_blockscale_gemm` (opt-in via `use_expert_skills=true`).

## Tie-break / determinism
- Plain dense GEMM has no argmax/tie-break; output order is deterministic per solution. Split-K /
  Stream-K introduce a different fp32 reduction order → tiny ULP drift (still within band), so pin the
  solution if exact reproducibility across runs is required.

## How to gate
- Same-math swap: `max_reldiff < 0.05` vs untuned reference on the live shape (gradlib does this).
- Quant: run the task eval (e.g. accuracy on the serving workload) A/B; accept only if no regression.
- Watch the input-distribution effect: GEMM perf varies >20% by input values, but accuracy gating must
  use representative (not all-zero) tensors or you under-detect quant error.

## Sources
- fp8 GEMM on CDNA4 (E4M3FN, bf16 out, fp32 accum): ROCm CDNA4 GEMM blog.
- PTPC-FP8 accuracy recipe: vLLM blog 2025-02-24.
- err_ratio<0.05 gate: `ROCm/aiter@HEAD:gradlib/gradlib/gemm_tuner.py`.
- mxfp4 near-lossless: ROCm MXFP4 online-rotation blog (see fusion.md / scaled_quant_gemm).
- fp8 a8w8 block-scale (arbitrary fp32) vs E8M0 HW scale + the software fp32 post-MFMA fix: E8M0 =
  power-of-two ([[quantization/block_scaling_mxfp.md]], matrix-cores-cdna blog); CK oracle
  `gemm_a8w8_blockscale` (`ROCm/aiter:csrc/ck_gemm_a8w8`). Applied recipe: gated expert skill
  `flydsl_fp8_blockscale_gemm`.
