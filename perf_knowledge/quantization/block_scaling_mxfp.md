---
title: Block scaling / microscaling (MXFP) — E8M0, block-scaled MFMA, FP6@FP4 rate
kind: technique
gens: [gfx950]
dtypes: [fp4_e2m1, fp6_e2m3, fp6_e3m2, mxfp4, mxfp6, fp8_e4m3]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
  - https://arxiv.org/pdf/2310.10537
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
  - https://salykova.github.io/matrix-cores-cdna
---

# Block scaling / microscaling (MXFP)

> **TL;DR.** Microscaling = **block floating point**: a group of **32** low-bit elements shares one
> **E8M0** (8-bit, exponent-only) scale, chosen so the block's largest element lands at the top
> representable power-of-2. That per-block self-normalization is what makes 4-bit usable. CDNA4 runs it
> *in the matrix core* (block-scaled MFMA); CDNA3 only simulates. The bit-exact per-block algorithm
> (amax, `f32_to_e8m0`, shuffle) is in [[operators/quant_fp4_mxfp]] — this page is the *why/strategy*
> above it. HW: [[hardware/cdna4_mi350]].

## Why a shared block scale
FP4 E2M1 has **15 levels** (max ±6) — hopeless for real activations/weights with any dynamic range. A
single per-tensor scale wastes range on outliers. MX fixes this by giving **every 32 consecutive
elements** their own scale: each block self-normalizes, so quiet blocks keep precision while loud blocks
don't clip. This is the 4-bit analog of per-token/per-block FP8 ([[scaling_strategies.md]],
[[operators/quant_dequant_fp8]]).

## The E8M0 block scale
- **8 bits, exponent only, no sign, no mantissa.** Value = `2^(E−127)` — literally an FP32 biased
  exponent. `E=127` ⇒ ×1; `E=255` reserved NaN; range `2^-127 … 2^127`.
- **One scale per 32 elements** — group size **32 is fixed by the OCP MX spec; do not tune it.**
- Effective cost ≈ element_bits + 8/32 = element_bits + **0.25** b/elt (MXFP4 ≈ 4.25, MXFP6 ≈ 6.25).
- Because the scale is power-of-2 only, scaling is an exponent add — no per-block multiply hardware
  needed for the scale itself.

## Scale selection: power-of-2 amax (strategy)
Per block (detail in [[operators/quant_fp4_mxfp]], aiter `per_1x32_f4_quant`):
1. `block_amax = max(|x|)` over the 32 elements.
2. Use the **power-of-2** element max, e.g. `FP4_MAX = 2^floor(log2(6)) = 4.0` — not 6 — because E8M0
   scales are powers of two and the largest element must land on the top representable power.
3. `s = f32_to_e8m0(block_amax / element_max_pow2)`; encode `x/2^(s−127)` into the element format.
The OCP rule: choose the scale so the block's max element reaches the **max element-type exponent**,
maximizing range without overflow. Quark exposes an **`even_round`** block-scale rounding mode.

## Scale layout & shuffle (hardware correctness)
The block-scaled MFMA (`v_mfma_scale_f32_32x32x64_f8f6f4`, builtin
`__builtin_amdgcn_mfma_scale_f32_*_f8f6f4`) expects E8M0 scales in **specific operand slots** (Ax 32×2,
Bx 2×32 for the 32×32×64 shape). aiter `shuffle=True` (`e8m0_shuffle`) rearranges + pads the scale
tensor (`((m+255)//256*256, ((n+31)//32+7)//8*8)`) to match. **Wrong shuffle = silent corruption.**
Verify with `amd_matrix_instruction_calculator --get-register --Ax/--Bx`. Full detail:
[[operators/quant_fp4_mxfp]], [[hardware/cdna4_mi350]], [[operators/scaled_quant_gemm]].

## Block-scaled MFMA element type codes
The scaled MFMA takes a per-operand type code, enabling **mixed precision** (e.g. FP4 weights × FP6/FP8
activations on independent A/B): `0 = E4M3 (fp8)`, `1 = E5M2 (bf8)`, `2 = E2M3 (fp6)`, `3 = E3M2 (bf6)`,
`4 = E2M1 (fp4)`. Pick A/B types from the accuracy gate ([[accuracy_evaluation.md]]).

## FP6 runs at the FP4 rate (the key throughput fact)
On CDNA4, **FP6 and FP4 share the same data path** → both peak at the FP4 rate (~10 PF with sparsity /
~9.2 PF dense, [[hardware_support_matrix.md]]). So **FP6 is "free" extra accuracy**: same FLOPs as FP4,
more mantissa/range. (FP6 may run marginally slower in practice due to power limits.) This differs from
NVIDIA Blackwell, where FP6 is rated at the FP8 rate. Consequence for strategy:
- Default to **MXFP6** (or mixed MXFP4/MXFP6) over plain MXFP4 unless MXFP4 already passes the gate —
  you pay nothing in throughput.
- AMD's "near-lossless MXFP4" recipe layers **rotations (QuaRot/Hadamard) + SmoothQuant** before the
  cast ([[calibration_and_quark.md]], [[operators/quant_fp4_mxfp]]).

## CDNA3 vs CDNA4
| | CDNA3 (gfx942) | CDNA4 (gfx950) |
|---|---|---|
| FP4/FP6 MFMA | ✗ no HW | ✓ (~10 PF) |
| block-scaled MFMA (E8M0) | ✗ | ✓ `v_mfma_scale_*` |
| MXFP4 inference | software sim (dequant-on-the-fly to fp16) | native |
On MI300 you can *simulate* MXFP for accuracy validation (Quark on-the-fly dequant) but get **no
throughput win**; value is footprint + forward-compat. `VLLM_ROCM_USE_AITER_FP4BMM=0` on gfx942 (FP4BMM
crashes — no HW).

## Pitfalls
- **Group size ≠ 32** — spec-fixed; `dynamic_mxfp4_quant` literally comments "Do not tune this."
- **Wrong E8M0 shuffle/layout** → silent corruption (run the calculator).
- **Per-tensor FP4** — collapses; MX block scaling is mandatory.
- **Choosing FP4 over FP6 "for speed"** — same rate; FP6 just adds accuracy.
- **Assuming MX accelerates on CDNA3** — simulation only.

## Verify
- Round-trip weights through MXFP4/6; per-block error + end-task accuracy vs FP16
  ([[accuracy_evaluation.md]]).
- `amd_matrix_instruction_calculator --architecture cdna4 --instruction
  v_mfma_scale_f32_32x32x64_f8f6f4 --get-register --Ax/--Bx` before wiring scales.

## Sources
- OCP MX spec (group 32, E8M0, scale selection): https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
- MX paper: https://arxiv.org/pdf/2310.10537
- block-scaled MFMA, type codes, FP6@FP4 rate, CDNA3 vs CDNA4: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
- MFMA scale-instruction syntax/operands: https://salykova.github.io/matrix-cores-cdna
- per-block algorithm, shuffle/pad, even_round: [[operators/quant_fp4_mxfp]] (`ROCm/aiter@a6bb49937`).
