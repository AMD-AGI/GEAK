---
name: fnuz-fp8-operands-can-silently-disable-the-native-matrix-cor-quantized-gemm-gfx950-compute-bound
description: Bitcast fnuz fp8 dot operands to the part's native OCP e4m3 and fold the bias into an existing scale: ~9.2x standalone on a block-scaled A8W8 GEMM
keywords: [fp8, dtype-dialect, isa-check, valu-emulation, mfma, dequant, quantized-gemm, compute-bound]
kernels: [_gemm_a8w8_blockscale_kernel]
platforms: [gfx950]
kernel_class: quantized_gemm
regime: compute-bound
key: fnuz-vs-OCP e4m3 fp8 dot operands in a Triton block-scaled A8W8 GEMM on gfx950/CDNA4, tall-M compute-bound shapes
lifecycle: archived
type: lever
confidence: ★★
effect: 9.30 / 9.14 / 9.21x standalone at M=2k / 32k / 64k, and the precondition for the run's 23.45x director-verified geomean (25.6x at the smallest M, 24.8x mid, 20.3x at the largest M)
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 2
toolchain: triton 3.6.0 / torch 2.11.0+gitd0c8b1f / gfx950 CDNA4
last_seen: 2026-08-10
---
# fnuz fp8 operands can silently disable the native matrix core: bitcast to OCP and fold the bias
- lever: If the harness hands fp8 operands in a bias-8 'fnuz' dtype on a target whose matrix cores implement only OCP e4m3 (bias 7), the compiler may silently upcast both operands and run a software GEMM at a few percent of fp8 peak; bitcast both operands to the OCP type at the dot and fold the 2x exponent-bias difference (a factor 0.25 for a product of two operands) into a scale that is already being multiplied.
- apply: Bitcast at the dot operands inside the k loop, then scale by folding 0.25 into an existing per-tile scale multiply -- applying the identical factor as a separate accumulator op instead cost 2-3%; where the factor is k-invariant, hoisting it to the epilogue is bit-exact (it is an exact power of two) and paid a further ~1%.
- verify: Dump the assembly and look for the native fp8 matrix instruction plus the disappearance of the conversion VALU block (16593 static instructions fell to 1605 here), and re-run the correctness gate; a compile-time upcast UserWarning is the tell that the tax is being paid.
- caution: Also verify the quantizer's saturation behaviour before shipping this: the fnuz NaN code reinterprets as OCP -0 and the fnuz maximum magnitude reinterprets as OCP NaN, so operands that saturate to the fnuz max can become NaN even when a gate on typical inputs passes.
- source: run kernel_20_geak_0808_4h 2026-08-10
