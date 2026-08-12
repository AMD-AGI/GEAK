---
key: block-scaled fp8 A8W8 linear in Triton on gfx950/CDNA4, where the tensors carry the previous arch's fp8 flavour
type: lever
confidence: ★★
effect: 7.83x isolated vs frozen baseline, non-overlapping; per-case 6.15x at M=2048, 8.81x at M=32768, 9.04x at M=65536 (N=4096, K=3072), no case regressed
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-12
name: reinterpret-legacy-fp8-bits-to-the-arch-native-fp8-type-to-r-quantized-gemm-gfx950-compute-bound
description: Legacy-flavour fp8 operands get silently emulated in fp16 on CDNA4; a zero-copy bit reinterpretation to the native fp8 type engages the matrix core, ~7.8x
keywords: ['fp8', 'mfma', 'quantized-gemm', 'bit-reinterpret', 'emulation-fallback', 'isa-census', 'block-scale']
kernels: ['_gemm_a8w8_blockscale_kernel']
platforms: ['gfx950']
kernel_class: quantized_gemm
regime: compute-bound
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-08-12
---
# Reinterpret legacy fp8 bits to the arch-native fp8 type to reach the matrix core
- lever: When a quantized GEMM's operands carry the previous arch's fp8 flavour, the compiler may emulate the format (per-element upcast to fp16, then the fp16 MFMA) and never touch the native fp8 matrix core; a zero-copy bit reinterpretation to the native type is worth trying before any tile or occupancy work.
- apply: view() both operands to the native fp8 dtype behind a guard (source dtype is the legacy flavour AND last dim unit-stride), drop input_precision='ieee' on the dot so it lowers to the native fp8 path, and fold one constant fixup for the exponent-bias remap into the epilogue; no extra memory pass, no extra launch, no per-element branch.
- verify: ISA census must show the native fp8 mfma opcode present and the fp16 mfma opcode at zero, with the emulation cmp/cndmask population collapsing by two orders of magnitude; parity gate unchanged (cos ~0.999999, err_ratio 0).
- pitfall: parity failed at cos ~0.12 -> the fixup constant was inverted: the native read is 2x the legacy value per operand, so the dot comes out 4x large and the fixup is 0.25 rather than 4 -> cos ~0.12 is the self-diagnosing signature of a 16x scale error.
- caution: also verify the shipped tuning configs afterwards: the native instruction has a different K depth, so tile / nonkdim / num_stages were tuned for an instruction that no longer exists, and a re-sweep paid a further ~1.32x here; and also verify the quantizer cannot emit the legacy NaN/edge encodings, which alias under the remap.
- source: run gemm_a8w8_blockscale-own16h, 2026-08-12, round 1 direction d0, director-validated
