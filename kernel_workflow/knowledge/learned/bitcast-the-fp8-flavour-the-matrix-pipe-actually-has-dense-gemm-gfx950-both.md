---
key: block-scaled fp8 dense GEMM on gfx950/CDNA4 under Triton, where the operand fp8 flavour has no native MFMA and the compiler silently emulates it per element
type: lever
confidence: ★★
effect: ~12x isolated standalone vs frozen baseline, non-overlapping, on all three cases (M=2048/32768/65536, N=2624, K=6144); ~54% of the campaign's 22x total; lifts the op from ~1% to ~28% of the arch's dense-fp8 roof
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 2
toolchain: unknown
last_seen: 2026-08-12
name: bitcast-the-fp8-flavour-the-matrix-pipe-actually-has-dense-gemm-gfx950-both
description: Bitcast a non-ISA fp8 operand type to the native one so real MFMA issues instead of per-element emulation: ~12x alone on block-scaled fp8 GEMM
keywords: ['fp8', 'mfma', 'dtype-bitcast', 'dense-gemm', 'emulation-fallback', 'block-scale', 'triton', 'gfx950']
kernels: ['_gemm_a8w8_blockscale_kernel']
platforms: ['gfx950']
kernel_class: dense_gemm
regime: both
layer: learned
lifecycle: active
cost: L3
verified_on: 2026-08-12
origin_kernels: ['_gemm_a8w8_blockscale_kernel']
---
# Bitcast the fp8 flavour the matrix pipe actually has
- lever: when an operand dtype is an fp8 variant the ISA has no MFMA for, bitcast both operands to the native fp8 type and fold the constant exponent-bias ratio into the epilogue scale
- apply: reinterpret the loaded tiles at the dot boundary only; multiply the epilogue scale by the bias ratio (0.25x between these two flavours); host wrapper and layouts unchanged
- verify: dump the ISA and confirm the wide f8f6f4 MFMA opcode issues with zero per-element compare/upconvert fixups, then frozen-baseline A/B plus oracle parity
- pitfall: verified score came back 0 while the standalone measurement showed a real ~12x -> the diff was captured in a workspace with no git root and recorded the harness repo instead of the source tree -> re-scope the patch to the source file and diff against canonical before claiming
- caution: also verify numerics against the oracle rather than bit-equality: the bias fold changes rounding, so confirm the max relative error sits at the output dtype's own floor
- source: run _gemm_a8w8_blockscale_kernel-own16h, 2026-08-12, rounds 1-2
