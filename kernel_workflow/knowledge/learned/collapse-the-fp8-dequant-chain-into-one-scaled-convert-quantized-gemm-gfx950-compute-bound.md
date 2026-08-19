---
key: VALU-emulated fp8 dequant feeding an f16 MFMA GEMM on gfx950, where native fp8 MFMA is closed off by a bit-exact parity gate
type: lever
confidence: ★★
effect: cumulative 8.98x then 9.17x vs the frozen baseline (+1.5-1.8% for the single-convert step alone), bit-exact; per-case the transposed-B staging carried +15% on the two larger shapes and ~0 on the smallest
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 2
toolchain: unknown
last_seen: 2026-08-12
name: collapse-the-fp8-dequant-chain-into-one-scaled-convert-quantized-gemm-gfx950-compute-bound
description: Fold the fp8 format-recovery constant into the scaled-convert scale operand and lift sign from the byte's sign bit: one convert per pair, shorter dep chain.
keywords: ['fp8', 'block-scaled-gemm', 'bit-exact-gate', 'critical-path', 'dequant', 'packed-loads', 'lds-staging', 'gfx950']
kernels: ['_w8a8_triton_block_scaled_mm']
platforms: ['gfx950']
kernel_class: quantized_gemm
regime: compute-bound
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-08-12
origin_kernels: ['_w8a8_triton_block_scaled_mm']
---
# Collapse the fp8 dequant chain into one scaled convert
- lever: Fold the exact power-of-two format-recovery factor into the fp32 scale operand of the native scaled-convert, derive magnitude from that single convert and the sign from the byte's sign bit, and feed it with int32-packed 4-bytes-per-dword loads.
- apply: Retires the separate multiply stream, the second convert and the abs; also stage the B operand by loading it transposed into LDS and transposing back, which trades most global loads for LDS reads.
- verify: Parity first (the folded constant has to be exact in the convert's output format), then the frozen isolated A/B; grep the ISA for the convert / and / multiply counts to confirm the stream actually disappeared.
- pitfall: Cutting op count alone measured ~0 -> the kernel was dependency-chain-bound, not issue-bound at the margin -> the paying version was the one that shortened the chain; occupancy dropped one wave here while throughput rose ~20%.
- caution: Also verify the recovery constant is representable exactly in the convert's format before folding, and re-check parity on near-cancellation elements.
- source: run _w8a8_triton_block_scaled_mm-ch16h, 2026-08-12, rounds 1-2 of the final wave
