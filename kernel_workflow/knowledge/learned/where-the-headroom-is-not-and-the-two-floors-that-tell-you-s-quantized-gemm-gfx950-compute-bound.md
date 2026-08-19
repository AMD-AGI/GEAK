---
key: a register-limited scale-applying fp8 linear on gfx950 already close to the on-box vendor-library floor
type: anti-pattern
confidence: ★★
effect: seven directions measured at or below parity on all cases (M=2048/32768/65536): barrier removal ~0.95x, num_warps=8 0.634x, >=256 tiles 0.897x, a Gluon rewrite 0.84x, split-K and VALU-count and epilogue rework each ~1.00x
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 7
toolchain: unknown
last_seen: 2026-08-12
name: where-the-headroom-is-not-and-the-two-floors-that-tell-you-s-quantized-gemm-gfx950-compute-bound
description: Occupancy, geometry, split-K, barriers, VALU count, epilogue and a Gluon rewrite all returned <=1.00x; price the library floor vs a scale-free floor first
keywords: ['anti-pattern', 'closed-axis', 'roofline', 'split-k', 'occupancy', 'quantized-gemm', 'isa-census', 'block-scale']
kernels: ['_gemm_a8w8_blockscale_kernel']
platforms: ['gfx950']
kernel_class: quantized_gemm
regime: compute-bound
layer: learned
lifecycle: active
cost: L1
verified_on: 2026-08-12
origin_kernels: ['gemm_a8w8_blockscale']
---
# Where the headroom is not, and the two floors that tell you so
- lever: Before funding loop-schedule rounds, measure two floors on the same box: the vendor library at your shape, and the best implementation in your own language with the quantization/scale machinery deleted. When the library sits below your language's scale-free floor, the in-language axis set is closed and the remaining rounds are better spent on correctness or on leaving the language.
- apply: Build the scale-free variant as a throwaway measurement artifact (numerics wrong on purpose, parity off) purely to price the language floor, then decompose the current time as language-floor + exact-scale-apply + slack, in ratios of the case total.
- verify: Each closed axis earns its own disconfirming measurement rather than an argument -- removing the per-iteration barriers succeeded (count cut 4x at zero spills) and measured slower; occupancy-limited claims need the register census plus the clock, since split-K raises total CTAs and not concurrent ones when occupancy is register-limited.
- pitfall: two arms with byte-identical compile metadata (VGPR/LDS/spill/opcode census) sat ~12% apart in wall clock, reproduced five times -> the census is blind to addressing and to VMEM issue order -> treat metadata as a rejector only, and re-derive every knob verdict after a tile or accumulator-shape change (five verdicts went void that way).
- caution: also verify a control that closed an axis was closed for the right reason: one control here was VALU-identical rather than scale-free, so the same physical quantity was quoted three different ways across rounds and one axis was re-opened after being declared dead.
- source: run gemm_a8w8_blockscale-own16h, 2026-08-12, rounds 4-13 dead-end table, director-validated
