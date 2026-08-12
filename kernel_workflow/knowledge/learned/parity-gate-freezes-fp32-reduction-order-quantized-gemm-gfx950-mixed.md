---
key: accumulation-order changes on an fp32 MFMA K-reduction judged against a golden f16-MFMA reference, gfx950 Triton GEMM
type: anti-pattern
confidence: ★★
effect: 0 of 4 accumulation-order variants cleared parity on any case: 2-way contiguous-half accumulator split max_rel=0.1146 at cos=1.0, even/odd split identical, compensated (Kahan) summation ~100% rel err, native fp8 MFMA max_rel=65 - while the order-PRESERVING regrouping of the same reduction passed at max_rel=0 and paid 1.16x
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 8
toolchain: unknown
last_seen: 2026-08-12
name: parity-gate-freezes-fp32-reduction-order-quantized-gemm-gfx950-mixed
description: Under a near-bit-exact parity gate the fp32 MFMA K-reduction order is structural: ILP splits, Kahan and native fp8 MFMA all fail parity, not perf
keywords: ['bit-exact', 'parity-gate', 'reduction-order', 'reassociation', 'mfma', 'fp8', 'quantized-gemm', 'gfx950', 'anti-pattern']
kernels: ['_w8a8_triton_block_scaled_mm']
platforms: ['gfx950']
kernel_class: quantized_gemm
regime: mixed
layer: learned
lifecycle: active
---
# parity-gate-freezes-fp32-reduction-order
- lever: When the correctness gate is cos>=0.99 AND max_rel<1e-2 with a small denominator clamp, treat the reference's fp32 accumulation ORDER as part of the spec: cancellation-to-near-zero elements make the relative-error term near-bit-exact, so any reassociation is a parity failure long before it is a perf question.
- apply: Spend the round on regrouping that preserves order (fewer, wider dots over the same linear accumulator) instead of splitting the accumulator; before authoring, run the parity gate on a one-line proof-of-order variant to learn the gate's true tightness cheaply.
- verify: Report max_rel alongside cosine - cosine stayed at 1.0 for a split that missed the gate by an order of magnitude, so a cosine-only check reads a parity failure as a pass.
- pitfall: pitfall: a hardware-native low-precision MFMA looked like free throughput -> it accumulates in a different fp32 order than the emulated reference -> it fails the same gate for a reason no tuning can remove.
- caution: This holds where the reference is itself an emulated fp32-accumulate reduction with a tight relative-error gate; also verify your gate's clamp and tolerance first - a looser or cosine-only gate reopens the whole axis.
- source: 16h single-kernel time-budget campaign, chuschen16h wave, rounds 3-4 rigorous-negative lanes, 2026-08-11
