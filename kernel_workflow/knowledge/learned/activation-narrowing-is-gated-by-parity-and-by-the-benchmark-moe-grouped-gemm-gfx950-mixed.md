---
key: narrowing the ACTIVATION operand of a grouped GEMM that already stores weights in fp4, gfx950 / Triton, judged by a cosine parity gate on synthetic inputs
type: anti-pattern
confidence: ★★
effect: closed axis: 1.00x banked across three activation-narrowing directions on all cases; the only working variant was ~+13% on the cumulative chain but landed at cosine 0.9883 against a 0.99 gate and stayed off; the outlier-preserving variant ceilinged at ~+5%
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 3
toolchain: unknown
last_seen: 2026-08-12
name: activation-narrowing-is-gated-by-parity-and-by-the-benchmark-moe-grouped-gemm-gfx950-mixed
description: Weight-side fp4 pays; activation-side narrowing closed three ways - parity floor, missing fp6 lowering, iid synthetic inputs with no outliers
keywords: ['moe', 'grouped-gemm', 'fp4', 'fp6', 'weight-quantization', 'parity-gate', 'anti-pattern', 'dot-scaled']
kernels: ['fused_moe_kernel']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: mixed
layer: learned
lifecycle: active
cost: L3
verified_on: 2026-07-30
---
# Activation narrowing is gated by parity and by the benchmark's input distribution
- lever: Narrow the operand that dominates traffic (weights, when they are streamed) and treat the activation side as a separate, parity-gated question rather than a symmetric follow-up; its payoff is small because it is the minority traffic.
- apply: Before proposing mixed-precision or outlier-preserving activation formats, check what the harness feeds: iid Gaussian activations have no channel outliers, so an outlier-preserving split degenerates to picking random channels.
- verify: Re-run the cosine parity gate per case at the intended format, and record the achieved margin - a variant a hair under the gate is worth caching behind a flag so it flips instantly if the gate is ever relaxed or the data changes.
- pitfall: e3m2/e2m3 activations were accepted in the plan and rejected at compile -> the Triton frontend of that vintage refuses those formats and the arch has no fp6 MFMA lowering -> confirm both frontend acceptance and a real hardware lowering path before budgeting a round on a format.
Before proposing a cheaper software dequant, confirm the format is software-unpacked at all: consumed natively by MFMA there is nothing to cheapen.
- caution: Also verify whether the parity floor is intrinsic to the format or to the tested shapes - the same variant may clear the gate on real activations with different dynamic range, so cache it rather than discarding the patch.
- source: run fused_moe_kernel-ch16h, 16h per-kernel time-budget campaign, 2026-07-30
