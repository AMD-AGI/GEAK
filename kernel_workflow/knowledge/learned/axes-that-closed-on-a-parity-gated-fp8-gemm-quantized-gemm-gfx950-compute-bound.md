---
key: closed optimization axes on a parity-gated, VALU-dequant fp8 block-scaled GEMM on gfx950 already sitting at high cumulative speedup
type: anti-pattern
confidence: ★★
effect: five directions disconfirmed: manual SW pipelining ~0.90x (loads only) and ~0.83x (loads+convert) per-case; no-transpose B reshape ~0.80x; host graph replay <1.00x on every case; +1 wave of occupancy unreachable; reduction reassociation fails parity outright
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 5
toolchain: unknown
last_seen: 2026-08-12
name: axes-that-closed-on-a-parity-gated-fp8-gemm-quantized-gemm-gfx950-compute-bound
description: Five directions returned ~1.00x or worse on a parity-gated fp8 GEMM: occupancy, manual SW pipelining, B-side LDS reshape, graph replay, reduction reassociation.
keywords: ['fp8', 'block-scaled-gemm', 'bit-exact-gate', 'anti-pattern', 'occupancy', 'software-pipelining', 'lds-staging', 'gfx950']
kernels: ['_w8a8_triton_block_scaled_mm']
platforms: ['gfx950']
kernel_class: quantized_gemm
regime: compute-bound
layer: learned
lifecycle: archived
cost: L2
verified_on: 2026-08-12
---
# Axes that closed on a parity-gated fp8 GEMM
- lever: Spend rounds on order-preserving compute restructuring first; these five axes were each measured to ~1.00x or worse here, so they are low-yield seeds on a similar op.
- apply: Reassociating the fp32 reduction (half-split, even/odd, compensated summation) and native low-precision MFMA both change accumulation order and fail a max_rel gate on cancellation-heavy elements; the rest fail on performance, not parity.
- verify: Before re-opening one, read the ISA: if occupancy is unchanged while throughput moves, the op is issue/dep-chain-bound and adding waves cannot help; if a restructure raises VGPRs past the occupancy ceiling with no spill, the loss is purely the occupancy step.
- pitfall: Hand-hoisting the next K block's load behind the MFMA drain looked free -> the auto-pipeliner at two stages already prefetched those loads, and the extra live state crossed the register ceiling -> reverted to the compiler's schedule.
- caution: These closed on one shape family and one compiler version; also verify with a quick ISA diff before assuming the same axis is closed on your op, especially if your parity gate is looser.
- source: run _w8a8_triton_block_scaled_mm-ch16h, 2026-08-12, ledger of 5 dead-end directions across 4 rounds
