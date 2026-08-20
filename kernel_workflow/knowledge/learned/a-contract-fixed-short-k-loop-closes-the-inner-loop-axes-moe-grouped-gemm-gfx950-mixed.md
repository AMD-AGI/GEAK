---
key: CK two-stage fp8 block-scale MoE on gfx950 once tile and occupancy wins have landed — which further rounds inside the K loop returned nothing
type: anti-pattern
confidence: ★★
effect: ~1.00x across eight directions: seven schedule variants plus an accumulator split ~+0.4% (inside A/A drift); legal barrier removal 0% (an illegal zero-barrier positive control was itself slower); LDS double buffering -0.53% at 32768 / -0.34% at 65536 tokens; whole-op graph capture 1.011 / 0.996 / 1.003 per case; replacing the atomic combine +2.8% at 65536, nil at 32768, -2.7% at 2048; single-kernel fusion priced at ~+0.6% geomean
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-12
name: a-contract-fixed-short-k-loop-closes-the-inner-loop-axes-moe-grouped-gemm-gfx950-mixed
description: With K trips pinned by the operator contract and full-rate MFMA already emitted, inner-loop levers all returned ~1.00x; the wins sat in mapping and routing.
keywords: ['anti-pattern', 'closed-axis', 'instruction-schedule', 'double-buffering', 'atomic-combine', 'moe', 'grouped-gemm', 'fp8-blockscale']
kernels: ['fmoe_fp8_blockscale_g1u1']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: mixed
layer: learned
lifecycle: active
cost: L1
verified_on: 2026-08-09
origin_kernels: ['moe_gemm_fp8_blockscale']
---
# A contract-fixed short K loop closes the inner-loop axes
- lever: price the loop before funding a round inside it: when trip count is fixed by the operator contract and the arch's full-rate MFMA is already emitted, reordering, deleting, retiling, deeper prefetch and cross-stage fusion each measured ~1.00x, while grid-to-cache mapping and host routing still paid.
- apply: build the cheap bounding controls first — an exact-copy control, an empty scheduler, an illegal zero-barrier variant — to prove the knob is live and to cap the prize before writing the real candidate.
- verify: read the emitted ISA for the full-rate instruction and the arch macro before assuming an MFMA-rate gap, and normalize busy counters (per-SIMD vs per-resident-wave) before believing an occupancy figure; prefetch depth is capped by the register file, so check the wave/SIMD count after any depth change.
- pitfall: seven candidates were wrong at full speed and passed a speed-only A/B -> clamped fp8 conversion against an OCP-range tensor, a non-bijective remap, MFMA-16 against a shuffled operand, a pre-activation routed-weight multiply, accumulator reassociation, an out-of-range N extent returning NaN with no static assert -> SNR-gate every instance/layout arm.
an A/A control inherited the previous arm's knobs and turned a loss into a fake win -> a runtime wrapper with a compile-time trip count of 1 also cost ~15% on every instance -> make A/A a harness invariant and rebuild/install explicitly after header edits.
- caution: also verify your controls are inert: hold one cell constant across several arms so the collisions give a free reproducibility floor for the cell you are not testing.
- source: run kernel_20_geak_0808_16h, 2026-08-08..09, gfx950/MI355X; 13 rounds, 30 direction-credits, director-validated
