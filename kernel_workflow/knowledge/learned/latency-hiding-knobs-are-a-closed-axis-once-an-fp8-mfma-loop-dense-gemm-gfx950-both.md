---
key: scale-free fp8 MFMA GEMM loop on gfx950/MI355X whose residual stall is the LDS-read-to-MFMA dependency, Triton
type: anti-pattern
confidence: ★★
effect: Disconfirming, on top of the 20.1x champion and on all three cases: deeper pipelining (stages=3) 1.0x and only fits shared memory at half the N tile, which halves arithmetic intensity; a 256x64 tile 0.64x; mfma non-K dim 16 -> 32 0.916x; VGPR shave for a third wave 1.0x; a hand-scheduled register-resident rewrite bypassing shared memory produced no accepted patch. Five directions, zero gains above the ~1% noise floor.
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 5
toolchain: triton-on-rocm
last_seen: 2026-08-11
name: latency-hiding-knobs-are-a-closed-axis-once-an-fp8-mfma-loop-dense-gemm-gfx950-both
description: On a scale-free fp8 MFMA loop, deeper pipelining, bigger tiles, wider mfma non-K dim and VGPR shaving all measured <=1.0x: a closed axis
keywords: ['fp8', 'dense-gemm', 'mfma', 'occupancy', 'lds-tiling', 'num-stages', 'vgpr-pressure', 'gfx950']
kernels: ['gemm_a8w8_blockscale', '_gemm_a8w8_blockscale_kernel']
platforms: ['gfx950']
kernel_class: dense_gemm
regime: both
layer: learned
lifecycle: active
---
# Latency-hiding knobs are a closed axis once an fp8 MFMA loop is VGPR and LDS walled
- lever: Treat occupancy/latency-hiding as a closed axis once the loop is accumulator-VGPR-gated and its shared-memory budget is saturated; spend rounds elsewhere. The residual here is the current-iteration shared-memory read feeding its own consuming MFMA, which cannot be prefetched because the data is written in the same pipeline stage.
- apply: Cheap pre-check before spending a round: read the per-lane VGPR count against the waves-per-SIMD ladder (an fp32 128x128 accumulator over 8 warps eats a third of the budget by itself), and check unroll x stages against the per-CU shared-memory limit — both walls here refuse to compile rather than run slow, and the accumulator dominates the VGPR count so shaving elsewhere cannot reach the next occupancy step.
- verify: Confirm the knob engaged at all (a config that overflows shared memory fails the build, which is a different outcome from a measured 1.0x), then re-time per case against the frozen baseline; a tile enlargement that also lowers arithmetic intensity should be scored on intensity, not on occupancy.
- pitfall: an enlarged tile looked like an occupancy win on paper but measured 0.64x -> the op was already compute/intensity bound rather than occupancy bound -> re-classify the bound before spending rounds on occupancy at all.
- caution: This held for a scale-free loop already at high arithmetic intensity; on a variant that is still memory-bound or that reintroduces scale feeding into the loop, also verify the pipelining knobs from scratch rather than reading this as settled.
- source: 16h per-kernel time-budget campaign, block-scaled fp8 dense-GEMM lane, ledger dead_end entries, 2026-08-11
