---
key: fp8 w8a8 block-scale MoE grouped GEMM (up-projection) in Triton on gfx950/MI355X, where the caller pins a small BLOCK_M and a per-k-group A scale
type: lever
confidence: ★★
effect: 4.95x geomean vs the frozen baseline (director-verified, non-overlapping, correctness pass); per-case 3.09x on the smallest case and 6.33x / 6.22x on the two large ones; 4.53x against a same-day re-measured baseline
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 4
toolchain: unknown
last_seen: 2026-08-12
name: derive-the-tile-then-renegotiate-the-scale-contract-moe-grouped-gemm-gfx950-prefill
description: fp8 block-scale MoE grouped GEMM, gfx950: derive the consumer tile in-file and renegotiate the A-scale contract per-row -> ~4.95x geomean.
keywords: ['moe-grouped-gemm', 'fp8-block-scale', 'gfx950', 'super-tile', 'occupancy', 'quantization-contract', 'xcd-remap', 'async-copy', 'paired-ab-rig']
kernels: ['fused_moe_kernel']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: prefill
layer: learned
lifecycle: active
cost: L3
verified_on: 2026-08-12
roofline: memory / B-tile-traffic bound before -> MFMA/global-load interlock after, at ~54% of the achievable fp8 MFMA peak
origin_kernels: ['fused_moe_kernel']
---
# Derive the tile, then renegotiate the scale contract
- lever: A small BLOCK_M fixed by the caller is often a producer-side data-layout constraint, not a consumer tile constraint: let the file derive its own tile. Pair it with sizing num_warps from accumulator elements per lane instead of maximising occupancy, and with renegotiating the fp8 A-scale granularity so the in-loop rescale disappears.
- apply: Export the launch symbol as an object whose __getitem__(grid) picks an adaptive BLOCK_M (<=256 rows) floored to grid >= 4 blocks/CU and derives num_warps from 64 acc-elems/lane; switch the A scale from per-128-k-group to per-row, which deletes the in-loop rescale and unpins BN/BK (BN reaches 256); fold the row-scale in a separate memory-bound pre-pass that writes A' once and is read ~16x.
- stack: total 4.95x geomean per-case (director-verified) = four directions compounded
  - 1. self-deriving super-tile launcher - 2.59x standalone (VMEM-per-MFMA 8.05 -> 1.16)
  - 2. num_warps from acc-elems/lane + rescale collapse - 1.83x standalone; reconciling (1)+(2), which collide on the same launcher/num_warps knob, gave 3.59x, +38.7% over the better single arm
  - 3. per-row A-scale contract - 3.59x -> 4.70x, the largest single lever
  - 4. wide-tile-gated async-copy + XCD pid remap +3.13%, then a K_EXACT constexpr eliding a compile-time-dead masked remainder (vgpr 254/spill 12 -> 249/0) +0.08%
  - note: attribution is incremental in landing order, not independent.
- verify: Confirm the derived tile engaged by md5-ing the compiled artifact per config under isolated per-config cache dirs, then re-time every case against the frozen baseline; for any expected delta under ~1%, use interleaved same-session paired arms with a planted known-null arm in the sweep.
- pitfall: A cross-round win and a <1% integration delta disagreed -> cross-process median-of-3 drifts on this box (one planted null caught a ~0.6% phantom) -> an in-process paired rig settled it, combined arm winning the largest case 4/4.
Per-row fp8 scaling passes the harness gate but shifts numerics (cosine 0.999300, rel RMS 3.74%, max_rel ~8-9x) -> non-averaging e4m3 reduction over K=7168 -> ship it flagged and re-qualify under any tighter max_rel gate.
- caution: Also verify where each case's ceiling actually is before budgeting rounds: the smallest case here sat at its compulsory-miss memory roof at ~3.1x while the two large cases reached ~6.2-6.3x, so it capped the geomean and further work on it returned nothing.
- source: run fused_moe_kernel-own16h, 2026-08-12, director-validated (geomean 4.9568 director vs 4.9518 tech-lead)
