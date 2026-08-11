---
name: discount-the-nameplate-relative-roofline-against-the-cu-part-quantized-gemm-gfx950-compute-bound
description: Rescale peak by the CU partition the box exposes before funding a headroom chase: an apparent 33% of peak was ~72% on the large fp8 GEMM cases
keywords: [roofline, compute-bound, fp8, quant, measurement-method, harness-artifact]
kernels: [_gemm_a8w8_blockscale_kernel]
platforms: [gfx950]
kernel_class: quantized_gemm
regime: compute-bound
key: sizing remaining headroom on an fp8 / a8w8 quantized GEMM on a CU-partitioned gfx950 part, where the harness quotes peak from the full-part nameplate
lifecycle: active
type: anti-pattern
confidence: ★★
effect: Three deep_explore rounds entered with expectations of 1.06x, 1.30x and 1.28x against an apparent '33% of MFMA peak'; all three returned actual 0 with an empty patch, and the ceiling was re-confirmed six times independently. The two large cases in fact sit at ~72% of achievable peak: the device query reports a 118-CU partition of a 256-CU part, so realizable peak is ~46% of the nameplate figure the harness divided by, leaving those cases ~4% off the pure-fp8-GEMM floor rather than 33% with 67% of the roof still to take.
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 3
toolchain: rocm 7.x / triton 3.6.0 / torch 2.11.0
source: chuschen 16h time-budget campaign, 2026-08-11
last_seen: 2026-08-11
---
# Discount the nameplate-relative roofline against the CU partition the box exposes before funding a headroom chase
- lever: Before spending a round on a gap between measured throughput and peak, rescale peak by the partition the run actually has: read the CU count off the device query and multiply the full-part nameplate by that fraction. A number that looks like a third of peak can be three quarters of what the partition can do, and the rounds funded by the phantom two thirds come back with empty patches.
- apply: Recompute the fraction-of-achievable-peak yourself and re-price the round against it; the same check is worth applying to a roofline-emp figure quoted by any harness that derives peak from a part name.
- verify: Compare the recomputed figure with the dtype's own GEMM floor — when it lands within a few percent of that floor, the honest reading is that the axis is finished and the round is better spent on a different one.
- pitfall: three rounds budgeted at 1.06-1.30x all returned exactly 0 with an empty patch -> the headroom they were funded by was an artifact of dividing by the full-part nameplate on a partitioned device -> discount peak by CU fraction before pricing the round.
- caution: Also verify the partition the run actually holds rather than trusting what the part is named — a fraction-of-peak figure is only valid for the partition it was measured on, so re-derive it whenever the CU count changes.
- source: chuschen 16h time-budget campaign, 2026-08-11
