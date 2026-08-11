---
name: split-the-launch-config-per-m-bucket-on-the-host-before-rewr-moe-grouped-gemm-gfx950-compute-bound
description: Bucket cases by M and fit the host launch config per bucket before touching a dequant-bound body: 3.26x on its own, and no body rewrite beat it
keywords: [launch-config, config-sweep, moe, dequant, tile-shape, compute-bound, roofline, large-batch]
kernels: [fused_moe_kernel_gptq_awq]
platforms: [gfx950]
kernel_class: moe_grouped_gemm
regime: compute-bound
key: tuning the host-side launch/meta config separately per M-bucket on a weight-dequant-bound Triton MoE grouped GEMM (gfx950) instead of rewriting the body
lifecycle: active
type: lever
confidence: ★★
effect: The per-M-bucket host-side launch-config retune measured 3.2631x on its own and carried essentially the whole campaign: best accepted pass 3.3977x (reported 3.40x) over 49 passes, the one director-verified pass reading 3.3849x geomean, state cumulative 3.3326x, with the kernel source left byte-identical to the golden baseline. Per-case 2.584x on the smallest batch and 3.6848x / 3.8871x on the two larger ones; the buckets did not agree on one config (small kept GROUP_SIZE_M=8, the two large ones GROUP_SIZE_M=4). Three deep body rewrites over the same window measured 3.2579x, 3.3326x and (hand-scheduled MFMA) 7-13x SLOWER, i.e. none beat the host-only incumbent. Empirical roofline fraction 0.260 -> 0.980.
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 5
toolchain: rocm 7.x / triton 3.6.0 / torch 2.11.0
source: chuschen 16h time-budget campaign run, 15.72h / 49 passes, 2026-08-11
last_seen: 2026-08-11
---
# Split the launch config per M-bucket on the host before rewriting a dequant-bound body
- lever: When the per-output-element work is dominated by weight unpack/dequant, that cost is M-independent while tile occupancy and grid fill are not, so one launch config fitted across the whole batch sweep is leaving most of the win on the table.
- apply: Before opening the kernel body, bucket the cases by M and tune the launch/meta config separately in each bucket, keeping distinct winners rather than collapsing to a compromise; two independently-fitted host-side config patches stacked cleanly here because neither touched the body.
- verify: Two signals that the body is the wrong place to spend the round — per-call enqueue far below device time (~13.6x below here, and a wrapper CUDAGraph then measured geomean 0.985), and a tile count already saturating the CUs (~3072 tiles over 256 CUs, which made genuine split-K a net loss in every bucket).
- pitfall: three deep body rewrites over the same window all failed to beat a host-only config patch (the hand-scheduled MFMA variant by 7-13x) -> the op was dequant-bound with the CUs already saturated, so body scheduling had nothing to buy -> price the host config per bucket first and treat it as the incumbent every body rewrite has to beat.
- caution: Also verify the buckets really disagree before keeping separate configs — here they did (GROUP_SIZE_M=8 on the smallest batch, 4 on the two large ones), and a single compromise config would have hidden that.
- source: chuschen 16h time-budget campaign run, 15.72h / 49 passes, 2026-08-11
