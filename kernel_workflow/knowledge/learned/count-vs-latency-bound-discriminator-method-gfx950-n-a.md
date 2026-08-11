---
key: distinguishing an issue-throughput ceiling from a dependency-latency / register-pressure ceiling in a quantized-dequant GEMM inner loop on gfx950
type: instrument
confidence: ★★
effect: the discriminator fired on both large-M cases: VALU op count fell 1168 -> 1093 and converts 226 -> 98 in the bottleneck loop, yet latency rose +7% on both, small-M case neutral; the paired probes agreed - forcing occupancy 3 made the mid case ~4% worse, and graph replay came in at 0.985x with host enqueue under ~1/13 of per-call GPU time
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 3
toolchain: unknown
last_seen: 2026-08-11
name: count-vs-latency-bound-discriminator-method-gfx950-n-a
description: Three cheap probes separate issue-throughput from dependency-latency/VGPR ceilings in a quantized-dequant GEMM inner loop before spending a round.
keywords: ['instrument', 'roofline', 'valu-bound', 'dependency-latency', 'occupancy', 'launch-overhead', 'vgpr-pressure', 'int4-dequant']
kernels: ['fused_moe_int4_w4a16']
platforms: ['gfx950']
kernel_class: method
regime: n/a
layer: learned
lifecycle: active
---
# count-vs-latency-bound-discriminator
- lever: Before spending a round on reducing dequant instruction count, run the count-vs-latency pair: instruction counts on the bottleneck loop and the isolated latency together. Count down with latency up identifies a dependency-latency plus register-pressure ceiling, where any issue-count reduction is unlikely to pay.
- apply: Three L0/L1 probes - static VALU/convert counts from the compiled body, a forced-higher-occupancy build, and a host-enqueue vs GPU-time split taken from the wrapper.
- verify: Read them as ratios: the enqueue/GPU-time split settles the dispatch question, the forced-occupancy build settles register pressure, the instruction-count diff settles issue throughput; only if all three come back negative is the ceiling a latency wall.
- pitfall: A warm-up outlier on the first repeat of the largest case inflated run-to-run spread -> take the median of at least three full benchmark repeats and drop the first -> without that, a neutral direction reads as a win.
- caution: Also verify the counts came from the loop that dominates rather than the whole module - a module-wide count can fall while the bottleneck loop stays flat.
- source: 16h per-kernel time-budget campaign, instruction-count and occupancy probes across 3 directions, 2026-08-11
