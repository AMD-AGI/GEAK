---
key: fp16 row-wise softmax and similar row reduce-then-normalise ops on a 40-CU RDNA3.5 wave32 iGPU (gfx1151) with a shared LPDDR memory path, Triton, starting from a multi-pass baseline
type: lever
confidence: ★★
effect: 2.2132x geomean vs the frozen baseline, director re-measured and non-overlapping against a 0.4% box noise floor (in-run verify 2.2166x, engineer claim 2.2158x); per case 2.30x at rows x cols = 4096x2048, 1.98x at 4096x4096, 2.40x at 4096x8192, and faster than the vendor library on all three; correctness 14/14 at every shipped state
confirms_cited: 0
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-09-17
name: fuse-the-row-passes-first-then-key-num-warps-to-row-width-reduction-norm-gfx1151-memory-bound
description: Multi-pass fp16 row softmax on a 40-CU RDNA3.5 iGPU: fuse the passes to the 2x-traffic floor, then key num_warps to row width - 2.21x geomean
keywords: ['memory-bound', 'kernel-fusion', 'dispatch-collapse', 'num-warps', 'launch-config', 'triton', 'fp16', 'gfx1151', 'rdna', 'online-softmax', 'roofline', 'closed-axis', 'isa-inspection']
kernels: ['softmax_kernel']
platforms: ['gfx1151']
kernel_class: reduction_norm
regime: memory-bound
layer: learned
lifecycle: active
origin_kernels: ['softmax_kernel']
cost: L3
verified_on: 2026-09-17
roofline: memory-bound before and after, but the binding constraint moves: the multi-pass baseline moved 4x tensor bytes over 3 dispatches and was already at 96/101/83% of the correct wall on the bytes it actually moved (only ~25-40% of it on ideal bytes); the fused form moves the 2x narrow-dtype-in/out floor over 1 dispatch at 99.8% / 99.4% of the measured DRAM wall on the two streaming shapes and above the last-level-cache figure on the cache-resident one, with every case also beating a library copy of the same tensor
levers: ['algo.fusion', 'host.dispatch-collapse', 'host.launch-tuning']
---
# Fuse the row passes first, then key num_warps to row width
- lever: Split achieved bandwidth into ideal-bytes and actual-bytes BEFORE tuning anything: when the actual-byte number is already at the wall, per-byte efficiency is not the gap and the only lever is removing bytes - collapsing the separate row-max, exponent-sum and divide passes into one register-resident pass takes traffic to the dtype floor and buys almost exactly the traffic ratio, with the remainder from the dispatch collapse. Then recover what is left with a next_pow2(row width) -> (BLOCK, num_warps) launch table: the optimum moves monotonically with row width (4 warps at 2048 columns, 32 at 4096 and 8192) and one hardcoded value costs 9-40% on the widths it does not fit.
- apply: One program per row with BLOCK = next_pow2(cols) so the row stays in registers across the max, exponential and sum steps; convert to fp32 only inside the exponential and keep the resident row in the narrow dtype; keep a block-streamed online variant as the fallback above the register budget; put the shape -> (BLOCK, num_warps) map in the host wrapper, where it is substrate-independent and merges onto any kernel body. When a row fits inside one wave, launching at num_warps=1 makes both reductions intra-wave (no LDS, no barrier) - worth ~+1.8% on the cache-resident shape here and ~+5% at half that row width, so it grows as rows narrow.
- stack: total 2.2132x director-verified geomean = one landed direction carrying two separable mechanisms, attributed incrementally in landing order
  - 1. pass fusion, 3 dispatches -> 1, traffic 4x -> 2x tensor bytes - 2.1544x standalone (round 1, verified) - the bulk of the win, landing almost exactly on the 2.0x traffic ratio
  - 2. shape-keyed next_pow2 -> num_warps launch table on that same fused body - +2.9% on top of (1) (round 1, verified); it is what turned the widest shape from worst (83% of its wall) into best (99.4%)
  - note: a round-2 wave-per-row variant measured +1.8% on the one shape still off its wall but did not advance the cumulative best, so it is excluded from this total
- verify: Confirm dispatches per call collapsed to 1 and that measured traffic now equals the ideal multiple of tensor bytes, then score each case against its OWN wall - cache-resident shapes against a last-level-cache figure, streaming shapes against the measured DRAM number rather than the paper one. Take a same-language copy control (a plain streaming copy kernel in the same DSL) alongside the library copy before calling any residual headroom real.
- pitfall: a round was funded on the premise that per-thread load width was the residual -> the ISA already emitted the widest buffer load/store for a row-contiguous narrow-dtype block at zero spill -> dump the AMDGCN before funding a width direction, then build a cost ladder (copy -> +convert -> +1 cross-lane reduce -> +2 reduces -> full op), which attributed the whole residual to the row reduction and left the exponential noise-adjacent.
removing the power-of-two mask measured slightly worse at num_warps=4 and clearly better at num_warps=1 -> elements per lane scale as 1/num_warps, so a per-element predicate or precision cost is config-dependent -> re-run every per-element A/B at each num_warps you might ship, and keep the masked path for non-power-of-two widths.
porting the geometry of a fast streaming-copy probe (persistent grid-stride) produced the slowest family measured -> the probe was fast because it had no cross-lane reduction at all, not because it was persistent -> port what made a control probe fast, not its shape.
the cache-resident shape swung bimodally far beyond its within-mode spread -> last-level-cache placement of a freshly allocated output, not a kernel property -> gate any claim on such a shape on a median of at least 3 interleaved full runs.
- caution: also verify, rather than assume, the axes that all returned ~1.00x here once the fused form was in place: load- and store-side cache modifiers and eviction policy, waves_per_eu, num_stages, rows-per-program and 2-D row tiles, explicit per-lane vector indexing, and wrapper-level graph capture (replay measured slower than eager once dispatch count was 1). Also compute the maximum geomean upside from the per-case table before funding a round - when only one of several cases is off its wall, even a perfect fix can land within a few times the noise floor and be unresolvable by a single full-benchmark comparison.
- source: run team_softmax_kernel_20260917_130805_24873_21932, 2026-09-17 - TechLead final report, 2 rounds / 3 directions plus 13 in-direction negative-control probes, director re-verified (accepted), correctness 14/14 PASS
