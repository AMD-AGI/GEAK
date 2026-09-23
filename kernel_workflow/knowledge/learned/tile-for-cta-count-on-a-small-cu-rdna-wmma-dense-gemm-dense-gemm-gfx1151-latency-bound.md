---
key: fp16 square dense GEMM at small shapes on a 40-CU RDNA3.5 WMMA iGPU (gfx1151), Triton, with the Python launcher living in the same editable file
type: lever
confidence: ★★
effect: 2.53x geomean vs the frozen baseline, non-overlapping and reproduced in-session (2.534 / 2.537); per-case 2.99x at M=N=K=128, 2.98x at 256, 1.83x at 512; vs the tuned vendor library 1.51x / 1.83x at 128 / 256 and 0.87x at 512
confirms_cited: 0
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-09-17
name: tile-for-cta-count-on-a-small-cu-rdna-wmma-dense-gemm-dense-gemm-gfx1151-latency-bound
description: Small fp16 dense GEMM on a 40-CU RDNA3.5 WMMA part: tile for CTA count not instruction count, then delete the JIT launch floor that hid it - 2.53x
keywords: ['dense-gemm', 'fp16', 'triton', 'gfx1151', 'wmma', 'rdna', 'tile-geometry', 'grid-occupancy', 'cu-underfill', 'latency-bound', 'launch-overhead', 'host-runtime', 'lds-staging', 'num-warps', 'isa-inspection', 'occupancy']
kernels: ['gemm_kernel']
platforms: ['gfx1151']
kernel_class: dense_gemm
regime: latency-bound
layer: learned
lifecycle: active
origin_kernels: ['gemm_kernel']
cost: L2
verified_on: 2026-09-17
roofline: latency-bound (too few resident waves) before and after; device fill 2.5/10/40% -> 20/100/100% of the CUs, matrix-pipe occupancy ~3% / ~16% / ~35% of WMMA peak at 128 / 256 / 512, with the largest shape drifting toward balanced at ~41% VALU+matrix pipe occupancy
levers: ['compute.grid-occupancy', 'host.launch-overhead', 'mem.lds-staging']
---
# Tile for CTA count on a small-CU RDNA WMMA dense GEMM
- lever: On a small-CU RDNA WMMA part a small dense GEMM is limited by resident parallelism, not by the instruction stream: choose the tile PER SHAPE by how many CTAs it puts on the device and accept a worse per-matrix-instruction mix. Measured here at the mid shape, a 32x64x128 -> 16x64x64 retile raised instructions per matrix op ~70% and still ran faster, purely because CTAs doubled and device fill went 80% -> 100%.
- apply: A shape-specialized config table (here 64x32x128 nw8 ns3 / 16x64x64 nw4 ns3 we4 / 32x64x64 nw2 ns2 we2 at M=N=K 128 / 256 / 512) plus a largest-tile-that-still-fills-every-CU rule off-table; num_stages>=2 so operands stage through LDS and the scratch spill disappears; a pointer-advance K loop with index-wrap in place of per-load masks and the shape predicates packed into one constexpr (launch cost scales with parameter count); a guarded monomorphic inline-cache slot that calls the compiled kernel's raw C launcher instead of the JIT run path; per-shape WMMA-native LDS staging via the AMD in-thread-transpose compiler knob, set around your own compile and restored after.
- stack: total 2.53x director-verified geomean = three directions, cumulative in landing order
  - 1. shape-specialized tile/grid + scratch-spill removal - 1.37x standalone (round 1, verified) - the entire device-side win: scratch to zero, VGPR roughly halved, CTAs 1/4/16 -> 8/32/128
  - 2. raw-C monomorphic launch slot - 1.12x standalone (round 1, verified), but hand-merged with (1) and re-swept it gave 2.48x, i.e. 1.81x over the better single patch; the product of the two standalone numbers predicted only 1.53x
  - 3. WMMA-native LDS operand staging + the joint residency re-sweep it forced - +2% on top of (1,2) (round 2, verified); the knob alone was worth ~3% on the two larger shapes and the re-sweep carried the rest
  - note: attribution is incremental in landing order; (1) and (2) were isolated standalone, (3) was only ever measured on top of them
- verify: Prove the fill changed, not just the config: count CTAs against CU count, read waves/SIMD, and take allocated shared bytes from the compiled-kernel metadata rather than the profiler, which reported zero here; confirm scratch is zero. For the launch slot, compare per-call host cost against the smallest shape's device time. For the staging knob, confirm in the ISA that wide LDS loads replaced the narrow element-wise ones.
- pitfall: a 96-point config sweep spanned almost no range and concluded the tile was irrelevant -> it was fitted under a host launch floor larger than device time, so it measured the host -> re-sweep every config table after any change that moves the bottleneck; the small-shape answers inverted once the floor was gone.
two parallel lanes' patches would not stack -> textually and semantically incompatible edits to the same file -> hand-merge and re-run the tile sweep through the merged module; forcing the apply would have silently reverted one lane to its default config.
ranking variants by the LDS-load : matrix-op ISA ratio picked losers -> a low ratio correlates with fat per-warp tiles and so with register pressure -> use the ISA ratio to identify the mechanism and wall time to rank candidates.
instruction-count issue utilization read ~13% and looked idle -> it counts instructions, and one matrix op holds the pipe for many cycles -> weight the matrix op by its issue cost; true pipe occupancy was ~41%.
the smallest shape swung ~12% between runs of the unmodified file -> box-level bimodality far beyond the documented spread -> gate any sub-8% claim there on interleaved A/B with cooldowns.
- caution: also verify the occupancy ceiling before adding CTAs: the WMMA-native staging knob roughly doubled VGPR use here and halved the waves/SIMD ceiling, leaving only ~1.5x of margin over what was achieved, so sweep tile geometry and register count as one tuple. Also re-test that knob jointly with any new tile at the smallest shape, where it was rejected only under a single-K-iteration config in which its extra store side cannot amortise.
- source: run team_gemm_kernel_20260917_094257_2933_21058, 2026-09-17 - TechLead final report, 2 rounds + 1 hand-integration, director re-verified, correctness PASS on 7 checks
