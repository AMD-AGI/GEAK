---
key: skinny-M fp16 GEMV / decode projection (M=1..32 against square N=K weights that marginally exceed the last-level cache) in Triton on a 40-CU RDNA3.5 wave32 iGPU with WMMA and a shared LPDDR memory path
type: lever
confidence: ★★
effect: 1.4841x geomean vs the frozen baseline, director-verified and reproduced in-session (1.4841 / 1.4846) against a ~2% per-case and ~1.5% geomean noise bar; per case 1.66x at M=1, 1.59x at M=8, 1.24x at M=32 - and that last ratio is understated, because the frozen baseline hardcoded a tile height below M and computed only half that case's rows: against an honest-work reference of the same algorithm it is 1.90x. Correctness PASS on 3 gating and 4 advisory shapes.
confirms_cited: 0
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-09-18
name: fill-the-cus-to-the-plateau-then-bucket-the-inner-path-by-m-dense-gemm-gfx1151-decode
description: Skinny-M fp16 GEMV on a 40-CU RDNA3.5 iGPU: shape-keyed split-K to the WG/CU fill plateau with a fused reduce, then a per-M inner path - 1.48x
keywords: ['split-k', 'grid-fill', 'cu-underfill', 'gemv', 'skinny-m', 'm-bucket', 'dense-gemm', 'fp16', 'triton', 'gfx1151', 'rdna', 'wmma', 'dispatch-collapse', 'arrival-counter', 'valu-bound', 'l2-residency', 'tile-geometry', 'roofline', 'static-isa-screen', 'measurement-discipline', 'closed-axis', 'anti-pattern']
kernels: ['gemv_kernel']
platforms: ['gfx1151']
kernel_class: dense_gemm
regime: decode
layer: learned
lifecycle: active
origin_kernels: ['gemv_kernel']
cost: L3
verified_on: 2026-09-18
roofline: CU-underfilled / latency-bound (the baseline launch left a fifth of the CUs with no work at under 2 waves per SIMD) -> memory-bound with zero idle CUs at a 3-5 workgroup-per-CU plateau: the fraction of the box's demonstrated achievable read ceiling for this byte stream moves ~0.60 -> ~1.00 at M=1, ~0.51 -> ~0.82 at M=8, ~0.46 -> ~0.57 at M=32. The residual at the largest M is last-level-cache RESIDENCY (the weight matrix marginally exceeds the cache), not byte bandwidth; its tile is pinned by the 256-VGPR accumulator cap with zero scratch.
levers: ['algo.split-k', 'host.launch-shape', 'host.dispatch-collapse', 'compute.m-bucket-specialisation']
---
# Fill the CUs to the plateau, then bucket the inner path by M
- lever: On a small-CU part a skinny-M GEMV win is a PRODUCT of two independent axes, not one knob: (a) a SPLIT_K program axis sized off the DETECTED CU count to a measured 3-5 workgroup-per-CU fill plateau - not to maximum parallelism - with a fused single-dispatch cross-partition reduce (fp32 partials into a cached workspace, an acq_rel arrival counter per n-tile elects the last partition to sum and cast once and re-arm the counter for the next call); and (b) a per-M constexpr inner path picked in the wrapper, where the M=1 bucket drops the matrix instruction entirely for a pure-VALU broadcast-FMA loop that reaches the read roof while larger M stays on matrix tiles. Split depth is shape-keyed and scales inversely with M, down to no split at the largest M.
- apply: All wrapper-side plus constexpr: a static shape-keyed config table (no autotune on the hot path) giving (bucket, tile, SPLIT_K, num_warps, num_stages) per M bucket, with the grid derived from the runtime CU count times the fill target. Two rules carried the tile choice: BLOCK_M x BLOCK_N ~ 1024 at num_warps=2 (a 16-fp32-per-lane accumulator) pinned all three bucket winners (32x32, 16x64, and 1x512 on the VALU path) and every departure lost monotonically; and a per-lane load-width rule BLOCK_N/(32*num_warps) >= 8 fp16 per lane on the VALU path, which buys the widest vector load and keeps the BLOCK_K reduction intra-thread. num_stages=2 and num_warps=2 at these tile sizes.
- stack: total 1.4841x director-verified geomean = two directions hand-merged, attributed incrementally in landing order
  - 1. split-K fill target + fused single-dispatch reduce - 1.4184x standalone (round 1, verified) - the bulk of the win
  - 2. per-M constexpr inner path + the per-lane load-width rule - 1.2181x standalone (round 1, verified), but 1.58x at M=1 against 0.80x at the largest M, so it only pays where it is bucket-gated
  - 3. hand-merge of (1) x (2) - +4.9% over the better individual (round 1 integrate, verified): a structural product, carrying the fused reduce INTO the VALU path and re-picking the M=1 tile at the point where both lanes' rules agree
  - note: attribution is incremental, not additive - the two patches each rewrote the whole file, so they could not be stacked mechanically and only a hand-merge measured the product. A second round on the memory axis shipped nothing.
- verify: Confirm dispatches per call stays at 1.0 in a profiler after the split lands (a separate reduce launch costs a double-digit percent of the smallest case), confirm launched workgroups and workgroups-per-CU actually moved per shape, sweep split depth and fill target so you see the plateau rather than a point (over-splitting the smallest M ran several times slower), and re-time each case against the FROZEN per-case baseline table.
- pitfall: profiler reported the shared-memory block size as 0 on every dispatch, which read as a free LDS-staging lever -> Triton requests LDS as DYNAMIC shared memory, so the object's fixed group-segment field stays 0 while the launch still allocates it, and the matrix path was already staging through LDS -> read the compiled metadata's shared figure from an AOT compile before funding any 'idle LDS' direction.
an AOT ISA screen showed narrow 16-bit loads everywhere plus phantom register spills -> the compile was missing the runtime's 16-byte divisibility attributes on the pointer/int args -> pass that attribute for every aligned arg; with it the screen reproduces runtime VGPR and scratch exactly, needs no GPU lock, and correctly predicted every spill in the timed sweep.
per-lane load width looked too narrow at the largest M and seeded a whole dead round -> the total vector-load count had been divided by ONE operand's byte stream -> model the full stream (weights + the per-n-tile re-read of the activation block + split-K partials) and reconcile it against the vector-load counter before believing a width.
the modelled request count matched the counter to 0.002% yet cutting requests 4x made the largest-M case 1.5-2.4x slower across 20 arrangements -> request count is descriptive here, not causal; the exact fit was a coincidence of three shipped configs -> price any request-count direction against one measured arm before funding a round on it.
split-K that was the whole win at M=1 cost in proportion to its depth at the largest M, on only a few MB of fp32 partials -> the weight matrix marginally exceeds the last-level cache, so any extra resident working set evicts it onto the far slower DRAM path -> key split depth to M and price split-K as a cache-RESIDENCY cost when the weights sit at cache size.
a K-major / preshuffled weight layout built once outside the timed region removed the matrix pipe's inherent LDS transpose and was numerically correct, yet was slower on every matrix-path case -> each workgroup then walks a long contiguous row block, so the concurrent workgroups touch the whole weight matrix at once instead of sweeping a narrow k-band together -> compare streaming locality against cache size before trading it for a transpose.
a standalone config sweep mis-ranked configs by up to 14% against the harness -> no clock warm and no interleaved baseline trial -> interleave a frozen-baseline trial and warm clocks before EVERY config, and score against the frozen table rather than a re-timed denominator.
- caution: also verify each M bucket separately before crediting a global knob - the load-width rule and the split depth that carried the small-M cases did nothing or regressed on the largest-M matrix path here; also verify the baseline actually computes every row of the largest-M case before trusting that case's ratio, since a hardcoded tile height below M silently halves the work in the denominator; also verify on this arch which counters exist at all, since several standard vector-memory and wave counters are absent while the texture-load, L2 hit/miss and wave-cycle ones work.
- source: run team_gemv_kernel_20260917_225504_62861_759, TechLead report + director re-measurement, 2026-09-18 (2 rounds, 3 directions, round 2 shipped no patch)
