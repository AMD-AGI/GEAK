---
key: INT4 block-quantised weight-only GEMV / skinny-M GEMM (packed nibbles, one fp16 scale per 32-element block) hand-written in HIP on a 40-CU RDNA3.5 wave32 iGPU with a shared-LPDDR path and a cache-resident weight matrix, M=1..8
type: lever
confidence: ★★
effect: 54.78x geomean isolated vs the frozen baseline, director-verified and non-overlapping; per-case 22.6x at M=1 and 133.1x at M=8 (N=K=4096, block_size=32); M=8/M=1 time ratio 8.04x -> 1.35x; oracle parity holds at the naive baseline's own error (rel_max ~3.2e-4)
confirms_cited: 0
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-09-18
name: fix-the-wave-mapping-before-the-int4-dequant-algebra-quantized-gemm-gfx1151-decode
description: INT4 weight-only GEMV on a 40-CU RDNA3.5 wave32 iGPU: wave-per-row-group split-K mapping first, then LDS-staged pre-permuted activations + DPP16 reduce
keywords: ['int4', 'w4a16', 'weight-only-quant', 'int4-dequant', 'quantized-gemm', 'gemv', 'skinny-m', 'decode', 'gfx1151', 'rdna', 'raw-hip', 'lds-staging', 'dequant-hoist', 'packed-valu', 'ilp', 'occupancy', 'workgroup-mapping', 'software-pipelining', 'measurement-discipline', 'profiler-error']
kernels: ['matmul_nbits']
platforms: ['gfx1151']
kernel_class: quantized_gemm
regime: decode
layer: learned
lifecycle: active
origin_kernels: ['matmul_nbits']
cost: L3
verified_on: 2026-09-18
roofline: memory-bound at ~51x (M=1) / ~402x (M=8) of the cache-resident read roofline -> ~2.2x / ~3.0x of it; at the end M=1 is memory-bound on the weight stream alone (50% of that kernel, at 64-77% of the measured achievable last-level-cache read rate) while M=8 has shifted to latency/occupancy-bound (no pipe above ~62%, 5.5 of 8 waves/SIMD achieved)
levers: ['algo.wave-cooperative-split-k', 'compute.int4-dequant-algebra', 'mem.lds-staging', 'compute.dpp-reduction']
---
# Fix the wave mapping before the int4 dequant algebra
- lever: Give one wave32 a small group of output rows and split K across its 32 lanes, hoist the M loop INSIDE the K loop so the weight stream is read once for all rows, align the load width to the quantization block (one 16 B vector of packed nibbles == exactly one 32-element scale block, so the scale gather leaves the inner loop), unpack nibbles to fp16 with a magic-constant OR-in feeding packed dot2 with fp32 accumulate, and stage the activation tile once per workgroup in LDS in the pre-permuted order the unpack wants.
- apply: Template the kernel on M and let the host dispatcher pick an LDS-staging instantiation for the wider M and a staging-free one for M=1 (the tiny activation tile is already L0-resident there), keeping one dispatch per call with a generic fallback for off-spec shapes; put every knob (rows/wave, waves/WG, unroll, double-buffer, accumulator split) behind a template parameter chosen by a cached env switch so A/B pairs interleave inside one session bracketed by two reference runs.
- stack: total 54.78x isolated (geomean, director-verified) = three directions compounded
  - 1. wave-cooperative split-K mapping with M inside K - 42.3x standalone (round 1, verified); grid 16 -> 256 workgroups on 40 CUs, per-load-instruction contiguity 512 B, M-scaling 8.04x -> 1.65x
  - 2. int4 dequant algebra (block-aligned wide loads, magic-constant unpack, packed dot2, scale-after-dot) - 30.5x standalone under its own thread-per-row mapping (round 1, verified); hand-merged onto (1) it reached 48.9x cumulative, +15.5% over the better parent
  - 3. LDS-staged pre-permuted activations + DPP16 wave reduce + even/odd split accumulators for dual-issue - +12.3% on top (round 2, verified), attributed in-round as 11.5 / 7.0 / 1.3 points
  - note: attribution is incremental in landing order; (1) and (2) had to be hand-merged because both rewrite the same source region, and the merged body needed a full knob re-tune.
- verify: Expect each load instruction to cover a contiguous multi-cache-line run per wave, activation requests down ~85% and vmem instructions per lane down ~3.5x after staging, zero LDS bank conflicts, and the M-scaling ratio collapsing toward 1; take every decision from the 100-iteration benchmark mode against the frozen baseline.
- pitfall: the sub-technique that LOST round 1 (staging activations in LDS) was the round-2 winner -> it had been priced against the other parent's thread-per-row mapping, not against the merged body -> re-price a rejected partial after the mapping changes (the same parent's load-bearing scheduling barrier measured neutral under the new mapping, so transfer runs both ways)
- pitfall: a v_mov appeared in front of every packed-dot2 accumulate -> seeding a fresh accumulator per sub-chunk makes the destination alias an operand -> accumulate the UN-scaled dot across sub-chunks and apply the scale once per quantization block, which also deletes 3/4 of the scale FMAs
- pitfall: the cross-lane butterfly reduce contended with the staged tile reads -> __shfl_xor lowers to ds_bpermute, an LDS-pipe instruction, and 16 accumulators x 5 steps is a lot of them -> do 4 of the 5 wave32 steps with DPP16 quad/row permutes, which are pure VALU
- pitfall: a stream's measured cost came out 5 points too high (-55% vs -50.2%) -> the ablation zeroed the index, making the value wave-uniform and loop-invariant so the load and everything downstream hoisted (static VALU 792 -> 276) -> rewrite the index into a small window that stays lane- AND loop-varying, then ISA-verify load/VALU counts match the reference
- pitfall: two identical builds looked 17% apart -> the short benchmark's spread grows as the kernel shrinks (5-12% after a 25x win, against the 0.15% recorded at setup) -> decide on the longer 100-iteration mode and re-establish the noise bar after every large win
- pitfall: the profiler's kernel trace reported the small case ~3x slower than the benchmark -> a long CPU-only profiler startup leaves the GPU down-clocked and a sub-millisecond case runs entirely inside the clock ramp -> read its time from GPU-busy cycle counters or the benchmark instead
- caution: Also verify which resources jointly pin the occupancy ceiling before funding an occupancy round: here it was double-pinned by register count AND LDS bytes per workgroup, so relaxing either alone measured flat. Also verify the tail iteration of a software-pipelined prefetch: the final trip re-loaded and re-staged a chunk nothing reads, ~20% of every stream.
- source: run team_matmul_nbits_kernel_20260918_024328, 2-round HIP campaign 2026-09-18, director-verified 54.7846x (100-iteration benchmark, oracle parity PASS 4/4 shapes incl. two anti-hardcoding shapes)
