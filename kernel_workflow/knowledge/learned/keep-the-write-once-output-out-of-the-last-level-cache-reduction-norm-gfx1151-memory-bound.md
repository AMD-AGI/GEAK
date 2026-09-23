---
key: bf16 fused residual-add + RMSNorm and similar elementwise-plus-row-reduce ops that stream two re-read inputs and one write-once output, on a 40-CU RDNA3.5 wave32 iGPU (gfx1151) with a shared LPDDR path and a large last-level cache, Triton, from a two-launch two-pass baseline
type: lever
confidence: ★★
effect: 1.8129x geomean vs the frozen baseline, director re-verified (two full runs at 1.8063 / 1.8129 against an A/B instrument accurate to ~0.4%); per case 2.78x at rows x cols = 2048x4096 (working set just over the last-level cache), 1.49x at 8192x4096 and 1.44x at 8192x8192 (streaming); 96.3% of the honestly attainable geomean for this access mix; correctness 8/8 shapes, output bit-identical to the reference and the residual delta exactly zero
confirms_cited: 0
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-09-20
name: keep-the-write-once-output-out-of-the-last-level-cache-reduction-norm-gfx1151-memory-bound
description: Fused residual-add + RMSNorm on an RDNA3.5 iGPU: fuse to the traffic floor, then bypass last-level-cache allocation on the write-once output - 1.81x
keywords: ['reduction-norm', 'kernel-fusion', 'dispatch-collapse', 'cache-modifier', 'non-temporal-store', 'l2-residency', 'store-bandwidth', 'memory-bound', 'isa-inspection', 'num-warps', 'waves-per-eu', 'occupancy', 'noise-floor', 'stacking', 'triton', 'bf16', 'gfx1151', 'rdna']
kernels: ['addnorm_kernel']
platforms: ['gfx1151']
kernel_class: reduction_norm
regime: memory-bound
layer: learned
lifecycle: active
origin_kernels: ['addnorm_kernel']
cost: L3
verified_on: 2026-09-20
roofline: memory-bound wasteful -> memory-bound saturated: 1.25x excess traffic over the fused contract floor across 2 dispatches -> exactly the floor in 1; the two streaming shapes move from ~81% to 99-100% of the measured DRAM wall and the cache-resident shape from a 0.83x liability to ~98% of the measured last-level-cache ceiling; occupancy deliberately capped from ~92% to ~23% of wave slots with zero spills, VALU under 10%
levers: ['mem.store-cache-policy', 'algo.fusion', 'host.launch-tuning']
---
# Keep the write-once output out of the last-level cache
- lever: When a fused elementwise+row-reduce op streams two inputs it re-reads and one output it never re-reads, the binding constraint at working sets near the last-level cache is RESIDENCY, not per-byte efficiency: ask only the write-once stream to skip last-level-cache allocation while the re-read streams stay cacheable. On this RDNA3.5 part the no-allocate control is a single policy bit on the store, and it pays in BOTH regimes - roughly 3.4x the achieved store bandwidth of the default policy when the working set is ~1.5x the cache, and still +1.6 to +2.9% far past it - so no working-set gate is needed.
- apply: Emit the bit through an inline-asm elementwise store, because the portable store cache_modifier / eviction_policy arguments lower to nothing on this toolchain. That route accepts one packed operand only, so it caps the store at 64-bit: view each row as packed integer words (4 bf16 per word) and re-derive every launch formula in word units, e.g. num_warps = max(1, min(16, BLOCK_words // 128)) to pin 4 words resident per lane at any row width, plus a deliberate waves_per_eu occupancy cap. Keep masked plain stores on the ragged-width and wide-row fallback paths.
- stack: total 1.8129x director-verified geomean from a hand-merge of two independently verified directions, each measured standalone against the SAME frozen baseline (parallel, not incremental)
  - 1. cache policy, no-allocate on the write-once output - 1.6297x standalone (round 1, verified) - turns the cache-resident shape from 0.83x into 2.35x, on an untuned schedule that left the streaming shapes ~10% short
  - 2. fusion to one dispatch, one program per row, row register-resident across the mean-square reduction - 1.2447x standalone (round 1, verified) - reaches the exact traffic floor, but alone regresses the cache-resident shape to 0.83x
  - 3. merge premium +11.2% over the better individual (round 1 integrate), from re-deriving the width formula in word units and the occupancy cap (~+5% by itself at 8 warps); never run as its own round
  - note: (1) and (2) are orthogonal REGIME winners - policy decides the cache-resident shape, width and occupancy decide the streaming ones - so the compounding lives in the merge, and a later 9-point launch-schedule grid added nothing on top
- verify: Diff the emitted machine code for the policy bit on the store before crediting any cache-hint timing, then A/B one shape whose working set sits just above the last-level cache against one far past it, all arms interleaved inside a single session, n>=3 reps.
- pitfall: portable store cache_modifier / eviction_policy read as timing noise -> six policy spellings compiled to byte-identical machine code, the argument being dropped silently -> emit the bit through inline asm and confirm it in the ISA
- pitfall: porting the vendor non-temporal store builtin verbatim reached only ~0.57x of the bandwidth of the no-allocate bit alone, and -19% in the streaming regime -> the builtin sets two extra policy bits and one of them is separately costly -> sweep the policy bits one at a time instead of copying the builtin
- pitfall: the fused kernel regressed the cache-resident shape to 0.83x -> three streams contend for the last-level cache once they share one kernel -> land the store policy in the same patch and judge the pair, not either half
- pitfall: gating the policy on working-set size cost ~3% -> the bit is positive, not merely neutral, in the streaming regime -> leave it ungated
- pitfall: the winning num_warps constant carried across into the merged body scored 0.89x of the merged result -> that constant encoded a per-lane residency rule in the pre-merge data view's units -> transfer the rule as a formula in the surviving units
- pitfall: a width prior of 2-4 warps with 'spills above 8' inverted here - those are the two worst widths and nothing spilled up to 32 warps / 1024 threads -> the prior is CDNA-derived and this is a wave32 RDNA part -> re-read register pressure per width in the ISA before trusting any width prior
- pitfall: a sub-1% geomean effect looked repeatable over two reps and evaporated on a four-way interleaved re-run -> ~1% single-config spread plus ~2% cross-session drift -> interleave every arm in one session (A-B-C-D then D-C-B-A, n>=3) and decline directions whose honest ceiling is under ~2%
- caution: The inline-asm store carries no mask, so also verify it is predicated on a fully in-bounds word block and that the ragged-width and wide-row fallbacks keep their masked stores - a store past a row end is an out-of-bounds write, not merely a wrong number. Also verify a knob can actually bind before funding a sweep on it: one width cap here was unreachable by construction and compiled an identical binary.
- source: run team_addnorm_kernel_20260920_020850_1116265_4613, 2026-09-20 - TechLead final report, 2 rounds / 3 directions, frozen-baseline interleaved A/B plus 8/8 oracle parity
