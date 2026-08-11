---
name: read-the-register-count-out-of-the-code-object-before-spendi-attention-gfx950-memory-bound
description: Read .vgpr_count out of the code object before opening an occupancy round: with zero spill the occupancy pragmas are no-ops — 5 of 6 directions closed
keywords: [vgpr, code-object, occupancy, isa-check, waves-per-eu, attention, prefetch, tile-shape]
kernels: [_fwd_grouped_kernel_stage1, paged_attention_decode]
platforms: [gfx950]
kernel_class: attention
regime: memory-bound
key: occupancy, partition-size and prefetch levers on a decode attention kernel that preloads its whole K/V partition into registers, gfx950
lifecycle: active
type: instrument
confidence: ★★
effect: Six directions on the launch-geometry and occupancy axis, five closed. The code object read .vgpr_count=80 (32 K + 32 V + 16 Q), AGPR 0, scratch 0 — a hard floor with no spill, giving occ 6 — and against that the launch-bound and waves-per-eu pragmas were silently ignored: the occupancy/prefetch re-tune measured 0.9733, moving the V load after the first dot was re-hoisted by the compiler for 2-5% WORSE, and a software-pipeline/prefetch direction returned 1.0004 because the body already preloads the whole 256-token partition. Growing the partition to 512/1024 measured 0.7245 (-25 to -36%) by doubling the live K/V register arrays into a 21-VGPR spill; shrinking it failed correctness outright against a frozen scratch allocation. The one launch-bound change that did pay (1.0475, occ7->occ6) was neutralised entirely once the storage narrowing landed and had to be reverted. Compute-side confirmation from the same axis: an ILP/accumulator-split direction measured 1.004 (TRUE-NEUTRAL) and native narrow-dtype matrix-core PV measured ~1.006 all-cases with the two time-dominant heavy cases flat at 1.000x/1.004x.
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 6
toolchain: rocm 7.x / triton 3.6.0 / torch 2.11.0
source: chuschen 16h time-budget campaign run, 16.43h / 6 passes, 2026-08-11
last_seen: 2026-08-11
---
# Read the register count out of the code object before spending a round on occupancy
- lever: Take the register footprint from the ELF .vgpr_count of the built code object rather than from a profiler's shift analysis, and check for spill/scratch before opening an occupancy direction. When the count is a genuine floor with zero scratch, the launch-bound and waves-per-eu attributes have nothing to give back and are dropped without a diagnostic; the remaining lever is a source-level footprint reduction, which on a body that deliberately preloads its whole tile trades away the memory-level parallelism that was hiding the latency in the first place.
- apply: The same read prices the tile/partition axis in advance — a tile growth that doubles the live operand arrays is a spill prediction you can make before building.
- pitfall: occupancy and prefetch pragmas measured as pure noise -> with zero scratch there was no spill to reclaim, so the attributes were dropped silently -> read .vgpr_count and scratch from the code object first and score the axis closed.
- caution: Also verify the harness's scratch allocation before shrinking a partition: a shrink failed correctness here against an allocation frozen at the old size, and a launch-geometry win measured earlier evaporated once a storage-narrowing edit landed, so also re-measure any surviving win after each structural edit.
- source: chuschen 16h time-budget campaign run, 16.43h / 6 passes, 2026-08-11
