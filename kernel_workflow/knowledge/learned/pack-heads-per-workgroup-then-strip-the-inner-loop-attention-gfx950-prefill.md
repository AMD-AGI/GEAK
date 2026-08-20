---
key: top-k sparse MLA attention prefill with gathered KV rows (head_dim 512 = nope+rope, topk 512), Triton bf16 on gfx950/MI355X
type: lever
confidence: ★★
effect: 3.08x unweighted geomean vs the frozen baseline, non-overlapping (per case 4.36x on the large query block, 4.01x on the mid, 1.67x on the tiny dispatch-floored one); oracle parity on all three shapes with no tolerance loosened
confirms_cited: 0
confirms_blind: 0
losses: 0
attempts: 0
toolchain: unknown
last_seen: 2026-08-17
name: pack-heads-per-workgroup-then-strip-the-inner-loop-attention-gfx950-prefill
description: Sparse top-k MLA prefill on gfx950: pack heads per workgroup to delete gather amplification, then strip inner-loop VALU and decouple softmax - 3.08x geomean
keywords: ['attention', 'triton', 'gfx950', 'tile-geometry', 'grid-occupancy', 'online-softmax', 'valu-bound', 'latency-bound', 'mfma-tiling', 'occupancy']
kernels: ['_sparse_attn_prefill_ragged_kernel']
platforms: ['gfx950']
kernel_class: attention
regime: prefill
layer: learned
levers: ['algo.head-packing', 'compute.valu-strip', 'algo.decoupled-softmax']
cost: L3
lifecycle: active
verified_on: 2026-08-17
roofline: gather-amplified memory-bound (each KV row fetched once per head group) -> latency/issue-wait bound at ~1.9 of 4 waves per SIMD, ~15% of bf16 MFMA peak with HBM at ~11% of nameplate; the MFMA-busy floor still sits ~4.3x under the achieved time
---
# Pack heads per workgroup, then strip the inner loop
- lever: on top-k sparse attention prefill the first lever is the tile, not the memory path: one workgroup per query block holding the whole head group (here 16 -> 64 heads, grid /4) fetches each gathered KV row once instead of once per sub-group, and only then is it worth deleting inner-loop work (dead masks, two-sided bounds checks, the per-block softmax rescale).
- apply: widen the head tile until gather amplification reads 1x; strip dead tl.where / constexpr-known masks, switch exp to exp2, retune BLOCK_K; decouple the softmax so the first key block fixes the reference max and later blocks skip both the row-max and the full fp32 accumulator rescale; replace two-sided validity checks with one unsigned compare and peel the ragged tail; gate the wide tile on a query-count threshold with the narrow tile as fallback.
- stack: total 3.08x isolated (unweighted geomean, director-verified) = four steps compounded
  - 1. head packing into one workgroup - 1.87x standalone (round 1, verified) - deletes the 4x gather amplification
  - 2. inner-loop VALU strip + BLOCK_K doubling + exp2 + a waves_per_eu hint - 2.18x standalone (round 1, verified) - largest single patch
  - 3. hand-merge of (1)+(2) plus a joint tile x num_warps x BLOCK_K ladder and a small-shape tile fallback - 2.70x cumulative (round 1 integrate, verified) - +24% over the best single patch; neither parent's own num_warps survived
  - 4. decoupled softmax + unsigned validity compare + peeled ragged tail - +14.2% on the carry-in (round 3, verified) - a 32% VALU and 35% VMEM cut with the MFMA stream and register state bit-identical
  - note: attribution is incremental in landing order, not independent; (3) is a joint retune, so (1) and (2) do not add.
- verify: read gather amplification (bytes fetched / unique bytes) back at 1x after the tile change, and diff the compiled ISA to confirm the deleted instructions are gone with the MFMA count unchanged; then re-time every case against the frozen baseline, because the wide tile is a regression on shapes below the gate.
- pitfall: instruction deletion returned far less than its size (-32% VALU bought -12% time) -> at ~2 waves/SIMD the freed slots reappear as issue wait rather than deleting the latency they were hiding -> price an instruction-count direction at roughly 0.4x its instruction delta on a low-occupancy tile.
- pitfall: the accumulator rescale was named prime VALU suspect for two rounds and is only ~7% of the loop body -> attribution was guessed from the source, not measured -> attribute per source-location in the disassembly first; here 35% of the body existed only to materialise a small int32 index vector into the gather layout and turn it into a predicate.
- pitfall: a launcher-metadata field reading zero LDS was quoted as "this kernel uses no LDS" while the loop body issues tens of shared-memory ops per lane per iteration -> a launcher/metadata field is not evidence about the generated code -> read the ISA.
- pitfall: shrinking per-lane registers to buy occupancy lost monotonically and spill-free configs ran well behind the spilling carry-in -> the winning tile was already pinned at the register cap, so there was no residency to buy -> treat spill and occupancy as symptoms here and get residency structurally (a narrower head tile with a second grid axis) if at all.
- pitfall: a bare in-process launch-loop microbenchmark ranked the tiny-shape config backwards (a config that looked 10% better was 43% worse under the scored graph replay), and the first timed run of a fresh process reads high from clock ramp -> confirm small/short cases on the scored harness and report a median of >=6 runs, discarding the first.
- caution: also verify the ragged-tail and out-of-range-slot paths with your own edge shapes - a scored case set whose query counts are exact multiples of the key block never executes them, so oracle parity says nothing about that rewrite.
- caution: also verify how the framework lowers the two dots' warp tiling before budgeting a further round: here the QK and PV dots received different warp layouts, costing redundant QK matrix instructions plus heavy operand replication, and no exposed knob or block shape could change it.
- source: GEAK per-kernel campaign, sparse-MLA prefill lane, 2026-08-17, director-validated (3 rounds, isolated A/B vs frozen baseline, oracle PASS x3)
