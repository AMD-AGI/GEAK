---
title: XCD / L2 locality (chiplet-aware grid mapping)
kind: technique
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8]
regimes: [prefill, decode, training, both]
updated: 2026-06-05
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf
  - https://hc2024.hotchips.org/assets/program/conference/day1/23_HC2024.AMD.MI300X.ASmith(MI300X).v1.Final.20240817.pdf
---

# XCD / L2 locality

## TL;DR
MI300X is a chiplet GPU: **8 XCDs (Accelerator Complex Dies)**, each with its **own L2 cache** and a
slice of CUs. Workgroups that share input data should land on the **same XCD** so they hit a shared L2
instead of crossing the die boundary to another XCD's L2 / the Infinity Fabric. The practical recipe:
launch **≥1024 workgroups** (feed all XCDs/CUs), use **8-multiple tile counts** (so the round-robin grid
splits evenly across 8 XCDs), and apply an **XCD-aware / swizzled CTA order** so consecutive program-ids
that reuse the same B-tile co-locate on one XCD. See `[[hardware/cdna3_mi300/xcd_chiplet.md]]`,
`[[hardware/shared/l2_xcd_swizzle.md]]`, and `[[operators/dense_gemm/tuning.md]]`.

## Concepts (the hardware)
- **8 XCDs / MI300X**: each XCD has ~38 CUs (304 total) and a **partitioned L2** local to that die.
  L2 is *not* unified across XCDs — a miss serviced from another XCD's L2 or HBM costs more.
  CDNA4/MI350X keeps the multi-XCD design with a different CU/XCD count
  (`[[hardware/cdna4_mi350/arch.md]]`).
- **Round-robin block dispatch**: the hardware assigns workgroup ids to XCDs in round-robin; the
  *default* linear program-id → XCD mapping scatters data-sharing blocks across all 8 dies, defeating L2
  reuse.
- **L2 partition**: keeping the working set of a tile within one XCD's L2 turns repeated B/A re-reads
  into L2 hits.

## The levers
- **≥1024 workgroups**: ensures every XCD and CU gets work; sub-1024 grids leave dies idle on prefill
  (`[[operators/dense_gemm/tuning.md]]`, `[[optimization/wave_and_grid_sizing.md]]`).
- **8-multiple tile counts**: choose grid dims so the number of tiles is a multiple of 8; round-robin
  then balances exactly across the 8 XCDs, no straggler die.
- **Swizzled / XCD-aware CTA order**: remap `pid → (xcd, local_id)` so blocks reusing the same B column
  panel (same N-tile, varying M) are issued to the **same XCD** and reuse its L2. This is the
  GEMM "grouped"/super-grouping launch (triton `GROUP_SIZE_M` is the row-grouping form of the same idea;
  XCD swizzle is the die-grouping form). See `[[hardware/shared/l2_xcd_swizzle.md]]`.
- **8-multiple tile sizes** (BLOCK_M/N): a separate but compounding rule — keep BLOCK_M/N multiples of 8
  for MFMA/alignment friendliness (`[[optimization/vectorization_and_coalescing.md]]`).
- **Persistent kernels**: a persistent grid (one workgroup/CU looping over tiles) lets you assign the
  tile→XCD mapping explicitly instead of trusting the dispatcher
  (`[[optimization/wave_and_grid_sizing.md]]`).

## Example mapping (conceptual)
For a tiled GEMM with `T` tiles on 8 XCDs, instead of `xcd = pid % 8` (scatters reuse), compute
`group = pid / tiles_per_xcd; xcd = group; local = pid % tiles_per_xcd` so a contiguous run of
data-sharing tiles stays on one XCD's L2. Tune `tiles_per_xcd` to the L2 working-set size.

## Dispatch order is also a *makespan* lever, not only an L2 lever
The round-robin assignment is **static**: under SPX, workgroups reach the CUs in **x-fastest linear order**
and block `i` is handed to XCD `i % 8` with no migration — a block cannot move to whichever XCD happens to
be idle. Three consequences that are easy to miss if you only think about L2:

- **Put the expensive axis on the SLOWEST grid axis (longest-job-first).** When per-block cost varies
  systematically along one grid axis — the standard case is a causal attention kernel with a KV-outer
  schedule, where a block's q-tile count falls monotonically with its KV tile index — having that axis
  *fast* means the whole cost range of one sequence is dispatched before the next starts, so a maximal
  block can begin past the halfway point and become the tail. Moving it to the slow axis dispatches all
  the expensive blocks first. Measured **−20%** on a gfx942 fused attention backward (754 blocks ranging
  0–114 q-tile iterations), matching the greedy-list-scheduling prediction of 114/144 = **0.79**. Cost:
  two SALU to decode the block index plus a permuted grid in the launcher.
- **Model 8 static queues of ~38 CUs, not one global CU pool.** A single-pool model reproduces headline
  critical paths but mispredicts anything that changes grid *composition*, because the busiest XCD sets
  the makespan.
- **Empty workgroups are free at the end of the grid and expensive in the middle.** Appending 5504 wholly
  empty blocks cost nothing measurable; interleaving the *same* number between live blocks cost **+29%**,
  because inserting blocks repartitions every later block across the XCDs (XCD load went from a balanced
  3193–3293 to a skewed 2243–4259 iteration units). Early-exit blocks are cheap, but do not scatter them
  through the grid. This also qualifies the "≥1024 workgroups" lever above: pad by *extending* the grid,
  never by interleaving no-op blocks among live ones.

**Confirm the partition mode before trusting any of this** — SPX/NPS1 is what makes the mapping
predictable, and CPX changes both dispatch and device-scope atomic behaviour. If a dispatch-order ablation
comes out neutral when the model says it should be ~0.79, suspect the partition mode before the kernel.

## Pitfalls
- Default linear pid mapping ⇒ B/A panels re-fetched from HBM/foreign L2 instead of local L2 hits.
- Tile count not a multiple of 8 ⇒ one or more XCDs finish early (load imbalance, tail latency).
- <1024 workgroups on prefill ⇒ idle CUs/XCDs (`[[operators/dense_gemm/tuning.md]]`).
- Over-grouping (too many tiles pinned to one XCD) ⇒ L2 thrash; size groups to the L2 capacity.
- Assuming a unified L2 (it is **partitioned per XCD**).
- Optimizing the loop body before checking the grid axis order on a kernel with a structural cost gradient
  across one axis — the axis order was the single largest and cheapest win in the measured case.
- **Designing around L2 residency for device-scope atomics.** A private L2 per XCD means such an atomic
  must carry the uncached bit and execute past the fabric, so it misses L2 by construction (counted as
  *Write and Atomic (Uncached)*, with equal atomic counts at the L1→L2 and L2→fabric interfaces). Fanning
  atomics out across more slots to "spread contention" therefore only multiplies the DRAM footprint —
  measured monotonically worse at 1/2/4/8 slots.

## Verify
- Omniperf: L2 hit rate per channel / cross-XCD traffic, HBM read BW; XCD-aware order should *raise* L2
  hits and *lower* HBM reads at equal FLOPs (`[[profiling/]]`).
- A/B: linear vs swizzled pid mapping; compare HBM read volume and latency.
- Sanity: workgroup count ≥1024 and tile count % 8 == 0.

## Sources
- 8 XCDs, partitioned L2, round-robin dispatch: AMD CDNA3 whitepaper + MI300X Hot Chips 2024 architecture deck.
- ≥1024 WGs, 8-multiple tiles, XCD/L2 placement levers: ROCm MI300X workload optimization guide.
- Swizzle mechanics: `[[hardware/shared/l2_xcd_swizzle.md]]`.
- Longest-job-first grid axis (−20%, 114/144 model), static-XCD-queue modelling, empty-workgroup
  placement (+29% interleaved vs free appended), and uncached device-scope atomics: first-party on-box
  gfx942 MI300X SPX/NPS1, ROCm 7.1.0
