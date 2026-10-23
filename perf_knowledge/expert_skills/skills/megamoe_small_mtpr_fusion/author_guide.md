# MegaMoE small-batch fusion — Author guide

Continue the lane the direction names. The frozen baseline is only the root
commit; never recreate the lane or inspect external implementations. When the
prompt carries `REWIND_TO`, roll the lane back to that commit first (the old
HEAD stays tagged) and continue from there.

Treat `BASELINE_OPERATOR_MAP` and the supplied baseline kernels as concrete
implementation references. Reuse, extract, inline or rewrite each region to fit
the single persistent kernel. Function identity is not required. Preserve the
observable math, quantization, dtype/layout, routing and top-k behavior; weight
precision never changes.

## Development rule

Fix the earliest concrete blocker returned by Verify, commit it, and return the
same lane for independent verification. Preserve working numerical bodies and
already-passing evidence. Do not write token-shaped stubs for contract checks.

The skill.md Procedure gives the build order (a → e). Each turn closes one
step with real device code and an on-card check: compile, `path=MEGA` on 8
ranks, relL2 < 0.10 at the target size and on mixed 1..mtpr calls, launch
count. Once the single kernel is correct, every further step is a performance
step and must come with a paired graph A/B (`MEGA_AB_TOOL`, ≥ 3 interleaved
passes, consistent sign) in the turn's notes.

### Hard rules (fail-closed)

1. Never read the checker/verifier implementation as authoring input
   (`runtime_validation.py`, contract checks, the contract tool source, any
   oracle/reference tree). Your authority is the candidate's runtime behavior.
2. Every turn moves the kernel body or a measured lever past the last HEAD.
   Re-sealing an unchanged HEAD is not progress.
3. Runtime correctness is the only authority. When relL2 / launch count / path
   marker disagree with a static check, fix the behavior the runtime saw.
4. No debug code in a committed HEAD: no timeouts or watchdogs, in-kernel
   timestamps, stage-skip switches or experiment environment variables. Keep
   such instrumentation in a scratch copy outside the lane.

## Blocking invariants (every checkpoint)

Behavior, not spelling.

1. `AITER_MEGAMOE_FUSE_ALL=1` (exact) selects the fused path for supported
   buckets; anything else runs the frozen path. First forward of each operator
   instance prints `[megamoe] path=MEGA rank=<r>` (fused) or
   `[megamoe] path=FALLBACK rank=<r>` (switch on, bucket outside the fused
   range). Never mix paths across ranks.
2. The fused call is at most two launches: dispatch, GEMM1, GEMM2 and combine
   run in the stage1 persistent grid; input quantization runs in that grid or
   as one separate kernel before it. The host returns the combine output view.
3. Whole-grid residency: every CTA of the grid is resident at once (≤ 2
   CTAs/CU with ≤ 256 VGPR and `2 × LDS ≤ 160 KiB`), asserted on the host. Any
   wait on another CTA (ready word, reducer) deadlocks otherwise.
4. Cross-rank words are monotone (epoch / parity) and graph-replay safe; no
   launch handshake, no reset of peer-visible state. Mixed bucket/grid sizes on
   one operator share one epoch sequence.
5. Control words broadcast through LDS have their own slot; a word must not be
   overwritten before every wave has read it (the baseline's `work`/`has_work`
   race). Each wave derives `has_work` itself.
6. GEMM2 on an m-tile starts only after that m-tile's GEMM1 tiles are counted
   complete (stores retired before the count). No grid barrier.
7. Combine reduces top-k components in a fixed order in FP32 (bit-identical to
   the separate combine kernel).
8. The switch-off path of the candidate tree stays within noise of the frozen
   baseline; only the LDS-race fix may change its bits.

## HBM rules (measured; break them only with a paired A/B)

- Weights are read once, with per-workgroup contiguous chunks and nontemporal
  loads. Overlap through bytes that must be read anyway: issue B before the A
  gather, keep B in flight across K-steps and across tiles.
- Never add early HBM reads during the dispatch front end (prefetch, warm-up,
  preload): they always lost (+1 to +10 µs).
- Do not steal work across XCDs: the stolen unit re-reads A from HBM.
- Check the rejected ledger in skill.md before trying an idea.

## Memory-ordering rules (gfx950; see skill.md Mechanism 6)

An agent- or system-scope release fence writes back the issuing XCD's L2; an
acquire invalidates it. They are correct but expensive, so place them by role:

| handoff | producer | consumer |
|---|---|---|
| GEMM1 tile → GEMM2 on the same XCD | `s_waitcnt(0)` + barrier, thread 0 agent atomic count; the completing count raises the ready word (atomic max / store). **No release fence.** | poll ready word, then invalidate L1 only |
| P2P row / peer flag (uncached shmem heap) | `s_waitcnt(0)` + barrier; store completion is visibility. **No per-CTA release.** | poll flag, invalidate L1 only |
| exit ticket → combine signal | every CTA: `s_waitcnt(0)` + barrier + agent ticket, no fence; the **one** last ticket holder: system acquire + system release, then signal peers | reducers wait for all peer flags, invalidate L1 |
| work-counter reset before the gate | one release by the owner, once per launch | — |

- A fence per tile or per CTA is a defect, not caution: per-CTA system release
  measured +54 µs; per-tile agent release plus XCD-interleaved GEMM1 left a
  single kernel ≈ 75–100 µs slower than the ledger.
- The same-XCD rule depends on the GEMM1 split: GEMM1 units are XCD-contiguous
  m-major slices and GEMM2 units for those m-tiles are claimed from the same
  XCD counter. If a GEMM2 unit can run on another XCD than its GEMM1 tiles,
  the handoff needs an agent release/acquire pair, so fix the split instead.
- Only CTAs that executed work take exit tickets; CTAs past the work bound
  leave without touching shared atomics.

## Completion counters and hang checklist

The validated layout for GEMM1 → GEMM2 readiness, which never needed a
per-launch reset:

- One monotone counter per m-tile, each on its own 64-byte line, never cleared:
  `old = atomic_add_agent(cnt[m], 1)`; when `(old + 1) % N_TILES == 0` the
  same thread raises `ready[m]` with `atomic_umax_agent(ready[m], gate_epoch)`.
  GEMM2 waits for `ready[m] >= gate_epoch`. `gate_epoch` is the launch epoch
  every CTA already derives from its entry ticket, so mixed grid sizes and
  graph replay stay consistent.
- Counter and ready addresses are extra slots of the existing dispatch table
  (table size derived from the largest slot), allocated once with the
  workspace. A second pointer table or a per-launch clear loop adds an
  ordering edge (clear vs. first count) that a late CTA can lose.
- Count only after `s_waitcnt(0)` + barrier of the tile's stores.

When a fused build hangs, check in this order before redesigning:

1. Residency: read VGPR/SGPR/LDS of the emitted kernel (compile-only run,
   kernel metadata). VGPR > 256 or `2 × LDS > 160 KiB` halves residency and
   every inter-CTA wait (entry gate, ready word, reducer) deadlocks. A
   single added atomic or live value can cross the limit.
2. Addresses: every word a CTA waits on or atomically updates is inside its
   allocation for the largest token count and grid size (index × stride).
3. Reset ordering: no word is cleared by one CTA while another CTA may
   already count into it in the same launch.
4. Work order: a unit that waits (GEMM2) is never claimable before every unit
   it waits on (its GEMM1 tiles) has been claimed by some resident CTA.
5. Wait predicate: a monotone word is waited with `>=`, never `==`. A fast
   peer can already have written the next epoch before a late block polls.
   The baseline combine cross-device barrier has exactly this `==` wait and
   hangs intermittently when stage2/combine graphs replay back-to-back.

## FlyDSL idioms

- Runtime control flow exists only in compiled functions. FlyDSL rewrites
  Python `if`/`for`/`while` on runtime values into device control flow inside
  the kernel body and inside helpers decorated with `@flyc.jit`; a plain nested
  closure runs as ordinary Python at trace time and raises `cannot evaluate
  dynamic Boolean`.
- Masks and `select` choose VALUES; they never decide WHO performs a side
  effect. Keep thread-0 atomics and lane-0 waits as such.
- Put a workgroup barrier between two thread-0-only regions that are not
  already separated by one, including across the persistent loop's back edge
  (a completion count at the end of an iteration and the next work claim).
  Without it the divergent regions can be merged and the other lanes read a
  stale work index: the fused loop hangs as soon as the count is added.
- Keep copies 16-byte vectorized as in the baseline; resolve scalarization
  errors by matching the baseline's operand form, not by splitting stores.
- Build descriptors inside the unit frame; long live ranges across the
  persistent loop spill. Watch VGPR: the validated kernel has 6 VGPR of
  headroom below the 2-CTA/CU limit; read it from the emitted metadata after
  each change. The budget is 256 registers per lane in total (arch VGPR +
  AGPR, `rocdl.waves_per_eu` = 2); a profiler VGPR column alone (e.g. 128)
  is not the budget. Read `.vgpr_count`, `.agpr_count` and
  `.vgpr_spill_count` from the code object before attributing a loss to
  register pressure; a K-loop restructure that loses without spills is a
  schedule effect.
- Exact-size buffer descriptors let out-of-range prefetches be dropped by
  hardware; keep them exact when prefetching past the last tile.

## Measuring

- Use `MEGA_AB_TOOL` with the frozen tree as the first arm and the lane HEAD
  with `#AITER_MEGAMOE_FUSE_ALL=1`, `--expect-path cand=MEGA`, ≥ 3 passes.
  Graph noise is ≈ ±1 µs; a lever below 1 µs needs ≥ 5 passes.
- Timeline stamps in a scratch build are thread-0 side effects: they follow
  the barrier idiom above and must not add live values past the VGPR limit.
  A stamped build that hangs is dropped for the turn; it never blocks landing
  a ledger step in the lane.
- Correctness screens: official accuracy at the target size, 100+ repeated
  calls (bit-identical output for identical input), mixed 1..mtpr token counts
  on one operator, ragged per-rank token counts.
