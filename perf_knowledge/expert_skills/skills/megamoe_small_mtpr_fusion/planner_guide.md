# MegaMoE small-batch fusion — Planner guide

Use this guide after Analyze matches the EP8/gfx950 MegaMoE V2 profile and the
target guard is `<T>_uniform_mtpr<M>` with M in the Skill's range (≈ 32–128).
Outside that range use `MEGA_REGIMES_GUIDE` to decide what transfers.

## Planner job

The Skill is a seed recipe. Plan in two phases on one lane:

1. **Reproduce.** Continue `single_persistent_kernel` on its current lane and
   exact working HEAD, following the skill.md build order (a → e). Route only
   the earliest incomplete step. The first milestone is a correct single
   kernel (relL2 < 0.10, `path=MEGA` on 8 ranks, ≤ 2 launches, mixed-size calls
   correct); speed comes after.
2. **Search beyond.** Once the reproduced candidate is verified, each new
   direction is one lever with a paired graph A/B. Start from the "Remaining
   time" list in skill.md (GEMM1 XCD imbalance is the largest item); never
   re-route an idea from the rejected ledger without a new reason stated in
   the direction.

- Never open a duplicate lane, and never restart from the frozen baseline
  silently. `rewind_head` and `park_candidate_id` are for a lane whose
  committed structure breaks a blocking invariant of `author_guide.md` that
  cannot be restored incrementally.
- Consume `BASELINE_OPERATOR_MAP`: choose reuse, extraction, inlining or
  rewriting per region. Do not mandate function identity.
- The structural contract is ADVISORY: it does not block on-card Verify.
  Route a compiling, importing single kernel ON-CARD early.
- If the same blocker recurs for ≥ 3 rounds with no HEAD move or no measured
  result, route the single earliest evidence boundary of that blocker instead
  of the identical direction.

## Blocker order

1. Host construction, switch and markers.
2. FlyDSL trace/JIT construction.
3. `path=MEGA` on all ranks, ≤ 2 launches (quantization may stay separate),
   whole-grid residency.
4. relL2 at the target size, mixed 1..mtpr calls, then 128/512/8192 on the
   shared operator (FALLBACK allowed there).
5. Graph replay with route mutation, repeated-call determinism, liveness.
6. Frozen-relative rank-max speedup on the target guard.

When Verify returns a traceback or hang classification, route only that
failure. A hang in the fused kernel is almost always residency (grid larger
than resident capacity) or a cross-rank word read from a previous launch.

## Performance routing

- Compare the lane's measured graph time with the gain ledger in skill.md: a
  step that does not reproduce its expected gain is re-measured (≥ 5 passes)
  before anything is built on it.
- Read the ledger as ratios to its baseline row (≈ 344 µs graph). Verify's
  frozen arm reads ≈ 354 µs, so ledger µs × 1.03 ≈ Verify ms. The ledger's
  single-kernel row (≈ 326.5 µs, ≈ 0.95 of baseline) already includes the
  row clamps of step (b) (≈ 5–9 µs of GEMM1). Before step (b), a correct
  single kernel with at most two launches at ≈ 0.97–1.0x of frozen is on
  track: continue with steps (b) → (e), whose rows carry most of the gain
  (grid_mult 2 runs fewer CTAs per CU than the frozen grid_mult 3, ≈ 20 µs
  of GEMM1 bandwidth; the bytes-in-flight rows of step (e) recover that).
- A correct single kernel that is far slower than that (below ≈ 0.95x of
  frozen before step (b)) is routed to the memory-ordering table in `author_guide.md` and to the
  XCD-contiguous GEMM1 split first: count release/acquire fences per tile and
  per CTA, and check that GEMM2 units run on the XCD of their GEMM1 tiles.
  Do not tune tile shapes or launch config before those hold.
- A lane carrying `denominator_mismatch` has a switch-off path that drifted
  from the frozen baseline: route that shared-code regression first.
- While Procedure steps remain unfinished, the skill.md gain ledger is the
  breakdown: route the next unfinished step directly. A timeline is an
  optional scratch side task, never a prerequisite for landing a ledger step;
  if the instrumented build fails or hangs, the round still lands the step.
- Outside the ledger (search-beyond phase, or a step that did not reproduce
  its gain), missing breakdown means the next direction is a measurement
  (timeline or ablation in a scratch build), not a knob sweep.

The complete acceptance contract belongs to independent Verify. Reference
failed check IDs or GPU evidence paths instead of copying its contents.
