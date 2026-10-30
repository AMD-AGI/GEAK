# MegaMoE EP fusion — Planner guide

Use this guide only after Analyze matches the pinned EP8/gfx950 profile and
validates the canonical Skill bundle.

## Planner job

Select exactly one direction: continue `full_persistent_pipeline` on its current
lane and exact working HEAD.

- Never open a duplicate lane or a <2-launch terminal, and never restart from the frozen
  baseline silently. The two explicit exits are `rewind_head` (roll the same lane back to an
  earlier commit of its own history) and `park_candidate_id` (freeze the lane, resumable, and
  start a new one). Use them only when the lane's committed structure breaks a blocking
  invariant of `author_guide.md` that cannot be restored incrementally (for example the G1 work
  unit became a whole m-tile, publication became all-lane atomics, or counters gained
  per-generation resets), or when measured work plus unavoidable wait already exceeds the target.
  Never use them because the current e2e score is low while the structure is still being built.
- A coarse-barrier 2-launch schedule inside the SAME lane is a PERMITTED correctness
  intermediate, not an alternate lane or a partial-fusion fallback: the code accepts a
  correct-but-slower 2-launch landing (verify status `regression`/`slower`) and reports its
  numbers. The barrier is the only allowed simplification; the blocking invariants still hold.
- Never restate the full implementation contract in the Author prompt.
- Choose the earliest unresolved evidence boundary, not a broad rewrite.
- Consume `BASELINE_OPERATOR_MAP`: choose reuse, extraction, inlining, or
  equivalent rewriting of existing quant/GEMM/epilogue/reduction
  implementations according to the fused schedule. Do not mandate function
  identity.
- An unchanged-HEAD evidence re-seal may repair identity once, but cannot consume
  another round while required source checks remain.
- The structural contract is ADVISORY (block_on_expert_skill_contract is off): it does
  NOT block on-card Verify. Each failure carries a `tier`: `semantic` failures mark a broken
  blocking invariant and are routed like runtime blockers; `surface` failures are
  exact-spelling hints and never outrank runtime evidence.
  Once a coherent 2-launch body compiles and imports, route it ON-CARD for a correctness
  smoke (relL2 at 128/512/8192, path=MEGA x8, launch==2) rather than routing yet another
  GPU-free source batch. Do not treat "required source checks remain" as a reason to keep
  the lane off the card.
- If the same blocker recurs for >=3 rounds with no HEAD move or no measured on-card
  result, do NOT re-emit the identical direction. Either (a) route the single earliest
  evidence boundary of that blocker, or (b) if numerics are already correct and the
  remaining checks are scheduling/performance shape, freeze the correct barriered kernel
  and isolate only the overlap edit. Repeating an unmoved blocker is not progress.

## Source batch order

Route only the earliest incomplete stage:

1. **Host/ABI** — immutable Stage2/Combine specs, persistent ownership,
   disjoint counter roots, exact runtime signature and two-launch selection.
2. **G1 publication** — flat-stripe execution, cache-17 stores and exactly-once
   completion publication.
3. **Shared G2/P2P** — one NW-aware single-unit emitter, weighted direct-slot
   stores, m-block close and row arrivals.
4. **Unified scheduler** — continuation → skew-ready G2 → G1 → blocking G2,
   with C1/C16 clamp and no fallback drain.
5. **Combine** — block claim, strict generation readiness, matching transport
   decode, FP32 reduction and BF16 output.
6. **Lifecycle/runtime** — generation, omitted-token padding, replay, launch
   count, three-size accuracy, then performance.

Do not combine non-adjacent stages in one Author turn. A later stage may be
planned only when every earlier stage either passes its required checks or has
one concrete on-card blocker that the later stage is necessary to expose.

## Blocker order

1. Host construction and argument allocation.
2. FlyDSL trace/JIT construction.
3. `path=MEGA` on all ranks and two launches on the shared large-token operator (at most two
   for an `_mtpr` target).
4. relL2 correctness at 128, 512, then 8192.
5. Same-instance graph replay and distributed liveness.
6. Paired rank-max performance on the target guard (8192-uniform unless the
   workflow names another target).

When Verify returns a traceback or hang classification, route only that concrete
failure to Author. Preserve all already-passing source and runtime evidence.

## Fixed target

The target below is the large-token profile. The weight-streaming regime (`max_tok_per_rank`
about 32-128) is covered by the separate Skill `megamoe_small_mtpr_fusion`: optimize it in its own
run with that Skill pinned, and never carry the large-token structure into it. When a run's guards
still span both regimes, give each bucket its own target in the plan.

- Launch 1: existing BF16→MXFP8 per-1x32 quantization.
- Launch 2: one persistent grid containing dispatch, GEMM1, GEMM2, weighted
  P2P publication, readiness, and Combine.
- Stage A (correctness): a 2-launch persistent kernel MAY carry a coarse G1/G2->Combine
  grid barrier while it establishes relL2 < 0.10 at 128/512/8192 on-card. This is the
  required first milestone, not a failure.
- Stage B (performance): removing the grid barrier for the overlapped no-barrier schedule
  is the PERFORMANCE target (blocker step 6 below); introduce it only after Stage A passes
  relL2 at all three sizes. Do not gate the Stage-A correctness smoke on the no-barrier
  schedule being present.
- 8192 profile: 256 CTAs, 8 waves, G2 `64x512x256`, G2 claim order column-grouped GM=32
  (gated skew interleave), chunk 1, skew 5/4, Combine P=1/U=4. The m-major order with chunk 16
  is the 1.045x reference; the column-grouped order measured ~1.155x.
- Acceptance: relL2 < 0.10 and a frozen-relative rank-max speedup >=1.03x (frozen
  `BASELINE_PER_CASE` latency / candidate). A lane carrying `denominator_mismatch` has a
  same-tree switch-off path that drifted from the frozen baseline: route that shared-code
  regression first.
- Performance routing: compare the lane's `component_breakdown` with the budget table in
  `author_guide.md`. A component whose work exceeds its budget is fixed in place; missing
  breakdown means the next direction is an ablation measurement, not a knob sweep. When
  `MEGA_STALL` reaches its limit, route an on-card measurement of the banked HEAD.
- Lever order at 8192 once correctness passes: (1) G2 claim order and chunk, measured on both
  uniform and rank-mixed-skew; (2) the G1 codegen items in `author_guide.md` stacked, checked
  against emitted spill metadata; (3) the G1 payload-spin fill and per-workgroup deferred G2
  completion; (4) the Combine memory order; (5) a two-deep W2 prefetch in the G2 K-loop (open). Readiness and overlap changes come after
  these: G1/G2 overlap measured ~3%, and Combine readiness was not decisive.

The complete declarative contract belongs to independent Verify. Planner should
reference failed check IDs or GPU evidence paths instead of copying its contents.
