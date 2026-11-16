# MegaMoE EP fusion — Author guide

Continue the lane the direction names. The frozen baseline is only the root
commit; never recreate the lane or inspect external implementations. When the
prompt carries `REWIND_TO`, roll the lane back to that commit first (the old
HEAD stays tagged) and continue from there.

Treat `BASELINE_OPERATOR_MAP` and the supplied baseline kernels as concrete
implementation references. For each region, choose reuse, extraction, inlining,
or equivalent rewriting to fit the persistent schedule. Function identity is
not required. Preserve observable math, quantization, dtype/layout, routing and
top-k behavior unless a deliberate change passes current-run correctness.

## Development rule

Fix the earliest concrete blocker returned by Verify, commit it, and return the
same lane for independent verification. Preserve working numerical bodies and
already-passing evidence. Do not write token-shaped stubs for contract checks.

If preflight still has source failures and no GPU blocker exists, implement the
next dependency-closed stage below. Re-sealing an unchanged HEAD twice is not
progress.

### Hard rules (fail-closed)

1. Never read the checker/verifier implementation as authoring input. The
   contract's AST/check code, PlanIR checker fields, `runtime_validation.py`,
   the contract tool source, and any oracle/reference tree are OFF LIMITS for
   shaping source. The contract tells you WHAT to implement, never HOW. Your
   authority is the candidate's own runtime behavior (compile, on-card smoke,
   relL2, launch count, path marker). If you find yourself studying how a check
   computes its verdict, stop and author the code that check is asking for.
2. Every lease must move the fused kernel BODY past the last committed HEAD.
   Once Host/ABI is closed, committing only more host/ABI/spec scaffolding is a
   failed turn: two consecutive turns landing on the same rung is forbidden, and
   `body=deferred` is never an acceptable output once the host shell exists.
   Each lease commits at least one new dependency-closed body stage of real
   device code that the persistent kernel executes.
3. Runtime correctness is the only authority. When the observed relL2 / launch
   count / path marker disagrees with what a static check "wants", fix the
   behavior the runtime saw, not the shape that satisfies the checker.

## Blocking invariants (every checkpoint, Stage A included)

These are behavior, not spellings: names and file layout are free, the behavior is not. A
checkpoint that violates one is a defect even when it is correct today, because every later
optimization is built on it. Verify reports them as `semantic`-tier contract failures.

1. G1 work unit is one flat `(m_tile, n_stripe)` stripe claimed from its OWN 64-byte shard head
   (`claim = shard + local * SHARDS`). Each stripe publishes its own completion: cache-17 stores,
   `s_waitcnt` in every wave, one block barrier, ONE thread-0 system increment. Never claim a whole
   m-tile and run its stripes serially in one CTA.
2. G2 claims come from sharded 64-byte heads in contiguous chunks of claim positions; a new chunk
   waits only on its last in-range unit. The continuation carries two block-uniform scalars (next,
   remaining) and issues NO atomic, NO readiness poll and NO LDS round-trip. Positions map to units
   through the G2 claim order (skill.md, "G2 claim order"): column-grouped GM=32 with C1 is the
   measured best; C16 at >=4096 tokens belongs only to the m-major reference order. The mapping is
   a bijection over all units, so every CTA can claim every unit.
3. Skew preemption happens only when `ticket % 6 == 0` AND
   `max_expert_tiles * experts_per_rank * stage1_tile_m * 4 > valid_rows * 5` (the hottest expert's
   padded rows exceed 5/4 of the balanced share). About 1/6 of CTAs are eligible, and on a uniform
   route none preempt.
4. The scheduler broadcasts only `kind` and `unit` through LDS; no multi-word LDS scheduler frame.
5. Stage2 runs as one native NW=8, BN=512 slab (never two lockstep 4-wave halves); descriptors,
   route, weights and peer resources are built inside the single-unit frame.
6. Stage2 publication per m-block: every wave fences, one block barrier, thread 0 does one
   agent-scope close increment; the block that closes the m-block self-clears the close counter and
   its first BM threads each publish one remote arrival in parallel. One atomic per publishing
   thread — never an all-lane masked atomic.
7. Combine is the irreversible third queue: one block claim serves NW items, lane 0 alone waits for
   `topk * generation`, loads use cache 19 with no acquire fence, the output store uses cache 2.
8. Arrival counters are never cleared: the Combine generation increments and omitted tokens are
   padded to the current target. No per-generation resets, parity banks or re-arm sweeps.
9. Copies stay 16-byte vectorized (as in the baseline) on every shared path.
10. The unfused path of the candidate tree (switch off) stays within noise of the frozen baseline.
    If it does not, the change regressed shared code; fix that before anything else.

Stage A (correctness first) may add ONE simplification: a coarse barrier before Combine. It may not
break any invariant above.

## FlyDSL idioms

- Runtime control flow exists only in compiled functions. FlyDSL rewrites Python `if`/`for`/`while`
  on runtime values into device control flow inside the kernel body and inside helpers decorated
  with `@flyc.jit`; a plain nested closure runs as ordinary Python at trace time and raises
  `cannot evaluate dynamic Boolean`. When a helper needs "only thread 0", "only lane 0" or "only
  valid rows", decorate it with `@flyc.jit` and write the `if` (the baseline's `_publish_tile_range`
  in `dispatch.py` is such a helper), or move the code into the kernel body.
- Masks and `select` choose VALUES; they never decide WHO performs a side effect. Replacing a
  thread-0 atomic with an all-lane masked atomic, or a lane-0 wait with a whole-wave spin plus a
  fence, multiplies atomics/polls by 64-512 per event and is a performance defect.
- A backend error such as `Do not know how to scalarize this operator's operand` on a vector buffer
  store means an operand no longer matches the baseline's working 16-byte copy. Compare resource
  uniformity (`readfirstlane`), dtype and offset type with the baseline and restore that form. Do not
  split the copy into 4-byte stores: that multiplies instructions and remote transactions on the
  shared dispatch path and slows the unfused path too.
- Keep the G2 continuation as scalars carried by the outer loop. A nested loop around the GEMM2 body
  makes its whole live range loop-carried; the resulting spills cost about 2.6x.
- Build descriptors and per-unit resources inside the unit frame; hoisting them into the outer
  scheduler frame extends live ranges into Combine and spills SGPRs.

## Performance budget (8192 uniform, EP8 gfx950, rank-max)

The whole fusion gain is ~0.2 ms, so a single expensive construct erases it. Use these measured
costs to judge a design before tuning knobs:

| item | measured |
|---|---|
| frozen 4-launch baseline | ~4.70 ms |
| Stage1+Stage2 persistent, Combine still a separate launch | ~4.62 ms |
| full fusion target (Combine as third queue), m-major G2 order | ~4.50 ms (~1.045x) |
| same structure, column-grouped G2 order (GM=32, C1) | ~4.05-4.07 ms (~1.155x) |
| G2 order GM=1 / 8 / 16 / 32 / 64 at C1 | 4.681 / 4.419 / 4.089 / 4.047 / 4.023 ms |
| G2 chunk C1 / C2 / C4 / C16 at GM=16 | 4.089 / 4.202 / 4.424 / 4.880 ms |
| column-grouped order on rank-mixed-skew, without / with the gated interleave | 0.58x / 1.21x |
| Stage2 + Combine, measured separately | ~2.08 ms |
| two lockstep Stage2 halves instead of one NW8 slab | +0.53 ms |
| per-block L2 writeback instead of cache-19 write-through P2P stores | +0.41 ms |
| skew preemption active on a uniform route | +0.30 ms (saves 0.51 ms on skew) |
| quantization work-stealing atomics | +0.30 ms |
| nested G2 loop around the GEMM2 body (spills) | ~2.6x |

Hot-path budget per unit of work: G2 continuation 0 atomics and 0 polls; new G2 chunk 1 claim and 1
wait; G1 stripe 1 claim and 1 system increment; Stage2 m-block close <= 1 atomic per n-block plus <=
BM remote arrivals; Combine 1 claim per NW items and 1 lane-0 poll per item, no fences. When the
candidate misses the target, first measure where the time goes (ablation: disable one component in a
diagnostic build and re-time; or per-CTA phase timestamps) and compare each component with the
budget above; do not tune occupancy, chunk sizes or barriers blind. A fused G2 whose span is still
near standalone Stage2 (~2.0 ms at 8192) is not yet reusing W2; check the G2 claim order and chunk
(sweep GM and C on fresh inputs) before any other G2 lever.

## Dependency-ordered build recipe

Do not rewrite GEMM math. Reuse the baseline quantizer, GEMM1, GEMM2, weighted
scatter and Combine numerical bodies; fuse their ownership and scheduling.

### 1. Host/ABI

- Allocate persistent Stage2/Combine payloads, outputs, peer tables and four
  disjoint counter domains: G1 completion, G2 claims, Stage2 close, and Combine
  claim/generation.
- Build immutable Stage2 and Combine compile specs from one selected
  `p2p_quant`; include `NW` and all codegen switches in the JIT key.
- Pass dynamic roots as declared kernel runtime parameters in exact signature
  order. Do not capture pointers or blindly splat tuples.
- Select exactly quant + one persistent launch and return its Combine output.

Done: the active Host values reach the cached compile call and real launch ABI.

### 2. G1 publication

- Claim one flat G1 stripe, wait dispatch payload, execute baseline GEMM1.
- Store activation/scales with cache 17.
- waitcnt every wave, uniform barrier, then thread-zero system increment once.

Done: G2 waits read only the G1 completion domain.

### 3. Shared G2/P2P emitter

- Build one single-unit `(m_block,n_block)` emitter shared by standalone and
  persistent paths.
- Thread Host `NW` into A-load, GEMM2, scale and scatter indexing.
- Use the caller's materialized `SharedAllocator().allocate(...).peek()` handle;
  `.buf.ptr` belongs to that handle, not a raw storage value.
- Weight and store the peer row with cache 19; release, barrier, close the
  dedicated m-block counter, and publish one arrival per valid row.

Done: standalone and persistent distributors call the same numerical emitter.

### 4. Unified G1/G2 scheduler

One block-uniform loop chooses in this exact order:

1. carried G2 continuation;
2. ready skew-preempt G2 claim;
3. G1 claim/execute/publication;
4. blocking post-G1 G2 claim;
5. terminate.

Use C1 below 4096 tokens and C16 otherwise; clamp the final chunk. Continuation
carries only next/count and performs no claim or wait. Never append a second G1
drain after this loop.

Done: a CTA enters Combine only with no continuation and exhausted G2 shard.

### 5. Combine third queue

- Claim one block; item is `claim*waves+wave`.
- Lane zero strictly waits for `topk*generation`.
- Read all direct top-k slots with cache 19; decode from the same `p2p_quant`
  as Stage2; reduce FP32 through one U1/U2/U4 path; store BF16 with cache 2.
- A bounded diagnostic wait must be removed before the production checkpoint.

Done: no Stage2/Combine tail launch remains.

### 6. Lifecycle and acceptance

- Clear claim heads, increment generation, pre-arm omitted tokens, keep arrivals
  monotone.
- Verify active `path=MEGA` ×8 and two launches on the shared large-token operator (at most two
  for an `_mtpr` target) before using numerical evidence.
- Then run 128→512→8192 accuracy, route-changing replay, and finally paired
  performance on the target guard (8192-uniform unless the workflow names another).

After a source edit, CPU checks are preflight only. The next independent Verify
must execute the exact HEAD on eight GPUs.

With `AUTHOR_GPU_MODE=bounded_smoke` (the normal candidate-validation turn) the
Author runs bounded construction/JIT/correctness smokes of the HEAD being
edited, before and after source edits, at any contract state; a production fix
never revokes the device. This is development evidence only; final
correctness, liveness and performance remain independently verified.
Every such smoke must export `AITER_MEGAMOE_FUSE_ALL=1` plus the declared
Combine/quant switches, use `--mega-only` when the harness supports it, and
observe `path=MEGA` on all eight ranks. Otherwise it measured the scattered
fallback and is void.

Do not scaffold-and-bail. While the EP8 lease is available, keep the same turn
open across write → trace/JIT → smoke → traceback-guided repair. Return only
after the staged implementation produces a real smoke result or a concrete
bounded failure that requires the next turn.

When `PERSISTENT_JIT_CACHE_DIR` is supplied, export it as the FlyDSL/AITER
runtime cache root for every Author smoke. It is outside the candidate Git tree
and persists across Workflow waves; never substitute a per-workspace cache.

## Runtime progression

1. Host construction must create every Stage2/Combine buffer and pointer table.
2. FlyDSL trace/JIT must complete for the selected profile.
3. All ranks must report `path=MEGA`; profiler must observe the bucket profile's launch count.
4. Pass relL2 <0.10 at 128, 512, and 8192.
5. Pass changing-route and smaller→larger graph replays without stale reads/hang,
   including the `moving-hot-rank` route: the first replay after a route change
   must match a repeated replay (relL2 < 0.01), not only the reference (< 0.10).
6. Tune only after correctness; accept only a frozen-relative 8192-uniform
   speedup >=1.03x (frozen `BASELINE_PER_CASE` latency / candidate latency), or
   the workflow's own target guard and minimum when it names one (always >1.0x;
   an `_mtpr<M>` target must run `path=MEGA` with at most two launches). A
   same-tree switch-off arm is a diagnostic and must stay within noise of the
   frozen latency.

## Implementation invariants

- Host bundle values must reach cached compile specs and the real launch ABI.
- Treat reachability as load-bearing: `NW` and `p2p_quant` must flow from the
  selected Host spec through the cached compiler into every A-load, GEMM2,
  scatter and Combine address/format calculation. A same-named constant or dead
  helper does not satisfy this invariant.
- Cached specs are immutable/hashable; output returned by Host aliases kernel output.
- G1 completion uses cache 17, waitcnt, uniform barrier, then one system publish.
- G2 has its own sharded extent and readiness counters; continuation drains C1/C16
  without reclaiming; skew is 5/4 and gated by ticket mod 6.
- Shared Stage2 owns descriptors/A preload and serves both persistent and standalone
  paths; 8192 uses BM64/BN512/BK256/NW8.
- P2P payload and E8M0 stores use cache 19 before release/close; each routed row
  publishes to its decoded peer/token readiness address.
- Combine decodes block-FP8+E8M0, reduces exactly top-k in FP32, selects one U1/U2/U4
  main path, stores BF16 with cache 2, and uses grid-wide runtime partitioning.
- Arrival counters are monotone. Reset only the rank-local claim head; increment
  Combine generation and pad omitted tokens before PLAN_READY/startup publication.
- LDS uses max-of-role-lifetimes and must remain <=163840 bytes.
- Every workgroup that reads per-launch route metadata (pair order, grouping
  table, dispatch counts) waits on that launch's readiness flag itself. Do not
  put the wait inside a role or grouping branch: with a heavy source-to-expert
  pair (more than 4 payload producers per destination) some payload producers
  sit outside adaptive grouping. They then read the previous launch's order,
  which corrupts rows silently or hangs on a changed route. The frozen tree's
  dispatch has this defect; keep the wait unconditional in any dispatch you carry.
- The fused G2 BN is `block_n * NW / 4`, exactly 512 at 8192. The emitter covers at most 512
  columns per unit; a wider BN silently leaves columns unwritten, and a sequential sweep that
  reuses the same input buffers hides it behind the previous variant's output. Assert BN <= 512
  and give every sweep variant fresh inputs.

## Codegen levers (measured at 8192)

Read the emitted VGPR/SGPR, spill and private-segment metadata first; each item below was found
that way and kept the output bit-identical.

- G1 last K-step A offset: a compile-time-constant ping/pong offset let LLVM hoist the address out
  of the loop and spill it (32 extra scratch reloads per tile). Anchor it to the per-tile row base:
  VGPR spills 53 -> 20, private 216 -> 84 B, about +1%.
- G1 SiluQuant epilogue: process 8 columns per lane (one 8-byte FP8 store, 2 amax shuffles per 8
  columns): epilogue 245 -> 164 us per workgroup, about +1.5%.
- G1 A-tile start row is uniform across the workgroup. Kept in a VGPR it costs 4 `readfirstlane`
  waterfall loops per K iteration; load it into an SGPR: G1 K-loop -57 us per workgroup, about
  +0.85% at 8192. At bs128@mtpr8192 one build measured about -1.4% and another no measurable
  change; measure it per bucket.
- Stacked: the three G1 items above together (last-K anchor, 8-column SiluQuant, SGPR row base)
  measured 4.115-4.127 -> 3.925-3.933 ms on the column-grouped build (-4.6%), spills 75 -> 31,
  private 272 -> 92 B; more than the sum of their separate measurements on another build. Keep them behind a compile key of
  the large-token path so other buckets' kernels stay byte-identical.
- G2 K-loop: the in-order `vmcnt` wait drained the A prefetch behind later loads. Issue B and the
  A scales before the barrier and before the next A DMA: about +3%.
- These GEMM1 edits live in shared code, so they also speed the unfused path. Credit them with the
  same tree's switch-off control, not as fusion gains.

## Failure handling

- A Python/FlyDSL exception is an Author code defect unless evidence identifies a
  harness failure.
- A GPU timeout may be a deadlock. Use the saved process/log classification; do not
  call it an agent failure without checking the GPU command.
- Put a bounded wall-clock timeout on EVERY on-card smoke you launch yourself, sized to a
  few× the EXPECTED runtime (a bs=128 fused smoke returns in well under a couple of minutes;
  a full accuracy/liveness pass in a handful). Wrap each attempt as
  `timeout <bound> bash $SKILL_DIR/scripts/gpu_lock.sh $GPU_ID <cmd>` (or export
  `GEAK_GPU_RUN_TIMEOUT`/`GEAK_GPU_POOL_WAIT` from the supplied `GPU_RUN_TIMEOUT_S`/
  `GPU_WAIT_TIMEOUT_S`). If a run has NOT returned by its bound, STOP and do not wait it out:
  treat the overrun itself as a probable deadlock (mismatched collective order, a readiness
  counter never published, a coordinator↔worker cycle, a counter cleared into the next
  generation) — an overrun is a FAILED ATTEMPT with a concrete next probe, never a hang to
  sit through. Immediately classify with `crash_bisection.md` / the bounded bs=128 diagnostic
  path, dump the first unmet generation/address, and kill any orphaned rank processes before
  the next attempt or it will fail on an already-bound port / stale heap. You do not need a
  prescribed procedure to notice this — when the expected-time bound is exceeded, self-check
  which stage's readiness counter is not advancing and report that, not a bare timeout.
- Once the same HEAD has produced the same device-sync hang twice, do not rerun the
  uninstrumented command. Add bounded device-side waits and/or per-protocol progress
  counters, dump the first unmet generation/address after a short timeout, then run
  only bs=128 until it produces relL2 or a new bounded failure.
- Bounded waits and cut-point switches are diagnostic-only. Remove them before the
  production checkpoint: final Combine must not continue into payload reduction
  after a readiness timeout, and no diagnostic-off branch may earn correctness.
- A run that reports `path=MEGA` ×8 and two launches but a high relL2 (output
  near-zero or missing most tokens' contribution) is a COVERAGE/LIVENESS defect,
  not an address off-by-one: the publish/readiness path is not firing for most
  tokens. Before touching numerics, confirm on-card at bs=128 that every routed
  token's arrival counter reaches its expected count. Verify the claim loop is
  actually DRIVEN over the entire (block,tile) grid so coverage reaches the bound
  (a single-shot emit with a raw block index, or a scheduler that enumerates one
  fewer owner than the tile count, leaves observed arrivals at zero), that each
  owner publishes exactly once per valid row, and that each consumer waits on its
  OWN tile's counter rather than a coarse last-tile proxy. Dump the first token
  whose observed arrivals are below expected and fix the wiring there.
- A trace-time `cannot evaluate dynamic Boolean as Python bool during tracing`
  means a kernel RUNTIME value reached a Python `if`/`and`/`or` in code FlyDSL does not
  rewrite (a plain nested closure). Fix the location, not the semantics: decorate that
  helper with `@flyc.jit` or move the branch into the kernel body so it becomes device
  control flow (see FlyDSL idioms). A 0/1 mask or `select` is right only for choosing a
  value; never use it to turn a single-thread atomic or a lane-0 wait into an all-lane
  one. When the offending line sits on the SHARED Stage2 path it breaks BOTH the fused
  arm and the unfused path at trace time, so fix it before the next on-card run.
- A DETERMINISTIC on-device `Memory access fault ... Reason: Unknown` + SIGABRT that
  reproduces on every rank at bs=128, activates `path=MEGA` first, produces NO `[RESULT]`,
  and STILL reproduces when the routed scatter store is replaced by a plain global
  `_buffer_store`, is most likely a flyc CODEGEN frame fault (register-allocation
  pressure), not an addressing bug; confirm with that plain-store probe before spending a
  lease on offsets. Fix it by shrinking live ranges while KEEPING the schedule: the unit
  stays one flat stripe and its completion is still published once per stripe, in the same
  outer-loop iteration, right after the per-unit body returns. Move the GEMM body and its
  accumulators into a `@flyc.jit` per-unit helper so they die before the publish, keep
  descriptors inside the unit frame, and keep the continuation as carried scalars. If the
  publish right after the body still faults (seen in a single-launch build), publish
  at the head of the next claim instead: keep the previous work id in an LDS word, then
  `s_waitcnt 0`, barrier, agent atomic. That is still once per stripe. Never
  "fix" a frame fault by claiming a whole m-tile or publishing once after a multi-stripe
  loop: that removes the G1/G2 overlap the fusion exists for, and every later round then
  optimizes a structure that cannot reach the target.
- When the GPU-free structural preflight reports device-body checks at `0 nodes` or a
  broken declared-source->sink edge, the cause is almost always an EXACT-SPELLING mismatch,
  NOT a pruned/branch-selected subtree and NOT missing content. The structural checker is
  pure AST matching: a `depends` relation looks for a specific sink call by exact name and a
  specific keyword by exact case (e.g. the sink keyword must read `NW=`, not `nw=`), and its
  def-use walk abandons any intermediate that is assigned more than once and cannot trace a
  value that arrives as a function parameter; a `count`/assign relation matches unparsed text
  by `re.fullmatch`, so `rows_per_wave = BM // _nw` fails the pattern `BM // NW`. So a
  semantically-correct body fails purely because an identifier, keyword case, or helper name
  differs from the reference surface syntax. Do NOT keep threading "reachability edges" round
  after round chasing this. Each contract failure carries a `tier`: `semantic` failures are the
  blocking invariants above (route them), `surface` failures are spelling/plumbing hints (ignore
  them when runtime evidence shows the behavior is right). `next_blocker` lists every failing
  check ID, semantic first. Because the contract is ADVISORY here (it does not block a runtime
  smoke), the right move once a coherent 2-launch body compiles and imports is to go on-card
  for a correctness result, not to reproduce every exact spelling first.
- Your on-card authorization is `AUTHOR_GPU_MODE`, nothing else. With `bounded_smoke` (`GPU_ID`
  is a concrete device) you may run bounded smokes this turn at any contract state, before and
  after edits; `STAGED_GPU_AUTHORING=0` or an empty structural-evidence HEAD does not remove
  that. Do not invent a "GPU-authorized lease required" blocker. A partially complete structural
  contract is NOT a reason to stay GPU-free.
- Once the fused body is fully authored, do not keep spending leases on GPU-free static
  visibility. The authored body's real semantics (numerics, launch shape, liveness) only
  resolve on-card, never in the AST checker. Take the authored HEAD on-card at bs=128
  (export `AITER_MEGAMOE_FUSE_ALL=1` + the declared switches, observe `path=MEGA` ×8) to get a
  concrete runtime blocker — that is worth more than another round of 0-node chasing, and it is
  where the flyc frame-fault (above) and coverage/liveness defects actually surface.
- Sequence correctness before performance, keeping the fused entry's function interface fixed.
  STAGE A: land a 2-launch `path=MEGA` fusion whose numerics pass (`relL2<0.10` at bs=128, then
  512/8192). The only acceptable Stage-A simplification is a coarse barrier before Combine; the
  blocking invariants (flat stripes, sharded heads, one-atomic publication, monotone counters,
  vectorized copies) hold already, so Stage B does not have to rebuild the scheduler. STAGE B: with
  the interface unchanged, remove the barrier (Combine third queue, lane-0 per-token readiness) to
  reach the ≥1.03x target. Never let Stage-B scheduling completeness gate a Stage-A
  correctness smoke — the two are separable and Stage A is the one that breaks a GPU-free spin.
- Do not run full performance after a failed construction/JIT/small correctness smoke.
- Never claim launch count, correctness, liveness, or speed without on-card evidence.

The full Skill contract is intentionally not part of Author context; independent
Verify evaluates it after each committed checkpoint.
