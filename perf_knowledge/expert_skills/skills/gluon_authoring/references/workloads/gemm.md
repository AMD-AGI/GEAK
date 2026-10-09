# Workload: GEMM (a16w16 / a8w8 / a4w4)

The canonical tile-programming workload. Its layer roadmap follows the general
backbone (`../method/climb.md ## Layer Backbone`); the GEMM tutorial's measured
checkpoints below are per-layer reference points, not a separate path.

## Logical tiles

```text
C[M, N] += A[M, K] @ B[K, N]
A_tile [BLOCK_M, BLOCK_K] , B_tile [BLOCK_K, BLOCK_N] , C_tile [BLOCK_M, BLOCK_N]
K-loop accumulates partial sums into the AMDMFMALayout accumulator.
```

Default layout chain (gfx950): `BlockedLayout (global) -> Swizzled/PaddedSharedLayout
(LDS) -> DotOperandLayout (k_width) -> AMDMFMALayout(version=4) -> BlockedLayout
(store)` (see `../tile-programming/layout-recipes.md ## Standard GEMM chain (gfx950, FP16)`).
**gfx942 downgrade:** `AMDMFMALayout(version=3)` via `cdna3.mfma`, LDS budget 64 KiB / 32 banks
instead of 160 KiB / 64 banks (re-derive the padding/swizzle, do not copy the gfx950 one), and no
`ds_read_b64_tr` transposed read.

## Usual bound class

Large-K GEMM is **compute-bound** (intensity >> ridge; e.g. M=N=4096, K=8192
FP16 ~1638 ops/byte vs a datasheet fp16 ridge of roughly 290–310 on gfx950 MI350X / MI355X, and
roughly 250 on the gfx942 MI300X downgrade — recompute from `perf_knowledge/hardware/data/sku.json`
rather than carrying these). Primary metric: MFMA efficiency. Small-M/N
or small-K can be latency/wrapper-bound — classify per shape.

**Two things have to be settled before that comparison means anything, and both are cheap.**
They are stated once here because the regimes below all inherit them.

- **The ridge is per dtype, so compute it for yours.** The number above is one dtype's. The ridge
  is `peak[dtype] / peak_bandwidth`, so halving the element width roughly doubles it — a shape
  whose intensity sits between two dtypes' ridges is compute-bound in one and memory-bound in the
  other. The SKU table deliberately does **not** store a ridge for this reason
  (`perf_knowledge/hardware/data/sku.json` `_no_derived_ridge`); `scripts/hw_budget.py` recomputes it per call from
  the dtype you declare. Quoting a ridge from a different dtype's row puts the whole dispatch
  boundary in the wrong place.
- **A memory floor only applies once the working set leaves the memory-side cache.** This target
  has a large memory-side last level behind the L2, and a kernel whose footprint fits in it can
  measure *faster than* its own HBM floor. That does not make the kernel superhuman; it makes the
  floor inapplicable — not merely imprecise — and the compute floor becomes the binding one.
  `Verify:` price the footprint (`--footprint-mb`) against the memory-side cache **before** quoting
  any fraction of peak bandwidth, per
  `../hardware/roofline-models.md ### Zeroth question: does a roofline apply to this kernel at all?`.

## Small-M (decode / GEMV) regime — memory/occupancy-bound

When the parallel (M) dimension is tiny (a GEMV-shaped decode step: one or few
rows times a weight), arithmetic intensity collapses far **below** the ridge, so
the op is **memory/occupancy-bound, not compute-bound** — the metric is a byte
one, not MFMA efficiency. **Which** byte metric depends on the footprint check
above: a decode step's working set is often small enough to sit inside the
memory-side cache, and then "% of HBM peak" is not the denominator — the traffic
never reached DRAM. Settle that first, then judge against the level the bytes
actually came from. The levers are the memory/occupancy ones, not MFMA-continuity:

- **Occupancy by tile count.** A tiny M makes the default (large-N) tiling launch
  too few CTAs to fill the device (`saturation_ratio < 1`, see Stage-Plain below). Raise
  the tile/CTA count: **narrower-N tiles** and/or **split the K (wide-K) reduction**
  across CTAs so more workgroups run concurrently and cover load latency. This is a
  general small-M occupancy recipe (more parallel work), not a fixed tile size —
  sweep it.
- **Quantize-once vs fused-per-tile recompute (fusion trap).** Fusing a per-output
  recompute (e.g. an input re-quant/dequant) **into** the GEMM saves a separate
  launch and can win in **eager**, but it **repeats that work per N-tile** and is
  **inert-to-negative under a CUDA graph** where the extra launch was already free
  (the launch-fusion law, `../hardware/roofline-models.md ## Kernel time
  decomposition`). For a graph-served small-M op, prefer **quantize once + an
  occupancy-tuned GEMM** over per-tile fusion; validate at the production boundary.

### The mirror case: large M, tiny N — the same regime with the traffic reversed

`## Small-M` above assumes the weight stream dominates, which is what makes "more CTAs, narrower-N
tiles" the recipe. A gate or projection whose **N is tiny** (a few tens of columns against a long
K, with M large) is in the same memory regime for the opposite reason, and three things invert:

- **Activation dominates, not weight.** `M*K` is the large term while the whole weight fits in
  cache, so cutting weight bytes buys nothing and the byte account is about the activation stream.
- **"Narrower-N tiles" is already spent.** It was the small-M occupancy recipe; here N is *already*
  below one tile's N, so the parallelism has to come from M or from splitting K instead.
- **Cache polarity reverses with reuse, and reversing it is actively harmful.** At tiny M the
  weight is read once and never reused while the activation is reused across N tiles; at tiny N it
  is the weight that is cache-resident and re-read. The two streams want opposite cache treatment
  in both cases, and getting the polarity backwards does not merely forgo a win — it evicts the
  stream that was being reused with the one that never will be.

**A tile whose N exceeds the problem's N is under-filled at the instruction, not at the warp.**
Moving warps off a degenerate axis recovers warp-level waste and is the right move when the tile is
wider than the warps can cover; it cannot recover a matrix instruction whose own N extent is wider
than the data. When N sits below the instruction's N, the fill question is instruction selection or
a different decomposition — check which shapes the dtype offers before tuning the tile, and treat a
low matrix-utilization reading here as geometry rather than as scheduling
(`../method/profile.md ## Rule: low MfmaUtil has two distinct tile-size causes`).

**The decomposition that does fill it: spend the idle extents on K, then keep the diagonal.** If
the instruction's N is wider than the problem's, the unused columns are not recoverable as columns
— but they can carry a *different K shard*. Fold the reduction axis into the free M and N extents,
issue one full-tile matrix op, and the result holds every shard-pair product when only the matching
ones are wanted:

```text
acc  -> reshape (M, FOLD, N, FOLD)        # FOLD K-shards on each free axis
keep  where fold_i == fold_j              # the diagonal: shard k against shard k
sum   over both fold axes                 # the cross terms were never wanted
```

The trade is explicit: `FOLD^2` products computed for `FOLD` useful ones, paid in accumulator
registers and in the two-axis reduction, against a matrix pipe that was idle anyway. It is worth
it exactly when the instruction was under-filled and not otherwise, so price it against the
matrix-issue floor rather than adopting it because the shape looks degenerate.

Two details decide whether it works, and both are easy to get wrong:

- **Where the select goes relative to the reshape is a form to check, not an order to follow.**
  Masking the accumulator before the reshape and selecting on the folded axes after it express the
  same values, and it is tempting to read one as the cheaper addressing. Nothing in this pack
  guarantees that, so read what the loop actually emitted rather than adopting an order — and note
  the cost the folded-axis form can carry: a select over the folded axes may lower to a cross-lane
  exchange that costs more than the packing saved, which is the opposite of the trade this
  decomposition exists for. A rewrite whose justification rests on the ordering alone is a gap to
  take through `../method/triage.md` (`missing_lowering_behavior`), not an assumption to
  build on.
- **Use a select, not a multiply by zero.** The cross-shard products are not merely unwanted, they
  can be non-finite (they pair operands that were never meant to meet). A select discards them;
  scaling them by zero propagates a NaN into the sum.

This is a decomposition rather than a DSL mechanism — the same fold-and-take-the-diagonal shape is
expressible with plain tensor reshape/split primitives, so it is available before any explicit-tile
rewrite is on the table.

## Mid-M ridge (regime transition) — needs its own tile

Between the small-M (memory/occupancy-bound) regime and the large-M
(compute-bound) regime there is a **transition ridge** where neither the small-M
occupancy tile nor the large-M compute tile is well-matched, and a tile tuned for
one side **regresses** on the ridge. Treat the ridge as a **separate dispatch
bucket with its own mid-M tile**. Its location is data/shape-dependent — **find it
by sweeping the served M-range** (`../method/benchmark-hygiene.md ## Served-range sweep`),
never assume it at a fixed M coordinate.

## Layer roadmap (a16w16 FP16)

The order is the content here, not a trajectory: each layer's job is to make the **next** one
reachable, so a layer taken out of order either does nothing or regresses. Read the third column
as "what proves this layer closed", never as a number to match.

| Layer | Mechanism | Closed when — and what it unlocks |
| --- | --- | --- |
| anchor | explicit layouts recovered from TTGIR, `cdna3/4.mfma` | numerics match and the layouts are the comparator's own; this is the attribution baseline every later claim is measured against |
| memory | `buffer_load` -> `buffer_load_to_shared` async (gfx950; **gfx942 downgrade:** async only at 32-bit with `order=[1,0]`, so usually `buffer_load` + register-staged `ds_write`) | the load path no longer round-trips through a register staging `ds_write`, and the loop's branch count collapses to the loop-carried ones; unlocks a pipeline that has something to overlap |
| LDS | padding/swizzle conflict-free `ds_read` | the static audit shows the conflict-free `ds_read` interval for the access width, not a 2/4-way multiple; unlocks pipelining that is not just hiding bank conflicts |
| pipeline | hand-written, in the order `../tile-programming/pipeline.md` defines: register prefetch -> authored LDS ring (2-stage global prefetch + double buffer -> 3-stage local prefetch; `async_copy` + `commit_group`/`wait_group` on gfx950, sync staging on gfx942) -> `warp_pipeline_stage` only at `num_warps >= 8` | the buffers retire in the intended order and the MFMA no longer waits on the same-iteration `ds_read`; matrix-issue efficiency rises at unchanged occupancy. Deeper only if the occupancy budget says depth is the bound. `num_stages` does nothing here (dead on the Gluon path in 3.8.0); re-injecting plain's pipeliner is a below-parity diagnostic / last resort, labelled *injected* |
| slicing | slice-N -> slice-MN | register pressure drops back under the arch ceiling with spills gone, and the loads spread across regions instead of bursting; this is the layer that PAYS FOR the previous one when it over-unrolled |
| beyond | XCD-aware PID remap + `GROUP_SIZE_M` | DRAM read requests fall at unchanged kernel structure — an L2-locality win, so it composes with everything above rather than trading against it |

Each layer is closed with three-evidence gating and a reprofile before the next. **Do not carry
another kernel's per-layer gains as expectations**: the size of each step depends on the shape,
the SKU and the build, and on some of them a layer is worth nothing. The compiler-stack rungs in
particular are an author+compiler co-design that needs the paired source structure and the pinned
build — re-derive and re-verify per kernel / target / build
(`../tile-programming/compiler-contract.md ## Scope: reference for authoring, not a portable recipe`).

**Three levers sit beside this ladder rather than on it.** They do not close a layer and they are
not ordered against each other; each is reached for when a specific reading says so, and each has a
gate worth checking before the first attempt:

- **Per-compile scheduler strategy.** A function-attribute pass-through can select a scheduling
  strategy per compile, which is a granularity a process-wide environment variable cannot express.
  It helps **ILP-starved** bodies (low occupancy with reorderable slack) and is
  **neutral-to-negative on a serial dependency chain with no slack**, so it is a sweep against a
  named bound, never a default. Three things to carry into it, and the chapter derives all three:
  the attribute itself is version-gated; an unrecognized strategy value is **silently ignored**,
  which makes a flat sweep over circulating names indistinguishable from those names not existing,
  so you confirm it took by diffing the generated assembly; and the choice is **per kernel, not per
  archetype** — "GEMMs want this strategy" is not an available claim, and a value lifted from a
  neighbouring kernel still owes you the diff. Mechanism:
  `../tile-programming/llvm-fn-attrs.md`. Gate cell: `../hardware/capability-matrix.md`. Where it
  lands in the layer loop:
  `../tile-programming/instruction-scheduling.md ## llvm_fn_attrs — the portable, per-compile scheduler strategy`.
- **Draining below the group.** A bare wait on the outstanding-load counter can drain to a depth
  the group-level wait cannot express. It is a pacing lever with a real cost, not a readiness one,
  and what you give up is spelled out in
  `../tile-programming/pipeline.md ### Draining below the group: bare s_waitcnt instead of wait_group`.
- **Inline assembly for what the language has no spelling for.** The uses that recur on this path
  are a compiler motion boundary the abstraction does not offer and moving a warp-uniform scalar
  into a scalar register so the address arithmetic stops recomputing it per lane. Two things about
  it that the "escape hatch" framing hides, and both change how you treat it: surveyed **GEMM**
  files carry it at around a third, so it is an ordinary part of this archetype's vocabulary rather
  than an emergency; and its parameters (`pack`, the constraint string, sometimes the asm text and
  the comparison polarity) are **keyed on the tile**, so a site lifted from another shape is not a
  portable idiom. It is still gated, and the gate is the cost: the block is opaque to the
  optimizer, and its most-used class is its least durable across a compiler upgrade. Whether any of
  it is worth its cost on your kernel is not established here and has to be measured. Note too that
  a hand-written atomic does **not** inherit the cache-maintenance the language's atomic intrinsics
  emit for you. The DSL pack's inline-assembly reference carries the two-axis classification, the
  trap ranking and the idiom catalogue.

## Beyond hot loop (L2 / XCD locality)

gfx950 has 8 XCDs, 4 MiB L2 each (32 CUs per XCD on MI355X / MI350X; the gfx942 MI300X downgrade
has the same 8 × 4 MiB L2 with 38 CUs per XCD, so the remap carries over and only the per-XCD
program count moves — read `cus` from `perf_knowledge/hardware/data/sku.json`). Remap program IDs so
workgroups that share `A`/`B` rows land on the same XCD, and pick `GROUP_SIZE_M` to maximize L2 reuse:

```text
minimize  GM + ceil(P / GM)        # P = 32 (programs per XCD dimension)
P=32 -> GM in {4, 6, 8}            # GM=1 is column-heavy and worse
```

Validate with `TCC_EA0_RDREQ_DRAM_sum` (L2->DRAM) before/after.

**Gate the remap — it is a memory/L2-locality lever only.** XCD remap optimizes L2
hit rate, so it helps only when the kernel is **actually L2/HBM-bound**. On a
**latency-bound** kernel (`MemUnitStalled` ~0, HBM a few % of budget — e.g. a
latency-bound attention loop) it improves a non-binding resource and only **disrupts
the default linear L2-streaming order -> regresses**. Also check the default grid is
not **already XCD-aligned**: in SPX mode the hardware round-robins workgroups by
`pid % NUM_XCD`, so if the blocks that share an operand are already spaced a multiple
of `NUM_XCD` apart (e.g. heads a multiple of 8), they already land on one XCD and the
remap only breaks it. Confirm mem-bound + a non-aligned default grid before remapping.

### Worktile scheduling for load imbalance (LPT/SPT makespan)

XCD remap fixes *spatial* L2 locality; the *temporal* sibling is the order SMs/CUs
pick up worktiles when the grid is **load-imbalanced** (causal masking, varlen,
split-K with ragged tails). The grid linearization is a free knob — choose it to
minimize the **makespan** (the last CU to finish), a classic identical-parallel-
machines result, via **longest-processing-time-first (LPT)**: schedule the heaviest
worktiles first so no CU is left holding a long tail after the others drain. This is
**architecture-agnostic** (gfx942/950/1250) and needs no kernel-body change, only a
remap of `pid -> worktile`.

- **Do not naively sort longest-first across everything** — that breaks L2 reuse
  (different batches' KV won't hit L2; loading all heads first can thrash L2).
  Balance LPT against locality: process the **batch axis outermost**, divide heads
  into **L2-capacity-sized sections**, and within a section traverse heaviest-first.
- **Varlen**: the per-batch lengths are runtime data, so enforce LPT with a cheap
  **preprocessing kernel** that sorts batches by per-worktile time and writes a
  cached virtual->actual batch-index map the main kernel reads — sorted traversal
  at no per-run cost.
- Attention causal/varlen specifics (reverse-mblock, diagonal start, SPT for the
  deterministic-`dQ` write order): `../workloads/attention.md ## Scheduling
  (load-imbalanced grids)`.
- **Small-grid caveat — gate by wave count.** The makespan win assumes there are
  enough worktiles to fill several waves; at very small grids (≲~2 waves/CU, i.e. only a
  few worktiles per CU) a longest-first reversal can *worsen* balance by clustering the
  heavy tiles into one wave instead of spreading them. Check `grid / CU_count` first and
  A/B the remap; condition it on worktile/wave count (e.g. only remap when there are ≥N
  worktiles per CU), not unconditionally. The remap is still cheap to test — the point is
  to verify, not assume, on tiny grids.

## Low-precision

a8w8 (BF8) and a4w4 (MXFP4) reuse the slice-MN + beyond-hot-loop design plus a scale side path; see
`../tile-programming/low-precision.md`. a8w8: tile 256x256x128, `mfma_scaled`
"e5m2", M+N slice. a4w4: tile 256x256x256, `mfma_scaled` "e2m1", M+N slice, and a scale path that
is worth reading before copying — the LDS round trip is the earlier form, not the mature one.

**gfx942 downgrade:** neither recipe exists as written — there is no `mfma_scaled` and no fp4
matrix path. a8w8 runs on the regular `cdna3.mfma` fp8 instruction with **FNUZ** operands
(`e4m3fnuz` / `e5m2fnuz`; e4m3fnuz saturates at 240, not 448) and the block scale applied on the
fp32 accumulator as explicit VALU, which puts the kernel in the vector-between-matmul case below;
a4w4 has no matrix-core route and is either an upcast or out of scope.

## Stage-Plain reminder

Do not enter this roadmap unless the escalation gate fired. If `tl.dot` already
lowers to MFMA at high efficiency, or the bottleneck is wrapper/dispatch, stay
plain (Stage-Plain). Confirm with the plain-Triton profile + budget gap first.

## Stage-Plain GEMM (before escalation)

If staying plain, tune here first — the GEMM entries of `../method/front-end.md`:

- **Saturation first**: `grid_tiles = ceil(M/BLOCK_M)*ceil(N/BLOCK_N)`,
  `saturation_ratio = grid_tiles / CUs` (256 on gfx950 MI355X / MI350X; 304 on the gfx942 MI300X
  downgrade — take `cus` from `perf_knowledge/hardware/data/sku.json`). `<1` is critical
  under-utilization -> smaller tiles / more programs before any body work (a
  1024x1024 GEMM with 256x256 tiles is only 16 tiles; 64x64 gives 256).
- **E2E vs steady-state**: a 64x64 kernel at ~1us is setup+writeback dominated; a
  4096^3 kernel at hundreds of ms is steady-state. Tune matrix instructions only
  when steady-state-dominated; otherwise improve fill (shape/tile/dispatch).
- **Config knobs** (legal values are source/target-specific): `BLOCK_{M,N,K}`,
  `GROUP_SIZE_M`, `num_warps`, `num_stages` (test `=3` when `BLOCK_K>=64` and the
  K-loop trip count is high; `=2` for small/register-constrained), `waves_per_eu`,
  `cache_modifier`, split-K. Treat `BLOCK_K` as a coupled perf/resource knob. These are
  **plain-path** knobs: once the kernel is on the Gluon path `num_stages` is dead (3.8.0 has no pass
  that consumes it) and the plain value survives only as the champion's record, and a tile change
  is a `resweep_request` back to the front end rather than a climb lever.
- **`tl.dot` parameter tuning** (one named hypothesis at a time): `acc=tl.zeros`
  vs implicit accumulation; `input_precision="ieee"`; `max_num_imprecise_acc`
  (precision knob — tie to task tolerance). Re-test after Triton/ROCm updates.

## Matrix plan checklist (before any Gluon matrix path)

```text
target + Triton version / dtype path / plain Triton comparator
result layout / operand layouts / k_width / instr_shape
accumulator dtype / store dtype / epilogue+masks / convert_layout placement
capability-matrix status
```

If result layout, operand layouts, `instr_shape`, or accumulator/store dtype are
unknown, stop and gather evidence (`../hardware/capability-matrix.md`,
`../gluon/matrix-reference.md`) instead of guessing.

## Applying the framework to GEMM variants

This is the **GEMM fill** of a workload-neutral variable set (the set, the archetype-mismatch
check and the recompile-depth census: `intake.md`). A new GEMM variant —
grouped / MoE, preshuffled, block-scaled, stream-K, mixed-dtype — is read through the same
variables as any other kernel, not through its own recipe.

Start from what a **plain** GEMM's values are, because two of them are what make the GEMM-class
compiler levers apply at all, and a variant that changes those loses the levers:

- **Which matmuls, which reductions, on which axes.** Plain: one matmul, one reduction over K,
  clean on the parallel axes — which is why the structure decision is usually trivial here and the
  tile decision dominates. **Split-K** moves that reduction across workgroups, so it becomes an
  atomic-or-separate-reduce choice (`## Worktile scheduling for load imbalance`, and the
  recompute-vs-atomic comparison in `../tile-programming/mental-model.md`). **Stream-K** moves the
  reduction *boundary* per worktile, so the epilogue is no longer uniform. **Grouped / MoE** is a
  *set* of matmuls with a data-dependent assignment, which also flips variable 7.
- **What sits between the matmuls.** Plain: **nothing** — the accumulator chain runs matrix to
  matrix with only copies between, and that is precisely the assumption the GEMM-class stack is
  built on. Variants that break it: **block-scaled** (`acc += dot(a,b)*scale` is not what the
  matrix op accumulates — see below), **mixed-dtype / quantized** (a dequant unpack ahead of the
  dot), and a **fused in-loop epilogue**. Once broken, treat the kernel as the
  vector-stage case: the matrix-accumulator-chain knobs are off the table on any build
  (`../tile-programming/compiler-contract.md ## VALU-between-matmul: manual interleave, or author
  a pass`).
- **Where that third category issues, and whether it fits.** For GEMM the third category is
  usually the **scale or dequant pipeline**, not a softmax, but the arithmetic is identical:
  capacity against demand per compute region, with the same two cheap gates first. A low-precision
  GEMM whose unpack does not fit its shadow is over budget in exactly the sense
  `../tile-programming/llir-codesign.md ## Attention: the co-execution budget` describes, and
  `../tile-programming/low-precision.md` is where the dtype-specific costs live.
- **Operand and intermediate footprint.** The dominant term is the accumulator, and it is
  closed-form (see the register-pressure bullet below). This is the variable the slicing layer
  exists to serve, and the one that decides whether a deeper pipeline is affordable at all.
- **Precision side-paths.** Scale scope, descale point, overflow risk for a delayed descale —
  instances below.
- **Data access pattern.** Plain: dense, statically addressed. **Preshuffled** operands carry a
  transformation sequence that must be preserved rather than simplified (below). Grouped / MoE
  addressing is indirect, which puts an address chain into the loop.
- **Does the loop body's composition vary per iteration?** Plain: no — the K-loop is static, which
  is what makes a computed budget describe the real kernel. **Grouped / MoE is the GEMM variant
  where this flips**, and its three consequences are the same ones the attention page works
  through (`attention.md ## The dynamic-body variable`): the address chain is vector work that has
  to be *placed*, a data-dependent trip count removes the static-unroll premise, and the budget
  becomes a distribution rather than a number.

**On this page's shape.** The sections above are organized by **M regime** rather than by variant,
and that is not a different method — it is one variable (footprint against the occupancy and
bandwidth budget) whose effect on GEMM is strong enough to deserve its own dispatch buckets. The
regime is found by sweeping the served M-range (`## Small-M (decode / GEMV) regime`,
`## Mid-M ridge`), which is the honest way to evaluate that variable here. Reading variables and
sweeping shapes are the same framework at different variables, not two competing intakes.

## Preshuffled / block-scaled signals

The two recurring variants worth naming, as **values** of the variables above (access pattern, and
what sits between the matmuls) rather than as standalone recipes:

- **Preshuffled**: on `DistributedLinearLayout`, `reshape/permute/trans` unshuffle,
  K-divisibility assumptions, or packed scale layouts — preserve the
  transformation sequence before tuning the MFMA; wrong answers usually come from
  simplifying the unshuffle, not the MFMA call.
- **Block-scaled** `acc += dot(a,b)*scale`: MFMA accumulates `dot+acc`, not
  `dot*scale+acc`. A Gluon rewrite needs a fresh accumulator per K step + scale
  conversion/multiply/add — keep plain Triton unless the explicit path removes more
  work than it adds (`../pitfalls/negative-patterns.md`,
  `../tile-programming/low-precision.md`). Record scale scope / descale point /
  overflow risk for delayed descale.
- **The software scale group and the instruction's K extent are different numbers, and the ratio
  is the thing to write down.** A scaled matrix instruction's K extent is much wider than the
  hardware scale group it is built from, so one instruction spans **several** scale groups rather
  than one. Do not read a convenient-looking equality between a quantizer's group size and an
  instruction's K as "one instruction, one scale" — derive `instruction_K / hardware_group` and
  carry that many scales per operand row per instruction. The hardware group size is fixed and
  asserted by the scale-layout helper, and a producer's group size is a software convention the
  instruction has no notion of; the `norm` row's group-size trap is the same fact from the
  producer's side.
- **Matrix engagement needs a dtype-aware counter read on this path.** A scaled matrix pipeline
  does not necessarily increment the generic matrix-utilization counter, while a variant that
  upcasts to a wider dtype and uses the regular instruction does. Two dispatch entries for the
  same GEMM can therefore report opposite-looking engagement for the same real behaviour, and a
  before/after across that boundary reads as "the matrix engine went idle". `Verify:` use the
  unified matrix-engagement test in `../hardware/bound-class-signals.md` and state which path the
  measured entry took (`../hardware/optimization-gotchas.md`).
- **Register pressure before unrolling**: `result_acc_bytes = M*N*acc_bytes`,
  `operand_bytes = M*K*in_bytes + K*N*in_bytes`. If one dot tile already consumes a
  large fraction of the VGPR budget, body duplication is likely to spill.

## Stop conditions (matrix route)

Stop when the capability matrix marks the dtype/op `wrong-result` or a
target-specific blocker; the Gluon path adds unhoistable hot-loop `convert_layout`;
a plain/config path wins under the same boundary with no extra Gluon mechanism; or
tuned Gluon has only local winners not worth dispatch.
