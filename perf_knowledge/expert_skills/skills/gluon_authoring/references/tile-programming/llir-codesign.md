# LLIR schedule co-design (layer 4, below the DSL)

Read this when the structure is right, the overlap exists, and the matrix pipe still idles between
MFMAs. Overlap has to exist first (`pipeline.md`). Cycle constants are per shape / target:
`../hardware/isa-mechanisms.md ## Matrix/VALU co-execution — the CDNA-only overlap budget`.

## Route the loop first: who built the overlap

- **Author-built** (register prefetch, authored LDS ring, `warp_pipeline_stage` markers): markers
  survive lowering as scheduler-barrier boundaries, so the LLVM layer sees the author's stages.
- **Compiler-built** (plain's pipeliner, or its re-injection on the Gluon path — mainly a diagnostic,
  numbers "injected"): no markers; regions can only be *inferred* from instruction order, and plain's loop
  shape (LDS read at top, global load + LDS write at bottom) often forms a single region. The dumped IR
  (`kernel_workflow/scripts/kernel_tools/dump_ir.sh`) shows whether regions formed. Re-injected staging and authored regions cannot
  coexist in one loop (v3.8.0 `isSafeToPipeline` skips a loop with a barrier); an authored ring plus
  instruction scheduling on top can.

## Region routing: the model is a property of the region

```text
region contains          ->  model
mfma + memory, no valu   ->  THROUGHPUT   (pair each memory op with enough MFMA to cover its issue)
mfma + valu, no memory   ->  CO-EXECUTION (each vector op must sit in a specific MFMA's shadow)
mfma + valu + memory     ->  neither (no model here)
no mfma                  ->  phase pacing only

per region:  capacity = (#MFMA in region) x window_for_that_shape
             demand   = sum of non-matrix op issue costs;  exposed = max(0, demand - capacity)
```

**Throughput regions are expressible by hand** (the two classes come from two waves; staging + a
wave-level phase offset deliver it), so a flat scheduling result there more likely points at layout /
staging than at tuning. **Co-execution regions are not**: the 3.8.0 Gluon API has no way to say which MFMA's
shadow a vector op lands in. Derive the window from the MFMA the region actually contains;
carrying one shape's window to another over-fills by the ratio of their intervals.

## Attention: the co-execution budget

Gate, in order: (1) two resident waves per SIMD (vector work shares a wave with the MFMA while memory
runs in the other wave; below that, `scheduling-model.md` covers the options); (2) bound class is matrix
issue; (3) `demand <= capacity` per region on paper — if not, **no ordering wins** within this model
(exposed = demand − capacity regardless of order); shrinking or moving work is what changes it.

Balancing the regions that share a budget:

- **The move must be free.** `ttgl.amd.slice` of a distributed tensor is free **only if
  layout-preserving** (register-only partition; `slicing.md ## Slice recipe (ttgl.amd.slice)`); a
  non-preserving slice costs a shuffle that eats the imbalance you were recovering.
- **A producer moves together with its consumer**, carrying the raw value across. A pure elementwise
  producer left behind is dragged toward its consumer (no chain edge holds it; see `MachineSink` below).
- **Cut points are set by the dependency chain.** A reduction pins itself and everything upstream; a cut
  works only where one value crosses. In streaming softmax the **natural cut is after the `exp`** (its
  consumers, row sum and downcast, are both downstream; AMD's gfx950 attention tutorial derives the same).
  Per column slice the cut can also sit before the subtract (the raw slice carried across), which is the
  balancing tunable: how many column slices are cut there vs carried raw (halves = 1 choice, eighths = 7).
- **The ratio is per (head-dim, block-N, dtype)**; it was not found to transfer across shapes.
- **Budget = estimate with error bars**: the backend adds address math the inventory did not price, so
  visible room in both regions is safer than a configuration that just fits.
- Removing vector work (skip a rescale, fold a scale upstream) only **frees** budget; attribute the
  enabling change and the change that spends it separately.

`coexec` is the stock answer for this region class on gfx1250 only (`compiler-contract.md ## What
upstream 3.8.0 actually gives you`); on gfx950/gfx942 balancing is by hand.

## Declaring the interleave

Gluon 3.8.0 does not expose `sched_group_barrier` (`instruction-scheduling.md ## The declarative hints that
are not there`), so declaring is for pass authors (a plugin or rebuild). Facts about
**IGroupLP** (the AMDGPU group-pipeline builder in the machine scheduler) that explain silent no-ops:
it **scans upward**, so a declaration at the top of a region yields empty groups; declare in issued
*instructions*, not elements; and **an unfillable group abandons the pipeline for the whole region**
(ops folded into source modifiers or fused reduction steps vanish before issue). A bare "do not cross"
barrier is advisory: codegen can still consolidate the last sub-regions; only `sched.barrier(0)` is a wall.

## Two preconditions the kernel must protect

- **Placement:** `MachineSink` drags producers out of their balanced region — `DISABLE_LLVM_OPT=disable-machine-sink`
  (stock 3.8.0); diff the `.s` region with and without it.
- **Form:** `SIPreEmitPeephole` splits packed f32 VALU that lands in an MFMA shadow (packed f32 cannot
  co-execute there; packed f16/bf16 co-issues like scalar VALU, measured on gfx950). Packed math lets it split only there, while blanket scalarization costs two issues where
  one packed op retired two elements — though on one VALU-heavy indexer body a scalar fold measured ahead of
  the packed one, so the form is worth an A/B (if aiter is available, the same note is a comment in `aiter/ops/triton/_gluon_kernels/gfx950/attention/sparse_mla_fp8.py`;
  `../hardware/isa-mechanisms.md ## Instruction-rate facts that flip a lever's sign (CDNA3/4)`).

## Phase pacing: moving a burst instead of removing work

Where a region's 3-source vector work coincides with the other wave's LDS-return burst, they collide on
register-file read ports; register reassignment inside the region did not fix it in the cases examined.
Two levers: **move the burst** (a small fixed `s_nop` delay at the head of the memory stage, `MEMNOP`; the
response was not smooth, so the small integer range is swept exhaustively) or **remove the third source** (fold a scale into an operand
loaded once upstream, `SCALE_ON_Q`; count 3-source VALU ops before/after).

**An `s_nop` right before a DPP / `permlane` / cross-lane op is an ISA-mandated wait state**, not padding
or pacing: removing it gives wrong results. Discriminate by the next instruction, not the count.

## The plugin tier

Not demonstrated on the stock 3.8.0 wheel (`TRITON_EXT_ENABLED` is a CMake option, off by default; the wheel's
`libtriton.so` does read `LLVM_PASS_PLUGIN_PATH`). Reported, not re-tested here: an out-of-tree LLVM pass
plugin fails to load with an undefined symbol on a hidden-visibility build, and even on a default-visibility
build upstream builds **no target machine** when a plugin is set, so plugin-on vs
plugin-off is not a clean A/B. Fork-only, if you have such a build: the plugin must match the exact LLVM
commit (`cmake/llvm-hash.txt`; a moved pin segfaults), and it resolves host symbols only on a **cold**
compile (Python's local symbol scope) — a warm cache hides a broken plugin, so prove it ran.

## Applicability gate: shapes the tool does not model

A schedule tool's per-shape cost model **silently skips** any region whose MFMA shape it cannot price;
this reads exactly like a hardware ceiling. Check which `v_mfma` shapes the loop emits and whether the
tool (and its window model, calibrated per target — gfx942 co-issue counts are not gfx950's minus a
constant) covers them. If it does not, the result is a scoped *tool* ceiling, and the interleave can be
authored by hand or with `warp_pipeline_stage` (`scheduling-model.md ## inter_wave — realize with warp_pipeline_stage`).

**When the schedule is not the problem:** demand > capacity (shrink/move work), or demand fits, knobs come
back neutral twice and window fill stays low — the bubble is likely a structural operand/convert dependency,
which operand production addresses and ordering does not.

## The same levers under other names

| Name you may meet | What it is here |
| --- | --- |
| `MEMNOP`, "mem-stage nops" | phase pacing (move the burst) |
| `SCALE_ON_Q`, "pre-scale Q" | remove the third source; `../workloads/attention.md` |
| `VEC1` / `VEC2` | the two halves of a split softmax, one per compute region (`## Attention: the co-execution budget`) |
| "lazy rescale" | `../workloads/attention.md ### Conditional rescaling (skip the per-block acc *= alpha)` |
| "declare with `sched_group_barrier`" | `## Declaring the interleave` |
| "coexec scheduler" | `amdgpu-sched-strategy=coexec`, gfx1250 only |

`MEMNOP` and `SCALE_ON_Q` reported together are two mechanisms, tuned separately.
