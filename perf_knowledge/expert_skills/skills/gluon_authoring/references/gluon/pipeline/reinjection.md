# Gluon pipeline — re-injecting plain's pipeliner (diagnostic / last resort)

> **Status of this route: lowest priority on the Gluon path.** The pipeline order is hand-written
> first — register-level prefetch, then an authored LDS ring, then `warp_pipeline_stage` and the
> scheduling-model choice — and it is defined once in `../../tile-programming/pipeline.md`.
> Re-injecting plain's auto-pipeliner sits below all of them and is admissible in exactly two
> situations:
>
> 1. **as a diagnostic below the parity gate** — to measure how much of the residual is
>    `lost_pipeline` debt (the `plain@ns=1` control below is the cheaper half of that diagnosis);
> 2. **as a last resort** — when a hand-written pipeline has been tried and demonstrably cannot reach
>    parity.
>
> Every number produced on this route is labelled **`injected`**, is never reported as a win, and
> the route is **never applied to an incumbent** (a kernel that arrived already written in Gluon) —
> there, only the hand-written ring is on offer. Its ceiling is plain's own overlap: it has no way to
> express anything plain could not. The step-by-step last-resort procedure (three conditions, tail
> pairing, the cache trap, the IR landing criteria) is `../../method/recover.md`; this page is the
> mechanism and the measured outcomes.

## Re-injecting plain's pipeliner — the measured recipe

`add_schedule_loops` and `add_pipeline` are present in `libtriton` on all four versions;
only the Python pass list omits them. Two ways to reach them, and the first edits nothing:

```python
import gluon_swp                        # scripts/gluon_swp.py -- wraps gluon_to_ttgir
with gluon_swp.pipelined(2, buffer_ops=True):
    out = my_kernel[grid](...)          # compile INSIDE the block; Triton caches
```

`gluon_swp.py` runs the passes as a second pass manager over the module the stock
`gluon_to_ttgir` returns. **Verified byte-identical TTGIR to the** `compiler.py` **splice on
3.6.0 / 3.7.0 / 3.7.1 / 3.8.0** — same md5 both armed and unarmed — so nothing is given up by
not touching the file, while a read-only or shared site-packages, a `pip --force-reinstall`,
and a crash between apply and revert all stop being hazards. `scripts/patch_reinject.py` is
the on-disk form, kept for when you want the pass list visible in `compiler.py` while reading.
That equivalence was verified across **minors on gfx942**, not across architectures: on gfx950
`add_pipeline` defaults to its async path and one more pass (`add_coalesce_async_copy`) joins the
sequence, so check it on your build there (`../../tile-programming/pipeline.md ## Reproduce plain's
software pipeline on the Gluon path (the parity-recovery route)`).

**One process per arm.** Triton's in-process JIT cache is keyed on `(function, signature,
constexprs)` and knows nothing about the injection, so an armed and an unarmed compile in one
process silently share a binary. Run each arm (unarmed, armed at each depth, each `buffer_ops`
setting) in its own process with its own `TRITON_CACHE_DIR`, and confirm each arm's own `.ttgir`
carries the landing tell before reading its clock.

**There is no env knob for this.** Names of the form `TRITON_GLUON_SWP_PIPELINE` /
`TRITON_GLUON_COOP_LDS` / `TRITON_GLUON_PINGPONG` circulate for it, but a three-way search —
upstream `main`, the `v3.8.0` tag, and every tag of the vendor fork lineage, plus
`git log --all -S` over both repositories — finds **no referent for any of them**. They are not a
fork feature we lack; they name nothing, and an env var nothing reads is tolerated and inert, so
the failure is silent and reads as "the technique does not work here". (An earlier GEAK note
attributed them to a vendor fork's `GetEnv.h`; the later search does not reproduce that, and the
practical consequence is the same either way — on a stock build they do nothing.) The fork-only
variables that *do* exist on the fork lineage are listed with their upstream routes in
`../../tile-programming/non-upstream-reserve.md`.

**Two conditions are required and neither alone does anything.** From a 2×2 on a dot-free
kernel, all four cells numerically identical, `PIPELINED` read off the IR (peeled prologue
loads + a loop-carried `iter_arg`):

| loads written as           | loop                          | pipelined                        |
| -------------------------- | ----------------------------- | -------------------------------- |
| `gl.amd.cdna3.buffer_load` | `range(...)`                  | ✗                                |
| `gl.amd.cdna3.buffer_load` | `tl.range(..., num_stages=2)` | ✗                                |
| `gl.load`                  | `range(...)`                  | ✗                                |
| `gl.load`                  | `tl.range(..., num_stages=2)` | **✓** 2→4 loads, `iter_args` 0→1 |

(The 2×2 was run on gfx942, hence `cdna3`; on gfx950 the same rows hold with
`gl.amd.cdna4.buffer_load` — the condition is "a buffer op", not the namespace.)

1. **The loop must be a pipelining CANDIDATE, and whether it is depends on the DOT.**
   `add_schedule_loops` takes the launch-level `num_stages` as the default for a loop it
   considers a candidate, and it decides that from the loop's contents:

   | loop                | needs an annotation?                                        |
   | ------------------- | ----------------------------------------------------------- |
   | contains a `tl.dot` | **no** — a bare `range` pipelines from `num_stages` alone   |
   | dot-free            | **yes** — `tl.range(..., num_stages=N)`, or nothing happens |

   Measured both ways on plain at the same launch `num_stages`: a bare-`range` **GEMM** scaled
   its load count with the depth and gained `ttg.memdesc_index`, with **no** `tt.num_stages` **in
   the IR at all**, while a bare-`range` **dot-free reduction** stayed byte-identical. Same on
   the Gluon side under injection: an un-staged GEMM with a bare `range` pipelines
   indistinguishably from the `tl.range` version.
   This is why a real GEMM can be pipelined while its source writes plain
   `for k in range(...)`, and why a dot-free kernel has to carry the annotation.
   Gluon exposes no `range` of its own (only `static_range`, which unrolls), but `tl.range`
   **is** usable from a `gluon_jit` body when you need the dot-free case. And
   `tl.range(..., num_stages=None)` **inherits** the launch value. Real tuned kernels use this
   deliberately — a `num_stages = None if ENABLE_PIPELINING else 1` constexpr on the inner loop,
   with the number arriving from the autotune config — so `None` on the loop is not "unset", and
   reading the source alone will not tell you the depth. **None of this makes `num_stages` a Gluon
   knob:** without the injection no pass reads it (`loop-knobs-and-targets.md ## Roll the loop (cut
   i-cache pressure)`); it is only the depth argument this route hands to plain's passes.
2. **The loads must still be** `tt.load` **when the pipeliner runs.** Plain's own `make_ttgir`
   orders `add_schedule_loops` #15, `add_pipeline` #16, `add_convert_to_buffer_ops` **#28** —
   twelve passes later. So plain's pipeliner only ever sees `tt.load`. An anchor written the
   way the transcription runbook asks (explicit `gl.amd.cdna4.buffer_load` on gfx950,
   `gl.amd.cdna3.buffer_load` on gfx942, because `gluon_to_ttgir` runs no buffer conversion) hands
   the pipeliner ops it cannot recognise.

> **Those two pieces of guidance pull against each other**, and that is not a bug in either.
> Buffer ops are worth real performance on a memory-bound body; the pipeliner needs
> `tt.load`. The splice resolves it by restoring plain's ORDER: pipeline first, then
> `add_convert_to_buffer_ops`. Write `gl.load`, arm `TRITON_GLUON_SWP_BUF=1`, and the final
> IR is buffer ops *and* pipelined — measured 4 `amdg.buffer_load`, 0 `tt.load`.
> That half is **opt-in**: arming it on an anchor whose loads are already `buffer_load`
> aborts with `PassManager::run failed`.

### On a dot kernel: un-write the staging

The faithful anchor shape — `allocate_shared_memory` + `_barrier()` +
`smem.load(A_DOT_OPERAND)` — carries every blocker at once, and building the LDS path is
exactly what the pipeliner exists to do. So hand the loop back to it: drop the explicit
staging and let the pass create it.

**Measure three arms, not two.** The middle one is the trap:

| arm                          | what it tells you                                                                        |
| ---------------------------- | ---------------------------------------------------------------------------------------- |
| hand-staged, no injection    | the faithful baseline                                                                    |
| **un-staged, injection OFF** | **a REGRESSION vs the hand-staged arm** — you removed the staging and nothing rebuilt it |
| un-staged, injection ON      | the recovery (`injected`)                                                                |

Reporting only the first and third makes the injection look like it did all the work, and
reporting only the second makes un-staging look like a mistake. The two halves go together or
not at all. Landed correctly, the pass creates the allocations, writes and reads that the
hand-staged arm had, plus a peeled prologue, plus `ttg.memdesc_index` for the multi-buffering —
and needs no authored barrier.

**Do not stack hand-written prefetch on top of it.** A hand register-prefetch next to a re-injected
pipeliner is not additive — it consumes the slot the pass wanted. The two routes are exclusive per
loop; pick one.

### And on attention — two dots chained through a softmax

The shape worth checking separately, because the second dot's A operand is the first dot's
output and the accumulators are loop-carried, so a pipeliner that prefetches a GEMM's operands
might refuse it. On a minimal FA-forward body it does not refuse: same signature as the GEMM —
prologue peeled, K/V staging created by the pass, multi-buffered, no barrier authored, numerics
unchanged.

> **A minimal body does NOT settle attention.** On a real sparse-paged attention kernel the
> same recipe was a **large regression** while the injection was demonstrably firing — op
> census identical to plain's, and the best wait profile of any arm. The mechanism is the
> language gap, not the pipeline: the injected pass builds
> the V staging on `swizzled_shared<vec=1, perPhase=1, maxPhase=1>` — no vectorisation, no
> swizzle — where plain gets `amd_rotating_shared<vec=4, perPhase=4, maxPhase=4>`, which
> Gluon cannot express — narrow scalar LDS reads where plain gets wide vectorised ones. So on
> a body whose staging plain puts on a rotating-shared layout, the
> pipeliner can fire perfectly and still lose, because the layout it has to fall back to is
> the one that blocks the faithful anchor too.

> `buffer_ops=True` **is not free either.** On that same kernel it was a further penalty on
> top — the opposite of "restore plain's order and get both". Pass-by-pass attribution put the
> base cost on `add_pipeline` itself, not on `optimize_dot_operands`. Measure the flag on your
> body; do not assume it pays.

> `buffer_ops=True` **also conflicts with buffer STORES**, not just loads: a loop that stores
> through `gl.amd.cdna{3,4}.buffer_store` dies with `LLVM ERROR: Fatal pipeliner error`, which
> **kills the interpreter** rather than raising. Write both sides as `gl.load`/`gl.store` on an
> arm you intend to arm with it.

> **`convert_layout` scratch allocated after the pipeliner pass is outside hazard analysis**, and
> a hazard there fails silently. A layout conversion that the injected pipeline leaves in the loop
> is a correctness question before it is a cost.

**Size the prize before spending the round, and size it at the champion's own tile.** On one
tuned attention champion the pipeline's own contribution was small and **changed sign across
sequence lengths** — a gain at some, a small pessimisation at others. Its own tuning notes
advertised a much larger win, but that number was the *combined* effect of a smaller `BLOCK_N`
**and** `num_stages=2` against the shipped tile. Only `plain@ns=1` **at the champion's tile**
separates the two, and a debt that flips sign with shape cannot be reported as one number.

### What to expect, and what to measure yourself

**The measured outcomes, in one table.** Two statements about this route circulated in this pack
and read as a contradiction — "it recovered the full pipeline gap on two kernels and on all four
versions" and "on a measured kernel the net stayed negative on most toolchain versions". They are
both measurements, and they are measured **against different baselines**: the first compares the
injected arm with the shipped plain kernel on a kernel whose debt was real; the second compares
de-staged-plus-injected with the hand-staged faithful anchor it replaced. The table keeps them
apart. All rows are **gfx942**; nothing here was measured on gfx950, where `add_pipeline` takes a
different (async) path, and CDNA5 / RDNA are untested.

| kernel class (gfx942) | baseline compared against | 3.6.0 | 3.7.0 | 3.7.1 | 3.8.0 | numerics |
| --- | --- | --- | --- | --- | --- | --- |
| dot-free reductions, GEMMs, minimal FA-forward body | does the injection **fire** (IR tells) | ✓ | ✓ | ✓ | ✓ | unchanged |
| two kernels whose `plain@ns=1` showed a real debt | injected arm vs shipped plain | closed — reached or slightly exceeded plain | same | same | same | unchanged |
| one de-staged dot kernel | injected vs un-staged-OFF (the injection's own effect) | consistent speedup | same | same | same | unchanged |
| the same kernel | de-staged + injected vs the hand-staged faithful anchor | **net negative on most minors**; which minors were positive is not recorded | | | | unchanged |
| a real sparse-paged attention kernel | injected vs plain | **large regression** while firing (rotating-shared gap above); minor not recorded | | | | unchanged |
| a tuned attention champion | pipeline's own share (`plain@ns=1` vs shipped) | small, **changes sign** across sequence lengths; minor not recorded | | | | — |
| every kernel measured | `num_stages=3` vs `2` | worse at 3 on every kernel; one refused to launch | | | | — |

What the rows have in common: **the injection fires on every shape and every version, the
magnitude does not transfer** — it varied by kernel, by shape, and by Triton version, in both
directions, and a ratio measured on one version is not evidence about another. One correlation did
hold on every version tested: **where plain itself gains nothing from `num_stages=2` over
`num_stages=1` on a version, recovering the pipeline for a transcription did not pay off there
either.** That makes the plain-side comparison a pre-check you can run before rewriting the body.

So the only number worth carrying between kernels is the one you measure: `plain@ns=1` **at the
champion's own config**. That control is what tells you the size of the debt before you spend a
round, and comparing your anchor against it is what says whether the residual is the pipeline
or something else. Judge any injected net on the **same-window per-rep ratio** of the armed arm over
the original anchor — differencing two percentages against a shipped baseline whose own spread is a
few percent cannot resolve an effect this size. Every such number is recorded as `injected`.

**Depth is a knob, not a monotone.** `num_stages=3` was worse than 2 on every kernel measured
here, and on one it refused to launch at all. Two mechanisms, both readable in advance:

- a deeper schedule genuinely double-buffers, so LDS grows — and if two workgroups' worth
  crosses **the LDS capacity of your CU**, occupancy halves. That divisor is arch-specific and
  owned by `../smem-lds-reference.md ## LDS is a second, independent occupancy limiter` (gfx950 has
  2.5× gfx942's capacity), so a gfx942 occupancy verdict must not be carried to gfx950.
  `recover`'s `LDS:` line reports the total and divides by the figure for the `--arch` you
  passed. (At depth 2 the pass may build a *single*-buffered rotating stage instead: prologue
  peeled so the global load overlaps the MFMA, LDS unchanged. That is why 2 often wins.)
- a depth plain itself cannot compile is not available to you either — the LDS requirement is
  byte-identical, because it is the same pass.

**A kernel whose loop has no trip count has nothing to pipeline.** If the dispatched config
makes the loop run once, arming the injection only pays for a peeled prologue and an epilogue
that never overlap anything — a clear regression, on a body where the shipped `num_stages=1` was
the right choice. Check the trip count before reading the debt.

### What this does not do

- **It is not upstream.** No upstream version calls these passes from `gluon_to_ttgir`; a
  patched `compiler.py` or an armed `gluon_swp` is a local change and every measurement taken under
  it must say so — and say `injected`.
- **It is not a win, and not a climb lever.** Its ceiling is plain's own overlap. Above the parity
  gate it has nothing to offer; the climb is the hand-written pipeline.
- **It is not for incumbents.** A kernel that arrived already in Gluon has no plain pipeline to
  have lost; only the authored ring applies there.
- `add_block_pingpong` still will not fire on hand-authored staging: it only collects
  `local_load`s whose source is a loop-carried `BlockArgument`, and a hand-written one is
  sourced from `memdesc_index`. Un-writing the staging is what makes it reachable.
- `warp_pipeline_stage` is a **different** mechanism (`marker-and-version-gates.md`), not this one.

### The ping-pong window — only reachable on this route (gfx942 measurement)

Ping-pong (`add_block_pingpong`) does fire on gfx942 under injection, but in a narrow window.
Measured: a 256×256×64 tile at `num_warps=8`, `ns=2` → **8 `s_setprio`** in the ISA;
the same tile at `num_warps=4`, and 128×128×64 at `num_warps=8`, both → **0**. It never fires on
hand-authored staging (the `BlockArgument` rule above). **Judge it from the ISA, never from a source
config** — and do not try to earn it by satisfying `is_pingpong_schedule_enabled`: that predicate is
not arch-symmetric, but the decisive point is that it is consulted **only inside `make_ttgir`**,
which the Gluon entry point does not go through, so meeting its condition from Gluon source buys
nothing on either generation. Ping-pong on this path needs the schedule decision spliced, not the
predicate satisfied. A missing `s_setprio` is also not counter-evidence until you have seen it on the
reference arm: plain's own `ns=2` build of a kernel whose loop shape `add_block_pingpong` rejects has
`s_setprio == 0` too.
