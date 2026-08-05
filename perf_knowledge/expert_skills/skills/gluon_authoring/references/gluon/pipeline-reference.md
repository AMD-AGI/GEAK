# Gluon — pipeline

Backbone layer 4. Gluon has **no** CuTeDSL `PipelineTmaAsync` / mbarrier library. It also
ships no auto-pipeliner: `gluon_to_ttgir` never calls `add_schedule_loops` / `add_pipeline`
on **any** upstream version (checked 3.6.0 / 3.7.0 / 3.7.1 / 3.8.0). So overlap is either
**authored** (§ below) or **re-injected** — and re-injection is measured, not theoretical:
it recovered the full pipeline gap on two kernels and on all four versions.

---

## Re-injecting plain's pipeliner — the measured recipe

`add_schedule_loops` and `add_pipeline` are present in `libtriton` on all four versions;
only the Python pass list omits them. Two ways to reach them, and the first edits nothing:

```python
import gluon_swp                        # scripts/gluon_swp.py -- wraps gluon_to_ttgir
with gluon_swp.pipelined(2, buffer_ops=True):
    out = my_kernel[grid](...)          # compile INSIDE the block; Triton caches
```

`gluon_swp.py` runs the passes as a second pass manager over the module the stock
`gluon_to_ttgir` returns. **Verified byte-identical TTGIR to the `compiler.py` splice on
3.6.0 / 3.7.0 / 3.7.1 / 3.8.0** — same md5 both armed and unarmed — so nothing is given up by
not touching the file, while a read-only or shared site-packages, a `pip --force-reinstall`,
and a crash between apply and revert all stop being hazards. `scripts/patch_reinject.py` is
the on-disk form, kept for when you want the pass list visible in `compiler.py` while reading.

**`TRITON_GLUON_SWP_PIPELINE` is not the knob.** It, `TRITON_GLUON_COOP_LDS` and
`TRITON_GLUON_PINGPONG` belong to a vendor fork's `GetEnv.h`; no upstream version reads them.
Measured on clean 3.7.1 and 3.8.0 they are *tolerated and inert* — as is a knob invented on the
spot — so the failure is silent and reads as "the technique does not work here".

**Two conditions are required and neither alone does anything.** From a 2×2 on a dot-free
kernel, all four cells numerically identical, `PIPELINED` read off the IR (peeled prologue
loads + a loop-carried `iter_arg`):

| loads written as | loop | pipelined |
| --- | --- | --- |
| `gl.amd.cdna3.buffer_load` | `range(...)` | ✗ |
| `gl.amd.cdna3.buffer_load` | `tl.range(..., num_stages=2)` | ✗ |
| `gl.load` | `range(...)` | ✗ |
| **`gl.load`** | **`tl.range(..., num_stages=2)`** | **✓** 2→4 loads, `iter_args` 0→1 |

1. **The loop must be a pipelining CANDIDATE, and whether it is depends on the DOT.**
   `add_schedule_loops` takes the launch-level `num_stages` as the default for a loop it
   considers a candidate, and it decides that from the loop's contents:

   | loop | needs an annotation? |
   | --- | --- |
   | contains a `tl.dot` | **no** — a bare `range` pipelines from `num_stages` alone |
   | dot-free | **yes** — `tl.range(..., num_stages=N)`, or nothing happens |

   Measured both ways on plain at the same launch `num_stages`: a bare-`range` **GEMM** scaled
   its load count with the depth and gained `ttg.memdesc_index`, with **no `tt.num_stages` in
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
   reading the source alone will not tell you the depth.
2. **The loads must still be `tt.load` when the pipeliner runs.** Plain's own `make_ttgir`
   orders `add_schedule_loops` #15, `add_pipeline` #16, `add_convert_to_buffer_ops` **#28** —
   twelve passes later. So plain's pipeliner only ever sees `tt.load`. An anchor written the
   way the transcription runbook asks (explicit `gl.amd.cdna3.buffer_load`, because
   `gluon_to_ttgir` runs no buffer conversion) hands the pipeliner ops it cannot recognise.

> **Those two pieces of guidance pull against each other**, and that is not a bug in either.
> Buffer ops are worth real performance on a memory-bound body; the pipeliner needs
> `tt.load`. The splice resolves it by restoring plain's ORDER: pipeline first, then
> `add_convert_to_buffer_ops`. Write `gl.load`, arm `TRITON_GLUON_SWP_BUF=1`, and the final
> IR is buffer ops *and* pipelined — measured 4 `amdg.buffer_load`, 0 `tt.load`.
> That half is **opt-in**: arming it on an anchor whose loads are already `buffer_load`
> aborts with `PassManager::run failed`.

### On a dot kernel: un-write the staging

The faithful anchor shape — `allocate_shared_memory` + `gl.barrier()` +
`smem.load(A_DOT_OPERAND)` — carries every blocker at once, and building the LDS path is
exactly what the pipeliner exists to do. So hand the loop back to it: drop the explicit
staging and let the pass create it.

**Measure three arms, not two.** The middle one is the trap:

| arm | what it tells you |
| --- | --- |
| hand-staged, no injection | the faithful baseline |
| **un-staged, injection OFF** | **a REGRESSION vs the hand-staged arm** — you removed the staging and nothing rebuilt it |
| un-staged, injection ON | the recovery |

Reporting only the first and third makes the injection look like it did all the work, and
reporting only the second makes un-staging look like a mistake. The two halves go together or
not at all. Landed correctly, the pass creates the allocations, writes and reads that the
hand-staged arm had, plus a peeled prologue, plus `ttg.memdesc_index` for the multi-buffering —
and needs no authored barrier.

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

> **`buffer_ops=True` is not free either.** On that same kernel it was a further penalty on
> top — the opposite of "restore plain's order and get both". Pass-by-pass attribution put the
> base cost on `add_pipeline` itself, not on `optimize_dot_operands`. Measure the flag on your
> body; do not assume it pays.

> **`buffer_ops=True` also conflicts with buffer STORES**, not just loads: a loop that stores
> through `gl.amd.cdna3.buffer_store` dies with `LLVM ERROR: Fatal pipeliner error`, which
> **kills the interpreter** rather than raising. Write both sides as `gl.load`/`gl.store` on an
> arm you intend to arm with it.

**Size the prize before spending the round, and size it at the champion's own tile.** On one
tuned attention champion the pipeline's own contribution was small and **changed sign across
sequence lengths** — a gain at some, a small pessimisation at others. Its own tuning notes
advertised a much larger win, but that number was the *combined* effect of a smaller `BLOCK_N`
**and** `num_stages=2` against the shipped tile. Only `plain@ns=1` **at the champion's tile**
separates the two, and a debt that flips sign with shape cannot be reported as one number.

### What to expect, and what to measure yourself

Across the kernels this was validated on — dot-free reductions, GEMMs, attention, on
3.6.0 / 3.7.0 / 3.7.1 / 3.8.0 — the pattern that held was:

- **the injection fires on every shape and every version**, with the numerics unchanged;
- **where the debt was real, it closed** — the recovered arm reached or slightly exceeded the
  shipped plain kernel;
- **the magnitude did not transfer.** It varied by kernel, by shape, and by Triton version, in
  both directions. A ratio you measured on one version is not evidence about another.

So the only number worth carrying between kernels is the one you measure: **`plain@ns=1` at the
champion's own config**. That control is what tells you the size of the debt before you spend a
round, and comparing your anchor against it is what says whether the residual is the pipeline
or something else.

**Depth is a knob, not a monotone.** `num_stages=3` was worse than 2 on every kernel measured
here, and on one it refused to launch at all. Two mechanisms, both readable in advance:

- a deeper schedule genuinely double-buffers, so LDS grows — and if two workgroups' worth
  crosses **the LDS capacity of your CU**, occupancy halves. That divisor is **arch-specific**:
  CDNA3 (gfx942) has 64 KiB/CU, CDNA4 (gfx950) has **160 KiB/CU**, so the same allocation is
  2.5× less pressure there and a gfx942 occupancy verdict must not be carried to gfx950.
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
  patched `compiler.py` is a local change and every measurement taken under it must say so.
- `add_block_pingpong` still will not fire on hand-authored staging: it only collects
  `local_load`s whose source is a loop-carried `BlockArgument`, and a hand-written one is
  sourced from `memdesc_index`. Un-writing the staging is what makes it reachable.
- `warp_pipeline_stage` is a **different** mechanism (below), not this one.

---

## Authored overlap (no compiler patch)

The upstream-only path. Still the right choice when a patched `compiler.py` is unacceptable,
and the only option on 3.6 for warp-level pipelining:

```text
buffer_load_to_shared / cdna4.async_copy / gfx1250 TDM
  → commit_group (producer)
  → wait_group   (consumer)
  → mfma / wmma
```

LLVM passes (`LLIR_SCHED`, `RA_HINTS`, `AMDGCN_AS`) interleave the hot loop — toggle +
IR-verify (`../tile-programming/compiler-contract.md`).

### `gl.amd.warp_pipeline_stage` — the official marker path

Upstream, opt-in, and **independent of `num_stages`**: `WarpPipeliner.cpp` bails out unless
the loop body contains at least one border marker, so nothing happens to a loop you did not
annotate. `gluon_to_ttgir` already calls `add_warp_pipeline` on 3.7+, so this needs **no
compiler patch**.

```python
for k in range(K // BK):                       # bare range is fine here
    with gl.amd.warp_pipeline_stage("load", priority=1):
        ...                                    # cluster 1
    with gl.amd.warp_pipeline_stage("mfma", priority=3):
        ...                                    # cluster 2
```

Constraints, from the pass itself: **≥ 2 clusters** or it errors; a barrier / `async_wait` /
`AsyncTDMWait` may sit only *between* stages, never inside one; and no `scf.for` / `scf.while`
inside a stage. It emits `s_setprio` + `sched_barrier` around each cluster in `make_llir`.

**Availability is the thing to check first — it is the one capability that is version-gated
rather than probe-gated:**

| | 3.6.0 | 3.7.0 | 3.7.1 | 3.8.0 |
| --- | --- | --- | --- | --- |
| `gl.amd.warp_pipeline_stage` | **absent** | ✓ | ✓ | ✓ |
| `add_warp_pipeline` in `libtriton` | **absent** | ✓ | ✓ | ✓ |
| `add_schedule_loops` / `add_pipeline` in `libtriton` | ✓ | ✓ | ✓ | ✓ |
| called from `gluon_to_ttgir` | ✗ | ✗ | ✗ | ✗ |
| `gl.amd.cdna4.async_copy` (all 5 entry points) | ✓ | ✓ | ✓ | ✓ |

On 3.6 there is no marker path at all — re-injection is the only pipelining available.
`cdna4.async_copy` imports on every version including 3.6; the `cdna4` naming is a module
name, not a hardware gate — `TargetFeatures::supportsBufferLoadToLocal()` covers CDNA3 and
CDNA4, so direct-to-LDS exists on gfx942 at 32-bit vector width (128-bit needs CDNA4).

## vs CuTeDSL (concept filter)

| CuTeDSL | Gluon equivalent |
| --- | --- |
| `PipelineTmaAsync` + mbarrier tx_count | `commit_group`/`wait_group` + `s_waitcnt` |
| TMA multicast | gfx1250 `cluster` + multi-CTA (CDNA: XCD remap, not multicast) — note TMA is NVIDIA-proprietary; gfx1250 uses `cluster`/TDM, not NVIDIA TMA |
| `cutlass.range(prefetch_stages=)` | manual prologue/steady-state in source |
| warp-spec warpgroups | `warp_specialize` + `warp_pipeline_stage` (gfx950) |
| CLC | **none** on CDNA; gfx1250 cluster scheduling only |

## gfx950 CDNA4 path

```python
# producer
gl.amd.cdna4.async_copy(gmem_ptr, lds_ptr, ...)
gl.amd.cdna4.commit_group()
# consumer
gl.amd.cdna4.wait_group(n)
# mfma
acc = gl.amd.cdna4.mfma(...)
```

Plain Triton on gfx950 may get `tritonamdgpu-pipeline` — **Gluon does not**; hand-write
sync when escalating from plain.

## gfx1250 path

TDM descriptor async copy replaces buffer_load_lds for matrix operands
(`gl.amd.gfx1250.tdm`). Wave **32** — separate minimal anchor from gfx950.

## gfx942 downgrade

Async hardware exists; verify `commit_group` lowering. No scaled MFMA.

## Roll the loop (cut i-cache pressure)

Gluon has **no** `num_stages`/`loop_unroll_factor` knob (only `ttgl.static_range`, which
fully unrolls, and plain `range`, which does not). When the Instr Cache hit rate is low,
this is a structural edit, not an API call: use plain `range` for large trip counts and
factor a fat hot body into a shared `@gluon.jit` helper to avoid duplicated code.

```python
for k in range(0, K, BLOCK_K):        # plain range -> rolled loop (static_range would unroll all)
    acc = _dot_step(a, b, acc)        # fat body factored into one @gluon.jit subroutine
```

There is no arch-specific primitive here (unlike `tl.range(loop_unroll_factor=)` in
plain Triton). Verify: L1I hit up, fetch-latency bubble down.

## Footguns

- `wait_group` depth mismatch → hang or stale LDS read (`../debug-async.md`).
- Deeper stages without occupancy headroom → regression (`../tile-programming/slicing.md`).

## Anchors

- [triton-lang/triton `cdna4/`](https://github.com/triton-lang/triton/tree/main/python/triton/experimental/gluon/language/amd/cdna4)
- [triton-lang/triton `gfx1250/`](https://github.com/triton-lang/triton/tree/main/python/triton/experimental/gluon/language/amd/gfx1250)
