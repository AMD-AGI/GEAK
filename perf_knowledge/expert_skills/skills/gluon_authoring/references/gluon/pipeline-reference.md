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

The upstream-only path. Reach it when the loop has **no `tt.dot`** for the re-injected
pipeliner to anchor on, or when you want something the auto-pipeliner cannot express at all.
It does **not** compose with re-injection in the same loop — hand-written staging is exactly
what starves that pass (`../tile-programming/pipeline.md ## Route 2`).

### What is actually on offer, and which architecture has it

Two categories: mechanisms that **move data**, and mechanisms that only **influence
scheduling**. Conflating them is how a hint gets budgeted as a prefetch.

**A — data movement**

| # | mechanism | CDNA3 gfx942 | CDNA4 gfx950 | CDNA5 gfx1250 | RDNA3/4 |
| --- | --- | --- | --- | --- | --- |
| A1 | async global→LDS multi-buffer (`commit_group` / `wait_group`) | **does not lower** ✗ | `cdna4.async_copy` | different API, see below | **absent** |
| A2 | per-tensor pipeline depth | **✓ over A5** | over A1 or A5 | ✓ | over A5 |
| A3 | several independent chains at staggered depths | over A5 | over A1 | ✓ | over A5 |
| A4 | sub-buffer splitting (`.index(i)` / `.slice(...)`) | **✓ over A5** | ✓ | ✓ | over A5 |
| A5 | sync staging: `allocate_shared_memory` + `.store()` / `.load()` + barrier | **✓** | ✓ | ✓ | **✓ — the only option** |

**B — scheduling hints (move no data; must be measured, never assumed)**

| # | mechanism | on gfx942 | availability |
| --- | --- | --- | --- |
| B1 | `warp_pipeline_stage` cluster markers | **✓** — emits `s_setprio` | 3.7.0+ (absent on 3.6.0) |
| B2 | `sched_barrier` / `sched_group_barrier` / `set_prio` (iglp) | **✗ symbol absent** | absent from `gl.amd.cdna3` and `.cdna4` on **all four** versions |
| B3 | `warp_specialize` producer/consumer partitioning | **✗** `PassManager::run failed` | symbol present in core `gl` on all four |

Every gfx942 cell above is from a compile-and-run probe on this box, not from whether the
symbol imports. Three of them are worth stating plainly because the import would mislead you:

- **B2 does not exist here at all.** aiter's `pa_decode_gluon` imports `sched_barrier` /
  `sched_group_barrier` / `set_prio` from `gl.amd.cdna3` inside a `try/except` and defines
  **no-op stubs** on `ImportError`. Those symbols are absent on 3.6.0 / 3.7.0 / 3.7.1 / 3.8.0,
  so on these builds that production kernel is running the stubs and its iglp hints are dead
  code. Do not copy the pattern expecting scheduling control.
- **B3 imports but does not lower on CDNA3.** `gl.warp_specialize` is in core `gl` on every
  version; a minimal two-partition kernel still fails in the pass manager on gfx942.
- **B1 does work on gfx942**, which the version-only table below does not tell you — but it is
  still a hint. `../platform-known-issues.md` records it as source-available yet
  performance-negative, and `../hardware/capability-matrix.md` adds that scheduling hints are
  not inherently profitable. It compiles and it reorders; whether it pays is a measurement.

**So on gfx942 the authored-overlap surface is exactly two mechanisms: A5 sync staging, and
B1 as a hint on top.** A2 / A3 / A4 are things you build *out of* A5 there, not separate
features. Everything asynchronous belongs to CDNA4 and later.

### Worked examples — every one compiled and numerics-checked on gfx942

Copy these as starting points. They are deliberately the CDNA3 set, because that is the target
whose surface is smallest and most often mis-documented; the `ds_read` / `ds_write` /
`s_setprio` counts after each are what the probe measured, so you can tell immediately whether
your own build reproduces them. All four are runnable as
`scripts/pipeline_examples_cdna3.py` — run it rather than trusting the numbers below.

**A5 — the base. Two LDS buffers, alternate, barrier between phases.** On CDNA3 this is the
staging path; there is no async variant to fall back to.

```python
BLK: gl.constexpr = gl.BlockedLayout([1, 4], [16, 4], [4, 1], [1, 0], [])
SH: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])

@gluon.jit
def a5_sync_double_buffer(out, inp, M: gl.constexpr, N: gl.constexpr, ITERS: gl.constexpr):
    s = gl.allocate_shared_memory(gl.float32, [2, M, N], SH)
    o = (gl.arange(0, M, layout=gl.SliceLayout(1, BLK))[:, None] * N
         + gl.arange(0, N, layout=gl.SliceLayout(0, BLK))[None, :])
    acc = gl.zeros([M, N], gl.float32, layout=BLK)
    s.index(0).store(gl.load(inp + o))                 # prologue fills buffer 0
    for i in range(ITERS):
        cur = i % 2
        nxt = (i + 1) % 2
        gl.barrier()
        if i + 1 < ITERS:
            s.index(nxt).store(gl.load(inp + o))       # stage i+1 while consuming i
        acc += s.index(cur).load(BLK)
        gl.barrier()
    gl.store(out + o, acc)
```

`ds_write=8 ds_read=8` at `ITERS=4`, M=N=32. Note `i % 2` — a runtime index, and it cost
nothing here (`../tile-programming/pipeline.md ### Hand-built buffering rules` has the
measurement and the scope of the compile-time-index rule).

**A2 — two tensors at different lead distances in one loop.** This is the one the
auto-pipeliner structurally cannot do: it assigns a single stage schedule to the whole loop,
so a small tensor rides in the big one's stage whether that helps or not.

```python
@gluon.jit
def a2_per_tensor_depth(out, big, small, M: gl.constexpr, N: gl.constexpr, ITERS: gl.constexpr):
    sb = gl.allocate_shared_memory(gl.float32, [2, M, N], SH)   # double-buffered
    ss = gl.allocate_shared_memory(gl.float32, [1, M, N], SH)   # single, no lead
    o = (gl.arange(0, M, layout=gl.SliceLayout(1, BLK))[:, None] * N
         + gl.arange(0, N, layout=gl.SliceLayout(0, BLK))[None, :])
    acc = gl.zeros([M, N], gl.float32, layout=BLK)
    sb.index(0).store(gl.load(big + o))                # big leads by one iteration
    for i in range(ITERS):
        cur = i % 2
        nxt = (i + 1) % 2
        gl.barrier()
        if i + 1 < ITERS:
            sb.index(nxt).store(gl.load(big + o))
        ss.index(0).store(gl.load(small + o))          # small has no lead at all
        gl.barrier()
        acc += sb.index(cur).load(BLK) * ss.index(0).load(BLK)
    gl.store(out + o, acc)
```

`ds_write=16 ds_read=16`. aiter's block-scaled GEMM is the production form of this shape,
leading its operands by two K-iterations and its scales by one, and its docstring names that
split as the main win over the Triton version: keeping the scales in the operands' stage both
puts scale-load latency on the critical path and spends `ds_read` bandwidth on a tiny tensor.
A3 (several chains at staggered depths) is the same freedom applied more than twice —
`mla_gluon` runs a page-index chain alongside a KV chain and drains them separately.

**A4 — one buffer consumed as independent half-slices.** Each slice is a descriptor in its own
right, so the halves can be filled or drained on different schedules.

```python
HALF: gl.constexpr = gl.BlockedLayout([1, 4], [8, 8], [4, 1], [1, 0], [])

@gluon.jit
def a4_subbuffer_slice(out, inp, M: gl.constexpr, N: gl.constexpr, ITERS: gl.constexpr):
    s = gl.allocate_shared_memory(gl.float32, [M, N], SH)
    ...
    for _i in range(ITERS):
        gl.barrier()
        s.store(gl.load(inp + o))
        gl.barrier()
        top = s.slice(0, M // 2, 0).load(HALF)          # rows [0, M/2)
        bot = s.slice(M // 2, M // 2, 0).load(HALF)     # rows [M/2, M)
```

`ds_write=8 ds_read=8`. The slice takes `(offset, length, dim)`; the consuming layout must
match the sliced shape, not the parent's — that mismatch is the usual first failure.

**B1 — cluster markers. A hint: it reorders, it stages nothing.**

```python
@gluon.jit
def b1_warp_pipeline_stage(out, inp, M: gl.constexpr, N: gl.constexpr, ITERS: gl.constexpr):
    ...
    for _i in range(ITERS):
        with gl.amd.warp_pipeline_stage("load", priority=1):
            v = gl.load(inp + o)
        with gl.amd.warp_pipeline_stage("compute", priority=3):
            acc += v * 2.0
    gl.store(out + o, acc)
```

`s_setprio=10`, and `ds_read`/`ds_write` stay **0** — which is the point: no LDS traffic
appears because no staging happened. Pair it with A5 when you want both.

### Getting `wait_group` right

Copy the semantics from the API, do not reason from the name:

- `wait_group(num_outstanding)` blocks until the number of outstanding commit groups is
  **less than or equal to** `num_outstanding`. **Uncommitted async operations are waited on
  even when `num_outstanding` is 0**, so a missing `commit_group` does not merely delay a
  wait — it silently converts an async copy into a blocking one.
- `buffer_load_to_shared` vs `global_load_to_shared` is a trade, not a preference: the buffer
  form takes a scalar base plus a 32-bit offset tensor and gets **hardware out-of-bounds
  masking**; the global form takes a pointer tensor and gets the **64-bit indexing range**.
  Production code picks per call site by whether a mask is needed.
- `load_shared_relaxed` is **not** a faster `.load()`. It deliberately omits the cross-warp
  `ds_read` fence, and that is only sound because the caller already paired the copy with a
  `wait_group` that synchronised the wave. Used without that pairing it is a race.

> **The trap that is not in any rule list: do not interleave async copies with ordinary
> loads/stores in the same loop.** Both entry points document that an async copy still
> completes **in order** with `ttgl.load`/`store` and `buffer_load`/`store`, so a stray
> ordinary load in the body serialises the copies you built the pipeline for. This is the
> easiest way to author a pipeline that measures like no pipeline at all.

### CDNA5 (gfx1250) is a different model — do not port the CDNA3/4 shape

Only `commit_group` / `wait_group` carry over. The copy entries are renamed and re-shaped
(`global_to_shared`, `shared_to_global`, `mbarrier_arrive`), there is an `mbarrier` object
model and a `cluster` scope, and the descriptor path is a separate `tdm` module
(`make_tensor_descriptor`, `async_load` / `async_store` / `async_gather` / `async_scatter`,
`async_wait`, `prefetch`). The matrix op is `wmma`, not `mfma`. Treat it as its own target.

### RDNA3 / RDNA4 have no async surface at all

`gl.amd.rdna3` and `gl.amd.rdna4` expose exactly one thing: `wmma`. No buffer ops, no async
copy, no TDM. Authored overlap there is A5 only — core `gl.allocate_shared_memory` staging with
an explicit barrier, hand LDS ping-pong (`rdna-wmma-reference.md ## Pipeline`).

**Re-injection has not been tested on CDNA5 or RDNA.** All four-version evidence for it comes
from gfx942; do not extrapolate the route table there.

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

**`cdna4.async_copy` imports on gfx942 but does not lower there. Importing is not lowering,
and this row used to claim otherwise.** The module and all five entry points import fine on
every version including 3.6, which is what the old claim rested on — but a compiled probe on
gfx942 fails at the backend, identically on 3.6.0 and 3.8.0:

| entry, called on gfx942 | result |
| --- | --- |
| `buffer_load_to_shared` | `failed to translate module to LLVM IR` (`builtin.unrealized_conversion_cast` on the pointer) |
| `global_load_to_shared` | `PassManager::run failed` |
| `load_shared_relaxed` | **OK** — it is only a shared-read hint, no async copy |
| `gl.allocate_shared_memory` + `.store()` / `.load()` (sync staging) | **OK** |

Three things make this the op and not the call site: `load_shared_relaxed` from the *same*
module compiles and runs correctly, so the import and the surrounding scaffolding are sound;
the two async entries fail with two *different* backend errors, both past the frontend; and
every per-thread vector width (32 / 64 / 128-bit) fails identically, so it is not the
width-gated behaviour a `supportsBufferLoadToLocal()` reading would predict. aiter's own
async-copy Gluon kernels agree — they are CDNA4-targeted and recorded as unreachable on gfx942.

**So on CDNA3 the authored-overlap path is sync staging, not async copy.** The async table
below applies to CDNA4. Not re-checked on CDNA4 hardware here (none available), so treat the
gfx950 column as inherited from those production kernels rather than probed.

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
