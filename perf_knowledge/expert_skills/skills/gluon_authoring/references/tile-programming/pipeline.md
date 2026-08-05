# Pipeline Layer (gfx950)

Read this for the pipeline layer (after LDS layout, before slicing). Goal: hide
HBM and LDS latency by staggering AC / LR / DOT so the MFMA unit never waits.

## Which route are you on

Three routes, and the first one that applies is the one to take. They are **not**
alternatives to try in any order, and the first two are **mutually exclusive** in the
kernel body -- see the warning on Route 2.

| you are | the pipeliner | go to |
| --- | --- | --- |
| **plain Triton** | already ran in `make_ttgir` | nothing to reproduce. Skip to `## Budget before deepening` to decide whether to deepen, and to Route 3 only if you have the build it needs |
| **Gluon, loop has a `tt.dot`** | not run by default, **but re-injectable** | **Route 1** -- reproduce plain's own passes. Default; no rebuild |
| **Gluon, dot-free loop** | not run, and injection has no anchor | **Route 2** -- author the overlap yourself |
| **Gluon, Route 1 landed and you want more than plain had** | -- | **Route 2**, for what the auto-pipeliner cannot express (per-tensor depth, staggered chains) |

Route 3 (LLIR co-design) is **not** a fourth option in the same sense: it is a
build-specific scheduler toggle that most readers do not have and that asserts on
attention-shaped kernels. It sits last for that reason.

## What the default Gluon lowering actually runs (source-verified)

Verified in `third_party/amd/backend/compiler.py`: the **plain** lowering `make_ttgir` runs
the automatic software pipeliner (`add_schedule_loops(num_stages)` + `add_pipeline`) and the
arch-gated automatic ping-pong (`add_block_pingpong(num_stages)`). The **Gluon** lowering
`gluon_to_ttgir` runs **none of those three** -- it runs `add_warp_pipeline`. Two consequences,
and the second is the one that used to be stated wrongly here:

- `num_stages` at the launch is **not** a pipelining trigger on the default Gluon path. It
  survives as a **budget parameter** (buffer count) -- the bandwidth model uses
  `in_flight = min(..., num_stages - 1)`.
- **"Gluon has no software pipeliner" does not follow.** The passes are absent from the
  default lowering, not from `libtriton` -- they are present on 3.6.0 / 3.7.0 / 3.7.1 / 3.8.0
  and re-injectable without a rebuild (Route 1). Reach for a hand-built pipeline or a scoped
  schedule ceiling only after Route 1 is ruled out, not before.

## Reproduce plain's software pipeline on the Gluon path (Route 1 — default, no rebuild)

The auto-pipeliner overlap **is re-injectable** into an explicit Gluon loop — run plain's own
TTGIR passes on the Gluon IR. Plain's overlap = `add_schedule_loops(num_stages)` +
`add_pipeline(...)` in `make_ttgir`. **No upstream `gluon_to_ttgir` calls them** — verified on
3.6.0 / 3.7.0 / 3.7.1 / 3.8.0 — while the passes themselves are present in `libtriton` on all
four, so no `libtriton.so` rebuild is needed anywhere.

### Reach them WITHOUT editing an installed file

These live in the **tile-programming-gluon** pack, not this one: they wrap
`gluon_to_ttgir`, which plain Triton never goes through.

```python
import gluon_swp                       # tile-programming-gluon: scripts/gluon_swp.py
with gluon_swp.pipelined(2, buffer_ops=True):
    out = my_gluon_kernel[grid](...)   # compile INSIDE the block; Triton caches
```

`gluon_swp.py` wraps `HIPBackend.gluon_to_ttgir` in-process and runs the passes as a second
pass manager over the module the stock function returns. Verified **byte-identical TTGIR to
the `compiler.py` splice on all four versions** — same md5 armed and unarmed — so nothing is
given up by not editing the file, while a read-only or shared site-packages, a later
`pip install --force-reinstall`, and a crash mid-experiment all stop mattering.
`gluon_swp.capabilities()` reports what the build has and refuses to install on a fork that
already splices the passes in (running them twice is not the same experiment).

`scripts/patch_reinject.py apply|revert` (also tile-programming-gluon) is the on-disk
alternative, kept for when you want the pass list itself visible in `compiler.py` while reading. It is env-armed
(`TRITON_GLUON_SWP=N`) and verified byte-identical to stock when unarmed.

> **Do NOT reach for `TRITON_GLUON_SWP_PIPELINE`.** That name, and
> `TRITON_GLUON_COOP_LDS` / `TRITON_GLUON_PINGPONG`, are additions to a **vendor fork's**
> `GetEnv.h`; no upstream version reads any of them. Measured on clean 3.7.1 and 3.8.0: they
> are **tolerated and inert** — so is a knob invented on the spot — which is the worst of the
> three possible outcomes. Nothing errors, nothing changes, and the null result reads as "the
> technique does not work here" rather than "that variable does not exist in this build".

The pass sequence either mechanism installs, which is plain's own order:

```python
add_optimize_dot_operands; add_schedule_loops(ns); add_pipeline(use_async_copy, use_block_pingpong)
# then, to get buffer ops back (plain runs convert_to_buffer_ops twelve passes LATER, at #28):
canonicalizer; canonicalize_pointers; canonicalizer; convert_to_buffer_ops
```

### The two conditions the kernel must meet

Measured on real gap kernels, every cell bit-exact. **The load form is a hard requirement; the
loop form is only one of two ways the depth can arrive:**

| loads written as | loop | where the depth comes from | pipelined |
| --- | --- | --- | --- |
| `gl.amd.cdna<N>.buffer_load` | either | either | ✗ |
| `gl.load` | bare `range(...)` | the loop / the launch | ✗ |
| **`gl.load`** | **`tl.range(..., num_stages=2)`** | the loop | **✓** |
| **`gl.load`** | **bare `range(...)`** | **the `add_schedule_loops(pm, ns)` argument** | **✓** |

1. **The depth has to reach `add_schedule_loops`; rewriting the loop is not the only way.**
   The pass reads `tt.num_stages` off the loop, and a Gluon `for` over the builtin lowers to
   an `scf.for` carrying none — Gluon has no `range` of its own, only `static_range`, which
   unrolls. So on the **loop-annotation** path the loop must be a `tl.range`, and
   `tl.range(num_stages=None)` inherits the launch value (which is how aiter's `_attn_fwd`
   works) while a bare `range` gets nothing.
   **The injection path does not read the loop at all**: `gluon_swp` passes the depth to
   `add_schedule_loops(pm, ns)` as an argument, so a bare `range` pipelines normally provided
   the loop has a `tt.dot`. Measured on a bare-`range` MoE GEMM, injection off → on:
   `ttg.memdesc_index` 0 → 4, `ttg.local_alloc` 0 → 2, `ttg.local_load` 0 → 10.
   **Do not rewrite a loop to `tl.range` just to use the shim** — that only matters when the
   loop or the launch has to carry the depth, i.e. a dot-free loop, where the annotation is
   the sole way to request the transform.
2. **The loads must still be `tt.load` when the pipeliner runs.** It anchors on global
   `tt.load`s whose forward slice reaches a `tt.dot`; an anchor written with explicit
   `gl.amd.cdna<N>.buffer_load` — which the transcription runbook asks for, because
   `gluon_to_ttgir` runs no buffer conversion — hands it ops it cannot see. Those two pieces
   of guidance genuinely pull against each other; `buffer_ops=True` resolves it by restoring
   plain's order rather than making you choose.

3. **`add_pipeline` is not the end of plain's pipeline — splice plain's whole tail.** Plain's
   `make_ttgir` keeps going after the pipeliner, and two of those passes decide whether the arm
   even runs. Without `remove_layout_conversions` the pipeliner's `local_load` stays in a blocked
   layout with a separate `convert_layout` to the dot operand still attached, so every operand
   takes a **second trip through LDS** on top of the staging; on a tile whose staging already
   fills the budget the arm then fails to **launch** on shared memory, which reads as "the
   injection does not work here" when the injection worked. Without `in_thread_transpose`
   (upstream gates it per-arch, so probe the gate — a static namespace read cannot see it)
   `optimize_dot_operands` picks a degenerate shared layout and one wide LDS read becomes many
   narrow ones. That pass also decides layout **fidelity**: with it the pipeliner emits the
   rotating shared layout Gluon has no constructor for, so an operand staging that is
   UNRECOVERABLE by hand comes back for free and `verify` can reach PASS instead of RECONCILED.
   The shim's default recipe splices the full tail and reports what it actually applied; read
   that report, because a pass absent on one minor is reported rather than silently skipped.
   **This tail was found missing twice, and neither symptom looked like "a pass is missing"** —
   once a launch-time out-of-resource, once a collapsed LDS access width.

At the TTGIR level an explicit Gluon loop is just a NON-pipelined `scf.for` — the exact object
the AMD pipeliner is built to consume. `add_schedule_loops` anchors on the loop's global
`tt.load`s whose forward slice reaches a `tt.dot` (tracing through `convert_layout`), assigns
pipeline stages, and `add_pipeline`'s `PipelineExpander` rewrites the loop into the multi-buffer
cross-iteration form: `local_load`-current at the top / global `tt.load`-next + `local_store`-next
at the bottom, buffers carried as `iter_args`. Downstream `UpdateAsyncWaitCount` then emits relaxed
`s_waitcnt lgkmcnt(N>0)` in place of the full-drain `lgkmcnt(0)`. That is the whole
`MfmaUtil↑ / cadence↓ / waitcnt-relax` win at unchanged registers and occupancy.

**Kernel side (give the pipeliner room):** load K/V **in-body** with **NO hand register-prefetch**
(the pipeliner *is* the prefetcher — a manual prefetch uses up the slot), and **do not hand-write
the LDS staging at all** — no `allocate_shared_memory` / `local_store` / `local_load`, no
`gl.barrier()`. Building that path is what the pass exists to do, and the faithful-anchor shape
starves it. Measured on a 2048³ fp16 GEMM: hand-staged 2/2/2 LDS ops with 2 barriers and no
multi-buffering = 1.000; un-staged with the injection = **2/4/4, `memdesc_index 4`, zero
barriers, 1.088**. Note the trap in between — un-staged with the injection OFF is **0.811**, a
19% regression, so the two halves go together or not at all. Also **split the causal mask into
TWO loops** (a clean branch-free full-region loop; a single loop with a loop-variant `scf.if`
blocks the pass and `BlockPingpong`). `num_stages=2` (ns3 usually worse; ns4 chained-dot
regresses). Composes on top of LPT causal remap + XCD (`../workloads/attention.md`).

**Proof it landed (all must move; TFLOPS alone is not proof):** TTGIR `local_alloc`/`local_store`/
`local_load` counts jump toward plain and `ttg.memdesc_index` appears — that last one is the
multi-buffer tell and it is the cheapest single signal. Read it off the dumped `.ttgir`, or off
`gluon_swp` armed vs unarmed. (`probe_levers.py` answers a **different** question — whether the
symbols exist in this `.so` — and it takes **no positional argument**: `--all` is the whole CLI,
so the `probe_levers.py reinject_ttgir_pipeliner` form some notes still carry exits non-zero.
Symbols existing is not the pass biting; those are two hypotheses and they come apart here.)
Then: `asm_loop_audit.py` shows `ds_read`/full-drain `lgkmcnt(0)`
down + relaxed `lgkmcnt(N>0)` appearing; `mfma_efficiency.py` cadence down; rocprofv3 `MfmaUtil`
up with occupancy unchanged; correctness rel < 2e-2 + determinism maxdiff == 0. Machine lever:
`references/hardware/lever-cards.json` group `pipeline` → `reinject_ttgir_pipeliner` (Route 1).
Route 2 = author the overlap yourself (below); Route 3 = LLIR co-design (last section). In practice
on gfx942 attention this reached near-parity with / matched plain across most shapes.

## Auto-recovering the pipeline structure (reproduce, then improve)

This is the standard start of the pipeline layer, not a discouraged opt-in. The
faithful layouts-only anchor stays the **attribution baseline** (built first, in
transcribe); the pipeline layer then **reproduces plain's pipeline and improves on
it**:

1. **Reproduce.** The post-pipeliner plain `.ttgir` physically contains the double
   buffer (`ttg.local_alloc` multi-buffer + `async_copy`/`commit`/`wait`), and Gluon
   can express all of it, so `scripts/recover_gluon.py --with-pipeline` (or `dump_ir.sh
   --emit-gluon pipeline`) — gluon pack — emits a prologue/loop/epilogue scaffold with the recovered
   `nBuffers` / `wait_group(N)` / mask shape and the recovered layouts wired in.
   Kernel-specific addressing (base ptrs / offsets / mask guard) is left as a `TODO`
   skeleton placeholder (intentional — you fill it from the algorithm skeleton, not the
   pipeline structure). This recovers
   the dominant lost-pipeline gap (`../phases/transcribe.md ## Re-profile +
   recalibrate`) and gets the Gluon line back near plain quickly.
2. **Improve.** Then go beyond what plain did: deeper buffering (3-stage local
   prefetch), operand/`LR` prefetch, manual `AC`/`LR`/`DOT` interleave, and
   ping-pong **only where the data dependency allows** (an online-softmax recurrence
   blocks symmetric ping-pong, `../workloads/attention.md`). Attribute each gain
   separately and guard every register-buffered step with the **tri-lemma /
   occupancy-cliff** check (`slicing.md ## Occupancy budget (P8)`): predict the
   post-change waves/CU and verify against the no-overlap baseline before keeping it.

Attribution note: keep the faithful layouts-only anchor as the baseline so a
recovered+improved pipeline's gain is attributable; do not fold the pipeline into
the transcription step. Whether the explicit stages actually *interleave* (vs MFMA /
`ds_read` clumping) still depends on the build-specific LLIR scheduler (a post-TTGIR
concern, not recovered) — `compiler-contract.md`.

## Route 2: author the overlap yourself (what the pipeliner cannot express)

Reach here in one of two situations: the loop has **no `tt.dot`**, so Route 1 has nothing to
anchor on; or Route 1 already landed and you want something the auto-pipeliner does not do at
all -- per-tensor pipeline depth, several independent async chains at staggered depths, or
sub-buffer splitting. The Gluon-native surface for those, with worked examples from production
kernels and the per-architecture availability, is `../gluon/pipeline-reference.md ## Authored
overlap`; this section is the scheduling model and the correctness rules behind it.

> **Route 1 and Route 2 do not compose in the same loop.** Hand-written staging is exactly what
> starves the pipeliner (it collects `local_load`s whose source is a loop-carried
> `BlockArgument`; hand staging reaches LDS through `memdesc_index` instead), so a kernel moving
> to Route 1 must have its staging **removed** first, and partial removal does not count -- one
> `ttg.barrier` left in the loop makes the pass skip it entirely and in silence. The refusal you
> get for keeping it names the op, not the cause
> (`'ttg.local_alloc' op pipeliner doesn't know how to predicate this op`); read it as "the hand
> staging is still there" rather than as a language limit, which one transcription concluded and
> had to retract.
>
> **The two halves go together, and even together they do not always win.** De-staging alone is a
> large regression, the injection has to earn that back before it earns anything, and on a
> measured kernel the net stayed NEGATIVE on most toolchain versions while the injection itself
> was worth a consistent speedup on all of them. So:
>
> - **Pre-check before rewriting the body.** Compare the *plain* kernel's own `num_stages=1`
>   against its `num_stages=2`. Where plain itself gains nothing from pipelining on a version,
>   recovering the pipeline for a transcription did not pay off there either -- that correlation
>   held on every version tested.
> - **Judge the net on the same-window per-rep ratio** of armed-and-de-staged over the original
>   anchor. Differencing two percentages against a shipped baseline whose own spread is a few
>   percent cannot resolve an effect this size.
>
> Per-kernel numbers live in the trial report, not here.

### Stages

- **2-stage global prefetch + double buffer**: prefetch tile `k+1`'s `AC` while
  computing `DOT(k)`; `nBuffers=2`, `wait_group(1)`. Hides HBM latency
  (~400 cycles).
- **3-stage local prefetch**: also prefetch `LR(k+1)` (LDS->reg) while `DOT(k)`
  runs, so the MFMA does not wait on the same-iteration `ds_read`.

```text
2-stage:  AC(k+1) , LR(k)   , DOT(k)
3-stage:  AC(k+2) , LR(k+1) , DOT(k)
```

### Independence rule (correctness of pipelining)

`DOT(k)` must not depend on the same-slot `LR(k+1)` or `AC(k+2)`. Each buffer
must be retired (`wait_group`) before it is overwritten. Violations show as
wrong results or as the scheduler refusing to interleave.

### Hand-built buffering rules (correctness + scheduling footguns)

These rules apply when you are authoring the staging yourself (Route 2). They do **not**
apply to a Route 1 body, which has no hand-written staging at all. Three of them are easy
to violate and hard to debug:

- **Buffer index may need to be compile-time — scope this one before paying for it.** The
  concern is real only where the *async* hazard analysis has to prove overwrite-safety:
  with `smem.index(k % nBuffers)` the buffer lifetime is not statically visible, so the
  scheduler can decline to interleave, and the fix is to unroll by the buffer period and
  give each static sub-iteration a literal index (`smem.index(0)`, `smem.index(1)`, ...).
  **Two pieces of evidence say it is not a blanket rule.** Over *sync* staging on gfx942 the
  two forms compiled to identical ISA — same `ds_read` / `ds_write` / `s_barrier` counts and
  the same VGPR count — so there was nothing to buy; and aiter's shipped block-scaled GEMM
  indexes its async main loop with exactly `k_iter % NUM_STAGES`, unrolling only its
  wind-down and citing **register allocation** for that (removing a PHI that would push the
  dot operands out of AGPRs), not scheduling. The async form is untested here because
  `cdna4.async_copy` does not lower on gfx942 (`../gluon/pipeline-reference.md ## Authored
  overlap`). So: measure both forms on your target before unrolling for this reason alone —
  the unroll has its own cost (next bullet but one, and the WAR-barrier corollary below).
- **`wait_group(N)` must match the groups intentionally left in flight.**
  Recompute `N` whenever the prologue issue count, region count, or unroll
  factor changes -- never copy a value from another shape or stage count. A
  stale `N` either drains too early (kills overlap) or lets `smem.load` read a
  buffer before its async group retired (wrong results).
- **`nBuffers` must equal the stage depth.** A 3-stage pipeline needs 3 buffers
  (one in flight, one consumed-current, one in transit); dropping to 2 forces
  `DOT` to wait on `AC` and silently collapses back to 2-stage.
- **Stage through shared memory the pipeline pass can see (phase-visibility).** A
  transform's correctness depends on what is visible to it **at the phase it runs**.
  Implicit/compiler-managed scratch created in a *later* phase — e.g. the shared
  scratch a `convert_layout` allocates after the warp-pipeline pass has already run
  — is **not protected by that pass's hazard analysis**, so staggered groups can
  race on it (silent wrong results / NaNs). Stage cross-stage data through
  **explicitly allocated** shared memory (`gl.allocate_shared_memory` +
  `buffer_load_to_shared`) that the pass can see, not through scratch a later pass
  introduces.

### Vetted double-buffer skeleton (copy, then specialize)

A correct 2-stage double buffer (`nBuffers` == stage depth == 2). The point is the
**retire ordering**: the prologue issues `k=0`; each iteration prefetches the
*other* buffer, keeps exactly one async group in flight (`wait_group(1)`), and DOTs
the current buffer; the last iteration drains (`wait_group(0)`). Copy this instead
of re-deriving the staging per kernel.

```python
s = gl.allocate_shared_memory(dtype, [2, *tile], shared_layout)  # nBuffers == 2

_acp.buffer_load_to_shared(s.index(0), base_ptr, off(0))         # prologue: issue k=0
_acp.commit_group()

for i in range(NUM):
    cur = i % 2
    nxt = (i + 1) % 2
    if i + 1 < NUM:
        _acp.buffer_load_to_shared(s.index(nxt), base_ptr, off(i + 1))  # prefetch k+1
        _acp.commit_group()
        _acp.wait_group(1)        # retire all but the one in-flight prefetch
    else:
        _acp.wait_group(0)        # drain on the last iter
    a_k = s.index(cur).load(dot_operand_layout)   # LDS -> reg for THIS iter only
    acc = gl.amd.cdna4.mfma(a_op, a_k, acc)       # DOT(k): independent of s.index(nxt)
```

Specialize: for 3 stages use `nBuffers == 3` + `wait_group(2)`, and recompute
`wait_group(N)` whenever the prologue / region / unroll changes
(`### Hand-built buffering rules`). The `i % 2` form here is the readable correctness
template; whether the literal-index unroll also schedules better is target-dependent and
measurable — see that section before assuming it does. `off(k)` is the per-iteration global
offset; `_acp` is the cdna4 async-copy module.

### Epilogue: interleave stores, do not burst

The prologue/epilogue are part of the schedule, not free drain code:

- Convert the accumulator into an **explicit blocked store layout**
  (`convert_layout(acc.to(out_dtype), gStoreLayoutC)`) before `buffer_store` /
  `gl.store`; an implicit per-thread store layout tends to emit uncoalesced or
  oddly-strided writes.
- **Interleave the stores with the final `DOT` regions** so each store rides an
  MFMA-cycle gap. Bursting all stores after `wait_group(0)` clusters the write
  traffic into a tail the hardware cannot hide and can undo hot-loop wins.

## Budget before deepening

```text
stall_hbm = max(0, latency_hbm_to_LDS - work_after_AC)
stall_lds = max(0, latency_LDS_to_reg - work_after_LR)
effective_pipeline_depth = min(num_stages - 1,
                               floor(32KiB / (active_waves * data_per_request)))
```

Gate before adding a stage (all must hold): (a) the async copy passed its **smoke
test** in isolation (`memory-path.md ## Async copy: smoke-test before wiring (mandatory)`);
(b) `DOT(k)` is independent of the same-slot `LR(k+1)` / `AC(k+2)`
(`### Independence rule`); (c) the **occupancy budget** shows pipeline depth is the
bound -- not LDS/occupancy (`slicing.md ## Occupancy budget (P8)`): a deeper
pipeline costs LDS buffers + prefetch registers, which can drop waves/CU and
regress. **If waves are already VGPR/LDS-capped, software prefetch (register OR LDS
double-buffer) regresses** because it adds the wave-capping resource — raise
occupancy first. This is the prior for occupancy-latency-hidden access patterns
(e.g. gather / decode where the gathered data is L2-resident and its latency is
already hidden by occupancy): check `R_total` / waves-per-CU before prefetching.
This is one face of the **overlap / occupancy / ILP tri-lemma** (overlap costs
VGPR, occupancy needs few VGPR, deep-unroll ILP costs LDS — pick two); predict the
post-change waves/CU before deepening (`slicing.md ## Occupancy budget (P8)`).

Then deepen only if `stall_reduction > extra_cost`. Extra cost = extra LDS stages
(capacity), prefetch registers (`R_prefetch` in the budget), prologue/epilogue
drain, and `wait_group` complexity. A 3rd stage that overflows LDS or pushes
`R_total` past 512 will regress (this is the over-unroll spill lesson — see
`slicing.md`).

**Unroll has a shallow sweet spot for hiding a fixed-latency hazard.** A 2-block
unroll lets block B's MFMA fill block A's MFMA-write -> VALU-read gap (`s_nop`,
`../hardware/planning-constants.md ## Extended planning (attention / fused kernels)`), so it is usually the
win. But **unrolling further does not keep helping**: the compiler does not reliably
exploit the extra independent blocks to hide that hazard, and the larger loop body
adds scheduling stalls (measured: a 4-block unroll *raised* `s_nop`/iter and
regressed vs 2-block). Treat ">2× unroll" as a tri-lemma cost (more LDS buffers, more
VGPR) that must beat the no-extra-unroll baseline, not a free ILP knob.

Corollary: **double-buffering purely to DROP a WAR barrier is not free either.** *If* your
target turns out to need a compile-time buffer index (`### Hand-built buffering rules` —
measure, do not assume), alternating buffers to remove the write-after-read barrier forces
unrolling by the buffer period,
which doubles the live compute state of the unrolled body and can SPILL even with **no
explicit prefetch**. Budget the post-unroll `R_total` before assuming the saved barrier
is a net win; on a VGPR-capped (1-wave) kernel it usually is not.

## Route 3: LLIR scheduler co-design (last, and probably not available to you)

**Applicability gate -- all three must hold, or skip this section entirely:**

1. the kernel is a **GEMM-shaped** pure MFMA -> MFMA accumulator chain. With VALU between
   matmuls (softmax / scale / dequant) the scheduler emits **invalid IR** -- it asserts, it does
   not merely fail to help;
2. you are on a build that **has** the pass. `TRITON_ENABLE_LLIR_SCHED` is not upstream: it was
   absent from all four clean upstream containers checked here (3.6.0 / 3.7.0 / 3.7.1 / 3.8.0);
3. Routes 1 and 2 are already exhausted, so the residual really is instruction interleave rather
   than missing overlap.

The hand-built AC/LR/DOT interleave in Route 2 is the portable answer when (1) or (2) fails.

### The toggle and its target interleave

> For the LLIR-scheduler **cadence arithmetic** (`mfmaPerGR`, the `needed=...` anchor budget) + the
> `hasOnlyPrefetchedMFMA` **fire-condition** stated as a derivation rule, and for **authoring a NEW
> scheduler/RA/amdgcn pass from scratch** on a stock triton with no reference source, see
> `llvm-codesign-handbook.md` (Part 8 + Part 0-5). This section is the toggle-and-target summary.

The author writes independent AC/LR/DOT; the compiler interleaves at throughput
rates only with `TRITON_ENABLE_LLIR_SCHED=1`. Target interleave:

- FP16: ~4x 16-cyc MFMA per `buffer_load` (64-cyc issue spacing), ~4x MFMA per
  `ds_read_b128`.
- BF8: ~2x 32-cyc MFMA per mem op (unless `BLOCK_K` is doubled).

`LLIR_SCHED` is **GEMM-only** (pure MFMA -> MFMA accumulator chain); on
attention / VALU-between-matmul kernels it **asserts** (invalid IR). There is no
portable knob for that case — interleave AC/LR/DOT manually (above), or, under
sanctioned co-design, author a dependency-preserving LLIR pass referencing
`llir-sched` (`compiler-contract.md ## Scenario B`; the local build's `ATTN_SCHED`
is one build-specific instance, not portable). Toggle order and IR verification live
in `compiler-contract.md`.

## IR / asm acceptance signals

| Change | Confirm in IR/asm |
| --- | --- |
| 2/3-stage prefetch | `wait_group(N)` present; loads for `k+1`/`k+2` issue before `DOT(k)` |
| + llirSched | MFMA interleaved with `buffer_load` / `ds_read` (not all MFMA clustered); compare the iter-end `v_accvgpr_mov` block |
| pipeline correct | `wait_group` retires each LDS buffer before the next `AC` overwrites it |

## Reprofile signal

After a pipeline change, MFMA efficiency should rise toward the budget target;
if it does not, check (1) the scheduler knob is actually on (IR), (2) the LDS
layout is conflict-free (`ds_read` 16-cyc), (3) no new spills. Then reclassify —
the next bound class may now be register or memory.
