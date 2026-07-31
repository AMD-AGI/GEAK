# Pipeline Layer (gfx950)

Read this for the pipeline layer (after LDS layout, before slicing). Goal: hide
HBM and LDS latency by staggering AC / LR / DOT so the MFMA unit never waits.

## Stages

- **2-stage global prefetch + double buffer**: prefetch tile `k+1`'s `AC` while
  computing `DOT(k)`; `nBuffers=2`, `wait_group(1)`. Hides HBM latency
  (~400 cycles).
- **3-stage local prefetch**: also prefetch `LR(k+1)` (LDS->reg) while `DOT(k)`
  runs, so the MFMA does not wait on the same-iteration `ds_read`.

```text
2-stage:  AC(k+1) , LR(k)   , DOT(k)
3-stage:  AC(k+2) , LR(k+1) , DOT(k)
```

## Independence rule (correctness of pipelining)

`DOT(k)` must not depend on the same-slot `LR(k+1)` or `AC(k+2)`. Each buffer
must be retired (`wait_group`) before it is overwritten. Violations show as
wrong results or as the scheduler refusing to interleave.

## Hand-built buffering rules (correctness + scheduling footguns)

You build the multi-buffer pipeline by hand (Gluon runs no automatic
pipeliner -- see the `num_stages` section). Three rules are easy to violate and
hard to debug:

- **Buffer index must be compile-time.** `smem.index(k % nBuffers)` (runtime
  modulo) is an anti-pattern: the buffer lifetime is no longer statically
  visible, so the scheduler cannot prove overwrite-safety and may refuse to
  interleave. Unroll by the buffer period and give each static sub-iteration a
  literal index (`smem.index(0)`, `smem.index(1)`, ...).
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

## Vetted double-buffer skeleton (copy, then specialize)

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
(`## Hand-built buffering rules`). For the scheduler to actually *interleave*, give
each buffer a **compile-time** index by unrolling the buffer period -- the `i % 2`
form here is the readable correctness template; the literal-index unroll is the
scheduling-optimal one (same rules section). `off(k)` is the per-iteration global
offset; `_acp` is the cdna4 async-copy module.

## Epilogue: interleave stores, do not burst

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
(`## Independence rule`); (c) the **occupancy budget** shows pipeline depth is the
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

Corollary: **double-buffering purely to DROP a WAR barrier is not free either.** Since
the buffer index must be compile-time (`## Hand-built buffering rules`), alternating
buffers to remove the write-after-read barrier forces unrolling by the buffer period,
which doubles the live compute state of the unrolled body and can SPILL even with **no
explicit prefetch**. Budget the post-unroll `R_total` before assuming the saved barrier
is a net win; on a VGPR-capped (1-wave) kernel it usually is not.

## Compiler co-design (mandatory for the pipeline to land)

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

## `num_stages` / auto-pipeliner do not carry to Gluon (source-verified)

Verified in `triton @ gfx950-tutorial` (`third_party/amd/backend/compiler.py`):
the **plain** lowering `make_ttgir` runs the automatic software pipeliner
(`add_schedule_loops(num_stages)` + `add_pipeline`) and the arch-gated automatic
ping-pong (`add_block_pingpong(num_stages)`). The **Gluon** lowering
`gluon_to_ttgir` runs **none of these** (it runs `add_warp_pipeline`). So in Gluon:

- the automatic software pipeline and automatic block-ping-pong **do not run** —
  you build the multi-buffer pipeline **by hand** (as in the Stages section
  above): `nBuffers` shared buffers
  (`gl.allocate_shared_memory([nBuffers, ...])`),
  `gl.amd.cdna4.async_copy.buffer_load_to_shared(smem.index(g_idx), ...)` +
  `commit_group()` / `wait_group(N)`, and an explicit prologue + loop order that
  issues the `k+1` / `k+2` loads before `DOT(k)` (the GEMM-tutorial prefetch pattern);
- `num_stages` survives only as a **budget parameter** (buffer count): the
  bandwidth model uses `in_flight = min(..., num_stages - 1)`; it is not the
  pipelining trigger in Gluon;
- Gluon also exposes `with amd.warp_pipeline_stage("load" | "prep" | "compute",
  priority=0..3):` to mark stages (`priority` lowers to `s_setprio`, 0-3) — but
  the GEMM tutorial uses the manual `wait_group` buffering above, not this marker;
- whether the explicit stages actually interleave (vs MFMA / `ds_read` clumping)
  depends on the build-specific LLIR scheduler (`compiler-contract.md`), not on
  `num_stages`. On a build without it, the buffers still hide some latency but the
  default LLVM scheduler can re-cluster MFMA — record a scoped ceiling.

> **CORRECTION (verified FA-fwd gfx942):** the second bullet ("`num_stages` is not the
> pipelining trigger in Gluon") is only true for the *default* lowering. Those two passes
> **are re-injectable** into `gluon_to_ttgir`, and once spliced in (Python-only `compiler.py`
> edit, no rebuild) **launch `num_stages>1` becomes the trigger** and reproduces plain's
> multi-buffer loop — see the next section. So prefer **Route 1** before hand-building or
> recording a schedule ceiling.

## Reproduce plain's software pipeline on the Gluon path (Route 1 — default, no rebuild)

The auto-pipeliner overlap **is re-injectable** into an explicit Gluon loop — run plain's own
TTGIR passes on the Gluon IR. Plain's overlap = `add_schedule_loops(num_stages)` +
`add_pipeline(...)` in `make_ttgir`; `gluon_to_ttgir` skips them. Re-inject them (default-OFF,
gated by launch `num_stages>1` so it is cache-key-safe; **NO `libtriton.so` rebuild** — the
passes already exist in the `.so`):

```python
# in gluon_to_ttgir, after add_combine_tensor_select_and_if, when options.num_stages>1:
add_optimize_dot_operands; add_schedule_loops(ns); add_pipeline(use_async_copy, use_block_pingpong)
; add_convert_to_tensor_ops; canonicalizer; remove_layout_conversions; reduce_data_duplication
; (in_thread_transpose if enabled); move_up_prologue_loads; canonicalizer; cse
```

At the TTGIR level an explicit Gluon loop is just a NON-pipelined `scf.for` — the exact object
the AMD pipeliner is built to consume. `add_schedule_loops` anchors on the loop's global
`tt.load`s whose forward slice reaches a `tt.dot` (tracing through `convert_layout`), assigns
pipeline stages, and `add_pipeline`'s `PipelineExpander` rewrites the loop into the multi-buffer
cross-iteration form: `local_load`-current at the top / global `tt.load`-next + `local_store`-next
at the bottom, buffers carried as `iter_args`. Downstream `UpdateAsyncWaitCount` then emits relaxed
`s_waitcnt lgkmcnt(N>0)` in place of the full-drain `lgkmcnt(0)`. That is the whole
`MfmaUtil↑ / cadence↓ / waitcnt-relax` win at unchanged registers and occupancy.

**Kernel side (give the pipeliner room):** load K/V **in-body** with **NO hand register-prefetch**
(the pipeliner *is* the prefetcher — a manual prefetch uses up the slot), and **split the causal
mask into TWO loops** (a clean branch-free full-region loop; a single loop with a loop-variant
`scf.if` blocks the pass and `BlockPingpong`). Launch `num_stages=2` (ns3 worse; ns4 chained-dot
regresses). Composes on top of LPT causal remap + XCD (`../workloads/attention.md`).

**Proof it landed (all must move; TFLOPS alone is not proof):** TTGIR `local_alloc`/`local_store`/
`local_load` counts jump toward plain (`scripts/probe_levers.py reinject_ttgir_pipeliner`, or run
the two passes on a dumped `.ttgir`); `asm_loop_audit.py` shows `ds_read`/full-drain `lgkmcnt(0)`
down + relaxed `lgkmcnt(N>0)` appearing; `mfma_efficiency.py` cadence down; rocprofv3 `MfmaUtil`
up with occupancy unchanged; correctness rel < 2e-2 + determinism maxdiff == 0. Machine lever:
`references/hardware/lever-cards.json` group `pipeline` → `reinject_ttgir_pipeliner` (Route 1).
Route 2 = hand-built cross-iteration double-buffer (below); Route 3 = authored pass. In practice
on gfx942 attention this reached near-parity with / matched plain across most shapes.

## Auto-recovering the pipeline structure (reproduce, then improve)

This is the standard start of the pipeline layer, not a discouraged opt-in. The
faithful layouts-only anchor stays the **attribution baseline** (built first, in
transcribe); the pipeline layer then **reproduces plain's pipeline and improves on
it**:

1. **Reproduce.** The post-pipeliner plain `.ttgir` physically contains the double
   buffer (`ttg.local_alloc` multi-buffer + `async_copy`/`commit`/`wait`), and Gluon
   can express all of it, so `scripts/recover_gluon.py --with-pipeline` (or `dump_ir.sh
   --emit-gluon pipeline`) emits a prologue/loop/epilogue scaffold with the recovered
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
