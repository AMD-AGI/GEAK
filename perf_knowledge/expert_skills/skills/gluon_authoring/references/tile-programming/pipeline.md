# Pipeline Layer (gfx950 baseline, gfx942 downgrade)

Hide HBM and LDS latency by staggering AC (async global→LDS) / LR (LDS→reg) / DOT so the MFMA unit does
not wait. Written for gfx950 (CDNA4) on upstream Triton 3.8.0; gfx942 (CDNA3) changes are marked
*gfx942 downgrade* (sync staging; async only at 32 bit per lane with a destination `order` matching the source's fast dim; no
`ds_read_b64_tr`; 64 KiB LDS / 32 banks vs 160 KiB / 64). Runnable examples:
`scripts/pipeline_examples_cdna4.py` (C1–C5) and `scripts/pipeline_examples_cdna3.py` (A5, B1);
`--compile-only` checks them on CPU. This page is the single home for the overlap options and the signals that point at each.

## Where the overlap comes from, and it is not the same question per tier

| tier | who builds the overlap | starting point |
| --- | --- | --- |
| plain Triton | the compiler (`make_ttgir` pipeliner) | afford it, confirm it fired (`## Plain Triton: the pipeliner runs for you`) |
| Gluon | **you** — `gluon_to_ttgir` has no pipeliner | walk `### The overlap options and the signal for each` |
| Gluon below parity | you, re-authoring what plain built | the same rungs from the recovered scaffold (`## Recovering the structure, then improving on it`); re-injection mainly sizes the debt (diagnostic) |

**Price a ring before writing it:** LDS/workgroup = Σ slots × tile bytes; workgroups/CU = min(LDS limit,
register limit) (`slicing.md ### Occupancy budget (P8)`). A second slot that halves workgroups/CU trades
cross-workgroup hiding for in-workgroup overlap. A ring on one operand only is the fallback when two do not
fit. **Shrinking `BLOCK_K` to fit a ring does not change barriers per MFMA** (half the MFMAs, twice the
iterations). **On a single slot a prefetch can move the stall into the barrier** — read the barrier share of
stalls alongside the memory-wait share before calling it overlap.

### The overlap options and the signal for each

The rungs below are options with the signal that points at each, not a fixed sequence; the numbering is the
order the sections appear in.

0. **Does this loop want a pipeline at all?** (`### First: does this loop want a pipeline at all?`)
1. **Register-level prefetch** — fits when the tile is register-resident or each iteration issues
   few MFMAs (decode GEMM, chunk-recurrence walks) (`### Register-level prefetch (step 1)`). AMD's gfx950 GEMM
   tutorial builds its compute-bound LDS-staged GEMM tiles from step 2 (the async direct-to-LDS ring) and adds
   the LDS->VGPR prefetch on top of it. Measured on gfx950 / Triton 3.8.0 (bf16 GEMM): on decode tiles
   register prefetch beat or tied every ring depth tried; on a compute-bound LDS-staged tile it only tied the
   synchronous body (or spilled at 256x256) while the async ring paid — pick the first overlap by regime.
2. **An authored LDS ring** — pays when the staged block feeds enough MFMA work to hide the copy.
   gfx950: `gl.amd.cdna4.async_copy.buffer_load_to_shared` / `global_load_to_shared` (destination first) +
   `commit_group()` + `wait_group(N)`, `load_shared_relaxed` for a read a `wait_group` retired (API semantics:
   `triton/experimental/gluon/language/amd/cdna4/async_copy.py` docstrings; tell: the staging `ds_write`
   disappears). **Lowering preconditions on Triton 3.8.0 (else LLVM translation fails, no fallback; compile-checked): per-lane
   width exactly 128 or 32 bit on gfx950 (gfx942: 32 only; 64 fails on both)**, coalesced writes, swizzle within a warp; a padded destination needs interval ≥ 64 lanes × per-lane bytes and a producer derived from its
   `offset_bases` (`../gluon/memory-reference.md ### Shared-layout family + transpose-on-read (layout
   dependency)`). gfx942: sync staging is the usual base (the 32-bit async form lowers but was recorded slower in the cases tried;
   recorded, not re-measured here — no gfx942 device; A/B it if the copy is the bound).
3. **Scheduling model** (`scheduling-model.md ## How to choose`): `inter_wave` →
   `gl.amd.warp_pipeline_stage` (needs `num_warps >= 8`, waits only between stages; `warp-pipeline.md`);
   `compiler_interleave` / intra-wave pacing → empty-asm fence, placed `s_nop`, `llvm_fn_attrs` strategy
   (`instruction-scheduling.md`). The stock `coexec` strategy is gfx1250-only (automatic at
   `num_warps <= 4`); on gfx950/gfx942 forcing it via `llvm_fn_attrs` draws LLVM's "only supported for
   gfx1250" warning and a non-coexec schedule (compile-checked, 3.8.0). The stock 3.8.0 `compiler.py` exposes no
   hook for inserting an authored LLVM pass.
4. **Re-inject plain's pipeliner** (mainly a diagnostic; `../gluon/pipeline/reinjection.md`). It replaces
   hand-built staging rather than adding to it, so on an incumbent it discards the authored body; it does not
   combine with rungs 1–2 in the same loop (v3.8.0 `isSafeToPipeline` skips a loop containing a `ttg.barrier`).

**`num_stages` is not on this list:** no pass in `gluon_to_ttgir` reads it (3.6.0–3.8.0, read from the
compiler source). It is only a budget parameter; plain's tuned value has no effect as a Gluon knob.

### The four layers, and how often surveyed kernels use them

| layer | mechanism | sites in surveyed gfx950 kernels | availability |
| --- | --- | --- | --- |
| 1 static staging | register prefetch, then the authored ring | dominant | stock 3.8.0 |
| 2 intra-wave | empty-asm fence, `s_nop`, `llvm_fn_attrs` | clear second | stock 3.8.0, both tiers (`make_llir`) |
| 3 inter-wave | `warp_pipeline_stage`, or hand-rolled `s_setprio` | rare | Gluon only; symbol 3.7.0+, race-free GPU-verified on 3.8.0 / 3.9.0 only (3.7.1: wrong results) |
| 4 re-inject plain's pipeliner | stage hook / `gluon_swp.py` | absent | diagnostic (`../gluon/pipeline/reinjection.md`) |

Static site counts, not measurements; the production Gluon kernels surveyed set no env vars and some pass
`num_stages=1` explicitly. A pipeline regression can be a layout question: in the cases recorded, an overlap
mechanism lost to a synchronous LDS-tiling / MFMA-layout restructure, so the layout is worth checking before
the overlap direction is closed.

## Plain Triton: the pipeliner runs for you

`make_ttgir` runs `add_schedule_loops(num_stages)` + `add_pipeline` and, arch-gated, `add_block_pingpong`.
`num_stages` buys buffers and trades against `BLOCK_K` for the same LDS, so the two interact in a sweep.

- **Confirm it fired:** it declines silently when LDS/registers do not fit. Tell: multi-buffer `local_alloc`
  plus `ttg.memdesc_index` in the TTGIR.
- **BlockPingpong gate** differs per chip on 3.8.0 (gfx950 only with async copy on; gfx942 on by default; both
  overridable by `knobs.amd.use_block_pingpong`, and it needs `num_stages > 1`): `triton/backends/amd/compiler.py`
  `is_pingpong_schedule_enabled`.
- Hand ping-pong details that transfer to Gluon: a two-cluster schedule uses a bare `s_barrier` (not a
  local-fencing one) to keep the LDS wait out of the compute slice; the chained-dot schedule pins
  `s_waitcnt lgkmcnt(0)` at the end of the memory cluster and the `s_barrier` at the loop top (at the bottom
  the backend sinks induction scalars into compute).

If the residual gap is about *where* the prefetch lands or how many buffers exist, plain exposes no direct
control for it (`num_stages` sets depth, not placement) — that points to Gluon more than to an untried
`num_stages`.

## Gluon: the pipeliner does not run

`gluon_to_ttgir` (3.8.0) runs nine passes; the pipeliner trio is not among them (`add_warp_pipeline` is).

- `num_stages` at launch has no effect on the Gluon path (3.6.0–3.8.0, compiler source); the unroller only writes
  `tt.num_stages` onto an unroll epilogue.
- The passes are absent from the lowering, not from `libtriton` — re-injectable without a rebuild (rung 4).
- **`loop_unroll_factor` is live on 3.8.0** (the unroller entered the Gluon pass list at 3.8.0): a keyword on
  `tl.range(...)` inside the body, not a launch option. Unrolling duplicates the body, which moves a
  hand-scheduled ring's buffer rotation and barriers; the value that **suppresses** unrolling keeps the ring as
  written
  (`../gluon/pipeline/loop-knobs-and-targets.md ## Roll the loop (cut i-cache pressure)`).

### First: does this loop want a pipeline at all?

A pipeline has little latency to hide when the tile is register-resident, the loop unrolls fully at trace
time (`static_range`: no `scf.for`), or the body issues only a handful of MFMAs per iteration. **One option
there is source-level reordering of independent work into the dependency shadow** — no buffer, no barrier.

## Authoring the overlap yourself (the climb default on the Gluon path)

Rungs 1–2. Also the route for incumbents, and for a loop with no `tt.dot` (the re-injected pipeliner is
built around dot operands; `../gluon/pipeline/reinjection.md`). API surface and per-arch
availability: `../gluon/pipeline-reference.md ## Authored overlap (no compiler patch)`.

### Register-level prefetch (step 1)

Reminder: `DEPTH` named loop-carried bundles, a dynamic loop stepping `DEPTH` with a `static_range` slot
(literal index), refill **before** the MFMA, tail guarded; costs `DEPTH × bundle` VGPRs, same on gfx950/gfx942.
Combined with a multi-stage LDS ring in the same loop it has shown up as a register regression; a
`gl.static_assert` can keep the two apart.

### Independence rule (correctness of pipelining)

2-stage `AC(k+1), LR(k), DOT(k)` (2 buffers, `wait_group(1)`); 3-stage `AC(k+2), LR(k+1), DOT(k)`; retire each
buffer before it is overwritten.

### Hand-built buffering rules (correctness + scheduling footguns)

- `wait_group(N)` counts commit groups (`async_copy.py` `wait_group` docstring) and takes a trace-time int:
  `N = groups_per_stage × stages_in_flight`, recomputed whenever prologue, region count or unroll changes; in
  a dynamic loop branch between two literals for the tail (skeleton below).
- **A runtime buffer index `smem.index(k % N)` is legal and cheap**: on gfx950 (3.8.0 compile, sync and
  async staging) it lowered to a couple of scalar ops per iteration, no VALU or extra LDS traffic. Literal
  unrolling doubles the body and raised VGPRs in the cases compiled (an earlier gfx942 sync-staging compile
  gave the same ISA as the literal form), so unroll only if measured better.
- `nBuffers` = async lead distance + 1 (tiles resident in LDS at once); depth sets prologue/drain and can
  exceed it (`warp-pipeline.md ### How long the drain is, and why the skeleton's formula does not generalize`).
- Stage cross-stage data through `gl.allocate_shared_memory`, not `convert_layout` scratch (allocated after
  the warp-pipeline pass, unprotected by its hazard analysis).

### Prefetch has three orthogonal degrees of freedom, and an unpinned one does not land

Which piece; which side of which sync point (non-monotone where measured, so a sweep covers points a bisection
skips); how deep. **A hoisted LDS read that is not pinned is re-scheduled toward its consumer and then measures as no
change.** Pin it by passing the independent work that should follow (e.g. the accumulator) through an empty
`"=v,0"` asm placed after the read; tying the read result itself did not hold it (3.8.0 compile, gfx950) —
see the empty-asm fence
(`instruction-scheduling.md ## The empty-asm scheduling fence — the most-used and least-documented lever
here`); the `.s` shows whether the `ds_read` moved and its `s_waitcnt` did not, which is worth confirming
before timing.

### Vetted double-buffer skeleton (copy, then specialize)

gfx950 async ring, 2 buffers. Numerics-checked versions: `scripts/pipeline_examples_cdna4.py` C1–C4.

```python
_acp = gl.amd.cdna4.async_copy        # MODULE scope: binding a module inside the @gluon.jit body is a compile error
s = gl.allocate_shared_memory(dtype, [2, *tile], shared_layout)  # nBuffers == 2; swizzled destination first

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
    a_k = _acp.load_shared_relaxed(s.index(cur), dot_operand_layout)  # LDS -> reg, THIS iter only
    acc = gl.amd.cdna4.mfma(a_op, a_k, acc)       # DOT(k): independent of s.index(nxt)
```

3 stages: 3 buffers + `wait_group(2)`. **gfx942 downgrade — sync-staged double buffer** (A5):

```python
s = gl.allocate_shared_memory(dtype, [2, *tile], shared_layout)  # e.g. SwizzledSharedLayout(..., order=[1, 0])
s.index(0).store(gl.load(ptrs(0)))
for i in range(NUM):
    cur = i % 2
    nxt = (i + 1) % 2
    gl.barrier()                                  # buffer `cur` written by every wave
    if i + 1 < NUM:
        s.index(nxt).store(gl.load(ptrs(i + 1)))
    a_k = s.index(cur).load(dot_operand_layout)
    acc = gl.amd.cdna3.mfma(a_op, a_k, acc)
    gl.barrier()                                  # nobody refills `cur` while a wave still reads it
```

### What the compiler inserts after wait_group: barriers and load_shared_relaxed

On 3.8.0 (compile-checked, gfx950 and gfx942) an async ring needs no `gl.barrier()`: each `wait_group(N)`
lowers to `llvm.amdgcn.wait.asyncmark(N)` + an `s_barrier`, and membar adds an `s_barrier` before each async copy into a buffer an earlier read may still use.

`load_shared_relaxed` (module function; `async_copy.py` docstring) removes no barrier after a `wait_group` —
it only adds alias-scope metadata. **After a hand-issued `s_waitcnt` it drops the RAW barrier, a real race**
(GPU-verified): use it only after a `wait_group` that retired this buffer's group.
Verify: `grep -n "s.barrier\|wait.asyncmark" <kernel>.llir` (the barrier's debug location is the `wait_group`
line), or `s_barrier` right after `s_waitcnt vmcnt(k)` in the `.amdgcn` loop.

**The derived `vmcnt` is LLVM's guess and can be too loose, with no diagnostic.** On 3.8.0 `wait_group(N)`
emits only `s_wait_asyncmark N` (a commit-group count, unclamped); v3.8.0's AMD backend has no
`SWaitcntOp::create` on this path (3.7.0 emitted `s_waitcnt vmcnt(min(63, numInst))` itself), so the vmcnt
constant comes from LLVM's asyncmark analysis. The upstream gfx950 Gluon attention tutorial documents the
failure: a `ds_read` issued ahead of the async copy it depends on — a correctness bug, not a slow one. On a
simple ring the derivation was exact (`wait_group(1)` with one copy per group → `vmcnt(1)`, compile-verified
3.8.0 gfx950), and in GPU race tests (gfx950 / 3.8.0; committed rings of depth 2-4 at 1 and 4 warps, ordinary
loads before or after each copy) the derived wait was exact whenever every copy was committed and the depth
was right — a too-loose wait appeared only with a missing `commit_group` or a wrong user depth. So the risk
lives in the commit/wait structure; the tutorial's case is the reason to still read the ISA. Check: in the `.amdgcn` loop, before the first `ds_read` of a buffer, the last `vmcnt(k)` must
leave at most the copies issued *after* that buffer's group outstanding. A wrong-looking `k` is not evidence
your `groups_per_stage` arithmetic is off (that arithmetic does not pick the number); the only way to name the
constant yourself on the 3.8.0 Gluon API (which has no vmcnt-level wait) is a hand `s_waitcnt` (next
section), then race-test.

### Three shapes production actually builds, and the barrier placement that differs between them

Structural observations from gfx950 / 3.8 kernels, no timing attached:

1. **Rolling FIFO, per-slot drain:** named slots (`ab0..ab3`), one commit per slot; steady state
   `wait_group(SLICES - 2)`, tail `wait_group(SLICES - 1 - k)` for slot `k` (a constexpr wait table).
2. **Asymmetric A/B depth** (`A_STAGES`, `B_STAGES`) over modular indexing, chosen from a host-side per-M
   table; unequal depths need two prologue commits.
3. **Single slot, `wait_group(0)` every iteration**, overlap bought back by wave priority around the MFMA
   block and a fence pinning the accumulator.

Barriers are compiler-placed (previous section): shape 1's hazard is read-before-fill (barrier after the
wait), shape 2's overwrite-before-read (barrier before the refill), shape 3 pays both. A ring deep enough to
separate a slot's read from its refill takes the second barrier off the critical path.

### Draining below the group: bare s_waitcnt instead of wait_group

`wait_group(N)` retires whole commit groups; draining by individual loads (`vmcnt(8)` → `(4)` → `(0)` across
one stage) needs a hand-issued `s_waitcnt` via inline asm (`../gluon/inline-asm-reference.md ## Class 4 —
synchronization`). On 3.8.0 `wait_group` emits only the asyncmark wait and LLVM picks the `vmcnt` (3.7.0
emitted `s_waitcnt vmcnt(min(63, numInst))` itself); the 3.8.0 Gluon API has no vmcnt-level wait, so a hand
`s_waitcnt` is the only way to name the constant there.

- **Inline `s_waitcnt` lacks `MemWaitOpTrait`, so membar puts no `s_barrier` after it.** It still puts one
  before a plain `.load()` of the filled buffer and **none before `load_shared_relaxed`** (compile-verified
  3.8.0) — and that relaxed read then races (previous section). Readiness for a
  relaxed read must come from a `wait_group`.
- `is_pure=False` asm is ordered by the Triton passes (generic read/write effects) but issues no wait and
  establishes no visibility; an `is_pure=True` asm whose result is unused is deleted (compile-checked 3.8.0).
- **Pacing before readiness:** a bare `s_waitcnt` caps VMEM in flight where no consumer reads yet; `wait_group`
  is the readiness form the compiler tracks (a bare `s_waitcnt` as readiness held for a plain `.load()` in the
  GPU race test but raced with `load_shared_relaxed`). For the staged-drain readiness form: put
  the first tier before an ordinary `.load()`, place every CTA barrier yourself, and verify by subtraction —
  your `s_waitcnt` appears and **no** compiler `vmcnt` wait sits between it and the consumer (a second wait
  means it bought nothing) — then race-test (`../method/benchmark-hygiene.md ## Determinism race-test (async /
  barrier / pipeline / layout changes)`).

**Epilogue:** an explicit blocked store layout for the accumulator before `buffer_store` is the usual form;
interleaving stores with the final DOT regions is the alternative to bursting them after `wait_group(0)`.

## Budget before deepening

Add a stage only if (a) the async copy passed its smoke test (`memory-path.md ### Async copy: smoke-test
before wiring (mandatory)`), (b) the independence rule holds, (c) the occupancy budget says depth, not
LDS/VGPR, is the bound — on VGPR/LDS-capped waves prefetch adds the capping resource and has regressed in the
cases recorded. Unroll: in the case measured, 2 blocks filled the MFMA→VALU `s_nop` gap and more raised
`s_nop`/iter, so a deeper unroll carries a cost to beat. Double-buffering only to drop a WAR barrier forces unroll by the buffer period only
if you also need literal indices; budget the unrolled VGPRs first (on a 1-wave kernel it usually spills).

## Intra-wave: pacing the stream the staging created

Layer 2, applied in `make_llir` (shared by both tiers, stock 3.8.0). Mechanisms live in
`instruction-scheduling.md`. Hand-rolled `s_setprio` exists beside the marker API because the marker refuses
an async wait in the middle of a stage; a ring that cannot be cut into legal stages can still take inline-asm priority
(`warp-pipeline.md ### The no-wait-inside-a-stage rule is what makes layering this over an authored ring
hard`). A **matrix+VALU** region (attention softmax) is where same-wave co-issue placement matters and where
the hand-authored path has the least control; a matrix+memory region usually gets its overlap from staging
and the phase offset.

## Inter-wave: the two-group phase offset

Layer 3 — mechanism in `warp-pipeline.md`; it layers on top of a ring. The 4-wave vs 8-wave comparison has
gone both ways: on one gfx950 fp16 GEMM family a 4-wave deep-ring build beat the 8-wave two-group build at the
K values tested (upstream measurement with AMD's tutorial tooling; the verdict reversed across toolchain pins),
while on stock Triton 3.8.0 / gfx950 an upstream ring-kernel family's 8-warp builds beat its 4-wave builds,
and on the warp-pipeline snippet kernel a 4-warp 2-CTA/CU ring came close to the warp-pipelined build. A/B it
on the kernel at hand
(`warp-pipeline.md ### The fifth question: is this better than spending the same resources on the ring?`).

## Recovering the structure, then improving on it

The parity-recovery path described here is an **authored** body. `scripts/recover_gluon.py --with-pipeline` (or
`dump_ir.sh --emit-gluon pipeline`) emits a prologue/loop/epilogue scaffold from plain's post-pipeliner
`.ttgir` with recovered `nBuffers` / `wait_group(N)` / mask shape and layouts; addressing is left as `TODO`.
The faithful layouts-only anchor serves as the attribution baseline for the improvements (deeper buffering,
operand prefetch, manual interleave; ping-pong where the dependency allows — an online-softmax recurrence
limits symmetric ping-pong), checking occupancy after each register-buffered step.

## Reproduce plain's software pipeline on the Gluon path (the parity-recovery route)

Rung 4 — mainly a diagnostic, numbers labelled `injected`; on an incumbent it would replace the authored body.
It does not share a loop with hand-written staging or prefetch (v3.8.0: one `ttg.barrier` in the loop and
`isSafeToPipeline` skips it).
Mechanism, kernel conditions, error strings, cache trap, IR tells and measured outcomes have one home:
`../gluon/pipeline/reinjection.md ## Re-injecting plain's pipeliner — the measured recipe`.

## IR / asm acceptance signals

| change | confirm |
| --- | --- |
| 2/3-stage prefetch | `wait_group(N)` present; loads for `k+1`/`k+2` issue before `DOT(k)`; `ds_write` gone (async) |
| pipeline correct | each LDS buffer retired before the next `AC` overwrites it; race-test passes |
| scheduling layered on top | MFMA interleaved with `buffer_load` / `ds_read` (`asm_loop_audit.py`, `mfma_efficiency.py`) |

If MFMA efficiency does not rise: check the knob is actually on (IR), the LDS layout is conflict-free, and no
new spills — then reclassify the bound.
