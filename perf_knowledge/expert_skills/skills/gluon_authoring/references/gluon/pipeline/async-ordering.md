# Gluon pipeline — the async forms, the drain, and the ordering class

## The async forms — gfx950 only, and the shape differs from A5

**These are the primary worked examples for the LDS ring on the main target, gfx950.** The four
cases in `authored-overlap.md` are the gfx942 downgrade set, and none of them is an async copy: A5
stages through registers, which is the only thing gfx942 can do at a useful width (it has async
copy only at 32 bits per thread — `marker-and-version-gates.md ## The gfx942 async-copy width gate —
available, and narrow`).

**The precondition is arithmetic, not stylistic: a ring needs a tile large enough and a staged block
reused enough to hide the transfer.** If a staged block feeds only one or two tiles of matrix work,
there is nothing to hide the copy behind and the ring is cost plus a correctness surface. Raising the
reuse of the staged block is the enabling edit; the ring is downstream of it
(`../../workloads/moe.md ### Two reuse axes, and this page used to carry only one` is the worked
instance). Where the ring sits among the hand-written forms is `../../tile-programming/pipeline.md`.

**The async forms below are a different loop shape, not A5 with a different call in it** — the
barrier moves, the wait replaces one of the two barriers, and the ring gains a wait depth that
has to match the buffer count. They are runnable as `scripts/pipeline_examples_cdna4.py`, which
keeps A5 as its control; the census columns there add the direct-to-LDS counters, because
**the falsifiable signature of a real async copy is that the staging `ds_write` disappears** —
a sync loop cannot move data global→LDS without the register round trip. A run where every case
reports `ds_write > 0` has measured a sync fallback, whatever the source says.

All four share the preamble, and `BLK` here covers the `[32, 32]` tile **exactly** — that is the
width contract from the blockquote in `authored-overlap.md`, not a style choice:

```python
from triton.experimental.gluon.language.amd.cdna4 import async_copy as _acp

BLK: gl.constexpr = gl.BlockedLayout([1, 4], [8, 8], [4, 1], [1, 0], [])   # [1*8*4, 4*8*1] = [32, 32]
SH: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])          # order=[1,0] is required
```

**The `SwizzledSharedLayout(1, 1, 1)` in that preamble is load-bearing, not incidental**, for two
reasons owned by `../memory-reference.md`: the op's three docstring conditions (per-thread width
128 or 32 bits, coalesced writes, swizzle only within a warp) all follow from the destination LDS
address being warp-uniform (`## Async Copy To Shared`), and a padded destination has been measured
failing translation — or, spliced into lowering, running and returning wrong data — where this
swizzled one lowers and verifies, while `max_phase = 1` is the one swizzle setting that cannot hit
the silent-wrong-data condition (`### Shared-layout family + transpose-on-read (layout dependency)`).
So **do not substitute a padded destination, or a `max_phase > 1` swizzle, into these skeletons as a
first edit**; get the ring running as written, then make each change a separate step with its own
copy-back check. **A compile is not a check on this path.**

**C1 — async double buffer via `global_load_to_shared`.** At this skeleton's 128-bit per-thread
width (four fp32) CDNA3 cannot express it: on gfx942 this op fails the pass manager (the 32-bit
downgrade is the width gate in `marker-and-version-gates.md`).

```python
@gluon.jit
def c1_async_double_buffer(out, inp, M: gl.constexpr, N: gl.constexpr, ITERS: gl.constexpr):
    s = gl.allocate_shared_memory(gl.float32, [2, M, N], SH)
    o = (gl.arange(0, M, layout=gl.SliceLayout(1, BLK))[:, None] * N
         + gl.arange(0, N, layout=gl.SliceLayout(0, BLK))[None, :])
    acc = gl.full([M, N], 0.0, gl.float32, layout=BLK)
    _acp.global_load_to_shared(s.index(0), inp + o)                  # prologue: tile 0
    _acp.commit_group()
    for i in range(ITERS):
        cur = i % 2
        nxt = (i + 1) % 2
        if i + 1 < ITERS:
            _acp.global_load_to_shared(s.index(nxt), inp + (i + 1) * M * N + o)
        _acp.commit_group()
        _acp.wait_group(1)            # let the i+1 copy stay in flight
        _barrier()
        acc += s.index(cur).load(BLK)
    gl.store(out + o, acc)
```

Read the loop against A5 and the differences are the whole lesson: **one** barrier instead of
two (the `wait_group` does the other half of A5's job), the `commit_group` issued
unconditionally even on the last iteration where no copy was enqueued, and no `.store()`
anywhere — the copy writes LDS directly.

**C2 — same depth, `buffer_load_to_shared` instead.** A separate lowering path, not a spelling:
at this width on gfx942 this one fails LLVM translation rather than the pass manager. It is the
default of the two entry points (`### Getting wait_group right`, second bullet). Scalar base plus an
int32 offset tensor, so the offsets move where the pointer arithmetic was:

```python
    _acp.buffer_load_to_shared(s.index(0), inp, o)                   # prologue
    ...
        _acp.buffer_load_to_shared(s.index(nxt), inp, o + (i + 1) * M * N)
```

**C3 — C1 plus `load_shared_relaxed`, which drops the redundant wait before the LDS read.**
One line changes, and it is sound only because the `wait_group(1)` above it already retired the
group that filled `s.index(cur)` (### Getting `wait_group` right):

```python
        acc += _acp.load_shared_relaxed(s.index(cur), BLK)           # not a method on the descriptor
```

**C4 — three buffers, two copies in flight.** Depth beyond 2 is where gfx950's LDS capacity
matters (2.5× gfx942's — `../smem-lds-reference.md ## LDS is a second, independent occupancy
limiter`); on the gfx942 downgrade, deep multi-buffering of real tiles runs out of LDS before it
runs out of latency to hide. Note the prologue issues **two** groups with a
`commit_group` after each, and the wait depth rises with them:

```python
    s = gl.allocate_shared_memory(gl.float32, [3, M, N], SH)
    _acp.global_load_to_shared(s.index(0), inp + o)                  # tile 0
    _acp.commit_group()
    _acp.global_load_to_shared(s.index(1), inp + M * N + o)          # tile 1
    _acp.commit_group()
    for i in range(ITERS):
        if i + 2 < ITERS:
            _acp.global_load_to_shared(s.index((i + 2) % 3), inp + (i + 2) * M * N + o)
        _acp.commit_group()
        _acp.wait_group(2)            # two groups may stay outstanding
        _barrier()
        acc += s.index(i % 3).load(BLK)
```

> **Each iteration must read a DIFFERENT tile or the numerics cannot catch a broken ring.**
> The input above is `[ITERS, M, N]` and iteration `i` consumes tile `i`, so the reference is the
> sum over tiles rather than one tile times `ITERS`. Feed every stage the same tile — which an
> earlier version of these examples did — and reading the wrong stage, or overwriting a buffer
> before it is consumed, both still sum to the right answer. Multi-buffering is the only thing
> these cases exist to demonstrate, so the data has to be able to falsify one.

## Getting `wait_group` right

Copy the semantics from the API, do not reason from the name:

- `wait_group(num_outstanding)` blocks until the number of outstanding commit groups is
**less than or equal to** `num_outstanding`. **Uncommitted async operations are waited on
even when** `num_outstanding` **is 0**, so a missing `commit_group` does not merely delay a
wait — it silently converts an async copy into a blocking one.
**Draw the other consequence of that `≤` before you write a literal: `wait_group(n)` is a no-op
whenever fewer than `n` groups are outstanding.** It is easy to read `≤` as "waits until exactly
`n`", and the difference only shows up where fewer groups exist — the prologue and, fatally, the
tail. A literal depth at the last steps of a ring therefore waits for nothing at all, and the
matrix core reads LDS the copy has not filled yet.
- **`buffer_load_to_shared` is the default of the two, and the API says so** — its own docstring
directs you to "prefer to use `buffer_load_to_shared` when possible for better performance". The
trade is not symmetric: the buffer form takes a scalar base plus a 32-bit offset tensor and gets
**hardware out-of-bounds masking**; the global form takes a pointer tensor, and the *only* thing it
buys is the **64-bit indexing range**, at the documented cost of higher register pressure and no
hardware masking. So reach for the global form when the addressing genuinely needs 64 bits, not by
default. (That is the API's own statement of preference, not a measurement made here — if you swap
the two on a real kernel, the swap is a change to measure like any other.)
- `load_shared_relaxed` is **not** a faster `.load()`, and it is not a different load instruction.
It is `shared_load` with one attribute set on the result — `ttg.amdg.syncedViaAsyncWait` — which
*asserts* that a `wait_group` already retired the group that filled this buffer, so the backend may
drop the wait it would otherwise insert ahead of the read. It is sound only while that assertion is
true; used without the pairing it is a race that still compiles. The downstream effects and the
fallback are owned by
`../../tile-programming/pipeline.md ### Vetted double-buffer skeleton (copy, then specialize)`.
- **`num_outstanding` counts groups, not copies.** The finest drain this API can express is one
`commit_group()`'s worth, so releasing after a specific number of *individual* loads — `vmcnt(8)`,
then `vmcnt(4)`, then `vmcnt(0)` inside one stage — has no group boundary to name. Below that floor
the only route is a hand-issued `s_waitcnt` through inline asm, and it is a **pacing** lever rather
than a readiness one: it also costs `load_shared_relaxed` its soundness, because the pass that
proves the fill retired cannot see inside an asm string. Confirm the drain you want really has no
group boundary before going there. The trade is owned by
`../../tile-programming/pipeline.md ### Draining below the group: bare s_waitcnt instead of wait_group`;
the asm-block rules by `../inline-asm-reference.md ## Class 4 — synchronization`.
- **`num_outstanding` is a value you compute, and the skeletons showing a literal understate
that.** A constant depth is the steady-state answer; the loop also has a prologue where fewer
groups exist yet and an epilogue where fewer remain, and those are what the literal cannot say.
The form that covers all three at once is a **rolling drain** — the depth expressed against how
much of the loop is left rather than against the buffer count. **Read what follows as scoped to a
body where that expression folds at trace time** (a statically unrolled loop, the usual case here);
where the trip count is a runtime value the expression does not fold, `N` has to be a compile-time
property, and the branch is the mechanism rather than the thing to remove — see
`../../tile-programming/pipeline.md ### Hand-built buffering rules (correctness + scheduling footguns)`,
which is written for that case and therefore reaches the opposite default:

  ```python
  # steady state asks for DEPTH-1; the tail asks for less, automatically
  wait_group(min(DEPTH - 1, STEPS - step - 1))
  ```

  **What this buys first is correctness at the tail, not tidiness.** The literal is not merely
  less elegant — combined with the no-op rule above it is a silent wrong answer. Measured: a
  3-buffer ring written with a literal `wait_group(2)` in every step, with only the *issue* side
  guarded at the tail, compiled with no diagnostic, ran with no hang, measured **4.8 % faster** (of
  course it did — it had stopped waiting), and was **bit-exact at two shapes and wrong at a third**
  (`rel_norm 6.58e-04`). A tail race is a race, so it does not fire at every shape, and a
  tolerance-based correctness gate very likely passes it. The closed form removes the failure
  because the expression already collapses to `0` on the last step.

  The tidiness is real but secondary: the
  `if step + 1 < STEPS: wait_group(DEPTH - 1) else: wait_group(0)` branch that a literal depth
  forces into the body **stops existing** — one fewer branch in the hot loop, and one fewer place
  for the drain and the buffer index to disagree.

  **The detection rule that follows: gate every async-ring change on bit-exactness against the
  pre-change kernel at *every* shape you run**, not on a tolerance and not on one shape. That is
  the only check that caught this one.

  Related shapes, all the same idea of naming the depth against a structural quantity instead of a
  number: `STAGES - 2` for a fixed ring, the same minus a tail term where the last stage is
  shorter, one per output panel where a stage fills several, one per slice where a buffer is split,
  and `PREFETCH - 1` where the lead distance rather than the ring depth is what you are draining
  to. Pick the quantity that changes when the structure changes, so the drain follows a retune
  instead of being re-derived after it.

  **Two rungs above the closed form live on the other page, not here**, because they are choices
  about the ring rather than about the expression: a **per-M host lookup table** that selects the
  depth instead of searching for it, and an **asymmetric A/B ring** whose two operands carry
  independent depths so the tail wait tightens against the loop counter rather than a literal. Both
  are shapes production builds, and neither is reachable by generalizing the formula above —
  `../../tile-programming/pipeline.md ### Three shapes production actually builds, and the barrier
  placement that differs between them`. Go there before deciding this expression is the whole
  answer.

  Two cautions. The expression has to **fold where the call sites are** — in a statically unrolled
  body it is ordinary trace-time arithmetic, which is the usual case; a drain depth that is only
  known at runtime is a different claim about the API and worth confirming on your build before
  relying on it. And an expression is not exempt from the rule above it: it still counts **groups**,
  so every term in it has to be a group count, not a copy count.

> **The trap that is not in any rule list: do not interleave async copies with ordinary
> loads/stores in the same loop.** Both entry points document that an async copy still
> completes **in order** with `ttgl.load`/`store` and `buffer_load`/`store`, so a stray
> ordinary load in the body serialises the copies you built the pipeline for. This is the
> easiest way to author a pipeline that measures like no pipeline at all.

## Ordering the fill against the read — class S

This section is a router, not a second account: the placement rule is owned by
`../../tile-programming/pipeline.md ### Three shapes production actually builds, and the barrier
placement that differs between them`, which names the two hazards and the two placements. What
belongs *here* is the reason you have to make that decision explicitly at all on this path.

**Two calls, two questions, and neither one implies the other.** This is the whole of why the
decision is explicit here. Do not read the second as insurance for the first:

- `wait_group` retires async groups and **provides no CTA synchronization** — that is explicit in
  the op's own contract, and the owner page above states it as the reason the barrier is a separate
  decision.
- `gl.barrier()` is a rendezvous that also carries LDS-write visibility, so it is the half that
  makes another wave's fill readable. It is **not** merely an execution fence: the barrier taxonomy
  in `../../tile-programming/warp-pipeline.md ## Barrier taxonomy` distinguishes the bare
  `s_barrier` (no memory ordering) from the local-scope `ttg.barrier` that `gl.barrier()` lowers to,
  and only the second is what these skeletons rely on. That distinction is also why a barrier can
  cost more than a rendezvous: dropped at the head of a compute cluster it pulls a wait in with it.

The consequence for authoring is narrow and worth stating plainly: **a depth change is a correctness
change.** Move the drain, merge the branches, or deepen the ring, and both the wait argument and the
barrier's placement have to be re-derived against the two hazards — the owner page's rule, "if you
cannot say which hazard your barrier is guarding, that is the thing to resolve before tuning the
depth", is the gate.

**S2 is not a third mechanism; it is an assertion that S1 already ran.** What the attribute is and
what it buys downstream are covered once, under `### Getting wait_group right` and its owner page.
The part that belongs *here* is the precondition encoded in the attribute's own name — the data was
synced **via an async wait** — and the consequence: S2 is a claim about S1, so on an A5 ring, which
has no S1 to point at, the claim has no referent. Whether it is then inert or harmful has not been
probed on this path; do not write it there and find out.

**Check both halves in the asm before you check them in a number.** `s_waitcnt` for S1 and
`s_barrier` for S4, counted against the ring you intended
(`../../method/triage.md ## gfx950 / Gluon fill`). A count that is right for the ring you *wrote* and
wrong for the ring you *meant* is the case these two greps exist to catch, and neither the numerics
nor the clock is guaranteed to catch it at the shape you happen to run.

## CDNA5 (gfx1250) is a different model — do not port the CDNA3/4 shape

Only `commit_group` / `wait_group` carry over. The copy entries are renamed and re-shaped
(`global_to_shared`, `shared_to_global`, `mbarrier_arrive`), there is an `mbarrier` object
model and a `cluster` scope, and the descriptor path is a separate `tdm` module
(`make_tensor_descriptor`, `update_tensor_descriptor`, `async_load` / `async_store` /
`async_gather` / `async_scatter`, `async_wait`, `prefetch`). The matrix op is `wmma`, not `mfma`.
Treat it as its own target.

## RDNA3 / RDNA4 have no async surface at all

`gl.amd.rdna3` and `gl.amd.rdna4` expose exactly one thing: `wmma`. No buffer ops, no async
copy, no TDM. Authored overlap there is A5 only — core `gl.allocate_shared_memory` staging with
an explicit barrier, hand LDS ping-pong (`../rdna-wmma-reference.md ## Pipeline`).

**Re-injection has not been tested on CDNA5 or RDNA** (nor measured on gfx950 in this pack — the
versioned table in `reinjection.md ### What to expect, and what to measure yourself` is gfx942-only);
do not extrapolate it there.

LLVM-level scheduling (the backend scheduler, steerable per compile via `llvm_fn_attrs` /
`amdgpu-sched-strategy`, plus the stock `coexec` strategy for matrix-plus-VALU regions) interleaves
the hot loop — steer +
IR-verify (`../../tile-programming/compiler-contract.md`).
