# Gluon pipeline — the stage marker, the fork-only predicate, and the gfx942 async width gate

## `gl.amd.warp_pipeline_stage` — the official marker path

Upstream, opt-in, and **independent of** `num_stages`: `WarpPipeliner.cpp` bails out unless
the loop body contains at least one border marker, so nothing happens to a loop you did not
annotate. `gluon_to_ttgir` already calls `add_warp_pipeline` on 3.7+, so this needs **no
compiler patch**.

```python
for k in range(K // BK):                       # a bare `range` is required, see below
    with gl.amd.warp_pipeline_stage("mem", priority=1):
        ...                                    # cluster 1 — LDS reads, refills, address VALU
    with gl.amd.warp_pipeline_stage("mfma", priority=0):
        ...                                    # cluster 2 — the matrix ops
```

**The memory stage takes the HIGHER priority**, and it is the one parameter that is easy to get
backwards — backwards costs the whole schedule rather than a few percent, because a compute stage
that outranks memory starves the other group's address updates. The derivation and the three
sources that agree on it are in `../../tile-programming/warp-pipeline.md ## Stage priority: memory outranks compute`.

The API-level rule that belongs here: priority is optional, and **if any stage in the loop sets
one, every stage that does not is reset to 0**; if no stage sets one, no `s_setprio` is emitted
at all. So `mem=1` with `mfma` left bare is equivalent to the pair above.

**The consequence is a configuration that looks finished and is not**, so verify the lever landed
before reading a time. Stages marked with no `priority` anywhere gives you `sched_barrier` region
boundaries and **no `s_setprio` at all** — the region cuts without the wave-priority mechanism they
were cut for. `../../hardware/bound-class-signals.md ## Lever gating laws` makes this an
efficacy-before-timing case, which fixes the order:

1. Dump the asm (`kernel_workflow/scripts/kernel_tools/dump_ir.sh`; `scripts/dump_ir.sh` is the
   pack shim) and confirm `s_setprio` appears in the hot loop, at the
   boundaries you marked. Absent means the lever never engaged, and a flat timing there answers a
   different question than the one you asked.
2. Only then time it — and time **both** assignments. The paragraph above puts the backwards case
   at *the whole schedule* rather than a few percent, and an effect that large in the wrong
   direction is exactly what one-direction testing cannot distinguish from "this lever is dead".

Cost is why it is worth the two runs: `s_setprio` is a SALU instruction and consumes no LDS, no
VGPR, no AGPR, no matrix issue slot and no L2 bytes, so unlike most of layer 4 it cannot drop you a
resource tier. The conflict to rule out first is a body that **already hand-writes `s_setprio`
through inline asm** — a form that sits on top of rings the marker path cannot be applied to at all
(`../../tile-programming/warp-pipeline.md`). Two priority sources in one loop fight, and the
hand-rolled one is the half the pass cannot see.

Constraints, from the pass itself: **≥ 2 clusters** or it errors; a barrier / `async_wait` /
`AsyncTDMWait` may sit only *between* stages, never inside one; and no `scf.for` / `scf.while`
inside a stage. It emits `s_setprio` + `sched_barrier` around each cluster in `make_llir`.

This page owns the **API surface**; what your build has is the four-version table in
`../pipeline-reference.md`. The rest of the mechanism is in
`../../tile-programming/warp-pipeline.md`, and it is not optional reading before the first
build — three things there decide whether any of this pays:

- **The applicability gates.** This buys compute utilisation only on a compute-bound loop that
  stages through LDS at 2 waves/SIMD. Each of those failing is *silent*: the marker compiles,
  `s_setprio` is emitted, and nothing overlaps.
- **The authoring rules**, which fail conversion rather than running slowly — a dynamic `range`
  is required, every op in the body must belong to a stage, and the loop covers only
  `K_ITERS - (NUM_BUFFERS - 1)` tiles so the remainder needs an epilogue outside any stage.
- **A skeleton to copy**, with the multi-buffering, the wait placement and the launch parameters
  in one piece — which this reference deliberately does not duplicate.

**Availability is the thing to check first — it is the one capability that is version-gated
rather than probe-gated: `warp_pipeline_stage` and `add_warp_pipeline` are absent on 3.6.0 and
present from 3.7.0.** The full four-version ledger for this reference (marker path, pipeliner
passes, async and shared-memory surface, loop knobs) lives in one place,
`../pipeline-reference.md ## Which spelling exists on which build — the four-version probe table`,
and is not repeated here. On 3.6 there is no marker path at all; staging by hand is what remains.

**Gate 0 comes before any of the authoring rules: `num_warps >= 8`.** Below that the warp index is
identically zero and the inter-wave phase shift silently does not exist, while the border marker's
`sched_barrier` still lands as an intra-wave scheduling wall. Both effects are real; they are not the
same effect, and a timing change from the second reported as the first is a misattribution
(`../../tile-programming/warp-pipeline.md ## Gate 0: does the launch have two wave groups at all?`).
The gfx950 runnable case for the marker (C5 in `scripts/pipeline_examples_cdna4.py`) is the one case
in that script that launches at `num_warps=8` for this reason.

## `gl.warp_predicate` — a fork-only extension, and a design boundary

Its row in the four-version table (`../pipeline-reference.md`) is not a probe result to re-check
per minor: `warp_predicate` **is absent from upstream entirely** and is added by a vendor fork (on the AMD tutorial lineage, from the
`v2.0`-era tag). It executes a body under a per-wave condition, lowering to
`s_and_saveexec_b64` + `s_cbranch_execz` — a **warp-uniform branch with no cross-wave
reduction and no barrier**.

Why it earns a row of its own rather than a footnote: it is the difference between eliding
work and multiplying by a no-op. A row whose correction factor is 1 is numerically a no-op,
but its multiply **still issues and still costs its issue slot**, so any "skip the work when
the common case holds" design needs real control flow to pay
(`../../workloads/attention.md ### Conditional rescaling (skip the per-block acc *= alpha)`). Two consequences:

- **On a clean upstream build that design is not writable at all.** Not slower — absent. When
comparing against a published kernel that uses it, the honest comparison is against what
upstream can express, and the gap is a **language** ceiling recorded against the build, not
a result about your kernel. Inline asm does not lift it — the one door takes tensors and
returns tensors, so it cannot enclose the region you wanted skipped
(`../inline-asm-reference.md ## Class 4 — synchronization`), and an ordinary `if` is
CTA-uniform by construction because every route to a 0-d condition is a whole-tile
reduction.
- **Where the branch goes is a kernel-author decision the compiler cannot make.** Control flow
is scheduled ahead of everything else in its region, so a predicate placed inside a compute
region issues *before* that region's first matrix op and the matrix pipe waits on it. Put it
in a memory region, where the same cost lands against memory latency instead
(`../../tile-programming/llir-codesign.md ## Attention: the co-execution budget`).

Probe it as a plain symbol (`probe_levers.py` / `hasattr`) and record its absence as a
**version/API ceiling** with the build identity beside it
(`../../tile-programming/compiler-contract.md ## Toolchain identity`), never as a mechanism
rejection.

## The gfx942 async-copy width gate — available, and narrow

**The gfx950 baseline this section downgrades from.** On gfx950 the same `cdna4.async_copy` entry
points lower at **128 or 32 bits per thread** (`supportsDirectToLdsLoadBitWidth`: CDNA4 = {128, 32}),
which is what the C1–C4 rings in `async-ordering.md` use; the contract every arch shares (one access
per lane, coalesced writes, swizzle within a warp, swizzled destination first) is owned by
`../memory-reference.md ## Async Copy To Shared`. This section is **the gfx942 downgrade**: what
changes, and by how much.

| | gfx950 (CDNA4) | gfx942 (CDNA3, downgrade) |
| --- | --- | --- |
| legal per-thread widths | 128 / 32 bit | **32 bit only** |
| bytes moved per instruction at the widest width | 16 B/lane | 4 B/lane — a quarter |
| destination `order` | as the contract requires | **must be `[1, 0]`** (below) |
| measured value vs sync staging | A/B it | **slower on every minor measured** (below) |
| namespace | `gl.amd.cdna4.async_copy` | the **same** `cdna4` entry points — `gl.amd.cdna3` has no `async_copy` submodule |

`cdna4.async_copy` **DOES lower on gfx942 — at 32-bit per thread, and only there.** An earlier
revision of this section claimed it did not lower at all, on the strength of a probe that used a
`BlockedLayout` whose threads did not tile the contiguous dimension. That is the layout-contract
failure `../memory-reference.md ## Async Copy To Shared` warns about, wearing the same
`unrealized_conversion_cast` costume as a real ceiling. Re-probed on gfx942, bf16, sweeping the
per-thread chunk with a clean tiling:


| bytes/thread   | gfx942                                  | why                                             |
| -------------- | --------------------------------------- | ----------------------------------------------- |
| **4 (32-bit)** | **lowers, runs, bit-exact**             | `supportsDirectToLdsLoadBitWidth`: CDNA3 = {32} |
| 8 / 16 / 32    | `failed to translate module to LLVM IR` | 64/128-bit direct-to-LDS is CDNA4-only          |


The control that settles it: the *same* 32-bit chunk fails with a ragged tiling
(`4 threads x 1 elem` over a 64-wide dim) and succeeds with a clean one (`64 x 1`). Same arch,
same width, same op — only the layout contract differs.

**A third constraint the width table cannot show: the destination shared layout's** `order` **must be**
`[1, 0]`**.** With everything else pinned at the configuration above, flipping only the `order`
turns a bit-exact lowering into `failed to translate module to LLVM IR` — and it does so both at
a trivial swizzle (`vec=1, max_phase=1`) and at a real one (`vec=4, max_phase=16`), so `order`
gates this on its own rather than through the swizzle. Satisfying the width and the tiling is
therefore not enough; a reader who checks only those two still hits the same error message.
This one only surfaces on a real kernel: an attention decode stages V in `order=[0, 1]` because
that is what plain's `amdg.in_thread_transpose` semantics require, which collides head-on. The
way out is a second shared layout for the async destination that differs *only* in `order` —
writer and reader reference the same `memdesc`, so the round trip stays bit-exact either way.

So on CDNA3 the async path is **available but narrow**: 4 B/thread moves a quarter of what CDNA4
does per instruction, which tends to make it `s_waitcnt`-bound. What is genuinely absent on CDNA3
is the *width*, not the op.

**Measured on a real kernel, narrow lost.** Converting one sync-staged KV buffer of an attention
decode to `async_copy` compiled and stayed **bit-exact on all four minors** (plus a repeated
same-input launch test for determinism, which a shared-memory synchronization change needs), and
was **slower than the sync staging it replaced on every one of them** — from roughly break-even
on the oldest minor to tens of percent on the newest. Read that trend carefully: the async arm
was flat across minors and the *sync* arm got faster, so this is the baseline improving, not
async regressing. Treat CDNA3 async copy as a correctness-preserving option to measure, not as
an upgrade — and carry both arms' **absolute** times per minor rather than their ratio, or you
will credit the async arm for the sync arm's improvement
(`../../method/benchmark-hygiene.md ## Cross-version ratios: attribute the move before claiming it`).
