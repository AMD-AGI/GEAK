# Gluon — pipeline (API router)

Backbone layer 4, **API surface only**. This page answers two questions and no others: *which
spelling exists on which build*, and *which chapter owns the mechanism you are holding*. It does
**not** rank the mechanisms — the order to reach for them (hand-written pipeline first, re-injection
last) is defined once, in `../tile-programming/pipeline.md`, and is not restated here.

Gluon has **no** CuTeDSL `PipelineTmaAsync` / mbarrier library, and it ships no auto-pipeliner:
`gluon_to_ttgir` never calls `add_schedule_loops` / `add_pipeline` on **any** upstream version
(checked 3.6.0 / 3.7.0 / 3.7.1 / 3.8.0). So the overlap on this path is **authored**. Plain's passes
can be re-injected over Gluon TTGIR, but that route is a diagnostic below the parity gate or a last
resort, its numbers are labelled `injected`, and it is never applied to an incumbent Gluon kernel
(`pipeline/reinjection.md`, whose measured outcomes are one versioned table there).

**How to read the version claims in these chapters.** Each mechanism is written in its **3.8.0
spelling** and for **gfx950 (CDNA4) first**; where an older minor or gfx942 (CDNA3) differs, the
difference is a downgrade note attached to that mechanism — there is no separate older-version
recipe to find. The table below is the authority for *what exists where*: a downgrade note tells you
what to write instead, the table tells you whether you have to. Where either disagrees with your
build, re-probe rather than picking one (`../../scripts/probe_levers.py --all`).

## Chapters

| Read this when | Chapter |
| --- | --- |
| you are on gfx950 and the loop stages through LDS asynchronously (`async_copy` + `commit_group` / `wait_group`), or you have to decide a drain depth and where the barrier goes | `pipeline/async-ordering.md` |
| you are building the overlap by hand: register-level prefetch, which mechanism exists on which architecture, the A / B / S classes, and the gfx942 downgrade worked set | `pipeline/authored-overlap.md` |
| you want the stage marker (`warp_pipeline_stage`), the fork-only `warp_predicate`, or the gfx942 async-copy width gate | `pipeline/marker-and-version-gates.md` |
| you are picking a per-target path (gfx950 / gfx1250 / gfx942), or reaching for a loop knob (`num_stages`, `loop_unroll_factor`) | `pipeline/loop-knobs-and-targets.md` |
| **diagnostic / last resort only:** you are below the parity gate and want to size the `lost_pipeline` debt, or a hand-written pipeline has demonstrably failed to reach parity | `pipeline/reinjection.md` |

**`num_stages` is dead on the Gluon path in 3.8.0** (and on every older minor) — no pass consumes
it — so writing it, omitting it, or finding it absent in someone else's kernel is not evidence about
pipelining in either direction. It survives only as a budget parameter and a champion-record field
(`pipeline/loop-knobs-and-targets.md ## Roll the loop (cut i-cache pressure)`).

## Which spelling exists on which build — the four-version probe table

The version ledger for this whole reference: the marker path, the pipeliner passes, the async and
shared-memory surface, and the loop knobs are each gated differently.

> **Read the rows by what kind of claim they are, because they do not all have the same evidence.**
> A row about a `gl.*` symbol is a statement about the Python surface and is checkable from a source
> tag directly. A row about **which passes `gluon_to_ttgir` calls**, or about a C++ pass being
> present in `libtriton`, lives in `third_party/amd/` and in the compiled library — so it comes from
> a build or a container probe, not from a Python-only source extraction. The 3.8.0 column is
> confirmed against a complete tree; the three older columns are container probes. That distinction
> matters the moment a row disagrees with your build: re-probe rather than re-reading the table
> (`../../scripts/probe_levers.py --all` reports the knob half of this as
> `version_disjoint_knobs`).

| | 3.6.0 | 3.7.0 | 3.7.1 | 3.8.0 | owner |
| --- | --- | --- | --- | --- | --- |
| `gl.amd.cdna4.async_copy` (all 5 entry points: `global_load_to_shared`, `buffer_load_to_shared`, `commit_group`, `wait_group`, `load_shared_relaxed`) | ✓ | ✓ | ✓ | ✓ | `pipeline/async-ordering.md` |
| `cdna4.compute_efficient_padded_shared_layout` | **absent** | **absent** | **absent** | ✓ | `layout-reference.md ## Let 3.8 Compute The Padded Layout For You (CDNA4)` |
| `gl.barrier` (3.6.0 spells it `gl.thread_barrier`; no alias either way) | `thread_barrier` | ✓ | ✓ | ✓ | `pipeline/authored-overlap.md` (the `getattr` shim) |
| `gl.amd.warp_pipeline_stage` | **absent** | ✓ | ✓ | ✓ | `pipeline/marker-and-version-gates.md` |
| `add_warp_pipeline` in `libtriton` | **absent** | ✓ | ✓ | ✓ | same |
| `add_schedule_loops` / `add_pipeline` in `libtriton` | ✓ | ✓ | ✓ | ✓ | `pipeline/reinjection.md` |
| … called from `gluon_to_ttgir` | ✗ | ✗ | ✗ | ✗ | same |
| `num_stages` read by any pass on the Gluon path | ✗ | ✗ | ✗ | ✗ | `pipeline/loop-knobs-and-targets.md` |
| `passes.ttir.add_loop_unroll` from `gluon_to_ttgir` (makes `loop_unroll_factor` live) | ✗ | ✗ | ✗ | **✓** | same |
| shared-descriptor `.gather` / `.scatter` | **absent** | ✓ | ✓ | ✓ | `smem-lds-reference.md` |
| shared-descriptor `atomic_scatter_*` | **absent** | **absent** | **absent** | ✓ | `smem-lds-reference.md` |
| `._reinterpret` with defaulted `dtype` / `shape` / `layout` (all three explicit works everywhere) | **absent** | **absent** | **absent** | ✓ | `smem-lds-reference.md` |
| `gl.dot_fma` with a rank-3 (batched) accumulator | **absent** | **absent** | **absent** | ✓ | `matrix-reference.md` |
| `llvm_fn_attrs` compile option | **absent** | **absent** | **absent** | ✓ | `../tile-programming/llvm-fn-attrs.md` |
| `sched_barrier` / `sched_group_barrier` / `set_prio` under `gl.amd.cdna3` / `.cdna4` | **absent** | **absent** | **absent** | **absent** | `pipeline/authored-overlap.md` (B2) |
| `gl.warp_predicate` (fork-only) | **absent** | **absent** | **absent** | **absent** | `pipeline/marker-and-version-gates.md` |

Two further version facts that are not symbols, and that this table's rows are often confused with:

- **3.8.0 ships the co-execution scheduler strategy and turns it on by default** for the arch when
  `num_warps <= 4` (`TRITON_HIP_USE_COEXEC_SCHEDULER` overrides the gate either way). Check whether it
  is already active before concluding a matrix-plus-VALU interleave needs anything else
  (`../tile-programming/non-upstream-reserve.md ## 0. Read this first: one capability here is upstream after all`).
- **Fork-only environment variables are not rows here** — they name nothing on a stock build. The
  list and the upstream route for each is `../tile-programming/non-upstream-reserve.md`.

Three rows are read wrongly more often than the others:

- **The `add_loop_unroll` row is what makes `loop_unroll_factor` a real knob, and only on 3.8.0**
  (`pipeline/loop-knobs-and-targets.md ## Roll the loop (cut i-cache pressure)`). It is the one line
  of this table that moved between minors for a reason other than a new symbol appearing.
- `compute_efficient_padded_shared_layout` **is arch-scoped AND version-scoped, and the two are easy
  to conflate.** It is gfx950-only (it asserts an `AMDMFMALayout` of `version == 4`); it is *also*
  absent from `cdna4.__all__` before 3.8.0, so on a 3.7.1 build the symbol does not exist even on the
  right architecture. A `hasattr` probe that fails there has found a version gate, not an arch one.
- **`.gather` / `.scatter` on the shared-memory descriptor arrived in 3.7.0**, which is a different
  gate from the async-copy row — the async entry points have been complete since 3.6.0. (The async
  row is a symbol row: on gfx942 the same `cdna4.async_copy` entry points lower only at 32 bits per
  thread, `pipeline/marker-and-version-gates.md ## The gfx942 async-copy width gate — available, and
  narrow`.)

On 3.6 there is no marker path at all; hand-written staging (register prefetch, a sync or async LDS
ring) is the pipelining available, with re-injection still only a diagnostic / last resort.

## Section map

Every heading below is a **stub**, kept at this address because other pages cite it by name. Each
carries what the section is for and a pointer that is fully qualified — **file and heading** — so
the second hop is mechanical: `context acquire` that file, query that exact heading. They are all
siblings on purpose, so a bounded query on one returns that one and not the neighbourhood. They are
listed gfx950-first, then the downgrades, then the diagnostic / last-resort route.

## The async forms — gfx950 only, and the shape differs from A5

You are on gfx950 and the loop stages async. C1–C4, and the falsifiable signature of a real async
copy.

→ **`pipeline/async-ordering.md ## The async forms — gfx950 only, and the shape differs from A5`**

## Getting `wait_group` right

You have to decide a drain depth. Group semantics, `load_shared_relaxed`, and the rolling drain.

→ **`pipeline/async-ordering.md ## Getting wait_group right`**

## Ordering the fill against the read — class S

You are deciding where the barrier goes in an authored ring. Two calls, two questions, and why a
depth change is a correctness change.

→ **`pipeline/async-ordering.md ## Ordering the fill against the read — class S`**

## gfx950 CDNA4 path

You are writing the staged loop on CDNA4. `async_copy` is a module; the five entry points.

→ **`pipeline/loop-knobs-and-targets.md ## gfx950 CDNA4 path`**

## Authored overlap (no compiler patch)

You are building the overlap yourself on an upstream build. Register-level prefetch, the
upstream-only path, why it does not compose with re-injection, and the direct-to-LDS width rule
whose failure text is indistinguishable from an absent op.

→ **`pipeline/authored-overlap.md ## Authored overlap (no compiler patch)`**

## What is actually on offer, and which architecture has it

You need to know which mechanism exists on your target before writing anything. The A (data
movement) / B (hints) / S (ordering) tables.

→ **`pipeline/authored-overlap.md ### What is actually on offer, and which architecture has it`**

## `gl.amd.warp_pipeline_stage` — the official marker path

You want the stage marker. API surface, Gate 0 (`num_warps >= 8`), the priority rule, and the
efficacy-before-timing order.

→ **`pipeline/marker-and-version-gates.md ## gl.amd.warp_pipeline_stage — the official marker path`**

## `gl.warp_predicate` — a fork-only extension, and a design boundary

You found the spelling somewhere and want to know whether you can write it.

→ **`pipeline/marker-and-version-gates.md ## gl.warp_predicate — a fork-only extension, and a design boundary`**

## Roll the loop (cut i-cache pressure)

You are reaching for a loop knob (`num_stages`, `loop_unroll_factor`). Both knobs, which pass reads
them, and **`loop_unroll_factor` in both directions** — it is a keyword on the loop construct and
not a launch option, and the value reached for most often is the one that *suppresses* unrolling.

→ **`pipeline/loop-knobs-and-targets.md ## Roll the loop (cut i-cache pressure)`**

## Where the factor goes: on the loop, never on the launch

You are about to pass the factor as a launch option.

→ **`pipeline/loop-knobs-and-targets.md ### Where the factor goes: on the loop, never on the launch`**

## `=1` is documentation, not a muzzle — and the amplify case is narrower than it looks

You are unrolling and have not considered the other direction.

→ **`pipeline/loop-knobs-and-targets.md ### =1 is documentation, not a muzzle — and the amplify case is narrower than it looks`**

## Auditing a tree for this knob — do not grep for `tl.range`

You are counting how often a corpus reaches for the unroll factor.

→ **`pipeline/loop-knobs-and-targets.md ### Auditing a tree for this knob — do not grep for tl.range`**

## The downgrade is a structural edit, not a different keyword

You are moving a staged loop down a minor version and expect a keyword swap.

→ **`pipeline/loop-knobs-and-targets.md ### The downgrade is a structural edit, not a different keyword`**

## Footguns

You are about to ship the loop.

→ **`pipeline/loop-knobs-and-targets.md ## Footguns`**

## gfx942 downgrade

You are writing the staged loop on CDNA3 and the CDNA4 form does not exist at the width you wanted.

→ **`pipeline/loop-knobs-and-targets.md ## gfx942 downgrade`**

## The gfx942 async-copy width gate — available, and narrow

You are asking whether gfx942 has an async copy at all. 32 bits per thread, clean tiling,
`order=[1, 0]`.

→ **`pipeline/marker-and-version-gates.md ## The gfx942 async-copy width gate — available, and narrow`**

## Worked examples — every one compiled and run on gfx942

You want the downgrade code, not the table. A5, A2, A4, B1 — the CDNA3 set, labelled as downgrade
examples; the gfx950 cases are under `## The async forms — gfx950 only, and the shape differs from A5`.

→ **`pipeline/authored-overlap.md ### Worked examples — every one compiled and run on gfx942`**

## gfx1250 path

You are writing the staged loop on gfx1250.

→ **`pipeline/loop-knobs-and-targets.md ## gfx1250 path`**

## CDNA5 (gfx1250) is a different model — do not port the CDNA3/4 shape

You are carrying a CDNA3/4 async ring forward a generation.

→ **`pipeline/async-ordering.md ## CDNA5 (gfx1250) is a different model — do not port the CDNA3/4 shape`**

## RDNA3 / RDNA4 have no async surface at all

You are on RDNA and looking for the async path.

→ **`pipeline/async-ordering.md ## RDNA3 / RDNA4 have no async surface at all`**

## vs CuTeDSL (concept filter)

You are porting a concept from the NVIDIA side and need to know what has no counterpart here.

→ **`pipeline/loop-knobs-and-targets.md ## vs CuTeDSL (concept filter)`**

## Anchors

You need the source anchors behind the per-target claims.

→ **`pipeline/loop-knobs-and-targets.md ## Anchors`**

## Re-injecting plain's pipeliner — the measured recipe

**Diagnostic / last resort.** You are below the parity gate and want to size the `lost_pipeline`
debt, or hand-written overlap has failed to reach parity. The two conditions, the 2x2, the
`gluon_swp` route, and the one versioned table of measured outcomes. Never on an incumbent; numbers
are `injected`.

→ **`pipeline/reinjection.md ## Re-injecting plain's pipeliner — the measured recipe`**

## On a dot kernel: un-write the staging

You are re-injecting on a loop you already staged by hand. Three arms, and why the middle one is
the trap.

→ **`pipeline/reinjection.md ### On a dot kernel: un-write the staging`**

## And on attention — two dots chained through a softmax

The body is two dots chained through a softmax, and you want to know whether re-injection reaches
it. Also why a minimal body does not settle it.

→ **`pipeline/reinjection.md ### And on attention — two dots chained through a softmax`**

## What to expect, and what to measure yourself

You have re-injected and need to know what counts as it working. The `plain@ns=1` control, the
versioned outcome table, and depth as a non-monotone.

→ **`pipeline/reinjection.md ### What to expect, and what to measure yourself`**

## What this does not do

You are about to describe re-injection in a report. Not upstream; not a win; not
`add_block_pingpong` on hand-authored staging; not the marker path.

→ **`pipeline/reinjection.md ### What this does not do`**
