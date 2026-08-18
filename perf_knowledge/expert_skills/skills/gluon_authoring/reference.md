# gluon_authoring — reference router

17 files, ~3.8 k lines, all **lazy**: load one only when [`skill.md`](skill.md) cites it, or when you
reach for the construct it documents. Two groups only — the API, and what not to write.

## First, check which generation you are on

`match.gens` claims gfx942 **and** gfx950. The mechanics transfer; several *verdicts* do not — the LDS/CU
divisor, the bank-conflict stride, the MFMA shape set, whether async copy lowers, and whether one shipped
limitation announces itself or degrades silently. All of it is one table:
[`skill.md ## Arch dispatch`](skill.md#arch-dispatch-gfx942-cdna3-vs-gfx950-cdna4). **Read it before
trusting a figure or asserting a digest across generations**, and take the values from
`hardware/hw_constants.json` below rather than from prose.

## Start above this skill, not in it

Onboarding is **not** duplicated here. If you have not written Gluon before, read GEAK's language layer
first — it is the canonical home for the concepts and for the measured GEMM ceilings:

| | |
| --- | --- |
| what Gluon is vs Triton, when to reach for it | [`languages/gluon/overview.md`](../../../languages/gluon/overview.md) |
| the abstraction: explicit layouts / pipeline stages / register budget / MFMA intrinsics, ping-pong vs interleave, MXFP4 numerics | [`languages/gluon/programming_model.md`](../../../languages/gluon/programming_model.md) |
| the near-peak GEMM recipe v0→v9 and the measured ceilings | [`languages/gluon/gemm_cookbook.md`](../../../languages/gluon/gemm_cookbook.md) |

The division of labour: **that layer teaches Gluon; this skill is the API-page detail behind it plus the
two mechanics for migrating an existing Triton kernel.** If you find yourself explaining what a layout is,
you are in the wrong file.

## The transcription runbook (`references/phases/`)

| you need | read |
| --- | --- |
| the executable form of step 1 — recover, **apply**, compile, verify, record, attribute | `phases/transcribe-runbook.md` |

Load this one **before writing the anchor**, not after it fails. Two of its stages carry decisions no
other file states: *Apply* (a body left on `AutoLayout` compiles, passes the oracle, and is several
times slower than the champion) and the classification of each `ttg.local_alloc` (staged vs pass-through).

Three corrections to that classification, all measured after the runbook was written and all owed
upstream:

1. It is a property of **the dump's `num_stages`**, not of the kernel. At the shipped depth the staging you
   are looking at is often the *pipeliner's*, which a faithful un-pipelined anchor has nothing to
   transcribe — so classify against a **`ns=1` dump**. Confirmed independently on two kernels, both of
   which read "staged" at the shipped depth and textbook "pass-through" at `ns=1`.
2. **`--verify` is blind to the choice only when the layout is `UNRECOVERABLE`.** That is the case the
   runbook's advice was written for: a layout with no Gluon constructor is excluded from the comparison, so
   dropping its buffer is invisible. Where the shared layout *is* recoverable, `verify` sees it and returns
   a hard FAIL naming the missing `swizzled_shared`. What `verify` cannot see in either case is allocation
   *size*.
3. **The performance sign is kernel-dependent.** Of three kernels authored both ways, pass-through was
   slower on two and faster on one. Judge it from `s_barrier` counts, from which limiter actually binds,
   and from whether an arch instruction was lost — see `skill.md ## Do-no-harm notes`.

## Per-arch constants (`references/hardware/`)

| you need | read |
| --- | --- |
| the machine-readable per-arch table the occupancy probe reads — LDS/CU and banking, VGPR file and the wave-step table, MFMA cadence and layout family, `ds_read_tr` / `scaled_mfma` / direct-to-LDS widths | `hardware/hw_constants.json` |

Prefer it over a figure quoted in prose, and **pass the arch rather than defaulting it**: the LDS/CU
divisor differs by 2.5× across the two generations this skill claims, and one applied to the other is a
confidently wrong occupancy verdict rather than a rounding error. `scripts/probe.py` shipped with exactly
that bug. Occupancy is capped **jointly** by LDS and by registers, so read both sides of the probe's
report and act on whichever binds — a generous LDS figure is not headroom if registers are the limit, and
the more LDS the generation gives you the more likely that is the case.

## Gluon API surface (`references/gluon/`)

| you need | read |
| --- | --- |
| where to start, what exists | `gluon/index.md` |
| blocked / MFMA / shared / linear / slice / dot-operand layouts | `gluon/layout-reference.md` |
| MFMA intrinsics, operand layouts, `kWidth` | `gluon/matrix-reference.md` |
| global access, `buffer_load`, async copy to shared | `gluon/memory-reference.md` |
| shared allocation, LDS sizing | `gluon/smem-lds-reference.md`, `gluon/shared-aot-reference.md` |
| barriers, `commit_group` / `wait_group`, `warp_pipeline_stage`, **and the measured re-injection recipe** — the two conditions, the dot-candidacy rule, per-shape and per-version numbers, the ping-pong window, why async copy is unreachable from plain on gfx942 | `gluon/pipeline-reference.md` |
| reductions, scans, elementwise atoms | `gluon/atoms-reference.md` |
| imports, launch, AOT | `gluon/imports-and-launching.md` |
| a runnable skeleton to copy | `gluon/gfx950-minimal-examples.md` |

## What not to write

| you need | read |
| --- | --- |
| constructs that compile and are then silently wrong or slow | `gluon-negative-patterns.md` |
| ROCm / driver / toolchain constraints and broken paths | `platform-known-issues.md` |
| the condensed list | `skill.md ## Knobs & pitfalls` |

## How much a direction may cost

| you need | read |
| --- | --- |
| which gates apply in your entry state, **and the depth contract** — why a layout-coupled direction cannot be run as a one-lever round, and the three rules that keep it reachable | `entry-modes.md` |

Read the depth contract **before writing the brief**, not after the deadline. This skill owns no
round loop, and the callers that do own one measure progress in *directions closed* — so the default
reading is one experiment per direction, under which LDS swizzle/padding choice and LDS dedup (the
two things plain Triton has no syntax for, and the only reason to be here) are unreachable, because
both touch several layouts at once. One in-place run diagnosed its residual correctly, named the
coupled fix, declined it as "not a one-lever round", closed 22 directions and moved nothing
structural, against an `expects.isolated_speedup_min` of 1.10. The tell is in the file: **a
direction that invalidates more than one layout constructor is a rewrite, not a lever.**

## Layout and overlap — the two mechanics, by entry state

| you need | read / run |
| --- | --- |
| TTGIR → Gluon layout recovery map (incl. the manual `convert_layout` step) | `tile-programming/layout-recipes.md` — **port only**; there is nothing to recover from when the source is already Gluon |
| **porting: does this kernel owe a pipeline at all** | `skill.md ## Procedure` step 2a — the `plain@ns=1` control |
| **already Gluon: does this loop overlap at all** | `skill.md ## Procedure` step 2d — read the tell **that matches the loop's shape**; `ttg.memdesc_index` is a false negative on a dot-free loop |
| adding overlap where the loop **has a dot** — the three conditions and the measured recipe | `gluon/pipeline-reference.md` (re-injection), `scripts/USAGE.md` for `gluon_swp.py` |
| adding overlap where the loop is **dot-free**, so the pipeliner has nothing to anchor on | `gluon/pipeline-reference.md ## Authored overlap` — on CDNA3 that is sync staging plus the `warp_pipeline_stage` hint, **not** async copy |
| what is unavailable on this arch despite importing | `skill.md ## Knobs & pitfalls`, and `gluon/pipeline-reference.md`'s per-arch A/B tables |
| the pass list, proof-it-landed signals, and the hand-built cross-iteration double buffer | `tile-programming/pipeline.md` |
| the tools | `scripts/USAGE.md` |

## Not here, on purpose

**No process.** The round loop, hardware budget / roofline, profiling (rocprof / ATT / PMC), the
lever-card catalogue, bound-class signals, the escalation gate, orchestration, experiment records,
benchmark hygiene, the transcription protocol page and the workload strategy pages were not vendored.
GEAK's `kernel_workflow` / `e2e_workflow` own deciding what to try and how much to spend — the only thing
this skill fixes is the port itself (`skill.md ## Procedure` steps 1–3), and only because those moves are
mechanical. Step 4 ranks what the port newly made expressible; it still does not decide.

Retained files still cite those dropped paths in 64 places. **Those pointers are dead here by design.**
Do not hunt for the files, and do not read a citation as an instruction to rebuild the regime. For the
hardware facts among them — instruction availability per gen, LDS banking, VGPR/AGPR budget, occupancy —
use GEAK's own [`perf_knowledge/hardware/`](../../../hardware/) (`cdna3_mi300/`, `cdna4_mi350/`,
`shared/`) and [`perf_knowledge/languages/gluon/`](../../../languages/gluon/).
