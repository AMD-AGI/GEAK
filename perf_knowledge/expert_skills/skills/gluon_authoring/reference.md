# gluon_authoring — reference router

14 files, ~2.4 k lines, all **lazy**: load one only when [`skill.md`](skill.md) cites it, or when you
reach for the construct it documents. Two groups only — the API, and what not to write.

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

## The two port mechanics

| you need | read / run |
| --- | --- |
| TTGIR → Gluon layout recovery map (incl. the manual `convert_layout` step) | `tile-programming/layout-recipes.md` |
| **whether this kernel owes a pipeline at all**, then the two conditions and the measured recipe | `gluon/pipeline-reference.md`, and `skill.md ## Procedure` step 2a for the `plain@ns=1` control |
| the pass list, proof-it-landed signals, and the hand-built cross-iteration double buffer for where the pass will not bite | `tile-programming/pipeline.md` |
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
