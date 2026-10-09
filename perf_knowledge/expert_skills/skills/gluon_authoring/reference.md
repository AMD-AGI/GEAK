# gluon_authoring — reference router

Every file below is **lazy**: [`skill.md`](skill.md) names the stage you are in, and this page names the one
file that continues it. Load that file, query the section you need by heading
(or `scripts/toolctl.py context query --markdown-heading "<heading>"`), and stop. gfx950 is
the main line in every file; gfx942 appears as the downgrade beside it.

Onboarding is not here — read GEAK's language layer first if you have not written Gluon before:
[`languages/gluon/overview.md`](../../../languages/gluon/overview.md),
[`programming_model.md`](../../../languages/gluon/programming_model.md),
[`gemm_cookbook.md`](../../../languages/gluon/gemm_cookbook.md).

## By stage (`skill.md ## Procedure`)

| stage | read | then, only if needed |
| --- | --- | --- |
| the whole flow, the gates, the non-negotiable rules | [`references/method/index.md`](references/method/index.md) | — |
| **entry** — modes A/B/C, champion assertion, comparator, depth contract, escalation seam, the launch args | [`references/method/entry.md`](references/method/entry.md) | [`references/method/front-end.md`](references/method/front-end.md) — what the plain front end already covered |
| **budget** — `hw_budget`, bound-class prior, floor, roofline reporting contract | [`references/method/budget.md`](references/method/budget.md) | [`references/hardware/roofline-models.md`](references/hardware/roofline-models.md), [`planning-constants.md`](references/hardware/planning-constants.md) |
| **transcribe** — recover, apply, compile, the four equivalence checks, divergence ledger | [`references/method/transcribe.md`](references/method/transcribe.md) | [`references/tile-programming/layout-recipes.md`](references/tile-programming/layout-recipes.md), [`references/gluon/index.md`](references/gluon/index.md) |
| **recover** — attribution, `plain@ns=1`, the three suspects, parity gate, last-resort re-injection | [`references/method/recover.md`](references/method/recover.md) | [`references/gluon/pipeline/reinjection.md`](references/gluon/pipeline/reinjection.md) (diagnostic / last resort only) |
| **evidence** — profiler entry, the four dials, PMC / ATT / SOL reading | [`references/method/profile.md`](references/method/profile.md) | [`references/hardware/bound-class-signals.md`](references/hardware/bound-class-signals.md) |
| **measure** — acceptance timing, screening, hygiene | [`references/method/benchmark-hygiene.md`](references/method/benchmark-hygiene.md) | [`references/method/records.md`](references/method/records.md) `9a. Anomaly validation` |
| **climb** — Layer Backbone, reversed-intuition traps, one layer per round, AMD lever cards | [`references/method/climb.md`](references/method/climb.md) | the layer file below |
| **close** — outcome enum, final report, closure self-review, cost discipline | [`references/method/close.md`](references/method/close.md) | [`references/method/records.md`](references/method/records.md) |
| runtime failure, deadlock, missing doc | [`references/method/triage.md`](references/method/triage.md) | — |
| which GEAK role runs which stage, GPU lock / broker, optional record tools (toolctl) | [`references/method/orchestration.md`](references/method/orchestration.md) | `skill.md ## Roles in GEAK` |

## By layer (`skill.md ## Mechanism`, Layer Backbone)

| layer | read |
| --- | --- |
| 1 — transcription anchor | [`references/method/transcribe.md`](references/method/transcribe.md) |
| 1.5 — scheduling model | [`references/tile-programming/scheduling-model.md`](references/tile-programming/scheduling-model.md), [`warp-pipeline.md`](references/tile-programming/warp-pipeline.md) |
| 2 — memory path | [`references/tile-programming/memory-path.md`](references/tile-programming/memory-path.md), [`references/gluon/memory-reference.md`](references/gluon/memory-reference.md) |
| 3 — LDS layout | [`references/tile-programming/layout-recipes.md`](references/tile-programming/layout-recipes.md), [`references/gluon/smem-lds-reference.md`](references/gluon/smem-lds-reference.md) |
| 4 — pipeline (hand-written first; the single order is defined here) | [`references/tile-programming/pipeline.md`](references/tile-programming/pipeline.md), then [`references/gluon/pipeline-reference.md`](references/gluon/pipeline-reference.md) for the spelling on your build |
| 4+ — instruction scheduling | [`references/tile-programming/instruction-scheduling.md`](references/tile-programming/instruction-scheduling.md), [`references/gluon/inline-asm-reference.md`](references/gluon/inline-asm-reference.md) |
| 5 — slicing / registers / occupancy | [`references/tile-programming/slicing.md`](references/tile-programming/slicing.md) |
| 6 — beyond the hot loop | [`references/workloads/gemm.md`](references/workloads/gemm.md), [`attention.md`](references/workloads/attention.md) |
| 7 — matrix engine + low precision | [`references/tile-programming/low-precision.md`](references/tile-programming/low-precision.md), [`references/gluon/matrix-reference.md`](references/gluon/matrix-reference.md) |
| compiler co-design (LLVM / LLIR, 3.8.0 surface, fork-only knobs) | [`references/tile-programming/compiler-contract.md`](references/tile-programming/compiler-contract.md), [`llvm-codesign-handbook.md`](references/tile-programming/llvm-codesign-handbook.md) |
| the archetype — which layers have surface | [`references/workloads/index.md`](references/workloads/index.md) (the row `hw_budget.py --workload` prints), [`intake.md`](references/workloads/intake.md) |

## Gluon API (`references/gluon/`)

| you need | read |
| --- | --- |
| where to start, gfx950 ↔ gfx942 at a glance | [`gluon/index.md`](references/gluon/index.md) |
| imports, launch, AOT | [`imports-and-launching.md`](references/gluon/imports-and-launching.md), [`shared-aot-reference.md`](references/gluon/shared-aot-reference.md) |
| layouts | [`layout-reference.md`](references/gluon/layout-reference.md) |
| MFMA / scaled MFMA | [`matrix-reference.md`](references/gluon/matrix-reference.md), [`atoms-reference.md`](references/gluon/atoms-reference.md) |
| global access, `buffer_load`, async copy | [`memory-reference.md`](references/gluon/memory-reference.md) |
| pipeline spellings per build | [`pipeline-reference.md`](references/gluon/pipeline-reference.md) → `pipeline/{authored-overlap,async-ordering,marker-and-version-gates,loop-knobs-and-targets,reinjection}.md` |
| inline asm | [`inline-asm-reference.md`](references/gluon/inline-asm-reference.md) → `inline-asm/*.md` |
| a runnable skeleton | [`gfx950-minimal-examples.md`](references/gluon/gfx950-minimal-examples.md), `scripts/pipeline_examples_cdna4.py` (gfx942 downgrade: `pipeline_examples_cdna3.py`) |
| RDNA WMMA (reference only; out of `match.gens`) | [`rdna-wmma-reference.md`](references/gluon/rdna-wmma-reference.md) |

## What not to write (`skill.md ## Knobs & pitfalls`)

| you need | read |
| --- | --- |
| constructs that compile and then cost you; Quick Reject; the inline-asm gate | [`references/pitfalls/negative-patterns.md`](references/pitfalls/negative-patterns.md) |
| ROCm / driver / toolchain constraints | [`references/pitfalls/platform-known-issues.md`](references/pitfalls/platform-known-issues.md) |

## Hardware (`skill.md ## Arch dispatch`)

| you need | read |
| --- | --- |
| per-arch constants, per-SKU peaks, thresholds (data — the single source) | `perf_knowledge/hardware/data/{hw_constants,sku,thresholds,workload_models}.json` |
| capability by target and version | [`references/hardware/capability-matrix.md`](references/hardware/capability-matrix.md), [`atlas.md`](references/hardware/atlas.md) |
| the gfx942 downgrade in one page | [`references/hardware/cdna3-gfx942.md`](references/hardware/cdna3-gfx942.md) |
| what the hardware actually does (ISA mechanisms, primary sources) | [`isa-mechanisms.md`](references/hardware/isa-mechanisms.md), [`primary-sources.md`](references/hardware/primary-sources.md) |
| SKU tables | `references/hardware/amd-*-skus.md` |
| AMD lever cards (data) and the term index | `references/hardware/lever-cards.json` (via `scripts/lever_index.py`), [`term-index.md`](references/hardware/term-index.md) |
| GEAK's own hardware cards | [`perf_knowledge/hardware/`](../../../hardware/) (`cdna4_mi350/`, `cdna3_mi300/`, `shared/`) |

## Tools

[`scripts/USAGE.md`](scripts/USAGE.md) — every tool by stage, including the GEAK-shared ones in
`kernel_workflow/scripts/kernel_tools/` and the GEAK infrastructure the skill defers to.
