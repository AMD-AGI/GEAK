# expert_skills — human-authored, e2e-validated optimization recipes

> **Contract (read this first).** An expert skill is an **advisory prior**, not a mandate. It packages a
> human expert's *proven, reusable optimization recipe* (the regulated steps + knobs + pitfalls) for a
> specific `operator × scenario`. A skill that has passed validation carries a high prior and a
> reproducible procedure — but it **never overrides on-box measurement**. The consuming workflow
> (`e2e_workflow` / `kernel_workflow`) treats a matched skill as a *candidate to reproduce first*, then
> decides the winner by its own A/B gate. When a skill conflicts with the live measurement, the
> measurement wins and the skill is flagged `stale` for re-review.
>
> This is the same discipline as the sibling `perf_knowledge/` base — *seed/locate candidates faster,
> never reduce a result below its measured baseline* — just with stronger, validated, opinionated recipes.

## Why this exists

`perf_knowledge/` holds *facts* (APIs, knobs, which backends exist). The `e2e_workflow/knowledge/`
ledgers hold the *agent's own rolling experience*. Neither captures a **human expert's end-to-end
recipe** — e.g. "port the MLA decode core from TileLang to Triton on gfx942", or "the FlyDSL fp8 a8w8
blockscale down-proj playbook that won +67% e2e". `expert_skills/` is that third tier: contributed,
indexed, and **validated to actually move e2e (or an isolated kernel) without regressing the baseline**.

## Layout

```
expert_skills/
├── README.md                  # this file (the contract)
├── index.yaml                 # machine-queryable selector + validation status (AUTO-MAINTAINED)
├── skills/<id>/               # one SUBDIRECTORY per skill
│   ├── skill.md               #   the main recipe (frontmatter selector + body) — REQUIRED
│   └── ...                    #   optional extra files: reference kernels, configs, validation manifest
├── _template/                 # SKILL_TEMPLATE.md + validation_manifest.yaml
└── _contribute/               # the "add a skill to GEAK" skill: scaffold / validate / make_pr / SKILL.md
```

Each skill lives in its own directory `skills/<id>/` so a skill that needs more than prose — a reference
kernel, tuned config JSONs, a custom validation manifest — can carry those files alongside its
`skill.md`. The selector (`index.yaml`) always points at `skills/<id>/skill.md`.

## How a skill is selected by the workflows

Each skill's frontmatter has a `match:` block. Expert skills are **opt-in** — the workflows ignore this
directory entirely unless the run passes `use_expert_skills=true` (default OFF; when OFF the workflow
behaves byte-identically to a build without this feature). When enabled, the workflow filters
`index.yaml` by the current bottleneck:

```
match.operator == bottleneck.operator
AND gen ∈ match.gens
AND model_arch_class ∈ match.arch_class   (or match.arch_class contains '*')
AND (migration skills) from_backend/to_backend fit the live path
AND validation.status == validated        (stale/draft/failed are NOT auto-applied)
```

There is **no ranking** here (same as `capability_index.yaml`). Every match enters the candidate set;
the on-box A/B picks the winner.

## scope: kernel vs e2e

A skill declares `scope: kernel | e2e`. It controls which harness validates it and which workflow layer
consumes it:

| scope  | validated by      | consumed by                          | pass criteria |
|--------|-------------------|--------------------------------------|---------------|
| kernel | `kernel_workflow` (isolated A/B vs the immutable unittest oracle) | `kernel_workflow` author/optimize | `isolated_speedup ≥ expects.isolated_speedup_min` + parity |
| e2e    | `e2e_workflow` (Director same-session A/B) | `e2e_workflow` routing/integration | `e2e_delta ≥ expects.e2e_delta_min_pct` + parity + non-trigger inertness |

## Validation status lifecycle

`draft` → (validate passes) → `validated` → (staleness: age / aiter·triton version drift) → `stale`
→ (re-validate) → `validated`. A failed validation sets `failed` and the PR cannot land as `validated`.
Only `validated` skills are auto-applied; `stale` is demoted to a plain reference (re-measure forced).

## Contributing

See [`_contribute/SKILL.md`](_contribute/SKILL.md). Short version:

```
python _contribute/scaffold.py --id <slug> --operator <op> --scope <kernel|e2e>   # make skeleton + register
$EDITOR skills/<slug>/skill.md                                                           # fill Procedure/Mechanism/Do-no-harm
bash   _contribute/make_pr.sh <slug>                                               # validate (by scope) → set status → open PR
```

## Canonical copy

This directory is maintained ONLY in the canonical `geak_v4/GEAK` tree. Other PerfSkills snapshots are
downstream copies — do not edit skills there; they sync from here.

## Sources
- Design discussion 2026-06-17 (this repo).
- Sibling contract: `perf_knowledge/README.md` (reference-only doctrine).
