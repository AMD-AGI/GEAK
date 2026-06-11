# Conventions — file format, frontmatter, naming

Every content file follows these rules so the base stays navigable and machine-queryable.

## Frontmatter (YAML, required on every operator/backend/hardware/language file)
```yaml
---
title: <human title>
kind: hardware | language | backend | operator_overview | sota_card | technique | quant | profiling | workflow | case_study | reference
# for sota_card (operators/<op>/backends/<backend>.md):
operator: dense_gemm
backend: flydsl            # one of the controlled backend ids (see taxonomy.md)
gens: [gfx942, gfx950]     # gfx906=MI100, gfx90a=MI200, gfx942=MI300, gfx950=MI350
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e5m2_fnuz, fp4_e2m1, fp6, int8]
regimes: [prefill, decode, training, both]
status: sota | competitive | legacy | experimental | na
updated: 2026-06-08
sources: [<url-or-repo@commit>, ...]
---
```

## Controlled vocabularies
Operator ids, backend ids, gen ids, dtype ids, and regime ids are defined in
[`taxonomy.md`](taxonomy.md). Use exactly those ids in frontmatter so `sota_registry.yaml` can be
generated/validated from the files.

## Section order
- **operator_overview**: TL;DR → math contract → shape regimes → Amdahl/where-it-matters →
  backend landscape (link table) → fusion neighbors → numerics → how-to-bench → Sources.
- **sota_card** (see [`_templates/sota_card_template.md`](_templates/sota_card_template.md)):
  TL;DR decision → SOTA implementation table → knobs/config space → numerics/parity → integration
  (rebind seam) → pitfalls → how to verify → alternatives → Sources.
- **hardware/language/backend/technique**: TL;DR → concepts → the levers → pitfalls → verify → Sources.

## Status badges (used in `sota_matrix.md`)
`🟢 sota` · `🟡 competitive` · `🧪 experimental` · `🟤 legacy` · `⚪ na`

## Performance number format
`<value> @ <hw>, ROCm <ver>, <lib>@<commit/ver>, <date>` — e.g.
`+2.23% e2e @ MI300X gfx942, sglang 0.5.11 / aiter, 2026-06-08`. Prefer median of ≥3 warm repeats;
note spread. Never present theoretical peak as achievable.

## Naming
- dirs/files: `snake_case`. Operator dirs match the operator id. Backend cards are
  `<backend_id>.md` under `<op>/backends/`.
- One fact per file where practical; link related files with relative paths.

## Sources
- This file defines repo conventions; no external source required.
