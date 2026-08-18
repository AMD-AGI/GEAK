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

## Structured index frontmatter (additive — plan Part 1.3)
On top of the block above, the structured index layer (`_gen_index.py` / `kb_resolve.py`) consumes these.
All are OPTIONAL for authoring — `_backfill_kb.py` auto-derives `platforms`/`kernel_class`/`layer`; a human
fills `levers`/`cost`/`bound_type` over time (empty = validator *warning*, not error).
```yaml
layer: reference | learned | artifact   # auto: reference for perf_knowledge/, learned for learned/ cards
platforms: [gfx942, gfx950]             # auto-derived from `gens:`; [] = platform-independent
skus: [mi355x]                          # only when SKUs differ materially (see taxonomy.md)
kernel_class: gemm.dense                # auto from operator map / learned `key:` line
levers: [config.per-shape-tune]         # the means this file documents
cost: L1                                # construction cost of that means (L0<L1<L2<L3)
risk: parity-safe
bound_type: [mfma_compute]              # roofline routing key(s)
# lifecycle three clocks (plan Part 2.1-2.2):
lifecycle: active                       # candidate | active | stale | archived
verified_on: 2026-07-20                 # last ON-BOX measurement date, or null
verified_stack: {rocm: "7.1", aiter: a6bb4993}
upstream_rev: {ROCm/aiter: a6bb4993}
```

## Controlled vocabularies
Operator ids, backend ids, gen ids, dtype ids, regime ids, and the structured axes above
(`kernel_class`, `lever`, `cost`, `bound_type`, `sku`, `lifecycle`) are defined in
[`taxonomy.md`](taxonomy.md), with a machine-readable mirror in [`_kb_vocab.py`](_kb_vocab.py) — the single
source both the generators and the validator import. Use exactly those ids in frontmatter so
`sota_registry.yaml` + `views/` + `kb_manifest.yaml` can be generated/validated from the files.

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
