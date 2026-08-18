---
title: ROCm / library version matrix — the stack every perf number is pinned to
kind: reference
updated: 2026-08-10
layer: reference
lifecycle: active
platforms: [gfx942, gfx950]
verified_on: null
verified_stack: {rocm: "7.1", aiter: a6bb499375849eec45d68c5ccaebc8865fd422c0, sglang: "0.5.11"}
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0
  - https://github.com/ROCm/aiter
  - reference/repo_index.md
---

# ROCm / library version matrix

`sourcing_rules.md` §2 requires every performance number to be **version-tagged**
(`value @ hardware, ROCm <ver>, <lib>@<commit/ver>, <date>`). This file is the single place that records
**which stack each measured number was taken on**, so the lifecycle machinery can detect *stack drift*:
when the live stack crosses a row below, every card pinned to the old row is a candidate for
`lifecycle: stale` (see [`../index/taxonomy.md`](../index/taxonomy.md) and `_ingest_web.py`'s drift signal).

It is **not** a source of peak numbers (those live in
[`../../e2e_workflow/knowledge/analysis_skills/roofline/peaks.md`](../../e2e_workflow/knowledge/analysis_skills/roofline/peaks.md))
and it is **not** the pin registry ([`repo_index.md`](repo_index.md) owns `repo@commit`). It maps
*measurement epochs* → *the stack that epoch assumed*.

## How to read this

- A card's `verified_stack: {rocm, aiter, sglang, ...}` frontmatter names the row it belongs to.
- When the on-box stack moves to a **newer minor** of ROCm or aiter than a card's `verified_stack`, that
  card's numbers are **unverified on the current stack** until a run re-measures them. The compensating
  rule (plan Part 2.3): a card whose `verified_on` is `null` is *permanently* shown as `⚠ unverified` —
  the monthly review walks that list. Stack drift → `stale`; only `_promote.py` (≥2 on-box reproductions)
  restores `active`.

## Current epoch (what the repo's numbers assume unless a card says otherwise)

| component | version / pin | arch(s) | notes |
|---|---|---|---|
| ROCm | 7.1 | gfx942, gfx950 | base runtime for all current perf numbers |
| aiter | `a6bb499375849eec45d68c5ccaebc8865fd422c0` (v0.1.12.post1-150) | gfx942, gfx950 | central kernel engine; see [`repo_index.md`](repo_index.md) |
| flydsl (pip) | `0.1.5` | gfx942, gfx950 | MLIR-Python DSL (FLIR→ROCDL) |
| sglang | 0.5.11 | gfx942, gfx950 | serving / attention-backend selection |
| hipBLASLt | ROCm-7.1 bundled | gfx942, gfx950 | solidx tables are build-specific — never portable |
| Composable Kernel | ROCm/rocm-libraries (projects/composablekernel) | gfx942, gfx950 | CK-Tile path |

> gfx1250 / CDNA5 / MI450: **no verified stack yet.** No perf number in this repo is pinned to it. The
> only reference is `expert_skills/skills/gluon_authoring/references/gluon/atoms-reference.md:50`. Any
> gfx1250 row here must stay empty until an on-box stack is available.

## Measurement epochs (append-only; never overwrite)

Each row is a stack snapshot a batch of numbers was taken on. When a number is re-measured on a newer
stack, **append** a new row (per `sourcing_rules.md` §Maintenance) so the trend stays visible.

| epoch (date) | ROCm | aiter | sglang | flydsl | who / trigger |
|---|---|---|---|---|---|
| 2026-06-08 | 7.1 | a6bb4993 (v0.1.12.post1-150) | 0.5.11 | 0.1.5 | initial perf_knowledge seeding |

## Stack-drift → lifecycle (cross-reference)

| trigger | detected by | effect |
|---|---|---|
| ROCm/aiter crosses a **minor** version vs a card's `verified_stack` | monthly review + `_ingest_web.py` | affected cards → `lifecycle: stale` (still resolvable, `⚠`-flagged) |
| upstream `repo@commit` in `upstream_rev` moves | `_ingest_web.py` git source | card → `stale` |
| on-box run reproduces a card's claim ≥2× on the current epoch | `_promote.py` | `verified_on` / `verified_stack` refreshed → `active` |

## Sources

- Stack pins mirror [`repo_index.md`](repo_index.md) (single source of truth for `repo@commit`).
- Version-tagging + append-only policy: [`../index/sourcing_rules.md`](../index/sourcing_rules.md) §2, §Maintenance.
- Lifecycle / drift semantics: [`../index/taxonomy.md`](../index/taxonomy.md); on-box promotion: `../index/_promote.py`.
