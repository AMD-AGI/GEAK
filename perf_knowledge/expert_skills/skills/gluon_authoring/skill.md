---
id: gluon_authoring
title: "Gluon on AMD CDNA (gfx950, gfx942) — reference fact store: queryable cards (API, errors, ISA, hardware, traps, tools), each with scope and evidence"
kind: expert_skill
authors: [qiongz]
scope: kernel
revision: "2026-11-26"
match:
  operator: '*'
  arch_class: ['*']
  gens: [gfx950, gfx942]
  dtypes: ['*']
  regimes: ['*']
  from_backend: [triton, gluon]
  to_backend: gluon
expects:
  isolated_speedup_min: 1.10
  parity: required
validation:
  status: draft
  last_verified: ""
  gpu: ""
  model: ""
  measured: {isolated: "", e2e_pct: "", parity: ""}
  artifact: ""
role: reference_fact_store
supersedes: []
---

# gluon_authoring - fact cards for Gluon on AMD CDNA (reference, not a recipe)

## When to use

Writing, porting (plain Triton -> Gluon) or debugging Gluon / Triton kernels on gfx950 (MI350X/MI355X)
or gfx942 (MI300X/MI325X), Triton 3.8.0: an API spelling, a compile error, a layout / MFMA / LDS /
async-copy lowering rule, a compiler knob, an ISA hazard or rate, a profiler trap. It prescribes no
procedure and no levers: query it, then judge each card yourself.

## Evidence and scope

Card fields: `claim` (scoped, conditional), `scope` (arch, triton, dtype, regime), `evidence`, `signal`,
`check` (how to confirm it on your box), `keys`, `see`, `overrides`. Evidence: `compile-checked` (CPU
AOT compile, Triton 3.8.0), `gpu-measured` (microbenchmark on gfx950), `source:<file:line>@v3.8.0`,
`doc:<ref>`, `unverified` (ranked last and labelled; run its `check` first). Outside a card's scope a card is a hypothesis.

**Precedence over the GEAK knowledge base:** a card whose `overrides[]` names a `perf_knowledge/` (or
`kernel_workflow/knowledge/`) file supersedes that file for the card's scope while this skill is loaded;
unnamed KB content still applies. `q.py` prints `overrides:` on such cards.

## How to query

`$SKILL` = `perf_knowledge/expert_skills/skills/gluon_authoring`, `$KT` = `kernel_workflow/scripts/kernel_tools`
(paths relative to the GEAK root).

```bash
python3 $SKILL/scripts/q.py "<words or a pasted error line>" [--arch gfx950] [--dtype fp8] [--kind error]
python3 $SKILL/scripts/q.py --id <card-id>             # whole card; --toc lists groups
python3 $SKILL/scripts/q.py --kind tool [<words>]      # tools: one line each, or search them
```

Paste a compiler/runtime error line verbatim (exact strings rank first). `$SKILL/facts.jsonl` holds the
cards; `$SKILL/INDEX.md` (generated: ids, keys, error string -> id) is large: grep it, do not read it
whole. Arch traps (`scripts/q.py --id ...`): `pipe.gluon-no-pipeliner` (num_stages does
nothing on @gluon.jit), `lowp.fnuz-on-gfx950-upcast` (FNUZ fp8 silently becomes fp16 MFMA on gfx950),
`mfma.gfx950-fp8-shape-registry` (which fp8 instr_shapes exist), `async.direct-to-lds-width` (128/32-bit
only; 32 on gfx942), `occ.occupancy-comment-register-only` (`; Occupancy:` ignores LDS).

Tools live in `$KT` (shared) and `$SKILL/scripts` (Gluon-specific); `--help` is each one's interface.
Refusal tokens (`WITHHELD`, `NOT CALIBRATED`, `SCOPED TOOL-GAP`) are answers, not errors. GEAK owns
rounds, GPU locking, timing and acceptance; nothing here is a gate.

## Sources

Triton 3.8.0 source and CPU AOT compiles, CDNA3/CDNA4 ISA documents (`$KT/amd_isa.py`), AMD product
pages (`$KT/extract_sku.py`), gfx950 microbenchmarks; each card names its basis. Origin: `PROVENANCE.md`.
