---
id: REPLACE_with_kebab_slug
title: "REPLACE — one line: operator + scenario + (migration from->to)"
kind: expert_skill
authors: [REPLACE]
scope: kernel            # kernel | e2e  — decides validation harness AND consuming layer
# ---- selector: the workflow matches these against the live bottleneck ----
match:
  operator: REPLACE                 # MUST exist in ../index/capability_index.yaml (validated by scaffold)
  arch_class: ['*']                 # e.g. [deepseek_mla]; '*' = any model arch class
  gens: [gfx942]                    # gfx942 (MI300X) | gfx950 (MI350X) | ...
  dtypes: [bf16]                    # bf16 | fp8_e4m3_fnuz | fp4_e2m1 | mxfp4
  regimes: [decode]                 # prefill | decode | training
  from_backend: ""                  # migration skills only: source backend (e.g. tilelang)
  to_backend: ""                    # migration/author skills: target backend (e.g. triton, flydsl)
  profile_signature:                # optional extra trigger gate
    op_name_regex: ""
    min_pct_gpu: 0.0
# ---- expected effect: the validation gate's pass criteria ----
expects:
  isolated_speedup_min: 1.10        # kernel scope: isolated A/B vs the immutable oracle
  e2e_delta_min_pct: 1.0            # e2e scope: Director same-session e2e delta
  parity: required                  # required | relaxed(<tol)
# ---- validation: AUTO-FILLED by validate_skill.py — do NOT hand-edit ----
validation:
  status: draft                     # draft | validated | stale | failed
  last_verified: ""
  gpu: ""
  model: ""
  measured: {isolated: "", e2e_pct: "", parity: ""}
  artifact: ""                      # path to the validation eval dir
role: advisory_prior                # the consuming workflow treats this as advisory, never a mandate
supersedes: []
---

## When to use
<!-- 1-2 sentences: the exact bottleneck/shape/arch where this recipe applies. -->

## Mechanism
<!-- WHY it works: the hardware / numerics / scheduling reason. This is what lets an agent transfer it. -->

## Procedure
<!-- The REGULATED STEPS. Write them so a workflow author-agent can reproduce the win directly:
     concrete entrypoints, files, kernel structure, the lever (e.g. "fuse dequant into one fp8 MFMA").
     This is the high-prior candidate the workflow reproduces, then gates by measurement. -->

## Knobs & pitfalls
<!-- Tunable knobs and their safe ranges; known failure modes. -->

## Do-no-harm notes
<!-- Where this MUST be kept off (e.g. "decode small-M must stay generic — wide-N tile tanks it 0.6x").
     The validation gate verifies these boundaries are respected (no regression when not triggered). -->

## Sources
<!-- eval dirs, ledger entries, commits, papers. Every claim needs a pointer. -->
