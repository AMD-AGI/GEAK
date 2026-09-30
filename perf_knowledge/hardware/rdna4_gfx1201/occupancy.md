---
title: RDNA4 / gfx1201 — occupancy
kind: hardware
gens: [gfx1201]
dtypes: []
regimes: [both]
updated: 2026-09-21
sources:
  - ../../expert_skills/skills/gluon_authoring/references/hardware/hw_constants.json
  - ../../expert_skills/skills/gluon_authoring/scripts/amd_occupancy.py
---

# RDNA4 / gfx1201 — occupancy

This card is a pointer, not a second prose copy of the occupancy table.

- Workflow-facing rules and the human-readable table:
  [`kernel_workflow/knowledge/amd_rdna4.md`](../../../kernel_workflow/knowledge/amd_rdna4.md)
- Machine-readable constants: Gluon `hw_constants.json`, key `gfx1201`
- Reproduction command:
  `amd_occupancy.py --compiler-sweep --arch gfx1201 --format json`

## TL;DR
> GEAK's HIP/Triton workflow uses the compiler-derived static occupancy model:
> cap **16 waves/SIMD**, VGPR allocation granule 24, and a 1536-VGPR SIMD
> file. `llvm-mc -mcpu=gfx1201` accepts `s_alloc_vgpr`; dynamic allocation has
> not been demonstrated through the workflow's HIP/Triton paths, so do not use
> it in occupancy planning without separate toolchain evidence.
