---
title: RDNA3.5 / gfx1151 (Strix Halo) — occupancy
kind: hardware
gens: [gfx1151]
dtypes: []
regimes: [both]
updated: 2026-11-06
sources:
  - ../data/hw_constants.json
  - ../../../kernel_workflow/scripts/kernel_tools/amd_occupancy.py
---

# RDNA3.5 / gfx1151 — occupancy

This card is a pointer, not a second copy of the occupancy table.

- Workflow-facing rules: [`kernel_workflow/knowledge/amd_ryzen.md`](../../../kernel_workflow/knowledge/amd_ryzen.md) §1–2
- Machine-readable constants: [`hw_constants.json`](../data/hw_constants.json), key `gfx1151`
- Reproduction command (from the repository root, with a ROCm LLVM compiler):
  `python3 kernel_workflow/scripts/kernel_tools/amd_occupancy.py --compiler-sweep --arch gfx1151 --format json`

## TL;DR
> The occupancy steps are compiler-derived on gfx1151 itself (`hw_constants.json`). On this part a
> spill costs DRAM bandwidth that the kernel is usually already short of
> ([isa_notes.md](isa_notes.md)), so stay at or below a step rather than spilling to reach the next.
