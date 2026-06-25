---
type: Anti-Pattern
title: Non-temporal / streaming loads regress on re-read data
description: NT (streaming) loads bypass L2/Infinity cache; they hurt whenever the data is re-read across blocks.
tags: [domain-moe, domain-gemm, domain-attention, bottleneck-memory, lever-kernel-body, gfx942]
bottleneck: memory
lever_class: kernel-body
timestamp: 2026-06-22T00:00:00Z
---

# The trap
"HBM saturated → use non-temporal/streaming loads to move bytes faster." On these kernels
NT loads **regressed** (e.g. 0.87→1.01–1.04, large cases worst), strongly corroborated
across all 4 harnesses on the MoE GEMMs.

# Why
The weights are re-read across an expert's token-blocks and stay hot in **L2 / Infinity
Cache**. NT bypasses that cache, so every re-read goes back to HBM. This directly refutes
a PROFILE claim of "HBM saturated, optimize bytes moved" — the kernel was occupancy-bound,
not HBM-bound (see [bottleneck-first](/methodology/bottleneck-first-classification.md)
and [CK V3→V1](/patterns/ck-pipeline-v1-occupancy.md)).

# When NT actually helps
Only when each byte is read **exactly once** and never reused — e.g. `block_size=1` paged
KV streaming where each KV page is touched a single time. There NT-K load was accepted
(see [paged_attention_ragged](/cases/paged-attention-ragged.md)).

# Evidence
- [moe_stage1](/cases/moe-stage1.md) / [moe_gemm_fp8_blockscale](/cases/moe-gemm-fp8-blockscale.md) — NT loads OFF is the better setting at the reuse-heavy token=1024 regime.

# Citations
1. AgentKernelArena/tasks/campaign20/baseline/moe_stage1/RESULTS.md (§3 negative results)
2. spare_kernels/arena_tasks/hip2hip/moe_gemm_fp8_blockscale/RESULTS.md
