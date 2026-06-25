---
type: Kernel Case Study
title: _fwd_grouped_kernel_stage1 (grouped GEMM stage-1)
description: Launch-bound Triton grouped GEMM stage-1 sped up 1.18x geomean by adding a host-side HIP-graph capture/replay launcher (compute backend unchanged).
tags: [domain-moe, bottleneck-launch, lever-host-side, gfx942]
speedup: 1.18x geomean
correctness: PASS (same kernel/args replayed; correctness-preserving by construction)
kept: kept-deployed
timestamp: 2026-06-22T00:00:00Z
---

# Baseline
- Backend: Triton (`@triton.jit`), grouped GEMM stage-1 (`_fwd_grouped_kernel_stage1`), pos15 of campaign20.
- Target: AMD MI300X (gfx942 / CDNA3); regime input/output seqlen=1024, geomean over concurrency {2, 32, 64}.
- Baseline frozen in `baseline/`, never edited; speedup independently re-validated by the Director vs the TRUE baseline.
- Per-shape baseline latency not reported in the campaign summary (this entry is grounded only on the summary's relative numbers).

# What changed (the win)
- Single lever: **host-side HIP/CUDA-graph capture + replay** of the same kernel — launch-overhead elimination.
- The compute backend is unchanged (still Triton); only a graph-replay launcher was added on the host side, collapsing the per-call host/dispatch floor on this launch-bound kernel.
- Correctness-preserving by construction: the identical kernel and arguments are replayed.

# Result
| concurrency | speedup |
|-------------|---------|
| c2          | 1.348x  |
| c32         | 1.174x  |
| c64         | 1.039x  |
| **geomean** | **1.18x** |

- Status: accepted / kept-deployed.
- Speedup is largest at low concurrency (c2 1.35x) and decays toward the GPU-bound regime (c64 1.04x), consistent with a launch-overhead-bound kernel.
- Correctness: PASS — same kernel/args replayed (no separate SNR/bit-exact figure broken out in the summary).
- Note: the source is the campaign summary only; per-run docs are brief and no per-shape baseline latency is given.

# What was tried and reverted
- None documented for this kernel in the campaign summary. The only recorded outcome is the accepted HIP-graph launcher; no negative attempts are listed for pos15.

# Patterns
- [Host-side graph replay](/patterns/host-graph-replay.md)

# Citations
1. head_kernels/campaign20/FINAL_REPORT.md (pos15; Results table line, lever #2 "HIP/CUDA-graph capture+replay", and Per-case backend table)
