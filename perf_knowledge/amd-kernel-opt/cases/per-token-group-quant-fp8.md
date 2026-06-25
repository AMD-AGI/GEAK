---
type: Kernel Case Study
title: _per_token_group_quant_fp8 (per-token-group fp8 quantization)
description: Memory-bound per-token-group fp8 quantization kernel rewritten in-place (Triton kernel body), reaching 2.90x geomean on gfx942.
tags: [domain-quant, bottleneck-memory, lever-kernel-body, gfx942]
speedup: "2.90x geomean"
correctness: PASS (Director re-validated vs TRUE baseline on the timed path; per-shape SNR not separately reported)
kept: kept-deployed
timestamp: 2026-06-22T00:00:00Z
---

# Baseline
- Backend: Triton (`@triton.jit`) per-token-group fp8 quantization.
- Target: AMD MI300X (gfx942 / CDNA3); regime input/output seqlen=1024, geomean over concurrency {2, 32, 64}.
- Memory-bound kernel (no per-shape absolute baseline latency given in the campaign summary).

# What changed (the win)
- Memory-bound quantization rewrite of the Triton kernel body. Compute backend unchanged (Triton -> Triton, "changed? = no" in the per-case backend table).
- The campaign summary documents this only as a "memory-bound quant rewrite" / "kernel-body" lever; the specific in-body changes (e.g. vectorization / load-store widening / reduction layout) are NOT enumerated in the source report.

# Result
| case | c2 | c32 | c64 | geomean |
|------|-----|------|------|---------|
| speedup | 1.603 | 3.843 | 3.974 | 2.90x |

- Status: accepted. Strong scaling with concurrency — modest at c2 (1.60x), large at c32/c64 (~3.8-4.0x), consistent with a memory-bound kernel where higher concurrency exposes more bandwidth headroom to recover.
- Correctness: PASS — every speedup independently re-validated by the Director vs the TRUE frozen baseline on the timed path. The campaign added an explicit anti-gaming clause (timed path == correctness path) after pos13; no gaming was flagged for this kernel.
- Bit-exactness / SNR: not separately reported for this kernel in the source (FINAL_REPORT pos12). Documentation for this entry is brief; treat the win as Director-verified rather than fully characterized.

# What was tried and reverted
- No negative / reverted attempts are documented for pos12 in the source report. The report is a campaign-level summary and does not include this kernel's per-round insight log.

# Patterns
- No patterns or anti-patterns linked (none supplied for this entry, and the source lacks the body-level detail to attribute one confidently).

# Citations
1. head_kernels/campaign20/FINAL_REPORT.md (pos12; Results table line 20; per-case backend table line 84)
