---
type: Kernel Case Study
title: paged_attention (vLLM decode, single-pass flash)
description: vLLM paged-attention decode rewritten as a single-pass flash kernel with online softmax that writes final_out directly, eliminating the second reduce-kernel launch for 1.18x geomean.
tags: [domain-attention, bottleneck-launch, lever-kernel-body, gfx942]
speedup: 1.18x geomean
correctness: PASS (official harness performance run; all 8 cases validated)
kept: kept-deployed
timestamp: 2026-06-22T00:00:00Z
---

# Baseline
- Original vLLM paged-attention decode: two-kernel design — partition-attention kernel followed by a separate reduce kernel that combines partial results.
- Workload is tiny per case: bf16, head_size=128, block_size=16, seq_lens=32 — effectively single-partition, so the second (reduce) kernel launch is pure launch overhead.
- Measured baseline geomean: 22.95 µs (per-case 0.0221–0.0240 ms).
- Hardware: MI300X / gfx942, HIP_VISIBLE_DEVICES=0. Harness: `scripts/task_runner.py performance`, 5 warmup + 30 timed CUDA-event iterations averaged, 8 test cases.

# What changed (the win)
- Winner **cc**: collapse the two-kernel pipeline into ONE single-pass flash kernel.
- Use **online softmax** to accumulate within the single pass and write **final_out directly**, so the second **reduce kernel launch is fully removed** (not just no-op early-returned).
- This is the core differentiator: the launch-overhead bottleneck is eliminated at the source rather than short-circuited.

# Result
| version | geomean speedup (vs original) | reduce launch removed? |
|---------|-------------------------------|------------------------|
| **cc** (this winner) | **1.18x** (measured) | **yes** |
| cursor | 1.05x (recorded) | no — still launches, only no-op return |
| geak_v3 | 1.04x (recorded) | no — centralized evaluator tuning |

- Per-case cc speedup range: **1.10x – 1.24x** (0.0204 ms case is the floor at 1.103x; best is sig_b167bd1c728f at 1.236x).
- cc geomean latency: 19.46 µs vs 22.95 µs baseline.
- Correctness: all 8 cases pass under the official performance harness. Report does not state bit-exactness or an explicit SNR figure.
- Note: only `original` and `cc` were re-measured on GPU0 this run; cursor/geak numbers are recorded values from other devices/times, used only for trend comparison.

# What was tried and reverted
- No reverted attempts documented for the cc winner itself.
- Contrasting (non-winning) approaches from the report: cursor's single-partition fast path + early-return in reduce + short-partition tile skip, and geak_v3's centralized-evaluator microtuning. Both correctly diagnosed "tiny workload, single partition" but kept the reduce kernel launch, capping their gains at ~1.04–1.05x.

# Patterns
- [Single-pass attention](/patterns/single-pass-attention.md)

# Citations
1. task_specific_skills/paged_attention_vllm/perf_results.md
