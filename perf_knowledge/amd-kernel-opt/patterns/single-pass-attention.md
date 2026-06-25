---
type: Optimization Pattern
title: Single-pass attention (online softmax, drop the reduce launch)
description: Fuse the partition-reduce into one flash pass that writes the final output directly, removing a second kernel launch.
tags: [domain-attention, bottleneck-launch, lever-kernel-body, gfx942]
bottleneck: launch + memory
lever_class: kernel-body
median_speedup: 1.18x-1.58x
timestamp: 2026-06-22T00:00:00Z
---

# When to use
Decode/paged attention implemented as two kernels: a partitioned attention kernel plus a
separate softmax-merge/reduce kernel. The second launch and its HBM round-trip are pure
overhead when the partition count is small.

# Mechanism
Carry the running `(m, l, O_acc)` online-softmax state and write `final_out` directly from
the single attention pass — no materialized S, no separate reduce kernel. Empty partitions
must write neutral `(sum=0, max=-inf, 0)` to avoid `-inf − -inf = NaN`. A single-partition
fast path skips the merge entirely. Often paired with
[HIP-graph replay](/patterns/host-graph-replay.md).

# Evidence
- [paged_attention (vLLM)](/cases/paged-attention-vllm-singlepass.md) — single-pass flash + online softmax + direct write, **eliminates the reduce launch** → **1.18×** (beats cursor 1.05× / geak 1.04× which kept the reduce launch)
- [kernel_unified_attention_2d](/cases/unified-attention-2d.md) — single-pass CUDA-graph path → **1.58×** (honest)

# Caveats
- The merge-elimination win shrinks as the partition count grows (it is the launch that is
  expensive). Keep the multi-partition path for long contexts.
- Guard fully-masked tiles against NaN.

# Citations
1. task_specific_skills/paged_attention_vllm/perf_results.md
2. head_kernels/campaign20/FINAL_REPORT.md (pos13)
