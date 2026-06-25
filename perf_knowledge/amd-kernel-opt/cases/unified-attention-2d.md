---
type: Kernel Case Study
title: kernel_unified_attention_2d (Triton decode paged attention)
description: Triton decode paged-attention whose only real win is a host-side single-pass CUDA-graph replay (1.58x honest); all in-kernel body edits were <1% wall-clock noise because the kernel is at a hard launch/latency floor.
tags: [domain-attention, bottleneck-launch, lever-kernel-body, gfx942]
speedup: 1.58x honest (campaign20 single-pass graph) / ~1.00x (KernelForge body-only run)
correctness: PASS; single-pass path bit-exact. Note: a separate "splitKV" graph was FLAGGED as gamed (max_rel_err 1.75/2.24) and excluded.
kept: kept-deployed (graph launcher + low-LDS body); body micro-opts neutral
timestamp: 2026-06-22T00:00:00Z
---

# Baseline
- Backend: Triton (`@triton.jit`), decode paged-attention on MI300X/gfx942.
- Workload: B in {2,32,64}, q_len=1, kv_len=1024, 64 q heads / 8 kv heads, head_size=64, block_size=64, TILE=64, BLOCK_M=16.
- Harness pins all launch meta (num_warps=2, num_stages=3, waves_per_eu=2); only the kernel body is editable in the KernelForge run.
- Measured baseline latency (KernelForge, official runner): c2 0.0662 ms, c32 0.0649 ms, c64 0.0652 ms — flat across 32x batch growth.
- Diagnosis: latency flat across B => latency/launch-bound on the serial 16-tile online-softmax chain, NOT throughput-bound. Kernel ~44us + ~17us fixed launch overhead.

# What changed (the win)
The headline 1.58x comes from a **host-side single-pass CUDA-graph capture+replay** launcher (campaign20 pos13), collapsing the per-call dispatch floor on this launch-bound kernel. Compute backend unchanged.

KernelForge body-level edits (kept, but neutral on wall-clock):
- v1: hoist loop-invariant KV addressing; scalar block-index load instead of 64-wide gather when TILE==BLOCK_SIZE.
- v3: simplify hot-loop masking (drop loop-invariant query masks; gate m_j fixup behind SLIDING_WINDOW>0).
- v7: `tl.range(num_stages=1)` cut LDS **34816 B -> 16384 B** (34KB->16KB). At 34KB only 1 workgroup/CU fits; 16KB allows up to 4 — a genuine occupancy resource win even though it did not move wall-clock here.

# Result
| path | metric | value |
|------|--------|-------|
| campaign20 single-pass graph | honest geomean | **1.58x** |
| campaign20 | per-case c2/c32 | not separately reported (only c64 single-pass path was correctness-exact) |
| KernelForge body-only | geomean | **~1.00x** (neutral) |
| KernelForge official runner | c2/c32/c64 | 0.0662 / 0.0662 / 0.0646 ms vs 0.0662 / 0.0649 / 0.0652 baseline |

- Drift-controlled interleaved A/B (v0/v3/v7, 12x100 alternating launches): all within <1% — earlier sequential "wins" were GPU clock/contention drift.
- Correctness: PASS on all kept changes; single-pass graph bit-exact. The faster "splitKV" graph was numerically wrong and FLAGGED as benchmark gaming (timed path != correctness path).
- Reconciliation: the 1.58x is real but lives entirely in the host launcher; the editable kernel body is at a hard floor, so KernelForge (body-only mandate) correctly found ~1.00x.

# What was tried and reverted
- **Fold qk_scale into Q (pre-loop)** — REVERTED. Correctness FAIL (max_rel_err 3.13 from bf16 rounding of Q*scale); no perf gain.
- **Remove `.cg` cache modifier on K/V** — REVERTED. Worse (~67-68us vs 62-65); `.cg` streaming helps.
- **2-wide tile unroll, COMBINED softmax update** — REVERTED. Correctness FAIL (mre 0.647); sharing max across tiles changes exp2 normalization order.
- **2-wide unroll, SEQUENTIAL updates** — REVERTED. Bit-identical and PASS but perf neutral; num_stages=3 already pipelines loads.
- **splitKV timed graph (campaign20)** — FLAGGED/excluded as gamed; numerically wrong.

# Patterns
- [Single-pass attention](/patterns/single-pass-attention.md)
- [Host graph replay](/patterns/host-graph-replay.md)
- [Launch-bound body opts are invisible (anti-pattern)](/anti-patterns/launch-bound-body-opts-invisible.md)

# Citations
1. KernelForge/results/kernel_unified_attention_2d/tasks/cli/90645755-6cd3-46af-82c9-5266a8ec7afd/workspace/optimization_report.md
2. head_kernels/campaign20/FINAL_REPORT.md
