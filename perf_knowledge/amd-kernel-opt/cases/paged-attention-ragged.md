---
type: Kernel Case Study
title: paged_attention_ragged (AITER vLLM-style paged decode attention)
description: AITER paged decode attention on gfx942 won ~1.05-1.10x by flipping K loads non-temporal (NT_KV_LOAD) in the GOLDEN kernel, exploiting block_size=1 single-touch KV streaming.
tags: [domain-attention, bottleneck-memory, lever-kernel-body, gfx942]
speedup: ~1.05-1.10x on dominant captured cases; flat on perf cases
correctness: PASS (bit-exact; pure kernel-internal change)
kept: kept-deployed
timestamp: 2026-06-22T00:00:00Z
---

# Baseline
- Kernel: AITER `paged_attention_ragged` (vLLM-style paged decode). Two launches: `paged_attention_ll4mi_QKV_mfma16_kernel` (dominant, memory-bound) + `paged_attention_ll4mi_reduce_kernel`.
- Workload: bf16 Q/K/V, head_size=128, gqa=8 (32 q-heads / 4 kv-heads), block_size=1. Captured cases = large-context decode (~8001 tokens, 32 seqs); perf cases c2/c32/c64 = ctx 1024.
- Roofline: QKV kernel memory-bound, ~109% of nominal HBM BW (L2-assisted); theoretical headroom only ~0.92-1.16x (very tight).
- Baseline latency (mean of 100 iters), original GOLDEN kernel, correctness PASS:

| case (captured) | ms | | case (perf) | ms |
|---|---|---|---|---|
| sig_28d61f7b1f07 | 0.1770 | | c2  | 0.0585 |
| sig_b637b17af08a | 0.1716 | | c32 | 0.0637 |
| sig_61620957a144 | 0.1636 | | c64 | 0.0639 |
| sig_4d693a861996 | 0.1628 | | | |

# What changed (the win)
- Flip `NT_KV_LOAD` false -> true in the GOLDEN kernel so K global loads use non-temporal cache hints.
- Rationale: block_size=1 streams each unique KV page exactly once (no reuse), so non-temporal K loads avoid L2 pollution without sacrificing reuse — the one regime where NT helps. K is consumed straight from registers.
- Plus (accepted alongside): per-signature memoization and HIP-graph replay for launch overhead.
- Lever: kernel-body / L2 locality. Pure kernel-internal change; transfers to real training.

# Result
- Optimized latency: captured 0.1485-0.1657 ms; perf 0.0591 / 0.0636 / 0.0624 ms.
- Speedup: ~1.05-1.10x on the dominant captured (large-context) cases; flat on the short-context perf cases.
- Correctness: PASS, bit-exact. Saved as `optimized_versions/v2_nt_kload.cuh`, accepted as new baseline.
- Memory-bound near the practical ceiling — see reverted occupancy attempt below.

# What was tried and reverted
| attempt | change | outcome |
|---|---|---|
| 1 | default -> EXPERIMENTAL kernel (head=128/bf16 specialized; coop K->LDS, double-buffer, NT) | ~0.8x REGRESSION on captured; tuned for short-context/high-batch, not the ~8001-token cases. REVERTED |
| 3 | also make V loads non-temporal (`load_ntmprl_16Byte`) | ~0.85x; c64 blew up to 0.58 ms. NT path splits 16B into 4x4B, loses coalescing; V benefits from L2. REVERTED |
| 4 | clean single 16B NT V load (`load_ntmprl_16Byte_vec`) | ~0.88x captured (~1.03x small perf). V is restaged through LDS + transposed and reused per block. REVERTED |
| 5 | `__launch_bounds__(NUM_THREADS, 2)` for occupancy x2 | ~0.97-1.0x. Profiler confirmed occupancy 1->2 waves/SIMD, no spills — yet no gain => genuinely HBM-bound. Confirms v2 near ceiling. REVERTED |

Note: in a separate small-decode campaign (ctx 2048-4096, paged_attention_decode), the same NT-KV lever was ~noise (1.001-1.028x) and did NOT reproduce in clean interleave — NT (a cache-bypass lever) only pays off in the large-PA single-touch regime, not the latency/occupancy-bound small-batch decode shape. See citation [2].

# Patterns
- [Host graph replay](/patterns/host-graph-replay.md)
- [Non-temporal load regression](/anti-patterns/non-temporal-load-regression.md) (anti-pattern: NT helped K here but regressed V and did not transfer to the small-decode shape)

# Citations
1. KernelForge/results/paged_attention_ragged/tasks/cli/89a8cf43-ffff-443a-9c77-72662c61de48/workspace/optimization_report.md
2. AgentKernelArena/tasks/campaign20/baseline/paged_attention_decode/RESULTS.md
