---
type: Kernel Case Study
title: MLA prefill flash-attention forward (_attn_fwd, head_dim_qk=192)
description: MLA prefill flash-attention forward kernel sped up 1.214x geomean by retuning the launch config (smaller BLOCK_M, 2-stage pipelining, fewer warps) plus an empty output buffer.
tags: [domain-attention, bottleneck-compute, lever-config, gfx942]
speedup: "1.214x geomean"
correctness: PASS (atol/rtol = 2e-2, all 3 cases)
kept: kept-deployed
timestamp: 2026-06-22T00:00:00Z
---

# Baseline
Stock Triton flash-attention `_attn_fwd`, MLA prefill: head_dim_qk=192 (nope 128 + rope/PE 64), head_dim_v=128, causal, bf16. AMD MI300X (gfx942, ROCm). batch=1, heads=16, seqlen 512/1024/2048.

Because pe_head_dim = 192-128 = 64, `_get_config` selected the `"pe"` branch of `gfx942-MHA-DEFAULT.json`:
`BLOCK_M=256, BLOCK_N=64, num_warps=8, waves_per_eu=1, num_stages=1, PRELOAD_V=True`.

Launch grid = `batch * heads * cdiv(seqlen, BLOCK_M)`. With BLOCK_M=256 that is only **32 / 64 / 128 workgroups** for s512/s1024/s2048 on a **304-CU** device → heavily under-occupied. Small cases (s512, s1024) are launch/overhead-bound; s2048 is where tiling/pipelining matters.

Baseline latency (median of 3 full-benchmark runs, ms): s512 0.089434 | s1024 0.100336 | s2048 0.173183 | geomean 0.115779.

# What changed (the win)
1. **Retuned the PE config (main win).** Replaced the stock `"pe"` config with
   `BLOCK_M=128, BLOCK_N=64, num_warps=4, waves_per_eu=1, num_stages=2, PRELOAD_V=True`:
   - **BLOCK_M 256 → 128**: doubles workgroups (64/128/256), lifting occupancy on short seqs.
   - **num_stages 1 → 2**: 2-stage software pipelining overlaps K/V global loads with the QK/PV MFMA chain in `_attn_fwd_inner`. (`num_stages ≥ 3` overflows the 64 KB LDS for the 192/128 head config → 2 is the cap.)
   - **num_warps 8 → 4, waves_per_eu=1**: keeps VGPR/LDS pressure low enough to realize the 2 pipeline stages with the smaller tile.
   - Found via a curated config sweep driven through `flash_attn_func(config=...)`, leaving the immutable harness untouched. Best geomean of all combos tried.
2. **`torch.zeros` → `torch.empty` output buffer (small win).** Kernel writes every `(row < seqlen_q, full head_dim)` element and internally zeroes causal early-exit rows, so pre-zeroing is redundant. Removed a memset, shaving ~2-4 µs/case (helps overhead-bound small shapes most).

Changes confined to `kernel_jit.py` + `host.py`.

# Result
Median of 3 full-benchmark runs (100 iters, gpu_lock, caches cleared, variance < 5%):

| case  | baseline (ms) | optimized (ms) | speedup |
|-------|--------------:|---------------:|--------:|
| s512  | 0.089434      | 0.084149       | 1.063x  |
| s1024 | 0.100336      | 0.083655       | 1.199x  |
| s2048 | 0.173183      | 0.123365       | 1.404x  |
| **geomean** | **0.115779** | **0.095409** | **1.214x** |

Per-shape range 1.063x – 1.404x. Correctness: `CORRECTNESS_OVERALL: PASS` on all three cases (atol/rtol = 2e-2). Not bit-exact (tolerance-based bf16 check).

# What was tried and reverted
- `num_stages ≥ 3` and `BLOCK_N=128` with BLOCK_M=128: LDS over-allocation ("out of resource: shared memory") at compile time.
- `BLOCK_M=64`: more workgroups but worse data reuse on s2048 (slower geomean).
- `waves_per_eu` 2/3 and `num_warps=8`: consistently regressed vs the chosen config.
- Bypassing `torch.autograd.Function.apply` for small-case launch overhead: not pursued (would alter public autograd behavior, out of scope); small cases stay launch-bound at ~84 µs.

# Patterns
- [Launch config autotune](/patterns/launch-config-autotune.md)
- [Output empty, not zeros](/patterns/output-empty-not-zeros.md)

# Citations
1. spare_kernels/k04_fmha_prefill/reference_solution/OPT_NOTES.md
