---
type: Reference
title: Consolidated kernel speedups (all sources)
description: Human-readable consolidated speedup tables across all campaigns (e2e, campaign20, KernelForge, spare_kernels) with provenance and trust notes.
tags: [registry, data, gfx942]
timestamp: 2026-06-22T00:00:00Z
---

# Kernel Optimization Results & Speedups — Consolidated List

> **Distilled success experience (techniques & anti-patterns) → [OKF knowledge base](/index.md)** (this bundle).

> Aggregated 2026-06-22 from all result sources under `agent-kernel-arena/`:
> **1161 `task_result.yaml` runs (107 distinct kernels)**, the audited
> `head_kernels/campaign20/FINAL_REPORT.md`, per-kernel KernelForge /
> spare_kernels reports, and the MiniMax-M2.5 e2e serving benchmarks.
>
> Companion file: **`kernel_speedups_task_results.csv`** — full 107-kernel
> table (best run per kernel, with `base_ms`, `best_opt_ms`, and the source
> path of the best run).

## Data trust / provenance

The same serving kernels recur across several sources (campaign20 / head_kernels
/ KernelForge / spare_kernels / `task_result.yaml`), but numbers differ because
**harness, measurement date, and shape regime differ**. Example —
`paged_attention_decode`: FINAL_REPORT 4.39× vs task_result.yaml 3.40× vs
KernelForge 1.19× (different baselines).

**Authoritative source = `head_kernels/campaign20/FINAL_REPORT.md`** — Director
independently re-validated each speedup vs the frozen TRUE baseline, and
retracted a 17.39× false positive (a harness grid bug on
`chunk_scaled_dot_kkt_fwd_kernel`). Incomplete reports (baseline only):
`_w8a8_triton_block_scaled_mm`, `paged_attention_large`, KernelForge-results/moe-stage1.

---

## A. End-to-end serving (MiniMax-M2.5, SGLang+aiter, 4×MI300X TP=4)

| Run | baseline | optimized | Speedup | Note |
|---|---|---|---|---|
| exp/e2e_minimax_20260615 | 1973 tok/s | 2706 tok/s | **+37.1% (1.37×)** | MoE 2-stage CK dispatch + CK bpreshuffle GEMM (early, config not strictly controlled) |
| e2e_minimax_runs/repro_37pct (06-21) | 2237 tok/s | 2998 tok/s | **+34.0% (1.34×)** | Reproduction; decode bm64→bm16 + bpreshuffle; statistically robust |
| e2e_runs/minimax_m25 | 2200 tok/s | 2741 tok/s | **+24.6% (1.25×)** | MoE 2-stage dispatch only |
| exp/e2e_minimax_freshrun (06-17) | 2004 tok/s | 2269 tok/s | **+13.2% (1.13×)** | Honest number after strict variable control |

Key levers (consistent across runs): decode-path `block_m 64→16` (bm64 has
~87–97% routing padding on sparse decode) + enable the gfx950-gated CK
bpreshuffle on gfx942 (`ARENA_GEMM_BPRE=1`). The kernel-level winners for
paged_attention / dense GEMM are **NOT integrable** into SGLang (KV cache
layout incompatibility).

---

## B. campaign20 production serving kernels (head_kernels/FINAL_REPORT — AUTHORITATIVE)

19 verified kernels, **geomean ≈ 1.43×**.

| # | kernel | Speedup (geomean) | Status / lever | backend change |
|---|---|---|---|---|
| 14 | paged_attention_decode | **4.39×** | faster bf16 ASM route (pa_bf16_noquant_gqa8) + V-shuffle outside timed path | CK→**ASM** |
| 12 | _per_token_group_quant_fp8 | **2.90×** | memory-bound quant rewrite | — |
| 16 | _topk_forward | **1.90×** | CUDA-graph replay + num_warps/stages | — |
| 11 | gemm_a8w8_blockscale | **1.82×** | HIP graph + per-shape config + dequant fold | — |
| 2 | fused_moe_kernel_gptq_awq | **1.63×** | launch-config (warps=1, nonkdim=16, kpack=2) | — |
| 13 | kernel_unified_attention_2d | **1.58×** (honest) | single-pass CUDA-graph + .cv | — |
| 9 | _gemm_a8w8_blockscale_kernel | **1.55×** | launch-proxy HIP opts (nonkdim=32/kpack=2) | — |
| 5 | _gemm_a16_w16_kernel | **1.52×** | in-kernel super-grouping (L2 reuse) + K-peel | — |
| 8 | chunk_scaled_dot_kkt_fwd_kernel | **~1.5×** | HIP-graph(c2)+autotune — **17.39× was RETRACTED** (harness grid bug) | — |
| 7 | moe_stage2 | **1.31×** | block_m heuristic (bm32-V1/id2 dense regime) | CK instance swap |
| 18 | write_req_to_token_pool_triton | **1.28×** | HIP-graph launcher (overhead-bound) | — |
| 15 | _fwd_grouped_kernel_stage1 | **1.18×** | HIP-graph replay (launch-overhead elim) | — |
| 17 | paged_attention_ragged | **1.10×** | per-sig memoization + HIP-graph (8 GPU-bound cases at ceiling) | — |
| 3 | fused_moe_kernel | **1.064×** | int32 index + contiguity hints | — |
| 4 | moe_gemm_fp8_blockscale | 1.005× | at ceiling (hand-tuned aiter ASM baseline) | — |
| 1 | fused_moe_int4_w4a16 | 1.00× | at ceiling (prior L10/L11 winner) | — |
| 6 | moe_stage1 | 1.00× | at ceiling (~96% MFMA peak; lever in off-limits .cu) | — |
| 20 | wvSplitK | 1.00× | at ceiling (~67% launch-floor/HBM) | — |
| 10 | _w8a8_triton_block_scaled_mm | 1.00× (0.994) | at ceiling (throughput levers break numerics gate) | — |
| 19 | paged_attention_large | — | **BLOCKED** (benchmark gate timeout) | — |

---

## C. Highest-speedup standalone kernels (`task_result.yaml` aggregate, 107 kernels, best correctness-passing run)

Mostly hip2hip-extracted 3D/PointNet ops (mmcv `others`) and `extracted-v2`
kernels — far higher than serving kernels. Full table in the CSV.

| kernel (task_name) | best speedup | base→opt (ms) | #runs |
|---|---|---|---|
| hip2hip-extracted-v2/hip_skinny_gemm | **53.52×** | 0.774→0.013 | 9 |
| others/roiaware_pool3d | **21.73×** | 0.545→0.025 | 46 |
| others/roipoint_pool3d | **18.59×** | 0.708→0.030 | 50 |
| others/ball_query | **13.74×** | 0.704→0.043 | 67 |
| others/three_nn | **6.63×** | 0.229→0.029 | 42 |
| others/assign_score_withk | **6.48×** | 0.742→0.066 | 88 |
| others/knn | **6.45×** | 0.528→0.077 | 57 |
| puyuan_lv12/moe_align_block_size | **5.96×** | 0.081→0.013 | 1 |
| extracted-v2/hip_paged_attention_decode | **4.60×** | 0.635→0.135 | 9 |
| extracted-v2/hip_grouped_topk | **4.38×** | 0.074→0.017 | 9 |
| campaign20/baseline/paged_attention_decode | **3.40×** | 0.096→0.034 | 11 |
| extracted-v2/hip_topk_softmax | **3.33×** | 0.039→0.012 | 9 |
| vllm-v2-cursor/hip_silu_and_mul | **1.95×** | 0.014→0.0095 | 5 |
| others/three_interpolate | **1.63×** | 0.0145→0.0084 | 42 |
| others/gather_points | 1.43× | 0.011→0.0076 | 51 |

The remaining ~90 kernels cluster at **1.0–1.3×** (launch-floor / HBM-bound ops:
rms_norm, reshape_and_cache, rotary_embedding, wvSplitK, etc.). A few are
`correctness=FAIL` with no valid speedup: fused_moe_int4_w4a16, _topk_forward,
vllm_wvsplitk, aiter_flash_attn_varlen, rotary_embedding.

---

## D. spare_kernels reference solutions (arena, independent measurement)

| kernel | speedup | base→opt |
|---|---|---|
| fused_moe_int4_w4a16 (triton2triton, geak winner) | **5.19×** | 38.3ms→7.1ms |
| paged_attention_decode (→pa_fwd_asm) | **1.51×** | geomean |
| moe_stage2 | 1.31× | 0.626→0.476 ms |
| gemm_a8w8_blockscale | 1.23× | 0.280→0.228 ms |
| MLA prefill flash-attn (k04, head_dim 192) | 1.21× | 0.116→0.095 ms |
| moe_gemm_fp8_blockscale | 1.19× | 1-stage ASM→2-stage CK |
| fused_moe_gptq_awq (k01 / kimi) | 1.10× | per-iter divide hoist |
| moe_stage1 | 1.08× | 0.867→0.805 ms |

---

## Source file index

- E2E: `e2e_runs/minimax_m25/REPORT.md`, `e2e_minimax_runs/repro_37pct_*/REPRO_REPORT.md`, `exp/e2e_minimax_*/RESULTS_SUMMARY.md`
- campaign20: `head_kernels/campaign20/FINAL_REPORT.md` (+ `phaseA_results.json` baselines)
- KernelForge per-kernel: `KernelForge/results/<kernel>/tasks/cli/*/workspace/optimization_report.md`
- AKA campaign20: `AgentKernelArena/tasks/campaign20/baseline/<kernel>/RESULTS.md`
- spare_kernels: `spare_kernels/arena_tasks/**/RESULTS.md`, `spare_kernels/k0*/.../OPT_NOTES.md`
- Raw per-run speedups: 1161 `AgentKernelArena/**/task_result.yaml` → see `kernel_speedups_task_results.csv`
