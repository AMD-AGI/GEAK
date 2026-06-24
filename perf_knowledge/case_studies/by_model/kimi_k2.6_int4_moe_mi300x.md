---
title: Kimi-K2.6 int4 MoE serving throughput on MI300X / vLLM — the memory-free fused-MoE config win
kind: case_study
operator: fused_moe_grouped_gemm
backend: vllm_kernels
gens: [gfx942]
dtypes: [int4_w4a16]
regimes: [prefill, decode]
status: sota
updated: 2026-06-17
sources:
  - test_results/kimi_2.6_20260610/RESULTS.md
  - test_results/kimi_2.6_20260610/director_e2e_validation.json
  - test_results/kimi_2.6_20260610/kernels/moe_int4_tune/tune_report.md
  - GEAK/e2e_workflow/knowledge/gemm_tuning/moe_int4_tuning.md
---

# Kimi-K2.6 / vLLM / MI300X — int4 fused-MoE config tune (the reproducible +16%)

> The model-specific numbers/shape live HERE; the *generic* recipe that reproduces
> them lives in [`../../../e2e_workflow/knowledge/gemm_tuning/moe_int4_tuning.md`](../../../e2e_workflow/knowledge/gemm_tuning/moe_int4_tuning.md)
> (shape derived from `config.json`+TP, prefill batch from the workload, gated by
> the existing e2e Integrator) — nothing here is wired as bespoke per-model code.

## Context
- **Model:** moonshotai-Kimi-K2.6 (`KimiK25ForConditionalGeneration`, DeepSeek-V3-class
  text decoder), **~1T-param int4 MoE**, compressed-tensors **w4a16** (weight-only,
  `num_bits=4`, `group_size=32`, symmetric, `input_activations: null`).
- **Per-rank fused-MoE shape @ TP=8:** `E=384` routed experts, `N=256`
  (`moe_intermediate_size 2048 / TP8`), `K=7168` (hidden), `topk=8`, bf16 compute.
- **Stack:** vLLM v1 (0.21.0), TP=8 on GPUs 0–7, `gpu_memory_utilization 0.95`,
  `max_model_len 9472`. **Hardware:** MI300X (gfx942), noise band 0.5%.
- **Workload:** ISL/OSL/conc = **8192/1024/64**, `random_range_ratio=0` (fixed len),
  num_prompts=192, 3 repeats. **Prefill-dominated.**

## The bottleneck
torch profile: the **int4 expert grouped-GEMM is ~57% of GPU time** and ran on
vLLM's **slow default fallback** — vLLM ships **no** tuned Triton config for
`E=384,N=256,int4_w4a16` (the "Using default MoE config" warning). That missing
per-shape JSON is the entire opportunity.

## The win (measured-by-us, +16.4% e2e, parity-preserved)
| metric | baseline | final | Δ |
|---|---|---|---|
| output throughput | 514.55 tok/s | **598.98** (director-verified) | **+16.4%** (1.164×) |
| GSM8K (5-shot greedy) | 0.965 | **0.9733** | parity preserved |
| TPOT | 119.89 ms | 100.83 ms | better |
| TTFT | 3297 ms | 6067 ms | worse (mnbt trade-off) |

Two compounding, **zero-source / zero-extra-HBM** levers:
1. **Tuned int4_w4a16 fused-MoE Triton config** via `VLLM_TUNED_CONFIG_FOLDER`
   (the decisive +16.4%). Per-M-bucket kernel speedups from the faithful
   `fused_experts` micro-sweep (parity rel<1e-2 per bucket):

   | M | default ms | best ms | speedup | best tile |
   |---|---|---|---|---|
   | 4096 | 4.27 | 2.79 | **1.53×** | M128 N128 K32 G8 w8 |
   | 8192 | 7.55 | 4.74 | **1.59×** | M256 N256 K32 G8 w8 |
   | 16384 | 14.87 | 9.72 | **1.53×** | M128 N128 K64 G8 w8 |

   (`BLOCK_SIZE_M=256` is a legal Triton tile vLLM accepts from the JSON; it matches
   AMD's MI350X reference trend. Small-M decode buckets gain little — the win is the
   prefill mass.)
2. **`--max-num-batched-tokens 16384`** (+1.5% alone): bigger prefill batches push M
   into the large buckets where the tuned config wins. The recipe derives this from
   the workload (`clamp(2·ISL, 8192..32768)`), not a fixed constant.

## The trap (why the agentic kernel path failed at e2e)
A separate run let the head-kernel track author an **fp8-fold rewrite** of the same
MoE op: op-level **1.51×**, but it caches a **second fp8 weight copy** → at memory
parity it **OOMs at KV-cache init** (e2e-undeployable). The config tune above gets a
comparable kernel speedup with **no memory cost**. **Lesson: for an int4 MoE, tune the
Triton config (memory-free) BEFORE — and usually instead of — any quant rewrite; gate
a quant rewrite on a real HBM-footprint check at the run's `gpu_memory_utilization`.**

## How to reproduce (generic, framework-native)
Driven through GEAK's normal e2e path: the **Op Benchmarker** routes the int4 fused-MoE
head op to the [`moe_int4_tuning`](../../../e2e_workflow/knowledge/gemm_tuning/moe_int4_tuning.md)
recipe (writes the shape-generic driver into `$EVAL_DIR`, sweeps, emits a
`winner_kind=env` with `apply_env=VLLM_TUNED_CONFIG_FOLDER` + recommended mnbt), and the
**e2e Integrator** runs the tight A/B (baseline vs +tuned-config+mnbt), the engagement
check ("Using default MoE config" gone), parity, and the Amdahl + `mem_footprint_starves_kv`
gates — no bespoke per-model code. The measured quantity is the **ratio B/A under one
identical 口径**, so the +16% relative win holds even when the absolute baseline differs
(e.g. variable-length `range_ratio` lowers both legs together).

## Lessons
1. **A missing vendor config is a first-class, zero-HBM win** — check for the
   "default MoE config" warning before reaching for a rewrite.
2. **Memory parity is the real e2e gate on a big MoE.** A faster kernel that doubles
   weight bytes starves the KV cache and never deploys.
3. **Pair the tuned config with a large prefill batch** so the tuned large-M buckets
   actually execute.
4. **Report the A/B ratio, not absolute tok/s** — it survives box/口径 drift.

## Cross-links
- Operator tuning: [`../../operators/fused_moe_grouped_gemm/tuning.md`](../../operators/fused_moe_grouped_gemm/tuning.md)
- Head-kernel routing: [`../../../e2e_workflow/roles/op_benchmarker.md`](../../../e2e_workflow/roles/op_benchmarker.md)
- Reproduce: recipe [`../../../e2e_workflow/knowledge/gemm_tuning/moe_int4_tuning.md`](../../../e2e_workflow/knowledge/gemm_tuning/moe_int4_tuning.md) (driver written into `$EVAL_DIR` at runtime; gated by the e2e Integrator)

<!-- MANIFEST: Kimi-K2.6 int4 w4a16 MoE on MI300X/vLLM — missing VLLM_TUNED_CONFIG_FOLDER int4 fused-MoE Triton config tuned per-M-bucket (1.53-1.59x prefill) + mnbt → +16.4% e2e, parity-preserved, ZERO extra HBM; fp8-fold rewrite OOMs at memory parity. Generic recipe (knowledge, no shipped scripts): e2e_workflow/knowledge/gemm_tuning/moe_int4_tuning.md, routed by op_benchmarker + gated by e2e_integrator. -->
