---
title: Serving a Llama-class model in fp8 on MI300X — the deployment recipe
kind: case_study
operator: dense_gemm
backend: aiter
gens: [gfx942, gfx950]
dtypes: [fp8_e4m3_fnuz, fp8_e4m3]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - https://docs.sglang.io/platforms/amd_gpu.html
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html
  - https://docs.vllm.ai/en/stable/features/quantization/quark/
  - https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/quantization.md
---

# Llama-class fp8 serving on MI300X

> This is a **synthesized recipe** from the quantization/deployment pages plus **vendor-reported**
> kernel speedups (AMD ROCm docs / SGLang docs), labelled inline. The fp8 *serving wiring* and
> the *gate* are the validated, repeatable parts; the per-kernel multipliers are vendor figures
> you should re-confirm on your box with the accuracy probe below. Note: on **Qwen3.5-27B** our
> own e2e run **rejected** `--quantization fp8` at the accuracy gate
> ([`qwen3.5-27b_sglang_e2e.md`](qwen3.5-27b_sglang_e2e.md)) — fp8 is a per-model accuracy
> decision, never assumed.

## Context
A standard dense Llama-class decoder (e.g. Llama-3.1-8B/70B-Instruct): MHA/GQA attention + dense
MLP. The serving win from fp8 is twofold — **half the weight/activation bytes into every GEMM**
(GEMM is the Amdahl head) and **half the KV-cache traffic** on decode (memory-bound). On AMD the
two levers are: (1) a **dialect-correct fp8 checkpoint**, (2) the **AITER fused-kernel master
switch** that routes those GEMMs/attention/MoE through the fp8 kernels. See
[`../../quantization/deployment_recipes.md`](../../quantization/deployment_recipes.md) and
[`../../quantization/formats_overview.md`](../../quantization/formats_overview.md).

## Baseline
- bf16 Llama serving on MI300X (full-byte GEMM inputs + bf16 KV cache) — the throughput and
  accuracy reference.
- The accuracy reference for the fp8 candidate: **GSM8K / MMLU through the live endpoint**, same
  harness/seed, compared to the bf16 baseline ([`../../quantization/accuracy_evaluation.md`](../../quantization/accuracy_evaluation.md)).

## The recipe (what works)

### 1. Pick a dialect-correct checkpoint first (the #1 fp8 trap on MI300X)
- **Fastest:** AMD's Hugging Face **Quark OCP-FP8 / `*-FP8-KV`** collection (Llama, Mixtral,
  DeepSeek-V3/R1, …). These are **OCP**; on **CDNA3 (gfx942) the loader must convert to FNUZ** —
  an OCP checkpoint used un-converted on MI300 is a **silent ~2× error**
  ([`../../quantization/fnuz_vs_ocp.md`](../../quantization/fnuz_vs_ocp.md)).
- **Produce your own:** Quark `quantize_quark.py --quant_scheme w_fp8_a_fp8 --kv_cache_dtype fp8
  --quant_algo autosmoothquant` ([`../../quantization/calibration_and_quark.md`](../../quantization/calibration_and_quark.md)).
- **On-the-fly:** vLLM ROCm dynamic FP8 / PTPC-FP8 at startup (no pre-quant; +2–5 min startup).

### 2. Turn on the AITER master switch + quant flags
```bash
# vLLM (ROCm V1)
export VLLM_ROCM_USE_AITER=1                       # master switch: ROCm fused fp8/MoE/attn kernels
vllm serve amd/Llama-3.1-8B-Instruct-FP8-KV --dtype auto          # pre-quantized Quark FP8+KV
# or on-the-fly from a bf16 ckpt:
vllm serve meta-llama/Llama-3.1-8B-Instruct --quantization fp8 --kv-cache-dtype fp8

# SGLang  (NOTE the different switch name)
export SGLANG_USE_AITER=1
python3 -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --quantization fp8 --host 0.0.0.0 --port 30000
```

### 3. Stack the GEMM tune on top (the bf16 lesson transfers)
fp8 dense GEMM still flows through the aiter dispatch; reach the fp8 (a8w8) GEMM and FlyDSL-fp8
path via the same DB tune (`libtype=flydsl`) — see
[`../../kernel_workflow/gemm_tuning_workflow.md`](../../kernel_workflow/gemm_tuning_workflow.md) (the fp8/CDNA4
note) and [`../by_kernel/gemm_aiter_db_tuning.md`](../by_kernel/gemm_aiter_db_tuning.md).

## What didn't / the traps (kept honestly)
- **OCP checkpoint on MI300 without FNUZ conversion → silent 2× error.** The most common fp8
  deployment bug on CDNA3.
- **Mixing up the AITER switch:** `VLLM_ROCM_USE_AITER` (vLLM) vs `SGLANG_USE_AITER` (SGLang).
- **Marlin / ModelOpt / gguf methods (`awq_marlin`, `gptq_marlin`, `modelopt_fp8/fp4`) are
  NVIDIA-specific → unsupported on ROCm.** Use a ROCm-native method (FP8, AWQ-Triton, GPTQ-HIP,
  compressed-tensors, Quark).
- **CDNA3 FP4 trap:** `VLLM_ROCM_USE_AITER_FP4BMM=0` on gfx942 — FP4BMM crashes (no HW); MXFP4 is
  CDNA4-only (sim on CDNA3, no speedup).
- **Benchmarking the JIT cold start:** AITER is JIT-compiled; first run may take 30–40 s or fail
  with "Child process unexpectedly failed" → **rerun, warm up, then bench**.
- **fp8 KV + prefix caching on RDNA3 → crash** (Instinct unaffected).
- **fp8 is not free accuracy:** our Qwen3.5-27B run rejected fp8 at the accuracy gate. Always run
  the GSM8K/MMLU probe.

## Final result (numbers, vendor-reported)
| lever | value | source / label |
|---|---|---|
| AITER fused block-scale MoE | **~3×** (MI300X) | **vendor** — vLLM ROCm optimization doc |
| AITER MLA decode (DeepSeek-class) | **~17×** (MI300X) | **vendor** — same doc / aiter-mla blog |
| RMSNorm + fp8 dynamic-quant fusion | 1–6% e2e latency | **vendor** — sglang #18466 (see [`../by_kernel/fused_norm_quant_win.md`](../by_kernel/fused_norm_quant_win.md)) |
| fp8 weight/activation + fp8 KV | ½ GEMM input bytes, ½ KV traffic | structural (the recipe's mechanism) |

**Our own e2e measurement on this exact recipe for a Llama-class model is not in the eval dirs**
— the auditable e2e fp8 datapoint we *do* have is the **Qwen3.5-27B fp8 rejection at the
accuracy gate**. Treat the multipliers above as the vendor floor to beat, and gate on accuracy.

## Lessons
1. **Checkpoint dialect (FNUZ vs OCP) before anything else** — the silent 2× error on CDNA3 is
   the highest-impact fp8 mistake.
2. **AITER master switch is the lever that *realizes* the fp8 kernels** — the quant flag alone
   without the switch leaves performance on the table.
3. **ROCm-native quant methods only** — Marlin/ModelOpt are a dead end on AMD.
4. **fp8 is an accuracy/throughput trade, gated per model** — same `--quantization fp8` flag was
   accepted as a recipe here and rejected on Qwen3.5-27B; the GSM8K/MMLU probe decides.
5. **Warm the JIT cache before benchmarking** — cold-start time is not steady-state throughput.

## Cross-links
- Deployment wiring (the source of this recipe): [`../../quantization/deployment_recipes.md`](../../quantization/deployment_recipes.md)
- fp8 dialect: [`../../quantization/fnuz_vs_ocp.md`](../../quantization/fnuz_vs_ocp.md) · checkpoints: [`../../quantization/calibration_and_quark.md`](../../quantization/calibration_and_quark.md)
- Accuracy gate: [`../../quantization/accuracy_evaluation.md`](../../quantization/accuracy_evaluation.md) · KV: [`../../quantization/kv_cache_quantization.md`](../../quantization/kv_cache_quantization.md)
- fp8 GEMM: [`../../operators/scaled_quant_gemm/`](../../operators/scaled_quant_gemm/) · [`../by_kernel/gemm_aiter_db_tuning.md`](../by_kernel/gemm_aiter_db_tuning.md)
- aiter backend: [`../../backends/aiter/overview.md`](../../backends/aiter/overview.md)

## Sources
- fp8/MXFP4 serving recipe, AITER switches, FNUZ/OCP, containers, JIT warmup, ~3× MoE / ~17× MLA: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html and `GEAK/perf_knowledge/quantization/deployment_recipes.md`.
- SGLang AMD fp8/Quark methods, `SGLANG_USE_AITER`: https://docs.sglang.io/platforms/amd_gpu.html and https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/quantization.md.
- Quark fp8 load (`quantization='quark'`, `*-FP8-KV`): https://docs.vllm.ai/en/stable/features/quantization/quark/.
- Norm+quant 1–6% e2e: https://github.com/sgl-project/sglang/issues/18466.

<!-- MANIFEST: Llama-class fp8 serving on MI300X — synthesized recipe (FNUZ-correct Quark ckpt + AITER master switch + GEMM tune), vendor ~3× MoE / ~17× MLA; fp8 gated per-model on GSM8K/MMLU (rejected on Qwen3.5-27B). -->
