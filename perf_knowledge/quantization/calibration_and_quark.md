---
title: Calibration & AMD Quark — GPTQ/AWQ/SmoothQuant/QuaRot on ROCm
kind: technique
gens: [gfx942, gfx950]
dtypes: [fp8_e4m3_fnuz, fp8_e4m3, mxfp4, mxfp6, int8, int4]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - https://docs.vllm.ai/en/stable/features/quantization/quark/
  - https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/gpu_dev_optimize/fp8_quantization_quark_vllm.html
  - https://rocm.blogs.amd.com/artificial-intelligence/quark/README.html
  - https://arxiv.org/pdf/2211.10438
---

# Calibration & AMD Quark

> **TL;DR.** **AMD Quark** is the first-party toolkit that *produces* quantized checkpoints for MI GPUs:
> weight + activation + KV-cache quant, FP8/INT8/INT4/MXFP4/MXFP6, with algorithms from Min-Max →
> SmoothQuant/AWQ/GPTQ → rotations (QuaRot). It exports Hugging-Face-format checkpoints that **vLLM and
> SGLang load directly** (`quantization='quark'`). This page covers *which algorithm, what calibration,
> what the serving stack consumes*. Granularity choices: [[scaling_strategies.md]]; gates:
> [[accuracy_evaluation.md]]; serve: [[deployment_recipes.md]].

## What Quark is
A quantization pipeline for LLMs/VLMs on AMD Instinct. It supports **FP8 and INT8** for activations,
weights, and **KV cache** (including FP8 attention), and a two-level **INT4-weight / FP8-compute** scheme
for ~4× compression at FP8-class accuracy. Scales across multiple GPUs for ultra-large models
(e.g. Llama-3.1-405B). Open-source; roadmap adds **MXFP4/MXFP6** and on-the-fly quant in vLLM/SGLang.

## Algorithm menu (compose into a pipeline)
| algorithm | flag/idea | what it does | when |
|---|---|---|---|
| Min-Max / Percentile | default observers | simple amax-based scales | fast baseline |
| **SmoothQuant / AutoSmoothQuant** | `--quant_algo autosmoothquant` | migrate activation outliers into weights (α per-layer by MSE) | W8A8 / per-token FP8 ([[scaling_strategies.md]]) |
| **AWQ** | `--quant_algo awq` | activation-aware weight scaling, protect salient channels | INT4/INT8 weight-only |
| **GPTQ** | `--quant_algo gptq` | second-order error-compensated weight rounding | INT4/INT8 weight-only |
| **QuaRot** (rotations) | Hadamard transform | rotate away activation outliers before low-bit cast | near-lossless **MXFP4** ([[operators/quant_fp4_mxfp]]) |
Quark's API lets you **compose** these (e.g. rotation → SmoothQuant → GPTQ).

## Calibration — what & how much
- **Calibration data**: a small representative corpus (e.g. C4, Pile, or task-domain text). The Quark
  scripts default to a few hundred samples; the FP8 tutorial uses `--num_calib_data 512`.
- **Static** activation/KV scales need calibration; **dynamic** activation scales do not
  ([[scaling_strategies.md]]). Weight scales are computed from the weights directly (no data).
- More calibration data ≠ strictly better — match the **distribution** to the serving workload;
  validate with the task gate ([[accuracy_evaluation.md]]).

## Example: produce an FP8 checkpoint
```bash
python3 quantize_quark.py \
  --model_dir meta-llama/Llama-3.1-8B-Instruct \
  --output_dir ./Llama-3.1-8B-Instruct-FP8 \
  --quant_scheme w_fp8_a_fp8 \
  --kv_cache_dtype fp8 \
  --quant_algo autosmoothquant \
  --num_calib_data 512 \
  --model_export hf_format \
  --tasks gsm8k
```
`--quant_scheme w_fp8_a_fp8` = FP8 weights + FP8 activations; `--kv_cache_dtype fp8` adds FP8 KV
([[kv_cache_quantization.md]]); `--tasks gsm8k` runs an accuracy check as it exports. MXFP4 uses the
analogous `mxfp4_quantization_quark_vllm` recipe (Llama-3.3-70B, MXFP4 for vLLM).

## What the serving stacks consume
- **vLLM**: load a Quark checkpoint with `quantization='quark'` (+ `kv_cache_dtype='fp8'` for FP8 KV).
  vLLM also supports **GPTQ** (HIP kernels, HF GPTQ models out of the box) and **AWQ** (Triton kernels)
  on ROCm, plus on-the-fly FP8/PTPC-FP8 at startup.
- **SGLang**: loads Quark-exported HF checkpoints; same scheme/KV flags ([[deployment_recipes.md]]).
- **Other stacks**: native PyTorch, ONNXRuntime, llama.cpp can also load Quark output.
- **Pre-quantized checkpoints**: AMD publishes the **Quark OCP-FP8 model collection** on Hugging Face
  (Llama, Mixtral, Grok-1, …) — skip calibration entirely. Mind the **dialect**: these are **OCP**;
  on CDNA3 the loader must apply FNUZ conversion ([[fnuz_vs_ocp.md]]).

## Dialect & arch awareness (don't skip)
Quark targets OCP FP8 (portable, matches H100/Blackwell and CDNA4). On **CDNA3 (gfx942)** the runtime
must convert OCP→FNUZ (scale ×2/÷2; ±448→±224) — handled by the loader, but a mismatch is the silent 2×
trap ([[fnuz_vs_ocp.md]]). On **CDNA4 (gfx950)** OCP loads natively.

## Pitfalls
- **Calib distribution ≠ serving distribution** → static scales drift; revalidate.
- **Loading an OCP checkpoint on MI300 without conversion** → 2× error ([[fnuz_vs_ocp.md]]).
- **Over-calibrating** — diminishing returns; gate on task accuracy, not sample count.
- **Picking GPTQ/AWQ when you need activation quant** — those are weight-only; use SmoothQuant/FP8 for
  W8A8/per-token activations.
- **Assuming Quark MXFP4 speeds up MI300** — sim only on CDNA3 ([[block_scaling_mxfp.md]]).

## Verify
- Run Quark's built-in `--tasks gsm8k`/`mmlu` during export; then re-check e2e under the real serving
  stack ([[accuracy_evaluation.md]]). Confirm the checkpoint's FP8 dialect matches the target arch.

## Sources
- AMD Quark in vLLM (schemes, algorithms, load `quantization='quark'`): https://docs.vllm.ai/en/stable/features/quantization/quark/
- Quark FP8 tutorial (quantize_quark.py, num_calib_data, kv_cache_dtype): https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/gpu_dev_optimize/fp8_quantization_quark_vllm.html
- Quark accuracy/perf on AMD GPUs (vLLM + SGLang): https://rocm.blogs.amd.com/artificial-intelligence/quark/README.html
- SmoothQuant: https://arxiv.org/pdf/2211.10438
