---
title: Deployment recipes — serving fp8/MXFP4 models on sglang & vLLM (AMD)
kind: technique
gens: [gfx942, gfx950]
dtypes: [fp8_e4m3_fnuz, fp8_e4m3, mxfp4, int8, int4]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - https://docs.sglang.io/platforms/amd_gpu.html
  - https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/quantization.md
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html
  - https://docs.vllm.ai/en/stable/features/quantization/quark/
---

# Deployment recipes — serving quantized models on AMD

> **TL;DR.** Two stacks, one mental model: get a **dialect-correct** quantized checkpoint
> ([[fnuz_vs_ocp.md]], [[calibration_and_quark.md]]), turn on the **AITER** fused-kernel master switch,
> and gate on task accuracy ([[accuracy_evaluation.md]]). vLLM: `VLLM_ROCM_USE_AITER=1` + `quantization`
> / `--kv-cache-dtype`. SGLang: `SGLANG_USE_AITER=1` + `--quantization`. MXFP4 is **CDNA4 + AITER only**;
> on CDNA3 it is sim, FP8 is the real lever ([[hardware_support_matrix.md]], [[block_scaling_mxfp.md]]).

## Pick the right checkpoint first
- **Pre-quantized (fastest)**: AMD's Hugging Face **Quark OCP-FP8** / `*-FP8-KV` collection
  (Llama, Mixtral, DeepSeek-V3/R1, …). These are **OCP**; on **CDNA3** the loader must convert to FNUZ
  ([[fnuz_vs_ocp.md]]).
- **Produce your own**: Quark `quantize_quark.py` (`--quant_scheme w_fp8_a_fp8`, `--kv_cache_dtype fp8`,
  `--quant_algo autosmoothquant`) ([[calibration_and_quark.md]]).
- **On-the-fly**: vLLM ROCm dynamic FP8 / PTPC-FP8 at startup (no pre-quant; +2–5 min startup).

## vLLM on AMD (ROCm V1)
```bash
# AITER master switch — turns on ROCm fused FP8/MoE/attention kernels (MI300X/MI325X/MI350)
export VLLM_ROCM_USE_AITER=1

# A) pre-quantized Quark FP8 (+ FP8 KV)
vllm serve amd/Llama-3.1-8B-Instruct-FP8-KV --dtype auto
# loaded explicitly: LLM(model=..., quantization='quark', kv_cache_dtype='fp8')

# B) on-the-fly dynamic FP8 from a bf16 checkpoint
vllm serve meta-llama/Llama-3.1-8B-Instruct --quantization fp8 --kv-cache-dtype fp8
```
- **AITER sub-flags** default to 1 under the master switch; rarely tune individually. Specialized ones
  retain their own defaults (e.g. FP4 paths).
- **CDNA3 FP4 trap**: `VLLM_ROCM_USE_AITER_FP4BMM=0` on gfx942 — FP4BMM crashes (no HW)
  ([[block_scaling_mxfp.md]]).
- Quant methods that work on ROCm: FP8, AWQ (Triton), GPTQ (HIP), compressed-tensors, Quark.
  **Marlin/NVIDIA-specific** (`awq_marlin`, `gptq_marlin`, `modelopt_fp8/fp4`, `gguf`) **do not** —
  pick the ROCm-native method ([[calibration_and_quark.md]]).

## SGLang on AMD
```bash
export SGLANG_USE_AITER=1   # SGLang's AITER master switch (NOT VLLM_ROCM_USE_AITER)

# FP8 (Aiter or Triton path); pre-quantized FP8 (e.g. DeepSeek-V3/R1) works out of the box
python3 -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --quantization fp8 --host 0.0.0.0 --port 30000

# MXFP4 (CDNA3/CDNA4) — requires SGLANG_USE_AITER=1
#   --quantization mxfp4   (real speedup only on CDNA4; sim on CDNA3)

# DeepSeek-style MoE FP8 on 8×MI300X — note EP requirement
SGLANG_ROCM_AITER_BLOCK_MOE=1 CK_BLOCK_GEMM=1 RCCL_MSCCL_ENABLE=0 \
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 \
  --tp 8 --ep 8 --quantization fp8 --trust-remote-code
```
- SGLang supports FP8, AWQ, MXFP4, W8A8, GPTQ, compressed-tensors, Quark on AMD. AMD-only MoE method:
  `quark_int4fp8_moe` (CDNA3/CDNA4 online quant).
- **MoE EP requirement**: on AMD use `--ep` with `--tp` for MoE; without EP the intermediate dim split
  (e.g. N=320) triggers an AITER CK GEMM incompatibility (both BF16 and FP8).

## AITER_CONFIG / tuning seam
- AITER is **JIT-compiled**: first run may fail with "Child process unexpectedly failed" or take
  30–40 s — **rerun**; artifacts cache to `aiter/aiter/jit/*.so`. Pre-warm before benchmarking.
- AITER fused FP8 kernels are where the quant win is realized: e.g. block-scale fused MoE ~3× and MLA
  decode ~17× on MI300X (AMD-reported). Tie kernel selection to the SOTA cards
  ([[operators/scaled_quant_gemm]], [[operators/fused_moe_grouped_gemm]], [[operators/mla_attention]]).
- Per-op tuning knobs (GEMM tile/split-K, MoE block sizes) live on the backend cards under
  `backends/aiter/` and the operator backend cards — this page is the *serving wiring*, not the kernel
  tuning.

## Containers (matched ROCm + arch)
- MI300X/MI325X: `lmsysorg/sglang:<ver>-rocm700-mi30x`.
- MI350X/MI355X: `lmsysorg/sglang:<ver>-rocm700-mi35x`.
- Use the arch-matched image so the FP8 **dialect** and MXFP HW paths compile correctly
  ([[fnuz_vs_ocp.md]], [[hardware_support_matrix.md]]).

## Gotchas (the short list)
- **OCP checkpoint on MI300 without conversion** → silent 2× error ([[fnuz_vs_ocp.md]]).
- **Mixing up the AITER switch**: `VLLM_ROCM_USE_AITER` (vLLM) vs `SGLANG_USE_AITER` (SGLang).
- **MXFP4 on CDNA3** → no speedup (sim); FP4BMM crashes gfx942.
- **MoE without `--ep`** on AMD → AITER CK GEMM incompatibility.
- **Benchmarking the JIT cold start** → warm up first.
- **Marlin/ModelOpt methods on ROCm** → unsupported; use ROCm-native quant.
- **FP8 KV + prefix caching on RDNA3** → crash (Instinct unaffected, [[kv_cache_quantization.md]]).

## Verify
- After serving: run GSM8K/MMLU through the live endpoint (same harness/seed) and compare to the bf16
  baseline; record tok/s as median of ≥3 warm runs, tagged per [[index/conventions.md]]
  ([[accuracy_evaluation.md]]).

## Sources
- SGLang AMD platform (FP8/MXFP4/Quark, `SGLANG_USE_AITER`, containers, EP): https://docs.sglang.io/platforms/amd_gpu.html
- SGLang quantization methods on AMD (Aiter paths, quark_int4fp8_moe): https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/quantization.md
- vLLM ROCm V1 optimization (`VLLM_ROCM_USE_AITER`, sub-flags, AITER speedups): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html
- vLLM Quark load (`quantization='quark'`, `*-FP8-KV`): https://docs.vllm.ai/en/stable/features/quantization/quark/
