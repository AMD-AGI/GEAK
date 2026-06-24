---
title: Model bring-up checklist — fast first run on MI300X / MI350X
kind: workflow
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3, fp4_e2m1, fp6]
regimes: [prefill, decode, both]
status: sota
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html
  - https://rocm.blogs.amd.com/artificial-intelligence/scaling-ai-inference/README.html
  - https://github.com/vllm-project/vllm/issues/36337
  - GEAK/e2e_workflow/knowledge/gemm_tuning/aiter_gemm_tuning.md
---

# Model bring-up checklist (MI300X / MI350X)

## TL;DR
Get a new model **running correctly and reasonably fast** on MI300X (gfx942 / CDNA3) or
MI350X (gfx950 / CDNA4) before you optimize. Order: pin the stack → enable AITER → pick the
quant format (mind FNUZ vs E4M3FN) → pick the attention backend → set parallelism within one
XGMI island → **accuracy smoke test FIRST** (silent quant/MoE corruption is the #1 bring-up
bug) → then tune (GEMM DB, fused norms, MoE, collectives). Each tuning lever is its own
workflow, linked below. After bring-up, run [`optimize_e2e_model.md`](optimize_e2e_model.md).

## 0. Pin the stack (reproducibility)
- **ROCm 7.2.x** stable; MI350X/MI355X supported since ROCm 7.0.0. **Pin the ROCm version
  the model card specifies** — a gfx950 MXFP4 dequant regression between 7.1 and 7.2 made
  Kimi-K2.5-MXFP4 emit gibberish.
- Pull the official pre-built vLLM ROCm / vllm-omni Docker image (no source build needed
  since Jan 2026). aiter is JIT — first call pays a compile; warm before benchmarking.
- Note memory budget: MI300X 192GB, MI325X 256GB, MI350X/MI355X 288GB HBM3e.

## 1. Enable AITER (the master perf switch)
```bash
export VLLM_ROCM_USE_AITER=1     # vLLM master switch: GEMM/RMSNorm/MoE/attn (on by default in sglang ROCm)
export HIP_FORCE_DEV_KERNARG=1   # faster kernel launch
```
Per-family sub-switches (`VLLM_ROCM_USE_AITER_LINEAR/_MOE/_MLA/_BLOCK_GEMM`) default on with
the parent. Diagnostic ladder if AITER paths break: `AITER_ONLINE_TUNE=1` →
`VLLM_ROCM_USE_AITER_MOE=0` → `VLLM_ROCM_USE_AITER=0` (full Triton fallback). See
[`../backends/aiter/integration.md`](../backends/aiter/integration.md).

## 2. Quant format (mind FNUZ vs E4M3FN)
- **gfx942 (CDNA3): fp8 E4M3 FNUZ.** sglang/vLLM re-quantize gfx942 checkpoints to FNUZ.
- **gfx950 (CDNA4): standard fp8 E4M3FN + native block-scaled MXFP8 / MXFP6 / MXFP4**
  (FP6 runs at FP4 rate; e8m0 group scales). **Porting a quantized checkpoint across gens
  must convert FNUZ↔E4M3FN.**
- Easiest path: AMD pre-quantized **Quark** checkpoints (FP8 W8A8, MXFP4 preview). Custom →
  Quark toolchain (AutoSmoothQuant; MXFP4 typically mixes fp8 per-tensor weight/act/kv +
  fp4 per-group). KV cache: add `--kv-cache-dtype fp8` for ~2× seq/batch headroom.
- See [`../quantization/`](../quantization/), [`../operators/quant_fp4_mxfp/`](../operators/quant_fp4_mxfp/),
  [`../operators/kv_cache_quant/`](../operators/kv_cache_quant/), [`../hardware/cdna4_mi350/`](../hardware/cdna4_mi350/).

## 3. Attention backend
Defaults are usually best (auto-select with `VLLM_ROCM_USE_AITER=1`): ROCM_AITER_FA for
MHA/GQA, ROCM_AITER_MLA for MLA. Hybrid/gated-delta models route attention to Triton kernels.
Pick + A/B per [`attention_backend_selection.md`](attention_backend_selection.md). Avoid
ROCM_ATTN on models with unsupported KV head dims (silent Triton fallback → big regression).

## 4. Parallelism (collective config)
- **TP-only within a single ≤8-GPU XGMI island**; add PP only beyond 8 GPUs / across nodes
  (`--tensor-parallel-size 8 --pipeline-parallel-size 2` for 2×8).
- Small models: N single-GPU instances usually out-throughput one `-tp N` instance.
- **DP on ROCm:** `VLLM_ALL2ALL_BACKEND="allgather_reducescatter"` +
  `--disable-nccl-for-dp-synchronization`.
- Collectives: **QuickReduce** (up to 3× allreduce on MI300X, no code change) / RCCL / MoRI
  (MI355X + sglang for cost-efficient MoE). See [`../operators/allreduce/`](../operators/allreduce/),
  [`../backends/mori_rccl/`](../backends/mori_rccl/).
> Note: the e2e *tuning* runs use **TP=1** for deterministic Amdahl accounting
> ([`optimize_e2e_model.md`](optimize_e2e_model.md)); collective config is tuned here at
> bring-up, separately.

## 5. Fused norms / MoE path / collectives (kernel levers)
- **Fused norms:** fused add+RMSNorm, fused RMSNorm-quant, LayerNorm/SiLU fp8 block quant —
  small Amdahl mass each → **stack**. [`../operators/fused_add_rmsnorm/`](../operators/fused_add_rmsnorm/),
  [`../operators/fused_norm_quant/`](../operators/fused_norm_quant/).
- **MoE:** aiter `fused_moe` auto-selects the quant kernel (moe-2stages, fp8-blockscale_g1u1,
  MXFP4 w4a4). Tune via `AITER_CONFIG_FMOE`. Watch the **AITER-gating bug** (expert mask
  gated on global AITER-on vs the actual matmul backend → wrong-expert routing when MXFP4
  falls through to emulation). [`../operators/fused_moe_grouped_gemm/`](../operators/fused_moe_grouped_gemm/),
  [`../backends/aiter/fmoe.md`](../backends/aiter/fmoe.md).

## 6. ACCURACY SMOKE TEST (do this BEFORE perf tuning)
Run gsm8k / lm_eval / a translation probe (greedy/temp=0, ≥10 prompts). This catches silent
quant-dequant corruption and MoE wrong-expert routing **before** you waste time tuning a
broken model (the Kimi-K2.5-MXFP4 gibberish and DeepSeek-V4 MoE-mask bugs both passed
"loads + serves" but failed accuracy). Quant always → accuracy-gate, never byte-parity.

## 7. GEMM DB tune (the banked lever)
Dense GEMM is usually the bulk of GPU time (~78% on Qwen3.5-27B). Run the aiter DB tune:
live `AITER_TUNE_GEMM=1` capture → gradlib → `AITER_CONFIG_GEMM_BF16` → verify
`is tuned on cu_num`>0. Banked **+2.23% e2e** on Qwen3.5-27B. Full recipe + traps:
[`gemm_tuning_workflow.md`](gemm_tuning_workflow.md).

## 8. Profile → optimize e2e
Once correct + the cheap levers are set, hand off to the full flow:
[`optimize_e2e_model.md`](optimize_e2e_model.md) (Profile → Strategize → ConfigSweep →
HeadKernel → Milestone → Finalize → Report → Validate). Profile with ROCm Compute Profiler
(FP4/FP6 counters, MI350X/MI355X support). See [`../profiling/`](../profiling/).

## Bring-up gotchas
- **Silent quant/MoE corruption** — always accuracy-smoke-test before tuning.
- **FNUZ vs E4M3FN** mismatch when porting checkpoints across gfx942/gfx950.
- **ROCm version drift** — pin to the model card's version (gfx950 MXFP4 regression).
- **MoE `device_gemm` errors** → `AITER_ONLINE_TUNE=1`, then disable AITER MoE, then full off.
- **Port flakiness** in bench harness (grpc=port+10000 can exceed 65535) → pin a low PORT.
- **Stretching small models with `-tp N`** when N independent instances are faster.

## Cross-links
- e2e flow: [`optimize_e2e_model.md`](optimize_e2e_model.md) · GEMM: [`gemm_tuning_workflow.md`](gemm_tuning_workflow.md)
- Attn: [`attention_backend_selection.md`](attention_backend_selection.md) · Backends: [`choosing_a_backend.md`](choosing_a_backend.md)
- Wire-in: [`integrating_a_new_kernel.md`](integrating_a_new_kernel.md) · Quant: [`../quantization/`](../quantization/)
- aiter integration + env table: [`../backends/aiter/integration.md`](../backends/aiter/integration.md)

## Sources
- AITER master switch, TP/PP/DP topology, FP8/FP4 quant, pre-quantized Quark, env vars: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html
- MI355X scaling / MoRI MoE: https://rocm.blogs.amd.com/artificial-intelligence/scaling-ai-inference/README.html
- gfx950 MXFP4 ROCm 7.1→7.2 regression (accuracy-test-first lesson): https://github.com/vllm-project/vllm/issues/36337
- GEMM DB +2.23%, port/fork-storm gotchas: `GEAK/e2e_workflow/knowledge/gemm_tuning/aiter_gemm_tuning.md`
