---
title: vLLM kernels on ROCm — overview (two kernel worlds, env hierarchy)
kind: backend
backend: vllm_kernels
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp4_e2m1, mxfp4, int8]
regimes: [prefill, decode, both]
status: sota
updated: 2026-06-08
sources:
  - https://github.com/vllm-project/vllm/blob/main/vllm/envs.py
  - https://github.com/vllm-project/vllm/blob/main/vllm/platforms/rocm.py
  - https://github.com/vllm-project/vllm/tree/main/csrc/rocm
  - https://rocm.blogs.amd.com/software-tools-optimization/vllm-omni/README.html
---

# vLLM kernels on ROCm / MI300X (V1)

## TL;DR
ROCm is a **first-class** vLLM platform. There are **two kernel worlds**: (1) vLLM's **own hand-written
HIP custom ops** in `csrc/rocm/` (custom PagedAttention, skinny GEMMs — [rocm_kernels.md](rocm_kernels.md)),
and (2) **AITER** tuned kernels wired in via `vllm/_aiter_ops.py` ([aiter_integration.md](aiter_integration.md)).
The optimizer picks per shape via the `--attention-backend` enum and the `VLLM_ROCM_USE_AITER*` env
hierarchy. All guidance is **V1** (V0 is gone). Master gate: `VLLM_ROCM_USE_AITER=1` (default 0).

## Concepts — the env hierarchy (`vllm/envs.py`)
`VLLM_ROCM_USE_AITER` is the **master switch (default 0)**; every `VLLM_ROCM_USE_AITER_*` is gated by it.
Key sub-flags (defaults as of 2026-06 main):
| env | default | gates |
|---|---|---|
| `VLLM_ROCM_USE_AITER` | **0** | master (turn on first) |
| `VLLM_ROCM_USE_AITER_LINEAR` | 1 | AITER quant ops + GEMM for linears |
| `VLLM_ROCM_USE_AITER_MOE` | 1 | AITER fused-MoE |
| `VLLM_ROCM_USE_AITER_RMSNORM` | 1 | AITER RMSNorm (+fused add/quant) |
| `VLLM_ROCM_USE_AITER_MLA` | 1 | DeepSeek MLA |
| `VLLM_ROCM_USE_AITER_MHA` | 1 | MHA; `0` → Triton/ROCM_ATTN |
| `VLLM_ROCM_USE_AITER_FP4BMM` | 1 | FP4 batched matmul — **CRASHES MI300X (no FP4 HW)**, set `0` |
| `VLLM_ROCM_USE_SKINNY_GEMM` | 1 | vLLM's own skinny GEMM (`csrc/rocm/skinny_gemms.cu`) |
| `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT` | 0 | ROCM_AITER_FA only; `1` for concurrency ≥32 |
| `VLLM_ROCM_CUSTOM_PAGED_ATTN` | 1 | vLLM custom paged-attn decode |

(Full table: [rocm_kernels.md].)

## The levers
- Recommended ROCm env block: `TORCH_BLAS_PREFER_HIPBLASLT=1 HIP_FORCE_DEV_KERNARG=1 SAFETENSORS_FAST_GPU=1
  VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_FP4BMM=0` (+ `NCCL_MIN_NCHANNELS=112` multi-GPU).
- Attention: `--attention-backend {ROCM_AITER_FA, ROCM_AITER_MLA, ROCM_ATTN, TRITON_ATTN}` bake-off
  ([rocm_kernels.md](rocm_kernels.md)).
- GEMM: hipBLASLt (default) / vLLM skinny GEMM / AITER GEMM / TunableOp warm pass.
- MoE/EP: `--data-parallel-size N --enable-expert-parallel`; all2all backend (MoRI/DeepEP,
  [../mori_rccl/mori_ep.md](../mori_rccl/mori_ep.md)).

## Images
**Use upstream `vllm/vllm-openai-rocm`** (official since ~Jan 20 2026). `rocm/vllm`, `rocm/vllm-dev` are
**deprecated** (Jan 2026). ROCm 7.0+ (MI300X fully supported since 7.0.0); engine **V1 only**.

## Quant traps (MI300X)
- **FP4 crashes gfx942** (`FP4BMM` defaults 1, MI300X has no FP4 HW; issue #34641) → `VLLM_ROCM_USE_AITER_
  FP4BMM=0`. FP4 ASM paths are CDNA4 (MI350/355) only.
- **FP8 fnuz**: gfx942 uses fnuz FP8 (bias off-by-one vs OCP) → wrong-dialect read off by exactly 2×;
  `VLLM_ROCM_FP8_PADDING=1` required for the fast FP8 linear path.
- **AITER MLA accuracy**: caused gsm8k loss with Kimi-K2 DP2TP4 (aiter #1455) — accuracy-gate.

## Verify
rocprofv3 Top-N → kernel name → world: `paged_attention_ll4mi_*` / `wvSplitK*` / `LLMM1` = vLLM custom HIP;
`*ck_*`/`fmha_*` = AITER/CK; Triton carries the Python name. Greedy/temp=0 parity after a backend swap.

## Sources
- vLLM env vars: https://github.com/vllm-project/vllm/blob/main/vllm/envs.py
- vLLM ROCm platform dispatch: https://github.com/vllm-project/vllm/blob/main/vllm/platforms/rocm.py
- vLLM ROCm custom HIP kernels: https://github.com/vllm-project/vllm/tree/main/csrc/rocm
- ROCm first-class in vLLM (FP8/FP4/MXFP4): https://rocm.blogs.amd.com/software-tools-optimization/vllm-omni/README.html
- Deeper env/flag reference: perf_knowledge rocm_kernels.md
