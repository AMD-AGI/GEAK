---
title: SGLang kernels on ROCm — overview (dispatch, env, where kernels live)
kind: backend
backend: sglang_kernels
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp4_e2m1, mxfp4, int8]
regimes: [prefill, decode, both]
status: sota
updated: 2026-06-08
sources:
  - https://github.com/sgl-project/sglang/blob/main/docs/platforms/amd_gpu.md
  - https://github.com/sgl-project/sglang/blob/main/docker/rocm.Dockerfile
  - https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/attention_registry.py
  - https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html
---

# SGLang kernels on ROCm / MI300X

## TL;DR
On MI300X almost every hot kernel in SGLang (attention, GEMM, MoE, RMSNorm, all-reduce) has **2–4
candidate implementations** — AITER (asm/CK/hipBLASLt), Triton, TileLang, CK, or a torch fallback. The
optimizer's job is to pick the fastest *correct* one per shape regime via flags+env. SGLang is
Amdahl-dominated by **GEMM** (~70–80% GPU time on dense models) → attention → MoE; spend in that order.
Master gate: `SGLANG_USE_AITER=1`. Attention backend selection is in
[attention_backends.md](attention_backends.md); the source-tree map is in
[where_kernels_live.md](where_kernels_live.md).

## Concepts — the kernel providers
| provider | what | engage |
|---|---|---|
| **AITER** (`ROCm/aiter`) | AMD tuned library: attention, MoE, GEMM, RMSNorm, custom-AR; runtime-dispatches CK/ASM/Triton/hipBLAS | `SGLANG_USE_AITER=1` (master gate) |
| **TileLang** (`tile-ai/tilelang`) | tile-DSL kernels; **default AMD attention backend**, pinned in `rocm.Dockerfile` | `--attention-backend tilelang` (default) |
| **sgl-kernel** | SGLang's own C++/HIP custom PyTorch ops (gemm/moe/attention/quant) | built via `sgl-kernel/setup_rocm.py` |
| **Triton** | editable, autotunable; the universal fallback + Tier-C rewrite path | `--attention-backend triton`, fused-MoE Triton |
| **CK** | composable-kernel instances under AITER | indirect (AITER) |
| **MoRI** | EP all-to-all + KV transfer (separate, pinned in Docker) | `--enable-deepep-moe` / mori all2all backend (see [../mori_rccl/mori_ep.md](../mori_rccl/mori_ep.md)) |

## The levers (top env vars)
- `SGLANG_USE_AITER=1` — master AITER switch (keep on MI300X; missing wheel → `ImportError`).
- `HSA_NO_SCRATCH_RECLAIM=1` — near-mandatory on MI300X (stops scratch-reclaim idle gaps/hangs).
- `TORCH_BLAS_PREFER_HIPBLASLT=1`, `HIP_FORCE_DEV_KERNARG=1`, `GPU_MAX_HW_QUEUES=2`.
- `SGLANG_USE_AITER_AR/AG` (custom AR/AG), `SGLANG_AITER_MLA_PERSIST`, `SGLANG_ROCM_FUSED_DECODE_MLA`,
  `SGLANG_USE_AITER_MOE_GU_ITLV`, `SGLANG_MOE_PADDING`, `SGLANG_MOE_CONFIG_DIR`. (Full table:
  [where_kernels_live.md].)

## Images
Use a prebuilt image (`lmsysorg/sglang:*-rocm*`, `rocm/sglang-staging:latest`) — it ships a matched AITER +
tuned hipBLASLt + CK + TileLang + MoRI. **Never** `pip install aiter` ad-hoc (ABI must match the image).
Build from source: `python sgl-kernel/setup_rocm.py install` then `pip install -e "python[all_hip]"`.

## Where the budget goes (Amdahl)
1. **GEMM** — hipBLASLt (default) / AITER GEMM / Triton; TunableOp warm pass is the cheap first move.
2. **Attention** — AITER / TileLang / Triton bake-off ([attention_backends.md](attention_backends.md)).
3. **MoE** — AITER CK 2-stage / Triton fused-MoE; EP all-to-all via MoRI/DeepEP.

## Pitfalls
- `SGLANG_USE_AITER=1` with no aiter wheel → `ImportError: aiter is required ...` (image mismatch tell).
- AITER CK MoE can crash under HIP-graph capture for novel shapes (`device_gemm does not support this GEMM
  problem`, issue #16025) → disable graph capture for that model or force Triton fused-MoE.
- **FP8 fnuz** on gfx942: MI300X uses fnuz FP8 (exponent bias off-by-one vs OCP) → a byte read in the wrong
  dialect is off by exactly 2×. Always accuracy-gate FP8 on MI300X.
- AITER custom AR has had segfaults (#1542) → `SGLANG_USE_AITER_AR=0` to fall back.

## Verify
- Confirm the backend banner in the server log; rocprofv3 Top-N kernels to see which provider ran.
- Greedy/temp=0 e2e parity after any backend swap (reduction order differs across AITER/Triton/CK/TileLang).

## Sources
- SGLang AMD GPU docs: https://github.com/sgl-project/sglang/blob/main/docs/platforms/amd_gpu.md
- SGLang ROCm Dockerfile (AITER/TileLang/MoRI/Mooncake pins): https://github.com/sgl-project/sglang/blob/main/docker/rocm.Dockerfile
- AITER ROCm blog: https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html
- Deeper env/flag reference: perf_knowledge where_kernels_live.md
