---
title: FlashAttention-ROCm — CK (Composable Kernel) backend
kind: backend
gens: [gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16]
regimes: [prefill, decode, training]
status: competitive
updated: 2026-06-08
sources:
  - https://github.com/Dao-AILab/flash-attention
  - https://github.com/ROCm/composable_kernel
  - https://rocm.blogs.amd.com/software-tools-optimization/ck-tile-flash/README.html
---

# FlashAttention-ROCm — CK backend

## TL;DR
The **CK (Composable Kernel) backend is the default** in ROCm FlashAttention. Kernels come from
`ROCm/composable_kernel` (git submodule), built into the wheel automatically — no env flag needed.
Mature, fp16/bf16, **head_dim ≤ 256** forward and backward, MI200/MI250/MI300/MI355X + RDNA3/4. It is the
backend you select (`VLLM_USE_TRITON_FLASH_ATTN=0`) when you need **sliding-window** half-precision
attention or to dodge Triton compilation issues (e.g. Phi3V shared-memory overflow). It does **not** do
fp8 or arbitrary head dims — use the Triton backend for those.

## Concepts
- **ck_tile FMHA.** The CK backend implements FlashAttention-2 via CK's tile-programming framework
  (ck_tile) — the same templates AMD documents in the "CK-Tile FlashAttention" blog. Templated over tile
  shape, head dim, causal/varlen, dtype.
- **Default path.** No env flag at runtime; built from the `composable_kernel` submodule at install. If you
  don't set `FLASH_ATTENTION_TRITON_AMD_ENABLE`, you are on CK.
- **Capability envelope:** fp16 / bf16 only; **head_dim ≤ 256** (fwd + bwd). RDNA3 has **no backward**;
  RDNA4 backward only with `deterministic=False`.

## The levers
- **Select CK** in vLLM: `export VLLM_USE_TRITON_FLASH_ATTN=0` (turns off Triton FA → CK, else SDPA).
- **Build arch list:** `FX_GFX_ARCHS=gfx90a;gfx942` (default = MI200 + MI300); `FA_BRANCH` pins the
  ROCm/flash-attention CK branch.
- **2-GEMM fusion epilogue:** for any kernel fusing 2 GEMMs (FlashAttention is one), AMD's MI300X guide
  recommends `OPTIMIZE_EPILOGUE=1` (store in MFMA layout, skip reblock).
- **head_dim ≤ 256** — above it, CK can't run; switch to Triton.

## When to use CK
- **Sliding-window attention** at half precision (Mistral, Mixtral, Qwen2) — Triton SWA is WIP, so CK is
  the path (`VLLM_USE_TRITON_FLASH_ATTN=0`).
- **Triton compile failures** — e.g. ROCm Triton FA overflows shared memory for Phi3VForCausalLM; disable
  Triton (→ CK).
- **Default, stable fp16/bf16 FMHA** within head_dim ≤ 256 where you don't need fp8 / ALiBi / arbitrary
  head dim.
- **Prefill / training** where the mature CK backward is wanted (and not RDNA3).

## Pitfalls
- **No fp8, no arbitrary head dim, no ALiBi/rotary/paged** in the core CK FA path — those are Triton-side.
- **head_dim > 256 unsupported** — hard limit.
- **RDNA backward gaps** (RDNA3 none; RDNA4 `deterministic=False` only) — irrelevant on Instinct but note
  for portability.
- **Build it in:** the CK kernels must be compiled into the wheel (`BUILD_FA=0` drops FA entirely → SDPA).
- **vLLM V1 backend selection** may not honor the legacy flag — verify the active backend from logs.

## Verify
- `pytest tests/test_flash_attn_ck.py` (CK correctness).
- vLLM log confirms CK (not Triton) when `VLLM_USE_TRITON_FLASH_ATTN=0`.
- Micro-bench CK vs Triton at your `(B,H,S,D,causal,dtype)`; for decode also compare AITER FA.

## Alternatives / cross-links
[overview.md](overview.md) · [triton_backend.md](triton_backend.md) · CK library card
(`backends/composable_kernel_lib/`) · attention operators
(`operators/attention_prefill_fmha/`, `operators/attention_decode_paged/`).

## Sources
- FlashAttention ROCm README (CK default, head_dim ≤ 256, fp16/bf16, GPUs, RDNA backward gaps):
  https://github.com/Dao-AILab/flash-attention
- Composable Kernel repo (CK FMHA kernels / submodule): https://github.com/ROCm/composable_kernel
- CK-Tile FlashAttention blog (ck_tile FMHA design): https://rocm.blogs.amd.com/software-tools-optimization/ck-tile-flash/README.html
- vLLM ROCm install (`VLLM_USE_TRITON_FLASH_ATTN=0` → CK; SWA/Phi3V; `FX_GFX_ARCHS`, `FA_BRANCH`):
  https://docs.vllm.ai/en/v0.6.5/getting_started/amd-installation.html
- 2-GEMM fusion `OPTIMIZE_EPILOGUE=1`: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
