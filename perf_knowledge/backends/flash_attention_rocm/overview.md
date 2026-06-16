---
title: FlashAttention-ROCm — two-backend FMHA (CK default + Triton)
kind: backend
gens: [gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3]
regimes: [prefill, decode, training]
status: competitive
updated: 2026-06-08
sources:
  - https://github.com/Dao-AILab/flash-attention
  - https://github.com/ROCm/flash-attention
  - https://blog.vllm.ai/2026/02/27/rocm-attention-backend.html
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# FlashAttention-ROCm (`fa_rocm`)

## TL;DR
The ROCm build of FlashAttention (Dao-AILab/flash-attention, ROCm/flash-attention fork) ships **two
backends implementing FlashAttention-2**: **Composable Kernel (CK)** — the **default** — and **Triton**
(kernels supplied by the **aiter** submodule). Selection is by env: `FLASH_ATTENTION_TRITON_AMD_ENABLE`
at the FA level, and `VLLM_USE_TRITON_FLASH_ATTN` inside vLLM. **CK**: fp16/bf16, **head_dim ≤ 256**,
MI200/MI300/MI355X + RDNA3/4, no fused dropout/ALiBi extras. **Triton**: fp16/bf16/fp32 + fp8 (FA-v3
interface), **arbitrary head dim**, causal/varlen/MQA-GQA/dropout/rotary/ALiBi/paged. On vLLM, Triton FA
is the **default on ROCm**; set `VLLM_USE_TRITON_FLASH_ATTN=0` to fall back to CK (needed for
**sliding-window** models). For decode/MLA, the newest AITER backends often beat both. Details:
[ck_backend.md](ck_backend.md), [triton_backend.md](triton_backend.md).

## Concepts
- **Two backends, one package.** `Dao-AILab/flash-attention` (ROCm) exposes the FA-2 API over either CK or
  Triton; the choice is a build-time + runtime env flag, not separate packages.
- **CK backend** = `ROCm/composable_kernel` (git submodule) generates the kernels; built into the wheel by
  default. The mature, default path.
- **Triton backend** = kernels from `ROCm/aiter` (`third_party/aiter` submodule, auto-installed). Enabled
  by `FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"` at install **and** runtime. More feature-rich, supports
  fp8 and arbitrary head dims.
- **Framework selectors:** vLLM uses `VLLM_USE_TRITON_FLASH_ATTN` (default Triton on ROCm; `=0` → CK or
  SDPA). Newer vLLM V1 adds dedicated backends (`TRITON_ATTN`, `ROCM_ATTN`, `ROCM_AITER_FA`, MLA variants).

## Backend comparison
| aspect | CK (default) | Triton |
|---|---|---|
| kernels from | `ROCm/composable_kernel` | `ROCm/aiter` |
| enable | default | `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` (install + runtime) |
| GPUs | MI200x/250x/300x/355x + RDNA3/4 | CDNA (MI200/MI300) + RDNA |
| dtypes | fp16, bf16 | fp16, bf16, fp32, **fp8** (FA-v3 iface) |
| head dim | **≤ 256** (fwd + bwd) | arbitrary |
| backward | yes (RDNA3 none; RDNA4 only `deterministic=False`) | fwd + bwd |
| features | core FA-2 | causal, varlen, MQA/GQA, dropout, rotary, **ALiBi**, paged; SWA = WIP |

## The levers
- **Pick the backend by feature need:** sliding-window / fp32 fallback / Phi3V shared-mem-overflow →
  **CK** (`VLLM_USE_TRITON_FLASH_ATTN=0`); fp8 / arbitrary head dim / ALiBi / rotary / paged → **Triton**.
- **head_dim ≤ 256** is the CK hard limit — models above it need the Triton backend.
- **Triton autotune:** `FLASH_ATTENTION_TRITON_AMD_AUTOTUNE=TRUE` (one-time warmup) or pin a config with
  `FLASH_ATTENTION_FWD_TRITON_AMD_CONFIG_JSON`. See [triton_backend.md](triton_backend.md).
- **Build args** (vLLM/source): `FX_GFX_ARCHS=gfx90a;gfx942` (CK arch list), `BUILD_TRITON=1`,
  `FA_BRANCH`, `BUILD_FA=0` to drop FA entirely (→ SDPA).
- For **decode / paged / MLA**, evaluate the AITER backends (`ROCM_AITER_FA`, AITER MLA) — vLLM reports
  **1.2–4.4× higher TPS** vs the generic FA paths.

## Pitfalls
- **CK has no sliding-window** → Mistral/Mixtral/Qwen2 SWA at half precision need CK only via the SWA-
  supporting path; the **Triton** backend's SWA is **WIP** — so for SWA you generally set
  `VLLM_USE_TRITON_FLASH_ATTN=0` (CK). Confirm per model/version.
- **vLLM V1 may ignore the legacy flag** — users report "Using Triton Attention backend on V1 engine" even
  with `VLLM_USE_TRITON_FLASH_ATTN=0`; on V1 select the backend explicitly.
- **`ROCM_ATTN` decode fallback cliff:** 2.7–4.4× slower TPS when a model's KV head size is unsupported by
  HIP paged attention (falls back to Triton decode).
- **fp32 → Triton/SDPA only** (CK FA doesn't do fp32).
- **Backend must be enabled at install too** for Triton (not just runtime), or the kernels aren't built.

## Verify
- Confirm the active backend from vLLM/FA logs (e.g. "Using Triton Attention backend"); don't trust the
  env flag alone on V1.
- Per-shape micro-bench CK vs Triton at your `(B,H,S,D, causal, dtype)`; for decode also bench AITER FA.
- Tests: CK `pytest tests/test_flash_attn_ck.py`; Triton
  `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE pytest tests/test_flash_attn_triton_amd.py`.

## Sources
- FlashAttention ROCm README (two backends, env flags, head_dim ≤ 256, feature table, build):
  https://github.com/Dao-AILab/flash-attention ; ROCm fork https://github.com/ROCm/flash-attention
- vLLM ROCm attention backends (Triton default, `VLLM_USE_TRITON_FLASH_ATTN`, AITER 1.2–4.4×, V1 backends):
  https://blog.vllm.ai/2026/02/27/rocm-attention-backend.html
- SWA / Phi3V CK fallback, build args: vLLM ROCm install docs
  https://docs.vllm.ai/en/v0.6.5/getting_started/amd-installation.html
- MI300X attention tuning (2-GEMM fusion, OPTIMIZE_EPILOGUE):
  https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- backend detail: [ck_backend.md](ck_backend.md) · [triton_backend.md](triton_backend.md)
