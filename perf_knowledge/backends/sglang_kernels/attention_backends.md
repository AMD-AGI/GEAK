---
title: SGLang attention backends on ROCm — flag → kernel map
kind: backend
backend: sglang_kernels
operator: attention_decode_paged
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode, both]
status: sota
updated: 2026-06-08
sources:
  - https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/attention_registry.py
  - https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/attention_backend.md
  - https://docs.sglang.io/platforms/amd_gpu.html
---

# SGLang attention backends (ROCm)

## TL;DR
SGLang picks an attention kernel via `--attention-backend <name>` (and the split
`--prefill-attention-backend` / `--decode-attention-backend`). On MI300X the live candidates are **AITER**
(default & fastest for supported MHA/MLA), **TileLang** (the AMD default in recent images), **Triton**
(universal fallback + Tier-C rewrite seam), and CK under AITER. Registry:
`python/sglang/srt/layers/attention/attention_registry.py` (`@register_attention_backend(...)`). The
op-unittest is the judge — bake off AITER vs TileLang vs Triton per model/regime.

## Flag → kernel map
| `--attention-backend` | file (`layers/attention/`) | kernel | MI300X fit |
|---|---|---|---|
| `aiter` | `aiter_backend.py` (`AiterAttnBackend`) | AITER `mha_batch_prefill`, `mla_decode_fwd`/`mla_prefill_fwd`, paged ragged KV | **fastest** for supported MHA (Llama/Qwen) & MLA (DeepSeek) |
| `tilelang` | TileLang-authored | TileLang attention (FlashMLA path) | **AMD default** in recent images; competitive, editable in TileLang |
| `triton` | `triton_backend.py` + `triton_ops/` | Triton FlashAttention | **fallback** + path to Triton rewrites |
| `aiter` + `SGLANG_USE_AITER_UNIFIED_ATTN=1` | `triton_ops/aiter_unified_attention.py` | AITER unified Triton (chunked prefill + decode in one kernel) | launch-bound small batch |
| `wave` | `wave_backend.py` | AMD **Wave** DSL attention | experimental |
| `dsa` / `nsa` / `dsv4` | `dsa_backend.py`, `nsa_backend.py`, `deepseek_v4_backend*.py` | DeepSeek sparse attn (indexer + flashMLA), `SGLANG_OPT_USE_AITER_INDEXER` | DeepSeek-V3.2/V4; ROCm coverage maturing |
| `fa3` | `flashattention_backend.py` | FlashAttention-3 | **Hopper** default; FA3 vision needs CUTLASS/TMA → **not** an MI300X path, use Triton vision attn |
| `flashmla`/`flashinfer_mla`/`cutlass_mla`/`trtllm_mla` | resp. files | CUDA-leaning MLA variants | avoid on MI300X unless verified |
| `torch_native` / `flex_attention` | `torch_native_backend.py`, `torch_flex_backend.py` | reference torch / FlexAttention | correctness reference |

## Practical ranking (bake-off order)
- **MHA decode (Llama/Qwen):** `aiter` → `tilelang` → `triton`
- **MHA prefill:** `aiter` → `tilelang` → `triton` (try unified-attn for launch-bound small batches)
- **MLA decode (DeepSeek):** `aiter` (+`SGLANG_ROCM_FUSED_DECODE_MLA=1`, `SGLANG_AITER_MLA_PERSIST=1`) →
  `tilelang` → `triton`
- **MLA prefill:** `aiter` → `tilelang` → `triton`
- **New/unsupported arch where AITER CK errors** (`device_gemm does not support this GEMM problem`, #16025)
  → `--attention-backend triton`.

## Knobs
- `SGLANG_AITER_MLA_PERSIST=1` (persistent-kernel MLA decode), `SGLANG_ROCM_FUSED_DECODE_MLA=1` (fused MLA
  decode + RoPE), `SGLANG_USE_AITER_UNIFIED_ATTN=1`, `SGLANG_AITER_FP8_PREFILL_ATTN`,
  `SGLANG_AITER_UNIFIED_VERIFY` (spec-decode), `--page-size N` (KV granularity),
  `--kv-cache-dtype fp8_e4m3` (accuracy gate; fnuz on MI300X).

## Dispatch surface to read when debugging
`AiterAttnBackend` imports (`aiter_backend.py` ~L37-49): `from aiter import (...)`, `from aiter.mla import
mla_decode_fwd, mla_prefill_fwd`, `from aiter.ops.triton.attention.unified_attention import
unified_attention` — that block is the literal "which AITER kernel ran" surface.

## Pitfalls
- `fa3` is a Hopper backend; selecting it on MI300X for vision falls off a real path — use Triton vision attn
  (Step-3 on Instinct blog).
- MLA accuracy: AITER MLA has shown eval regressions on some models — accuracy-gate when enabling.
- FP8 prefill attn uses scaled inputs; re-check parity (fnuz dialect on gfx942).

## Verify
Isolated FMHA/MLA bench at the served shape across `{aiter, tilelang, triton}`; greedy/temp=0 parity vs a
reference; confirm the backend banner in the log.

## Alternatives / cross-links
[overview.md](overview.md) · [where_kernels_live.md](where_kernels_live.md) · CK-Tile FMHA card:
`operators/attention_prefill_fmha/backends/ck.md` · operators `attention_prefill_fmha`,
`attention_decode_paged`, `mla_attention`.

## Sources
- Attention registry (source): https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/attention_registry.py
- Attention backend docs: https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/attention_backend.md
- SGLang AMD GPU docs (TileLang default, AITER): https://docs.sglang.io/platforms/amd_gpu.html
- AITER CK MoE/attn crash (#16025): https://github.com/sgl-project/sglang/issues/16025
- Step-3 on AMD Instinct (FA3 vision → Triton): https://rocm.blogs.amd.com/artificial-intelligence/step3-model/README.html
