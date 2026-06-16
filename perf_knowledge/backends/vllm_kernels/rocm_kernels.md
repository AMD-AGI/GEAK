---
title: vLLM's own ROCm HIP kernels — custom PagedAttention & skinny GEMM
kind: backend
backend: vllm_kernels
operator: attention_decode_paged
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [decode, prefill, both]
status: sota
updated: 2026-06-08
sources:
  - https://github.com/vllm-project/vllm/tree/main/csrc/rocm
  - https://github.com/vllm-project/vllm/blob/main/vllm/platforms/rocm.py
  - https://github.com/vllm-project/vllm/blob/main/csrc/rocm/torch_bindings.cpp
---

# vLLM's own ROCm HIP kernels (`csrc/rocm/`)

## TL;DR
Beyond AITER, vLLM ships its **own hand-written HIP kernels** in `csrc/rocm/`: a custom **PagedAttention**
decode kernel (`attention.cu`, the `ROCM_ATTN` backend) and **skinny GEMMs** for decode (`skinny_gemms.cu`:
`LLMM1`, `wvSplitK`, `wvSplitKrc`, `wvSplitKQ`). These are the **editable HIP source** for a Tier-C rewrite
and are strong on **decode** (M=batch) without AITER. The authoritative op list is
`csrc/rocm/torch_bindings.cpp`. Operators: `attention_decode_paged`, `skinny_gemv_decode`, `scaled_quant_gemm`.

## Attention — the ROCM_ATTN path
| backend | file (`v1/attention/backends/`) | kernel | fit |
|---|---|---|---|
| `ROCM_AITER_FA` | `rocm_aiter_fa.py` | AITER flash-attn (prefill+decode, KV shuffle/gather) | **default MHA** |
| `ROCM_AITER_MLA` | `mla/rocm_aiter_mla.py` | AITER MLA decode | **default DeepSeek MLA** |
| `ROCM_ATTN` | `rocm_attn.py` | **vLLM custom HIP paged-attn** (`csrc/rocm/attention.cu`) + `VLLM_ROCM_CUSTOM_PAGED_ATTN` | strong decode, no AITER needed |
| `TRITON_ATTN` | `triton_attn.py` | Triton unified attention | universal fallback |

vLLM's PagedAttention HIP kernel names to grep in a profile:
- `paged_attention_ll4mi_QKV_mfma16_kernel` — MFMA-16 main path
- `paged_attention_ll4mi_QKV_mfma4_kernel` — MFMA-4 small-head path
- `paged_attention_ll4mi_reduce_kernel` — cross-block softmax reduce

Bound by `__launch_bounds__`, templated on `BLOCK_SIZE`, KV dtype, FP8 KV. Exposed via
`csrc/rocm/torch_bindings.cpp` (`rocm_ops.def("paged_attention", ...)`).

Dispatch order on gfx942 (`vllm/platforms/rocm.py` `get_attn_backend_cls`, ~L382-408): ROCM_ATTN →
ROCM_AITER_UNIFIED_ATTN → TRITON_ATTN, with AITER MLA/MHA inserted when
`rocm_aiter_ops.is_mla_enabled()/is_mha_enabled()`.

## Skinny GEMM — `csrc/rocm/skinny_gemms.cu`
For **decode** the GEMM is skinny (M=batch); vLLM's own kernels often beat a generic library kernel:
| op | use |
|---|---|
| `LLMM1` | skinny matmul (small M) |
| `wvSplitK`, `wvSplitKrc` | split-K skinny GEMM |
| `wvSplitKQ` | FP8 split-K skinny GEMM |

Engage with `VLLM_ROCM_USE_SKINNY_GEMM=1` (default). `csrc/rocm/q_gemm_rdna3.cu` is RDNA3 quantized GEMM
(not MI300X). The full editable op list is in `torch_bindings.cpp`: `LLMM1`, `wvSplitK`, `wvSplitKrc`,
`wvSplitKQ`, `paged_attention`.

## Practical ranking (op-unittest decides)
- **MHA decode:** `ROCM_AITER_FA` → `ROCM_ATTN` (custom HIP) → `TRITON_ATTN`
- **MHA prefill:** `ROCM_AITER_FA` → `TRITON_ATTN`
- **MLA decode:** `ROCM_AITER_MLA` → `ROCM_AITER_TRITON_MLA` → `TRITON_MLA`
- **AITER missing/erroring for a new shape:** `TRITON_ATTN` (slower but correct), or `ROCM_ATTN` for decode.

## Numerics / parity
fp32 online-softmax accumulate in paged-attn; FP8 KV uses scaled reads (fnuz on gfx942 — wrong dialect off
by 2×). Custom-HIP vs AITER vs Triton reduction order differs → re-check greedy/temp=0 parity.

## Integration (rebind seam)
- Select via `--attention-backend ROCM_ATTN`; the HIP source (`attention.cu`, `skinny_gemms.cu`) is the
  Tier-C edit seam (rebuild vLLM after editing). Autotune MFMA `matrix_instr_nonkdim`, `waves_per_eu`,
  split-K when rewriting.
- `torch_bindings.cpp` is the registration surface (`rocm_ops.def(...)`).

## Pitfalls
- `V0`-era vars (`VLLM_USE_TRITON_FLASH_ATTN`, `VLLM_USE_ROCM_FP8_FLASH_ATTN`) are **silently ignored** on
  V1 — selection is the `--attention-backend` enum + AITER hierarchy.
- Editing `csrc/rocm/*.cu` needs a vLLM rebuild; not a Python-only change.
- FP8 KV is an accuracy gate; fnuz re-check on MI300X.

## Verify
rocprofv3 kernel-trace → confirm `paged_attention_ll4mi_*` / `wvSplitK*` actually ran (not a Triton
fallback); isolated decode bench vs `ROCM_AITER_FA`/`TRITON_ATTN` at the served batch.

## Alternatives / cross-links
[overview.md](overview.md) · [aiter_integration.md](aiter_integration.md) · CK-Tile FMHA:
`operators/attention_prefill_fmha/backends/ck.md` · operators `attention_decode_paged`,
`skinny_gemv_decode`.

## Sources
- vLLM ROCm custom HIP kernels (attention.cu, skinny_gemms.cu): https://github.com/vllm-project/vllm/tree/main/csrc/rocm
- ROCm platform dispatch (backend order, gfx9 list): https://github.com/vllm-project/vllm/blob/main/vllm/platforms/rocm.py
- ROCm op registration (torch_bindings.cpp): https://github.com/vllm-project/vllm/blob/main/csrc/rocm/torch_bindings.cpp
- Deeper backend/env reference: perf_knowledge aiter_integration.md
