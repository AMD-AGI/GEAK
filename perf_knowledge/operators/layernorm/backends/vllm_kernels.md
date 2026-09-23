---
title: layernorm on vllm_kernels — SOTA card
kind: sota_card
operator: layernorm
backend: vllm_kernels
gens: [gfx942, gfx950, gfx1151]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu
  - https://github.com/vllm-project/vllm/blob/main/csrc/torch_bindings.cpp
---

# layernorm × vllm_kernels

## TL;DR
vLLM uses LayerNorm mainly for **vision encoders / non-RMS models**; the decoder hot path is RMSNorm. Two
worlds as usual: native HIP (`csrc/layernorm_kernels.cu`) and AITER (via `model_executor/layers/`). On
MI300X with AITER on, AITER's CK/asm wins; native HIP is the editable default with `USE_AITER=0`.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| AITER `layernorm2d_fwd[_with_add]` (via layer) | `model_executor/layers/` | gfx942/950, bf16/fp16 | live when `VLLM_ROCM_USE_AITER=1` | MI300X serving default — [aiter.md](aiter.md) |
| native HIP LayerNorm | `csrc/layernorm_kernels.cu` | gfx942/950 | vectorized, bandwidth-bound | `USE_AITER=0` / Tier-C edit |

## Config space / knobs
- `VLLM_ROCM_USE_AITER=1` (+ norm gate) → AITER; `=0` → native HIP.
- Native: vector width, LDS-cached γ,β, int64 strides.
- fp8/int8 quant fusion via the AITER smoothquant/dynamicquant variants.

## Numerics / parity
fp32 μ/σ²; γ,β fp32-promote (#42325 class); biased variance; fnuz fp8 on gfx942. AITER vs native order →
greedy re-gate. See [../numerics.md](../numerics.md).

## Integration (rebind seam)
- Python: `model_executor/layers/layernorm.py` (LayerNorm class, `forward_hip`).
- Native HIP: `csrc/layernorm_kernels.cu` + `torch_bindings.cpp` (`_C::layer_norm`); rebuild to edit.
- torch.compile: AITER custom-op preserved; ROCm fusion pass for norm+quant.

## Pitfalls & anti-patterns
- ⚠ Image mismatch (`USE_AITER=1`, no aiter) → import error.
- ⚠ γ/β dtype regression #42325.
- ⚠ CUDA13/CCCL3 `cub::Sum` build break (#24464) — pin CCCL if cross-building.

## How to verify
rocprofv3 Top-N kernel name (AITER `*ck_*` vs native vs Triton); isolated bench; greedy parity.

## Alternatives / cross-links
[aiter.md](aiter.md) · [hip.md](hip.md) · [triton.md](triton.md) · [miopen.md](miopen.md) ·
[[backends/vllm_kernels/rocm_kernels]].

## Sources
- vLLM HIP layernorm kernels + bindings: https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu, https://github.com/vllm-project/vllm/blob/main/csrc/torch_bindings.cpp.
- AITER norm gate: perf_knowledge [[backends/vllm_kernels/overview]].

## On gfx1151 (RDNA3.5, Strix Halo)
`gfx1151` appears in `gens:` because this backend's **source is portable** to RDNA — it
compiles/JITs there with no vendor asset table. It is **not** a claim that anything on this
card was measured on RDNA: every perf number, ranking and tuning recipe above is CDNA
(gfx90a/942/950) evidence. Three differences bite before any of it transfers — WMMA not MFMA,
wave32 not wave64, and 40 CU behind a 32 MB MALL on ~229 GB/s shared LPDDR5X rather than HBM —
so tile shapes, occupancy targets and the roofline ceiling all move. Any `fp8_*` entry in
`dtypes:` above is CDNA-only: gfx1151 has **no fp8 matrix instruction and no block-scaled
FP4/FP6**, so an fp8 candidate there runs EMULATED — it passes correctness and loses
performance silently. See [`../../../hardware/rdna35_gfx1151/`](../../../hardware/rdna35_gfx1151/)
and MEASURE on the box.
