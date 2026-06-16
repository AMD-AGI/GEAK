---
title: quant_dequant_fp8 on vllm_kernels — SOTA card
kind: sota_card
operator: quant_dequant_fp8
backend: vllm_kernels
gens: [gfx942, gfx950]
dtypes: [fp8_e4m3_fnuz, fp8_e4m3]
regimes: [prefill, decode, both]
status: sota
updated: 2026-06-08
sources:
  - vllm-project/vllm@HEAD:csrc/quantization/fp8/common.cu
  - vllm-project/vllm@HEAD:csrc/quantization/fp8/amd/quant_utils.cuh
  - vllm-project/vllm@HEAD:csrc/quantization/compressed_tensors/int8_quant_kernels.cu
  - https://github.com/vllm-project/vllm/blob/main/vllm/envs.py
  - https://github.com/vllm-project/vllm/blob/main/csrc/quantization/fp8/common.cu
---

# quant_dequant_fp8 × vllm_kernels

## TL;DR
vLLM ships its **own** HIP FP8 quant kernels in `csrc/quantization/fp8/common.cu` (per-tensor static,
per-tensor dynamic, per-token dynamic) plus int8 kernels in
`compressed_tensors/int8_quant_kernels.cu`. These run when AITER's quant path is off, and they are the
canonical path for **`compressed-tensors` / Quark-exported** FP8 checkpoints. The defining ROCm detail:
the **dynamic** path clamps to **224.0** (not the 448 OCP max, nor even the 240 FNUZ max) to avoid
accuracy loss, and the FP8 type is `c10::Float8_e4m3fnuz` on ROCm (selected via `#ifdef USE_ROCM`). With
`VLLM_ROCM_USE_AITER_LINEAR=1` the AITER quant path generally supersedes these for the linear hot path;
vLLM's own kernels remain the correct, portable fallback — and the only path that already speaks the
compressed-tensors scheme.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `scaled_fp8_quant_kernel` (static per-tensor) | `csrc/quantization/fp8/common.cu:184` | gfx942/950, e4m3fnuz | vectorized (`scaled_fp8_conversion_vec`, float4 lanes) | calibrated weights/activations |
| `dynamic_per_token_scaled_fp8_quant_kernel` | `:198` | gfx942/950 | per-row amax reduce + scale | activations, no calibration |
| `segmented_max_reduction` (per-tensor dynamic amax) | `:69` | gfx942/950 | `atomicMaxFloat` into a global scale | per-tensor dynamic |
| `static_scaled_int8_quant_kernel` / `dynamic_scaled_int8_quant_kernel` | `compressed_tensors/int8_quant_kernels.cu` | gfx942/950, int8 | round-to-nearest-even, per-token | int8 W8A8 / SmoothQuant |

### SOTA detail — the 224.0 dynamic cap (ROCm branch, `common.cu`)
```cpp
// vllm-project/vllm: csrc/quantization/fp8/common.cu (ROCm path)
#ifdef USE_ROCM
  using FP8_TYPE = c10::Float8_e4m3fnuz;
  // FNUZ max is 240, but dynamic quant clamps the scale denominator to 224.0:
  // empirically 240 hurts accuracy → FP8_E4M3_MAX = 224.0f
  static constexpr float FP8_E4M3_MAX = 224.0f;
#else
  using FP8_TYPE = c10::Float8_e4m3fn;       // OCP, max 448
  static constexpr float FP8_E4M3_MAX = 448.0f;
#endif
// is_scale_inverted=true → store 1/s so the per-element op is a multiply
```
So the dynamic scale on ROCm is `scale = amax / 224.0` — using 240 or 448 here either under-utilizes the
range or saturates outliers.

## Config space / knobs
| knob | values | effect |
|---|---|---|
| `is_scale_inverted` | true/false | store `1/s` so per-element op is a multiply (`scaled_fp8_conversion<true>`) |
| vector width | `float4` lanes via `scaled_fp8_conversion_vec` | coalesced FP8 store |
| `VLLM_ROCM_FP8_PADDING` | 1 | pad weights for the fast FP8 linear on gfx942 |
| `FP8_E4M3_MAX` | 224 (ROCm) / 448 (CUDA) | dynamic-quant denominator |
| granularity | per-tensor / per-token | from the quant scheme (compressed-tensors / Quark export) |

## Measured performance
| config | metric | value @ hw / ver / date | source |
|---|---|---|---|
| static per-tensor FP8 quant | bound | vectorized `float4` store, HBM-bound | `common.cu:184` |
| dynamic per-tensor | amax method | two-pass: `segmented_max_reduction` (atomicMax) then quant | `common.cu:69,198` |
| ROCm dynamic-quant cap | constant | `FP8_E4M3_MAX = 224.0f` (240 hurts accuracy) | `common.cu`, web-confirmed |
| vs AITER linear | who wins | AITER linear path supersedes when `VLLM_ROCM_USE_AITER_LINEAR=1` | `vllm/envs.py` |

> vLLM kernels are correctness-portable, not the throughput SOTA on ROCm — the AITER fused path wins the
> linear hot loop. Bench on your model; no vendor speedup asserted.

## Numerics / parity
- **`FP8_TYPE = c10::Float8_e4m3fnuz` on ROCm**; **`FP8_E4M3_MAX = 224.0f`** for dynamic quant (the
  in-source rationale: 240 hurts accuracy). FNUZ ("Finite, NaN, Unsigned-Zero") has no Inf / no −0 and
  one extra exponent of range vs OCP.
- FNUZ↔OCP dialect must match the checkpoint. OCP-quantized weights read on gfx942 are off by ~2×
  ([[../../../quantization/fnuz_vs_ocp]]).
- int8 path: round-to-nearest-even, per-token or per-tensor scale; gate on task accuracy.
- Gate on task accuracy, never byte parity → [[numerics.md]].

## Integration (rebind seam)
- `torch.ops._C.static_scaled_fp8_quant` / `torch.ops._C.dynamic_per_token_scaled_fp8_quant` /
  `dynamic_scaled_int8_quant` (registered in `csrc/torch_bindings.cpp`).
- In V1, the linear quant path is selected by the model's quant config plus `VLLM_ROCM_USE_AITER_LINEAR`.
- Editing the `.cu` is a **Tier-C rewrite** requiring a full vLLM rebuild — overlay via env/quant-config
  first. → [[../../../reference/env_vars]].

## Pitfalls & anti-patterns
- **Assuming 240 / 448 max on ROCm dynamic quant — it is 224.** Hard-coding 448 saturates outliers;
  240 measurably hurts accuracy (the in-source reason for 224).
- **FNUZ↔OCP mismatch** with a Quark / compressed-tensors export → ~2× error.
- `VLLM_ROCM_USE_AITER_FP4BMM=1` **crashes gfx942** (no FP4 HW) — unrelated to fp8 but a common
  co-occurring trap; keep it 0 → [[operators/quant_fp4_mxfp]].
- Forgetting `#ifdef USE_ROCM` when reading the source — the CUDA branch (448, `e4m3fn`) is *not* what
  runs on MI300/MI350.

## How to verify
- rocprofv3 → confirm `scaled_fp8_quant_kernel` / `dynamic_per_token_scaled_fp8_quant_kernel` ran (not an
  AITER or Triton symbol).
- Round-trip error + gsm8k parity on a compressed-tensors/Quark FP8 checkpoint.
- Inspect the scale: dynamic scale should equal `amax/224` on ROCm.

## Alternatives / cross-links
[aiter.md](aiter.md) (live HIP path) · [hip.md](hip.md) · [triton.md](triton.md) ·
[numerics.md](../numerics.md) · [overview.md](../overview.md) · [[../../../quantization/fnuz_vs_ocp]] ·
[[../../../quantization/calibration_and_quark]] · [[operators/quant_fp4_mxfp]].

## Worked example
Static per-tensor FP8 quant of a calibrated weight on gfx942:
```python
import torch
w = torch.randn(8192, 8192, dtype=torch.bfloat16, device="cuda")
scale = torch.tensor([w.abs().max() / 224.0], device="cuda")   # ROCm dynamic cap = 224
qw = torch.empty_like(w, dtype=torch.float8_e4m3fnuz)          # FNUZ on ROCm
torch.ops._C.static_scaled_fp8_quant(qw, w, scale)
```
Using `torch.float8_e4m3fn` (OCP, max 448) here would mis-scale on gfx942 — the FNUZ type and the 224 cap
are both required.

## Sources
- vLLM FP8 quant (224 cap, fnuz type, per-token, segmented max):
  `vllm-project/vllm@HEAD:csrc/quantization/fp8/common.cu`
  (`segmented_max_reduction:69`, `scaled_fp8_quant_kernel:184`,
  `dynamic_per_token_scaled_fp8_quant_kernel:198`).
- ROCm FP8 helpers: `vllm-project/vllm@HEAD:csrc/quantization/fp8/amd/quant_utils.cuh`.
- int8 W8A8 kernels: `vllm-project/vllm@HEAD:csrc/quantization/compressed_tensors/int8_quant_kernels.cu`.
- 224 vs 240 vs 448 cap, `Float8_e4m3fnuz` on ROCm:
  https://github.com/vllm-project/vllm/blob/main/csrc/quantization/fp8/common.cu
- Env gates (`VLLM_ROCM_USE_AITER_LINEAR`, `VLLM_ROCM_FP8_PADDING`):
  https://github.com/vllm-project/vllm/blob/main/vllm/envs.py
