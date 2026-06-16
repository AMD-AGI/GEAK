---
title: quant_fp4_mxfp on aiter — SOTA card
kind: sota_card
operator: quant_fp4_mxfp
backend: aiter
gens: [gfx950]
dtypes: [mxfp4, mxfp6, fp4_e2m1]
regimes: [prefill, decode, both]
status: sota
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/quant.py
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/utility/fp4_utils.py
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/quant/fused_mxfp4_quant.py
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
  - https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
---

# quant_fp4_mxfp × aiter

## TL;DR
aiter is the live MXFP path. `per_1x32_f4_quant` (with `_hip` / `_triton` variants) does the per-32-block
E8M0 MXFP4 cast with an optional HW **scale shuffle**; `per_1x32_f4_quant_for_dot_scaled` preps both
operands (LHS pack_dim=-1, RHS pack_dim=0) for `tl.dot_scaled`; `per_1x32_f8_scale_f8_quant` is the MXFP8
(block-FP8 + E8M0) variant; and the fused `fused_*_mxfp4_quant` family removes the standalone pass. The
MXFP4 tensor + E8M0 scales feed aiter's block-scaled GEMM (`gemm_a4w4`) and A4W4 MoE. **gfx950-only for a
HW win** — on gfx942 (CDNA3) there is no FP4 matrix-core support, so FP4 is pure simulation with no
speedup (and `VLLM_ROCM_USE_AITER_FP4BMM=1` crashes).

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `per_1x32_f4_quant(shuffle=True)` | `aiter/ops/quant.py:75` | gfx950, mxfp4 | per-block amax → e8m0 → shuffle | feeds HW block-scaled MFMA |
| `per_1x32_f4_quant_hip` → `dynamic_per_group_scaled_quant_fp4` | `aiter/ops/quant.py:359` | gfx950 | HIP group-32 cast, padded shuffle alloc | the HIP backend |
| `per_1x32_f4_quant_for_dot_scaled` | `:146` | gfx950 | LHS pack_dim=-1 / RHS pack_dim=0 | `tl.dot_scaled` GEMM |
| `per_1x32_f8_scale_f8_quant` | `:172` | gfx950 | MXFP8 (block fp8 + e8m0), `dtypeMax=2^8=256` | mxfp8 path |
| fused `fused_rms_mxfp4_quant`, `fused_dynamic_mxfp4_quant_moe_sort` | `fused_mxfp4_quant.py` | gfx950 | removes a pass | production norm/MoE |

### SOTA excerpt — per-1×32 MXFP4 cast, pow-2 dtypeMax + E8M0 (`aiter/ops/quant.py:75`)
```python
def per_1x32_f4_quant(x, scale=None, quant_dtype=dtypes.fp4x2, shuffle=False, pack_dim=-1):
    assert quant_dtype == dtypes.fp4x2
    block_size = 32
    F4E2M1_MAX = 6.0                                   # e2m1 representable max
    MAX_POW2 = int(torch.log2(torch.tensor(F4E2M1_MAX)).item())   # = 2
    dtypeMax = 2.0**MAX_POW2                           # = 4.0  (NOT 6.0 — scale to a power of 2)
    x = x.view(-1, block_size)
    max_abs = torch.amax(torch.abs(x.float()), 1)      # per-32-block amax
    scale_e8m0_biased = fp4_utils.f32_to_e8m0(max_abs / dtypeMax)   # power-of-2 E8M0 scale
    scale_f32 = fp4_utils.e8m0_to_f32(scale_e8m0_biased)
    y = fp4_utils.f32_to_mxfp4(x.float() / scale_f32.view(-1, 1))
    scale = scale_e8m0_biased.view(m, -1).view(torch.uint8)
    if shuffle:
        scale = fp4_utils.e8m0_shuffle(scale)          # HW layout for the scaled-MFMA scale operand
    return y, scale.view(dtypes.fp8_e8m0)
```
Note `dtypeMax = 2^floor(log2(6)) = 4.0`, **not** 6.0: the block scale is forced to a power of two so the
shared E8M0 exponent divides cleanly (this corrects the loose "FP4_MAX=4" — the *representable* e2m1 max is
6.0, but the *scaling denominator* is 4.0).

### SOTA excerpt — E8M0 scale shuffle layout (`aiter/utility/fp4_utils.py:72`)
```python
def e8m0_shuffle(scale):                               # scale: [m, n] uint8 (one e8m0 per 32-block)
    m, n = scale.shape
    scale_padded = torch.empty((m+255)//256*256, (n+7)//8*8, ...)   # pad to 256×8
    scale_padded[:m, :n] = scale
    sm, sn = scale_padded.shape
    scale = scale_padded.view(sm//32, 2, 16, sn//8, 2, 4)          # the HW tile decomposition
    scale = scale.permute(0, 3, 5, 2, 4, 1).contiguous().view(sm, sn)
    return scale                                       # interleaved so the scaled-MFMA reads it directly
```

## Config space / knobs
| knob | values | effect |
|---|---|---|
| group/block size | **32** (fixed) | OCP MX block; do not change |
| `shuffle` | True/False | HW MFMA layout (`e8m0_shuffle`) vs unshuffled (`tl.dot_scaled` / sim) |
| `pack_dim` | -1 (LHS) / 0 (RHS) | which dim packs 2 FP4 → 1 byte for `tl.dot_scaled` |
| `num_rows` / `num_rows_factor` | tensor / int | ragged / MoE token counts |
| format | MXFP4 / MXFP8 (`per_1x32_f8_scale_f8_quant`) / MXFP6 | accuracy gate → [[numerics.md]] |
| `scaling_mode` | "even" | Quark even_round |
| shuffle alloc pad | `(m+255)//256*256` × `((n+31)//32+7)//8*8` | HIP path padding for the shuffled scale |

## Measured performance
| config | metric | value @ hw / ver / date | source |
|---|---|---|---|
| MXFP4 vs FP16 GEMM peak | throughput | MXFP4 ~10 PF vs FP8 ~5 PF on MI355 (AMD-reported peak) | matrix-cores-cdna blog |
| **downstream MXFP4 GEMM (achieved)** | TFLOPS | **Gluon 5255 TFLOPS @ 92.41% MFMA eff** (4096×4096×32768, MI355X) — the practical MXFP4 ceiling | Gluon GEMM tutorial |
| MXFP6 vs MXFP4 | rate | **same throughput** (MXFP6 = MXFP4 rate on CDNA4) | matrix-cores-cdna blog |
| gfx942 FP4 | speedup | **none** — simulation only (no FP4 MFMA on CDNA3) | code + HW matrix |
| quant cast itself | bound | per-block amax, HBM-bound (no MFMA) | `quant.py:75` |

> Peak PF figures are AMD-reported architecture peaks; the Gluon **5255 TFLOPS @ 92.41%** is the achieved
> downstream MXFP4 GEMM bar (the cast itself is HBM-bound). For the FP8 block-scale GEMM that consumes the
> companion path, the HIP/C++ **8-wave ping-pong** kernel hits **3204 TFLOPS** @ 8192 (beats hipBLASLt, no
> asm), with **4-wave interleave** as the successor — scheduling from **HipKittens** (arXiv 2511.08083). See
> [[operators/scaled_quant_gemm/backends/asm]] / [[operators/scaled_quant_gemm/tuning]] /
> [[optimization/mfma_scheduling]]. Bench `gemm_a4w4` at your shapes.

## Numerics / parity
- **E8M0 shared scale**, one per 32-element block, power-of-two (8-bit exponent, no mantissa, biased by
  127). `f32_to_e8m0` extracts bits `(u32>>23)&0xFF` with round-to-nearest; NaN→0xFF.
- **MXFP4 = e2m1**: 2 exp, 1 mant, ±0.5 (subnormal) … ±6.0 (max). Scaling denominator is `2^2 = 4.0`.
- **MXFP8** (`per_1x32_f8_scale_f8_quant`): block fp8 (e4m3) + E8M0 scale, `dtypeMax=2^8=256` (pow-2 of
  448). **MXFP6** (e2m3 or e3m2): the accuracy fallback when MXFP4 degrades.
- Per OCP MX v1.0: group 32, E8M0 scale 2^−127..2^127, one NaN reserved; element formats independent of
  the scale.
- **Gate:** task accuracy, never byte parity; MXFP6 / mixed fallback if MXFP4 fails → [[numerics.md]].
  Wrong shuffle = silent corruption (verify with the matrix-instruction calculator).

## Integration (rebind seam)
- `aiter.ops.quant.per_1x32_f4_quant*` / `per_1x32_f4_quant_for_dot_scaled`; routed by
  `get_hip_quant(QuantType.per_1x32)` / `get_triton_quant(...)`.
- Consumed by `gemm_a4w4` / A4W4 MoE. In vLLM the MXFP4 path is gfx950-only;
  `VLLM_ROCM_USE_AITER_FP4BMM=0` on gfx942 (crash). → [[../../../reference/env_vars]].

## Pitfalls & anti-patterns
- **gfx942: no FP4 HW** — simulation only, no speedup; **FP4BMM crashes** ([[../../../quantization/hardware_support_matrix]]).
- **Wrong scale shuffle** (shuffled when the GEMM wants unshuffled, or vice versa) → corruption with no
  error. `tl.dot_scaled` wants unshuffled; the raw scaled-MFMA wants `e8m0_shuffle`.
- Using `dtypeMax=6.0` instead of `4.0` → off-by-1.5× scale (must be a power of two for E8M0).
- Per-tensor FP4 or group ≠ 32 → not representable in the MX format / breaks the MFMA.

## How to verify
- Round-trip per-block error vs bf16; gate `tol_err_ratio=0.05`.
- `amd_matrix_instruction_calculator --get-register --Ax/--Bx` to confirm the scaled-MFMA scale layout
  matches `e8m0_shuffle`.
- e2e gsm8k on gfx950; `AITER_LOG_MORE=1` to confirm `gemm_a4w4` dispatched.

## Alternatives / cross-links
[triton.md](triton.md) · [hip.md](hip.md) · [ck.md](ck.md) · [overview.md](../overview.md) ·
[numerics.md](../numerics.md) · [[hardware/cdna4_mi350]] · [[operators/scaled_quant_gemm]] ·
[[../../../quantization/block_scaling_mxfp]] · [[../../../quantization/hardware_support_matrix]].

## Worked example
Quantize an A4W4 GEMM's two bf16 operands for the HW scaled-MFMA on gfx950:
```python
import torch
from aiter import dtypes
from aiter.ops.quant import per_1x32_f4_quant
A = torch.randn(4096, 8192, dtype=torch.bfloat16, device="cuda")   # (M, K)
B = torch.randn(8192, 8192, dtype=torch.bfloat16, device="cuda")   # (K, N)
a_fp4, a_scale = per_1x32_f4_quant(A, shuffle=True, pack_dim=-1)   # K packed → (M, K//2), scale (M, K//32)
b_fp4, b_scale = per_1x32_f4_quant(B, shuffle=True, pack_dim=0)    # K packed → (K//2, N), scale (K//32, N)
# a_fp4/b_fp4 + shuffled e8m0 scales feed gemm_a4w4 (scaled v_mfma_*_f8f6f4)
```
For a `tl.dot_scaled` Triton GEMM instead, pass `shuffle=False` (unshuffled scale).

## Sources
- aiter MX quant + fused: `ROCm/aiter@a6bb49937:aiter/ops/quant.py` (`per_1x32_f4_quant:75`,
  `per_1x32_f4_quant_for_dot_scaled:146`, `per_1x32_f8_scale_f8_quant:172`, `per_1x32_f4_quant_hip:359`),
  `aiter/utility/fp4_utils.py` (`f32_to_e8m0:49`, `e8m0_to_f32:61`, `e8m0_shuffle:72`),
  `aiter/ops/triton/quant/fused_mxfp4_quant.py`.
- Block-scaled MFMA, MXFP6=MXFP4 rate, FP6/FP4 type codes:
  https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
- OCP MX v1.0 (group 32, E8M0, e2m1 max 6.0):
  https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
- Downstream Gluon MXFP4 GEMM 5255 TFLOPS @ 92.41% (MI355X): https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html ; FP8 8-wave ping-pong 3204@8192 (>hipBLASLt): https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html ; scheduling origin: arXiv 2511.08083.
