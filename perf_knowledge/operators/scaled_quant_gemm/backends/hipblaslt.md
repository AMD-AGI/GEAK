---
title: scaled_quant_gemm on hipblaslt — SOTA card
kind: sota_card
operator: scaled_quant_gemm
backend: hipblaslt
gens: [gfx942, gfx950]
dtypes: [fp8_e4m3_fnuz, fp8_e5m2_fnuz, fp4_e2m1]
regimes: [prefill, decode]
status: competitive
updated: 2026-06-05
sources:
  - https://github.com/ROCm/hipBLASLt
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
---

# scaled_quant_gemm × hipblaslt

## TL;DR
> hipBLASLt supports **fp8 scaled GEMM** (and CDNA4 block-scaled mxfp solutions) with scale pointers in the
> matmul descriptor; solution selection picks per shape. No Python rebind to force a custom scaled kernel —
> get it on the live path through aiter's per-shape DB (which races hipBLASLt scaled solutions).

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| hipBLASLt fp8/mxfp scaled solution (`Cijk_*`, scale ptr in descriptor) | `ROCm/hipBLASLt@HEAD` | gfx942 fp8 FNUZ; gfx950 fp8/mxfp | no first-party number reproduced; selected per shape | covered fp8/fp4 shapes |

## Config space / knobs
- Not user-authored: scaled solutions, scale mode (tensor vs block on gfx950), epilogue are Tensile-baked.
  Influence selection via aiter's tuned DB ([./aiter.md](aiter.md)) / solution index.

## Numerics / parity
- Scale-after-dot, fp32 accumulate; FNUZ (gfx942) vs OCP (gfx950); accuracy gate → [../numerics.md](../numerics.md).

## Integration (rebind seam)
- ⚠ No clean rebind for a custom scaled kernel; **use aiter** to select the best hipBLASLt scaled solution
  on the live serving path. `HIPBLASLT_TUNING_FILE`/TunableOp don't engage the aiter path.

## Pitfalls & anti-patterns
- Scale-mode mismatch (tensor-scaled solution vs block-scaled model) → wrong results or no candidate.
- Tuning through PyTorch hooks bypassed by aiter → 0 engagement (see dense aiter card).

## How to verify
- rocprof shows a scaled `Cijk_*` kernel; A/B via aiter DB swap + bf16 accuracy gate.

## Alternatives / cross-links
[triton.md](triton.md) · [aiter.md](aiter.md) · [ck.md](ck.md) · [asm.md](asm.md) · [hip.md](hip.md) · [../overview.md](../overview.md)

## Sources
- hipBLASLt: https://github.com/ROCm/hipBLASLt
- Matrix Core (fp8 scaling): https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
