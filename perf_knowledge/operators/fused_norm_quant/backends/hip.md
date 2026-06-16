---
title: fused_norm_quant on hip — SOTA card
kind: sota_card
operator: fused_norm_quant
backend: hip
gens: [gfx942, gfx950]
dtypes: [fp8_e4m3_fnuz, int8]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - /sgl-workspace/aiter/aiter/ops/gated_rmsnorm_fp8_group_quant.py
  - https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
---

# fused_norm_quant × hip

## TL;DR
Hand-written HIP is the reference and is the actual impl for some aiter fused norm-quant kernels — e.g.
`gated_rmsnorm_fp8_group_quant` is a HIP kernel (gated RMSNorm + fp8 group-128 quant in one pass). vLLM's
`rms_norm_kernel` also has an FP8-quant variant (PR #40860). Reach for HIP for a quant fusion the library
lacks or to own the scale layout.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `gated_rmsnorm_fp8_group_quant` (HIP) | `aiter/ops/gated_rmsnorm_fp8_group_quant.py` | gfx942/950, fp8 group-128 | head_dim=128, group=128 only | gated rmsnorm → block-scale GEMM |
| vLLM `rms_norm_kernel` FP8-quant variant | `vllm/csrc/layernorm_kernels.cu` (PR #40860) | gfx942/950, fp8 | fp32 γ-multiply (correct for quant) | editable HIP norm+quant |
| aiter asm `module_rmsnorm_quant` | [aiter.md](aiter.md) | gfx942/950 | floor | aiter asm path |

## Config space / knobs
- Block ×64; one block/row (or persistent). Compute `y` in fp32, abs-max (per-token or per-group via
  sub-wave reduces), `scale`, then `y_q = round_RNE(y/scale)` clamped to fp8/int8 range.
- Group-128: each group = a contiguous 128-lane span; `transpose_scale` for the GEMM's scale layout.
- Vector load x; `__restrict__`; `hipcc --offload-arch=gfx942 -O3`.

## Numerics / parity
fp32 norm + fp32 scale + RNE; **fnuz fp8 on gfx942** (off-by-2×); quantize the fp32 `y`; group size matches
GEMM. Task gate. See [../numerics.md](../numerics.md).

## Integration (rebind seam)
- aiter HIP: JIT-compiled `module_gated_rmsnorm_quant`; called via `gated_rmsnorm_fp8_group_quant(...)`.
- vLLM native: `csrc/layernorm_kernels.cu` + `torch_bindings.cpp`; rebuild to edit.

## Pitfalls & anti-patterns
- ⚠ `gated_rmsnorm_fp8_group_quant` is **head_dim=128, group=128 ONLY** — asserts otherwise.
- ⚠ FNUZ/OCP dialect (off-by-2×).
- ⚠ The vLLM #42325 lesson: fp32 γ-multiply is correct *here* (quant) but was wrongly mirrored to the
  plain kernel — keep the quant and non-quant kernels' γ handling distinct/correct.

## How to verify
isolated dequant vs fp64 norm; gsm8k delta; fnuz on gfx942; `--save-temps` ISA; rocprofv3.

## Alternatives / cross-links
[aiter.md](aiter.md) · [triton.md](triton.md) · [[gated_rmsnorm]] n/a → [[rmsnorm]] · [[quant_dequant_fp8]] ·
[[languages/hip_cpp/patterns]] §1.

## Sources
- aiter HIP gated rmsnorm + fp8 group quant: `/sgl-workspace/aiter/aiter/ops/gated_rmsnorm_fp8_group_quant.py`.
- vLLM HIP rms_norm FP8-quant variant: https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu (PR #40860).
- wave64 reduce: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html.
