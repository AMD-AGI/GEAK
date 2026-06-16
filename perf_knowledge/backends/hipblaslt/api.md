---
title: hipBLASLt API — hipblasLtMatmul, epilogues, FP8 scaling, hipblaslt_ext
kind: backend
backend: hipblaslt
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e5m2_fnuz, fp8_e4m3, int8]
regimes: [prefill, decode, both]
status: sota
updated: 2026-06-05
sources:
  - https://rocm.docs.amd.com/projects/hipBLASLt/en/develop/reference/ext-reference.html
  - https://github.com/ROCm/hipBLASLt
---

# hipBLASLt API

## TL;DR
hipBLASLt is AMD's "lt" GEMM library (the cuBLASLt analog) and the **default BLAS** executed for dense
bf16/fp16/fp8 GEMM on Instinct — directly under PyTorch and indirectly under aiter (which calls a chosen
**solution index** via `hipb_mm`). You build a **matmul plan** once (layouts + descriptor + algo) and reuse
it, with **epilogues** (bias/activation/aux) and **FP8 scaling** fused in. There is no first-class pip
module for general use; frameworks reach it through PyTorch (`torch.matmul`, `torch._scaled_mm`).

> Repo note: `ROCm/hipBLASLt` is being consolidated into `ROCm/rocm-libraries`; APIs below remain current,
> binaries still install under `/opt/rocm`.

## What it computes
```
D = Activation( alpha * op(A) * op(B) + beta * op(C) + bias )
```
`op(X)` ∈ {N, T}; `alpha`/`beta` host or device scalars; `bias` is a length-rows(D) vector;
`Activation` is the epilogue.

## Core C++ objects

| Object / call | Purpose |
|---|---|
| `hipblasLtHandle_t` (`hipblasLtCreate`) | library handle |
| `hipblasLtMatrixLayout_t` | per-matrix dtype, rows, cols, ld, batch, batch stride |
| `hipblasLtMatmulDesc_t` | compute/scale type, transposes, epilogue, bias/scale pointers |
| `hipblasLtMatmulPreference_t` | constraints (max workspace) for the heuristic |
| `hipblasLtMatmulAlgoGetHeuristic` | ranked candidate algos for the problem |
| `hipblasLtMatmul` | execute with a chosen algo |

Minimal flow: create layouts → `hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32F, HIP_R_32F)` →
set `TRANSA/TRANSB`, `EPILOGUE`, `BIAS_POINTER` → set a workspace budget on the preference →
`hipblasLtMatmulAlgoGetHeuristic(...)` → `hipblasLtMatmul(..., &res[0].algo, workspace, ws, stream)`.

### Epilogue enum (`hipblasLtEpilogue_t`)
`DEFAULT` (none) · `BIAS` · `RELU`/`RELU_BIAS` · `GELU`/`GELU_BIAS` · `GELU_AUX`/`GELU_AUX_BIAS`
(writes pre-activation aux for backward) · `SWISH_EXT`/`SWISH_BIAS_EXT` (SiLU). ROCm 7.2 CK/hipBLASLt
added clamp epilogues (`..._CLAMP_EXT`, `..._CLAMP_BIAS_EXT`).

## `hipblaslt_ext` (the framework-facing API)

| Class | Purpose |
|---|---|
| `hipblaslt_ext::Gemm` | single GEMM (set problem → get algos → run) |
| `hipblaslt_ext::GroupedGemm` | grouped/variable GEMM over a list of problems |
| `GemmProblemTypeV2`, `GemmEpilogueV2`, `GemmInputsV2`, `GemmPreferenceV2`, `GemmTuningV2` | structured descriptors |
| `setScalingAType` / `setScalingBType` | FP8 scaling mode (scalar=0 / vector=1; only when A,B both FP8) |

> API churn: original non-V2 structs were unified into V2 (non-V2 removed); the V2 names are themselves now
> being deprecated as the API re-stabilizes — check your version's `ext-reference` docs.

## FP8 GEMM & scaling (gfx942 FNUZ, gfx950 OCP/MX)
```
D = scaleD * Activation( scaleA * op(A_fp8) * scaleB * op(B_fp8) + beta*C )
```
| Descriptor attribute | Mode | Meaning |
|---|---|---|
| `_A_SCALE_MODE`/`_B_SCALE_MODE` | `..._SCALE_SCALAR_32F` | per-tensor scalar fp32 |
| same | `..._SCALE_OUTER_VEC_32F` | per-row/col vector fp32 |
| same | `..._SCALE_VEC32_UE8M0` | 32-elem block scale (MX-style, UE8M0 exponent) |
| `_A_/_B_/_D_SCALE_POINTER` | — | device scale tensor(s) |

gfx942: `f8_r`=E4M3FNUZ, `bf8_r`=E5M2FNUZ; gfx950 adds OCP FP8 + FP4/MXFP4. FP8 **swizzle GEMM**
(order `HIPBLASLT_ORDER_COL16_4R8`) is exposed for higher FP8 throughput.

## Integration / rebind seam
- Raw torch: `TORCH_BLAS_PREFER_HIPBLASLT=1` makes PyTorch prefer hipBLASLt; FP8 via
  `torch._scaled_mm(a, b, scale_a, scale_b, out_dtype=...)`.
- **Under aiter** (sglang/vLLM): hipBLASLt is invoked by `hipb_mm(inp, weights.t(), solidx, ...)` with a
  solution index from aiter's DB — the torch BLAS dispatch and `HIPBLASLT_TUNING_OVERRIDE_FILE` are
  **not** consulted on that path. Deploy serving wins via aiter
  ([../aiter/tuned_gemm.md](../aiter/tuned_gemm.md)).

## Numerics / parity
Same math, different tiling/solution → parity-safe; cross-solution argmax flips possible but rare.

## Pitfalls
- Solution indices are **ROCm/hipBLASLt-version-locked** — re-tune on upgrade.
- No general pip module — Python access is through PyTorch.
- On serving under aiter, the override file alone does nothing (known dead-end).

## How to verify
`HIPBLASLT_ENABLE_MARKER=1` / roctx markers; `--algo_method index` re-bench a specific solution. For
serving, verify via aiter engagement.

## Cross-links
[offline_tuning.md](offline_tuning.md) · [tensilelite.md](tensilelite.md) · [env.md](env.md) ·
[when_wins.md](when_wins.md) · [`operators/dense_gemm/backends/hipblaslt.md`](../../operators/dense_gemm/backends/hipblaslt.md).

## Sources
- hipBLASLtExt reference (Gemm/GroupedGemm/scaling): https://rocm.docs.amd.com/projects/hipBLASLt/en/develop/reference/ext-reference.html
- hipBLASLt repo (API, tensilelite, clients): https://github.com/ROCm/hipBLASLt
- FP8/epilogue/scaling details: https://rocm.blogs.amd.com/software-tools-optimization/hipblaslt-offline-tuning-part2/README.html
- aiter on-box call (`hipb_mm` by solidx): `ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/tuned_gemm.py`.
