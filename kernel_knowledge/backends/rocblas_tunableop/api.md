---
title: rocBLAS — classic BLAS / Tensile GEMM API on CDNA
kind: backend
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8, fp32]
regimes: [both]
status: competitive
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/projects/rocBLAS/en/latest/
  - https://rocm.docs.amd.com/projects/rocBLAS/en/docs-6.2.1/reference/enumerations.html
  - https://github.com/ROCm/rocBLAS
---

# rocBLAS (Tensile GEMM API)

## TL;DR
rocBLAS is AMD's **classic full Level-1/2/3 BLAS** (the cuBLAS analogue), GEMM via **`rocblas_gemm_ex`** /
`rocblas_gemm_strided_batched_ex`, backed by **Tensile** (same generator family as hipBLASLt's
TensileLite). It has **no fused bias/activation epilogues** (that's hipBLASLt's job) but supports fp8 on
gfx942 (FNUZ). Its main role in the modern stack is as **one of the two libraries PyTorch TunableOp races**
(rocBLAS vs hipBLASLt) — see [tunableop.md](tunableop.md). You select a specific Tensile kernel with
`rocblas_gemm_algo_solution_index`. **CRITICAL:** like hipBLASLt, rocBLAS sits under PyTorch dispatch, so
on sglang/vllm/aiter the live GEMM path **bypasses it** — TunableOp gets 0 engagement there (see
[when_wins.md](when_wins.md)).

## Concepts
| aspect | rocBLAS |
|---|---|
| role | full BLAS; GEMM via `rocblas_gemm_ex` / `rocblas_gemm_strided_batched_ex` |
| backend | **Tensile** (per-arch logic YAML picks a solution) |
| arch | gfx908/gfx90a/gfx942/gfx950 (codename gfx942 = aqua_vanjaram, CDNA3 MI300A/MI300X) |
| solution DB | `library/src/blas3/Tensile/Logic/asm_full/.../gfx942/`; kernels ship as `TensileLibrary_*_gfx942.hsaco` (fallback + tuned variants per arch) |
| fp8 | yes on gfx942 (FNUZ) via fp8 compute; hipBLASLt usually preferred for fp8 + epilogues |
| epilogues | **none** (no fused bias/act — use hipBLASLt) |
| tuning override | via Tensile logic / TensileLite or TunableOp; **no** env override file like hipBLASLt's `HIPBLASLT_TUNING_FILE` |

## The API
```cpp
rocblas_handle h; rocblas_create_handle(&h);
rocblas_gemm_ex(h,
    rocblas_operation_none, rocblas_operation_none,
    m, n, k, &alpha,
    A, rocblas_datatype_bf16_r, lda,
    B, rocblas_datatype_bf16_r, ldb, &beta,
    C, rocblas_datatype_bf16_r, ldc,
    D, rocblas_datatype_bf16_r, ldd,
    rocblas_datatype_f32_r,                 // compute type (fp32 accumulate)
    rocblas_gemm_algo_solution_index,       // pick a specific Tensile solution...
    solution_index,                         // ...by index
    rocblas_gemm_flags_none);
```

## The levers
- **`rocblas_gemm_algo`** — `rocblas_gemm_algo_standard` (heuristic) vs
  `rocblas_gemm_algo_solution_index` (pin a specific Tensile kernel by `solution_index`). *Negative indices
  to mean "default" were deprecated* — use the standard algo for default.
- **`rocblas_gemm_flags`** — `rocblas_gemm_flags_none`; `rocblas_gemm_flags_check_solution_index` (validate
  a chosen index); a flag to pick "highest efficiency per CU" (for many small concurrent problems; takes
  precedence over the handle's perf metric; `*_ex` only).
- **compute type** = `rocblas_datatype_f32_r` for bf16/fp16/fp8 inputs (fp32 accumulate).
- **TensileLite custom tuning** for hot shapes the pooled logic misses (esp. skinny LLM-decode GEMMs).
- Env: `ROCBLAS_VERBOSE_TENSILE_ERROR=1` to surface "Tensile solution found but exception thrown".

## Pitfalls
- **No fused epilogues** — bias/activation must be a separate kernel; for fused paths use hipBLASLt.
- **Solution indices are version-locked** to the rocBLAS/Tensile build; re-tune on upgrade.
- **"Tensile solution found, but exception thrown"** is shown only once unless
  `ROCBLAS_VERBOSE_TENSILE_ERROR` is set.
- **Large/odd workspace:** some `gemm_ex` solutions allocate an unusually large workspace (not a leak) —
  bound workspace when tuning.
- **"Could not initialize Tensile host: No devices found"** in containers → arch/library mismatch (e.g.
  missing `gfx942` `.hsaco`); ensure the rocBLAS build matches the GPU arch.
- **Bypassed by aiter/sglang/vllm** — rocBLAS (and the TunableOp race) only engages the **PyTorch dispatch
  path**. See [when_wins.md](when_wins.md).

## Verify
- `ROCBLAS_LAYER=2` / `ROCBLAS_VERBOSE_TENSILE_ERROR=1` to log shapes/solutions.
- For PyTorch, inspect the TunableOp CSV: `Gemm_Rocblas_<idx>` rows prove rocBLAS won that shape (vs
  `Gemm_Hipblaslt_*`).

## Sources
- rocBLAS docs (`rocblas_gemm_ex`, Tensile, solution index): https://rocm.docs.amd.com/projects/rocBLAS/en/latest/
- rocBLAS enumerations (`rocblas_gemm_algo_solution_index`, flags):
  https://rocm.docs.amd.com/projects/rocBLAS/en/docs-6.2.1/reference/enumerations.html
- rocBLAS repo (Tensile logic / `.hsaco` per arch; negative-index deprecation in CHANGELOG):
  https://github.com/ROCm/rocBLAS
- gfx942 = CDNA3 aqua_vanjaram, MI300A/MI300X: AMD CDNA3 ISA guide (see sourcing_rules.md).
- tuning: [tunableop.md](tunableop.md) · when it wins / bypass: [when_wins.md](when_wins.md)
