---
title: hipBLASLt environment variables — full reference table
kind: reference
backend: hipblaslt
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode, both]
status: sota
updated: 2026-06-05
sources:
  - https://rocm.docs.amd.com/projects/hipBLASLt/en/develop/reference/env-variables.html
---

# hipBLASLt environment variables

## TL;DR
The env vars you actually use cluster into: **logging/dump** (`HIPBLASLT_LOG_MASK`, `HIPBLASLT_LOG_FILE`),
**offline tuning** (`HIPBLASLT_TUNING_FILE` / `_OVERRIDE_FILE` / `_USER_MAX_WORKSPACE`), **profiling**
(`HIPBLASLT_ENABLE_MARKER`), **PyTorch routing** (`TORCH_BLAS_PREFER_HIPBLASLT`), and **Stream-K/Origami**
(`TENSILE_*`). Solution indices written by tuning are version/arch-locked.

## Logging & dump

| Var | Purpose | Default |
|---|---|---|
| `HIPBLASLT_LOG_LEVEL` | verbosity 0–5 (1=errors, 2=trace, 3=hints, 4=info, 5=API trace) | 0 |
| `HIPBLASLT_LOG_MASK` | bit mask: 1 Error, 2 Trace, 4 Hints, 8 Info, 16 API-trace, **32 Bench**, 64 Profile, 128 Extended-profile (combine bits) | 0 |
| `HIPBLASLT_LOG_FILE` | log path (`%i` → PID, e.g. `log_%i.log`) | stdout |

`HIPBLASLT_LOG_MASK=32` ("Bench") is what dumps **bench-replayable GEMM command lines** for offline tuning
(see [offline_tuning.md](offline_tuning.md)).

## Offline tuning

| Var | Purpose | Default |
|---|---|---|
| `HIPBLASLT_TUNING_FILE` | **write** best solution indices per shape during search | unset |
| `HIPBLASLT_TUNING_OVERRIDE_FILE` | **read** tuned indices at runtime (override heuristic) | unset |
| `HIPBLASLT_TUNING_USER_MAX_WORKSPACE` | cap workspace (bytes) the tuned solution may use | 128 MiB (`128*1024*1024`) |

## Profiling & PyTorch routing

| Var | Purpose | Default |
|---|---|---|
| `HIPBLASLT_ENABLE_MARKER` | emit ROCProfiler/roctx markers | 0 (off) |
| `TORCH_BLAS_PREFER_HIPBLASLT=1` | make PyTorch prefer hipBLASLt over hipBLAS/rocBLAS | varies by version |

## Stream-K / Origami (skinny & decode shapes)

| Var | Purpose | Default |
|---|---|---|
| `TENSILE_SOLUTION_SELECTION_METHOD` | 0 = standard tuned libs (no Stream-K); 2 = Origami + Stream-K. **No effect on MI350 (gfx950).** | 0 |
| `TENSILE_STREAMK_DYNAMIC_GRID` | dynamic grid selection (6 = auto-pick optimal WG count; 0–5 alternatives) | 6 |
| `TENSILE_STREAMK_FIXED_GRID` | fix the grid to N workgroups (e.g. 64) | unset |
| `TENSILE_STREAMK_MAX_CUS` | cap CUs used by Stream-K kernels (e.g. 32) | all CUs |

Stream-K is the lever for decode/skinny GEMMs where a normal grid leaves CUs idle — relevant on gfx942;
disabled-by-config on gfx950.

## Type overrides

| Var | Purpose | Default |
|---|---|---|
| `HIPBLASLT_OVERRIDE_COMPUTE_TYPE_XF32` | override XF32 compute type: -1 off, 0 F32, 1 XF32/TF32, 2 F32_BF16 | -1 |

(Note: TF32 is N/A on CDNA4 — see taxonomy.)

## Pitfalls
- ⚠ Tuning files store **version/arch-locked** solution indices — re-tune on every ROCm/hipBLASLt bump.
- ⚠ Under aiter (sglang/vLLM), `HIPBLASLT_TUNING_OVERRIDE_FILE` is **not** consulted — deploy via aiter's
  DB ([../aiter/tuned_gemm.md](../aiter/tuned_gemm.md)).
- `HIPBLASLT_LOG_MASK=32` logs only GEMMs that actually hit a hipBLASLt kernel, with many duplicates.

## Cross-links
[offline_tuning.md](offline_tuning.md) · [api.md](api.md) · [tensilelite.md](tensilelite.md) ·
[when_wins.md](when_wins.md).

## Sources
- hipBLASLt environment variables (official reference): https://rocm.docs.amd.com/projects/hipBLASLt/en/develop/reference/env-variables.html
- Tuning-mode bench defaults + LOG_MASK usage: https://rocm.blogs.amd.com/software-tools-optimization/hipblaslt-offline-tuning-part2/README.html
