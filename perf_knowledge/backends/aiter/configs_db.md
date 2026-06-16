---
title: aiter configs DB — CSV schema, env overrides, merge semantics
kind: reference
backend: aiter
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8, fp4_e2m1]
regimes: [prefill, decode]
status: sota
updated: 2026-06-05
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/configs/
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/jit/core.py
---

# aiter configs DB

## TL;DR
aiter's per-shape tuned configs live as **CSV files under `aiter/configs/`**, one (tuned, untuned) pair
per op family. Each is reachable by an `AITER_CONFIG_*` env var that **overrides** the shipped path and is
**`:`-mergeable** (your tuned rows overlay the shipped table without editing site-packages). This is the
deploy seam for every aiter tuning win.

## The files (`aiter/configs/`)

| Op family | tuned CSV | untuned CSV | env override |
|---|---|---|---|
| dense bf16/fp16 GEMM | `bf16_tuned_gemm.csv` | `bf16_untuned_gemm.csv` | `AITER_CONFIG_GEMM_BF16` |
| bf16 batched GEMM | `bf16_tuned_batched_gemm.csv` | `bf16_untuned_batched_gemm.csv` | `AITER_CONFIG_BF16_BATCHED_GEMM` |
| a8w8 (fp8/int8) GEMM | `a8w8_tuned_gemm.csv` | `a8w8_untuned_gemm.csv` | `AITER_CONFIG_GEMM_A8W8` |
| a8w8 bpreshuffle | `a8w8_bpreshuffle_tuned_gemm.csv` | `a8w8_bpreshuffle_untuned_gemm.csv` | `AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE` |
| a8w8 block-scale | `a8w8_blockscale_tuned_gemm.csv` | `a8w8_blockscale_untuned_gemm.csv` | `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE` |
| a8w8 blockscale+bpreshuffle | `a8w8_blockscale_bpreshuffle_tuned_gemm.csv` | `..._untuned_...` | `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE` |
| a8w8 batched GEMM | `a8w8_tuned_batched_gemm.csv` | `a8w8_untuned_batched_gemm.csv` | `AITER_CONFIG_A8W8_BATCHED_GEMM` |
| a4w4 block-scale GEMM | `a4w4_blockscale_tuned_gemm.csv` | `a4w4_blockscale_untuned_gemm.csv` | `AITER_CONFIG_GEMM_A4W4` |
| fused MoE | `tuned_fmoe.csv` | `untuned_fmoe.csv` | `AITER_CONFIG_FMOE` |
| asm a8w8 list | `asm_a8w8_gemm.csv` | — | (asm kernel catalog) |

(Each `AITER_CONFIG.*_FILE` property in `aiter/jit/core.py` resolves env → default path → shipped CSV.)

## Schemas (real headers)

**`bf16_tuned_gemm.csv`** (key + result):
```
gfx, cu_num, M, N, K, bias, dtype, outdtype, scaleAB, bpreshuffle,   # key
libtype, solidx, splitK, us, kernelName, err_ratio, tflops, bw       # result
```
Lookup at serving time uses the 9-tuple `(cu_num, M(padded), N, K, bias, dtype, outdtype, scaleAB,
bpreshuffle)` (the `gfx` column documents provenance; `M` is the bucketed `padded_M`). See
[tuned_gemm.md](tuned_gemm.md).

**`bf16_untuned_gemm.csv`** (capture output, no result columns):
```
M, N, K, bias, dtype, outdtype, scaleAB, bpreshuffle
```
Written by `AITER_TUNE_GEMM=1` (`save_shapes`), deduped. `bias` = `bias is not None`.

**`a4w4_blockscale_tuned_gemm.csv`**:
```
gfx, cu_num, M, N, K, kernelId, splitK, us, kernelName, tflops, bw, errRatio
```
(real rows are `gfx950` FP4 BpreShuffle kernels, e.g.
`_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E`).

**`tuned_fmoe.csv`**:
```
cu_num, token, model_dim, inter_dim, expert, topk, act_type, dtype, q_dtype_a, q_dtype_w, q_type,
use_g1u1, doweight_stage1, block_m, ksplit,   # key
us1, kernelName1, err1, us2, kernelName2, err2, us, run_1stage, tflops, bw, _tag   # result
```
(stage-1 + stage-2 kernel names; see [fmoe.md](fmoe.md)).

## Env override + merge semantics (`aiter/jit/core.py`)
- `get_config_file(env_name, default_file, tuned_file_name)`: if `os.getenv(env_name)` is set it is used
  (else the shipped path). When the value is a `:`-joined (`os.pathsep`) list, the **shipped default is
  prepended** and `update_config_files` merges all existing CSVs into one (last-wins on duplicate keys),
  so you can overlay a small tuned file on top of the shipped table:
  ```bash
  export AITER_CONFIG_GEMM_BF16=/abs/my_bf16_tuned.csv          # replace
  export AITER_CONFIG_GEMM_BF16=/abs/a.csv:/abs/b.csv           # merge (shipped prepended)
  ```
- `AITER_LOG_TUNED_CONFIG=1` logs every DB hit (`... is tuned on cu_num = N in <file>, libtype is ...`).

## Pitfalls
- **Version/build-specific**: hipBLASLt `solidx` and asm `kernelName` are tied to the aiter/ROCm build.
  Never ship a hand-copied tuned table as portable — re-tune on upgrade (sourcing rule #2).
- **`cu_num` and `bias` in the key** — a DB tuned on a different CU count or bias flag won't hit
  (see [tuned_gemm.md](tuned_gemm.md) pitfalls).
- A `flydsl` row needs the FlyDSL package present to be honored (see [flydsl_path.md](flydsl_path.md)).

## Cross-links
[tuned_gemm.md](tuned_gemm.md) · [fmoe.md](fmoe.md) · [flydsl_path.md](flydsl_path.md) ·
[integration.md](integration.md).

## Sources
- On-box: `ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0`: `aiter/configs/*.csv` (real headers/rows),
  `aiter/jit/core.py` (`AITER_CONFIG.*_FILE`, `get_config_file`, `update_config_files`),
  `aiter/tuned_gemm.py` (`save_shapes`, index columns).
