---
title: hipBLASLt offline tuning — 3-stage dump → bench → override
kind: workflow
backend: hipblaslt
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
status: sota
updated: 2026-06-05
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/hipblaslt-offline-tuning-part2/README.html
  - https://rocm.docs.amd.com/projects/hipBLASLt/en/develop/how-to/how-to-use-hipblaslt-offline-tuning.html
---

# hipBLASLt offline tuning

## TL;DR
Offline tuning picks the best **solution index** per GEMM shape ahead of time and overrides the runtime
heuristic — **zero rebuild, deterministic, no runtime search cost**. Three stages: **(1) dump** shapes with
`HIPBLASLT_LOG_MASK=32`, **(2) tune** each with `hipblaslt-bench --algo_method all` under
`HIPBLASLT_TUNING_FILE`, **(3) override** at runtime with `HIPBLASLT_TUNING_OVERRIDE_FILE`. This is the
primary path for **raw-torch** stacks. On sglang/vLLM, hipBLASLt runs *under aiter*, which bypasses the
override file — deploy serving wins via aiter's DB instead ([../aiter/tuned_gemm.md](../aiter/tuned_gemm.md)).

## The 3 stages

```
 workload ──▶ 1. DUMP (LOG_MASK=32) ──▶ 2. TUNE (TUNING_FILE) ──▶ 3. OVERRIDE (OVERRIDE_FILE) ──▶ prod
```

### Stage 1 — dump bench-replayable shapes
```bash
export HIPBLASLT_LOG_MASK=32                 # 32 = log each GEMM as a hipblaslt-bench command line
export HIPBLASLT_LOG_FILE=dump_gemm_shapes.txt
python serve.py        # short run; every decode step reuses the same shapes
unset HIPBLASLT_LOG_MASK HIPBLASLT_LOG_FILE
```
Only GEMMs that actually call a hipBLASLt kernel are logged (and with many duplicates — de-dup the file).
Each line is a ready-to-run `hipblaslt-bench` invocation with the shape/dtype/transpose/ld already filled.

### Stage 2 — tune (best solution index per shape)
```bash
export HIPBLASLT_TUNING_FILE=tuning.txt                       # tuning mode: record best solidx
export HIPBLASLT_TUNING_USER_MAX_WORKSPACE=$((256*1024*1024)) # cap workspace (default 128 MiB)
/opt/rocm/bin/hipblaslt-bench \
  --a_type f8_r --b_type f8_r --c_type bf16_r --d_type bf16_r \
  --compute_type f32_r --scale_type f32_r \
  --transA N --transB T -m 4096 -n 4096 -k 4096 \
  --algo_method all --cold_iters 50 --iters 50
unset HIPBLASLT_TUNING_FILE
```
In **tuning mode** the bench defaults change automatically: `-i 1000`, `--cold_iters/-j 1000`,
`--algo_method all`, `--requested_solution -1`, `--rotating 512`.

`--algo_method`:
| Mode | Behavior |
|---|---|
| `heuristic` | search the top heuristic candidates then tune them (fast) |
| `all` | tune **every** solution in the pool (slowest, best; the default + `find_exact.py` fixed value) |
| `index` | benchmark one specific `--solution_index` |

Accuracy tip (AMD): use an **idle** GPU (`rocm-smi`) — tuning is latency-sensitive. Use `--flush` /
`--rotating` so cache reuse doesn't inflate numbers.

### Stage 3 — override at runtime
```bash
unset HIPBLASLT_TUNING_FILE
export HIPBLASLT_TUNING_OVERRIDE_FILE=tuning.txt
python serve.py
```
`hipblasLtMatmulAlgoGetHeuristic` (C) / `algoGetHeuristic` (C++) now returns the tuned solution whenever a
matching (shape, dtype, layout) entry exists, else falls back to the heuristic.

## Verify it took effect
Re-run with `HIPBLASLT_LOG_MASK=32` and confirm the **solution index in the log matches** the tuning file.
For raw-torch, compare tok/s before/after. (For serving, see aiter engagement.)

## Higher-level automation
- **QuickTune** (Quark) — one-click offline tuning over hipblaslt-bench.
- **Primus `offline_tune_gemm.py`** — batch-tune many dumped shapes across GPUs → emits an override file.
- **`find_exact.py` + TensileLite** — generate *new* kernels when no pooled solution is good enough
  ([tensilelite.md](tensilelite.md)).

## Pitfalls
- ⚠ Solution indices are **NOT portable** across ROCm/hipBLASLt versions or archs — re-tune every upgrade.
- ⚠ Under aiter (sglang/vLLM) the override file is **not consulted** — tuning it alone does nothing there.
- `hipblaslt-bench` may be absent in some images (built with `--clients`); then aiter/gradlib is the tuner.
- Tuning with cache hot (no `--flush`/`--rotating`) → optimistic numbers that don't hold in serving.

## Cross-links
[api.md](api.md) · [tensilelite.md](tensilelite.md) · [env.md](env.md) · [when_wins.md](when_wins.md) ·
[../aiter/tuned_gemm.md](../aiter/tuned_gemm.md) (serving deploy seam).

## Sources
- GEMM Tuning within hipBLASLt — Part 2 (bench, FP8, override): https://rocm.blogs.amd.com/software-tools-optimization/hipblaslt-offline-tuning-part2/README.html
- GEMM Tuning within hipBLASLt — Part 1 (find_exact / recompilation): https://rocm.blogs.amd.com/software-tools-optimization/hipblaslt-offline-tuning-part1/README.html
- Using hipBLASLt offline tuning (official docs, tuning-mode defaults): https://rocm.docs.amd.com/projects/hipBLASLt/en/develop/how-to/how-to-use-hipblaslt-offline-tuning.html
- QuickTune day-0 guide / Primus offline_tune: https://rocm.blogs.amd.com/artificial-intelligence/hipblaslt_offline_tuning/README.html · https://github.com/AMD-AGI/Primus/blob/main/examples/offline_tune/README.md
