---
title: PyTorch TunableOp — env-driven rocBLAS-vs-hipBLASLt GEMM autotuner
kind: backend
gens: [gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp32]
regimes: [both]
status: competitive
updated: 2026-06-08
sources:
  - https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/cuda/tunable
  - https://rocm.blogs.amd.com/artificial-intelligence/pytorch-tunableop/README.html
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# PyTorch TunableOp

## TL;DR
TunableOp (PyTorch ≥ 2.3 on ROCm) intercepts GEMM ops at the **PyTorch dispatch** layer, **races rocBLAS
vs hipBLASLt** solutions per unique shape, picks the fastest, and persists it to a **CSV**. It's the
lowest-friction way to squeeze GEMM perf from any PyTorch model on Instinct without touching code — typical
reported gains ~6–8% latency (TGI/ROCm 6.1) up to ~22% throughput (single-matmul / Gemma-2B blog cases),
but **diminishing as math libs mature** (no guarantee it beats the hipBLASLt heuristic). **CRITICAL
caveat:** it only helps the **PyTorch dispatch path**. On **sglang/vllm + aiter** the live GEMM is
dispatched by aiter and **bypasses PyTorch dispatch entirely → TunableOp gets 0 engagement**; the live
lever there is aiter's per-shape DB, not this CSV. See [when_wins.md](when_wins.md) and the dense_gemm
aiter card.

## Environment variables (complete)
| Variable | Default | Meaning |
|---|---|---|
| `PYTORCH_TUNABLEOP_ENABLED` | 0 | master on/off (`1` = enable) |
| `PYTORCH_TUNABLEOP_TUNING` | 1 | `1` = tune unseen shapes; `0` = use existing CSV only (ship mode) |
| `PYTORCH_TUNABLEOP_FILENAME` | `tunableop_results.csv` | results CSV (per-GPU; ordinal note below) |
| `PYTORCH_TUNABLEOP_VERBOSE` | 0 | `1` basic / `2` tuning status / `3` full trace |
| `PYTORCH_TUNABLEOP_VERBOSE_FILENAME` | `err` | `err`/`out`/filename for verbose output |
| `PYTORCH_TUNABLEOP_RECORD_UNTUNED` | 0 | `1` = record encountered-but-untuned shapes (offline collection) |
| `PYTORCH_TUNABLEOP_UNTUNED_FILENAME` | `tunableop_untuned.csv` | where untuned-shape records go |
| `PYTORCH_TUNABLEOP_NUMERICAL_CHECK` | off | e.g. `1e-5_1e-5` (atol_rtol) to reject divergent candidates |
| `PYTORCH_TUNABLEOP_ROCBLAS_ENABLED` | 1 | `0` excludes rocBLAS from the race |
| `PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED` | 1 | `0` excludes hipBLASLt from the race |
| `PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS` | 30 | per-op tuning time budget |
| `PYTORCH_TUNABLEOP_MAX_TUNING_ITERATIONS` | 100 | per-op tuning iteration cap |
| `PYTORCH_TUNABLEOP_MAX_WARMUP_DURATION_MS` | 0 | warmup time before measuring (0 = off) |
| `PYTORCH_TUNABLEOP_MAX_WARMUP_ITERATIONS` | 0 | warmup iterations (0 = off) |
| `PYTORCH_TUNABLEOP_ICACHE_FLUSH_ENABLED` | 1 | flush I-cache between candidates |
| `PYTORCH_TUNABLEOP_ROTATING_BUFFER_SIZE` | L2 size | MiB pool to rotate params (avoid cache-reuse skew); `0` disables |
| `PYTORCH_TUNABLEOP_BLAS_LOG` | 0 | `1` logs BLAS parameters into the CSV |

## `torch.cuda.tunable` Python API
```python
t = torch.cuda.tunable
t.enable(True); t.tuning_enable(True)            # == ENABLED / TUNING env
t.record_untuned_enable(True)                    # offline collection pass
t.set_max_tuning_duration(30); t.set_max_tuning_iterations(100)
t.set_filename("results.csv", insert_device_ordinal=True)   # -> results_0.csv, results_1.csv ...
t.set_numerical_check_tolerances(True, atol=1e-5, rtol=1e-5)
t.get_results(); t.get_validators()
t.read_file("results.csv")                       # ship mode load
t.tune_gemm_in_file("untuned.csv")               # OFFLINE single-GPU tune
t.mgpu_tune_gemm_in_file("untuned%d.csv", 8)     # OFFLINE multi-GPU (pattern needs %d wildcard)
```
No Python `write_file` — results auto-write on process exit / mode switch (C++ `WriteFile` only).
Supported ops: GEMM, batched GEMM, GEMM+bias, scaled (fp8) GEMM; recent: TF32 GEMM, ScaledGEMM &
submatrix offline tuning.

## CSV format
```
Validator,PT_VERSION,2.4.0
Validator,ROCM_VERSION,6.0.0.0-91-08e5094
Validator,HIPBLASLT_VERSION,0.6.0-592518e7
Validator,GCN_ARCH_NAME,gfx942:sramecc+:xnack-
Validator,ROCBLAS_VERSION,4.0.0-88df9726-dirty
GemmTunableOp_float_NN,nn_1024_512_2048,Gemm_Hipblaslt_NN_52565,0.0653662
GemmTunableOp_float_NN,nn_256_128_512,Gemm_Rocblas_21,0.00793602
```
- **Validator lines** pin PT / ROCm / hipBLASLt / GCN arch / rocBLAS versions. If **any** differ at load,
  TunableOp **rejects** the file (and a tuning run **overwrites** it) — the parity guard. A CSV tuned on
  gfx942 + ROCm 6.0 is invalid on ROCm 6.4 or gfx950.
- **Entry** = `op_name, params, solution_name, avg_time_ms`. `op_name` = dtype + transposes
  (`GemmTunableOp_float_NN`); `params` = transA/transB + M,N,K (`nn_1024_512_2048` — PyTorch may
  swap/commute A,B so M/N can look transposed); `solution_name` `Gemm_Hipblaslt_*` or `Gemm_Rocblas_<idx>`
  tells you which library won.
- **Incremental:** existing entries are not re-tuned; new shapes append one line.

## Workflow A — online (tune during the run)
```bash
# 1) Tune: race both libs for every shape, write CSV (warmup must hit all real shapes)
PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=1 PYTORCH_TUNABLEOP_VERBOSE=1 \
PYTORCH_TUNABLEOP_FILENAME=tunableop_results.csv python serve_or_bench.py
# 2) Ship: load CSV, no per-shape benchmarking
PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0 \
PYTORCH_TUNABLEOP_FILENAME=tunableop_results.csv python serve_or_bench.py
# 3) Baseline
PYTORCH_TUNABLEOP_ENABLED=0 python serve_or_bench.py
```
Cut unique shapes (less tuning, more reuse): static KV-cache + pad seq lengths
(`cache_implementation="static"`, `pad_to_multiple_of=8`).

## Workflow B — offline (PyTorch ≥ 2.6): collect then tune
```bash
# Pass 1 collect (cheap, no benchmarking)
PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0 PYTORCH_TUNABLEOP_RECORD_UNTUNED=1 \
PYTORCH_TUNABLEOP_UNTUNED_FILENAME=tunableop_untuned.csv python serve_or_bench.py
```
```python
# Pass 2 tune offline (single or 8x MI300X)
torch.cuda.tunable.enable(True)
torch.cuda.tunable.set_filename("tunableop_results.csv", insert_device_ordinal=True)
torch.cuda.tunable.tune_gemm_in_file("tunableop_untuned.csv")          # single
torch.cuda.tunable.mgpu_tune_gemm_in_file("tunableop_untuned%d.csv", 8) # multi-GPU
```
Then ship with `ENABLED=1 TUNING=0`.

## Pitfalls
- ⚠ **0 engagement on aiter/sglang/vllm live GEMM** — aiter bypasses PyTorch dispatch; the CSV (and
  `HIPBLASLT_TUNING_FILE`) does nothing on that path. Use aiter's DB instead. See [when_wins.md](when_wins.md).
- **Validator mismatch silently rejects/overwrites** the CSV — pin PT/ROCm/hipBLASLt/rocBLAS/arch in the
  serving image, re-tune on upgrade.
- **OOM during tuning** on MI300X — tuning allocates large workspaces; a workload that fits with tuning off
  can OOM with it on. Reduce `MAX_TUNING_ITERATIONS` / disable rotating buffer.
- **Slow tuning** (1–2 min TGI, much more for many shapes) — use offline + ship mode in prod.
- **Not always a win** — always A/B vs `ENABLED=0`; keep the CSV only if faster.

## Verify
- A/B: `ENABLED=0` baseline vs `ENABLED=1 TUNING=0` (CSV) at the target ISL/OSL/conc; accept only if
  faster and parity holds.
- Inspect CSV: count `Gemm_Rocblas_*` vs `Gemm_Hipblaslt_*` rows; absence of new rows on a serving stack is
  the smoking gun for the aiter-bypass (then 0 engagement).
- Enable `PYTORCH_TUNABLEOP_NUMERICAL_CHECK=1e-5_1e-5` during tuning for parity; disable for ship.

## Sources
- PyTorch TunableOp source & README (env vars, `torch.cuda.tunable`):
  https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/cuda/tunable
- AMD ROCm blog "Accelerating models on ROCm using PyTorch TunableOp" (CSV format, examples):
  https://rocm.blogs.amd.com/artificial-intelligence/pytorch-tunableop/README.html
- MI300X workload optimization (TunableOp + GEMM tuning):
  https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- TunableOp OOM behavior: https://github.com/pytorch/pytorch/issues/138532
- aiter-bypass / 0-engagement evidence: `operators/dense_gemm/backends/aiter.md` (perf_knowledge e2e run 2026-06-08).
- API/rocBLAS side: [api.md](api.md) · when it wins / bypass: [when_wins.md](when_wins.md)
