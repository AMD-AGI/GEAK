# rocBLAS + PyTorch TunableOp — GEMM autotuning on MI300X (CDNA3 / gfx942)

> AMD-only. rocBLAS is AMD's classic BLAS (analog of cuBLAS), also Tensile-generated. **PyTorch TunableOp** is the env-driven autotuner that, at runtime, races **rocBLAS vs hipBLASLt** solutions per GEMM shape and ships the winner in a CSV. This is the lowest-friction way to squeeze GEMM perf out of any PyTorch model on Instinct without touching code.

This file covers: the rocBLAS API & Tensile backend, when rocBLAS beats hipBLASLt, the full TunableOp env var set + `torch.cuda.tunable` Python API, the warmup→populate→ship CSV workflow (online and offline), the CSV format incl. validators, multi-GPU/parity considerations, and how to bake it into a serving stack.

---

## 1. rocBLAS in one screen

| Aspect | rocBLAS |
|---|---|
| Role | Full Level-1/2/3 BLAS; GEMM via `rocblas_gemm_ex` / `rocblas_gemm_strided_batched_ex` |
| Backend | **Tensile** (same generator family as hipBLASLt's TensileLite). Per-arch logic YAML picks a solution. |
| Arch | gfx942 (MI300X/MI325X), gfx950, gfx90a, … |
| Solution DB | `library/src/blas3/Tensile/Logic/asm_full/.../gfx942/` |
| FP8 | Yes on gfx942 (FNUZ) via `rocblas_gemm_ex` with fp8 compute; hipBLASLt is usually preferred for FP8 + epilogues |
| Epilogues | No fused bias/activation epilogues (that's hipBLASLt's strength) |
| Tuning override | Solution-fitness via Tensile logic; offline tuning typically done through TunableOp or rocBLAS' own tuning, not an env override file like hipBLASLt |

rocBLAS GEMM (C++) sketch:

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
    rocblas_gemm_algo_solution_index,       // pick a specific Tensile solution
    solution_index, rocblas_gemm_flags_none);
```

### When rocBLAS beats hipBLASLt

There is no universal winner — it is shape/dtype-specific, which is exactly why TunableOp races both. Empirically:
- rocBLAS sometimes wins on **small/odd M** (decode-time tall-skinny GEMMs) and certain strided-batched cases.
- hipBLASLt usually wins on **large square / FP8 / epilogue-fused** GEMMs.
- TunableOp result files routinely contain a **mix** (`Gemm_Rocblas_21` for some shapes, `Gemm_Hipblaslt_NN_52565` for others) — proving both libraries are needed.

---

## 2. PyTorch TunableOp — what it does

TunableOp (PyTorch ≥ 2.3, ROCm) intercepts GEMM ops and, for each unique shape it sees, benchmarks up to thousands of rocBLAS + hipBLASLt algorithms, picks the fastest, and persists the choice to a CSV. Supported ops: **GEMM, batched GEMM, GEMM+bias, scaled (FP8) GEMM**; recent additions: TF32 GEMM, ScaledGEMM offline tuning, submatrix offline tuning.

Caveat: as the math libs mature there is less headroom; there's no guarantee tuning beats the default hipBLASLt heuristic. Typical reported gains: ~6–8% latency (TGI on ROCm 6.1) up to ~22% throughput (single-matmul / Gemma-2B blog examples).

---

## 3. Environment variables (complete)

| Variable | Default | Meaning |
|---|---|---|
| `PYTORCH_TUNABLEOP_ENABLED` | 0 | Master on/off. `1` to enable TunableOp dispatch. |
| `PYTORCH_TUNABLEOP_TUNING` | 1 | `1` = tune unseen shapes; `0` = only use existing CSV (ship mode). |
| `PYTORCH_TUNABLEOP_FILENAME` | `tunableop_results.csv` | Results CSV path (per-GPU; see ordinal note). |
| `PYTORCH_TUNABLEOP_VERBOSE` | 0 | `1` basic, `2` tuning status, `3` full trace. |
| `PYTORCH_TUNABLEOP_VERBOSE_FILENAME` | `err` | `err`/`out`/filename for verbose output. |
| `PYTORCH_TUNABLEOP_RECORD_UNTUNED` | 0 | `1` = record encountered-but-untuned GEMMs (for offline tuning collection). |
| `PYTORCH_TUNABLEOP_UNTUNED_FILENAME` | `tunableop_untuned.csv` | Where untuned-shape records go. |
| `PYTORCH_TUNABLEOP_NUMERICAL_CHECK` | off | e.g. `1e-5_1e-5` (atol_rtol) to verify each candidate's correctness. |
| `PYTORCH_TUNABLEOP_ROCBLAS_ENABLED` | 1 | `0` to exclude rocBLAS from the race. |
| `PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED` | 1 | `0` to exclude hipBLASLt from the race. |
| `PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS` | 30 | Per-op tuning time budget (ms). |
| `PYTORCH_TUNABLEOP_MAX_TUNING_ITERATIONS` | 100 | Per-op tuning iteration cap. |
| `PYTORCH_TUNABLEOP_MAX_WARMUP_DURATION_MS` | 0 | Warmup time before measuring (0 = off). |
| `PYTORCH_TUNABLEOP_MAX_WARMUP_ITERATIONS` | 0 | Warmup iterations (0 = off). |
| `PYTORCH_TUNABLEOP_ICACHE_FLUSH_ENABLED` | 1 | Flush instruction cache between candidates. |
| `PYTORCH_TUNABLEOP_ROTATING_BUFFER_SIZE` | L2 size | MiB pool to rotate params and avoid cache reuse skew; `0` disables. |
| `PYTORCH_TUNABLEOP_BLAS_LOG` | 0 | `1` to log BLAS parameters into the CSV. |

---

## 4. `torch.cuda.tunable` Python API

```python
import torch
t = torch.cuda.tunable

t.enable(True)                 # == PYTORCH_TUNABLEOP_ENABLED=1
t.is_enabled()
t.tuning_enable(True)          # == PYTORCH_TUNABLEOP_TUNING (tune new shapes)
t.tuning_is_enabled()
t.record_untuned_enable(True)  # collect untuned shapes (offline collection pass)
t.record_untuned_is_enabled()

t.set_max_tuning_duration(30)        # ms per op
t.set_max_tuning_iterations(100)
t.set_filename("results.csv", insert_device_ordinal=True)  # -> results_0.csv, results_1.csv ...
t.get_filename()
t.set_numerical_check_tolerances(True, atol=1e-5, rtol=1e-5)

t.get_results()        # (op_signature, params, solution, time)
t.get_validators()     # (validator_key, validator_value)
t.read_file("results.csv")           # load a CSV (ship mode)
t.tune_gemm_in_file("untuned.csv")              # OFFLINE: tune all shapes in a file
t.mgpu_tune_gemm_in_file("untuned%d.csv", num_gpus=8)  # OFFLINE multi-GPU; pattern needs wildcard
```

Note: there is no Python `write_file` — results are written automatically on process exit (or when switching modes). `WriteFile(...)` exists only in the C++ API.

---

## 5. CSV result format

```
Validator,PT_VERSION,2.4.0
Validator,ROCM_VERSION,6.0.0.0-91-08e5094
Validator,HIPBLASLT_VERSION,0.6.0-592518e7
Validator,GCN_ARCH_NAME,gfx942:sramecc+:xnack-
Validator,ROCBLAS_VERSION,4.0.0-88df9726-dirty
GemmTunableOp_float_NN,nn_1024_512_2048,Gemm_Hipblaslt_NN_52565,0.0653662
GemmTunableOp_float_NN,nn_256_128_512,Gemm_Rocblas_21,0.00793602
```

- **Validator lines** record PT / ROCm / hipBLASLt / GCN arch / rocBLAS versions. If **any** differ at load time, TunableOp rejects the file (and a tuning run **overwrites** it). This is the parity guard — a CSV tuned on gfx942 + ROCm 6.0 is invalid on ROCm 6.4 or gfx950.
- **Entry** = 4 fields: `op_name`, `params`, `solution_name`, `avg_time_ms`.
  - `op_name` e.g. `GemmTunableOp_float_NN` (dtype + transposes).
  - `params` e.g. `nn_1024_512_2048` → transA/transB (`n`/`t`) + M, N, K. Note PyTorch may swap/commute A,B, so M/N can look transposed vs your code.
  - `solution_name` `Gemm_Hipblaslt_...` or `Gemm_Rocblas_<idx>` — tells you which library won.
- **Incremental:** existing entries are not re-tuned across runs; new shapes append one line. Great for iterating on a model — only new shapes pay tuning cost.

---

## 6. Workflow A — online (tune during the run)

```bash
# 1) Tune: race rocBLAS+hipBLASLt for every shape your model hits, write CSV
PYTORCH_TUNABLEOP_ENABLED=1 \
PYTORCH_TUNABLEOP_TUNING=1 \
PYTORCH_TUNABLEOP_VERBOSE=1 \
PYTORCH_TUNABLEOP_FILENAME=tunableop_results.csv \
python serve_or_bench.py        # warmup must exercise all real shapes

# 2) Ship: load CSV, do NOT tune (no per-shape benchmarking at startup)
PYTORCH_TUNABLEOP_ENABLED=1 \
PYTORCH_TUNABLEOP_TUNING=0 \
PYTORCH_TUNABLEOP_FILENAME=tunableop_results.csv \
python serve_or_bench.py

# 3) Baseline for comparison
PYTORCH_TUNABLEOP_ENABLED=0 python serve_or_bench.py
```

Reduce the number of unique shapes (less tuning, more reuse): static KV-cache + pad sequence lengths.

```python
model.generation_config.cache_implementation = "static"
inputs = tokenizer(prompt, return_tensors="pt", padding=True, pad_to_multiple_of=8).to("cuda")
```

---

## 7. Workflow B — offline (PyTorch ≥ 2.6): collect, then tune separately

Decouples tuning from the workload so you don't re-run the model after a math-lib upgrade.

```bash
# Pass 1 — collection: record untuned shapes only (cheap, no benchmarking)
PYTORCH_TUNABLEOP_ENABLED=1 \
PYTORCH_TUNABLEOP_TUNING=0 \
PYTORCH_TUNABLEOP_RECORD_UNTUNED=1 \
PYTORCH_TUNABLEOP_UNTUNED_FILENAME=tunableop_untuned.csv \
python serve_or_bench.py
```

```python
# Pass 2 — tune the collected shapes offline (single or multi-GPU)
import torch
torch.cuda.tunable.enable(True)
torch.cuda.tunable.set_filename("tunableop_results.csv", insert_device_ordinal=True)
# single GPU:
torch.cuda.tunable.tune_gemm_in_file("tunableop_untuned.csv")
# 8x MI300X in parallel (pattern must contain a wildcard):
torch.cuda.tunable.mgpu_tune_gemm_in_file("tunableop_untuned%d.csv", 8)
```

Then ship with `PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0`.

---

## 8. Multi-GPU & parity considerations

- TunableOp produces a **per-GPU CSV** (use `insert_device_ordinal=True` or `%d` in the filename). Each GPU tunes independently, no inter-GPU communication.
- For **data-parallel** workloads the per-GPU shapes are identical → redundant tuning. Tune on one GPU and reuse, or use `mgpu_tune_gemm_in_file` to parallelize the offline pass.
- **Validator parity:** the CSV is bound to PT/ROCm/hipBLASLt/rocBLAS/arch versions. Pin those in your serving image and re-tune on upgrade.
- **Numerical parity:** enable `PYTORCH_TUNABLEOP_NUMERICAL_CHECK=1e-5_1e-5` while tuning to reject any candidate that diverges; disable it for the ship run (it slows tuning).

---

## 9. Caveats / gotchas

- **OOM during tuning** on MI300X: tuning can allocate large workspaces and has been reported to spike memory (a workload that fits with tuning off may OOM with it on). Watch memory in the tuning pass; reduce `MAX_TUNING_ITERATIONS` / disable rotating buffer if needed.
- **Tuning is slow:** verbose tuning can add 1–2 min (TGI) or much more for many shapes. Use offline tuning + ship mode in production.
- **Not always a win:** always compare against `PYTORCH_TUNABLEOP_ENABLED=0`; keep the CSV only if it's faster.
- **Relationship to hipBLASLt offline tuning:** TunableOp races whole-library solutions through PyTorch; hipBLASLt's `HIPBLASLT_TUNING_FILE` tunes within hipBLASLt only. They are complementary — TunableOp is the easy front door; hipBLASLt offline/TensileLite is for deeper, library-level wins.

---

## 10. Serving-stack integration

- **TGI** (AMD image) bundles TunableOp: an extra warmup selects the best rocBLAS/hipBLASLt GEMM; enabled by default, ~1–2 min warmup, ~6–8% latency win (ROCm 6.1 / PT 2.3).
- **vLLM / SGLang:** set `PYTORCH_TUNABLEOP_ENABLED=1` (+ `TUNING=0` after the CSV exists) in the container env; combine with `TORCH_BLAS_PREFER_HIPBLASLT=1`. For FP8 paths, prefer AITER/hipBLASLt's own tuning; TunableOp covers the plain `torch.matmul`/`Linear` GEMMs.
- Bake the tuned CSV into the image and run ship mode (`TUNING=0`) so startup is fast and deterministic.

---

## Sources
- PyTorch TunableOp source & README (env vars, `torch.cuda.tunable` API): https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/cuda/tunable
- AMD ROCm Blog, "Accelerating models on ROCm using PyTorch TunableOp" (CSV format, examples): https://rocm.blogs.amd.com/artificial-intelligence/pytorch-tunableop/README.html
- ROCm "AMD Instinct MI300X workload optimization" (TunableOp + GEMM tuning): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- rocBLAS documentation (`rocblas_gemm_ex`, Tensile, solution index): https://rocm.docs.amd.com/projects/rocBLAS/en/latest/
- HuggingFace TGI AMD install docs (TunableOp integration): https://github.com/huggingface/text-generation-inference/blob/main/docs/source/installation_amd.md
- PyTorch issue #138532 (MI300X TunableOp memory/OOM behavior): https://github.com/pytorch/pytorch/issues/138532
