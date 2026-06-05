# hipBLASLt + Tensile/TensileLite — GEMM on MI300X (CDNA3 / gfx942)

> AMD-only. hipBLASLt is AMD's "lt" GEMM library (analog of cuBLASLt). It is the **preferred GEMM backend** for LLM inference on Instinct. Targets gfx942 (MI300X/MI325X), gfx950 (MI350X/MI355X), gfx90a, gfx110x. All FP8 GEMM on gfx942 is **FNUZ** (E4M3FNUZ / E5M2FNUZ).
>
> Repo note: `ROCm/hipBLASLt` is being consolidated into `ROCm/rocm-libraries`; the code, docs, and APIs below remain current. Binaries still install under `/opt/rocm`.

This file covers the `hipblasLtMatmul` API, layouts and epilogues, the Tensile/TensileLite backend and solution DB, gfx942 solution selection, offline tuning with `hipblaslt-bench` + `HIPBLASLT_TUNING_FILE` / `HIPBLASLT_TUNING_OVERRIDE_FILE`, FP8 GEMM + scaling modes, the "no tuned config" fallback, and the `hipblaslt_ext` API. Python access is via PyTorch `torch._scaled_mm` / TunableOp.

---

## 1. What hipBLASLt computes

```
D = Activation( alpha * op(A) * op(B) + beta * op(C) + bias )
```

- `op(X)` is transpose / non-transpose (`HIPBLAS_OP_N` / `HIPBLAS_OP_T`).
- `alpha`, `beta` are host or device scalars.
- `bias` is a length-rows(D) vector broadcast across columns.
- `Activation ∈ {none, GELU, ReLU, Swish/SiLU}` (the epilogue).

Versus `hipblas`/rocBLAS: hipBLASLt lets you build a **matmul plan** once (descriptors + algo) and reuse it across many calls, exposes **epilogues** (bias/activation/aux), and supports **FP8 scaling**. It is the recommended LLM GEMM path: set `TORCH_BLAS_PREFER_HIPBLASLT=1` to prefer it over hipBLAS in PyTorch.

---

## 2. Core C++ API objects

| Object / call | Purpose |
|---|---|
| `hipblasLtHandle_t` | Library handle (`hipblasLtCreate`) |
| `hipblasLtMatrixLayout_t` | Per-matrix layout: dtype, rows, cols, leading dim, batch, batch stride |
| `hipblasLtMatmulDesc_t` | Op descriptor: compute type, scale type, transposes, epilogue, bias, scale pointers |
| `hipblasLtMatmulPreference_t` | Constraints (max workspace size) for heuristic search |
| `hipblasLtMatmulAlgoGetHeuristic` | Returns ranked candidate algos for the problem |
| `hipblasLtMatmul` | Executes the GEMM with a chosen algo |

### 2.1 Minimal matmul (bf16, with bias + GELU)

```cpp
hipblasLtHandle_t handle;  hipblasLtCreate(&handle);

// Layouts: A[m,k] row/col-major via leading dim; here col-major NN
hipblasLtMatrixLayout_t aL, bL, cL, dL;
hipblasLtMatrixLayoutCreate(&aL, HIP_R_16BF, m, k, m);
hipblasLtMatrixLayoutCreate(&bL, HIP_R_16BF, k, n, k);
hipblasLtMatrixLayoutCreate(&cL, HIP_R_16BF, m, n, m);
hipblasLtMatrixLayoutCreate(&dL, HIP_R_16BF, m, n, m);

hipblasLtMatmulDesc_t op;
hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32F, HIP_R_32F);   // fp32 accumulate
hipblasOperation_t opN = HIPBLAS_OP_N;
hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

// Epilogue: bias + GELU
hipblasLtEpilogue_t epi = HIPBLASLT_EPILOGUE_GELU_BIAS;
hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi));
hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));

// Heuristic pick under a workspace budget
hipblasLtMatmulPreference_t pref; hipblasLtMatmulPreferenceCreate(&pref);
size_t wsMax = 128ull*1024*1024;
hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                      &wsMax, sizeof(wsMax));
hipblasLtMatmulHeuristicResult_t res[8]; int got = 0;
hipblasLtMatmulAlgoGetHeuristic(handle, op, aL, bL, cL, dL, pref, 8, res, &got);

// Execute the top candidate
hipblasLtMatmul(handle, op, &alpha, A, aL, B, bL, &beta, C, cL, D, dL,
                &res[0].algo, workspace, wsMax, stream);
```

### 2.2 Epilogue enum (`hipblasLtEpilogue_t`)

| Value | Effect |
|---|---|
| `HIPBLASLT_EPILOGUE_DEFAULT` | none |
| `HIPBLASLT_EPILOGUE_BIAS` | + bias |
| `HIPBLASLT_EPILOGUE_RELU` / `..._RELU_BIAS` | ReLU (+bias) |
| `HIPBLASLT_EPILOGUE_GELU` / `..._GELU_BIAS` | GELU (+bias) |
| `HIPBLASLT_EPILOGUE_GELU_AUX` / `..._GELU_AUX_BIAS` | GELU writing pre-activation aux (for backward) |
| `HIPBLASLT_EPILOGUE_SWISH_EXT` / `..._SWISH_BIAS_EXT` | Swish/SiLU (+bias) |

---

## 3. The `hipblaslt_ext` C++/Python-friendly API

`hipblaslt_ext` wraps the verbose descriptor flow in higher-level objects, used heavily by frameworks:

| Class | Purpose |
|---|---|
| `hipblaslt_ext::Gemm` | Single GEMM instance (set problem, get algos, run) |
| `hipblaslt_ext::GroupedGemm` | Grouped/variable GEMM: `D = alpha*(A*B) + beta*C` over a list of problems |
| `GemmProblemTypeV2`, `GemmEpilogueV2`, `GemmInputsV2`, `GemmPreferenceV2`, `GemmTuningV2` | Structured problem/epilogue/input/pref/tuning descriptors |
| `setScalingAType` / `setScalingBType` | FP8 scaling mode (scalar=0 / vector=1); only valid when DataTypeA == DataTypeB == FP8 |

> API evolution: the original non-V2 structs were unified with the V2 variants (old non-V2 removed); the V2 names are now themselves deprecated as the API re-stabilizes. Check your hipBLASLt version's `ext-reference` docs.

**Python:** there is no first-class pip `hipblaslt` module for general use — frameworks reach hipBLASLt through **PyTorch**: `torch.matmul`/`nn.Linear` (when `TORCH_BLAS_PREFER_HIPBLASLT=1`) and `torch._scaled_mm(a, b, scale_a, scale_b, out_dtype=...)` for FP8. SGLang/vLLM call `torch._scaled_mm` for FP8 linear on ROCm.

---

## 4. Tensile / TensileLite backend & the solution DB

hipBLASLt's kernels are generated by **TensileLite** (a hipBLASLt-internal fork of Tensile). Concepts:

| Concept | Meaning |
|---|---|
| **Solution** | One concrete generated kernel: tile size (MT0×MT1), depth-U (K-tile), MFMA instruction, global-split-K, workgroup, scheduling/pipeline. |
| **Solution index** | Integer ID of a solution in the built library for an arch. **Not stable across ROCm versions or archs.** |
| **Logic YAML** | Per-arch tuning DB mapping problem (transpose, dtype, M/N/K) → ranked solutions. For gfx942 lives under `library/src/.../Tensile/Logic/asm_full/aquavanjaram/gfx942/`. |
| **Heuristic** | At runtime, `AlgoGetHeuristic` consults the logic DB + size to rank candidate solutions. |

### 4.1 Build hipBLASLt for gfx942 only (fast compile, < ~2h)

```bash
git clone https://github.com/ROCm/hipBLASLt && cd hipBLASLt
python3 -m pip install -r tensilelite/requirements.txt
./install.sh -idc --logic-yaml-filter "gfx942/*/*" -a gfx942 -j 256 --build_dir build
# -i install, -d deps, -c clients (gives hipblaslt-bench)
```

---

## 5. gfx942 solution selection & the "no tuned config" fallback

When you call `hipblasLtMatmulAlgoGetHeuristic`:
1. The heuristic looks up the gfx942 logic DB by (transA, transB, dtypes, M/N/K bucket).
2. If a **tuned entry** exists, it returns that solution highly ranked.
3. If the **exact shape is not in the DB**, it falls back to the nearest/"Equality" or generic logic — often a *generically reasonable but not optimal* kernel. This is the common "config not found / not tuned" case where you leave 10–40% perf on the table for odd LLM shapes.
4. If `HIPBLASLT_TUNING_OVERRIDE_FILE` has a matching entry, the heuristic **overrides** its choice with your tuned solution index.

The remedy for (3) is offline tuning (next section) or TensileLite kernel generation for truly novel shapes.

---

## 6. Offline tuning workflow (`hipblaslt-bench` + tuning files)

`hipblaslt-bench` is the offline benchmark/tuner client (built with `--clients`, lands in `build/.../clients/staging/`).

### 6.1 Step 1 — capture the real GEMM shapes from your workload

```bash
export HIPBLASLT_LOG_MASK=32                  # log GEMM problems as bench command lines
export HIPBLASLT_LOG_FILE=dump_gemm_shapes.txt
python my_serving_run.py                      # run your actual model briefly
unset HIPBLASLT_LOG_MASK HIPBLASLT_LOG_FILE
```

### 6.2 Step 2 — tune each shape and write a tuning file

```bash
export HIPBLASLT_TUNING_FILE=tuning.txt       # enables tuning mode + records best solution index
# Example FP8 GEMM tune (M=N=K=4096, NT, E4M3FNUZ in, bf16 out):
./hipblaslt-bench \
  --a_type f8_r --b_type f8_r --c_type bf16_r --d_type bf16_r \
  --compute_type f32_r --scale_type f32_r \
  --transA N --transB T -m 4096 -n 4096 -k 4096 \
  --algo_method all --cold_iters 50 --iters 50
unset HIPBLASLT_TUNING_FILE
```

`--algo_method` modes:
| Mode | Behavior |
|---|---|
| `heuristic` | tune only the top heuristic candidates (fast) |
| `all` | tune every solution in the pool (slowest, best) |
| `index` | benchmark one specific solution index |

Useful knobs: `--cold_iters` (warmup), `--iters` (measured), `HIPBLASLT_TUNING_USER_MAX_WORKSPACE=<bytes>` (constrain the chosen solution's workspace; default `128*1024*1024`).

### 6.3 Step 3 — deploy at runtime via override

```bash
unset HIPBLASLT_TUNING_FILE
export HIPBLASLT_TUNING_OVERRIDE_FILE=tuning.txt   # heuristic now returns your tuned indices
python my_serving_run.py
```

> Re-tune after **any** ROCm / hipBLASLt upgrade or arch change — solution indices are not portable.

### 6.4 Env var cheat sheet

| Variable | Effect |
|---|---|
| `HIPBLASLT_TUNING_FILE` | Tuning mode: record best solution index per shape into file |
| `HIPBLASLT_TUNING_OVERRIDE_FILE` | Runtime: override heuristic with indices from file |
| `HIPBLASLT_TUNING_USER_MAX_WORKSPACE` | Constrain workspace during tuning (default 128 MiB) |
| `HIPBLASLT_LOG_MASK=32` | Emit GEMM problems as bench command lines |
| `HIPBLASLT_LOG_FILE` | Where logs/shape dumps go |
| `TORCH_BLAS_PREFER_HIPBLASLT=1` | PyTorch prefers hipBLASLt over hipBLAS |

### 6.5 Higher-level automation

- **Primus `offline_tune_gemm.py`** — batch-tunes many dumped shapes across multiple GPUs, then emits a file for `HIPBLASLT_TUNING_OVERRIDE_FILE`.
- **QuickTune (Quark team)** — one-click offline GEMM tuning wrapper over hipblaslt-bench.
- **`find_exact.py` + TensileLite** — recompilation path: generates *new* kernels (logic YAML) merged into the library source (e.g. `.../gfx942/Equality/`). Use when the shape needs a kernel the pool doesn't contain. Heavier (rebuild) but best for fixed shapes.

Online alternative: `VLLM_ROCM_USE_AITER_HIP_ONLINE_TUNING=1` makes the GEMM wrapper tune candidate algos at runtime on first sight of a new shape and cache the winner — no offline pass needed.

---

## 7. FP8 GEMM & scaling on gfx942

CDNA3 (gfx942) has native FP8-FNUZ MFMA. The scaled matmul applies per-tensor or per-block scales:

```
D = (scaleD) * Activation( scaleA * op(A_fp8) * scaleB * op(B_fp8) + beta*C )
```

Scaling is configured on the matmul descriptor:

| Descriptor attribute | Mode constant | Meaning |
|---|---|---|
| `HIPBLASLT_MATMUL_DESC_A_SCALE_MODE` / `_B_SCALE_MODE` | `HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F` | per-tensor scalar fp32 scale |
| same | `HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F` | vector (per-row/col) fp32 scale |
| same | `HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0` | block scaling (32-elem blocks, UE8M0 exponent) — MX-style |
| `HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER` / `_B_/_D_` | — | device pointer(s) to scale tensor(s) |

`hipblaslt-bench` FP8 winner CSV columns include `scaleA, scaleB, scaleC, scaleD, amaxD`. For FP8 throughput, recent hipBLASLt also supports **swizzle GEMM** (rearranged layouts) — order `HIPBLASLT_ORDER_COL16_4R8` for FP8/BF8 swizzle, exposed in `hipblaslt-bench`.

> gfx950 (MI350/MI355) adds OCP FP8 and FP4/MXFP4. On gfx942 stick to FNUZ types (`f8_r` = E4M3FNUZ, `bf8_r` = E5M2FNUZ in bench flags).

AMD reports up to ~2x from FP8 GEMM vs bf16 on the right shapes, and an improved heuristic search for 8/16/32-bit batched GEMM on gfx942 in ROCm 7.x.

---

## 8. End-to-end tuning recipe (LLM serving on MI300X)

1. `TORCH_BLAS_PREFER_HIPBLASLT=1` so PyTorch uses hipBLASLt.
2. Dump shapes: `HIPBLASLT_LOG_MASK=32 HIPBLASLT_LOG_FILE=shapes.txt python serve.py` (short run).
3. Tune: feed `shapes.txt` to `hipblaslt-bench --algo_method all` under `HIPBLASLT_TUNING_FILE=tuning.txt` (or Primus/QuickTune to batch it).
4. Deploy: `HIPBLASLT_TUNING_OVERRIDE_FILE=tuning.txt python serve.py`.
5. Verify: compare tok/s vs untuned; for shapes still slow, escalate to `find_exact.py`/TensileLite.
6. Re-run after any ROCm/hipBLASLt bump.

---

## Sources
- ROCm hipBLASLt repository (API, `tensilelite/`, clients): https://github.com/ROCm/hipBLASLt
- "Using hipBLASLt offline tuning" — hipBLASLt docs: https://rocm.docs.amd.com/projects/hipBLASLt/en/develop/how-to/how-to-use-hipblaslt-offline-tuning.html
- hipBLASLtExt API reference (`hipblaslt_ext`, GroupedGemm, scaling setters): https://rocm.docs.amd.com/projects/hipBLASLt/en/develop/reference/ext-reference.html
- AMD ROCm Blog, "GEMM Tuning within hipBLASLt – Part 2" (hipblaslt-bench, FP8): https://rocm.blogs.amd.com/software-tools-optimization/hipblaslt-offline-tuning-part2/README.html
- AMD ROCm Blog, "Day 0 Developer Guide: hipBLASLt Offline GEMM Tuning Script": https://rocm.blogs.amd.com/artificial-intelligence/hipblaslt_offline_tuning/README.html
- AMD ROCm Blog, "Customizing Kernels with hipBLASLt TensileLite GEMM Tuning": https://rocm.blogs.amd.com/artificial-intelligence/hipblaslt-tensilelite-tuning/README.html
- Primus offline_tune workflow: https://github.com/AMD-AGI/Primus/blob/main/examples/offline_tune/README.md
