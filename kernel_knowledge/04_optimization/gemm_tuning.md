# GEMM Tuning Playbook — AMD Instinct MI300X (CDNA3 / gfx942)

> Scope: AMD only. Target is MI300X (gfx942, CDNA3). Notes for gfx950 (MI350/MI355, CDNA4) are flagged inline. This file is the end-to-end playbook for getting a GEMM (or a whole serving workload's GEMM set) to peak on MI300X across the four tuning surfaces you will actually use in production:
>
> 1. **hipBLASLt** — offline tuning via `hipblaslt-bench` + `HIPBLASLT_TUNING_FILE` / override file, and full **TensileLite** kernel generation.
> 2. **PyTorch TunableOp** — online + offline (PyTorch 2.6+) auto-selection over hipBLASLt/rocBLAS solutions.
> 3. **Triton autotune** — config-space design for the AMD-specific knobs (`matrix_instr_nonkdim`, `waves_per_eu`, `kpack`, `BLOCK_*`, `SPLIT_K`, `num_stages`, `num_warps`, `GROUP_M`).
> 4. **Composable Kernel** — `ckProfiler` instance tuning + epilogue knobs.
>
> Everything below is concrete: commands, env vars, config snippets, and a step-by-step recipe to build a per-shape tuned config DB for a serving workload.

---

## 0. Why GEMM tuning matters on MI300X specifically

MI300X has enormous theoretical FLOPs (≈1307 TFLOP/s BF16/FP16 dense, ≈2615 TFLOP/s FP8 dense with the MFMA engines) and 192 GB HBM3 at ≈5.3 TB/s. But measured sustained efficiency on LLM serving GEMMs is typically **45–55% of peak** with stock libraries because:

- The chip is **8 XCDs × 38 CUs = 304 CUs**. A GEMM grid must produce **≥1024 workgroups** to fill the machine and tolerate the round-robin XCD scheduler; small/skinny GEMMs (decode: M=1..256) leave CUs idle → **split-K / stream-K** is mandatory.
- The best MFMA tile on CDNA3 is **`mfma_16x16` (`matrix_instr_nonkdim=16`)**, which beats `mfma_32x32` even for large tiles due to better power/clock behavior. Most CUDA-derived defaults pick 32x32 → leaving perf on the table.
- Register file is **512×32-bit VGPR per SIMD split 256 VGPR + 256 AGPR** when 1 wave/SIMD. MFMA accumulators live in **AGPRs**. VGPR is allocated in granules of 16, so being a few registers over a boundary silently halves occupancy.
- Library heuristics ship generic solutions; a per-shape exhaustive search routinely yields **1.5–7.2×** end-to-end gains (see Nscale MI300X GEMM-tuning benchmark).

> **Golden rule:** never trust the default solution for a production shape. Tune per (M,N,K, dtype, transA/B) and persist the result.

---

## 1. hipBLASLt offline tuning (the primary path)

hipBLASLt is the default BLAS backend for PyTorch/vLLM/SGLang on Instinct. Offline tuning lets you pick the best **solution index** per problem shape ahead of time, store it in a file, and override at runtime — **zero rebuild, deterministic, no runtime search cost.**

### 1.1 Three-stage workflow

```
                ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
   workload ──▶ │ 1. DUMP      │──▶│ 2. TUNE      │──▶│ 3. OVERRIDE  │──▶ production
                │ shapes       │   │ search best  │   │ apply tuned  │
                │ LOG_MASK=32  │   │ TUNING_FILE  │   │ OVERRIDE_FILE│
                └──────────────┘   └──────────────┘   └──────────────┘
```

### Stage 1 — Dump the GEMM shapes your workload actually issues

```bash
export HIPBLASLT_LOG_MASK=32                 # 32 = log GEMM problem (bench-replayable) lines
export HIPBLASLT_LOG_FILE=dump_gemm_shapes.txt
python -m vllm.entrypoints.openai.api_server --model <m> ...   # run a few steps; same shapes repeat
unset HIPBLASLT_LOG_MASK HIPBLASLT_LOG_FILE
```

`dump_gemm_shapes.txt` now holds one replayable `hipblaslt-bench` invocation per distinct GEMM. You only need a handful of iterations — every decode step reuses the same shapes.

### Stage 2 — Tune (exhaustive search → best solution index per shape)

Point `HIPBLASLT_TUNING_FILE` at an output file, then replay each shape. With `--algo_method all` (or `heuristic`) hipBLASLt benchmarks the candidate pool and records the winning `solution_index`:

```bash
export HIPBLASLT_TUNING_FILE=tuning.txt
# Optionally bound workspace so the picked solution fits your runtime budget:
export HIPBLASLT_TUNING_USER_MAX_WORKSPACE=$((256*1024*1024))   # default 128 MiB

/opt/rocm/bin/hipblaslt-bench --api_method c \
  -m 4096 -n 4096 -k 4096 \
  --lda 4096 --ldb 4096 --ldc 4096 --ldd 4096 \
  --alpha 1.0 --beta 0.0 --transA N --transB T --batch_count 1 \
  --a_type f16_r --b_type f16_r --c_type f16_r --d_type f16_r \
  --scale_type f32_r --compute_type f32_r \
  --algo_method all -i 100 -j 50           # -i iters, -j warmup; search all solutions
```

For FP8 (E4M3) shapes use `--a_type f8_r --b_type f8_r --c_type f16_r --compute_type f32_r` and add `--scaleA 1 --scaleB 1` (or the actual scale layout). To pin a specific solution and just measure it:

```bash
/opt/rocm/bin/hipblaslt-bench ... --algo_method index --solution_index 56073
```

### Stage 3 — Override at runtime

```bash
unset HIPBLASLT_TUNING_FILE
export HIPBLASLT_TUNING_OVERRIDE_FILE=tuning.txt    # runtime reads tuned indices
python -m vllm.entrypoints.openai.api_server ...     # now uses tuned solutions
```

At runtime `hipblasLtMatmulAlgoGetHeuristic` (C) / `algoGetHeuristic` (C++) returns the tuned solution whenever a matching (shape, dtype, layout) entry exists; otherwise it falls back to the heuristic. This is what vLLM/SGLang pick up transparently.

### 1.2 Key hipBLASLt env vars

| Env var | Purpose | Default |
|---|---|---|
| `HIPBLASLT_LOG_MASK=32` | dump bench-replayable GEMM problem lines | off |
| `HIPBLASLT_LOG_FILE=<path>` | where logs go | stderr |
| `HIPBLASLT_TUNING_FILE=<path>` | **write** tuned solution indices during search | unset |
| `HIPBLASLT_TUNING_OVERRIDE_FILE=<path>` | **read** tuned indices at runtime | unset |
| `HIPBLASLT_TUNING_USER_MAX_WORKSPACE=<bytes>` | cap workspace the tuned solution may use | 128 MiB |
| `TORCH_BLAS_PREFER_HIPBLASLT=1` | make PyTorch prefer hipBLASLt over rocBLAS | varies by version |
| `HIPBLASLT_ENABLE_MARKER=1` | emit roctx markers for profiling | off |

> ⚠️ **Solution indices are NOT stable across ROCm versions.** Re-run tuning on every ROCm/hipBLASLt upgrade. Pin the ROCm version of your tuning DB.

### 1.3 hipblaslt-bench essential flags

| Flag | Meaning |
|---|---|
| `-m -n -k` | GEMM dims (M = rows of C, N = cols, K = contraction) |
| `--transA / --transB` | `N`/`T` — `NT` (A row-major, B col-major) is the common LLM weight layout |
| `--lda/--ldb/--ldc/--ldd` | leading dims |
| `--a_type/--b_type/--c_type/--d_type` | `f16_r`, `bf16_r`, `f8_r`, `f32_r`, `i8_r` |
| `--compute_type` | usually `f32_r` |
| `--scale_type`, `--bias_type` | epilogue scale/bias types |
| `--algo_method` | `heuristic` \| `index` \| `all` |
| `--solution_index` | pin a solution (with `index`) |
| `-i / -j` | timing iters / warmup iters |
| `--initialization` | `rand_int` / `trig_float` / `hpl` |
| `--flush` | flush caches between iters (more honest cold numbers) |

### 1.4 Going further: TensileLite kernel generation

If no pooled solution is fast enough for your shape, generate **new** kernels with TensileLite (the kernel-gen layer under hipBLASLt). Workflow (from the ROCm "hipBLASLt TensileLite Tuning" advanced guide):

1. Dump shapes (Stage 1 above).
2. Use `find_exact.py` / the TensileLite `Tensile` driver with a problem YAML and a **tuning logic YAML** describing the solution search space (tile sizes, MFMA, depthU, GSU, etc.).
3. Build only gfx942 logic for speed:

```bash
./install.sh -idc --logic-yaml-filter "gfx942/*/*" -a gfx942 -j 256 --build_dir build
# -i install, -d deps, -c client; gfx942-only logic cuts build to < 2h
```

4. Merge the winning solutions into the library logic and rebuild, or export to an override file.

> **QuickTune** (AMD-AGI/Primus `examples/offline_tune`) wraps this whole loop into a near one-click tool — feed it a model + shapes, it produces the tuned solution DB.

---

## 2. PyTorch TunableOp

TunableOp is an in-PyTorch auto-tuner: for every GEMM call it benchmarks all available hipBLASLt/rocBLAS implementations and caches the fastest. It is the lowest-effort path for any PyTorch-based serving stack (vLLM, SGLang, raw HF).

### 2.1 Online tuning (simplest)

```bash
export PYTORCH_TUNABLEOP_ENABLED=1          # master switch (default 0)
export PYTORCH_TUNABLEOP_TUNING=1           # tune-on-miss (default 1)
export PYTORCH_TUNABLEOP_VERBOSE=1          # confirm it's active
export PYTORCH_TUNABLEOP_FILENAME=tunableop_results.csv
export TORCH_BLAS_PREFER_HIPBLASLT=1
python serve.py        # first pass is slow (it's tuning); results written on exit
```

Output `tunableop_results0.csv` (one file per GPU rank, suffix = device id) records the chosen op per (shape, dtype, transpose). An entry of `Default` means TunableOp found nothing faster than hipBLASLt's default for that shape.

### 2.2 Offline tuning (PyTorch 2.6+) — decoupled collect → tune

Avoids re-running the whole expensive workload every time you re-tune (e.g., after a math-lib upgrade). Two passes:

1. **Collection pass** — record the GEMMs the workload issues (set record-untuned mode), producing an untuned-GEMM file.
2. **Tuning pass** — replay just those GEMMs with the offline tuner (no model needed), writing the results CSV.

ROCm 7.x / PyTorch 2.9 additions: TF32 support in TunableOp, improved **ScaledGEMM (FP8)** offline tuning, **submatrix** offline tuning, and better logging for bias-less BLAS.

### 2.3 Key TunableOp env vars

| Env var | Purpose | Default |
|---|---|---|
| `PYTORCH_TUNABLEOP_ENABLED` | master on/off | 0 |
| `PYTORCH_TUNABLEOP_TUNING` | tune when no cached entry | 1 |
| `PYTORCH_TUNABLEOP_VERBOSE` | debug output | 0 |
| `PYTORCH_TUNABLEOP_FILENAME` | results CSV base name | `tunableop_results.csv` |
| `PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS` | cap per-op tuning time | — |
| `PYTORCH_TUNABLEOP_NUMERICAL_CHECK` | validate result correctness during tuning | 0 |
| `PYTORCH_TUNABLEOP_ROTATING_BUFFER_SIZE` | rotate inputs to defeat cache (honest timing) | — |

> **Caveat:** as hipBLASLt heuristics improve release-over-release, TunableOp's marginal win shrinks — sometimes it picks `Default` for most shapes. Always measure end-to-end before/after; don't assume a win.

---

## 3. Triton GEMM autotune — config-space design

When you write the GEMM in Triton (custom fused epilogue, MoE grouped GEMM, novel dtype), you own the autotune space. The AMD-specific knobs are what make or break it.

### 3.1 The MI300X tuning knobs

| Knob | What it controls | MI300X guidance |
|---|---|---|
| `BLOCK_M`, `BLOCK_N`, `BLOCK_K` | output tile + K-step | Start 128×128×64 / 256×128×64 for big GEMMs; 16/32×… for skinny decode |
| `matrix_instr_nonkdim` | MFMA size: 16→`mfma_16x16`, 32→`mfma_32x32` | **Prefer 16.** Beats 32 even on large tiles |
| `kpack` | K elements packed per MFMA issue (1 or 2) | Try 2 to raise per-instruction work; validate it helps |
| `waves_per_eu` | occupancy hint → LLVM lowers VGPR to fit N waves/EU | 1–4; raising it forces register diet (watch spills) |
| `num_warps` | wavefronts per workgroup (each = 64 lanes) | 4 or 8 typical; 8 for big tiles |
| `num_stages` | software-pipeline depth (global→LDS prefetch) | 0 for single-GEMM on CDNA; 1–2 also tried |
| `SPLIT_K` / `GROUP_M` | K-axis parallelism / L2-friendly block swizzle | SPLIT_K 2–16 for skinny GEMM; GROUP_M 4–8 |
| `OPTIMIZE_EPILOGUE` (env) | keep MFMA-layout result vs reblock for stores | 1 trims a reblock at cost of store efficiency |

> **CDNA `num_stages` rule of thumb** (from the MI300X workload guide):
> - single GEMM kernel → `num_stages = 0`
> - two fused GEMMs (FlashAttention) → `num_stages = 1`
> - 1 GEMM fused with a non-GEMM op → `num_stages = 0`
> - no-GEMM kernel → `num_stages = 1`

### 3.2 A real MI300X GEMM autotune config block

```python
import triton

def mi300x_gemm_configs():
    configs = []
    for BM, BN, BK in [(128,128,64), (128,256,64), (256,128,64),
                       (256,256,64), (128,128,128), (64,64,64)]:
        for nw in (4, 8):
            for wpe in (1, 2, 4):
                for sk in (1, 2, 4, 8):          # SPLIT_K — crucial for skinny GEMM
                    configs.append(triton.Config(
                        {"BLOCK_SIZE_M": BM, "BLOCK_SIZE_N": BN,
                         "BLOCK_SIZE_K": BK, "GROUP_SIZE_M": 8,
                         "SPLIT_K": sk,
                         "matrix_instr_nonkdim": 16,   # mfma_16x16 — MI300X sweet spot
                         "kpack": 2,
                         "waves_per_eu": wpe},
                        num_warps=nw,
                        num_stages=0))               # single-GEMM → 0 on CDNA
    return configs

@triton.autotune(configs=mi300x_gemm_configs(),
                 key=["M", "N", "K"])               # re-tune per shape bucket
@triton.jit
def gemm_kernel(...): ...
```

When `SPLIT_K > 1`, the kernel accumulates partial K-tiles across blocks and combines via `tl.atomic_add` (or a separate reduction kernel). This is what gets a decode-time M=1..256 GEMM to use all 304 CUs.

### 3.3 PyTorch Inductor max-autotune (uses the same knobs)

```bash
export TORCHINDUCTOR_MAX_AUTOTUNE=1
# or in code:
#   torch._inductor.config.max_autotune = True
```

The 2025 PyTorch PR (#143286) added ROCm-specific MM configs sweeping `waves_per_eu`, `kpack`, `matrix_instr_nonkdim`, and `GROUP_M` for `torch.compile`'s Triton GEMM template (HF inference geomean 1.35×→1.42×).

> **Analytical alternative:** `tritonBLAS` (ROCm, Dec 2025) replaces runtime autotune with an analytical model that picks the config directly, reaching ~94.7% of exhaustive-search perf on MI300X (Triton 3.4.0). Use it when autotune startup cost is intolerable.

---

## 4. Composable Kernel (CK) instance tuning

CK is the C++ template library behind many AITER/hipBLASLt paths. Use `ckProfiler` to find the best instance and `OPTIMIZE_EPILOGUE` for the store path.

### 4.1 ckProfiler GEMM sweep

```bash
# args: <op> <datatype> <layout> <verify> <init> <log> <time> M N K  strideA strideB strideC [splitK]
#  datatype: 0=fp32 1=fp16 ...   layout: 0=NN 1=NT 2=TN 3=TT
./bin/ckProfiler gemm_universal 1 1 1 1 0 1 4096 4096 4096 -1 -1 -1
```

It enumerates compiled CK instances (tile sizes, MFMA, pipeline, GSU/split-K) and reports TFLOP/s per instance + the best. Use the winning instance's traits to specialize a CK kernel, or feed it into AITER.

> CK is moving **header-only** in an upcoming ROCm major release: `ckProfiler` and static libs will no longer ship by default and must be built explicitly.

### 4.2 Epilogue knob for MI300X

```bash
export OPTIMIZE_EPILOGUE=1   # store MFMA result in MFMA layout directly (skip reblock)
```

`OPTIMIZE_EPILOGUE=1` avoids converting the MFMA accumulator to a blocked layout before the global store — saves a reblock at the cost of lower `global_store` vector width; net usually faster for fused epilogues. Default `0` maximizes store-vector length.

ROCm 7.2 CK added **GEMM+GEMM fusion** and **Fused Clamp GEMM** (`HIPBLASLT_EPILOGUE_CLAMP_EXT`, `..._CLAMP_BIAS_EXT`).

---

## 5. End-to-end recipe: per-shape tuned DB for a serving workload

```
Step 0  Pin environment
  export TORCH_BLAS_PREFER_HIPBLASLT=1
  export HIP_FORCE_DEV_KERNARG=1            # vLLM kernarg perf
  export NCCL_MIN_NCHANNELS=112             # MI300X collective channels (multi-GPU)
  Record exact ROCm + hipBLASLt + framework versions (tuning is version-locked).

Step 1  Collect shapes
  HIPBLASLT_LOG_MASK=32 HIPBLASLT_LOG_FILE=shapes.txt  python serve.py --warmup-only
  → de-dup shapes.txt into a unique (M,N,K,dtype,trans) set.

Step 2  Pick a tuning surface per shape class
  • Standard dense GEMM, dtype supported by hipBLASLt → hipBLASLt offline tune (§1).
  • PyTorch-internal linears you don't control → TunableOp offline (§2.2).
  • Custom fused / grouped-MoE GEMM → Triton autotune (§3) or CK (§4).

Step 3  Tune
  for each shape:  hipblaslt-bench --algo_method all ...     (HIPBLASLT_TUNING_FILE=tuning.txt)
  (or QuickTune / Primus offline_tune to batch the loop)

Step 4  Validate accuracy
  Replay each tuned GEMM with --algo_method index + a reference; check max-abs / rel error.
  (TunableOp: PYTORCH_TUNABLEOP_NUMERICAL_CHECK=1.)

Step 5  Deploy
  unset HIPBLASLT_TUNING_FILE
  export HIPBLASLT_TUNING_OVERRIDE_FILE=tuning.txt
  (TunableOp: PYTORCH_TUNABLEOP_ENABLED=1, TUNING=0, FILENAME=tunableop_results.csv)

Step 6  Measure end-to-end
  Compare tokens/s & TTFT/ITL before vs after. Keep the DB only if it actually wins.

Step 7  Re-tune on every ROCm/library upgrade. Solution indices are not portable.
```

### 5.1 Decision ladder: which surface?

```
Is the GEMM a standard dense matmul in a supported dtype?
  ├─ yes ─ Do you control the PyTorch graph but not the call sites?
  │         ├─ yes → TunableOp (online to explore, offline to deploy)
  │         └─ no  → hipBLASLt offline tune + OVERRIDE_FILE
  └─ no (fused epilogue / grouped MoE / exotic dtype)
            ├─ writing it in Triton → Triton autotune (matrix_instr_nonkdim=16, sweep SPLIT_K)
            └─ writing it in C++/CK  → ckProfiler instance sweep + OPTIMIZE_EPILOGUE=1
Still not fast enough on a hot shape? → TensileLite kernel generation (new instances).
```

### 5.2 Common pitfalls

- **Forgetting split-K/stream-K on decode GEMMs** → grid < 1024 blocks → idle CUs. Always sweep `SPLIT_K`.
- **`matrix_instr_nonkdim=32`** copied from CUDA tiling → use 16 on MI300X.
- **VGPR boundary**: 176 VGPR × 3 > 512 → only 2 waves/CU. Nudge `waves_per_eu` and watch for spills (AGPR/VGPR spill kills perf).
- **Tuning DB reused across ROCm versions** → stale/invalid solution indices, silent fallback.
- **Tuning with cache hot** (no `--flush`/rotating buffer) → optimistic numbers that don't hold in serving.

---

## Sources

- hipBLASLt offline tuning (official): <https://rocm.docs.amd.com/projects/hipBLASLt/en/develop/how-to/how-to-use-hipblaslt-offline-tuning.html>
- GEMM Tuning within hipBLASLt — Part 1 & 2 (ROCm Blogs): <https://rocm.blogs.amd.com/software-tools-optimization/hipblaslt-offline-tuning-part1/README.html> · <https://rocm.blogs.amd.com/software-tools-optimization/hipblaslt-offline-tuning-part2/README.html>
- hipBLASLt TensileLite tuning (advanced guide): <https://rocm.blogs.amd.com/artificial-intelligence/hipblaslt-tensilelite-tuning/README.html>
- Day-0 hipBLASLt offline GEMM tuning script / QuickTune & Primus offline_tune: <https://rocm.blogs.amd.com/artificial-intelligence/hipblaslt_offline_tuning/README.html> · <https://github.com/AMD-AGI/Primus/blob/main/examples/offline_tune/README.md>
- PyTorch Offline Tuning with TunableOp (ROCm Blogs): <https://rocm.blogs.amd.com/artificial-intelligence/pytorch-tunableop-offline/README.html>
- PyTorch ROCm GEMM tuning params PR (#143286): <https://github.com/pytorch/pytorch/pull/143286>
- AMD MI300X workload optimization guide (Triton/TunableOp/epilogue, num_stages): <https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html>
- tritonBLAS (analytical config selection, arXiv Dec 2025): <https://arxiv.org/html/2512.04226v1>
- Composable Kernel profiler README: <https://github.com/ROCm/composable_kernel/blob/develop/profiler/README.md>
- Nscale MI300X GEMM-tuning benchmark (up to 7.2×): <https://www.nscale.com/blog/nscale-benchmarks-amd-mi300x-gpus-with-gemm-tuning-improves-throughput-and-latency-by-up-to-7-2x>
