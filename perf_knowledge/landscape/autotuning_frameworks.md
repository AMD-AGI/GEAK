---
kind: landscape
title: Autotuning, cost-model, and tuned-config-DB frameworks for GPU kernels
updated: 2026-06-09
gens: [gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8, fp4_e2m1]
regimes: [prefill, decode, both]
sources:
  - https://triton-lang.org/main/python-api/generated/triton.autotune.html
  - https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/tunable/README.md
  - https://rocm.blogs.amd.com/artificial-intelligence/hipblaslt-tensilelite-tuning/README.html
  - https://github.com/ROCm/composable_kernel/blob/develop/profiler/README.md
  - https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html
  - https://kerneltuner.github.io/kernel_tuner/stable/optimization.html
  - https://commit.csail.mit.edu/papers/2014/ansel-pact14-opentuner.pdf
  - https://github.com/IBM/triton-dejavu
  - https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py
  - https://github.com/ROCm/aiter
---

# Autotuning, cost-model, and tuned-config-DB frameworks for GPU kernels

## TL;DR
Every framework here answers one question — *"which kernel variant is fastest for this exact
shape/dtype/hardware, and where do I persist that answer?"* — and they cluster into four search
families: **exhaustive/grid race** (TunableOp, ckProfiler, hipBLASLt offline, vLLM `benchmark_moe`,
cuBLASLt/cuDNN autotune, aiter gradlib), **heuristic-then-verify** (cuBLASLt
`AlgoGetHeuristic`, cuDNN Mode A/B, hipBLASLt grid logic), **classical black-box search**
(Kernel Tuner: BO/GA/PSO/annealing; OpenTuner: AUC-bandit ensemble), and **learned cost models**
(TVM Ansor/MetaSchedule XGBoost; 2025 NeuSight/SynPerf tile-level ML predictors). The
*persisted artifact* is the durable asset: a per-shape CSV (aiter, TunableOp, rocBLAS override),
a per-shape JSON (vLLM MoE, Triton-dejavu cache), a logic YAML (hipBLASLt/TensileLite), or a
JSON tuning-record DB (MetaSchedule). The **correctness gate** is the second durable asset and
the one most frameworks get *informally* — aiter gradlib's `err_ratio < 0.05` and TileLang's
`rtol/atol + max_mismatched_ratio` are the explicit ones worth copying.

For us (we already ship the aiter `AITER_TUNE_GEMM→gradlib→CSV` recipe, +2.23% e2e), the high-value
borrowings are: (1) a **version-validator header** on every tuned artifact (TunableOp's
`Validator,ROCM_VERSION,...` lines) so a stale CSV self-rejects instead of silently mis-dispatching;
(2) the **persist-and-restore-with-fallback-heuristic** pattern (Triton-dejavu) for the perf_knowledge
workflow; (3) treating the tuned DB as a **first-class index keyed by the same tuple as the live
dispatch** (vLLM's `E=…,N=…,device_name=…` filename = lookup key); and (4) a `tuned_config:` block in
`sota_registry.yaml` so the workflow can answer "is there a tuned artifact for this op/gen/dtype, what
gates it, where does it live?" without grepping site-packages.

See [`../optimization/autotuning_methodology.md`](../optimization/autotuning_methodology.md) and
[`../kernel_workflow/gemm_tuning_workflow.md`](../kernel_workflow/gemm_tuning_workflow.md) for our validated recipe.

---

## Triton `@triton.autotune` + the autotune cache
Author-time autotuner baked into `@triton.jit`. `configs=[Config(...)]` enumerates tile/warp/stage
knobs; `key=[...]` lists the *argument names* whose value-change re-triggers a full benchmark sweep
(otherwise the in-memory best config is reused). `prune_configs_by` carries `early_config_prune`
(drop configs before benching, e.g. register-spilling tiles) and a `perf_model`+`top_k` (a
*cost-model hook* — only bench the model's top-k). `cache_results=True` (or
`TRITON_PRINT_AUTOTUNING=1` to observe) writes timings to a **single on-disk file whose cache key
includes the full config list** — add/remove one config and the whole sweep re-runs (known wart,
issue #9822). Search = exhaustive over the (pruned) config list. **AMD:** fully supported (Triton has
a first-class AMD/ROCm backend; this is the author-time tuner behind most perf_knowledge triton kernels).
**License:** MIT. **Online** (first call per new key pays the search) unless results are cached/pinned.
- https://triton-lang.org/main/python-api/generated/triton.autotune.html
- https://github.com/triton-lang/triton/blob/main/python/triton/runtime/autotuner.py

## Triton-dejavu (IBM) — persistent autotune cache + fallback heuristics
A drop-in replacement for Triton's autotuner that **saves and restores the autotuner cache across
process lifetimes**, reducing autotune overhead "to zero" for known deployments (the gap the stock
on-disk cache leaves). Applicability of a stored cache is decided by an unambiguous tuple:
*(dejavu version, Triton version, CUDA/ROCm version, GPU type [+ user `tag`])* — a clean **validator
pattern**. Killer features for serving: `fallback_heuristic` + `TRITON_DEJAVU_FORCE_FALLBACK=1`
compute a config when a key is missing (no random autotune stall in a latency-critical path); and
app-provided configs skip the autotuner entirely. Caveat: it can't safely capture `prune_configs_by`
output, so restore-with-pruning is the user's responsibility. Reported >100% speedups vs un-cached
Triton; used in IBM's vLLM platform-portability work. **AMD:** yes (Triton-level, works on ROCm).
**License:** Apache-2.0. **Offline-restore + online-fallback.**
- https://github.com/IBM/triton-dejavu
- https://github.com/triton-lang/triton/issues/4020

## PyTorch TunableOp (hipBLASLt / rocBLAS) — the canonical CSV tuned-DB pattern
PyTorch's `torch.cuda.tunable` races up to thousands of rocBLAS + hipBLASLt GEMM algos (GEMM,
batched, GEMM+bias, scaled) and persists the winner to `tunableop_results<N>.csv` (one per GPU).
Each line = `operator_name, operator_params, solution_name, avg_us`; setting `solution_name=Default`
forces fallback. Crucially the CSV is **prefixed with `Validator` lines**
(`PT_VERSION`, `ROCM_VERSION`, `HIPBLASLT_VERSION`, `ROCBLAS_VERSION`) — on any version mismatch
TunableOp **rejects the file** rather than mis-dispatching. **Offline 3-stage flow** (v2.6+):
collect (`PYTORCH_TUNABLEOP_RECORD_UNTUNED=1`) → tune (`PYTORCH_TUNABLEOP_TUNING=1`) → deploy
(`PYTORCH_TUNABLEOP_ENABLED=1, TUNING=0`). **Incremental:** new shapes are tuned as seen, existing
ones never re-tuned across runs. Search = exhaustive race; no formal numeric gate (relies on library
correctness). **Critical for us:** TunableOp hooks `torch.addmm/matmul/F.linear` — it **does NOT reach
the aiter-dispatched sglang/vLLM GEMM path** (the −0.11%/−0.30% "wrong lever" lesson in our
methodology doc). **AMD:** native (this *is* the ROCm path). **License:** BSD-3 (PyTorch). **Offline/online.**
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/tunable/README.md
- https://rocm.blogs.amd.com/artificial-intelligence/pytorch-tunableop-offline/README.html

## AMD hipBLASLt offline tuning + TensileLite (select vs generate)
Two distinct levers. **(a) Offline tuning / `hipblaslt-bench`:** `HIPBLASLT_TUNING_FILE=<f>` races
solutions and writes `tuning.txt` (best solution *index* per shape); deploy with
`HIPBLASLT_TUNING_OVERRIDE_FILE=tuning.txt` — **no recompile**, overrides even the C/C++
`AlgoGetHeuristic` result. Or `find_exact.py` emits an **equality/grid logic YAML** to merge into
library source (requires rebuild, but survives ROCm upgrades because *you* own it). **(b) TensileLite:**
the assembly *generator* — `tensile_config_generator.py` (from `HIPBLASLT_LOG_MASK=32` shape dumps)
→ `Tensile config.yaml` → `merge.py` into arch logic (gfx942/gfx950). Selection picks from a finite
pool; TensileLite *generates new kernels* from a near-infinite param space. **Hard caveat:** solution
indices are **not portable across ROCm releases or GPU arch** → re-tune on every bump.
rocBLAS analog: `ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH=<csv>`. **AMD:** native. **License:** MIT.
**Offline.** (Same bypass caveat as TunableOp for an aiter-dispatched server.)
- https://rocm.blogs.amd.com/artificial-intelligence/hipblaslt-tensilelite-tuning/README.html
- https://rocm.blogs.amd.com/software-tools-optimization/hipblaslt-offline-tuning-part1/README.html
- https://rocm.docs.amd.com/projects/hipBLASLt/en/develop/how-to/how-to-use-hipblaslt-offline-tuning.html

## AMD Composable Kernel — ckProfiler
The CK benchmarking driver. `ckProfiler gemm_universal <dtype> <layout> <verify> <init> ... M N K
strideA strideB strideC <splitK> <warmup> <iters> <rotating-buffer-MB>` **iterates all compiled
kernel instances for a shape and reports the fastest** — the "best instance config." Notably exposes
a **correctness knob (`verify=1`)** and an **input-distribution knob (`init`)** because CK measured
>20% perf swing between best/worst input values and ~10% between int vs float init — a real
benchmarking-hygiene lesson. Build-time `GPU_TARGETS=gfx942;gfx950` and `DTYPES=...` prune which
instances exist (and thus the search space); `DISABLE_DL_KERNELS=ON` for MI (use XDL/MFMA instances).
No persisted DB of its own — it's the *measurement* layer; you bake winners into source/config.
Search = exhaustive over compiled instances. **AMD:** native (CDNA). **License:** MIT.
**Offline.** (Repo moved to `ROCm/rocm-libraries`.)
- https://github.com/ROCm/composable_kernel/blob/develop/profiler/README.md
- https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html

## aiter tuned-config CSV DB (gradlib) — the live-dispatch lever (ours)
The only tuner that reaches the live sglang/vLLM GEMM path. Capture real shapes with
`AITER_TUNE_GEMM=1` (appends to `aiter/configs/bf16_untuned_gemm.csv` with the **true bias/dtype**),
race hipBLASLt/rocBLAS/asm/triton/flydsl/skinny per shape via `gradlib/gemm_tuner.py`, gate each
candidate at **`err_ratio < 0.05`** (explicit numeric gate — the model to copy), deploy by
`AITER_CONFIG_GEMM_BF16=<tuned.csv>` (`:`-merge list; also `model_configs/` + `configs/` auto-merge),
verify with `AITER_LOG_TUNED_CONFIG=1` (`is tuned on cu_num`). **Lookup key = 9-tuple**
`(cu_num, padded_M, N, K, bias, dtype, outdtype, scaleAB, bpreshuffle)`; the 18-col CSV mirrors it +
`libtype, solidx, splitK, us, kernelName, err_ratio, tflops, bw`. `solidx`/`kernelName` are
**build-specific → never hand-ship; regenerate on the target stack.** MoE analog: `tuned_fmoe.csv`.
Search = exhaustive race; **partial tunes never regress** (uncovered shapes fall back). **AMD:**
native. **License:** MIT. **Offline-capture + offline-tune + static-deploy.** Validated **+2.23% e2e**
on Qwen3.5-27B (and our earlier +2.23% baseline note).
- https://github.com/ROCm/aiter  (`aiter/tuned_gemm.py`, `gradlib/gradlib/gemm_tuner.py`, `aiter/configs/`)
- [`../kernel_workflow/gemm_tuning_workflow.md`](../kernel_workflow/gemm_tuning_workflow.md)

## NVIDIA cuBLASLt — heuristic + autotune
`cublasLtMatmulAlgoGetHeuristic` returns a **ranked list** of algo variants; index 0 is the
predicted-best (the fast path). True autotune = enumerate the top-N (CUDA-L2 used up to 100), time
each, **cache the empirically-fastest per (problem, device)**. `cublasSetSmCountTarget()` lets you
reserve SMs (concurrency tuning). The 2025 CUDA-L2 paper quantifies the gap: exhaustive autotune
beats single-shot heuristic consistently (they report +22% over cuBLAS via RL-generated kernels).
No standard persisted DB (you build your own cache). **AMD:** no (NVIDIA-only; hipBLASLt is the
mirror). **License:** proprietary (CUDA EULA). **Online heuristic / offline autotune.**
- https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuBLASLt/LtSgemmSimpleAutoTuning
- https://docs.nvidia.com/cuda/cublas/
- https://arxiv.org/abs/2512.02551 (CUDA-L2)

## NVIDIA cuDNN v8+ frontend — heuristic modes + engine-config autotune
Graph API: a finalized op-graph yields candidate **engines**, each with **knobs**. Heuristics return
a ranked list of **engine configs**: **Mode A** (fast, broad), **Mode B** (more accurate, higher CPU
latency, falls back to A), **Fallback** (functional only). Recommended flow: query A/B, take the first
*supported* config; or **autotune** — `cudnnFindPlan()`/`build_plan_at_index` builds plans in parallel
and times each, picking the best per (graph, device). `setSMCount` targets hardware. Filterable by
numerical/behavior notes (e.g. exclude down-conversion engines — a *correctness/precision* filter).
**AMD:** no (MIOpen is the rough ROCm analog). **License:** proprietary. **Online heuristic / offline autotune.**
- https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/developer/overview.html
- https://deepwiki.com/NVIDIA/cudnn-frontend/2.5-heuristics-and-engine-configuration

## TVM Ansor / MetaSchedule — learned cost models + JSON tuning-record DB
The reference learned-cost-model autotuner. **AutoTVM** (gen1): template + XGBoost cost model +
simulated annealing. **Ansor** (gen2): template-free; generates *sketches* from the compute DAG,
evolutionary mutation of candidates, **XGBoost GBDT cost model on a 164-feature AST vector**, retrained
each round on real measurements (cost model reduces costly on-device timings). **MetaSchedule** (gen3):
unifies both on TIR, 30 transform primitives, pluggable `CostModel` (XGBModel default, MLPModel,
RandomModel). **Persisted DB** = two newline-JSON files: `database_workload.json` (one line per
unique workload = structural hash + IRModule) and `database_tuning_record.json` (workload idx + schedule
trace + measured times). **Workload reuse via structural hash** — same shape/dtype/compute ⇒ same hash ⇒
reuse records. `UnionDatabase`/`OrderedUnionDatabase` merge multiple DBs (globally-best vs first-match) —
a clean **multi-source registry** pattern. Search = evolutionary + learned cost model.
**AMD:** yes (TVM has a ROCm/HIP backend). **License:** Apache-2.0. **Offline.**
- https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html
- https://arxiv.org/pdf/2006.06762 (Ansor)

## Kernel Tuner (KAT/KT) — search-strategy buffet + output verification
General GPU auto-tuner (Python script drives a kernel; backends CUDA/HIP/OpenCL). Distinctive for
its **breadth of search strategies**: brute force (default), Random, Simulated/Dual Annealing,
Differential Evolution, **Genetic Algorithm**, **PSO**, Firefly, Basin-hopping, Nelder-Mead/Powell/
BFGS/COBYLA/SLSQP, multi-start/iterative local search, and **Bayesian Optimization** (their 2021 paper
adds constraint-aware BO with a contextual-variance acquisition that beat GA by ~20–65%). Reports up to
31.7× faster search than brute force. Builds in **output verification of every benchmarked config** (a
first-class correctness gate — the design we should mirror in our gate). No standard shipped DB; you
persist results yourself. **AMD:** yes (HIP backend). **License:** Apache-2.0. **Offline.**
- https://kerneltuner.github.io/kernel_tuner/stable/optimization.html
- https://arxiv.org/abs/2111.14991 (BO for GPU kernels)

## OpenTuner — ensembles of search techniques (AUC-bandit)
The general program-autotuning framework. Its signature idea: run an **ensemble of disparate search
techniques simultaneously** under an **AUC-Bandit meta-technique** (multi-armed bandit, sliding window,
area-under-curve credit assignment) that **dynamically reallocates trials to whichever technique is
winning**. Ensemble = GA variants + Nelder-Mead + PSO + pattern search + random; a bandit also picks
*which parameter mutation* to apply. Robust across wildly different search-space shapes; parallel-friendly
(fills idle measurement slots from other techniques). Demonstrated up to 2.8× on 16 benchmarks.
Domain-agnostic — you write the config representation + objective. **AMD:** agnostic (drives any program).
**License:** MIT. **Offline.**
- https://opentuner.org/
- https://commit.csail.mit.edu/papers/2014/ansel-pact14-opentuner.pdf

## TileLang `@tilelang.autotune` + Carver
Tile-DSL author-time tuner (strong AMD/CDNA support). `@tilelang.autotune(configs=..., warmup, rep,
ref_prog=, rtol=0.01, atol=0.01, max_mismatched_ratio=0.01, skip_check=False)` — note the
**explicit numeric gate built into the decorator** (rtol/atol + mismatch ratio vs a reference program),
the cleanest "tune *and* verify in one call" we found. `**Carver**` is a separate lightweight
generate-and-rank framework for tile configs (a cost-model-lite). Kernel caching via `tilelang.cache`.
On MI300X FA: 108 candidate configs via `itertools.product`, ~2.7× over PyTorch / 1.53× over Triton.
Search = exhaustive over generated configs (Carver ranks to prune). **AMD:** yes (CDNA3/4, RDNA, MFMA,
MXFP4). **License:** MIT. **Offline/JIT.**
- https://tilelang.com/tutorials/auto_tuning.html
- https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-tilelang-kernel/README.html

## vLLM `benchmark_moe.py` — shipped per-shape MoE JSON DB
The serving-stack "ship a tuned DB" exemplar. `benchmark_moe.py --model … --tp-size … --dtype … --tune
--save-dir` races Triton fused-MoE configs (BLOCK_M/N/K, warps, stages) and writes a JSON whose
**filename IS the lookup key**: `E={experts},N={shardedN},device_name={GPU}[,dtype=…][,block_shape].json`.
Loaded at startup; `VLLM_TUNED_CONFIG_FOLDER` overrides the built-in `fused_moe/configs/` DB (which the
community grows via PRs per GPU/shape). Miss ⇒ "Using default MoE config. Performance might be
sub-optimal!" (a *coverage-gap warning* worth copying). `VLLM_MOE_TUNE_CACHE_CLEAR_INTERVAL` avoids OOM
during tuning. Search = exhaustive grid (community patches shrink ~1900→8–64 for single-GPU). No formal
numeric gate. **AMD:** yes (same JSON DB path on ROCm). **License:** Apache-2.0. **Offline-tune + shipped DB.**
- https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py
- https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py

## NVIDIA TensorRT-LLM — timing cache + optimization profiles + Autotuner
Two layers. **(a) Classic TRT builder:** times candidate **tactics** per layer, picks the fastest, and
serializes a **timing cache** alongside the engine to skip re-timing on rebuilds; **per-shape** handled
by **optimization profiles** (min/opt/max shapes — multiple profiles ⇒ better per-shape kernels at the
cost of build time and slight nondeterminism). **(b) PyTorch-backend `Autotuner`** (`enable_autotuner`):
a newer Python kernel-tuning framework applied to fused MoE / NVFP4 linear. **AMD:** no (NVIDIA-only).
**License:** Apache-2.0 (code) over proprietary TRT. **Offline (build-time) + cache.**
- https://nvidia.github.io/TensorRT-LLM/latest/commands/trtllm-build.html
- https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/useful-build-time-flags.html

## ML cost-models for kernels (2025–2026)
Beyond TVM's XGBoost: the frontier is **tile-level ML predictors that replace on-device timing**.
**NeuSight** (ASPLOS 2025) decomposes a kernel into tiles, predicts per-tile utilization with an ML
model, bounds it by GPU architecture, and aggregates — 121%/30%→**2.3%** latency error on unseen H100
GPT-3. **SynPerf** (2026) is hybrid analytical+ML: an analytical model quantifies per-pipeline (Tensor/
FMA/LSU) demand, fed to an ML model for cross-pipeline contention. **CUDA-L2** (2025) uses RL to *generate*
matmul kernels beating cuBLAS by +22%. **SwizzlePerf** (2025) is hardware-aware LLM kernel opt. These are
research-grade (mostly NVIDIA-evaluated) but the **tile-decomposition + cost-model-instead-of-measurement**
idea is the direction for cutting our race time. **AMD:** mostly NVIDIA-evaluated; methodology portable.
**License:** mixed (papers/repos). **Offline (prediction).**
- https://arxiv.org/html/2407.13853v3 (NeuSight)
- https://arxiv.org/html/2601.14910v1 (SynPerf)
- https://arxiv.org/abs/2512.02551 (CUDA-L2) · https://arxiv.org/pdf/2508.20258 (SwizzlePerf)

---

## Comparison table

| framework | search method | persisted artifact | correctness gate | AMD | license |
|---|---|---|---|---|---|
| Triton `@autotune` | exhaustive over configs (+`early_config_prune`, `perf_model` top-k) | on-disk timing cache (key = config list) | none (relies on kernel) | yes | MIT |
| Triton-dejavu | exhaustive + **persist/restore + fallback heuristic** | versioned cache dir (dejavu/triton/cuda/GPU/tag) | none | yes | Apache-2.0 |
| PyTorch TunableOp | exhaustive race (rocBLAS+hipBLASLt) | `tunableop_results<N>.csv` + **Validator version header** | version validators (rejects stale) | native | BSD-3 |
| hipBLASLt offline / TensileLite | exhaustive (offline) / **generate** (Tensile) | `tuning.txt` index file / logic **YAML** | `NumElementsToValidate` (opt) | native | MIT |
| CK ckProfiler | exhaustive over compiled instances | none (bake into source) | `verify=1` flag | native | MIT |
| **aiter gradlib (ours)** | exhaustive race (hipBLASLt/asm/triton/flydsl/skinny) | **per-shape CSV**, 9-tuple key, `:`-merge | **`err_ratio < 0.05`** | native | MIT |
| cuBLASLt | heuristic rank → autotune top-N | self-built cache | none (lib) | no | proprietary |
| cuDNN v8+ frontend | heuristic Mode A/B → autotune plans | self-built; engine-config filters | numerical-note filters | no | proprietary |
| TVM Ansor/MetaSchedule | evolutionary + **XGBoost learned cost model** | **JSON workload+tuning-record DB** (struct-hash) | measured-time + Union/Ordered merge | yes | Apache-2.0 |
| Kernel Tuner (KAT) | BO/GA/PSO/annealing/local-search/brute | user-persisted | **output verification per config** | yes | Apache-2.0 |
| OpenTuner | **AUC-bandit ensemble** of techniques | user-defined | user-defined objective | agnostic | MIT |
| TileLang `@autotune`/Carver | exhaustive + Carver rank | `tilelang.cache` | **rtol/atol + max_mismatched_ratio** | yes | MIT |
| vLLM `benchmark_moe` | exhaustive grid | **per-shape JSON**, filename = key | none | yes | Apache-2.0 |
| TensorRT-LLM | tactic timing + opt-profiles + Autotuner | **timing cache** in engine | builder validation | no | Apache-2.0/proprietary |
| ML cost models (2025–26) | learned/tile-level prediction; RL gen | model weights | depends | mostly NV | mixed |

---

## What we borrow

### (a) For `tuning/` and `kernel_workflow/gemm_tuning_workflow.md`
- **Version-validator header on every tuned artifact** (TunableOp `Validator,ROCM_VERSION,…` +
  Triton-dejavu's identity tuple). Our CSV already encodes `gfx`/`cu_num`; add an explicit
  `# rocm=…,hipblaslt=…,aiter=…` header line so a stale CSV **self-rejects** instead of silently
  mis-dispatching. This formalizes our existing "re-tune on any ROCm/aiter bump" gotcha.
- **Persist-and-restore with a fallback heuristic** (Triton-dejavu) for any triton kernel we tune:
  store the autotune cache keyed on our identity tuple and provide a `fallback_heuristic` so a missing
  key never stalls the live path — the serving-safe version of our "uncovered shapes fall back" rule.
- **Make the correctness gate explicit and uniform.** We have `err_ratio < 0.05` (gradlib); adopt
  TileLang's `(rtol, atol, max_mismatched_ratio)` triple and Kernel Tuner's "verify every benchmarked
  config" as the documented standard for *non-GEMM* ops we tune (norm/activation/attention), where
  gradlib doesn't apply. Record the gate values in the artifact.
- **Benchmarking hygiene from ckProfiler:** vary input *distribution* (init int vs float) and use a
  rotating buffer / `CACHE_INVALIDATE_BUFFERS` (we already do for OOM) — CK's >20% input-dependent swing
  is a measurement trap to call out in the workflow.
- **Search-space pruning playbook:** document the vLLM single-GPU lesson (1900→8–64 configs) and
  Triton's `early_config_prune`/`perf_model` top-k as the sanctioned way to bound race time before
  reaching for a learned cost model.
- **Coverage-gap warning UX** (vLLM "Using default … sub-optimal"): our `AITER_LOG_TUNED_CONFIG=1`
  engagement grep is the equivalent — keep it mandatory in the gate step, plus log *misses*.

### (b) Turning `sota_registry.yaml` into a tuned-config-aware index
- **Add a `tuned_config:` block per entry** mirroring the live dispatch key, e.g.:
  ```yaml
  tuned_config:
    tool: aiter_gradlib            # aiter_gradlib | tunableop | vllm_moe_json | tilelang | hipblaslt_offline
    artifact: configs/<model>_bf16_tuned_gemm.csv   # path or glob
    key: [cu_num, M, N, K, bias, dtype, outdtype, scaleAB, bpreshuffle]   # = the live lookup key
    gate: {metric: err_ratio, max: 0.05}
    deploy_env: AITER_CONFIG_GEMM_BF16
    validators: {rocm: "7.2", aiter: a6bb499, gfx: gfx942}
    coverage: partial              # full | partial | none
  ```
  This lets the workflow answer, without grepping site-packages: *is there a tuned artifact for this
  op/gen/dtype, what tool made it, what gates it, how is it deployed, and is it still valid for this
  stack?*
- **Borrow vLLM's "filename = lookup key" discipline:** name every shipped artifact so its identity
  (op, gen, dtype, TP/sharded-N, GPU) is legible without opening it — the registry then just points at it.
- **Borrow MetaSchedule's structural-hash + Union/Ordered DB merge:** the registry's `coverage` field
  plus aiter's `:`-merge list = an `OrderedUnionDatabase` (first match wins: mine > model_configs > default).
  The generator (`index/_gen_registry.py`) can compute a structural key (op+gens+dtypes) so two ops that
  share a tuned artifact reuse it.
- **Keep `solidx`/`kernelName`-type fields out of the registry** — they're build-specific (the aiter
  lesson); the registry points at *how to regenerate*, never at hand-copied indices.

## Sources
- Triton autotune & cache: https://triton-lang.org/main/python-api/generated/triton.autotune.html · https://github.com/triton-lang/triton/blob/main/python/triton/runtime/autotuner.py · https://github.com/triton-lang/triton/issues/9822
- Triton-dejavu: https://github.com/IBM/triton-dejavu · https://github.com/triton-lang/triton/issues/4020
- PyTorch TunableOp: https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/tunable/README.md · https://rocm.blogs.amd.com/artificial-intelligence/pytorch-tunableop-offline/README.html
- hipBLASLt / TensileLite: https://rocm.blogs.amd.com/artificial-intelligence/hipblaslt-tensilelite-tuning/README.html · https://rocm.blogs.amd.com/software-tools-optimization/hipblaslt-offline-tuning-part1/README.html · https://rocm.docs.amd.com/projects/hipBLASLt/en/develop/how-to/how-to-use-hipblaslt-offline-tuning.html
- Composable Kernel ckProfiler: https://github.com/ROCm/composable_kernel/blob/develop/profiler/README.md · https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html
- aiter / gradlib: https://github.com/ROCm/aiter · [`../kernel_workflow/gemm_tuning_workflow.md`](../kernel_workflow/gemm_tuning_workflow.md) · [`../optimization/autotuning_methodology.md`](../optimization/autotuning_methodology.md)
- cuBLASLt / CUDA-L2: https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuBLASLt/LtSgemmSimpleAutoTuning · https://docs.nvidia.com/cuda/cublas/ · https://arxiv.org/abs/2512.02551
- cuDNN frontend: https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/developer/overview.html · https://deepwiki.com/NVIDIA/cudnn-frontend/2.5-heuristics-and-engine-configuration
- TVM Ansor / MetaSchedule: https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html · https://arxiv.org/pdf/2006.06762
- Kernel Tuner: https://kerneltuner.github.io/kernel_tuner/stable/optimization.html · https://arxiv.org/abs/2111.14991
- OpenTuner: https://opentuner.org/ · https://commit.csail.mit.edu/papers/2014/ansel-pact14-opentuner.pdf
- TileLang: https://tilelang.com/tutorials/auto_tuning.html · https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-tilelang-kernel/README.html
- vLLM MoE tuned DB: https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py · https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py
- TensorRT-LLM: https://nvidia.github.io/TensorRT-LLM/latest/commands/trtllm-build.html · https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/useful-build-time-flags.html
- ML cost models (2025–26): https://arxiv.org/html/2407.13853v3 (NeuSight) · https://arxiv.org/html/2601.14910v1 (SynPerf) · https://arxiv.org/pdf/2508.20258 (SwizzlePerf)
- Licenses: https://github.com/apache/tvm/blob/main/LICENSE · https://github.com/KernelTuner/kernel_tuner · https://github.com/ROCm/composable_kernel/blob/develop/LICENSE · https://github.com/ROCm/hipBLASLt/blob/develop/LICENSE.md
