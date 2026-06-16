---
title: The ROCm kernel ecosystem for LLM inference (beyond aiter/sglang/vllm)
kind: reference
updated: 2026-06-09
note: migrated from perf_knowledge v1 (02_libraries/rocm_ecosystem.md)
---

# The ROCm Kernel Ecosystem for LLM Inference on MI300X (gfx942)

> Scope: the rest of the ROCm stack an e2e inference optimizer must understand beyond sglang/vllm/RCCL —
> math libraries (hipBLASLt/rocBLAS/Tensile), MIOpen, primitive libs (rocPRIM/hipCUB/rocThrust),
> hipFFT/rocFFT, rocSPARSE, the HSA/HIP runtime, profiling (rocprofiler-sdk / rocprofv3 / AMD SMI), and
> **how PyTorch-ROCm maps ops onto these**. AMD-only (CDNA3/gfx942). Verified vs ROCm 7.x (2026).
>
> For LLM **inference**, the libraries that matter (in order): **hipBLASLt/rocBLAS** (GEMM = 70-80% of
> GPU time) → AITER/Triton (attention/MoE, see sglang_rocm.md/vllm_rocm.md) → RCCL (comm) → the rest are
> mostly irrelevant at inference time but show up in training/other workloads. Knowing which library a
> hot kernel came from tells you **which tuning lever** to pull.

---

## 1. Version / image matrix (ROCm 7.x, 2026)

| Component | Version (recent develop) | Role at inference |
|---|---|---|
| ROCm | 7.2.x stable (7.0 = first full MI300X line, Sep 2025; 7.13 preview) | platform |
| HIP runtime | ROCm 7.x | kernel launch / streams / graphs |
| PyTorch (ROCm) | 2.7+ | op dispatch surface |
| hipBLASLt | 1.2.2 | **primary GEMM backend** |
| rocBLAS | 5.2.0 | GEMM fallback / alt |
| Tensile | 4.45.0 | rocBLAS GEMM kernel generator/backend |
| MIOpen | 3.5.1 | conv/RNN/norm (mostly not inference-LLM) |
| AMD SMI | 26.2.2 | device mgmt (**replaces** ROCm SMI) |
| ROCm SMI | 7.8.0 | **deprecated** → maintenance mode |
| rocprofiler-sdk / rocprofv3 | 1.x | **the** profiler (rocprof/rocprofv2 EoS ~Q2 2026) |

Always serve from a matched prebuilt image (`vllm/vllm-openai-rocm`, `lmsysorg/sglang:*-rocm*`,
`rocm/sglang-staging`) — these ship hipBLASLt tuning DBs, AITER, and CK already matched to the ROCm
version. Mixing a pip wheel against a different system ROCm is the #1 source of "slow / wrong kernel".

---

## 2. Math libraries — GEMM backend stack (the head kernel)

```
PyTorch nn.Linear / matmul
        │  TORCH_BLAS_PREFER_HIPBLASLT=1
        ▼
   hipBLASLt ── (no solution for shape) ──► rocBLAS ──► Tensile kernels
        │                                       │
        └─ tuned solution DB                    └─ source-GEMM fallback (rocBLAS w/o Tensile)
```

| Library | What it is | Tuning lever |
|---|---|---|
| **hipBLASLt** | lightweight, extensible GEMM w/ epilogue (bias/act/scale); the default on modern arch | `HIPBLASLT_TUNING_FILE=<file>` (offline `hipblaslt-bench` solution sweep); `HIPBLASLT_LOG_MASK`/`HIPBLASLT_LOG_FILE` to see calls |
| **rocBLAS** | classic BLAS; uses **Tensile** kernels; fallback when hipBLASLt has no solution | `ROCBLAS_USE_HIPBLASLT=0` forces Tensile; `=1` prefer hipBLASLt w/ Tensile fallback; `ROCBLAS_TENSILE_*` |
| **Tensile** | benchmark-driven GEMM kernel generator (the actual asm) behind rocBLAS | regenerate/extend the Tensile logic for a custom shape (heavy) |
| **PyTorch TunableOp** | runtime auto-tuner; benchmarks 1000s of rocBLAS+hipBLASLt candidates per (M,N,K,dtype) and caches winner | `PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=1 PYTORCH_TUNABLEOP_FILENAME=<csv>` → warm pass, then ship `TUNING=0` |

Inference workflow: keep `TORCH_BLAS_PREFER_HIPBLASLT=1`; watch the log for `not found tuned config ...
using default config` (= generic = slow); if you see it, run a **TunableOp warm pass** or pin a
hipBLASLt solution. Caveat: as hipBLASLt matures, TunableOp's edge shrinks and is not guaranteed to beat
the default — measure.

---

## 3. MIOpen (conv / RNN / fused norm) — mostly NOT LLM-inference

MIOpen is AMD's deep-learning primitive library (the cuDNN analog): convolutions (incl. NHWC
channels-last on ROCm 7), pooling, batchnorm/layernorm, RNN, activation, softmax. **For transformer LLM
*inference* it is largely idle** — there are no convolutions; norms/softmax come from AITER/Triton fused
kernels, not MIOpen. It matters for: vision encoders (ViT/CLIP towers in VLMs), Whisper/conv front-ends,
diffusion, and training. PyTorch routes `conv2d`/`batch_norm` here.

| Lever | Env |
|---|---|
| auto-tune conv on first new shape | `MIOPEN_FIND_ENFORCE` / find-mode; results cached in user DB |
| logging | `MIOPEN_ENABLE_LOGGING=1`, `MIOPEN_ENABLE_LOGGING_CMD=1` |
| user DB path | `MIOPEN_USER_DB_PATH` |

> Note: Triton is *not* used if MIOpen/rocBLAS is faster for an op — PyTorch picks the library backend.

---

## 4. Primitive & specialty libraries

| Library | cuda analog | What | LLM-inference relevance |
|---|---|---|---|
| **rocPRIM** | CUB (device) | device-wide scan/reduce/sort/select primitives (HIP) | **indirect** — building block under fused MoE topk/sort, sampling; rarely tuned directly |
| **hipCUB** | CUB | thin CUB-compatible wrapper over rocPRIM | portability shim |
| **rocThrust** | Thrust | high-level parallel algorithms (sort/scan/transform) | sampling, sort-by-key in MoE routing; behind torch ops |
| **hipFFT / rocFFT** | cuFFT | FFTs | **not** standard LLM inference (relevant: some audio/conformer, diffusion) |
| **rocSPARSE** | cuSPARSE | sparse BLAS (SpMM/SpMV) | structured-sparsity / some MoE-sparse experiments; not mainstream dense LLM |
| **rocSOLVER / hipSOLVER** | cuSOLVER | dense linear solvers | not inference |
| **rocRAND / hipRAND** | cuRAND | RNG | sampling jitter; minor |
| **CK / ck_tile** | CUTLASS | Composable Kernel — tiled GEMM/attention/MoE building blocks (asm-grade) | **high** — backs AITER attention/MoE; tunable by instance, not rapidly editable for novel shapes |
| **AITER** | cuBLAS+cuDNN+FA+TE combined | AMD tuned-kernel library (asm/CK/Triton/hipBLAS dispatch) | **highest** — the inference fast path (see sglang/vllm docs) |
| **hipBLAS** | cuBLAS | BLAS API shim over rocBLAS/hipBLASLt | API layer |

Takeaway: at inference you almost never call rocPRIM/rocThrust/rocFFT/rocSPARSE directly — they sit
under PyTorch/AITER. The two specialty libs an optimizer *does* touch are **CK** (tune the attention/MoE
instance) and **AITER** (flag/env selection).

---

## 5. Runtime: HSA / HIP

| Layer | Role | Key knobs |
|---|---|---|
| **HSA** (ROCr runtime) | low-level queue/signal/memory; talks to kfd driver | `HSA_NO_SCRATCH_RECLAIM=1` (**critical on MI300X** — stops scratch reclaim stalls), `HSA_FORCE_FINE_GRAIN_PCIE=1` (P2P over PCIe), `HSA_ENABLE_SDMA` |
| **HIP** | CUDA-like API: kernels, streams, graphs, events | `HIP_FORCE_DEV_KERNARG=1` (device kernarg → lower launch latency), `GPU_MAX_HW_QUEUES=2` (more HW queues → comm/compute overlap), `HIP_VISIBLE_DEVICES` |
| HIP graphs | capture decode kernel stream → amortize launch | sglang `--cuda-graph-*`, vLLM avoids `--enforce-eager` |

MI300X launch-overhead reality: decode is a flood of tiny kernels; per-dispatch cost is real, so
`HIP_FORCE_DEV_KERNARG=1` + HIP-graph capture + `GPU_MAX_HW_QUEUES=2` together are worth several % at
decode. `HSA_NO_SCRATCH_RECLAIM=1` removes large sporadic idle gaps.

---

## 6. Profiling: rocprofiler-sdk / rocprofv3 (+ torch.profiler)

`rocprofv3` (CLI over **ROCprofiler-SDK**) is the supported profiler in 2026. Legacy `rocprof`,
`rocprofv2`, ROCTracer, ROCProfiler are **deprecated, EoS ~Q2 2026** — do not build new tooling on them.

| Task | Command / config |
|---|---|
| kernel trace (csv) | `rocprofv3 --kernel-trace --output-format csv -- <cmd>` |
| HIP + kernel + memcpy trace | `rocprofv3 --sys-trace -- <cmd>` |
| default output | **rocpd** (SQLite3 DB); omit `--output-format` |
| visualize | `--output-format pftrace` → open in `ui.perfetto.dev` |
| torch path | `torch.profiler` with the ROCprofiler-SDK backend (ROCm 7.2.x fixed the big idle-gap artifact in vLLM traces) |
| device mgmt / power / util | **`amd-smi`** (ROCm SMI deprecated) |

Optimizer use: rocprofv3 (or torch.profiler) gives the **Top-N kernels by GPU time** with names → the
name tells the library: `Cijk_*`/Tensile = rocBLAS GEMM; `*hipblaslt*` = hipBLASLt; `*ck_*`/`fmha_*` =
CK/AITER attention; `paged_attention_ll4mi_*` = vLLM custom HIP; Triton kernels carry the python kernel
name. That classification routes the tuning decision (hipBLASLt tune vs CK instance vs Triton rewrite).

---

## 7. How PyTorch-ROCm maps ops → libraries (the dispatch cheat-sheet)

| torch op | ROCm backend | Tuning entry point |
|---|---|---|
| `torch.matmul` / `nn.Linear` (dense) | hipBLASLt → rocBLAS/Tensile | TunableOp, `HIPBLASLT_TUNING_FILE` |
| scaled / FP8 GEMM | hipBLASLt FP8 / AITER `_rocm_aiter_w8a8_gemm` / vLLM `wvSplitKQ` | AITER tuned csv; fnuz dialect care |
| skinny/decode GEMM (M=batch) | vLLM `LLMM1`/`wvSplitK` (custom HIP), AITER skinny | `VLLM_ROCM_USE_SKINNY_GEMM`, Triton split-K |
| scaled-dot-product-attention | AITER FA / Triton FA / CK fmha / vLLM custom paged-attn | `--attention-backend`, AITER env |
| `conv2d` / `batch_norm` | MIOpen | MIOpen find-mode |
| `rms_norm` / `layer_norm` (transformer) | AITER fused / Triton (not MIOpen) | `*_AITER_RMSNORM` |
| `sort` / `topk` / `cumsum` (MoE routing, sampling) | rocThrust / rocPRIM / AITER topk | mostly automatic |
| all-reduce / all-gather | RCCL / AITER custom AR / Quick Reduce | see rccl_comm.md |
| `fft` | rocFFT/hipFFT | rarely in LLM |

Rule of thumb: **dense GEMM → math libraries (hipBLASLt/Tensile, TunableOp); attention/MoE/norm →
AITER/Triton/CK; comm → RCCL.** Everything else is downstream of these.

---

## 8. Optimizer checklist (ecosystem layer)

1. Run from a matched prebuilt ROCm image; set `TORCH_BLAS_PREFER_HIPBLASLT=1`, `HSA_NO_SCRATCH_RECLAIM=1`,
   `HIP_FORCE_DEV_KERNARG=1`, `GPU_MAX_HW_QUEUES=2`.
2. Profile with **rocprofv3** / torch.profiler → Top-N kernels; classify each by name → library.
3. GEMM-dominated? → TunableOp warm pass and/or hipBLASLt solution pin; check for "default config" logs.
4. Attention/MoE/norm hot? → AITER flags / CK instance / Triton rewrite (see sglang/vllm docs).
5. Comm hot? → RCCL/custom-AR (see rccl_comm.md).
6. Watch versions: rocprof/rocprofv2/ROCm-SMI are deprecated — use rocprofv3 / amd-smi.

---

## Sources
- rocprofv3 / ROCprofiler-SDK docs: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html
- ROCm 7.2.x release notes (versions, deprecations, vLLM profiling fix): https://rocm.docs.amd.com/en/latest/about/release-notes.html
- rocBLAS design & usage (hipBLASLt/Tensile backend selection): https://rocm.docs.amd.com/projects/rocBLAS/en/latest/how-to/what-is-rocblas.html
- MI300X workload optimization (HSA/HIP env, TunableOp, MIOpen): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- ROCm component versions / RELEASE.md: https://github.com/ROCm/ROCm/blob/develop/RELEASE.md
- AITER (AI Tensor Engine for ROCm): https://github.com/ROCm/aiter
- PyTorch profiling on ROCm (AMD HPC training examples): https://github.com/amd/HPCTrainingExamples/blob/main/MLExamples/PyTorch_Profiling/README.md
- MIOpen / hipBLASLt logging env (TheRock issue #2591): https://github.com/ROCm/TheRock/issues/2591
