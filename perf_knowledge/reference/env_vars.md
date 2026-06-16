---
title: Environment-variable dictionary (AITER / SGLang / vLLM-ROCm / HIP-ROCm / hipBLASLt / Triton / MIOpen / profiling)
kind: reference
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0 (aiter/jit/core.py, aiter/tuned_gemm.py)
  - https://docs.vllm.ai/en/stable/configuration/env_vars/
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html
  - https://docs.sglang.io/platforms/amd_gpu.html
  - https://rocm.docs.amd.com/projects/hipBLASLt/en/develop/reference/env-variables.html
  - https://rocm.docs.amd.com/en/latest/reference/env-variables.html
---

# Environment-variable dictionary

The single place perf_knowledge cards point to for env-var names, values, and effects. Cite this file by path
instead of restating var semantics in every card. Names below are verified against the on-box
`aiter` source (pinned in [`repo_index.md`](repo_index.md)) and the linked ROCm/vLLM/SGLang docs.
Where a name could not be confirmed verbatim it is marked **(verify exact name)**.

> Convention: "default" = behavior with the var unset. Booleans accept `0`/`1` unless noted. Many
> of these are *load-time* — set before importing torch / launching the server, not mid-run.

---

## AITER_*  (the AMD kernel engine)

`VLLM_ROCM_USE_AITER` / `SGLANG_USE_AITER` are the *framework-side* master switches (see their own
sections). The `AITER_*` vars below are read by the `aiter` package itself and govern the dense-GEMM
tuning path and JIT. Verified in `aiter/jit/core.py` and `aiter/tuned_gemm.py`.

| var | values | what it does | source |
|---|---|---|---|
| `AITER_TUNE_GEMM` | `0`/`1` | Enables the offline/online GEMM **tuner** path in `tuned_gemm.py` (sweep kernels for a shape, record best). Used by the tuning workflow, not normal serving. | aiter `tuned_gemm.py` |
| `AITER_ONLINE_TUNE` | `0`/`1` (default `0`) | On a `device_gemm … does not support this GEMM problem` miss, tune the shape **at runtime** instead of erroring. ROCm vLLM guide recommends this as the first remedy. | vllm-optimization.html; aiter |
| `AITER_LOG_TUNED_CONFIG` | `0`/`1` | Logs which tuned config row was selected per GEMM (which CSV / which kernel). Primary engagement-proof for the dense-GEMM path. | aiter `tuned_gemm.py` |
| `AITER_BYPASS_TUNE_CONFIG` | `0`/`1` | Skip the tuned-CSV lookup (force the heuristic/fallback kernel). Useful as an A/B baseline against the tuned path. | aiter `jit/core.py` |
| `AITER_CONFIG_GEMM_BF16` **(verify exact name)** | path/csv | Selects the BF16 tuned-GEMM CSV. On-box the live path **merges** `aiter/configs/bf16_tuned_gemm.csv` with per-model files (`dsv3_…`, `llama70B_…`, `qwen32B_…`) via the `AITER_CONFIG` loader; the env override symbol is `AITER_CONFIG_GEMM_BF*` in source. | aiter `configs/`, `jit/core.py` |
| `AITER_KSPLIT` | int | Split-K factor override for GEMM kernels (skinny / large-K shapes). | aiter `tuned_gemm.py` |
| `AITER_USE_NT` | `0`/`1` | Force NT operand layout for GEMM (vs auto). | aiter |
| `AITER_JIT_DIR` / `AITER_META_DIR` / `AITER_ASM_DIR` | path | Where JIT-built `.so`, metadata, and prebuilt ASM kernels live. Point at a warm, writable cache to avoid first-call build stalls. | aiter `jit/core.py` |
| `AITER_REBUILD` | `0`/`1` | Force a JIT rebuild of operators (ignore cached `.so`). | aiter `jit/core.py` |
| `AITER_AOT_IMPORT` | `0`/`1` | Use ahead-of-time-built ops instead of JIT. | aiter `jit/core.py` |
| `AITER_GPU_ARCHS` | e.g. `gfx942;gfx950` | Target arch list for JIT builds. | aiter `jit/core.py` |
| `AITER_LOG_LEVEL` / `AITER_LOG_MORE` / `AITER_TRITON_LOG_LEVEL` | int / `0`/`1` | aiter-side logging verbosity (separate from `AMD_LOG_LEVEL`). | aiter |
| `AITER_ROPE_TRITON_BACKEND` / `AITER_ROPE_NATIVE_BACKEND` / `AITER_ROPE_FUSED_QKNORM` | `0`/`1` | Select RoPE implementation and fused QK-norm path. | aiter |
| `AITER_QUICK_REDUCE_QUANTIZATION` / `AITER_QUICK_REDUCE_MAX_SIZE_BYTES_MB` | enum / int | Quick-reduce (custom allreduce) quantization mode and size cap. | aiter |
| `AITER_ENABLE_EXPERIMENTAL` | `0`/`1` | Gate experimental kernels. | aiter |

**GEMM tuning trio** = `AITER_TUNE_GEMM` (do the sweep) + `AITER_LOG_TUNED_CONFIG` (prove which row
hit) + the BF16 config selector / `AITER_ONLINE_TUNE` (where the tuned rows come from / on-miss
behavior). → workflow: [`../kernel_workflow/gemm_tuning_workflow.md`](../kernel_workflow/gemm_tuning_workflow.md).
Engagement proof: [`../profiling/engagement_verification.md`](../profiling/engagement_verification.md).
Dense-GEMM SOTA path: [`../operators/dense_gemm/`](../operators/dense_gemm/).

---

## SGLANG_*  (SGLang serving stack on ROCm)

| var / flag | values | what it does | source |
|---|---|---|---|
| `SGLANG_USE_AITER` | `0`/`1` (default `0`) | Master switch: route MoE / attention / GEMM / allreduce through aiter. Errors if `aiter` not installed. Required for MXFP4 on CDNA3/4. | docs.sglang.io/platforms/amd_gpu.html |
| `--attention-backend` | `aiter` \| `ROCM_ATTN` \| `triton` \| `fa3` … (CLI flag) | Pick the attention backend. `ROCM_ATTN` is the renamed prefill-decode attention. AMD usually auto-selects. | sglang attention_backend.md |
| `SGLANG_DISABLE_AITER_GREEDY_SAMPLE` **(verify exact name)** | `0`/`1` | Disable the aiter greedy-sampling kernel (fall back to torch sampling) — A/B knob for the sampling path. Confirm spelling in installed sglang. | sglang (verify) |
| `--mem-fraction-static` / `--context-length` | float / int (CLI) | Lower these when the aiter attention backend over-allocates KV cache → OOM (known issue). | sglang issues #18262 |

Cross-link: [`../kernel_workflow/attention_backend_selection.md`](../kernel_workflow/attention_backend_selection.md),
[`../kernel_workflow/choosing_a_backend.md`](../kernel_workflow/choosing_a_backend.md),
[`../quantization/deployment_recipes.md`](../quantization/deployment_recipes.md).

---

## VLLM_ROCM_*  (vLLM on ROCm; ~13 AITER sub-flags)

`VLLM_ROCM_USE_AITER` is the parent; the sub-flags only take effect when it is `1`. Defaults below
are from the vLLM source / ROCm guide.

| var | values (default) | what it does | source |
|---|---|---|---|
| `VLLM_ROCM_USE_AITER` | `0`/`1` (`0`) | Parent switch for all aiter ops in vLLM V1. | env_vars (vLLM) |
| `VLLM_ROCM_USE_AITER_MHA` | `0`/`1` (`1`) | aiter multi-head attention backend (set `0` to use Triton prefill-decode). | vllm-optimization.html |
| `VLLM_ROCM_USE_AITER_MLA` | `0`/`1` (`1`) | aiter MLA attention. | vLLM source |
| `VLLM_ROCM_USE_AITER_MOE` | `0`/`1` (`1`) | aiter fused-MoE kernels (disable to dodge unsupported-GEMM MoE errors). | vllm issues #22245 |
| `VLLM_ROCM_USE_AITER_LINEAR` | `0`/`1` (`1`) | aiter linear/GEMM op. | vLLM source |
| `VLLM_ROCM_USE_AITER_RMSNORM` | `0`/`1` (`1`) | aiter RMSNorm. | vLLM source |
| `VLLM_ROCM_USE_AITER_FP8BMM` / `…_FP4BMM` | `0`/`1` (`1`) | aiter Triton fp8/fp4 batched-matmul kernels. | vLLM source |
| `VLLM_ROCM_USE_AITER_FP4_ASM_GEMM` | `0`/`1` (`0`) | aiter fp4 ASM GEMM. | vLLM source |
| `VLLM_ROCM_USE_AITER_TRITON_ROPE` | `0`/`1` (`0`) | aiter Triton RoPE. | vLLM source |
| `VLLM_ROCM_USE_AITER_PAGED_ATTN` | `0`/`1` (`0`) | aiter paged-attention. | vLLM source |
| `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT` **(verify exact name)** | enum/`0`/`1` | Shuffle/repack KV-cache layout for the aiter attention kernel's preferred tiling. | vLLM ROCm source (verify) |
| `VLLM_USE_TRITON_FLASH_ATTN` | `0`/`1` (`1`) | Use vLLM's Triton FA. Set `0` for VL models and required-off on RDNA3/Navi. | env_vars (vLLM); vllm issues #4514 |
| `--aiter-config` | (proposed CLI) | Upcoming single flag to replace the 13 env vars (RFC #33163). | vllm issues #33163 |

Layout-shuffle operator: [`../operators/layout_shuffle/overview.md`](../operators/layout_shuffle/overview.md).
Cross-link: [`../kernel_workflow/attention_backend_selection.md`](../kernel_workflow/attention_backend_selection.md),
[`../quantization/deployment_recipes.md`](../quantization/deployment_recipes.md).

---

## HIP / ROCm runtime

| var | values | what it does | source |
|---|---|---|---|
| `HIP_VISIBLE_DEVICES` | comma list | Visible GPUs to HIP (post-init device masking). | ROCm/HIP env docs |
| `ROCR_VISIBLE_DEVICES` | comma list / UUIDs | Masks at the ROCr/HSA layer (applied before `HIP_VISIBLE_DEVICES`). | ROCR-Runtime env docs |
| `CUDA_VISIBLE_DEVICES` | comma list | Honored by the HIP CUDA-compat shim (torch). | ROCm |
| `GPU_MAX_HW_QUEUES` | int (e.g. `2`) | HIP's HSA hardware-queue pool size. ≤4 streams maximizes HW efficiency; `2` pairs with high-prio RCCL for FSDP. | workload.html |
| `HIP_FORCE_DEV_KERNARG` | `0`/`1` | Place kernel args in device memory → lower launch latency. Common inference win. | vllm-optimization.html |
| `AMD_LOG_LEVEL` | `0`–`4` | HIP runtime log verbosity (kernel launches, API). | ROCm/HIP env docs |
| `HSA_*` (e.g. `HSA_ENABLE_DEBUG`, `HSA_DISABLE_FRAGMENT_ALLOCATOR`, `HSA_TOOLS_LIB`) | various | ROCr/HSA debug + allocator knobs; perf-sensitive, debug-only. | ROCR-Runtime env docs |
| `HSA_OVERRIDE_GFX_VERSION` | e.g. `11.0.0` | Spoof arch for unsupported consumer cards. | ROCm |
| `TORCH_BLAS_PREFER_HIPBLASLT` | `0`/`1` (`1` recommended) | Prefer hipBLASLt over rocBLAS for torch GEMM. | vllm-optimization.html |
| `NCCL_MIN_NCHANNELS` | int (e.g. `112`) | RCCL channel count; multi-GPU only. | vllm-optimization.html |
| `SAFETENSORS_FAST_GPU` | `0`/`1` | GPU-accelerated safetensors load. | vllm-optimization.html |

---

## hipBLASLt / rocBLAS / PyTorch TunableOp

> **Validated caveat — TunableOp gets 0 engagement on sglang/vLLM/aiter dense GEMM.** PyTorch
> TunableOp and `HIPBLASLT_TUNING_FILE` hook the **PyTorch GEMM dispatch**. In sglang/vLLM-with-aiter
> the live dense GEMM is dispatched **by aiter and bypasses PyTorch dispatch**, so the tuned solutions
> are never consulted regardless of CSV quality. A `PYTORCH_TUNABLEOP_ENABLED=1` A/B that shows "no
> change" proves nothing about the kernel — only that the path was bypassed. Tune via the AITER trio
> instead. Full writeup: [`../profiling/engagement_verification.md`](../profiling/engagement_verification.md).

| var | values | what it does | source |
|---|---|---|---|
| `PYTORCH_TUNABLEOP_ENABLED` | `0`/`1` | Master on/off for TunableOp (rocBLAS/hipBLASLt GEMM search on the **torch** path). | pytorch tunable README |
| `PYTORCH_TUNABLEOP_TUNING` | `0`/`1` | Actually run tuning (vs. only consume existing results). | pytorch tunable README |
| `PYTORCH_TUNABLEOP_FILENAME` | path | Results CSV (validator header: PT/ROCM/HIPBLASLT/GCN/ROCBLAS versions; mismatch → re-tune). | pytorch tunable README |
| `PYTORCH_TUNABLEOP_VERBOSE` | `0`/`1` | Tuning debug output. | pytorch tunable README |
| `HIPBLASLT_TUNING_FILE` | path | hipBLASLt offline-tuning results file. **0 engagement on aiter dense path** (see caveat). | hipBLASLt env docs |
| `HIPBLASLT_LOG_MASK` / `HIPBLASLT_LOG_FILE` | int / path | Log every hipBLASLt call (large files); use to confirm whether GEMMs even reach hipBLASLt. | hipBLASLt env docs |

Cross-link: [`../kernel_workflow/gemm_tuning_workflow.md`](../kernel_workflow/gemm_tuning_workflow.md),
[`../operators/dense_gemm/`](../operators/dense_gemm/).

---

## Triton (AMD backend)

| var | values | what it does | source |
|---|---|---|---|
| `TRITON_CACHE_DIR` | path | Compiled-kernel cache; warm it to avoid first-call JIT stalls. | triton |
| `TRITON_PRINT_AUTOTUNING` | `0`/`1` | Print autotune config chosen per kernel (engagement proof). | triton |
| `AMDGCN_ENABLE_DUMP` / `TRITON_ALWAYS_COMPILE` **(verify exact name)** | `0`/`1` | Dump AMDGCN/LLVM-IR; force recompile. Confirm in installed triton. | triton (verify) |
| `MLIR_ENABLE_DUMP` | `0`/`1` | Dump MLIR through the AMD lowering pipeline. | triton |
| `FLASH_ATTENTION_TRITON_AMD_AUTOTUNE` | `0`/`1` | Autotune the Triton-AMD flash-attention kernel. | aiter / flash-attn |

Cross-link: [`../languages/`](../languages/), [`../kernel_workflow/authoring_a_kernel_with_geak.md`](../kernel_workflow/authoring_a_kernel_with_geak.md).

## MIOpen

| var | values | what it does | source |
|---|---|---|---|
| `MIOPEN_FIND_MODE` | `NORMAL`/`FAST`/`HYBRID`/`DYNAMIC_HYBRID` | Convolution algorithm search strategy. | MIOpen docs |
| `MIOPEN_USER_DB_PATH` | path | Per-user perf-DB (find results) location. | MIOpen docs |
| `MIOPEN_FIND_ENFORCE` | enum/int | Force (re)search or DB use. | MIOpen docs |
| `MIOPEN_LOG_LEVEL` | int | MIOpen logging. | MIOpen docs |

(Conv ops on MI are secondary for LLM serving; relevant to [`../operators/conv2d/`](../operators/conv2d/),
[`../operators/depthwise_conv/`](../operators/depthwise_conv/).)

## Profiling

| var | values | what it does | source |
|---|---|---|---|
| `ROCPROF_*` (rocprofv3) | various | rocprofv3 control (counters, output dir, ATT). | ROCm profiling docs |
| `ROCPROFILER_*` **(verify exact name)** | various | rocprofiler-sdk runtime config. | ROCm (verify) |
| rocprof-compute (omniperf) CLI | `--roof-only`, `-k`, … | Roofline / per-kernel profiling (renamed omniperf). | rocprof-compute docs |

Cross-link: [`../profiling/rocprof_compute_workflow.md`](../profiling/rocprof_compute_workflow.md),
[`../profiling/rocprofv3_counters.md`](../profiling/rocprofv3_counters.md),
[`../profiling/tooling_overview.md`](../profiling/tooling_overview.md).

---

## Sources
- On-box `aiter` source, pinned in [`repo_index.md`](repo_index.md): `aiter/jit/core.py`,
  `aiter/tuned_gemm.py`, `aiter/configs/*tuned_gemm.csv` (AITER_* names grepped from source).
- vLLM env vars — https://docs.vllm.ai/en/stable/configuration/env_vars/ ;
  vLLM V1 ROCm optimization — https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html ;
  AITER env-var refactor RFC — https://github.com/vllm-project/vllm/issues/33163
- SGLang AMD — https://docs.sglang.io/platforms/amd_gpu.html ;
  attention backends — https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/attention_backend.md
- hipBLASLt env vars — https://rocm.docs.amd.com/projects/hipBLASLt/en/develop/reference/env-variables.html ;
  PyTorch TunableOp — https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/tunable/README.md ;
  TunableOp 0-engagement finding — [`../profiling/engagement_verification.md`](../profiling/engagement_verification.md)
- ROCm/HIP/ROCR env vars — https://rocm.docs.amd.com/en/latest/reference/env-variables.html ;
  https://rocm.docs.amd.com/projects/HIP/en/latest/reference/env_variables.html ;
  MI300X workload optimization — https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- Names marked **(verify exact name)** were not confirmed verbatim in docs/source at write time.
