---
title: SGLang on ROCm — where the kernels live (sgl-kernel, AITER, TileLang, MoRI)
kind: backend
backend: sglang_kernels
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, mxfp4]
regimes: [prefill, decode, both]
status: sota
updated: 2026-06-08
sources:
  - https://github.com/sgl-project/sglang/blob/main/docker/rocm.Dockerfile
  - https://deepwiki.com/sgl-project/sglang
  - https://github.com/sgl-project/sglang/blob/main/docs/platforms/amd_gpu.md
---

# Where SGLang's kernels live (ROCm source map)

## TL;DR
When you need to **edit** or **trace** a SGLang kernel on MI300X, four trees own kernels: **sgl-kernel**
(SGLang's own C++/HIP custom ops), **AITER** (external `ROCm/aiter`, gated by `SGLANG_USE_AITER`),
**TileLang** (external `tile-ai/tilelang`, default AMD attention), and **MoRI** (external `ROCm/mori`, EP
all-to-all + KV). They're layered, not nested — AITER/TileLang/MoRI are pinned and built in
`docker/rocm.Dockerfile`, separate from sgl-kernel. This card is the "which repo, which file" index.

## The four kernel trees
| tree | what it owns | location | how engaged |
|---|---|---|---|
| **sgl-kernel** | core C++/HIP custom PyTorch ops (gemm, moe, attention, quant), wraps MSCCL++ etc. | `sgl-kernel/` in SGLang repo | built via `setup_rocm.py`; called as `torch.ops.sgl_kernel.<op>` |
| **AITER** | AMD tuned attention/MoE/GEMM/RMSNorm/custom-AR/FlashMLA (CK+ASM+Triton) | external `ROCm/aiter`, pinned `AITER_REPO`/`AITER_COMMIT` in Dockerfile | `SGLANG_USE_AITER=1` + flags |
| **TileLang** | tile-DSL attention/MoE kernels; **default AMD attention backend** | external `tile-ai/tilelang`, pinned `TILELANG_REPO`/`TILELANG_COMMIT` | `--attention-backend tilelang` |
| **MoRI** | EP dispatch/combine + KV transfer | external `ROCm/mori` | all2all backend / DeepEP toggle |

## sgl-kernel layout (the editable HIP ops)
- `sgl-kernel/setup_rocm.py` — ROCm build entry point (uses `setuptools.build_meta`, not scikit-build, for
  hipcc integration).
- `sgl-kernel/pyproject_rocm.toml` — ROCm packaging.
- `sgl-kernel/csrc/common_extension_rocm.cc` — **ROCm operator registration** (the op table).
- `sgl-kernel/csrc/{gemm,moe,attention}/` — kernel implementations.
- `sgl-kernel/include/sgl_kernel_ops.h` — C++ op declarations.
- `sgl-kernel/python/sgl_kernel/{gemm.py,moe.py,elementwise.py,__init__.py}` — Python wrappers; the loader
  detects the GPU and loads the matching lib; ops registered under the `sgl_kernel` namespace via
  `TORCH_LIBRARY_FRAGMENT`.
- `sgl-kernel/CMakeLists.txt` — driven by `CMAKE_ARGS="-DUSE_CUDA=OFF -DUSE_ROCM=ON -DROCM_PATH=/opt/rocm
  -DLLVM_CONFIG=${LLVM_CONFIG}"` in the Docker build.

> CUDA kernels are **not** directly portable to ROCm — they need arch-specific tuning. sgl-kernel keeps a
> distinct ROCm build path for this reason.

## Python-side dispatch (the runtime model layer)
- Attention registry: `python/sglang/srt/layers/attention/attention_registry.py` + per-backend files
  ([attention_backends.md](attention_backends.md)).
- ROCm linear helpers: `python/sglang/srt/layers/rocm_linear_utils.py` (block-scaled FP8, per-token quant).
- MoE: `python/sglang/srt/layers/moe/moe_runner/aiter.py`, `layers/moe/rocm_moe_utils.py` (AITER CK
  `ck_moe_stage1/2`), `layers/moe/fused_moe_triton/` (Triton fused-MoE, config json via
  `SGLANG_MOE_CONFIG_DIR`), `layers/quantization/rocm_mxfp4_utils.py` (MXFP4, needs CDNA3/4 + AITER).

## Build / Docker
- Prebuilt: `lmsysorg/sglang:*-rocm*` / `rocm/sglang-staging:latest` (matched AITER/TileLang/MoRI/hipBLASLt).
- From source: `python sgl-kernel/setup_rocm.py install` → `pip install -e "python[all_hip]"`.
- `docker/rocm.Dockerfile` pins and builds AITER, TileLang, MoRI, Mooncake independently; targets gfx942 +
  gfx950. AMD `srt_hip` extra pulls `petit_kernel`, `wave-lang`.

## Pitfalls
- "Config says `SGLANG_USE_AITER=1` but the aiter package isn't in the image" → ImportError; use a matched
  image, don't pip-install AITER ad-hoc.
- Editing sgl-kernel requires rebuilding (`setup_rocm.py install`) — it's compiled HIP, not Python.
- TileLang/AITER/MoRI live in **their own repos**; a SGLang version bump doesn't move them (Dockerfile note).
- Roadmap kernel languages on AMD: Triton, TileLang, FlyDSL, HipKittens (issue #23494) — expect new trees.

## Verify
- `rocprofv3 --kernel-trace` Top-N → kernel-name → tree: `sgl_kernel::*` = sgl-kernel HIP; `*ck_*`/`fmha_*` =
  AITER/CK; TileLang kernels carry tile names; Triton kernels carry the Python kernel name.
- Grep `common_extension_rocm.cc` for the authoritative ROCm op list when locating an editable kernel.

## Alternatives / cross-links
[overview.md](overview.md) · [attention_backends.md](attention_backends.md) ·
[../vllm_kernels/where_... ] analog: [../vllm_kernels/overview.md](../vllm_kernels/overview.md) ·
[../mori_rccl/mori_ep.md](../mori_rccl/mori_ep.md).

## Sources
- SGLang ROCm Dockerfile (sgl-kernel build, AITER/TileLang/MoRI pins, gfx942/gfx950): https://github.com/sgl-project/sglang/blob/main/docker/rocm.Dockerfile
- sgl-kernel build system & file layout: https://deepwiki.com/sgl-project/sglang
- SGLang AMD GPU docs (build steps, TileLang default): https://github.com/sgl-project/sglang/blob/main/docs/platforms/amd_gpu.md
- AMD roadmap kernel langs (#23494): https://github.com/sgl-project/sglang/issues/23494
