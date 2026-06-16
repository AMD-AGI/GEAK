---
title: Multi-backend kernel libraries & dispatchers — landscape survey
kind: landscape
updated: 2026-06-09
scope: prior art for an operator×backend SOTA registry + a serving-stack dispatcher
projects: [FlagGems, PyTorch_ATen_dispatcher, torch.compile_Inductor, ROCm_aiter, FlashInfer, Liger-Kernel, xFormers, NVIDIA_Transformer_Engine, vLLM, SGLang, BackendBench, HF_Kernel_Hub, Unsloth]
sources:
  - https://github.com/FlagOpen/FlagGems
  - https://pytorch.org/blog/flaggems-joins-the-pytorch-ecosystem-triton-powered-operator-library-for-universal-ai-acceleration/
  - https://github.com/pytorch/pytorch/issues/139602
  - https://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/
  - https://docs.pytorch.org/tutorials/advanced/extend_dispatcher
  - https://dev-discuss.pytorch.org/t/torchinductor-update-8-max-autotune-support-on-cpu-with-gemm-template/2439
  - https://deepwiki.com/ROCm/aiter
  - https://rocm.docs.amd.com/projects/radeon/en/latest/docs/advanced/vllm/gemm-tuning.html
  - https://github.com/flashinfer-ai/flashinfer
  - https://rocm.blogs.amd.com/artificial-intelligence/flashinfer/README.html
  - https://rocm.blogs.amd.com/artificial-intelligence/flashinfer-release2/README.html
  - https://github.com/linkedin/Liger-Kernel
  - https://embeddedllm.com/blog/cuda-to-rocm-portability-case-study-liger-kernel
  - https://github.com/facebookresearch/xformers
  - https://github.com/ROCm/xformers
  - https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/attention/attention.html
  - https://vllm.ai/blog/2026-02-27-rocm-attention-backend
  - https://docs.vllm.ai/en/latest/design/attention_backends/
  - https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/attention_backend.md
  - https://github.com/sgl-project/sglang/issues/20372
  - https://github.com/meta-pytorch/BackendBench
  - https://github.com/huggingface/kernels
  - https://huggingface.co/docs/kernels/index
  - https://unsloth.ai/docs/blog/unleash-the-power-of-amd-official-support-for-unsloth-is-here
---

# Multi-backend kernel libraries & dispatchers

## TL;DR

The field has converged on a small number of **proven design patterns** for "one operator → N
backend implementations → pick the best." Three layers recur:

1. **Registry / dispatch table** — a per-operator table keyed by `(dispatch_key)` that maps to an
   implementation. PyTorch's **ATen dispatcher** is the canonical example (vtable per op, key
   computed from tensor args + thread-local state). FlagGems and HF Kernel Hub layer their own
   tables *on top of* this same mechanism (out-of-tree registration via `PrivateUse1` /
   `TORCH_LIBRARY_IMPL`).
2. **Selection mechanism** — four families, often combined:
   - **env-flag / explicit override** (vLLM `VLLM_ATTENTION_BACKEND`, SGLang `--attention-backend`,
     TE `NVTE_FUSED_ATTN`) — cheapest, no recompile;
   - **priority-ordered capability list per platform** (vLLM/SGLang auto-select: try FA3 → FlashInfer
     → Triton, gated by arch/dtype/head-dim) — this is the dominant serving-stack pattern;
   - **per-shape autotune DB** (CSV/JSON keyed by shape×dtype×arch) — AITER `tuned_gemm`, PyTorch
     TunableOp, Inductor `max-autotune`;
   - **JIT cost/spec compile** (FlashInfer generates a kernel per attention variant).
3. **Correctness gate** — BackendBench shows the registry is only trustworthy if every cell passes
   the framework's own OpInfo tests before it can be claimed "SOTA."

For **perf_knowledge** the highest-leverage borrowings are: (a) model the registry as a **`(op, backend,
arch, dtype, regime)` → impl** table exactly like ATen's vtable + vLLM's per-platform priority list;
(b) make selection a **machine-readable priority list + capability predicates + a per-shape tuned-DB
pointer**, mirroring AITER's CSV-merge design and SGLang's `get_attention_backends()` plugin API; and
(c) require a **BackendBench-style correctness stamp** on every cell. **License-wise** the directly
reusable designs are Apache-2.0 (FlagGems, FlashInfer, TE) / MIT (AITER) / BSD (Liger, BackendBench).

---

## FlagGems (FlagOpen / BAAI)

URL: https://github.com/FlagOpen/FlagGems

A general operator library (180–216+ ops) written entirely in **Triton**, designed to be
**multi-backend single-source**: write the kernel once, JIT it on any backend. It plugs into PyTorch
by registering Triton implementations against ATen ops out-of-tree, intercepting ATen calls and
routing them to backend-specific Triton code, so models switch with no API change. [pytorch.org blog]

- **op → N backends:** one Triton source per op; the *backend* is the device/vendor the Triton
  package targets (NVIDIA CUDA, AMD ROCm, Ascend, AIPU, …). Per the multi-backend RFC, the dispatch
  *key* to register differs per backend, so FlagGems needs device/backend **detection** to pick the
  right key, and wrapper-function APIs must be device-agnostic. [pytorch #139602]
- **selection:** runtime per-function dispatch + a kernel cache (`LibEntry`) that bypasses Triton's
  Autotuner/Heuristics/JitFunction on cache hit, falling back to full tuning on miss. A C++ Triton
  dispatcher is in development to cut Python overhead. Per-vendor tuning configs live under
  `src/flag_gems/runtime/backend/` and are loaded by `configs_loader.py`; op registration is via
  `op_registrar.py`. (Exact per-vendor heuristic YAML schema — *mark: inferred from repo tree, not
  read line-by-line.*)
- **AMD support:** yes — explicitly supports ROCm via the ROCm Triton/torch stack; one of its stated
  goals is supporting alternative GPUs/DSAs without forking. (MI300X-specific tuned configs present
  but coverage-per-op not individually verified — *mark.*)
- **license:** **Apache-2.0** (verified, repo LICENSE).
- **borrow:** the *single-source + per-vendor config dir + cache-that-bypasses-autotune* layout maps
  almost 1:1 onto perf_knowledge's "operator card + per-backend tuned params." `op_registrar.py` +
  `configs_loader.py` is a concrete template for "registry entry → impl + tuned config."

## PyTorch ATen dispatcher

URL: https://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/ ·
https://docs.pytorch.org/tutorials/advanced/extend_dispatcher

The canonical production registry+dispatcher. Every operator owns a **table of function pointers**,
one entry per **dispatch key** (CPU, CUDA, XLA, Autograd, Autocast, …). At call time the dispatcher
computes the highest-priority key from the input tensors + thread-local state and does an indirect
jump. [ezyang]

- **op → N backends:** the per-op vtable *is* the op→backend map; keys layer cross-cutting concerns
  (autograd wraps backend, autocast wraps autograd). `BackendSelect` handles device-less factory
  functions (e.g. `randn`) by inspecting args and re-dispatching. [ezyang]
- **selection:** pure **key priority**, computed not benchmarked — deterministic, zero runtime
  search. Fallthrough kernels are masked out at static-init for speed.
- **AMD support:** ROCm reuses the **CUDA dispatch key** (HIP presents a CUDA-compatible surface);
  arch is read at runtime via `gcnArchName`. New accelerators register under `PrivateUse1` with
  `TORCH_LIBRARY_IMPL`. [vllm rocm blog; pytorch tutorial]
- **license:** BSD-3 (PyTorch).
- **borrow:** the **vtable-per-op keyed by a composite dispatch key** is the exact data model
  perf_knowledge's `sota_registry.yaml` should mimic — our "key" is `(arch, dtype, regime)` instead of
  `(device, autograd…)`. The `PrivateUse1` + `TORCH_LIBRARY_IMPL` pattern is how a perf_knowledge-blessed
  kernel can be injected without patching the framework.

## torch.compile / TorchInductor (max-autotune)

URL: https://dev-discuss.pytorch.org/t/torchinductor-update-8-max-autotune-support-on-cpu-with-gemm-template/2439

Graph-level backend selection. With `mode="max-autotune"`, Inductor **benchmarks candidate
implementations at compile time** and keeps the fastest. [pytorch tutorial]

- **op → N backends:** for GEMM/conv it considers ATen, Triton templates, and CUTLASS (CUDA) / C++
  templates (CPU). Backends gated by `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS`
  (default `ATEN,TRITON,CPP`). `use_[backend]_template()` predicates decide eligibility;
  `select_algorithm.py` does the actual timing.
- **selection:** **online autotune** (measure-and-pick), cached. Gated by `is_big_gpu` (≥68 SMs).
- **AMD support:** yes — torch.compile/Inductor runs on ROCm; Triton templates are the portable path.
- **license:** BSD-3.
- **borrow:** the `MAX_AUTOTUNE_GEMM_BACKENDS` **env-controlled candidate set** + `use_*_template`
  **eligibility predicates** + `select_algorithm` **measured pick** is a clean three-part contract
  perf_knowledge's dispatcher can copy: *candidate list → predicate filter → measured/looked-up winner.*

## ROCm/aiter (AI Tensor Engine for ROCm)

URL: https://deepwiki.com/ROCm/aiter · https://rocm.docs.amd.com/projects/radeon/en/latest/docs/advanced/vllm/gemm-tuning.html

AMD's high-perf op library; the per-shape `tuned_gemm` dispatch is exactly the registry+dispatcher
pattern we already use. AITER sits between framework and hardware and does **runtime kernel dispatch
on workload characteristics**, choosing among **CK, hand-written ASM, Triton, FlyDSL, hipBLAS(Lt)**
backends. [deepwiki]

- **op → N backends:** `tuned_gemm.py` selects per GEMM; selection considers **arch, shape, dtype,
  quantization, and tuning results**. MoE/GEMM tuned params live in **CSV** configs resolved by an
  `AITER_CONFIG` class; model-specific CSVs (DeepSeek-V3, Kimi-K2.5) are **merged at runtime** to
  pick the fastest variant. [deepwiki]
- **selection:** **per-shape autotune DB**. Offline tuners (`GemmTuner`, `FmoeTuner`) benchmark
  ASM/CK/FlyDSL to populate the CSVs; at runtime the CSV is the lookup. The vLLM/gradlib flow
  (`VLLM_TUNE_GEMM=1` records `untuned_gemm.csv` → `gemm_tuner.py` emits `tuned_gemm_tpN.csv`) is the
  collect-then-tune loop. Pre-tuned CSVs ship in the optimized vLLM Docker. [rocm gemm-tuning docs]
- **AMD support:** native — gfx942 (MI300X/325X), gfx950 (MI355X) etc.; `get_gfx()` arch detection
  picks configs. (perf_knowledge already documents this in `backends/`.)
- **license:** **MIT** (verified, per-file SPDX headers).
- **borrow:** AITER's **CSV-keyed, runtime-merged, collect-then-tune** GEMM DB is the proven
  reference design for perf_knowledge's per-shape tuned-DB layer — and it's MIT, so the schema/loader are
  freely reusable. Adopt its `(arch, shape, dtype, quant)` key and the *base-CSV + model-overlay
  merge* idea directly.

## FlashInfer

URL: https://github.com/flashinfer-ai/flashinfer · https://arxiv.org/abs/2501.01005

Attention engine for LLM serving (used by vLLM, SGLang, MLC). Its differentiator is a **JIT compiler
that generates a specialized kernel per attention variant** from a variant spec (functor + dtypes +
extra tensors), because hand-maintaining a CUDA kernel per variant doesn't scale. [arxiv 2501.01005]

- **op → N backends:** one *variant spec* → JIT-compiled kernel; variant logic is compile-time, seq
  lengths are runtime (for the load-balanced scheduler). On ROCm it adds an **AITER backend** option
  alongside the HIP backend (CK FMHA). [rocm flashinfer blogs]
- **selection:** JIT spec-compile + runtime scheduler; AOT or per-first-use JIT.
- **AMD support:** **yes, growing** — ROCm port covers decode (Oct 2025) and FA2-based prefill
  (single/batch/ragged) on CDNA3 (MI300X/325X) and CDNA4 (MI355X); experimental AITER backend for
  prefill. *In progress (mark):* FP8 prefill, cascade, MLA, RoPE/ALiBi. [rocm flashinfer-release2]
- **license:** **Apache-2.0** (verified). *Note (mark):* upstream depends on some proprietary binary
  artifacts; ROCm fork at github.com/ROCm/flashinfer.
- **borrow:** the **variant-spec → kernel** idea is how perf_knowledge can describe "one logical operator,
  many concrete shapes/masks/dtypes" without an entry per combination — encode the *spec/capability
  predicate* in the registry and let the impl be JIT/templated. Also: FlashInfer-on-ROCm exposing an
  AITER sub-backend is a model for "library X delegates to AMD-native kernels."

## Liger-Kernel (LinkedIn)

URL: https://github.com/linkedin/Liger-Kernel · https://embeddedllm.com/blog/cuda-to-rocm-portability-case-study-liger-kernel

Triton fused training kernels (RMSNorm, RoPE, SwiGLU, fused-linear-CE, DPO/ORPO/SimPO losses).
HF-compatible drop-ins. Not a dispatcher per se — a *portable kernel set* that proves Triton's
write-once portability to AMD.

- **op → N backends:** single Triton source; "backend" = device. Selection is just patch/no-patch
  (monkeypatch HF modules).
- **selection:** none dynamic; user opts in. Portability via `is_hip()` checks.
- **AMD support:** **full since v0.4.0** — multi-GPU training +26% speed / −60% memory on AMD; the
  only real change was warp size 32→64 (reduce `num_warps` 32→16). gfx942/MI300X tier-1 supported.
  [embeddedllm case study]
- **license:** **BSD-2-Clause** (verified, repo).
- **borrow:** the **MI300X warp-size 64 / num_warps adjustment** is a concrete portability rule for
  any perf_knowledge cell that ports a 32-warp CUDA Triton kernel. Liger is a good *source library* to
  catalog as the SOTA impl for the fused training ops it covers.

## xFormers (Meta)

URL: https://github.com/facebookresearch/xformers · https://github.com/ROCm/xformers

Composable transformer blocks with a **`memory_efficient_attention` dispatcher** that picks among
multiple "Ops" by input shape/dtype.

- **op → N backends:** `fmha` has multiple Ops; on CUDA = cutlass/flash; on AMD = **Composable
  Kernel (CK)** ops (`ckF`, `ckB`, `ck_decoderF`, `ck_splitKF`). User can force one via
  `op=` argument. [diffusers/xformers docs]
- **selection:** the dispatcher **filters Ops by capability** (head-dim, dtype) and picks a
  compatible one; errors if none. Notable AMD limits: CK rejects fp32 and head-dim>256 (no cutlass
  fallback on AMD). [composable_kernel #1757]
- **AMD support:** yes via ROCm fork / experimental upstream rocm7.1 wheels; CK-only backends.
- **license:** **BSD-3-Clause** (verified, repo).
- **borrow:** xFormers is the clearest small example of a **capability-predicate dispatcher**: each
  Op advertises supported dtype/head-dim, the dispatcher filters then picks. perf_knowledge's registry should
  carry the same **per-cell capability predicates** (dtype set, head-dim range, page-size) so the
  serving dispatcher can filter before choosing. Also a cautionary tale: an op that exists on CUDA
  may have *no* AMD equivalent — the registry must encode "not available on arch X."

## NVIDIA Transformer Engine

URL: https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/attention/attention.html

FP8/FP4 Transformer acceleration. Multiple attention backends per framework: framework-native
("unfused"), flash-attention, and **cuDNN fused attention** (the only FP8-DPA path, via sub-backend 2).

- **op → N backends:** per-op set {unfused, flash-attn, cuDNN-fused}; FP8 only on cuDNN sub-backend 2.
- **selection:** **env flags + capability fallback**. `NVTE_FUSED_ATTN=1/0`, `NVTE_FLASH_ATTN`,
  `DelayedScaling.fp8_dpa/fp8_mha` recipe options; if the chosen cuDNN kernel isn't available it
  **warns and falls back to unfused**. Perf note baked into docs: flash-attn wins on Ampere, cuDNN
  +20–50% on Hopper. [TE attention docs]
- **AMD support:** **no** (NVIDIA-only; there is a separate ROCm "TE" effort but upstream is NV).
  *(mark.)*
- **license:** **Apache-2.0** (verified).
- **borrow:** two ideas — (1) the **recipe-flag → backend** mapping with **automatic fall-back +
  warning** when a backend is unavailable (perf_knowledge dispatcher should degrade gracefully, not error,
  and log why); (2) **embedding the perf crossover ("cuDNN wins on Hopper") in the docs/registry** so
  selection is explainable, not magic — perf_knowledge cells already do this; TE validates the pattern.

## vLLM — attention/GEMM backend abstraction

URL: https://docs.vllm.ai/en/latest/design/attention_backends/ · https://vllm.ai/blog/2026-02-27-rocm-attention-backend

The reference serving-stack dispatcher. `get_attn_backend` registry + a **per-platform
priority-ordered list**; explicit override via `VLLM_ATTENTION_BACKEND` / `--attention-backend`.

- **op → N backends:** attention backends FLASH_ATTN(FA2/3/4 by SM), FLASHINFER, TRITON_ATTN,
  TORCH_SDPA, FLASHMLA, + ROCm: **ROCM_AITER_FA, ROCM_AITER_MLA, ROCM_AITER_TRITON_MLA, ROCM_ATTN,
  TRITON_ATTN**. MLA uses **separate prefill vs decode** backends.
- **selection:** explicit override (validated against dtype/head-dim/compute-cap, errors with reason
  if incompatible) → else **first compatible from priority list, per platform**. ROCm has its **own**
  selection logic: with `VLLM_ROCM_USE_AITER=1`, auto-selects ROCM_AITER_FA (MHA) / ROCM_AITER_MLA
  (MLA), falling back to TRITON_ATTN/ROCM_ATTN (incl. Radeon, no-AITER). Default ROCm backend changed
  to ROCM_ATTN in v0.19.0. [vllm rocm blog]
- **AMD support:** first-class; AITER gates GEMM/RMSNorm/MoE too, so `VLLM_ROCM_USE_AITER=1` is
  required even when overriding only attention.
- **license:** Apache-2.0.
- **borrow:** this is the **single most directly applicable design** for the serving-side dispatcher.
  Copy: (1) `get_attn_backend`-style **registry returning a class by key**; (2) **per-platform
  priority list** (the data perf_knowledge emits); (3) **override flag that is validated, not trusted**;
  (4) **separate prefill/decode selection**; (5) a single master switch (`VLLM_ROCM_USE_AITER`) that
  flips a whole family. perf_knowledge's job is to *produce the priority lists + capability predicates* vLLM
  (or our stack) consumes.

## SGLang — attention/GEMM backend abstraction + hardware plugin

URL: https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/attention_backend.md · https://github.com/sgl-project/sglang/issues/20372

Same family as vLLM, with an explicit **platform plugin interface** (issue #20372): a `Platform`
class exposes `get_attention_backends()`, `get_attn_backend_cls_str()`, `get_graph_runner_class()`.
`RocmPlatform` returns aiter/wave/flashinfer; `CudaPlatform` returns flashinfer/fa3/triton.

- **op → N backends:** MHA {FlashInfer, FA3/FA4, Triton, SDPA, FlexAttn, TRTLLM_MHA, AITER…} and MLA
  {FlashInfer, FA3, FlashMLA, Cutlass-MLA, TRTLLM_MLA, Triton…}; supports **hybrid prefill/decode
  backends** (`--prefill-attention-backend` + `--attention-backend`).
- **selection:** auto by **arch→backend map** (sm80/86/89→FlashInfer, sm90→FA3, sm100→FlashInfer/
  TRTLLM, AMD→Triton/AITER), with capability constraints (e.g. MLA page-size per backend: FlashInfer
  MLA=1, FlashMLA=64, Cutlass-MLA=128, TRTLLM_MLA∈{32,64}). **GEMM** auto-order: DeepGEMM → FlashInfer
  TRTLLM → CUTLASS → **AITER (AMD)** → Triton. Explicit override via `--attention-backend`.
- **AMD support:** Triton workhorse + AITER (`SGLANG_USE_AITER=1`); FlashInfer-ROCm emerging.
- **license:** Apache-2.0.
- **borrow:** the **`Platform.get_attention_backends()` plugin interface** is the cleanest API
  contract for "ask the registry what backends exist for *this* hardware, in priority order." perf_knowledge
  should expose exactly this shape: `get_backends(op, arch, dtype, regime) -> [backend...]` ordered,
  each with capability predicates (page-size, fp8-kv, spec-decode, sliding-window). The **per-backend
  page-size constraint table** is a model for perf_knowledge capability fields.

## BackendBench (Meta / meta-pytorch)

URL: https://github.com/meta-pytorch/BackendBench

Not a dispatcher — the **correctness+perf gate** for a registry. Lets you drop custom kernels in a
directory, dynamically **override PyTorch core ops at runtime** to form a pip-installable backend,
then validates against PyTorch's **OpInfo** test suite (271 ops) — "if it passes OpInfo it's likely
correct enough to upstream." [repo, docs/correctness.md]

- **op → N backends:** one directory per op you fill with an impl; runtime override builds a full
  backend.
- **selection:** n/a — it *measures*; correctness-first (most kernels 70–100% of eager, a few ~1.2×).
  Related Meta work TritorX generated kernels for 481 ATen ops passing 20k+ OpInfo tests.
- **AMD support:** runs wherever PyTorch+the kernels run (Triton path → ROCm). *(not AMD-specific.)*
- **license:** **BSD-3** (verified).
- **borrow:** adopt the **OpInfo-as-correctness-gate** rule: a perf_knowledge cell cannot be marked
  `status: sota` until its impl passes OpInfo (or our equivalent) for the claimed dtypes/shapes. The
  **directory-per-op + runtime-override-to-form-a-backend** layout is also a ready-made harness to
  *validate* perf_knowledge-registered kernels before the serving stack trusts them.

## Hugging Face Kernel Hub (`huggingface/kernels`)

URL: https://github.com/huggingface/kernels · https://huggingface.co/docs/kernels/index

The closest existing thing to a **machine-queryable operator registry served over a network**.
Kernels are a first-class Hub repo type; `get_kernel("kernels-community/activation")` downloads a
build **matched to your CUDA/Torch version** and registers it as a native `torch.ops` operator.

- **op → N backends:** `register_kernel_mapping()` / `use_kernel_mapping()` maps a **layer → kernel
  repo per backend** (`cuda`, `rocm`, `metal`, `xpu`, `cpu`) and **per mode** (training/inference/
  torch.compile) with a `Mode.FALLBACK` chain. `kernelize(model, mode=...)` swaps forwards in-place.
- **selection:** **(backend, mode) keyed mapping** + automatic backend detection + version pinning
  (major-version branches that never break API). Attention via `attn_implementation=
  kernels-community/flash-attn2 | vllm-flash-attn3 | paged-attention`.
- **AMD support:** **yes** — `rocm` is a first-class backend; CI build-variant order is CUDA, **ROCm**,
  XPU, Metal, CPU.
- **license:** Apache-2.0 (the `kernels` library).
- **borrow:** this is the **delivery + registry** model perf_knowledge should emulate for distribution:
  (1) a **(backend, mode) → kernel-repo mapping** that the serving stack queries; (2) **environment
  auto-matching** (ROCm/Torch version → correct build); (3) **major-version branches as a stability
  contract**; (4) `kernelize()`-style **in-place forward swap** so a model adopts perf_knowledge-blessed
  kernels with no modeling-code change. Our `sota_registry.yaml` ≈ their kernel-mapping, but with
  arch/dtype/regime keys added.

## Unsloth kernels

URL: https://unsloth.ai/docs/blog/unleash-the-power-of-amd-official-support-for-unsloth-is-here

Hand-written Triton kernels for fine-tuning (attention, RoPE, MLP, backprop). Like Liger, a portable
kernel *set* rather than a dispatcher.

- **op → N backends:** Triton source; routes through HIP via Triton `is_hip()`, **NVIDIA precedence
  on mixed systems**. Two tiers: *Fully Supported* (arch-specific tuning) vs *Supported* (generic
  HIP path).
- **selection:** multi-path ROCm detection (`rocminfo`/`amd-smi`/`hipconfig`); on AMD, FA2 is
  unavailable so it **auto-falls back to xFormers (CK)**.
- **AMD support:** **yes, official** — MI300X (gfx942, needs `HSA_OVERRIDE_GFX_VERSION=9.4.2`) +
  Radeon RDNA2/3/3.5/4; 95 CI tests run without physical AMD HW.
- **license:** Apache-2.0 (Unsloth open kernels). *(mark: confirm per-file; commercial tiers exist.)*
- **borrow:** the **two-tier support label** ("fully tuned" vs "runs generically") is a clean
  `status` taxonomy for perf_knowledge cells beyond binary sota/works. The **multi-path arch detection** and
  **`HSA_OVERRIDE_GFX_VERSION`** note are practical entries for perf_knowledge's env/detection reference.

---

## Comparison table

| Project | op → backend model | selection mechanism | AMD support | license |
|---|---|---|---|---|
| **FlagGems** | one Triton source / op, backend = device; out-of-tree ATen registration | runtime per-fn dispatch + `LibEntry` cache (bypasses autotune on hit); per-vendor config dir | Yes (ROCm); MI300X configs *(mark)* | Apache-2.0 |
| **ATen dispatcher** | per-op vtable keyed by dispatch key | key priority (computed, not benchmarked); `BackendSelect` for factory fns | ROCm = CUDA key (`gcnArchName`); `PrivateUse1` | BSD-3 |
| **torch.compile / Inductor** | candidate set per GEMM/conv: ATen/Triton/CUTLASS/CPP | online **max-autotune** (measure+pick), env-gated candidate set | Yes (Triton templates on ROCm) | BSD-3 |
| **ROCm/aiter `tuned_gemm`** | per-shape: CK / ASM / Triton / FlyDSL / hipBLAS | **per-shape CSV DB**, runtime-merged; offline tuners populate; `get_gfx` arch | Native (gfx942/950) | MIT |
| **FlashInfer** | variant-spec → JIT kernel; ROCm adds AITER sub-backend | JIT spec-compile + runtime scheduler | Yes, growing (CDNA3/4); FP8/MLA *in progress (mark)* | Apache-2.0 |
| **Liger-Kernel** | one Triton source / op (training fusions) | static opt-in monkeypatch; `is_hip()` | Full since v0.4.0 (warp 64) | BSD-2 |
| **xFormers** | `fmha` Ops; CUDA=cutlass/flash, AMD=CK | **capability-predicate** filter (dtype/head-dim) + pick; `op=` override | Yes (CK-only; no fp32) | BSD-3 |
| **NVIDIA TE** | {unfused, flash, cuDNN-fused}; FP8 on cuDNN | env flags + recipe + **fallback-with-warning** | No (NV-only) *(mark)* | Apache-2.0 |
| **vLLM** | `get_attn_backend` registry; per-platform list | validated override → first-compatible priority list; ROCm own logic; master `VLLM_ROCM_USE_AITER` | First-class | Apache-2.0 |
| **SGLang** | `Platform.get_attention_backends()` plugin | arch→backend map + capability constraints; hybrid prefill/decode; override flag | Triton + AITER | Apache-2.0 |
| **BackendBench** | dir-per-op → runtime override = backend | n/a (correctness gate via OpInfo; perf measured) | Wherever PyTorch runs | BSD-3 |
| **HF Kernel Hub** | `(backend, mode) → kernel-repo` mapping | mapping + auto env-match + version pin; `kernelize()` swap | First-class `rocm` | Apache-2.0 |
| **Unsloth** | one Triton source / op (fine-tune) | `is_hip()` route + multi-path detect; FA2→xFormers fallback | Official (MI300X + Radeon) | Apache-2.0 *(mark)* |

---

## What we borrow

Concrete and actionable for **perf_knowledge** (the registry/KB) and **e2e_workflow** (the dispatcher the
serving stack consumes):

- **(a) Registry data model = ATen vtable + AITER CSV key.** Model `sota_registry.yaml` as a
  per-operator table keyed by a **composite key `(arch, dtype, regime)`** (our analog of ATen's
  dispatch key), each cell pointing at an impl **plus a per-shape tuned-DB pointer** in AITER's
  `(arch, shape, dtype, quant)` CSV style with **base-CSV + model-overlay merge**. MIT AITER means we
  can reuse the schema/loader outright.
- **(b) Dispatcher API = SGLang's `get_attention_backends()` + vLLM's priority list.** Expose
  `get_backends(op, arch, dtype, regime) -> [backend...]` **ordered by priority**, each carrying
  **capability predicates** (dtype set, head-dim range, page-size, fp8-kv, spec-decode,
  sliding-window). The serving stack filters by predicate, then takes the first compatible — exactly
  vLLM/SGLang behavior. Support **separate prefill/decode** answers.
- **(c) Override that is validated, not trusted (vLLM/TE).** Every selection must accept an env/flag
  override but **validate it against the cell's capability predicate** and, on mismatch,
  **fall back with a logged reason** (TE's warn-and-fallback) rather than crash. Add a **single
  master switch** per family (à la `VLLM_ROCM_USE_AITER`).
- **(d) Correctness stamp = BackendBench OpInfo gate.** No cell becomes `status: sota` until its impl
  passes OpInfo (or our equivalent) for the claimed dtypes/shapes. Reuse BackendBench's
  **dir-per-op + runtime-override** harness (BSD-3) to validate perf_knowledge kernels before the dispatcher
  trusts them.
- **(e) Three-part selection contract (Inductor) + capability predicates (xFormers).** Make the
  dispatcher a clean pipeline: **candidate list → predicate filter → winner** (looked-up tuned-DB
  entry, or measured if cold). Carry per-cell capability predicates so "exists on CUDA, absent on
  gfx942" is first-class data, not a runtime surprise.
- **(f) Distribution & adoption = HF Kernel Hub.** For shipping perf_knowledge-blessed kernels, mirror the
  **`(backend, mode) → repo` mapping**, **environment auto-matching** (ROCm/Torch version →
  build), **major-version stability branches**, and a `kernelize()`-style **in-place forward swap**
  so models adopt our kernels with zero modeling-code change.
- **(g) Status taxonomy & portability rules.** Adopt Unsloth's **two-tier label** (fully-tuned vs
  generic-HIP) to enrich `status`, and record the concrete portability rules (**MI300X warp=64 →
  num_warps halved** from Liger; **`HSA_OVERRIDE_GFX_VERSION=9.4.2`** from Unsloth) in perf_knowledge's env/
  detection reference.

---

## Sources

- FlagGems repo — https://github.com/FlagOpen/FlagGems
- FlagGems joins PyTorch Ecosystem (PyTorch blog) — https://pytorch.org/blog/flaggems-joins-the-pytorch-ecosystem-triton-powered-operator-library-for-universal-ai-acceleration/
- FlagGems multi-backend RFC (PyTorch #139602) — https://github.com/pytorch/pytorch/issues/139602
- FlagGems runtime dir tree — https://github.com/FlagOpen/FlagGems/tree/master/src/flag_gems/runtime
- FlagGems LICENSE (Apache-2.0) — https://github.com/FlagOpen/FlagGems/blob/master/LICENSE
- PyTorch dispatcher (ezyang) — https://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/
- Extending dispatcher for a new backend — https://docs.pytorch.org/tutorials/advanced/extend_dispatcher
- TorchInductor max-autotune (CPU GEMM template) — https://dev-discuss.pytorch.org/t/torchinductor-update-8-max-autotune-support-on-cpu-with-gemm-template/2439
- Max-autotune GEMM backends env / gating — https://github.com/pytorch/pytorch/issues/125683
- ROCm/aiter (DeepWiki) — https://deepwiki.com/ROCm/aiter
- aiter LICENSE (MIT) — https://github.com/ROCm/aiter/blob/main/LICENSE
- GEMM tuning for vLLM (untuned→tuned CSV, gradlib) — https://rocm.docs.amd.com/projects/radeon/en/latest/docs/advanced/vllm/gemm-tuning.html
- FlashInfer repo — https://github.com/flashinfer-ai/flashinfer
- FlashInfer paper (arXiv:2501.01005) — https://arxiv.org/abs/2501.01005
- FlashInfer on ROCm (decode) — https://rocm.blogs.amd.com/artificial-intelligence/flashinfer/README.html
- FlashInfer on ROCm release 2 (prefill + AITER backend) — https://rocm.blogs.amd.com/artificial-intelligence/flashinfer-release2/README.html
- ROCm/flashinfer fork — https://github.com/ROCm/flashinfer
- Liger-Kernel repo — https://github.com/linkedin/Liger-Kernel
- Liger v0.4.0 (full AMD support) — https://github.com/linkedin/Liger-Kernel/releases/tag/v0.4.0
- Liger CUDA→ROCm case study (EmbeddedLLM) — https://embeddedllm.com/blog/cuda-to-rocm-portability-case-study-liger-kernel
- xFormers repo — https://github.com/facebookresearch/xformers
- xFormers ROCm fork — https://github.com/ROCm/xformers
- diffusers attention backends (xFormers CK ops) — https://github.com/huggingface/diffusers/blob/main/docs/source/en/optimization/attention_backends.md
- composable_kernel #1757 (CK dtype/head-dim limits) — https://github.com/ROCm/composable_kernel/issues/1757
- Transformer Engine attention backends / FP8 selection — https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/attention/attention.html
- Transformer Engine repo (Apache-2.0) — https://github.com/NVIDIA/TransformerEngine
- vLLM attention backend feature support — https://docs.vllm.ai/en/latest/design/attention_backends/
- vLLM on ROCm attention (AMD blog, 2026-02) — https://vllm.ai/blog/2026-02-27-rocm-attention-backend
- vLLM ROCm platform — https://docs.vllm.ai/en/stable/api/vllm/platforms/rocm/
- SGLang attention backend doc — https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/attention_backend.md
- SGLang hardware plugin system (issue #20372) — https://github.com/sgl-project/sglang/issues/20372
- SGLang attention backend default choice (issue #5064) — https://github.com/sgl-project/sglang/issues/5064
- BackendBench repo — https://github.com/meta-pytorch/BackendBench
- BackendBench correctness doc — https://github.com/meta-pytorch/BackendBench/blob/main/docs/correctness.md
- HF kernels repo — https://github.com/huggingface/kernels
- HF Kernel Hub docs (index + layers) — https://huggingface.co/docs/kernels/index
- HF "Learn the Kernel Hub in 5 Minutes" — https://huggingface.co/blog/hello-hf-kernels
- Unsloth official AMD support — https://unsloth.ai/docs/blog/unleash-the-power-of-amd-official-support-for-unsloth-is-here
- Unsloth AMD install guide — https://unsloth.ai/docs/get-started/install/amd
