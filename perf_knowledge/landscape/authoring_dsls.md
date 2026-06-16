---
title: Kernel-Authoring DSLs & Codegen Frameworks — landscape (AMD CDNA3/CDNA4 focus)
kind: landscape
updated: 2026-06-09
scope: DSLs and codegen frameworks for WRITING fast GPU kernels, scored on MI300X (gfx942, CDNA3) / MI350X (gfx950, CDNA4) usability
status: survey
sources:
  - https://arxiv.org/abs/2511.08083
  - https://triton-lang.org/main/gluon/index.html
  - https://github.com/tile-ai/tilelang
  - https://developer.nvidia.com/blog/achieve-cutlass-c-performance-with-python-apis-using-cute-dsl/
  - https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/index.html
  - https://www.modular.com/blog/modular-x-amd-unleashing-ai-performance-on-amd-gpus
  - https://docs.jax.dev/en/latest/pallas/gpu/index.html
  - https://arxiv.org/abs/2504.16214
  - https://exo-lang.dev/
  - https://iree.dev/guides/deployment-configurations/gpu-rocm/
  - https://github.com/NVIDIA/cudnn-frontend
  - https://arxiv.org/abs/2507.23194
---

# Kernel-Authoring DSLs & Codegen Frameworks — Landscape

A survey of the languages and codegen frameworks used to *write* fast GPU kernels, scored
specifically on whether they are a **real option on AMD CDNA3 (MI300X/MI325X, gfx942) and CDNA4
(MI350X/MI355X, gfx950) today**, and on what the GEAK / agentic kernel-authoring workflow should
borrow from each. Companion to `perf_knowledge/languages/` (per-language deep dives) and
`perf_knowledge/backends/` (library backends).

## TL;DR

- **Real and production-grade on MI300X/MI350X today:** **Triton** (first-class AMD backend, ships in
  ROCm 7.x, `amd_mfma`/`amd_wmma` layouts), **CK / CK-Tile** (AMD's own C++ tile library, the
  hand-tuned baseline to beat), **TileLang** (TVM-based Pythonic tile DSL, AMD-blessed, MI300X
  FlashMLA at AITER-assembly parity), and **Mojo / Modular MAX** (GA on MI300/MI325 since Jun 2025,
  `"hip"` backend, BF16 matmul beating hand-tuned). **Gluon** (Triton's low-level dialect) is real on
  AMD and is the most interesting *new* lever — AMD published a Gluon GEMM tutorial reaching near-peak
  on CDNA, including CDNA4 scaled-MFMA / MXFP4 intrinsics.
- **Real but newest / most opinionated:** **HipKittens** (HazyResearch C++ tile primitives, the AMD
  port of ThunderKittens, arXiv Nov 2025, MLSys 2026, MIT license, now an AITER backend) — it encodes
  *the* AMD-specific scheduling wisdom (8-wave ping-pong, 4-wave interleave; wave-specialization is the
  wrong default on CDNA).
- **NVIDIA-only (borrow ideas, not code):** **CuTe-DSL / CUTLASS 4.x** (Python layout algebra, the
  reference for "layouts as first-class types"), **JAX Pallas + Mosaic GPU** (Hopper+ only on GPU),
  **cuDNN frontend graph API** (declarative fusion-graph authoring), **Hidet / Hexcute / Tilus**
  (CUDA-targeted; Hexcute's automatic layout/task-mapping synthesis is the key transferable idea).
- **Hardware-agnostic scheduling research:** **Exo / Exo 2** (user-schedulable language; correctness-
  preserving scheduling-as-a-library; CPU/accelerator-focused, GPU is future work) — borrow the
  *cursor / scheduling-library* model for GEAK transformation primitives.
- **For GEAK specifically:** the highest-value pattern is **author in a tile DSL, not raw HIP/asm** —
  it shrinks the agent's search space and makes candidates compile-correct more often (cf. AMD's own
  GEAK Triton agents, and NVIDIA's μCUTLASS+Speed-of-Light agent paper, arXiv 2603.29010).

---

## Triton (+ AMD backend)

URL: https://rocm.blogs.amd.com/software-tools-optimization/optimizing-triton-kernel.html ·
https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py

- **Programming model:** Tile-level SPMD. You write per-"program" (block) code over N-D tiles; the
  compiler owns layouts, shared-memory allocation, vectorization, and software pipelining. Python DSL
  via `@triton.jit`, lowers through TritonGPU MLIR IR.
- **AMD/CDNA support:** First-class. Ships as Triton v3.3 in **ROCm 7.0**, auto-targets HIP. Has
  CDNA-specific GPU-IR layouts: `amd_mfma` (matrix core), `amd_wmma`, blocked/shared/sliced. Tunables
  include `matrix_instr_nonkdim` (16 → mfma_16x16, which usually wins on MI300X; 32 → mfma_32x32),
  `waves_per_eu`, `kpack`. Native FNUZ FP8 (`fp8e4b8`/`fp8e5b16`) on CDNA3; XF32 on gfx942. Dedicated
  `ROCm/triton` fork + `occ.sh` occupancy tooling. Also the FlashAttention fallback path on AMD
  (`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`).
- **Maturity:** Very high. The de-facto portable kernel DSL; huge ecosystem; AMD-validated tutorials
  on MI300X.
- **Best for:** Attention, fused elementwise/normalization, fused MoE, GEMM where you want portability
  + good (not always peak) perf with modest effort.
- **Strengths/limits:** Productivity + portability are unmatched. Limit: compiler auto-scheduling can
  leave perf on the table vs. hand-tuned asm/CK; advanced combos (autotune + FP8) still have rough
  edges on AMD; you don't control wave specialization/layout directly — that's what Gluon is for.
- **License:** MIT.
- **What we borrow:** This is GEAK's primary authoring target on AMD. Borrow the layout/tunable
  vocabulary (`matrix_instr_nonkdim`, `waves_per_eu`, `kpack`) as agent search dimensions; use the
  `amd_mfma` IR as the readable artifact for the reflector to reason about.

## Gluon (Triton low-level dialect)

URL: https://triton-lang.org/main/gluon/index.html ·
https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html

- **Programming model:** Same tile-based SPMD frontend as Triton (`@gluon.jit`, shared Python
  frontend), but **exposes the low-level knobs Triton hides**: explicit tile layouts, shared-memory
  allocation, data movement, warp/wave specialization, and target-specific instructions. Lowers to
  TritonGPU IR / MLIR. Positioned as "Triton when the compiler's auto-schedule isn't enough."
- **AMD/CDNA support:** Yes, and improving fast. AMD published a **Gluon GEMM tutorial** ("From Naive
  to Near-Peak") reaching near-peak on CDNA, and Gluon exposes **CDNA4 scaled-MFMA / MXFP4**
  intrinsics (e.g. `gl.amd.cdna4.mfma_scaled`, `v_mfma_scale_f32_16x16x128_f8f6f4`). ROCm's `iris`
  multi-GPU framework has an experimental Gluon backend. Requires ROCm 7.0+ and a recent Triton commit
  (experimental APIs).
- **Maturity:** Experimental but officially in-tree in `triton-lang/triton` and documented; the
  fastest-moving "new low-level option" on AMD.
- **Best for:** Squeezing the last 10-30% on GEMM/attention where Triton's auto-schedule plateaus, and
  for reaching CDNA4-only instructions (MXFP4 scaled-MFMA) that Triton doesn't expose.
- **Strengths/limits:** Same toolchain/frontend as Triton (low switching cost) but with explicit
  control → can approach hand-tuned without leaving Python. Limit: experimental/unstable APIs; requires
  real GPU hardware knowledge (layouts, banks, wave scheduling).
- **License:** MIT (part of Triton).
- **What we borrow:** **High-value.** Gluon is the bridge between "agent writes Triton" and "expert
  hand-tunes asm." GEAK should treat Gluon as a *second-tier escalation target*: when a Triton kernel
  is near but below SOL, escalate to Gluon to control layout/wave-specialization explicitly. Borrow its
  explicit-layout vocabulary for the agent's optimization moves.

## TileLang

URL: https://github.com/tile-ai/tilelang · https://arxiv.org/abs/2504.17577 ·
https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-tilelang-kernel/

- **Programming model:** Pythonic tile DSL on **Apache TVM**. Tiles are first-class; core ops
  (`GEMM`, `COPY`, `REDUCE`, `ATOMIC`) express dataflow, separate scheduling primitives/annotations
  express optimization. **Three abstraction levels**: L1 pure compute (no HW), L2 GPU-aware
  (Triton-like), L3 full thread-level control (near hand-written HIP). Built-in autotuner.
- **AMD/CDNA support:** Strong and AMD-endorsed. HIP/ROCm backend (`USE_ROCM`, builds for gfx942 and
  gfx950), uses MatrixCore + async copy. **MI300X FlashMLA at parity with AITER hand-tuned assembly**
  (Apr 2025). AMD's ROCm blog actively promotes TileLang as a more-AMD-friendly alternative to Triton
  (FA kernel "~80 lines vs >500 in CUDA, equivalent perf"). Dec 2025: added a CuTe-DSL backend.
- **Maturity:** Mid-high and rising fast; open-sourced Jan 2025, very active. Academic + Microsoft
  Research + PKU lineage.
- **Best for:** Attention/FlashMLA, GEMM, custom fused ops where you want more explicit tiling control
  than Triton but still in Python; AMD targets specifically.
- **Strengths/limits:** Explicit tile/schedule control + autotuner + multi-backend (CUDA/ROCm/Metal/
  Ascend/CPU). Limit: TVM dependency (heavier stack); HipKittens authors argue neither Triton nor
  TileLang "systematically" reaches AMD asm-level peak across the board (uncertain/contested claim).
- **License:** MIT.
- **What we borrow:** **High-value on AMD.** The 3-level abstraction ladder maps cleanly onto an
  agent escalation strategy (start L2, drop to L3 only where needed). The FlashMLA-at-AITER-parity
  result is a strong existence proof for "Python DSL can match asm on MI300X." Consider TileLang as a
  GEAK authoring target alongside Triton/Gluon.

## ThunderKittens + HipKittens (HazyResearch)

URL: https://arxiv.org/abs/2511.08083 · https://github.com/HazyResearch/HipKittens ·
https://hazyresearch.stanford.edu/blog/2025-11-09-hk

- **Programming model:** C++-embedded, PyTorch-inspired **tile primitives**. Tiles sized to matrix-core
  units; coalesced/bank-conflict-free memory ops; Python-inspired bulk-compute functions wrapping
  asm+HIP; async loads/stores via direct buffer loads to LDS. **Interface = tiles + ops on tiles;
  implementation = how tiles map to HW** (the explicit separation is the design thesis). ThunderKittens
  is the NVIDIA original; **HipKittens (HK) is the AMD port**.
- **AMD/CDNA support:** Native and the whole point. Supports **CDNA3 (gfx942, MI300X/MI325X — `cdna3`
  branch)** and **CDNA4 (gfx950, MI350X/MI355X — main)**. Validated on MI325X and MI355X.
- **Maturity:** New (arXiv Nov 2025), **accepted to MLSys 2026**, **now an official AITER backend**
  (first HK kernels landed in AITER). MIT license. No tagged releases yet; ~2.3k commits.
- **Best for:** BF16/FP8 GEMM, GQA/MHA attention fwd+bwd (head dim 64/128, causal/non-causal), RoPE,
  LayerNorm. Reported to compete with or beat **all** AMD baselines on average (PyTorch, AITER, CK,
  hipBLASLt, Triton), including AMD's raw-asm kernels.
- **Strengths/limits:** Encodes the *correct AMD scheduling patterns* — key finding: **NVIDIA-style
  producer/consumer wave specialization underperforms on CDNA3/CDNA4** (AMD's static register
  allocation makes producer waves waste registers, shrinking output tiles / arithmetic intensity).
  HK instead uses **8-wave ping-pong** and **4-wave interleave**. Limit: opinionated C++ (steeper than
  Python DSLs); small kernel surface today; needs deep HW knowledge to extend.
- **License:** MIT.
- **What we borrow:** **Highest-value AMD-specific knowledge.** The "tiles + ops on tiles" interface/
  implementation split, and especially the **scheduling rule: prefer 8-wave ping-pong / 4-wave
  interleave over wave specialization on CDNA**. This is a hard-won, citable, AMD-only insight that
  GEAK's optimizer and our `optimization/mfma_scheduling` + `optimization/pipelining` docs should
  encode as a prior. See `perf_knowledge/languages/hipkittens/`.

## CUTLASS / CuTe / CuTe-DSL (Python)

URL: https://developer.nvidia.com/blog/achieve-cutlass-c-performance-with-python-apis-using-cute-dsl/ ·
https://github.com/NVIDIA/cutlass · https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/

- **Programming model:** CuTe = a unified **layout algebra** (layouts, tensors, hardware "atoms",
  explicit thread/data hierarchy). CUTLASS 3.x exposes it in C++; **CuTe-DSL (CUTLASS 4.x, Beta)**
  brings the *same* model to Python via `@cute.jit` / `@cute.kernel`, JIT-compiled through MLIR → PTX →
  SASS. Hybrid AST-rewrite + tracing compilation; ~2 orders of magnitude faster compile than C++
  templates.
- **AMD/CDNA support:** **None.** NVIDIA-only (Ampere/Hopper/Blackwell tensor cores). (There is a
  separate community "CUTLASS port"/rocWMMA-style effort on AMD — see `perf_knowledge/languages/cutlass_port` —
  but CuTe-DSL itself does not target CDNA.)
- **Maturity:** CuTe C++ very mature; CuTe-DSL public beta, evolving. pip `nvidia-cutlass-dsl`.
- **Best for (on NV):** Dense/grouped GEMM, FMHA, peak tensor-core kernels.
- **Strengths/limits:** The gold standard for layout reasoning. Limit: NV-only; beta DSL.
- **License:** BSD-3-Clause.
- **What we borrow:** **Concepts, not code.** CuTe's **layout algebra as first-class types** is the
  single most influential idea in the whole space (Hexcute, Tilus, TileLang's CuTe backend all cite
  it). Use it as the mental model when documenting AMD `amd_mfma` layouts and tile→register mappings.
  CuTe-DSL's fast-JIT-for-agents argument directly motivates GEAK's preference for DSL over C++.

## AMD Composable Kernel (CK) + CK-Tile

URL: https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/index.html ·
https://rocm.blogs.amd.com/software-tools-optimization/building-efficient-gemm-kernels-with-ck-tile-vendo/

- **Programming model:** C++ template metaprogramming with **compile-time coordinate transforms** and
  **tile distribution**. CK-Tile is the tile-based abstraction layer: zero-overhead abstractions,
  thread cooperation, portable-performance templates. You define Problem (dtypes/layout/tile shapes/
  scheduler flags) + tunable params (BlockSize, M/N/K-PerBlock, M/N-PerXDL, AK1/BK1).
- **AMD/CDNA support:** **Native, first-party AMD.** XDL→MFMA instructions on gfx942; gfx950 support.
  This is the backend behind much of AITER/hipBLASLt/MIOpen. (Repo moved into `ROCm/rocm-libraries`.)
- **Maturity:** High, production. The reference hand-tuned C++ path on AMD.
- **Best for:** GEMM, batched GEMM, fused-MHA, fused-MoE, SmoothQuant/INT8, elementwise — anything
  needing peak with deep tuning.
- **Strengths/limits:** Peak or near-peak perf; the baseline everyone benchmarks against. Limit:
  heavy C++ template programming, steep learning curve, slow compiles, large config surface — exactly
  the friction that DSLs (and agents) try to remove.
- **License:** MIT.
- **What we borrow:** CK-Tile's **tunable-parameter taxonomy** (tile shapes, M/N-PerXDL, K-pack,
  scheduler flags) is a ready-made agent search space; its tile-distribution / coordinate-transform
  model documents *what* a good AMD layout looks like. Treat CK as the SOTA baseline + knob dictionary,
  not as an agent authoring target (too verbose for LLM emission). See `perf_knowledge/backends/composable_kernel_lib`.

## Mojo / Modular MAX

URL: https://www.modular.com/blog/modular-x-amd-unleashing-ai-performance-on-amd-gpus ·
https://docs.modular.com/mojo/manual/gpu/fundamentals/ · https://arxiv.org/html/2509.21039v1

- **Programming model:** Mojo is a Python-family, MLIR-based systems language. GPU kernels written with
  the stdlib `gpu` package: explicit threads/blocks (1-3D), `DeviceContext` for compile+launch, strong
  static typing + compile-time metaprogramming + portable hardware dispatch. Vendor-agnostic GPU
  modules embedded *in the language*, not a third-party lib.
- **AMD/CDNA support:** **GA since Jun 2025** across MI300/MI325 (MI355X added Sep 2025, Platform
  25.6). `DeviceContext(api="hip")` selects the AMD backend. Reported MI300X BF16 matmul **beating
  hand-tuned** while staying portable; MAX server shows up to +53% prefill / +32% decode throughput vs.
  baseline stacks.
- **Maturity:** Mid-high, fast-moving, commercially backed (Modular). Large open kernel repo
  (`modular/modular`).
- **Best for:** End-to-end portable kernels + serving (MAX); matmul/attention/custom ops where you want
  one source for NVIDIA + AMD.
- **Strengths/limits:** True single-source portability with peak-competitive perf and a real language
  (not just a kernel DSL). Limit: newer ecosystem; some pieces historically had non-OSS components
  (now largely open); team needs to learn Mojo.
- **License:** Apache-2.0 (with LLVM exceptions) for the open Modular Platform / Mojo stdlib.
- **What we borrow:** Existence proof that a **portable language can beat hand-tuned on MI300X**.
  The `"hip"`/`"cuda"` unified-dispatch model is a clean template for portability. A candidate GEAK
  authoring target where single-source NV+AMD matters. See `perf_knowledge/languages/mojo`.

## JAX Pallas (+ Mosaic GPU)

URL: https://docs.jax.dev/en/latest/pallas/gpu/index.html ·
https://docs.jax.dev/en/latest/pallas/gpu/reference.html

- **Programming model:** JAX-embedded kernel DSL. Same JAX API at a lower level — you reason about
  memory spaces (`plgpu.GMEM`, shared, registers, Blackwell `plgpu.TMEM`) and grid tiling. On GPU it
  lowers to **Mosaic GPU** (lower-level, more control than Triton; warp specialization + pipelining
  for FA3/ping-pong-style kernels); on TPU it lowers to Mosaic. Legacy Triton GPU backend exists but
  is best-effort only.
- **AMD/CDNA support:** **None for the Mosaic GPU backend (Hopper+ NVIDIA only).** No CDNA path. (TPU
  is the other first-class target.)
- **Maturity:** Experimental API but heavily used in production at Google (Splash Attention, MaxText).
- **Best for:** TPU kernels first-class; NVIDIA Hopper/Blackwell attention/GEMM in the JAX ecosystem.
- **Strengths/limits:** Excellent memory-hierarchy model, `interpret=True` CPU debugging, AutoGrad-
  adjacent. Limit: not an AMD option; experimental; needs a hand-written bwd for trainable kernels.
- **License:** Apache-2.0.
- **What we borrow:** **Concepts.** The explicit **memory-space typing** (GMEM/SMEM/registers/TMEM) is a
  clean way to document the AMD hierarchy (HBM/LDS/VGPR). `interpret=True` is a model for a
  cheap-correctness-check stage in GEAK's evaluator (run logic on CPU before profiling on GPU).

## Hidet (+ Hidet Script)

URL: https://github.com/hidet-org/hidet · https://pytorch.org/blog/introducing-hidet/

- **Programming model:** Python DL compiler (CentML) with **Hidet Script**, a Python-embedded DSL for
  tensor programs that can express C++/CUDA-level optimizations; `torch.compile` backend. Notable for a
  **GPU-centric, drastically-reduced autotuning search space** (~10^6 → few hundred candidates vs
  TVM/Ansor).
- **AMD/CDNA support:** **None found** — NVIDIA CUDA-only codegen and tensor-core usage. (Uncertain
  whether any AMD branch exists; not in mainline.)
- **Maturity:** Mature for NV inference; the base that **Hexcute** and **Tilus** extend.
- **Best for (on NV):** Fused inference operators, schedule autotuning.
- **License:** Apache-2.0.
- **What we borrow:** The **search-space pruning philosophy** — a GPU-aware problem formulation that
  cuts candidates by ~4 orders of magnitude is *directly* applicable to GEAK's autotuning/evolutionary
  loops (don't search 10^6 configs; encode HW priors to search hundreds).

## Hexcute (+ Tilus)

URL: https://arxiv.org/abs/2504.16214 · https://github.com/NVIDIA/tilus

- **Programming model:** Python tile DSL extending **Hidet** with tile-level primitives; exposes shared
  memory + registers; **treats per-thread data distribution as part of the tensor type** (thread-value
  layouts) and **automatically synthesizes layouts + task mappings** via a type-inference / constraint-
  programming algorithm. Tilus (NVIDIA) is a sibling that adopted Hexcute's auto-layout idea.
- **AMD/CDNA support:** **None** (CUDA-targeted; integrated into vLLM for NVIDIA). No CDNA path.
- **Maturity:** Research (arXiv Apr 2025) but vLLM-integrated; Tilus actively developed (Blackwell/
  Hopper). Strong results: 1.7-11.3× over DL compilers for **mixed-dtype** GEMM (key for quantized LLMs).
- **Best for (on NV):** Mixed-input-dtype matmul (e.g. W4A16/FP4) where layout choice dominates.
- **License:** Apache-2.0 (Tilus); Hexcute per paper/repo (uncertain — verify).
- **What we borrow:** **Automatic layout & task-mapping synthesis** is the single most valuable
  *algorithmic* idea for reducing kernel-authoring effort. If GEAK or a future AMD DSL can infer good
  `amd_mfma`/LDS layouts from the operation (instead of the agent guessing), correctness + perf go up.
  Mixed-dtype focus aligns with our quantization deep dives.

## Exo / Exo 2

URL: https://exo-lang.dev/ · https://github.com/exo-lang/exo (Exo 2: ASPLOS 2025)

- **Programming model:** A **user-schedulable language** built on **exocompilation** — target-specific
  instructions, memories, and config state live in *user libraries*, not the compiler. You write naive
  code + a schedule of correctness-preserving rewrites; the compiler guarantees each rewrite is sound.
  **Exo 2** adds **Cursors** (references/inspection/actions over code) so users build **reusable
  scheduling libraries** (~2k LOC amortized over 80+ kernels, order-of-magnitude less scheduling code).
- **AMD/CDNA support:** **None today** — CPU SIMD (AVX-512/AVX2/Neon) + Gemmini accelerator are the
  demonstrated targets; **GPU support is stated future work**. (No CDNA path; uncertain timeline.)
- **Maturity:** Active research (MIT CSAIL), PLDI 2022 + ASPLOS 2025; not a turnkey GPU tool.
- **Best for:** BLAS-class kernels on CPU/custom accelerators where you want verified, reusable
  schedules.
- **License:** MIT.
- **What we borrow:** **The scheduling-as-a-library + correctness-preserving-primitives model.** For
  GEAK, this argues for a *library of trusted transformation moves* (tile, vectorize, pipeline,
  re-layout) that the agent composes — each guaranteed semantics-preserving — rather than free-form
  rewriting that can silently break correctness. The Cursor idea (point at code → apply transform) maps
  to how an agent should target edits.

## IREE / MLIR codegen (+ rocMLIR)

URL: https://iree.dev/guides/deployment-configurations/gpu-rocm/ · https://github.com/ROCm/rocMLIR

- **Programming model:** Not a hand-authoring DSL — an **MLIR end-to-end compiler/runtime**. You import
  a model (PyTorch/ONNX → MLIR), `iree-compile --iree-hal-target-device=hip
  --iree-rocm-target=gfx942`, and it picks codegen pipelines (SIMT vs matrix-core) + tiling/
  vectorization per dispatch. **rocMLIR** is AMD's standalone MLIR kernel generator (GEMM/conv/
  attention/GEMM+GEMM), auto-using MFMA/WMMA; used by MIGraphX.
- **AMD/CDNA support:** **Yes** — HIP HAL driver, gfx942/gfx950 targets; rocMLIR is first-party AMD.
- **Maturity:** IREE mature as a deployment compiler; the *kernel-authoring* story is generated, not
  hand-written (µkernel-on-GPU path still maturing).
- **Best for:** Whole-model deployment + auto-generated kernels; not for hand-crafting one hot kernel.
- **License:** Apache-2.0 (IREE / LLVM); rocMLIR per ROCm.
- **What we borrow:** The **MLIR dispatch/pipeline-selection** model (multiple codegen pipelines per op,
  switch to isolate perf bugs) and rocMLIR as a *generator baseline* for GEMM/attention/conv on AMD.
  Useful as a "what does the compiler already do well" reference so GEAK doesn't re-derive it.

## NVIDIA cuDNN frontend (graph API)

URL: https://github.com/NVIDIA/cudnn-frontend ·
https://docs.nvidia.com/deeplearning/cudnn/backend/latest/developer/graph-api.html

- **Programming model:** **Declarative fusion-graph authoring** — describe a computation as a graph
  (nodes=ops, edges=tensors) in C++/Python; cuDNN's runtime fusion engines generate the kernel(s),
  with autotuning (`cudnnFindPlan`) to pick the best engine. Increasingly ships **OSS fused kernels**
  (SDPA/FlashAttention, grouped-GEMM+GLU/SwiGLU, RMSNorm+SiLU, RoPE, FP8/MXFP8).
- **AMD/CDNA support:** **None** (NVIDIA Hopper/Blackwell). AMD analog is MIOpen + CK fusions.
- **Maturity:** High, production (NV).
- **Best for (on NV):** Attention + GEMM-epilogue fusions without writing kernels by hand.
- **License:** MIT (frontend).
- **What we borrow:** The **graph-of-ops + autotune-over-engines** authoring pattern, and its *fusion
  catalogue* (which op chains are worth fusing: GEMM+epilogue, attention, norm+activation, grouped-GEMM
  +GLU for MoE) — a checklist for which AMD fusions GEAK should target via CK/AITER/Triton.

---

## Comparison table

| DSL / framework | Abstraction level | AMD CDNA3/4 support | Best-for | Maturity | License |
|---|---|---|---|---|---|
| **Triton (AMD backend)** | Tile-level SPMD (compiler owns layout) | ✅ First-class (ROCm 7.x, `amd_mfma`) | Attention, fused elementwise/MoE, portable GEMM | Very high | MIT |
| **Gluon** | Low-level tile (explicit layout/wave/SMEM) | ✅ Real, exposes CDNA4 scaled-MFMA/MXFP4 (experimental) | Last-10-30% GEMM/attn; CDNA4-only instrs | Experimental, in-tree | MIT |
| **TileLang** | 3 levels: compute → GPU-aware → thread | ✅ Strong, AMD-endorsed (FlashMLA = AITER asm) | FlashMLA/attention, GEMM, fused ops | Mid-high, fast | MIT |
| **HipKittens** | C++ tile primitives (interface/impl split) | ✅ Native (gfx942 + gfx950); AITER backend | BF16/FP8 GEMM, GQA/MHA fwd+bwd, RoPE, LN | New (MLSys'26) | MIT |
| **CK / CK-Tile** | C++ templates, tile distribution | ✅ First-party AMD (the SOTA baseline) | Peak GEMM/MHA/MoE/quant with deep tuning | High, production | MIT |
| **Mojo / MAX** | Python-family language, explicit threads | ✅ GA since Jun 2025 (`api="hip"`) | Portable NV+AMD kernels + serving | Mid-high | Apache-2.0 |
| **CuTe-DSL (CUTLASS 4)** | Layout algebra in Python (atoms/layouts) | ❌ NVIDIA-only | Dense/grouped GEMM, FMHA (NV) | C++ mature, DSL beta | BSD-3 |
| **JAX Pallas + Mosaic GPU** | Tile + explicit memory spaces | ❌ Hopper+ NV / TPU only | TPU kernels; NV attention in JAX | Experimental, prod-used | Apache-2.0 |
| **Hidet (+ Script)** | Python tensor-program DSL + autotune | ❌ NVIDIA-only (no AMD found) | NV fused inference ops | Mature (NV) | Apache-2.0 |
| **Hexcute / Tilus** | Tile DSL w/ auto layout+task-map synth | ❌ NVIDIA-only | Mixed-dtype quantized GEMM (NV) | Research / vLLM-integrated | Apache-2.0 (verify Hexcute) |
| **Exo / Exo 2** | User-schedulable lang, scheduling library | ❌ CPU/Gemmini; GPU = future work | Verified reusable BLAS schedules | Research (ASPLOS'25) | MIT |
| **IREE / rocMLIR** | MLIR compiler (generated, not authored) | ✅ gfx942/gfx950 (HIP HAL, rocMLIR) | Whole-model deploy + auto kernels | Mature compiler | Apache-2.0 |
| **cuDNN frontend** | Declarative fusion graph + autotune | ❌ NVIDIA-only | NV attention + GEMM-epilogue fusion | High (NV) | MIT |

Legend: ✅ real option on MI300X/MI350X today · ❌ no CDNA path (borrow ideas only).

---

## What we borrow

For our deep dives (`perf_knowledge/languages/`) and for the **GEAK / agentic kernel-authoring workflow**:

- **Author in a tile DSL, not raw HIP/asm.** Every successful agent result (AMD's GEAK Triton agents;
  NVIDIA's μCUTLASS+SOL agent, arXiv 2603.29010) shows that emitting *DSL* code instead of low-level
  code shrinks the search space and raises compile-correctness. On AMD, the agent's primary targets
  should be **Triton → Gluon → TileLang/HipKittens**, in increasing control.
- **Encode the AMD-specific scheduling prior from HipKittens.** Default to **8-wave ping-pong /
  4-wave interleave**; do **not** copy NVIDIA producer/consumer **wave specialization** on CDNA3/CDNA4
  (static register allocation wastes producer-wave registers). This is a citable, AMD-only rule.
- **Escalation ladder = abstraction ladder.** Borrow TileLang's L2→L3 and Triton→Gluon model: start
  high-level (auto-scheduled), measure against Speed-of-Light, and only drop to explicit-layout/
  wave-control when below SOL. Don't hand-tune what the compiler already nails.
- **Automatic layout / task-mapping synthesis (Hexcute) + search-space pruning (Hidet).** Prefer
  inferring good `amd_mfma`/LDS layouts over having the agent guess; prune the config space with HW
  priors (Hidet's 10^6→hundreds) before any evolutionary/autotune search.
- **Scheduling-as-a-library with correctness-preserving primitives (Exo 2 Cursors).** Give GEAK a
  fixed set of *trusted transformation moves* (tile, vectorize, k-pack, pipeline, re-layout, fuse
  epilogue) it composes — each semantics-preserving — instead of free-form edits that silently break
  correctness.
- **Layout algebra as the shared mental model (CuTe).** Use CuTe's first-class-layout vocabulary when
  documenting AMD tile→VGPR→MFMA mappings; it's the lingua franca every modern DSL cites.
- **Tunable-knob dictionaries are ready-made search spaces.** CK-Tile (BlockSize, M/N/K-PerBlock,
  M/N-PerXDL, AK1/BK1) and Triton (`matrix_instr_nonkdim`, `waves_per_eu`, `kpack`) define the exact
  dimensions GEAK's optimizer should sweep.
- **Cheap-correctness-then-profile evaluator (Pallas `interpret=True`).** Validate kernel *logic* cheaply
  (CPU/reference) before spending GPU time profiling — mirrors GEAK's cascaded functional→perf evaluator.
- **Fusion catalogue (cuDNN graph API).** A checklist of op-chains worth fusing on AMD via CK/AITER/
  Triton: GEMM+epilogue, attention (SDPA), norm+activation (RMSNorm+SiLU), grouped-GEMM+GLU/SwiGLU for
  MoE, RoPE+SDPA.
- **Speed-of-Light budgeting (NVIDIA μCUTLASS+SOL paper).** Use a roofline SOL estimate to (a)
  deprioritize problems already near peak, (b) budget agent iterations, and (c) flag benchmark-gaming
  kernels — saves 19-43% tokens at ≥95% of speedup. Transferable to AMD with CDNA roofline numbers.

### Uncertainties / to-verify
- HipKittens' "beats *all* AMD baselines on average" is the authors' claim on MI325X/MI355X; verify
  against our own `operators/*/backends/` measurements before treating as SOTA in the registry.
- TileLang vs. asm peak: the HipKittens paper's claim that "neither Triton nor TileLang systematically
  reaches AMD peak" is **contested** by TileLang's own FlashMLA-at-AITER-parity result — treat as
  operator-dependent, not absolute.
- Hexcute license: confirm from repo (Tilus is Apache-2.0; Hexcute unverified).
- Gluon on **gfx942** specifically: AMD's most advanced Gluon tutorials emphasize **gfx950** (CDNA4
  scaled-MFMA/MXFP4); gfx942 Gluon coverage exists but is less documented — verify before recommending
  Gluon as the gfx942 escalation path.
- Hidet/Exo AMD: no CDNA backend found as of 2026-06; both list GPU/AMD as future or absent — recheck
  periodically.

---

## Sources

- HipKittens (arXiv abstract + HTML; CDNA3/CDNA4, wave-specialization finding, ping-pong/interleave, baselines): https://arxiv.org/abs/2511.08083 · https://arxiv.org/html/2511.08083v1
- HipKittens blog (vision, ThunderKittens lineage): https://hazyresearch.stanford.edu/blog/2025-11-09-hk
- HipKittens repo (MIT license, gfx942 `cdna3` branch / gfx950 main, kernel list, AITER backend, MLSys'26): https://github.com/HazyResearch/HipKittens
- ThunderKittens (NVIDIA original): https://arxiv.org/abs/2410.20399
- Triton AMD backend compiler (gfx942 paths, FP8 FNUZ, XF32): https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
- Triton on AMD: optimizing Triton kernels (layouts, `matrix_instr_nonkdim`, mfma_16x16): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-triton-kernel.html
- Triton ships in ROCm 7.0 (v3.3): https://rocm.docs.amd.com/en/docs-7.0.0/about/release-notes.html
- Gluon overview (Triton low-level dialect): https://triton-lang.org/main/gluon/index.html
- Gluon dialect / GluonOps docs: https://triton-lang.org/main/dialects/GluonDialect.html
- Gluon GEMM on CDNA (near-peak; CDNA4 scaled-MFMA / MXFP4 `gl.amd.cdna4.mfma_scaled`): https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html
- Gluon "lower-level alternative to Triton" framing: https://biggo.com/news/202509190133_Gluon_GPU_Programming_Language
- ROCm Iris (experimental Gluon multi-GPU backend, ROCm 7.0+): https://github.com/ROCm/iris
- TileLang repo (MIT, USE_ROCM gfx942/gfx950, FlashMLA-AMD, CuTe-DSL backend Dec 2025): https://github.com/tile-ai/tilelang
- TileLang paper (3 abstraction levels, tile ops): https://arxiv.org/abs/2504.17577
- AMD ROCm blog promoting TileLang vs Triton (FA ~80 lines): https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-tilelang-kernel/
- CuTe-DSL / CUTLASS 4 (Python layout algebra, MLIR→PTX→SASS, fast JIT): https://developer.nvidia.com/blog/achieve-cutlass-c-performance-with-python-apis-using-cute-dsl/
- CUTLASS repo + Python DSL docs (BSD-3): https://github.com/NVIDIA/cutlass · https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html
- CK-Tile conceptual docs (tile distribution, coordinate transforms, MFMA/XDL): https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/index.html
- CK-Tile hands-on GEMM on MI300X (gfx942 build, tunables): https://rocm.blogs.amd.com/software-tools-optimization/building-efficient-gemm-kernels-with-ck-tile-vendo/
- CK repo (moved to ROCm/rocm-libraries; MIT): https://github.com/ROCm/composable_kernel
- Mojo + AMD GA (MI300/MI325, Jun 2025; +53%/+32% throughput): https://www.modular.com/blog/modular-x-amd-unleashing-ai-performance-on-amd-gpus
- Mojo GPU fundamentals (`DeviceContext`, `api="hip"`): https://docs.modular.com/mojo/manual/gpu/fundamentals/
- Mojo HPC kernels paper (MI300 since Jun 2025, BF16 matmul beats hand-tuned): https://arxiv.org/html/2509.21039v1
- Modular Platform repo (Apache-2.0): https://github.com/modular/modular
- JAX Pallas Mosaic GPU (memory spaces, Hopper+ only, warp specialization): https://docs.jax.dev/en/latest/pallas/gpu/index.html · https://docs.jax.dev/en/latest/pallas/gpu/reference.html
- Hidet (NVIDIA-only CUDA codegen, Hidet Script, search-space pruning): https://pytorch.org/blog/introducing-hidet/ · https://github.com/hidet-org/hidet
- Hexcute paper (auto layout + task-mapping synthesis, mixed-dtype, vLLM): https://arxiv.org/abs/2504.16214 · https://arxiv.org/html/2504.16214v1
- Tilus repo (adopts Hexcute auto-layout; Blackwell/Hopper): https://github.com/NVIDIA/tilus
- Exo language (exocompilation, user-schedulable): https://exo-lang.dev/ · https://github.com/exo-lang/exo
- Exo 2 (Cursors, scheduling libraries, ASPLOS 2025; GPU future work): https://www.researchgate.net/publication/388789422_Exo_2_Growing_a_Scheduling_Language · https://techxplore.com/news/2025-03-exo-language-high-code.html
- IREE ROCm/HIP guide (gfx target, HIP HAL): https://iree.dev/guides/deployment-configurations/gpu-rocm/
- rocMLIR (AMD MLIR kernel generator, MFMA/WMMA auto): https://github.com/ROCm/rocMLIR
- cuDNN frontend (graph API, runtime fusion engines, OSS fused kernels, MIT): https://github.com/NVIDIA/cudnn-frontend · https://docs.nvidia.com/deeplearning/cudnn/backend/latest/developer/graph-api.html
- GEAK Triton kernel AI agent (generator/reflector/evaluator/optimizer; correctness/speedup): https://arxiv.org/abs/2507.23194 · https://rocm.blogs.amd.com/software-tools-optimization/triton-kernel-ai/README.html
- GEAK-Triton v2 + GEAK-OpenEvolve (evolutionary search): https://rocm.blogs.amd.com/artificial-intelligence/geak-agents-family/README.html
- GEAK HIP optimization (Voxelization 2.07x, SwiGLU 1.68x): https://rocm.blogs.amd.com/software-tools-optimization/geak-hip-optimizations/README.html
- μCUTLASS DSL + Speed-of-Light agent guidance (DSL beats low-level codegen; SOL budgeting): https://arxiv.org/abs/2603.29010 · https://arxiv.org/html/2603.29010v1
