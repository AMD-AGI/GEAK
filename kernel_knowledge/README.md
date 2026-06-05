# kernel_knowledge — AMD MI300X Kernel Authoring Knowledge Base

A deep, AMD-only (CDNA3 / gfx942 first; gfx950 / MI355X deltas flagged) reference library whose job is
to let an agent **write super-high-quality kernel code and pick the right backend** for LLM inference
on AMD Instinct. It is the "how do I actually make this op fast on this chip" layer that sits under
the `workflow_e2e_team` optimizer — the System Architect / Op Benchmarker / kernel squad consult it
when choosing a backend, designing an autotune space, or hand-writing a Triton/HIP/CK/asm kernel.

- **39 documents, ~10k lines, 200 unique web sources** (ROCm docs + blogs, GitHub repos/PRs/issues,
  arXiv, AMD whitepapers, Hot Chips). Every doc ends with its own in-context `## Sources`; the union
  is aggregated in [`SOURCES.md`](SOURCES.md).
- Organized along 4 requested dimensions + a hardware foundation: **hardware → languages → libraries →
  operators → optimization strategy**.

## How an agent should use this library
1. **Start from the op.** Going to optimize a kernel? Open the matching `03_operators/*.md` — it gives
   you the algorithm, real kernel logic, the shape-regime split (prefill large-M vs decode skinny-M),
   the backend ladder, and a tuning-knob table.
2. **Pick the backend before writing code.** Cross-check `02_libraries/*` (is there an aiter/hipBLASLt/
   CK op that already wins? cheapest first) before reaching for a hand-written kernel. The doctrine is
   the same as the workflow's `backend_playbook.md`: env/flag swap → library tune → code rewrite.
3. **Then choose the language** (`01_languages/*`) and **mind the hardware** (`00_hardware/*`) — every
   tiling / occupancy / dtype decision traces back to CDNA3 facts (LDS 64 KB, 512 VGPR/EU, MFMA shapes,
   per-XCD L2, FNUZ fp8).
4. **Tune & verify** with `04_optimization/*` — the GEMM/attention autotune recipes, fusion patterns,
   memory/occupancy checklists, quantization accuracy gates, and the rocprof/omniperf → Top-N → Amdahl
   triage loop.

This library is read-only reference. It complements (does not replace) the workflow's own persistent
experience files: `../workflow_e2e_team/knowledge/backend_playbook.md` and `gemm_attention_backends.md`
(those carry *measured* per-run results; this library carries the *general* engineering knowledge).

## Map

### 00_hardware/ — the chip a kernel author must model
| file | what it gives you |
|---|---|
| [mi300x_cdna3_arch.md](00_hardware/mi300x_cdna3_arch.md) | chiplet/XCD topology, CU/wave64 model, clocks, Infinity Cache, SPX/CPX×NPS partitioning, grid/occupancy design rules |
| [memory_hierarchy_occupancy.md](00_hardware/memory_hierarchy_occupancy.md) | VGPR/AGPR, LDS banks & conflict rules, L1/L2(per-XCD)/Infinity Cache, coalescing, occupancy math (worked examples), direct-to-LDS double buffering |
| [matrix_cores_numerics.md](00_hardware/matrix_cores_numerics.md) | full MFMA/WMMA instruction table (shapes×dtypes×throughput), register/lane mapping, fp8/fp6/fp4 & OCP/FNUZ formats, intrinsic usage |

### 01_languages/ — language types (Triton / HIP / CK / asm)
| file | what it gives you |
|---|---|
| [triton_amd.md](01_languages/triton_amd.md) | AMD Triton backend, TTIR→AMDGCN pipeline, `tl.dot`→MFMA, wave64, fp8 fnuz, annotated GEMM + fused softmax |
| [triton_autotune_amd.md](01_languages/triton_autotune_amd.md) | AMD autotune deep dive: `matrix_instr_nonkdim`, `waves_per_eu`, `kpack`, `num_stages`(AMD), `SPLIT_K`, `GROUP_SIZE_M`; config space + baking winners |
| [hip_cpp.md](01_languages/hip_cpp.md) | HIP/C++ on CDNA3, hipcc/amdclang, `__launch_bounds__`, 64-bit `__shfl`/`__ballot`, complete tiled-LDS GEMM |
| [hip_intrinsics_async.md](01_languages/hip_intrinsics_async.md) | `__builtin_amdgcn_mfma_*`, buffer/ds builtins, direct-to-LDS, `s_waitcnt`/`sched_group_barrier`, MFMA microkernel + LDS double-buffer |
| [composable_kernel.md](01_languages/composable_kernel.md) | classic CK programming model, descriptors/coord-transforms, device ops, pipeline v1–v5, instance config & autotune |
| [ck_tile.md](01_languages/ck_tile.md) | ck_tile tile-programming model, GEMM + FMHA skeletons, pipeline/scheduler policies, when to use vs classic CK |
| [asm_mfma_intrinsics.md](01_languages/asm_mfma_intrinsics.md) | CDNA3 ISA for perf, MFMA encoding/register banking, inline asm, scheduling for MFMA/global-load overlap |
| [rocwmma.md](01_languages/rocwmma.md) | rocWMMA fragment API, gfx942 tile/dtype support, complete fragment GEMM, vs raw MFMA vs CK |

### 02_libraries/ — operator libraries (aiter / hipBLASLt / rocBLAS / CK / sglang / vllm / RCCL)
| file | what it gives you |
|---|---|
| [aiter.md](02_libraries/aiter.md) | AMD AITER op catalog, backend dispatch (asm>CK>Triton>HIP), real call examples, when it beats hipBLASLt/CK |
| [hipblaslt.md](02_libraries/hipblaslt.md) | `hipblasLtMatmul`, epilogues, Tensile DB, `HIPBLASLT_TUNING_FILE`, `hipblaslt-bench`, fp8 GEMM + scaling |
| [rocblas_tunableop.md](02_libraries/rocblas_tunableop.md) | rocBLAS + PyTorch TunableOp env-driven autotune, warmup→CSV→ship workflow, parity notes |
| [composable_kernel_lib.md](02_libraries/composable_kernel_lib.md) | CK as a consumed library, ckProfiler instance selection, how vllm/sglang/aiter call CK, fp8/fp4 |
| [sglang_rocm.md](02_libraries/sglang_rocm.md) | sglang ROCm: `--attention-backend`, `SGLANG_USE_AITER*`, quant flags, kernel dispatch paths, HIP-graph |
| [vllm_rocm.md](02_libraries/vllm_rocm.md) | vllm ROCm: custom PagedAttention HIP kernels (`csrc/rocm`), `VLLM_ROCM_USE_AITER*` defaults, fp8/fp4 paths |
| [rccl_comm.md](02_libraries/rccl_comm.md) | RCCL collectives for TP/EP, xGMI topology, `NCCL_*`/`RCCL_*` tuning, MoE all-to-all, custom all-reduce |
| [rocm_ecosystem.md](02_libraries/rocm_ecosystem.md) | MIOpen/rocPRIM/rocSPARSE/runtime overview, ROCm 6.x/7.x version matrix, PyTorch-ROCm op→lib dispatch |

### 03_operators/ — specific operators (GEMM family, MoE, quant, attention family)
| file | what it gives you |
|---|---|
| [gemm.md](03_operators/gemm.md) | dense GEMM: tiled-MFMA, split-K/stream-K, persistent, epilogue fusion, dtype variants, backend ladder, annotated Triton GEMM |
| [grouped_gemm.md](03_operators/grouped_gemm.md) | variable-size grouped GEMM (MoE backbone): segment offsets, group_id mapping, load balancing |
| [batched_gemm.md](03_operators/batched_gemm.md) | uniform batched GEMM: strided/array, CK/hipBLASLt/Triton, use cases |
| [moe.md](03_operators/moe.md) | fused_moe pipeline (gate→topk→permute→expert GEMM→combine), aiter/CK/Triton, fp8/fp4 MoE, tuning knobs |
| [moe_routing_ep.md](03_operators/moe_routing_ep.md) | top-k gating math, permute/combine kernels, expert/data parallel, DeepEP-style all-to-all, load balancing |
| [quantization_fp8.md](03_operators/quantization_fp8.md) | fp8 e4m3/e5m2, scaled MFMA GEMM, scaling granularity, kv-cache fp8, fp8 attention, quant/dequant logic |
| [quantization_fp4_fp6.md](03_operators/quantization_fp4_fp6.md) | OCP MXFP4/FP6 (block-32, E8M0), gfx950 native vs gfx942 dequant, block-scaled GEMM, accuracy |
| [rmsnorm_rope_activation.md](03_operators/rmsnorm_rope_activation.md) | fused RMSNorm(+residual+quant), RoPE(+qk-norm), SiLU/GELU act_and_mul — the fusion-everything memory-bound ops |
| [attention_prefill.md](03_operators/attention_prefill.md) | FA-2/3 prefill, online softmax, MFMA Q/K/V tiling, causal/varlen/GQA, CK/Triton/aiter, tuning knobs |
| [attention_decode_paged.md](03_operators/attention_decode_paged.md) | paged decode attention, KV gather, split-KV/flash-decoding, kv fp8, aiter default, tuning knobs |
| [sparse_attention.md](03_operators/sparse_attention.md) | block-sparse, sliding-window+sink, NSA/DeepSeek sparse, block-skipping kernel logic |
| [linear_attention.md](03_operators/linear_attention.md) | Mamba2 SSD, gated DeltaNet, RWKV, chunked-scan (intra-parallel + inter-recurrent) logic |
| [mla.md](03_operators/mla.md) | DeepSeek MLA math, weight absorption/matrix merging for decode, prefill vs decode, aiter/CK MLA, fp8 |
| [deepseek_v3_v4_attention.md](03_operators/deepseek_v3_v4_attention.md) | V3 MLA+MoE serving, V4/NSA sparse attention, MI300X gaps & PRs, ranked "what to optimize" plan |

### 04_optimization/ — optimization strategy (tuning, algorithms, fusion, memory, quant, profiling)
| file | what it gives you |
|---|---|
| [gemm_tuning.md](04_optimization/gemm_tuning.md) | full GEMM tuning playbook: Tensile/`hipblaslt-bench`, TunableOp, Triton autotune space, ckProfiler, per-shape DB recipe |
| [algorithms.md](04_optimization/algorithms.md) | split-K/stream-K, persistent, software pipelining, online softmax, chunked-prefill, spec-decode, ping-pong/wave-specialization |
| [fusion_patterns.md](04_optimization/fusion_patterns.md) | epilogue fusion, norm+quant, rope+qk-norm, act_and_mul, fused MoE; Amdahl framing + decision table |
| [memory_optimization.md](04_optimization/memory_optimization.md) | coalescing, 128-bit vec loads, LDS bank-conflict avoidance, register pressure, double/triple buffering, L2 reuse |
| [quantization_strategy.md](04_optimization/quantization_strategy.md) | fp16/fp8/fp4 decision ladder, what-to-quantize order, scaling tradeoffs, calibration, accuracy-gate procedure |
| [profiling_roofline.md](04_optimization/profiling_roofline.md) | rocprofv3 / rocprof-compute(omniperf) / roofline / arithmetic intensity, mem-bound vs compute-bound, Amdahl triage |

## Cross-cutting MI300X gotchas (surfaced by the research — read before writing any kernel)
These came up repeatedly and are correctness/perf landmines an agent must internalize:
- **FP8 on gfx942 is FNUZ, not OCP.** CDNA3 matrix cores run `e4m3fnuz`/`e5m2fnuz` (max 240, bias off
  by 1 vs OCP). OCP/NVIDIA fp8 checkpoints must be re-cast on load or values come out **exactly 2×
  off**. gfx950/MI355X adds OCP fp8 + native fp4/fp6. (See `matrix_cores_numerics.md`, both quant docs.)
- **Prefer `mfma_16x16` over `mfma_32x32`** on MI300X (power/clock, not payload) → in Triton set
  `matrix_instr_nonkdim=16`. (`triton_autotune_amd.md`, `gemm.md`.)
- **L2 is per-XCD; there is no global L2.** The 256 MB Infinity Cache is the only device-shared level →
  weight-reuse scheduling (`GROUP_SIZE_M`, XCD-multiple grid sizing) matters. (`memory_hierarchy_occupancy.md`.)
- **FA-3's Hopper tricks (warp-specialization, TMA, wgmma) do NOT port to CDNA3.** The MI300X
  equivalents are `buffer_load` async pipelining + LDS double-buffering + occupancy tuning + 8-wave
  ping-pong. (`attention_prefill.md`, `algorithms.md`.)
- **gfx942 has coverage gaps for the newest models** (DeepSeek V3.2/V4 sparse-MLA paths favor CDNA4 in
  aiter; some fall back to slow Triton) → often the highest-value optimization target.
  (`deepseek_v3_v4_attention.md`, `aiter.md`.)
- **Out-of-box Triton kernels in vllm/sglang are under-tuned** → first move is flip to aiter or supply
  a tuned config JSON before hand-writing. (`moe.md`, `gemm_tuning.md`.)
- **e2e is Amdahl-dominated** — only `pct_gpu_time × achievable_speedup` moves the headline. Tune the
  head kernel (GEMM/attention) and the cheap config knobs first. (`profiling_roofline.md`.)

## Provenance
Built 2026-06-04 by 9 parallel research agents, each mining ROCm docs/blogs, GitHub source/PRs/issues,
arXiv, and AMD whitepapers (2024-2026 material). See [`SOURCES.md`](SOURCES.md) for the full union.
