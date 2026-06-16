---
title: Ecosystem landscape — what to borrow
kind: landscape
updated: 2026-06-09
---

# Ecosystem landscape — kernels × backends, and what we borrow

A 2026-06 survey of the kernel/backend ecosystem, run as 6 parallel research sweeps (~2,100 lines,
~200 cited sources). It serves perf_knowledge's two jobs: **(1) write better kernels** and **(2) use the best
existing kernel**. Read this page for the cross-cutting picture; drill into the 6 files for evidence.

| file | scope | the one thing to take |
|---|---|---|
| [`multi_backend_libraries.md`](multi_backend_libraries.md) | FlagGems, ATen dispatch, aiter, FlashInfer, Liger, xFormers, TE, vLLM/SGLang kernel layers, BackendBench, **HF Kernel Hub** | key cells by `(arch,dtype,regime)` → ordered backends; HF Kernel Hub = network-served op registry with first-class `rocm` + `kernelize()` zero-code swap |
| [`authoring_dsls.md`](authoring_dsls.md) | Triton, Gluon, TileLang, HipKittens, CuTe-DSL, CK-Tile, Mojo, Pallas, Hidet, Hexcute, IREE | the AMD authoring ladder **Triton → Gluon → TileLang/HipKittens**; CK = baseline to beat |
| [`ai_kernel_agents.md`](ai_kernel_agents.md) | KernelBench/robust-kbench, TritonBench, KernelLLM, GEAK, Kevin-32B, Sakana, KernelEvolve, AutoTriton/TritonRL | the **reward-hacking-proof correctness gate** is the whole ballgame |
| [`autotuning_frameworks.md`](autotuning_frameworks.md) | Triton @autotune/dejavu, TunableOp, hipBLASLt/TensileLite, ckProfiler, Ansor/MetaSchedule, Kernel Tuner, ML cost models | version-validator headers + tuned-config DB as a first-class registry layer |
| [`amd_sota_2026.md`](amd_sota_2026.md) | current best impl per op on MI300X/MI350X; ROCm blogs, HipKittens, Gluon, MLPerf v6.0 | Gluon hits near-peak at HIP/Triton level — **hipBLASLt is no longer the bar** |
| [`serving_stack_registries.md`](serving_stack_registries.md) | vLLM, SGLang, TensorRT-LLM, TGI/HF-kernels, torch.library, cuDNN/cuBLASLt dispatch | copy vLLM `ir_op_priority` with **variant-qualified** backends + a `dispatch:` seam |

---

## The big picture

Every serious stack has converged on the **same shape**, and perf_knowledge currently has only half of it:

> **a per-`(operator, arch, dtype, regime)` ranked list of backends** (index 0 = default), each entry
> carrying **a dispatch seam** (how code applies it), **a tuned-config pointer** (the per-shape DB), and
> **version-scoped exclusions** (known-bad combos). Selection = filter the list by capability predicates,
> honor a tuned-config if present, fall back with a logged reason.

That's cuBLASLt/cuDNN (ranked heuristic + autotune cache + errata blocklist), vLLM (`ir_op_priority`),
SGLang (`ATTENTION_BACKENDS` registry), TRT-LLM (`Autotuner.choose_one` + cache), and HF Kernel Hub
(specificity-ranked `(backend,mode)→repo`) — all the same idea. **perf_knowledge today is a human catalog**
(status badges + cards); it has no selection key→winner, no dispatch seam, no autotune/errata layer.
Closing that gap is the highest-leverage borrow.

### Two reality checks from the AMD-specific sweep
- **The hardware moved.** On CDNA4, **Gluon** reaches FP16 98.75% / BF8 99.72% / MXFP4 92.41% of peak —
  i.e. near-peak GEMM now exists at *Triton/HIP* level, and AMD's own HIP FP8 GEMM beats hipBLASLt at
  8192³. The "you must drop to asm for peak" assumption is stale; our cards/matrix need a **Gluon column**
  and a **HipKittens backend**, and a FlyDSL/mori/QuickReduce numeric refresh.
- **The CDNA scheduling prior.** HipKittens (arXiv Nov 2025) established that NVIDIA-style **wave
  specialization underperforms on CDNA3/CDNA4** — use **8-wave ping-pong / 4-wave interleave**. This is a
  citable, AMD-specific rule that belongs in `optimization/mfma_scheduling.md` and in GEAK's optimizer prompt.

---

## Top 10 borrowings (prioritized)

**For using existing kernels (the registry/dispatch half):**
1. **Registry v3 = ranked list keyed by `(operator, gen, dtype_class, regime)`**, value = `{winner, alternates}`,
   index-0 = default. (cuBLASLt/cuDNN + vLLM `ir_op_priority`.) Emit it from `_gen_registry.py`.
2. **Variant-qualified backends** — `aiter/asm`, `aiter/ck`, `aiter/triton`, not bare `aiter`. Add a
   `variant:` field per card. (vLLM.) "aiter" alone is ambiguous and mis-routes.
3. **A `dispatch:` seam per card** — `{kind: env|flag|ir_op_priority|import_path, value}` — so the KB is
   *applied by code*, not just read. (vLLM/SGLang/HF-kernels.)
4. **A `tuned_config:` block mirroring the live 9-tuple key** — tool, artifact glob, key fields, gate
   metric+max, deploy env var, validators, coverage. Turns the registry into a tuned-config index.
   (aiter CSV + TunableOp + MetaSchedule OrderedUnionDB: mine > model > default.)
5. **Version-validator header on every tuned artifact** + version-scoped `exclusions[]` — stale CSV
   self-rejects on a ROCm/hipBLASLt/aiter bump; known-bad combos are blocklisted. (TunableOp + cuDNN errata.)
6. **Specificity tie-break** — when entries match, the narrower `gens` set wins (gfx950-specific beats
   all-CDNA); add a `torch_compile: survives|prefer_fused|opaque` flag so we never recommend a hand kernel
   the compiler beats by fusing. (HF Kernel Hub + PyTorch.)

**For writing kernels better (the authoring half):**
7. **Reward-hacking-proof correctness gate** (THE one): ban torch/cuBLAS delegation in candidate kernels,
   use fresh poisoned output buffers, fuzz shapes/layouts/dtypes across configs + forward&backward.
   (Kevin-32B, robust-kbench, TritonRL, Sakana's 10–100×→1.49× collapse.) Harden `op_benchmarker`'s gate.
8. **Author in a tile DSL, escalate by Speed-of-Light** — ladder Triton → Gluon → TileLang/HipKittens,
   drop a rung only when measured < roofline SOL (μCUTLASS+SOL: 19–43% fewer tokens at ≥95% speedup).
9. **Cascaded profiler-driven feedback loop** — cheap functionality test → profile only survivors →
   map rocprofv3/Omniperf metrics to concrete edits; multi-turn (8–10) beats one-shot (~2× success).
10. **Evolutionary/MAP-Elites tier + KB-as-retrieval** — a recombining archive of correct variants beats
    linear refinement past ~3 iters (GEAK-OpenEvolve 3.42–7.02×); feed perf_knowledge cards in as the agent's
    retrieval context (KernelEvolve hit 100% on novel HW exactly this way — validates perf_knowledge's thesis).

---

## Concrete next actions (proposed, not yet done)

- **perf_knowledge registry v3**: extend the card frontmatter schema (`variant`, `dispatch`, `tuned_config`,
  `exclusions`, `torch_compile`) and upgrade `index/_gen_registry.py` to emit a top-level
  `selection[(op,gen,dtype_class,regime)] → {winner, alternates}` block. Backward-compatible.
- **perf_knowledge content refresh** (from `amd_sota_2026.md`): add **Gluon** language + matrix column, add a
  **HipKittens** backend column, refresh FlyDSL/mori/QuickReduce numbers, add the CDNA ping-pong prior to
  `optimization/mfma_scheduling.md`.
- **e2e_workflow**: (a) repoint the knowledge base from `perf_knowledge` → `perf_knowledge` and fix the
  stale `02_libraries/03_operators/04_optimization` paths; (b) have `system_architect`/`op_benchmarker`
  query the v3 `selection` block; (c) harden the correctness gate (item 7); (d) add the SOL-gated DSL
  escalation ladder (item 8) to the single-kernel ladder.
- **Distribution (longer-term)**: mirror the HF Kernel Hub model — a `kernelize()`-style in-place forward
  swap so a model adopts perf_knowledge-recommended kernels with zero modeling-code change; gate each cell on a
  BackendBench/OpInfo correctness check before it earns `status: sota`.

## Sources
- The six landscape files in this directory, each with its own per-claim `## Sources` (~200 unique URLs total).
- Cross-checked against on-box `ROCm/aiter@a6bb4993` and the existing `index/sota_registry.yaml` (schema v2).
