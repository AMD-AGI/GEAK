---
title: aiter — backend overview (the central kernel engine on AMD)
kind: backend
backend: aiter
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e5m2_fnuz, fp4_e2m1, int8]
regimes: [prefill, decode, training, both]
status: sota
updated: 2026-06-05
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0
  - https://github.com/ROCm/aiter
  - https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html
---

# aiter — backend overview

## TL;DR
**aiter (`ROCm/aiter`, "AI Tensor Engine for ROCm") is the default kernel backend for LLM inference on
AMD Instinct, and the single most important backend in this knowledge base.** It is roughly *cuBLAS +
cuDNN + FlashAttention + TransformerEngine combined* for AMD: one library that owns GEMM, attention
(MHA/MLA), MoE, norm, RoPE, quant, sampling, and RCCL-bypass collectives. Critically, aiter is a
**dispatcher**, not a monolith — for each op it selects the fastest of **hipBLASLt / hand-tuned asm /
skinny HIP / Triton / FlyDSL / CK** from a per-shape config DB. On sglang/vLLM, aiter is the live path,
so it *subsumes* the underlying libraries: to improve a serving GEMM you tune aiter's DB, not hipBLASLt's
override file (which aiter bypasses — see [tuned_gemm.md](tuned_gemm.md)).

When NOT aiter: a novel op with no catalog entry (write Triton/CK), or a gfx942 shape that falls back to
generic Triton because the tuned asm/CK path only exists on gfx950 — verify a tuned path exists for *your*
shape before assuming a speedup.

## On-box version (this knowledge base)
All code citations below are pinned to the on-box checkout:
`ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0` (`v0.1.12.post1-150-ga6bb49937`, fetched 2026-06).
The public README positions aiter identically ("default kernel backend for LLM inference on AMD GPUs").

## Concepts

### 1. aiter is a multi-backend dispatcher
For a given op + (M,N,K) + dtype + arch, aiter picks an implementation ("libtype"). For dense bf16 GEMM
the libtypes are literally enumerated in the dispatch (`aiter/tuned_gemm.py:solMap`):

| libtype | what it is | typical winner for |
|---|---|---|
| `hipblaslt` | Tensile `Cijk_*` kernel by solution index | most mid/large dense shapes |
| `asm` | hand-tuned MFMA assembly (`gemm_a16w16_asm`) | curated hot shapes, bf16→fp32 out |
| `skinny` | small-M HIP kernels (`wvSpltK`, `LLMM1`, `wv_splitk_small`) | decode M≤16, N≤cu_num |
| `triton` | `aiter.ops.triton.gemm.basic.gemm_a16w16` | portable fallback / no-scale |
| `flydsl` | FlyDSL split-K HGEMM (and A4W4 MoE) | gfx950 split-K shapes; see [flydsl_path.md](flydsl_path.md) |
| `torch` | `F.linear` / `torch._scaled_mm` | gfx12 / no-solution fallback |

The selection is driven by the per-shape config DB, not a heuristic at call time. See
[configs_db.md](configs_db.md) for the CSV schema and [tuned_gemm.md](tuned_gemm.md) for the dispatch
key.

### 2. The op catalog (what aiter ships)
- **GEMM**: `tuned_gemm.gemm_a16w16` / `tgemm.mm` (bf16/fp16), `gemm_a8w8` (fp8/int8 + fused dequant),
  `gemm_a4w4` (4-bit weights via FlyDSL/CK), `deepgemm` (fine-grained fp8 block GEMM), batched variants.
- **Attention**: `flash_attn_func` (MHA prefill), paged/decode attention, `mla_decode_fwd` +
  MLA prefill (DeepSeek). See [attn_mla.md](attn_mla.md).
- **MoE**: `fused_moe` (sorting + grouped GEMM + activation + weighted combine fused), shared-expert
  fusion, block-scale fp8, A4W4 FlyDSL. See [fmoe.md](fmoe.md).
- **Norm/RoPE/quant**: `rms_norm`, `layernorm2d_with_add_asm`, `rope_fwd`, fused QK-norm+RoPE+KV-write+quant,
  gated-RMSNorm+fp8-group-quant.
- **Collectives**: `custom_all_reduce`, `quick_all_reduce` (RCCL bypass).

### 3. JIT + AOT build model
Most C++/HIP/asm kernels are compiled on first use into `aiter/jit/` (AOT blobs under `aot/`). First call
pays a one-time compile; later calls hit the cached `.so`. `AITER_LOG_MORE=1` prints build/dispatch.

### 4. Custom-op registration (survives torch.compile)
aiter wraps dispatchers like `gemm_a16w16` in `@torch_compile_guard(gen_fake=...)`
(`aiter/jit/utils/torch_guard.py`). On torch.compile, it registers the op into a `torch.library.Library`
with an inferred schema and a **fake/meta impl** (`aiter_lib._register_fake`), so the op survives Inductor
graph tracing without executing the real kernel. This is the AMD analog of vLLM's
`direct_register_custom_op`. The fake for GEMM (`gen_gemm_a16w16_fake_tensor`) just allocates the output
shape `[*A.shape[:-1], B.shape[0]]`.

## The levers (where you actually intervene)
1. **Per-shape tuned DB** — the only GEMM/MoE lever that engages the live serving path. Capture →
   tune → deploy via env. ([tuned_gemm.md](tuned_gemm.md), [fmoe.md](fmoe.md), [configs_db.md](configs_db.md))
2. **Framework switches** — `VLLM_ROCM_USE_AITER=1` (vLLM master switch, required even when forcing an
   attention backend); SGLang enables aiter by default on ROCm. ([integration.md](integration.md))
3. **Diagnostics** — `AITER_LOG_MORE=1`, `AITER_LOG_TUNED_CONFIG=1` (logs `is tuned on cu_num` per shape),
   `AITER_TUNE_GEMM=1` (capture live shapes).

## Measured impact (on-box validation)
- **+2.23% e2e** from tuning aiter's bf16 GEMM DB, stacked on `--attention-backend triton`
  (ref 1548.9 → cand 1583.5 tok/s, 5-rep non-overlapping, 246 `is tuned on cu_num` hits) @ MI300X gfx942,
  sglang 0.5.11 / aiter@a6bb49937, 2026-06-08 (perf_knowledge run `e2e_Qwen-Qwen3.5-27B_20260607_193315`).
- Vendor-reported (AMD, for context, not measured here): block-scale GEMM up to **2×**, block-scale fused
  MoE up to **3×**, MLA decode up to **17×** vs naive @ MI300X (tested 2025-03; see attn_mla/fmoe cards).

## Pitfalls
- ⚠ Tuning hipBLASLt's `HIPBLASLT_TUNING_OVERRIDE_FILE` does **nothing** on sglang/vLLM — aiter bypasses
  the PyTorch BLAS dispatch and calls `hipb_mm` by solution index directly.
- ⚠ DB lookup key includes `bias` and `cu_num`; a capture/live mismatch → every lookup misses → tuned
  CSV is inert (verified failure). See [tuned_gemm.md](tuned_gemm.md).
- gfx942 coverage gap: newest-model paths (sparse MLA, paged MQA logits) may only exist on gfx950; on
  gfx942 they fall back to Triton (several × slower).

## How to verify aiter is engaged
`grep -c 'is tuned on cu_num' server.log` (with `AITER_LOG_TUNED_CONFIG=1`) > 0, and `AITER_LOG_MORE=1`
to confirm hot shapes hit asm/CK rather than Triton fallback.

## Sub-pages
[tuned_gemm.md](tuned_gemm.md) · [flydsl_path.md](flydsl_path.md) · [fmoe.md](fmoe.md) ·
[attn_mla.md](attn_mla.md) · [configs_db.md](configs_db.md) · [integration.md](integration.md)

## Sources
- On-box source: `/sgl-workspace/aiter` = `ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0`
  (`v0.1.12.post1-150`): `aiter/tuned_gemm.py`, `aiter/jit/core.py`, `aiter/jit/utils/torch_guard.py`,
  `aiter/configs/`.
- aiter as the central engine / default backend: https://github.com/ROCm/aiter (README) ·
  https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html
- Vendor 2×/3×/17× figures: https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html
  (AMD-reported, MI300X, tested 2025-03).
- +2.23% e2e: perf_knowledge validation run `e2e_Qwen-Qwen3.5-27B_20260607_193315`, 2026-06-08.
