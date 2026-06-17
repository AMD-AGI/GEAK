---
id: flydsl_fp8_gemm_playbook
title: "FlyDSL fused fp8 a8w8 blockscale GEMM, capture-safe bare-core bind (down-proj head)"
kind: expert_skill
authors: [zihao]
scope: e2e
match:
  operator: dense_gemm
  arch_class: ['*']
  gens: [gfx942]
  dtypes: [fp8_e4m3_fnuz]
  regimes: [prefill, decode]
  to_backend: flydsl
  profile_signature:
    op_name_regex: "gemm_a8w8_blockscale|down_proj"
    min_pct_gpu: 10.0
expects:
  e2e_delta_min_pct: 1.0
  parity: required
validation:
  status: validated
  last_verified: 2026-06-14
  gpu: gfx942/MI300X
  model: Qwen-Qwen3.5-27B-FP8
  measured: {isolated: "2.432x (down-proj h1)", e2e_pct: "+67.4", parity: "pass (greedy)"}
  artifact: ../../../../examples/e2e_workflow/qwen3.5-27b-fp8_sglang_flydsl-downproj-autointegrate/
role: advisory_prior
supersedes: []
---

## When to use
An fp8 **a8w8 block-scale** model (weight_block_size [128,128]) on MI300X where a dense GEMM head is the
top GPU-time mass and the per-(N,K) Triton config-JSON overlay is already exhausted (config tiling gives
only ~1.04–1.06x). Specifically the **K-heavy / narrow-N down-proj** (e.g. N=5120, K=17408, ~18% GPU on
Qwen3.5-27B-FP8): config tuning is near its ceiling, so the remaining lever is a **code-level authored
core**, not more config JSON.

## Mechanism
The stock blockscale GEMM dequantizes per 128×128 block then runs a bf16 MFMA. For the K-heavy down-proj
the win is **killing the dequant**: author a FlyDSL kernel (aiter's SOTA fp8 GEMM DSL on gfx942, JIT,
`is_flydsl_available()==True`) that folds the block-scale into the operand scale and runs **one fused
full-K low-precision MFMA core**, caching only compact fp8/preshuffled weights. Isolated this is ~2.43x
on the down-proj shape — far beyond the ~1.06x config tile. The catch that blocked earlier attempts: the
authored kernel crashed sglang's CUDA-graph capture (nested-graph / capture-unsafe). The fix is a
**capture-safe BARE-CORE bind** (the `nocgraph` overlay): bind the bare fused core over the live
blockscale path so it runs inside the existing captured graph instead of creating its own.

## Procedure
1. Confirm the head: profile shows fp8 a8w8 blockscale GEMM is the mass; isolate the down-proj (N,K).
   Confirm config tiling is exhausted (wide-N tile BM=128/BN=256 only ~1.04–1.06x prefill).
2. Author a **FlyDSL** fused fp8 a8w8 blockscale core (Tier-C, flydsl FIRST): baseline reuses
   `gemm_a8w8_bpreshuffle_flydsl` / `flydsl_preshuffle_gemm_a8`; the lever is the fused full-K MFMA with
   the block-scale folded into the operand scale. fp32 accumulate.
3. **Capture-safe bare-core bind** (critical): overlay the BARE core over the live blockscale GEMM
   (`aiter.ops.triton.gemm_a8w8_blockscale`), NOT a nested-graph wrapper. The earlier
   `cand_flydsl_downproj` overlay crashed at capture; the accepted `cand_flydsl_downproj_nocgraph`
   binds the bare core so it runs inside the existing captured decode graph.
4. **Memory is a hard constraint**: cache only compact fp8/preshuffled weights (~the model's own fp8
   weight size), never a bf16 expansion across all layers (that forces mem-fraction down → KV-cache
   starves → net e2e regression). Cache weight prep once by `weight.data_ptr()`.
5. Stack with `--attention-backend triton` (+2.24% on its own). e2e-gate the overlay vs the accepted
   stack with the tight Director same-session A/B.

## Knobs & pitfalls
- The h0 up/gate GEMM (wide-N, 57% GPU) is a **reject** for this exact lever: iso 1.10x but decode stays
  generic → e2e −0.26%. The down-proj (h1) is where the fused core pays off (+60.09% on the reference).
- The h2 qkv/o win is nested-graph / capture-unsafe → e2e 0%. Only the bare-core bind ships.
- FlyDSL entrypoints vary by build: `flydsl_preshuffle_gemm_a8` may be absent while
  `gemm_a8w8_bpreshuffle_flydsl` is present — check at author time.

## Do-no-harm notes
- Keep **decode** on the generic config where the wide-N/large tile would tank tiny-M (config-tile
  variant regresses decode ~0.6x). The fused core must not slow the decode path it overlays.
- Capture safety is non-negotiable: any host sync in the steady-state hot path deadlocks graph capture
  → 0 live forwards → e2e correctly rejects it. Verify live-forward count > 0 after overlay.
- When not triggered (non-fp8-blockscale model, or down-proj < min_pct_gpu), the skill is inert.

## Sources
- `examples/e2e_workflow/qwen3.5-27b-fp8_sglang_flydsl-downproj-autointegrate/final_report.md` —
  931.593 → 1559.934 tok/s = **1.674x (+67.4%)**, parity pass; h1 down-proj iso 2.432x, e2e +60.09%.
- Live eval dir: `exp/e2e_Qwen-Qwen3.5-27B-FP8_20260613_195618_371691_18852/` (overlay
  `cand_flydsl_downproj_nocgraph/`).
- Ledger lineage (config-tile precursor): `e2e_workflow/knowledge/gemm_attention_backends.md`
  entries 2026-06-13f / 06-15d (WIDE-N tile ~1.06x; author_plan = flydsl FIRST).
- Related memory: `flydsl-e2e-graph-capture-gap` (the capture crash this skill's bare-core bind resolves).
