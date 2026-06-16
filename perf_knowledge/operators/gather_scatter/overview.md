---
title: gather_scatter — overview
kind: operator_overview
operator: gather_scatter
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [fp32, bf16, fp16, fp8_e4m3_fnuz, int8]
regimes: [both]
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - https://huggingface.co/blog/yiakwy-xpu-team/efficient-moe-align-sort-design-for-sglang
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/moe_sorting.py
---

# gather_scatter  (`out[i] = in[idx[i]]` / `out[idx[i]] += in[i]`)

## TL;DR
Index-driven data movement: **gather** reads rows at an index list, **scatter(-reduce)** writes/accumulates
rows at an index list. It is the irregular-memory backbone of **embedding lookup / embedding-bag**, the
**MoE permute/unpermute** (token→expert grouping and back), and any index_select/index_add. The dominant
fact on CDNA: the index side is **scattered → uncoalesced → bandwidth-starved**; the lever is to make the
**moved row contiguous and 128-bit vectorized** (tile the hidden dim, `global_load_dwordx4`) so only the
*row pointer* is irregular, not the bytes.

## Math contract
- **gather**: `out[i, :] = in[idx[i], :]`, `idx ∈ [0, R)`, `out:[N, H]`, `in:[R, H]`. dtype preserved.
- **scatter**: `out[idx[i], :] = in[i, :]` (overwrite) or **scatter-reduce** `out[idx[i], :] += in[i, :]`
  (needs atomics when multiple `i` map to one `idx` — the MoE unpermute case: each token is indexed `topk`
  times → `topk` rows reduce back to one).
- **embedding-bag**: gather `+` segment-reduce (sum/mean/max over a bag of indices) in one pass.
- **MoE permute** = gather into expert-contiguous order; **MoE unpermute** = scatter-reduce back with the
  router weight folded in. See [[operators/moe_routing_topk/overview.md]].

## Shape regimes
- **embedding** `[vocab→H]`: one gather per token at prefill; memory-bound, `H` ∈ {4k..8k}.
- **MoE permute** `[num_tokens·topk, H]`: H=7168 (DeepSeek), the gather/scatter that brackets the grouped
  GEMM; **memory-bound** (low arithmetic intensity) vs the compute-bound expert FFN.
- The op is **HBM-bandwidth bound**; ideal ≈ `2·moved_bytes / 5.3 TB/s`, *if* coalesced.

## Where it matters (Amdahl)
On a dense LLM, embedding gather is a small slice. On **MoE serving** the permute/unpermute pair is a real,
recurring memory-bound cost bracketing every MoE layer — community Triton MoE work measures the unpermute
scatter at **~54% of peak BW** (decent for irregular) and SGLang's AMD MoE align/sort at a **~30%
coalescing rate** (a known bottleneck). Fusing gather into the grouped-GEMM prologue / scatter into its
epilogue is where the win is (see fusion.md, [[operators/fused_moe_grouped_gemm/overview.md]]).

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| aiter | 🟢 sota (MoE sort/permute via `moe_sorting`; fused into FusedMoE) | [backends/aiter.md](backends/aiter.md) |
| triton | 🟢 sota (BLOCK_D-tiled gather/scatter, `tl.atomic_add`; Inductor index ops) | [backends/triton.md](backends/triton.md) |
| hip | 🟢 sota (full control: vectorized rows, HW fp atomics, direct-to-LDS gather) | [backends/hip.md](backends/hip.md) |

## Fusion neighbors
gather → **grouped-GEMM prologue** (don't materialize the permuted tensor); scatter-reduce →
**grouped-GEMM epilogue** (fold router-weight multiply + write-back). embedding gather → norm/RoPE.
See [fusion.md](fusion.md), [[operators/fused_moe_grouped_gemm/overview.md]].

## Numerics
gather = byte-exact. **scatter-reduce reduction order is non-deterministic with atomics** → small fp delta;
gate task accuracy, not byte parity. See [numerics.md](numerics.md).

## How to bench
Isolated: `torch.index_select` (gather) / `index_add_` (scatter) vs authored kernel on `[N,H]` with realistic
index distributions (uniform vs skewed/imbalanced); oracle `allclose` (gather exact, scatter within fp). For
MoE: bench the permute/unpermute inside the FusedMoE pipeline, not in isolation.

## Sources
- Coalescing / vectorize / LDS staging guidance (MI300X): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- MoE permute/scatter 54% BW, AMD 30% coalescing rate, BLOCK_D tiling: https://huggingface.co/blog/yiakwy-xpu-team/efficient-moe-align-sort-design-for-sglang · https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/
- aiter `moe_sorting` (token→expert permute): ROCm/aiter@a6bb49937:aiter/ops/moe_sorting.py.
