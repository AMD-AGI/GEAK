---
title: gather_scatter on aiter — SOTA card
kind: sota_card
operator: gather_scatter
backend: aiter
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3_fnuz, int8, fp4_e2m1]
regimes: [prefill, decode]
status: sota
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/moe_sorting.py
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/fused_moe.py
  - https://huggingface.co/blog/yiakwy-xpu-team/efficient-moe-align-sort-design-for-sglang
---

# gather_scatter × aiter

## TL;DR
aiter has **no standalone gather/scatter op** — instead the MoE permute/unpermute is **fused into
`fused_moe`** via `moe_sorting` (token→expert sort) feeding a two-stage grouped GEMM that **reads through
the sort index** (prologue gather) and **writes the router-weighted combine** in its epilogue
(`MulRoutedWeight1`). This is the SOTA gather/scatter on the AMD MoE serving path precisely because it never
materializes a permuted tensor. For a standalone gather (embedding) author Triton/HIP.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `moe_sorting` + grouped-GEMM consume-index | `ROCm/aiter@a6bb49937:aiter/ops/moe_sorting.py`, `aiter/fused_moe.py` | gfx942/950, bf16/fp8/int8/fp4 | part of up-to-**3×** fused MoE (AMD-reported, MI300X, 2025-03); gather fused, not separately timed | the MoE permute/unpermute on serving |
| (standalone gather) | — | — | **na** — author Triton/HIP | embedding, custom |

## Config space / knobs
- `moe_sorting` `block_size` (default `BLOCK_SIZE_M`) sets the padded per-expert tile granularity; padding =
  `topk_ids.numel() + num_experts*block_size - topk`.
- Quant signature (`quant_type`, `q_dtype_a/w`) selects the grouped-GEMM kernel the gather feeds.
- `doweight_stage1` chooses where the router-weight multiply (the scatter-side fold) lands.
- Tuned per shape in `tuned_fmoe.csv` ([[backends/aiter/configs_db.md]], [[backends/aiter/fmoe.md]]).

## Numerics / parity
combine reduction order (router-weighted) differs from a dense reference → accuracy-gate, not byte parity
([[operators/gather_scatter/numerics.md]], [[operators/moe_dispatch_combine/overview.md]]).

## Integration (rebind seam)
Engaged automatically inside `aiter.fused_moe` when `VLLM_ROCM_USE_AITER_MOE=1` / SGLang AITER MoE. Verify
`fmoe_stage1_*`/`moe_ck2stages_*` kernels fire (not a Triton MoE fallback). There is no separate
gather/scatter env — it's inseparable from the fused MoE.

## Pitfalls & anti-patterns
- ⚠ Expecting a standalone aiter gather op — there isn't one (it's fused).
- ⚠ CK stage-2 shape gaps ("device_gemm does not support this GEMM problem") for odd expert/inter shapes →
  pad/tune ([[backends/aiter/fmoe.md]]).
- ⚠ DB key mismatch (`cu_num`, `token` bucket, quant sig) → tuned MoE row misses, falls to default.

## How to verify
`AITER_LOG_MORE=1` → confirm the sort + grouped-GEMM kernels; compare MoE-model tok/s before/after a tuned
`tuned_fmoe.csv`. rocprofv3 → no materialized permute tensor between the GEMMs.

## Alternatives / cross-links
[backends/triton.md](triton.md) · [backends/hip.md](hip.md) · [[backends/aiter/fmoe.md]] ·
[[operators/fused_moe_grouped_gemm/overview.md]] · [[operators/moe_dispatch_combine/overview.md]].

## Sources
- On-box: `ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0`: `aiter/ops/moe_sorting.py` (sort/permute),
  `aiter/fused_moe.py` (consume-index, MulRoutedWeight epilogue).
- MoE sort/coalescing analysis: https://huggingface.co/blog/yiakwy-xpu-team/efficient-moe-align-sort-design-for-sglang
- 3× fused MoE (AMD-reported): https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html
