---
title: all_to_all_dispatch_combine on aiter — SOTA card
kind: sota_card
operator: all_to_all_dispatch_combine
backend: aiter
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3_fnuz, fp4_e2m1]
regimes: [prefill, decode, both]
status: sota
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/fused_moe.py
  - https://github.com/ROCm/mori/blob/main/docs/MORI-EP-GUIDE.md
  - https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html
---

# all_to_all_dispatch_combine × aiter

## TL;DR
aiter does not implement the EP collective itself — it **consumes** it. `aiter.fused_moe` is the
grouped-GEMM engine on either side of the all-to-all, and `aiter/moe_op/mori_all2all.py`
(`MoriAll2AllManager`) is the **integration seam** that wraps MoRI-EP dispatch/combine for FusedMoE. The
SOTA wide-EP path is **AITER FusedMoE + MoRI-EP**, with shared-expert fusion co-designed across the two. So
"aiter all-to-all" = the FusedMoE prologue/epilogue that the dispatched tokens flow through.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| AITER FusedMoE + MoRI-EP (dispatch/combine) | `aiter/fused_moe.py` + `aiter/moe_op/mori_all2all.py` + MoRI-EP | gfx942/950, bf16/fp8/fp4 | **32.3k in / 12.4k out tok/s/node** (32× MI300X 2P2D, AMD-reported, ROCm 6.3.1, 2025-11); per-collective GB/s → [[backends/mori_rccl/mori_ep.md]] | wide-EP DeepSeek serving |

## Config space / knobs
- The dispatched tokens land in the 3D DeepEP-compatible layout (`packed_recv_*`) that the FusedMoE
  grouped-GEMM consumes — needs MoRI built with `ENABLE_STANDARD_MOE_ADAPT=ON`.
- FusedMoE quant signature (`quant_type`, `q_dtype_a/w`) must match the **dispatch payload dtype** (FP8
  dispatch → FP8 grouped GEMM). `tuned_fmoe.csv` tunes the GEMM ([[backends/aiter/fmoe.md]]).
- shared-expert fusion done MoRI-side in wide-EP (`VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS` incompatible
  with MoRI).

## Numerics / parity
combine reduction + FP8 dispatch quant + grouped-GEMM quant → accuracy-gate (greedy/temp=0). Dispatch dtype
must match FusedMoE input dtype (FNUZ/OCP 2× hazard). See [[operators/all_to_all_dispatch_combine/numerics.md]].

## Integration (rebind seam)
`MoriAll2AllManager` (`aiter/moe_op/mori_all2all.py`); vLLM/SGLang `--all2all-backend mori_low_latency` +
`VLLM_ROCM_USE_AITER_MOE=1`. The collective is MoRI; aiter owns the GEMM the tokens pass through.

## Pitfalls & anti-patterns
- ⚠ Treating aiter as the comm layer — it's the **compute** layer; the collective is MoRI/DeepEP.
- ⚠ dispatch payload dtype vs grouped-GEMM quant mismatch → 2× FP8 error or a CK shape gap.
- ⚠ MoRI build/flag prerequisites (`ENABLE_STANDARD_MOE_ADAPT`, arch pin) apply.

## How to verify
`AITER_LOG_MORE=1` → FusedMoE kernels fire on the dispatched tokens; rocprofv3 → dispatch/combine overlap the
grouped-GEMM; numeric parity vs torch MoE.

## Alternatives / cross-links
[backends/mori.md](mori.md) (the collective) · [backends/hip.md](hip.md) ·
[[backends/aiter/fmoe.md]] · [[operators/fused_moe_grouped_gemm/overview.md]].

## Sources
- On-box: `ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/fused_moe.py` (FusedMoE), `aiter/moe_op/mori_all2all.py` (`MoriAll2AllManager`).
- MoRI-EP integration / 3D layout: https://github.com/ROCm/mori/blob/main/docs/MORI-EP-GUIDE.md
- Wide-EP AITER+MoRI 32-GPU numbers: https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html
