---
title: fused_norm_quant — overview
kind: operator_overview
operator: fused_norm_quant
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3, int8, mxfp4]
regimes: [prefill, decode, both]
updated: 2026-06-08
sources:
  - /sgl-workspace/aiter/aiter/ops/rmsnorm.py
  - /sgl-workspace/aiter/aiter/ops/gated_rmsnorm_fp8_group_quant.py
  - /sgl-workspace/aiter/aiter/ops/triton/normalization/rmsnorm.py
  - https://github.com/sgl-project/sglang/issues/18466
---

# fused_norm_quant  (`y_q, scale = quant(rmsnorm(x))` in one kernel)

## TL;DR
Fuses **(fused-add) RMSNorm/LayerNorm + dynamic fp8/int8 quant** so the norm output is written **already
quantized** — the downstream GEMM (qkv/up-gate/down) reads ½ bytes (fp8) or ¼ (fp4) instead of bf16. Since
norm is bandwidth-bound and quant is "free" while `y` is in fp32 registers, the fusion removes a whole
quant pass AND halves the GEMM input traffic. This is the **[[rmsnorm]]/[[layernorm]] + [[quant_dequant_fp8]]/
[[quant_int8]] seam** — read those first.

## Math contract
`y = norm(x)` (rmsnorm or layernorm, optionally with residual-add), then quantize:
- **per-token dynamic fp8**: `scale[m] = max(|y[m,:]|)/fp8_max`, `y_q = round(y/scale)` (fp8 fnuz on gfx942).
- **group fp8** (`gated_rmsnorm_fp8_group_quant`): scale per group of 128 (head_dim=128, group=128).
- **smoothquant int8**: per-channel smoothing scale × per-token scale.
- **mxfp4**: block-32 e8m0 scale (gfx950).
Outputs: `y_q` (fp8/int8/fp4), `scale` (fp32 or e8m0). `residual_out` (bf16) if fused-add. Norm math in
fp32; scale in fp32; quant rounding RNE.

## Shape regimes
Same as the underlying norm: `M=tokens` prefill / `M=batch` decode, `N∈{4096,5120,8192}`. The quant adds a
per-row (or per-group) max-reduction over the already-computed `y` — cheap, in-register.

## Where it matters (Amdahl)
The norm is 1–4% GPU time; the fusion's real value is the **downstream GEMM** reading quantized input
(the GEMM is the Amdahl head). SGLang reports **1–6% e2e latency / 1–2% throughput** from RMSNorm+FP8
dynamic-quant fusion on Qwen3 MI300X (#18466).

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| aiter | 🟢 sota (CK/asm + HIP group-quant + flydsl) | [backends/aiter.md](backends/aiter.md) |
| triton | 🟢 sota (`_quant_rms_norm_kernel` family) | [backends/triton.md](backends/triton.md) |
| hip | 🟢 sota (`gated_rmsnorm_fp8_group_quant` HIP) | [backends/hip.md](backends/hip.md) |

## Fusion neighbors
This op fuses [[rmsnorm]]/[[fused_add_rmsnorm]]/[[layernorm]] with [[quant_dequant_fp8]]/[[quant_int8]]/
[[quant_fp4_mxfp]]; further upstream the [[act_and_mul_silu_gelu]] output quant is the sibling on the MLP
down-proj side; the QK-norm+RoPE+quant variant (`fused_qk_rmsnorm_group_quant`,
`fused_qk_norm_rope_cache_quant`) is the attention-entry form. See [fusion.md](fusion.md).

## Numerics
fp32 norm + fp32 scale; fnuz fp8 on gfx942 (off-by-2× trap); group/per-token scale must match the consumer
GEMM dequant; **task-level accuracy gate**, not byte parity. See [numerics.md](numerics.md).

## How to bench
`op_tests/test_rmsnorm2d.py` with a `_dynamicquant` path; oracle = fp64 `quant(rmsnorm(x))`; e2e A/B
toggling the quant fusion + a small eval (gsm8k).

## Sources
- aiter norm+quant entrypoints (`rmsnorm2d_fwd_with_dynamicquant`, `_with_add_dynamicquant`, `_smoothquant`): `/sgl-workspace/aiter/aiter/ops/rmsnorm.py`.
- gated rmsnorm + fp8 group quant (HIP, head_dim=128, group=128): `/sgl-workspace/aiter/aiter/ops/gated_rmsnorm_fp8_group_quant.py`.
- Triton `_quant_rms_norm_kernel` / `_quant_fused_add_rmsnorm_kernel`: `/sgl-workspace/aiter/aiter/ops/triton/normalization/rmsnorm.py`.
- 1–6% e2e Qwen3 norm+quant: https://github.com/sgl-project/sglang/issues/18466.
