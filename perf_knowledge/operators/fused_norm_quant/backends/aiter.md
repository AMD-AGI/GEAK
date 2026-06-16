---
title: fused_norm_quant on aiter — SOTA card
kind: sota_card
operator: fused_norm_quant
backend: aiter
gens: [gfx942, gfx950]
dtypes: [fp8_e4m3_fnuz, int8, mxfp4]
regimes: [both]
status: sota
updated: 2026-06-09
sources:
  - ROCm/aiter@a6bb4993:aiter/ops/rmsnorm.py
  - ROCm/aiter@a6bb4993:aiter/ops/gated_rmsnorm_fp8_group_quant.py
  - ROCm/aiter@a6bb4993:aiter/ops/fused_qk_rmsnorm_group_quant.py
  - ROCm/aiter@a6bb4993:aiter/ops/fused_qk_norm_rope_cache_quant.py
  - ROCm/aiter@a6bb4993:aiter/ops/triton/_triton_kernels/normalization/rmsnorm.py
  - https://github.com/sgl-project/sglang/issues/18466
  - https://blog.vllm.ai/2025/02/24/ptpc-fp8-rocm.html
  - https://docs.vllm.ai/en/latest/design/fusions/
  - https://github.com/ROCm/aiter/releases
---

# fused_norm_quant × aiter

## TL;DR
aiter is the live norm+quant path and the SOTA choice. It ships asm/CK/HIP fused entrypoints across the
full matrix: per-token dynamic fp8, per-channel smoothquant int8, group-128 fp8, residual+norm+quant
triples, gated (silu) RMSNorm+fp8, and the **mega-fusions** at the attention entry
(`fused_qk_rmsnorm_group_quant`, `fused_qk_norm_rope_cache_quant`). The point of the fusion: a norm that
feeds an fp8/int8 GEMM would otherwise write bf16 → re-read → quantize → write fp8 (three extra HBM passes
and two launches); the fused kernel computes the norm in fp32, picks the scale, and **writes the quantized
tensor directly**. This is the kernel SGLang cites for **+1–6% e2e** on Qwen3 (label-sourced, #18466).

## SOTA implementation(s)
| impl | source (`repo@commit:path`) | gens / dtypes | measured perf | when it's best |
|---|---|---|---|---|
| `rmsnorm2d_fwd_with_dynamicquant` / `rmsnorm_quant` | `aiter/ops/rmsnorm.py` (`module_rmsnorm_quant`) | gfx942/950, fp8 | part of Qwen3 **+1–6% e2e** (#18466) | norm → per-token fp8 GEMM |
| `rmsnorm2d_fwd_with_add_dynamicquant` / `add_rmsnorm_quant` | same | gfx942/950, fp8 | residual+norm+quant triple | block input → fp8 GEMM |
| `rmsnorm2d_fwd_with_smoothquant` / `_with_add_smoothquant` | same (`module_rmsnorm`) | gfx942/950, int8 | per-channel SmoothQuant (xscale·norm) | int8 GEMM |
| `gated_rmsnorm_fp8_group_quant` (HIP) | `gated_rmsnorm_fp8_group_quant.py` (`module_gated_rmsnorm_quant`) | gfx942/950, fp8 group-128 | `head_dim=128`, `group_size=128` **only** | gated norm → block-scale GEMM |
| `fused_qk_rmsnorm_group_quant` | `fused_qk_rmsnorm_group_quant.py` | gfx942/950, fp8 / fp4x2 | QK-norm + group quant (+ optional residual) | attention entry |
| `fused_qk_norm_rope_cache_quant_shuffle` | `fused_qk_norm_rope_cache_quant.py` | gfx942/950, fp8 KV | QK-norm + RoPE + KV-cache write + quant in one kernel | **mega-fusion**, decode/prefill attn entry |

### What the SOTA kernel actually does (fused-quant output)
The norm output is computed in fp32, the **per-token amax** is taken, the scale `amax/DTYPE_MAX` is written
to the scale tensor, and the value is cast straight to fp8 — never materialized as bf16. The on-box Triton
reference shows the exact quant epilogue used by all the dynamicquant variants:

```python
# ROCm/aiter@a6bb4993:aiter/ops/triton/_triton_kernels/normalization/rmsnorm.py  (_per_token_quant)
scale_out  = row_max / DTYPE_MAX               # row_max = max|rmsnorm(x)| over the token
scale_out  = tl.where(scale_out == 0, 1.0, scale_out)
qx         = x * (1.0 / scale_out)             # x is the fp32 norm output
tl.store(y_scale_ptr + row_idx, scale_out.to(...))   # write per-token scale
return qx                                       # cast to fp8 at the store site
```
In the **blocked** path (`N > 65536/elt`) the kernel can't hold the whole row to find `row_max`, so it
stages the fp32 norm into an `aux` scratch tensor, reduces `row_max` across blocks, then re-reads `aux` and
quantizes — guaranteeing the scale matches the values bit-for-bit. SmoothQuant variants multiply by a
per-channel `xscale` (`rms_norm *= x_scale`) **before** taking amax.

### Dispatch + the mega-fusions
`aiter/ops/rmsnorm.py` routes `with_dynamicquant` to **CK** for `N>8192` (no group/shuffle on CK:
`assert group_size==0`), else the asm `rmsnorm_quant` (which *does* take `group_size` and `shuffle_scale`).
`gated_rmsnorm_fp8_group_quant` is a standalone HIP kernel: `norm(x)·silu(z)` per head, flatten, fp8
group-128. `fused_qk_norm_rope_cache_quant_shuffle` is the deepest fusion — QK-RMSNorm + RoPE +
paged-KV-cache write + fp8 KV quant in a single launch (its `fused_qk_rmsnorm` helper falls back to plain
`rmsnorm2d_fwd` when `M ≥ 16384`).

## Config space / knobs
| knob | where | values | effect |
|---|---|---|---|
| fusion entrypoint | call site | dynamicquant / smoothquant / group / gated / qk-rope | **the entrypoint IS the config**; pick the one matching the consumer GEMM |
| quant dtype | out tensor dtype | fp8 (fnuz gfx942), int8, fp4x2/mxfp4 (gfx950) | output dialect |
| `group_size` | arg | 0 (per-token) / 128 (group) | scale granularity; **must match the GEMM's dequant** |
| `shuffle_scale` / `transpose_scale` | arg | False / True | scale memory layout for the consumer GEMM (block-scaled FP8) |
| `scale_ub` / `clamp_out` | arg | None / tensor | clamp the dynamic scale upper bound (`CLAMP_MAX`) |
| `use_model_sensitive_rmsnorm` / `N>8192` | arg / data | — | CK tier (group/shuffle **not** supported on CK) vs asm |
| `gemma_norm` | arg (qk) | False/True | `(1+γ)` Gemma-style weight in the QK fusion |
| JIT warm | runtime | — | compiles `module_rmsnorm_quant` / `module_gated_rmsnorm_quant` / `module_fused_qk_*` |

## Numerics / parity
- **fp32 norm + fp32 amax + fp32 scale**, then a single cast to the quant dtype (RNE) — the scale is exact
  for the values written (one-pass amax in single-pass; `aux`-staged amax in blocked, never a re-derived
  approximation). No Welford (RMSNorm subtracts no mean; the only fp32 reduction is `Σx²`).
- **fnuz fp8 on gfx942** (off-by-2× trap if treated as OCP e4m3); group size must equal the GEMM dequant
  granularity or you silently de-scale wrong blocks.
- **Gate with a task metric** (gsm8k / lm-eval), **not** allclose — fused norm+quant changes rounding and a
  tiny logit shift is expected. aiter MLA fp8 had a real gsm8k loss precedent (#1455): treat any quant
  fusion swap as a parity event.
- **SmoothQuant**: the per-channel `xscale` is applied to the fp32 norm output before amax, so the int8
  range is balanced across channels — verify `xscale` is the calibrated tensor for the checkpoint.

## Integration (rebind seam)
- **vLLM**: `VLLM_ROCM_USE_AITER=1` (+ `_RMSNORM` / `_LINEAR`); the ROCm `rocm_aiter_fusion` pass stitches
  rmsnorm+quant so the linear layer consumes fp8 directly, and `ActivationFusionPass` adds **+8% throughput**.
  ⚠ Inductor-compiled torch-op quant can now **auto-fuse some patterns**, making the standalone SiLU+quant /
  RMSNorm+quant passes **obsolete except for custom-op cases** (attention, collectives, sub-byte quant) —
  check the compiled graph before wiring a pass. **PTPC-FP8** (per-token-per-channel) is **up to 2.5× vs
  naive** on MI300X; **AITER v0.1.12** ships the fused **gated RMSNorm + group-quant** kernel.
- **SGLang**: `rocm_linear_utils` block-scaled FP8 / per-token quant routes here; the QK-norm+RoPE+cache
  mega-fusion is wired into the Qwen3/Qwen3-VL attention path (#18466).
- **Verify**: `AITER_LOG_MORE=1` shows e.g. `RMSNORM_2D_FWD_DYNAMICQUANT: ...`; `rocprofv3` shows a single
  fused norm-quant kernel **and** the next GEMM reading fp8 (not a stray `cast`/`quant` kernel between them).

## Pitfalls & anti-patterns
- ⚠ FNUZ vs OCP fp8 dialect mismatch on gfx942 → off-by-2×.
- ⚠ `group_size` / scale layout ≠ consumer GEMM → wrong dequant (garbled output, not a crash).
- ⚠ `gated_rmsnorm_fp8_group_quant` supports **only** `head_dim=128` / `group_size=128` — other dims are
  unsupported (assert).
- ⚠ Passing `group_size>0` or `shuffle_scale=True` on the **CK** path (`N>8192`) → `assert` failure (CK
  takes neither; only the asm tier does).
- ⚠ fp4/mxfp4 on gfx942 → no HW (gfx950 only); fp4x2 in the QK fusion also rejects `transpose_scale=True`.
- ⚠ Gating with allclose instead of a task metric → false fail (quant always shifts logits slightly).

## How to verify (bench + oracle)
- **Isolated**: `op_tests/test_rmsnorm2d.py` with `_dynamicquant`; dequantize the fp8 output (×scale) and
  compare to an **fp64** reference norm (rel err within the fp8 floor for that amax).
- **Granularity**: assert the emitted scale shape matches the GEMM's expected layout
  (per-token `(M,)` vs group `(M, N/128)`); flip `transpose_scale` to match.
- **e2e**: gsm8k / lm-eval delta within band + `rocprofv3` single fused kernel + GEMM reads fp8 +
  `AITER_LOG_MORE=1` dispatch.

## Worked example (Qwen3 QKV-proj input, MI300X)
A linear feeding an fp8 GEMM at `N=4096`, decode `M=128`. Unfused: `rms_norm` writes 1.0 MiB bf16 → a
`quant` kernel reads 1.0 MiB, writes 0.5 MiB fp8 + a `(128,)` scale → GEMM reads 0.5 MiB fp8. That's
~2.5 MiB of extra norm/quant traffic and **two** launches. Fused `rmsnorm2d_fwd_with_dynamicquant`: read
1.0 MiB bf16, write 0.5 MiB fp8 + scale, **one** launch — the bf16 intermediate never touches HBM. At
~3.5 TB/s the fused floor is ~0.4 µs vs ~0.7 µs + a second launch unfused. Stacking the residual
(`_with_add_dynamicquant`) folds the post-attn add in too; pushing the whole QK-norm+RoPE+KV-write into
`fused_qk_norm_rope_cache_quant_shuffle` collapses the entire attention entry into one kernel — the
top of the #18466 stack.

## Alternatives / cross-links
[triton.md](triton.md) · [hip.md](hip.md) · [[rmsnorm/backends/aiter]] · [[fused_add_rmsnorm]] ·
[[quant_dequant_fp8]] · [[scaled_quant_gemm]] · [[backends/aiter/fmoe]] ·
[[quantization]] · [[optimization/kernel_fusion_strategy]].

## Sources
- aiter norm+quant entrypoints (dynamicquant/smoothquant/group, CK-vs-asm dispatch, group/shuffle only on
  asm): `ROCm/aiter@a6bb4993:aiter/ops/rmsnorm.py`.
- Gated RMSNorm + fp8 group-128 (head_dim=128 only): `ROCm/aiter@a6bb4993:aiter/ops/gated_rmsnorm_fp8_group_quant.py`.
- QK-norm+group-quant and the QK-norm+RoPE+cache+quant mega-fusion:
  `ROCm/aiter@a6bb4993:aiter/ops/fused_qk_rmsnorm_group_quant.py`,
  `ROCm/aiter@a6bb4993:aiter/ops/fused_qk_norm_rope_cache_quant.py`.
- Fused-quant epilogue (`_per_token_quant`, `aux`-staged amax in blocked):
  `ROCm/aiter@a6bb4993:aiter/ops/triton/_triton_kernels/normalization/rmsnorm.py`.
- Qwen3 norm+quant e2e tracking (label-sourced %): https://github.com/sgl-project/sglang/issues/18466;
  MLA fp8 gsm8k loss precedent: https://github.com/ROCm/aiter/issues/1455.
- PTPC-FP8 up to 2.5× vs naive (MI300X): https://blog.vllm.ai/2025/02/24/ptpc-fp8-rocm.html.
- vLLM Inductor fusion passes (ActivationFusionPass +8%; torch-op quant auto-fuse obsoletes some passes except custom-op): https://docs.vllm.ai/en/latest/design/fusions/.
- AITER v0.1.12 fused gated RMSNorm + group-quant: https://github.com/ROCm/aiter/releases.
