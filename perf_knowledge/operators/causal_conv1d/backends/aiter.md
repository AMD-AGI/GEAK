---
title: causal_conv1d on aiter — SOTA card
kind: sota_card
operator: causal_conv1d
backend: aiter
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [prefill, decode]
status: sota
updated: 2026-06-05
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/causal_conv1d.py
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/causal_conv1d.py
  - https://github.com/ROCm/aiter
---

# causal_conv1d × aiter

## TL;DR
aiter is **how causal_conv1d actually ships on AMD** — it is not a separate algorithm but the library
that packages, JIT-builds, and exposes both implementations: the **Triton** varlen-prefill + decode
kernels (`aiter.ops.triton.causal_conv1d`) and the **HIP** decode kernel
(`aiter.ops.causal_conv1d.causal_conv1d_update`, `@compile_ops`). On sglang/vLLM serving a GDN/Mamba
model, this aiter path is the live conv. Unlike dense GEMM there is **no per-shape tuned-DB dispatch**
here (the op is too cheap and shape-flat to race library kernels) — aiter's role is integration +
continuous-batching contract + the fused conv+QKV-split GDN kernels, not a tuner.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| Triton `causal_conv1d_fn` / `_update` | `ROCm/aiter@a6bb49937:aiter/ops/triton/causal_conv1d.py` | gfx942/950; bf16/fp16/fp32; width 2–4 | ~69 µs decode @ batch128/dim4096/w4 bf16, MI300X gfx942, ROCm 7.2.0, 2026-06-05 | varlen prefill, decode, GDN serving |
| HIP `causal_conv1d_update` (`@compile_ops`) | `ROCm/aiter@a6bb49937:aiter/ops/causal_conv1d.py` + `csrc/kernels/causal_conv1d_update.cu` | gfx942/950; same | comparable (launch-bound) | single-step decode, circular state |
| GDN fused conv+QKV-split | `.../_triton_kernels/gated_delta_rule/{decode,prefill}/causal_conv1d*split_qkv.py` | gfx942/950 | fewer launches | Qwen3-Next/GDN |

## Config space / knobs
No tuned CSV / `AITER_TUNE_*` lever for this op (contrast dense GEMM). The only knobs are the kernel
launch params (fixed in source — prefill `BLOCK_M=8,BLOCK_N=256,num_stages=2`; decode `BLOCK_N=256`;
HIP `kNThreads=64`) and the **choice of impl** (Triton fn for prefill, Triton-or-HIP update for decode).
Width 2/3/4, `activation∈{None,silu,swish}`. See [../tuning.md](../tuning.md).

## Numerics / parity
fp32 MAC + fp32 SiLU, single output rounding; same-math vs `F.conv1d(groups=dim)`,
`atol≈2e-2/rtol≈1e-2` bf16. No quant path. State indexing is the correctness risk —
[../numerics.md](../numerics.md).

## Integration (rebind seam)
- Triton: `aiter.ops.triton.causal_conv1d.{causal_conv1d_fn, causal_conv1d_update}` (activation str,
  continuous-batching args).
- HIP: `aiter.ops.causal_conv1d.causal_conv1d_update(...)` (`@compile_ops("module_causal_conv1d_update")`,
  explicit `out`, `use_silu` bool). First call JIT-builds into `aiter/jit/`; `AITER_LOG_MORE=1` shows the
  build/dispatch. Live call site on vLLM: `vllm/.../mamba/ops/causal_conv1d.py` (Qwen3-Next GDN mixer).
- Deploy an authored variant by rebinding the model's mixer call or the aiter op; e2e-gate decode tok/s.

## Pitfalls & anti-patterns
- ⚠ `assert num_cache_lines >= batch` (Qwen3-Next/GDN) when cudagraph capture size > mamba cache —
  reduce `--max-cudagraph-capture-size`.
- ⚠ Don't look for an `AITER_TUNE_GEMM`-style DB here — there isn't one; tuning ≠ this op's lever
  (integration + fusion + request-splitting are).
- ⚠ gfx942 coverage: the GDN conv+split kernels are the newest paths; confirm they are present (some GDN
  kernels are gfx950-first and fall back to generic Triton on gfx942 — see [overview of aiter](../../../backends/aiter/overview.md)).
- channel-last required; `zeros_like` out for pad slots; split prefill/decode requests (vLLM PR #17146).

## How to verify
`aiter/op_tests/{,triton_tests/}test_causal_conv1d.py` for parity; micro-latency bench in
[../tuning.md](../tuning.md); `AITER_LOG_MORE=1` to confirm which impl/module engages; e2e decode tok/s
with a trace.

## Alternatives / cross-links
[triton.md](triton.md) · [hip.md](hip.md) · [../overview.md](../overview.md) · backend deep-dive:
[`../../../backends/aiter/overview.md`](../../../backends/aiter/overview.md) ·
op cross-link: [[linear_attention_gated_delta]].

## Sources
- aiter HIP entry (`@compile_ops`) + Triton entry: `ROCm/aiter@a6bb49937:aiter/ops/causal_conv1d.py`, `aiter/ops/triton/causal_conv1d.py`.
- GDN fused conv+QKV-split kernels: `ROCm/aiter@a6bb49937:aiter/ops/triton/_triton_kernels/gated_delta_rule/`.
- aiter as the default AMD inference backend: https://github.com/ROCm/aiter
- cudagraph / `num_cache_lines>=batch` fix: https://github.com/vllm-project/vllm/issues/35945 ; split prefill/decode: https://github.com/vllm-project/vllm/pull/17146
- Measured µs: perf_knowledge on-box microbench, MI300X gfx942, ROCm 7.2.0, aiter@a6bb49937, 2026-06-05.
