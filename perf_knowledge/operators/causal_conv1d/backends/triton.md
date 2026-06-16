---
title: causal_conv1d on Triton — SOTA card
kind: sota_card
operator: causal_conv1d
backend: triton
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [prefill, decode]
status: sota
updated: 2026-06-05
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/causal_conv1d.py
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/_triton_kernels/causal_conv1d.py
  - https://docs.vllm.ai/en/stable/api/vllm/model_executor/layers/mamba/ops/causal_conv1d/
---

# causal_conv1d × Triton

## TL;DR
**Triton is the live causal_conv1d backend on AMD serving stacks** for the varlen prefill path
(`causal_conv1d_fn`) and a primary decode path (`causal_conv1d_update`). aiter ships these Triton kernels
and they carry the full vLLM/sglang continuous-batching contract (varlen, `query_start_loc`,
`cache_indices`, pad slots, spec-decode). Use Triton here because the op is memory/launch-bound (no MFMA
to lose to a library) and the varlen + state-cache logic is exactly what Triton's masking + pointer
arithmetic express cleanly. For pure single-step decode the HIP kernel ([hip.md](hip.md)) is an equally
good register-window alternative.

## SOTA implementation(s)
| impl | source | gens/dtypes/shapes | measured perf | when best |
|---|---|---|---|---|
| `causal_conv1d_fn` (varlen prefill) | `ROCm/aiter@a6bb49937:aiter/ops/triton/causal_conv1d.py::causal_conv1d_fn` + `_triton_kernels/causal_conv1d.py::_causal_conv1d_fwd_kernel` | gfx942/950; bf16/fp16/fp32; width 2–4; channel-last `[dim,cu_seqlen]` | launch/BW-bound; scales with total tokens | varlen prefill, continuous batching, has_initial_state |
| `causal_conv1d_update` (decode) | same file `::causal_conv1d_update` + `_causal_conv1d_update_kernel` | gfx942/950; same dtypes/widths; `[batch,dim,1..4]` | **~69 µs** @ batch=128/dim=4096/width=4 bf16, MI300X gfx942, ROCm 7.2.0, aiter@a6bb49937, 2026-06-05 (median/20) | single-step or spec-decode; continuous batching |
| GDN fused conv+QKV-split | `_triton_kernels/gated_delta_rule/{decode,prefill}/causal_conv1d*split_qkv.py` | gfx942/950 | fewer launches (fused split) | Qwen3-Next / GDN serving |

## Config space / knobs
Fixed (not autotuned) in aiter: prefill `BLOCK_M=8, BLOCK_N=256, num_stages=2`; decode `BLOCK_N=256`.
`KERNEL_WIDTH` and `NP2_STATELEN=next_pow2(state_len)` are `constexpr` so the ≤4-tap loop and the state
load fully unroll/mask. Grid: prefill `(batch, ceil(maxseq/BLOCK_M), ceil(dim/BLOCK_N))`, decode
`(batch, ceil(dim/BLOCK_N))`. If re-authoring, the only worthwhile sweep is `BLOCK_N∈{128,256,512}`,
`num_warps∈{2,4}` — there is no MFMA/stage pipeline to tune. See [../tuning.md](../tuning.md).

## Numerics / parity
fp32 MAC + fp32 SiLU, single output rounding; parity vs `F.conv1d(groups=dim)` at `atol≈2e-2/rtol≈1e-2`
(bf16). Same-math, no quant. State indexing (causal left-pad, circular vs linear, pad slots) is the real
correctness risk — see [../numerics.md](../numerics.md).

## Integration (rebind seam)
vLLM call site: `vllm/model_executor/layers/mamba/ops/causal_conv1d.py` (Qwen3-Next `_forward_core` →
`causal_conv1d_update(...)`). aiter exposes `aiter.ops.triton.causal_conv1d.{causal_conv1d_fn,
causal_conv1d_update}`. To swap an authored kernel, rebind at the model's mixer call site or the aiter op,
then e2e-gate on decode tok/s. Continuous-batching args (`query_start_loc`, `cache_indices`,
`has_initial_state`, `pad_slot_id`) **must** be threaded through or correctness breaks.

## Pitfalls & anti-patterns
- ⚠ **`assert num_cache_lines >= batch`** fails when the CUDA-graph capture size > mamba cache size
  (Qwen3-Next/GDN). Fix: reduce `--max-cudagraph-capture-size` (default 512). This is the #1 reported
  causal_conv1d crash.
- ⚠ Non-channel-last input → wrong/slow path; the kernel asserts `is_channel_last` under `validate_data`.
- ⚠ Mixed prefill+decode in one chunked-prefill batch tanks throughput — split into the two kernels
  (vLLM PR #17146).
- Output buffer must be `zeros_like` when pad slots are used (padded rows are skipped, not written).

## How to verify
Isolated: `aiter/op_tests/triton_tests/test_causal_conv1d.py` (parity vs `causal_conv1d_ref`, prefill +
decode). Micro-latency: the `causal_conv1d_update` bench in [../tuning.md](../tuning.md). e2e: run the GDN
model, confirm decode tok/s, read a trace for conv kernel duration/gaps.

## Alternatives / cross-links
[hip.md](hip.md) (register-window decode) · [aiter.md](aiter.md) (the integrating dispatcher) ·
[../overview.md](../overview.md) · language: [`../../../languages/triton_amd/`](../../../languages/triton_amd/) ·
op cross-link: [[linear_attention_gated_delta]].

## Sources
- aiter Triton conv1d (varlen fn + update, BLOCK_M/N, continuous batching): `ROCm/aiter@a6bb49937:aiter/ops/triton/causal_conv1d.py`, `.../_triton_kernels/causal_conv1d.py`.
- vLLM Mamba conv1d ops / Qwen3-Next call site, `num_cache_lines>=batch` assert + cudagraph fix: https://docs.vllm.ai/en/stable/api/vllm/model_executor/layers/mamba/ops/causal_conv1d/ ; https://github.com/vllm-project/vllm/issues/35945
- Split prefill/decode throughput: https://github.com/vllm-project/vllm/pull/17146
- Measured µs: perf_knowledge on-box microbench, MI300X gfx942, ROCm 7.2.0, aiter@a6bb49937, 2026-06-05.
