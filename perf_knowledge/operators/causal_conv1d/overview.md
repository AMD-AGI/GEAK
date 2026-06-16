---
title: causal_conv1d — overview
kind: operator_overview
operator: causal_conv1d
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [prefill, decode]
updated: 2026-06-05
sources:
  - https://github.com/Dao-AILab/causal-conv1d
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:csrc/kernels/causal_conv1d_update.cu
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/causal_conv1d.py
  - https://docs.vllm.ai/en/stable/api/vllm/model_executor/layers/mamba/ops/causal_conv1d/
---

# causal_conv1d  (`y = silu(depthwise_causal_conv(x, W) + b)`, width ∈ {2,3,4})

## TL;DR
A **short causal depthwise conv1d** (kernel width 2–4) that mixes local token context *before* the
recurrent/SSM state update in Mamba/Mamba2 and linear-attention blocks (Gated DeltaNet — Qwen3-Next,
Kimi Linear). It is **memory-bound and launch-bound**, not compute-bound: there is no MFMA here, so the
whole game is layout (channel-last), a register sliding-window, and keeping the conv state in-cache
during decode. On AMD the live path is **aiter** — a Triton kernel for varlen prefill
(`causal_conv1d_fn`) and either a Triton or a hand-written HIP kernel for single-step decode
(`causal_conv1d_update`). This is the LLM-relevant conv op; [[depthwise_conv]]/[[conv2d]] are vision
(MIOpen) territory.

## Math contract
Per channel `c` (depthwise = `groups == dim`), with left-only ("causal") padding of `width-1`:
```
y[b,c,t] = act( bias[c] + Σ_{k=0..width-1} W[c,k] · x[b,c, t - (width-1) + k] )
```
- **x**: `[batch, dim, seqlen]` (prefill) or `[batch, dim, 1]` (decode); **channel-last** required by the
  fast kernels (`x.stride(dim)==1`). **W**: `[dim, width]`. **bias**: `[dim]` or none. **act**: `None` /
  `silu` (= swish). Equivalent reference: `F.conv1d(x, W.unsqueeze(1), bias, padding=width-1, groups=dim)[..., :seqlen]`
  (aiter `op_tests/.../test_causal_conv1d.py::causal_conv1d_ref`).
- dtype: fp16/bf16/fp32 in; **fp32 accumulate** (the kernel promotes `x`/`W` to `float` for the MAC),
  cast back to input dtype on store.
- **State**: decode carries a `conv_state` `[*, dim, state_len]`, `state_len ≥ width-1` — the last
  `width-1` inputs per channel. It is updated **in place** each step (shift-left, or circular buffer).

## Shape regimes
- **prefill (`causal_conv1d_fn`)**: varlen, x is 2D `[dim, cu_seqlen]` with `query_start_loc` cumulative
  offsets and per-seq `conv_states` initialized from `cache_indices` + `has_initial_state` — the vLLM/
  sglang continuous-batching contract. `dim` ≈ conv channels of the GDN/Mamba mixer (often a few × the
  model hidden, e.g. `2·key_dim + value_dim`); seqlen = prompt length.
- **decode (`causal_conv1d_update`)**: x is `[batch, dim, 1..4]` (seqlen 1, or a few for spec-decode),
  one block per (batch, channel-tile). This is the **dominant** call at serving time (one per token per
  GDN layer) and is pure launch + state I/O — tens of µs regardless of arithmetic.

## Where it matters (Amdahl)
Tiny FLOP count, but in a **hybrid linear-attention model it runs in 75% of layers** (Qwen3-Next /
Kimi Linear use a 3:1 GDN:full-attn ratio). At decode it is a per-layer, per-step launch, so its cost is
**fixed-overhead-dominated**: the win is not FLOPs but (a) not launching a slow generic conv, (b) fusing
it with the QKV split / gating so the GDN mixer is fewer kernels, and (c) keeping `conv_state` resident.
On a dense transformer with no SSM/linear-attention layers this op is **absent** — verify the model uses
Mamba/GDN before optimizing it.

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| triton | 🟢 sota (live varlen prefill + decode on aiter/vLLM) | [backends/triton.md](backends/triton.md) |
| hip | 🟢 sota (decode `causal_conv1d_update`, register sliding-window) | [backends/hip.md](backends/hip.md) |
| aiter | 🟢 sota (the dispatcher/integration that ships both) | [backends/aiter.md](backends/aiter.md) |
| ck | ⚪ na (CK ships grouped-conv1d GEMM instances, but causal short-conv goes through Triton/HIP, not implicit-GEMM — see [[conv2d]]) | — |

## Fusion neighbors
QKV-split (one conv producing q/k/v slices), gated-delta gating, SiLU activation, and the recurrent/SSD
state update downstream → see [fusion.md](fusion.md). Cross-link: [[linear_attention_gated_delta]]
(causal_conv1d is the local-mixing front-end of that block).

## Numerics
fp32-accumulate MAC over ≤4 taps; SiLU in fp32; parity vs `F.conv1d` reference is exact-to-tolerance →
see [numerics.md](numerics.md).

## How to bench
Isolated: aiter `op_tests/triton_tests/test_causal_conv1d.py` (prefill+decode, parity vs `causal_conv1d_ref`).
Micro-latency: time `causal_conv1d_update` at `(batch, dim, width)` decode shapes, median of ≥3 warm reps.
Measured on-box: `~69 µs` median @ batch=128, dim=4096, width=4, bf16, decode (Triton) @ MI300X gfx942,
ROCm 7.2.0, aiter@a6bb49937, 2026-06-05 — see [tuning.md](tuning.md) for the full sweep + command.

## Sources
- Dao-AILab causal-conv1d (the reference op: causal depthwise conv1d, width 2/3/4, silu/swish): https://github.com/Dao-AILab/causal-conv1d
- Mamba `d_conv=4` local conv width; conv before selective scan: https://github.com/state-spaces/mamba
- On-box aiter HIP decode kernel (register sliding-window, fp32 MAC, width 2/3/4, 64-thread wavefront): `ROCm/aiter@a6bb49937:csrc/kernels/causal_conv1d_update.cu`.
- On-box aiter Triton varlen prefill + decode (channel-last, continuous batching, BLOCK_M=8/BLOCK_N=256): `ROCm/aiter@a6bb49937:aiter/ops/triton/causal_conv1d.py`.
- vLLM Mamba causal_conv1d ops (varlen + continuous batching contract): https://docs.vllm.ai/en/stable/api/vllm/model_executor/layers/mamba/ops/causal_conv1d/
- Measured µs: perf_knowledge on-box microbench, MI300X gfx942, ROCm 7.2.0, 2026-06-05.
