---
title: causal_conv1d — fusion
kind: operator_overview
operator: causal_conv1d
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [prefill, decode]
updated: 2026-06-05
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/_triton_kernels/gated_delta_rule/decode/causal_conv1d_split_qkv.py
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/_triton_kernels/gated_delta_rule/prefill/causal_conv1d_fwd_split_qkv.py
  - https://www.alibabacloud.com/blog/602580
  - https://github.com/Dao-AILab/causal-conv1d
---

# causal_conv1d — fusion

Because the op is **launch-bound**, fusion is *the* optimization: every separate kernel in the GDN/Mamba
mixer is fixed overhead, and the conv is so cheap that folding it into its neighbors removes more wall
time than any internal tweak.

## Where it sits in the Gated DeltaNet / Mamba mixer
```
hidden ─► [in-proj GEMM] ─► (q,k,v,b,a packed) ─► [SiLU·causal_conv1d] ─► split q/k/v
                                                          │
                                                          ▼
                              [gating: β, α decay] ─► [gated delta-rule recurrence / SSD] ─► [out GEMM]
```
The conv mixes local context (width 4) on the packed `2·key_dim + value_dim` channel block, *then* the
result is split into q/k/v for the delta-rule state update. See [[linear_attention_gated_delta]] for the
recurrence it feeds.

## The fusions aiter actually ships (the win)
1. **conv + activation (built-in).** SiLU/swish is a kernel arg (`activation="silu"`, `SILU_ACTIVATION`
   constexpr / `use_silu`), applied in fp32 inside the conv before store. **Never** run conv then a
   separate SiLU kernel — it doubles the launches and the HBM round-trip for a free epilogue.
2. **conv + QKV-split, fused (the important one).** aiter has dedicated kernels in the GDN path that do
   the causal conv1d **and write the split q/k/v slices directly** in one launch:
   - decode: `gated_delta_rule/decode/causal_conv1d_split_qkv.py`
     (`_causal_conv1d_update_split_qkv_kernel`, args `q_ptr,k_ptr,v_ptr, key_dim, value_dim`).
   - prefill: `gated_delta_rule/prefill/causal_conv1d_fwd_split_qkv.py`.
   This removes a separate slice/copy pass over the `[dim, *]` conv output — instead of conv→materialize→
   3× narrow copies, the kernel scatters each channel's output into the right q/k/v buffer as it computes.
   This is the canonical fusion for Qwen3-Next-style GDN serving.

## What fuses *upstream/downstream* (and what doesn't)
- **in-proj GEMM does NOT fuse into the conv.** The GEMM is matrix-core-bound and the conv is memory-
  bound — a merged kernel would spill and lose the GEMM's tuned tiling. Keep the GEMM separate (its
  CShuffle epilogue can at most pre-cast to the conv's dtype).
- **gating (β / α decay sigmoid) is a separate fused kernel** (`fused_gdn_gating_prefill`,
  `fused_sigmoid_gating_recurrent`) sitting between the conv and the recurrence — the conv feeds it but
  they are not one kernel (different access patterns: conv is `[dim,t]` local, gating is per-token scalar).
- **the recurrence / SSD does NOT fuse into the conv.** The delta-rule state update is its own family of
  kernels (chunked prefill: `chunk_delta_h`, `chunk_o`; decode: `fused_recurrent`). The conv is strictly
  the local-mixing front-end. See [[linear_attention_gated_delta]].

## Fusion support matrix
| fusion | aiter Triton | aiter HIP (`update`) | plain `causal_conv1d_fn` |
|---|---|---|---|
| + SiLU/swish epilogue | yes (`SILU_ACTIVATION`) | yes (`use_silu`) | yes (`activation`) |
| + QKV-split (q/k/v out) | **yes** (GDN split-qkv kernels) | **yes** (decode split-qkv) | no (generic conv) |
| + bias | yes (`HAS_BIAS`) | yes (`bias_ptr`) | yes |
| + in-proj GEMM (upstream) | no | no | no |
| + gating / recurrence (downstream) | no (separate fused kernels) | no | no |
| + state write (in-place) | yes (intrinsic) | yes (intrinsic, circular/linear) | yes |

## Where fusion moves e2e
In a 3:1 GDN model the conv runs in 75% of layers, once per decode step — so the GDN block is launched
~`0.75 × n_layers` times per token. Collapsing conv+SiLU+QKV-split from 3 kernels to 1 cuts launches and
HBM traffic in the block where it matters most. The realistic e2e signal is **fewer/shorter kernels in
the GDN mixer trace**, not a FLOP reduction — read it with a trace, gate on decode tok/s.

## Sources
- Fused conv1d + QKV-split kernels (decode + prefill) in the GDN path: `ROCm/aiter@a6bb49937:aiter/ops/triton/_triton_kernels/gated_delta_rule/{decode,prefill}/causal_conv1d*split_qkv.py`.
- GDN gating + recurrence neighbors (separate fused kernels): `ROCm/aiter@a6bb49937:aiter/ops/triton/_triton_kernels/gated_delta_rule/` (`fused_gdn_gating_prefill.py`, `fused_recurrent.py`, `chunk_*.py`).
- SiLU/swish as conv epilogue arg: `ROCm/aiter@a6bb49937:aiter/ops/triton/causal_conv1d.py`, `csrc/kernels/causal_conv1d_update.cu`.
- GDN = conv front-end before delta-rule recurrence; Qwen3-Next 3:1 ratio: https://www.alibabacloud.com/blog/602580
