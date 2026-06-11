---
title: cumsum_scan — overview
kind: operator_overview
operator: cumsum_scan
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [fp32, bf16, fp16, int32, int64]
regimes: [prefill, decode, both]
updated: 2026-06-08
sources:
  - https://triton-lang.org/main/python-api/generated/triton.language.associative_scan.html
  - https://triton-lang.org/main/python-api/generated/triton.language.cumsum.html
  - https://moderngpu.github.io/scan.html
  - https://srush.github.io/annotated-mamba/hard.html
---

# cumsum_scan  (`out[i] = ⊕_{j≤i} x[j]` — prefix sum / scan)

## TL;DR
Scan is the **generalization of reduction**: instead of one output it produces a running combine at every
position (inclusive/exclusive prefix sum, or a general first-order recurrence `x[t] = a[t]·x[t-1] + b[t]`
via an associative pair-operator). It is **bandwidth-bound with a log-depth dependency chain** — the
parallel algorithm is a tree (`2·⌈log₂N⌉` steps), not a left-to-right loop. On AMD the in-block scan is a
**wave64 cross-lane scan** then an **LDS** carry between waves; long axes are **chunked** into blocks with
a second pass to stitch per-block carries. In LLMs it's small but load-bearing: **MoE expert histograms /
offsets** (cumsum of token counts), **linear/gated-delta attention** and **SSM/Mamba** state recurrences,
and top-p/sampling CDFs. Use `tl.cumsum` for plain prefix-sum, `tl.associative_scan` with a `@triton.jit`
combine for recurrences.

## Math contract
- **Inclusive scan**: `out[i] = ⊕_{j=0..i} x[j]` (last element = full reduction).
- **Exclusive scan**: `out[i] = ⊕_{j=0..i-1} x[j]` (identity at i=0).
- **Reverse scan**: same from the right (`reverse=True`).
- **op**: `+` (cumsum, the common case), `max` (running max), `*`, or a **general associative combine**
  (SSM/EMA pair `(a,b)`: `(a₁,b₁)⊕(a₂,b₂) = (a₁·a₂, a₂·b₁ + b₂)`).
- **Associativity is required** (parallel tree); commutativity is *not* required but Triton's
  `associative_scan` has an operand-order bug for non-commutative ops (see numerics).
- dtype: accumulate fp32 for bf16/fp16; small ints upcast (`tl.cumsum` auto-upcasts <32-bit to avoid
  overflow).

## Shape regimes
- **MoE routing**: cumsum over `[num_experts]` token counts → expert offsets/histogram (tiny axis, many
  calls). The hot use on the serving path. See [`../moe_routing_topk/overview.md`](../moe_routing_topk/overview.md),
  [`../moe_dispatch_combine/overview.md`](../moe_dispatch_combine/overview.md).
- **Linear / gated-delta attention, SSM/Mamba**: scan over the **sequence axis** with a gated recurrence —
  long axis (1k–16k) → chunked, one channel per block. See
  [`../linear_attention_gated_delta/overview.md`](../linear_attention_gated_delta/overview.md),
  [`../causal_conv1d/overview.md`](../causal_conv1d/overview.md).
- **Sampling**: cumsum of sorted probs for top-p (nucleus) cutoff → [`../sampling_topk_topp/overview.md`](../sampling_topk_topp/overview.md).

## Where it matters (Amdahl)
Small %GPU-time, but on the **critical path** of MoE dispatch (every MoE layer) and *is the whole kernel*
for linear-attention/SSM models. For Mamba-class models the scan is the analogue of attention's GEMM — the
dominant op. For dense Transformers it's MoE-offset glue: optimize by **fusing** it into the routing kernel,
not by speeding the scan in isolation.

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| triton | 🟢 sota (`tl.cumsum`/`tl.associative_scan`; SSM/MoE kernels) | [backends/triton.md](backends/triton.md) |
| hip | 🟢 sota (Hillis-Steele / Blelloch, wave-scan + LDS, chunked) | [backends/hip.md](backends/hip.md) |
| pytorch_inductor | 🟡 (lowers `cumsum` to Triton scan; fuses) | see [`../../backends/pytorch_inductor/overview.md`] |
| aiter | 🟡 (scan inside fused MoE routing / linear-attn ops) | [`../../backends/aiter/overview.md`] |

## Fusion neighbors
Cumsum-of-counts fused into **MoE routing/topk**; scan fused into **gated-delta/SSM** recurrence;
cumsum-of-sorted-probs fused into **top-p sampling**. Pre-scan elementwise (gate computation) folds into
the scan's load. See [fusion.md](fusion.md).

## Numerics
fp32 accumulate; tree vs sequential order → LSB differences; ⚠ **Triton `associative_scan` operand-order
bug for non-commutative combines at seq ≥ 128**; mixing `tl.sum`+`tl.cumsum` in one kernel is buggy.
See [numerics.md](numerics.md).

## How to bench
Isolated: time the scan at the exact `[rows, axis]`; GB/s = `(read+write)/time` (scan reads and writes the
full tensor, unlike reduce). For chunked long-axis, report the 3-stage (block-scan / carry-reduce /
block-add) cost. Parity: fp32 atol vs `torch.cumsum` / a reference tree; **explicitly test seq=64 and
seq≥128** for non-commutative combines.

## Sources
- `tl.associative_scan` (combine_fn, axis, reverse; associative+commutative note): https://triton-lang.org/main/python-api/generated/triton.language.associative_scan.html
- `tl.cumsum` (axis, reverse, <32-bit upcast, bf16→fp32): https://triton-lang.org/main/python-api/generated/triton.language.cumsum.html
- scan = generalized reduce, tree depth 2·⌈log₂N⌉, inclusive/exclusive: https://moderngpu.github.io/scan.html
- chunked block-scan + carry-stitch, SSM recurrence as pair-scan: https://srush.github.io/annotated-mamba/hard.html
