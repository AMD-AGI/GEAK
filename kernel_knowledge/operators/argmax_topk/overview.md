---
title: argmax_topk — overview
kind: operator_overview
operator: argmax_topk
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [fp32, bf16, fp16]
regimes: [decode, prefill, both]
updated: 2026-06-08
sources:
  - https://triton-lang.org/main/python-api/generated/triton.language.argmax.html
  - https://triton-lang.org/main/python-api/generated/triton.language.max.html
  - https://github.com/triton-lang/triton/issues/6635
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
---

# argmax_topk  (`argmax`/`top-k` along an axis — greedy/sampling decode)

## TL;DR
argmax/top-k is an **index-returning reduction**: it carries a `(value, index)` pair through the wave/LDS
combine instead of just a value, so the AMD levers are the same (wave64 shuffle reduce → LDS), with the
extra burden of a **deterministic tie-break** (which index wins when values are equal). The hot use is
**greedy decode** (`argmax` over the vocab `[batch, vocab]`, vocab ~32k–256k → top-1) and **top-k
sampling** (the k largest logits). It's small %GPU-time but on the **token-emission critical path**, and
its **tie-break / NaN / reduction-order behavior is a parity minefield** vs torch (Triton issue #6635).

## Math contract
- **argmax/argmin**: `out[r] = argmax_c x[r, c]` → the **index** of the max (ties → left-most by default).
- **max+index**: value *and* index (`tl.max(..., return_indices=True)`).
- **top-k**: the `k` largest values **and their indices** along the axis (sorted desc).
- **tie-break**: default **left-most** index on equal values (`tie_break_left=True` /
  `return_indices_tie_break_left=True`). **NaN handling differs from torch.**
- dtype: compare in the input dtype (or upcast to fp32 for stability); index is int32/int64.

## Shape regimes
- **Greedy decode**: `[batch, vocab]` → `[batch]`, `vocab ∈ {32k, 128k, 152k, 256k}`, `batch` 1..256.
  One reduction per row over a **huge axis** → wave/LDS reduce + possibly split (few rows × huge vocab).
- **Top-k sampling**: same `[batch, vocab]`, return k (k ∈ {1..1024, often 40–100}) → partial sort /
  iterative argmax / bitonic top-k.
- **MoE routing top-k**: `[tokens, num_experts]` → top-k experts (k=1,2,8); tiny axis, many rows →
  [`../moe_routing_topk/overview.md`](../moe_routing_topk/overview.md).

## Where it matters (Amdahl)
Tiny FLOPs, but the **lm_head→logits→argmax/top-k→sample** chain runs **once per generated token** in
decode — latency-critical, not throughput-critical. The win is **fusing it with the logits/sampling**
(don't write logits to HBM just to argmax them) and **correct, fast tie-break** so it matches torch and the
serving sampler. See [`../sampling_topk_topp/overview.md`](../sampling_topk_topp/overview.md),
[`../lm_head_logits/overview.md`](../lm_head_logits/overview.md).

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| triton | 🟢 sota (`tl.argmax`/`tl.max(return_indices)`; iterative/bitonic top-k) | [backends/triton.md](backends/triton.md) |
| hip | 🟢 sota (wave `(val,idx)` shuffle reduce + LDS; bitonic top-k) | [backends/hip.md](backends/hip.md) |
| aiter | 🟡 (top-k inside fused MoE routing / sampling) | [`../../backends/aiter/overview.md`] |
| pytorch_inductor | 🟡 (lowers argmax to Triton; top-k often stays ATen) | see [`../../backends/pytorch_inductor/overview.md`] |

## Fusion neighbors
argmax fused into the **logits** kernel (greedy decode: skip writing full logits); top-k fused into
**sampling** (top-k → top-p → multinomial); top-k fused into **MoE routing**. See [fusion.md](fusion.md).

## Numerics
**Tie-break direction**, **NaN/Inf** handling, and **reduction order** all affect which index wins and can
diverge from torch (#6635, #802). For greedy decode a flipped argmax = a different token. See
[numerics.md](numerics.md).

## How to bench
Isolated: time argmax/top-k at `[batch, vocab]`; it's bandwidth-bound on the logits read (output tiny);
GB/s = `logits_bytes/time`. Top-k cost grows with k. Parity: index match vs `torch.argmax`/`torch.topk`
including the **tie and ±Inf edge cases** (#6635), and a greedy temp=0 token-match eval.

## Sources
- `tl.argmax(tie_break_left=True)` (left-most index, non-NaN): https://triton-lang.org/main/python-api/generated/triton.language.argmax.html
- `tl.max(return_indices, return_indices_tie_break_left)`: https://triton-lang.org/main/python-api/generated/triton.language.max.html
- top-k accuracy discrepancy with ±Inf (tie / reduction order): https://github.com/triton-lang/triton/issues/6635
- wave64 reduce / 64-bit masks for the (val,idx) combine: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
