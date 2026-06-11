---
title: cumsum_scan — fusion
kind: operator_overview
operator: cumsum_scan
gens: [gfx942, gfx950]
dtypes: [fp32, bf16, fp16, int32]
regimes: [both]
updated: 2026-06-08
sources:
  - https://github.com/ROCm/aiter
  - https://srush.github.io/annotated-mamba/hard.html
  - https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py
---

# cumsum_scan — fusion

A scan reads and writes the full tensor, so it's worth folding the elementwise that **produces the scan
input** (the gate/weight computation) and the elementwise that **consumes the scan output** (the divide/
gather) into the same kernel — the data is touched once.

## The patterns

| fusion | example | saves | where |
|---|---|---|---|
| **counts → cumsum → offsets** | MoE: histogram tokens → exclusive scan → bucket offsets | a separate offset kernel | [`../moe_routing_topk/overview.md`](../moe_routing_topk/overview.md) |
| **gate calc → scan** (SSM/gated-delta) | compute `a[t],b[t]` then pair-scan in one kernel | re-loading the gates | [`../linear_attention_gated_delta/overview.md`](../linear_attention_gated_delta/overview.md) |
| **scan → divide** (top-p) | cumsum of sorted probs → threshold/mask in one kernel | re-loading the CDF | [`../sampling_topk_topp/overview.md`](../sampling_topk_topp/overview.md) |
| **chunked scan carry** | block-scan + carry-add fused per chunk | an intermediate full write | tuning.md 3-stage |

## How it gets fused
- **Triton (manual)**: compute the pre-scan elementwise (gates), call `tl.associative_scan`/`tl.cumsum`,
  apply the post-scan elementwise, store once — all in one `@triton.jit`. The SSM/gated-delta and
  linear-attention kernels are built this way (scan is the core, gates fused in).
- **aiter (library-fused)**: the cumsum-of-counts is inside the **fused MoE routing** op; the recurrence
  scan is inside the **linear-attention / gated-delta** op — never a standalone scan on the serving path.
  See [`../../backends/aiter/overview.md`](../../backends/aiter/overview.md).
- **Inductor**: lowers `torch.cumsum` to a Triton scan and fuses surrounding pointwise; but the
  MoE/SSM-specific fusions are hand-written kernels, not Inductor's job.

## The MoE-offset worked example
MoE dispatch needs, per expert, the **start offset** of its token bucket = **exclusive cumsum** of the
per-expert token counts. Fusing histogram → exclusive-scan → offsets into the routing kernel avoids
launching a separate `cumsum` (tiny but on every MoE layer's critical path). Watch exclusive-vs-inclusive
([numerics.md](numerics.md)).

## Anti-patterns
- A separate `torch.cumsum` call for MoE offsets when the routing kernel could produce them — extra launch
  on every MoE layer.
- Fusing pre-scan elementwise so heavily the kernel spills (scan already carries the dependency chain in
  registers).
- Fusing across the chunk boundary incorrectly so the carry-in is lost (3-stage stitch must be respected).

## Verify
rocprof: the fused MoE/SSM kernel shows no separate scan kernel; total HBM traffic = one read + one write
of the tensor. For SSM, the scan and gate computation appear as one kernel.

## Sources
- aiter fused MoE routing / linear-attention ops: https://github.com/ROCm/aiter
- gate calc + scan fused, chunked carry: https://srush.github.io/annotated-mamba/hard.html
- Inductor lowers cumsum + fuses pointwise: https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py
