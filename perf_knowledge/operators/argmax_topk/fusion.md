---
title: argmax_topk — fusion
kind: operator_overview
operator: argmax_topk
gens: [gfx942, gfx950]
dtypes: [fp32, bf16, fp16]
regimes: [both]
updated: 2026-06-08
sources:
  - https://github.com/ROCm/aiter
  - https://github.com/triton-lang/triton/issues/1846
  - https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py
---

# argmax_topk — fusion

The lm_head produces a `[batch, vocab]` logits tensor that is **huge** (e.g. `256 × 152k × 2 B ≈ 78 MB`)
and is consumed *immediately* by argmax/top-k. Writing it to HBM just to read it back for the argmax is
the waste fusion kills — fold the reduction into the **logits GEMM epilogue** (greedy) or the **sampling**
chain.

## The patterns

| fusion | example | saves | where |
|---|---|---|---|
| **logits → argmax** (greedy) | lm_head GEMM epilogue emits argmax per row, never materializes full logits | a full `[batch, vocab]` write + read | [`../lm_head_logits/overview.md`](../lm_head_logits/overview.md) |
| **top-k → top-p → sample** | one sampling kernel: top-k, cumsum CDF (top-p), multinomial | re-loading logits/probs between stages | [`../sampling_topk_topp/overview.md`](../sampling_topk_topp/overview.md) |
| **routing logits → top-k experts** | MoE router GEMM epilogue → top-k(2/8) experts + weights | a separate top-k pass | [`../moe_routing_topk/overview.md`](../moe_routing_topk/overview.md) |
| **softmax → top-k** | sample over softmax probs without writing both | a full probs pass | [`../softmax/overview.md`](../softmax/overview.md) |

## How it gets fused
- **Greedy decode (the big win)**: a fused lm_head that computes logits per row in the GEMM accumulator and
  reduces to the argmax **without ever writing the full logits tensor** — for top-1 greedy you only need
  the running max per row. Removes the largest single HBM tensor in the decode loop.
- **aiter (library-fused)**: top-k is inside the **fused MoE routing** op and the **sampling** op on the
  serving path — not a standalone top-k. See [`../../backends/aiter/overview.md`](../../backends/aiter/overview.md).
- **Triton (manual)**: write the argmax/top-k in the same kernel as the producing reduction/softmax; for
  greedy, fuse into the logits Triton kernel (⚠ validate — matmul+argmax fused kernels have hit bugs, #1846).
- **Inductor**: lowers `argmax` to a Triton reduce and fuses surrounding pointwise; `topk` often stays ATen.

## The greedy-decode byte math
Materializing `[256, 152k]` bf16 logits = ~78 MB write + 78 MB read ≈ 0.04 ms just in traffic, *per token*.
Fusing argmax into the lm_head removes both. At high decode rates this is a real latency saver on the
critical path.

## Anti-patterns
- Materializing full logits for greedy decode when only the per-row max/argmax is needed.
- Fusing matmul+argmax without validating (segfault/wrong-result history, #1846).
- Fusing top-k so heavily it spills — top-k carries `k` candidates in regs/LDS; large k may need a split.
- A fusion that changes the reduction order → flips a tie ([numerics.md](numerics.md)); re-validate parity.

## Verify
rocprof: the fused greedy path shows **no** `[batch, vocab]` logits write; the sampling path is one kernel,
not three. Parity: greedy temp=0 token-match after fusion (a fused argmax that flips a tie shows up here).

## Sources
- aiter fused MoE routing / sampling top-k: https://github.com/ROCm/aiter
- matmul+argmax fused kernel hazard: https://github.com/triton-lang/triton/issues/1846
- Inductor argmax lowering / fusion: https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py
