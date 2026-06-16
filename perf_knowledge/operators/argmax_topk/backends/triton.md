---
title: argmax_topk on Triton — SOTA card
kind: sota_card
operator: argmax_topk
backend: triton
gens: [gfx942, gfx950]
dtypes: [fp32, bf16, fp16]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - https://triton-lang.org/main/python-api/generated/triton.language.argmax.html
  - https://triton-lang.org/main/python-api/generated/triton.language.max.html
  - https://github.com/triton-lang/triton/issues/6635
  - https://github.com/openai/triton/issues/802
---

# argmax_topk × triton

## TL;DR
`tl.argmax(x, axis, tie_break_left=True)` and `tl.max(x, axis, return_indices=True,
return_indices_tie_break_left=True)` give the index-returning reduce directly; top-k is iterative argmax
(small k) or a hand-written bitonic (larger k). SOTA for authoring and the lm_head/MoE-routing fusion —
**but parity is the hard part**: tie-break, ±Inf (#6635), and `num_warps`-dependent tie results (#802).

## SOTA implementation(s)
| impl | source | gens/dtypes | notes | when best |
|---|---|---|---|---|
| `tl.argmax` row reduce | this card | gfx942/950, fp32 compare | greedy decode top-1 | `[batch, vocab]` → token |
| iterative `tl.argmax`+mask | this card | gfx942/950 | k× reduce; small k | top-k sampling, k≤~16, MoE k=2/8 |
| chunked + 2-call combine | driver + kernel (tuning.md) | gfx942/950 | huge vocab, few rows | greedy decode batch=1 |

```python
@triton.jit
def greedy_argmax(logits_ptr, out_ptr, sr, vocab, BLOCK: tl.constexpr):
    r = tl.program_id(0); cols = tl.arange(0, BLOCK)
    x = tl.load(logits_ptr + r*sr + cols, mask=cols < vocab, other=-float("inf")).to(tl.float32)
    idx = tl.argmax(x, 0)                              # left-most tie, fp32 compare
    tl.store(out_ptr + r, idx)
# grid = (batch,);  BLOCK = next_pow2(vocab)  (chunk + 2-call if vocab too large)
```

## Config space / knobs
- `BLOCK = next_pow2(axis)`; chunk + 2-call combine for huge vocab.
- `num_warps`: 2/4 — **but ties are `num_warps`-sensitive (#802)**: lock it after parity validation.
- `tie_break_left`: set to match the reference/sampler.
- **fp32 compare** (`other=-inf` mask, `.to(tl.float32)`) — stable ordering for bf16 logits.
- top-k: iterative `tl.argmax` + `tl.where(idx==best, -inf, x)` mask, k times; or a bitonic in-kernel for
  larger k.
- `waves_per_eu=3/4` to hide the logits-load latency.

## Numerics / parity
**The hard part.** Left-most tie (match torch/sampler); ±Inf discrepancy (#6635 — sanitize or replicate
order); `num_warps`-dependent ties (#802); fp32 compare. Greedy = a flipped index is a different token. See
[../numerics.md](../numerics.md). ⚠ Validate fused matmul+argmax kernels (#1846).

## Integration (rebind seam)
Greedy: fuse into the lm_head Triton kernel (register opaque so Inductor keeps it). MoE/sampling: top-k is
inside the routing/sampling kernel. Standalone argmax can be a torch custom op.

## Pitfalls & anti-patterns
- Tie-break not matching the reference → divergent greedy tokens.
- ±Inf / NaN logits into `tl.argmax` → #6635 discrepancy.
- Changing `num_warps` after validation → ties flip (#802).
- `BLOCK < vocab` without chunking → can't reduce the full axis.
- Iterative top-k for large k → `O(k·vocab)`, slow; use bitonic/selection.

## How to verify
Index match vs `torch.argmax`/`torch.topk` on random **and** tie/±Inf/all-equal inputs; greedy temp=0
token-match over ≥10 prompts; re-run tie tests after any tuning change.

## Alternatives / cross-links
[hip.md](hip.md) (full order control, bitonic top-k) · [../tuning.md](../tuning.md) ·
[../fusion.md](../fusion.md) · [`../../reduction/backends/triton.md`](../../reduction/backends/triton.md).

## Sources
- `tl.argmax(tie_break_left)`: https://triton-lang.org/main/python-api/generated/triton.language.argmax.html
- `tl.max(return_indices, tie_break_left)`: https://triton-lang.org/main/python-api/generated/triton.language.max.html
- ±Inf top-k discrepancy: https://github.com/triton-lang/triton/issues/6635
- all-equal tie / num_warps dependence: https://github.com/openai/triton/issues/802
