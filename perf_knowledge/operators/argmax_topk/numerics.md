---
title: argmax_topk — numerics & parity (the tie-break minefield)
kind: operator_overview
operator: argmax_topk
gens: [gfx942, gfx950]
dtypes: [fp32, bf16, fp16]
regimes: [both]
updated: 2026-06-08
sources:
  - https://triton-lang.org/main/python-api/generated/triton.language.argmax.html
  - https://github.com/triton-lang/triton/issues/6635
  - https://github.com/openai/triton/issues/802
  - https://github.com/triton-lang/triton/issues/1846
---

# argmax_topk — numerics & parity

argmax/top-k returns **indices**, so the failure mode isn't a rounded value — it's the **wrong index**,
which in greedy decode is a **different token**. Three parity hazards: tie-break direction, NaN/Inf, and
reduction-order sensitivity.

## 1. Tie-break direction (must match torch)
Triton default: **left-most** index on equal values (`tie_break_left=True` /
`return_indices_tie_break_left=True`, ignoring NaN). `torch.argmax`'s tie behavior is **not guaranteed** by
docs (implementation-defined; often first index on CPU, can differ on CUDA). For **bit-exact parity** you
must (a) know which index the reference returns and (b) set the Triton flag to match — and accept that the
serving sampler's own tie convention is the real target. bf16 logits have **many ties** (low mantissa) →
this is not a rare edge case at the lm_head.

## 2. NaN / ±Inf (issue #6635)
Triton issue #6635: a top-k built on `tl.argmax` **excluded indices that a torch reference included** when
the input contained multiple `float("Inf")` — a tie/non-finite reduction-order discrepancy. Lessons:
- Sanitize logits (no raw NaN/Inf into argmax) or replicate the reference's exact NaN/Inf order.
- `tl.argmax` returns the left-most **non-NaN** index; AMD `v_max_f32` returns the non-NaN operand — so a
  NaN logit silently loses, which may differ from a torch path that propagates NaN.
- Always test the **±Inf and all-equal** edge cases (the #6635 / #802 repros), not just random logits.

## 3. Reduction-order sensitivity (issue #802)
Triton issue #802: argmax gave a **wrong result when all elements are equal** and it depended on
`num_warps` (wrong with `num_warps=1`, correct with `2`). The parallel reduce order resolves ties
differently than a sequential scan. Implication: a tuning change (warps/block) can **flip** a tie result.
Lock the launch config once parity is validated; re-validate ties after any tuning change.

## 4. fp32 compare for bf16/fp16
Compare in fp32 (upcast) for a stable, well-defined ordering; comparing raw bf16 makes ties even more
common and the result dtype-rounding-dependent. The index is exact (int) — no fp issue on the index
itself, only on which value "wins."

## 5. matmul + argmax interaction (issue #1846)
Issue #1846: a `matmul`-then-`argmax` kernel segfaulted — a reminder to **validate fused logits+argmax
kernels** specifically, not just standalone argmax (the lm_head fusion is exactly this pattern).

## Parity gate
- Index match vs `torch.argmax`/`torch.topk` on random logits **and** the tie / ±Inf / all-equal edge
  cases (#6635, #802).
- Greedy **temp=0 token-match** eval over ≥10 prompts vs the reference decode (a flipped argmax shows up as
  a divergent token).
- Re-run the tie tests after any `num_warps`/block change (#802).
- For top-k feeding sampling: match the **count** at the threshold (ties can yield >k or <k) to torch.

## Sources
- left-most tie-break, non-NaN: https://triton-lang.org/main/python-api/generated/triton.language.argmax.html
- top-k / ±Inf discrepancy vs torch: https://github.com/triton-lang/triton/issues/6635
- all-equal tie wrong, num_warps-dependent: https://github.com/openai/triton/issues/802
- matmul+argmax fused kernel hazard: https://github.com/triton-lang/triton/issues/1846
