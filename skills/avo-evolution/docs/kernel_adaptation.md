# Kernel Variant Adaptation (e.g. MHA → GQA)

The AVO paper showed that optimizations discovered on one kernel configuration
transfer to a related variant with little extra effort (MHA → grouped-query
attention in ~30 minutes). Use this doc when a variation step's direction is to
**adapt the current best kernel to a new variant** rather than to optimize the
same configuration further.

## Approach

1. Start from the **best committed version** (already in the worktree), not the
   baseline — you want to carry over its optimizations.
2. Identify the minimal structural change the variant requires (e.g. shared KV
   heads for GQA: change how K/V are indexed/broadcast, not the whole pipeline).
3. Preserve the optimizations that are variant-agnostic (tiling, pipelining,
   register allocation); only touch what the variant semantics force you to.
4. `save_and_test` against the variant's harness/shapes from `COMMANDMENT.md`.
5. Commit only when the adapted kernel is correct for the new variant AND meets
   or beats the relevant baseline.

## Pitfalls

- Do not re-derive the kernel from scratch — that discards transferred gains.
- Keep the harness contract intact: the variant's correctness reference and
  shapes come from `COMMANDMENT.md`, not from ad-hoc test code.
- If the variant needs different shapes, confirm they are reflected in the
  task's `USER TASK CONTEXT` / commandment before benchmarking.
