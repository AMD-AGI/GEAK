# Self-Pivot Checklist

This is the **agent-side** anti-stall check. It runs *before* the controller's
deterministic supervisor fires, so a good step pivots on its own and the
supervisor only intervenes for real dead ends.

## Trigger a self-pivot when ANY holds

- 3× `save_and_test` with speedup ≤ 1.001 on the same strategy.
- 2× correctness failure with the same root cause.
- The same file region edited 3× with no metric change.
- The profiler shows the bottleneck unchanged after your edit.

## Actions on pivot

1. `strategy_manager mark <index> failed --details "<one-line diagnosis>"`.
2. Write a one-paragraph postmortem (what you tried, why it did not help) so the
   supervisor has signal if it is invoked next.
3. `strategy_manager next` → switch to the next pending strategy.
4. If no pending strategies remain: run `profile_kernel`, `query` the RAG KB for
   an orthogonal approach, and document findings — then submit so the controller
   can re-plan.

## Do NOT

- Repeat an identical edit (the controller detects patch cycles and will force a
  redirect).
- Lower correctness tolerance to make a benchmark pass.
- Stop working or declare the run done. A step ends; the run continues.
