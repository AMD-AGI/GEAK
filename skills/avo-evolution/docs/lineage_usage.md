# Using the Lineage (P_t)

The lineage is the chain of **committed** kernel versions, each one a verified
improvement over its parent. It lives in `avo_state/lineage.json` and each
version is a git tag `avo-v{N}`. The controller summarizes the recent chain
into your task body.

## How to use it within a step

- **Start from the best.** The worktree is already reset to the best version;
  build your change on top of it, not on the original baseline.
- **Compare, don't guess.** When a direction is unclear, inspect what earlier
  versions changed (`git show avo-v{N}`) and which directions already paid off.
- **Avoid re-treading failures.** `strategy_manager show` plus the lineage tell
  you which directions are already `failed` — do not repeat them.

## What "committed" means

A version is committed only when it is correct AND its verified benchmark
speedup is ≥ the running best (within epsilon). Your intermediate attempts that
regress or fail correctness are recorded in `avo_state/attempts.jsonl` for
diagnosis but are **not** part of the lineage. That is expected — the paper's
40 committed versions came from 500+ internal attempts.

## Backtracking

The supervisor may occasionally reset the worktree to an earlier version if a
recent path looks like a dead end. If you suspect the current best is a local
trap (e.g. it blocks a class of optimizations), say so in your final message so
the supervisor can consider a backtrack.
