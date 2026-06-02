# Variation Step Boundaries

You execute **ONE** variation step. The outer AVO controller decides when the
whole run stops — never you.

## Inputs the controller provides

- A repo worktree already **reset to the current best lineage version**.
- An `## AVO Variation Step Contract` block in the task body with:
  - the lineage best id + its speedup,
  - the active direction (the strategy assigned by the supervisor).
- The usual GEAK context: `COMMANDMENT.md` (the test contract), `profile.json`,
  `baseline_metrics.json`, and a `.optimization_strategies.md` managed by
  `strategy_manager`.

## What you produce

- Patch attempts via `save_and_test` (the controller scans
  `patch_*.patch` / `patch_*_test.txt`).
- `strategy_manager` state updates (mark exploring / failed / successful).
- A final message; submit when you have a verified improvement OR have
  exhausted this direction and documented why.

## Forbidden

- Declaring the entire run complete ("optimization finished").
- Skipping `save_and_test` for "obvious" micro-edits — unverified edits cannot
  be committed by the controller.
- Hand-rolling a benchmark or lowering correctness rigor to make a number look
  better. The commit gate uses the verified benchmark only.

## Step end conditions

- **Verified improvement found** → submit; the controller applies the commit
  gate and tags `avo-v{N}`.
- **Direction exhausted** → mark the strategy `failed` with a one-line reason,
  request the next strategy, then submit so the controller can re-plan.
