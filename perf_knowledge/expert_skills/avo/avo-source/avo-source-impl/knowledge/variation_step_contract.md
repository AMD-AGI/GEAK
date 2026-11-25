# Variation Step Boundaries (avo-workflow)

You execute **ONE** variation step. The workflow's deterministic loop decides when
the whole run stops — never you.

## What the workflow provides

- `AVO_REPO` + `BEST_VERSION`: a git repo whose best version you start from (you
  create a private worktree/checkout at `BEST_VERSION`).
- `DIRECTION`: the active strategy assigned by the supervisor (may be empty early
  on — then pick the highest-value untried optimization yourself).
- `COMMANDMENT.md`: the test/benchmark contract (the scoring function `f`).
- `BASELINE_PER_CASE`, `PROFILE_SUMMARY`, `INSIGHTS`, `FAILED_DIRECTIONS`.

## What you produce

- ONE incremental patch `best → candidate` (`git diff $BEST_VERSION`) saved to your
  step dir, plus its hash.
- A structured result (`status`, `strategy`, `patch_file`, `per_case`,
  `rationale`). You do NOT commit — the workflow verifies independently and the
  deterministic commit gate decides.

## Forbidden

- Declaring the entire run complete.
- Skipping the COMMANDMENT benchmark for "obvious" micro-edits — unverified edits
  cannot be committed.
- Hand-rolling a benchmark or lowering correctness rigor. The commit gate uses the
  independently-verified benchmark only.
- Editing the test/harness/reference to pass (verification re-applies your patch on
  a clean base, so it never reaches the score).

## Step end conditions

- **Correct candidate produced** → emit the patch with `status=ok`; the workflow
  verifies + gates it (tags `avo-v{N}` if it beats the best).
- **Direction exhausted** → `status=failed` (set `marked_failed=true`) with a
  one-line reason in `notes`; the supervisor will re-plan.
- **No edit landed** (empty/0-byte patch) → `status=no_change`; you likely edited
  the wrong path — fix the workspace path next time.
- **Harness/env broken** (baseline benchmark yields nothing, unbound env var) →
  `status=infra_failure` so the loop can abort instead of burning budget.
