# GEAK-EVO architecture

The goal is simple: replace GEAK's search loop with EVO while using the same
task, test commands, time budget, and final result format.

## One shared workflow

```text
Hyperloom or a kernel task
          |
     Set up and profile
          |
     Choose the optimizer
       /           \
     GEAK          EVO
       \           /
      Check and report the result
          |
     Return the result
```

EVO owns how ideas and programs are chosen and developed. Its host uses supplied
model, workspace and execution services. The existing owner controls GPU access,
correctness checks, timing and process cleanup. GEAK and EVO should use those
same services and the same public handoff.

## EVO modules

```text
geak_evo/
  host.py          Connect the search to existing execution services
  command.py       Run an admitted search and final measurements
  dispatch.py      Connect the common optimizer files to the native owner
  result.py        Check the selected source and export its result
  loop.py          Run the next step; stop or resume safely
  planner/         Choose ideas, parents, and work budgets
  engineer/        Continue the edit, test, and repair loop
  archive/         Keep programs, ideas, and their history
  selection/       Choose useful work and the final program
  evaluate/        Use the shared correctness and timing tools
```

The archive stores exact source versions and separate records for ideas and
measurements. Every measurement points to its program, test cases, hardware,
baseline, units, and method. Parent and donor links explain where code came from.
Unfinished ideas and slower correct programs remain distinct from the selected
result. Large traces stay in artifact files, with links from concise records.

A selected program is not yet a measured improvement. The shared workload
check must compare it with the correct baseline before reporting a speedup.
An explicit choice to keep the original is a normal result; a missing or
unresolved choice is an error, not an invitation to invent a winner.

The host grants access to source, profiles, compiler output and hardware guides.
Each deployment must supply and qualify those capabilities. Test results decide
whether a change is kept; performance models help choose experiments.

## What must keep working

- Existing kernel and serving entry points, including Hyperloom's task/result files.
- Correctness and the full required test cases before a performance gain is accepted.
- Saved source, ideas, parent links, measurements, costs, and unfinished work.
- Resume without repeating work that has already finished or resetting its budget.
- One owner for GPU work, bounded model calls, and cleanup when a run stops.
- The original program as a valid outcome when no tested improvement is selected.
- Existing storage formats and hardware/backend support.

The current public command paths stay in place during migration. The first cleanup
does not change experiment versions, model choices, saved records, or live jobs.

## Execution paths

The [shared host](shared-host.md) constructs the search from supplied services.
An explicit model profile fixes the requested model, reasoning effort and route.
The [admitted task command](admitted-command.md) runs that search, checks the
final choice, measures both ORIGINAL and the selected program, and exports the
result. It uses the owner's existing budget and saved operation IDs for recovery.
The caller must leave time for final measurements within the original deadline.

Numeric export currently supports native-v2 tasks with published graph samples
and cache receipts. The [task reader](evaluation-port.md#generated-geak-tasks)
can also read Hyperloom's generated tasks, but their paired timing evidence still
needs a result adapter. Reading a task alone does not make it runnable here.

The kernel lane can dispatch an optimizer command directly with `--task` and
`--out` in a separate clone. Before adopting its commit, the lane checks the
process outcome, matching baseline/candidate measurements, required improvement,
commit ancestry and allowed file changes. Observed work that has not stopped
blocks further subprocess work. This path needs the standalone runtime's
`runProcess`; the Claude Code Workflow tool does not provide it.

The [native EVO adapter](dispatch-adapter.md) connects that dispatcher to the
admitted command. A small patch wires the existing owner entrypoint; it still
needs installation and live qualification. The command uses supplied services;
the bootstrap wheel has
no standalone EVO runner. These source paths have CPU checks but have not yet
completed a model/GPU search together or returned a validated Hyperloom workload
improvement. A deployment must pin and qualify the actual search, engineer,
evaluator and instruction versions it uses.

## Code to edit today

The public serving command stays in `interface/run_e2e.py`. Its result checks
live in `interface/e2e/normalization.py`, and report writing lives in
`interface/e2e/reports.py`.

The three JavaScript workflows are built from the files named in
`scripts/build_workflows.json`. The source folders follow their execution order:
setup, analysis, search, validation, and reporting. Shared helpers live in
`kernel_workflow/src/shared/`. Run `node scripts/build_workflows.mjs` after an
edit, and `node scripts/build_workflows.mjs --check` to check the committed output.

Generated scripts are retained for the existing single-file runner. They are
copies for execution; changes belong in the source folders. This improves
navigation and allows shared code, but it increases the checkout's total size.

## Readability rules

- Name the action: `save_program`, `choose_parent`, `run_engineer`.
- Keep shared logic in one place. Do not replace a copy with several wrapper layers.
- Write comments about requirements and non-obvious behavior. Keep dated incident
  narratives in development notes.
- Keep old record field names readable even if internal names change.
- Measure code size separately from tests and documentation. Retiring a feature
  must be an explicit decision; keep tests for every retained behavior.

Use separate `feat/evo-*` branches for later features and relevant checks before
merging. Saved archives stay readable. In-flight work needs its original runtime
and instructions or a tested migration; never change their pins to force resume.
