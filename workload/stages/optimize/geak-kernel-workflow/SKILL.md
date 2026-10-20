---
name: geak-kernel-workflow
description: Optimize one frozen-harness unit-test task with GEAK's own kernel_workflow.js — one task per GPU, attempts chained so improvements accumulate, until the time budget is exhausted.
---

# geak-kernel-workflow

The **default** implementation of the optimize stage. It hands each task directory
extracted by the profile stage (`tasks/<kernel>/`, containing `unittest.py`, `meta.json`,
`cases.py` and `kernel_src/`) to GEAK's `kernel_workflow/kernel_workflow.js`, which edits
`kernel_src/<mod>.py`.

This skill supplies the **optimization method** only; parallelism, timing, fault
tolerance and KEEP/revert belong to the stage-shared `scripts/worker.sh` and
`scripts/scheduler.py`. `prompt.sh` is the boundary between them:

    scripts/worker.sh  ──(env: K / TASK_DIR / KERNEL_SRC / ATTEMPT / BEST_SPEEDUP /
                            EVAL_DIR / ATTEMPT_CAP / PROMPT_OUT)──>  prompt.sh
                       <──────────────── $PROMPT_OUT ────────────────┘

So changing the optimization method means a different directory containing a `prompt.sh`
(see `subagent-raw/` alongside it), with no edit to scheduling or accounting.

## What it asks the agent to do

The prompt from `prompt.sh` instructs one unattended session to run
`kernel_workflow.js` with:

| Parameter | Value | Why |
| --- | --- | --- |
| `kernel_path` | `$TASK_DIR/kernel_src/<mod>.py` | the only editable file |
| `workflow_dir` | GEAK's `kernel_workflow/` | |
| `eval_dir` | `$EVAL_DIR` (this attempt's own directory) | intermediate artifacts from different attempts must not overwrite each other |
| `exp_root` | `$RUN_DIR` | |
| `budget` | config `optimize.params.budget` (default 4) | internal iterations within one attempt |
| `min_improve` | config (default 0.05) | |
| `apply_to_original` | `"true"` | **critical**: changes land back in `kernel_src/`, so the next attempt continues from there — this is what lets short attempts accumulate |
| `task` | natural-language task description | see below |

## Four things the steer must state

All four were written in because a previous 200 GPU-hour campaign saw each of them
actually violated:

1. **The harness is frozen.** Only `kernel_src/<mod>.py` may be edited. `unittest.py`,
   `cases.py` and `meta.json` are off limits — a "speedup" obtained by editing the
   harness measured nothing.
2. **The noise band.** An improvement below the keep threshold (default 1.045) is noise,
   not a result. Four kernels in the previous campaign whose source was md5-**identical**
   to the original re-measured at 1.0246–1.0453 from cold.
3. **It must generalize across the length sweep.** `cases.py` contains the full shape
   sweep; tabulating for one shape, or branching on shape to skip computation, shows up
   immediately in the aggregate, which is
   `sum(baseline_ms×calls)/sum(candidate_ms×calls)` rather than a best case.
4. **Not writing memory nobody reads is not correctness.** The unit-test leg allocates
   from a virgin allocator, so zeroed pages make "skip part of the output" pass every
   gate while failing in production — exactly what happened once in the previous
   campaign. Every output must be written in full.

## Output

This skill does not write `result.json`. Per-kernel results are recorded by
`scripts/worker.sh` in `$WORK_ROOT/state/<kernel>.status.json` (`best_speedup`,
`credited_seconds`, `attempts`, `history`), with the best source in
`$WORK_ROOT/runs/<kernel>/best_kernel_src`. Cold re-verification and the stage summary are
steps 2 and 3 of [the slot's README](../README.md) — they belong to the slot, not to this
implementation, so replacing the implementation does not require rewriting them.

## Preconditions

- `kernel_workflow/kernel_workflow.js` exists in the GEAK repository.
- A working runner (`runners/CONTRACT.md`). With credentials missing the runner returns
  78 and the worker stops immediately, rather than spending the time budget on retries
  that cannot succeed.
- Task directories come from the profile stage and already contain `kernel_src/`.
