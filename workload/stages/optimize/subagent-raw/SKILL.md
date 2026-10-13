---
name: subagent-raw
description: Optimize the kernel with a bare agent loop instead of the GEAK workflow — the alternative and control path required by instruct.md.
---

# subagent-raw

The **alternative** implementation of the optimize stage, and the one `instruct.md`
explicitly asks for: "besides geak, add an option that optimizes with a bare subagent".

Its only difference from `geak-kernel-workflow/` is the prompt that `prompt.sh` writes.
Instead of invoking `kernel_workflow.js`, it hands the agent the loop itself — read the
code, change it, run `unittest.py`, keep whatever is faster. Scheduling, timing, fault
tolerance and KEEP/revert reuse exactly the same `scripts/worker.sh` and
`scripts/scheduler.py`.

To enable:

    "optimize": { "skill": "subagent-raw" }

or without touching the config:

    python3 workload/scripts/stage.py --config <cfg> \
        --set stages.optimize.skill=subagent-raw run optimize

## Why it is kept

1. `instruct.md` requires the option.
2. It is the **control group**. Both paths share one frozen harness, one noise band and
   one cold re-verification gate, so "how much better is the GEAK workflow than a bare
   agent" is measurable rather than assumed.
3. When the GEAK workflow itself is the problem (a broken `kernel_workflow.js`, a missing
   dependency), this path depends on none of the GEAK runtime and keeps running.

## What it lacks relative to the GEAK path

Stated plainly, so the control group is not mistaken for an equivalent substitute: no
multi-candidate generation, no built-in evaluation and rollback loop, and no internal
iteration control such as `budget`/`min_improve`. How many rounds the agent attempts
within one attempt, and how, is entirely up to it. The prompt therefore spells out the
self-test command and the acceptance criteria far more explicitly.

## Output

Identical to `geak-kernel-workflow` — which is precisely what makes the comparison
possible: source lands in `$TASK_DIR/kernel_src/`, results are written by
`scripts/worker.sh` into `$WORK_ROOT/state/<kernel>.status.json`, and the stage summary
is produced by the slot adapter `stages/optimize/finalize.py`.
