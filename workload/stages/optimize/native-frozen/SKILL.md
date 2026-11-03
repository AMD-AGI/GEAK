---
name: native-frozen
description: Optimize one frozen-harness task with the runner's own coding tools — no Workflow tool, no kernel_workflow.js — on exactly the device set the task declared.
---

# native-frozen

An implementation of the optimize stage, for a runner that does **not** expose GEAK's
Workflow tool. Everything else is the same as `geak-kernel-workflow/`: the same
`scripts/worker.sh`, the same frozen harness, the same cold verification, the same
KEEP threshold. Only `prompt.sh` differs — it asks the session to edit
`kernel_src/` with whatever coding tools it has, instead of driving
`kernel_workflow.js`.

Choose it in config when the runner is a plain coding CLI:

```json
"optimize": {"skill": "native-frozen"}
```

## Why a separate method rather than a flag

`kernel_workflow.js` holds a pool lock that selects **one** GPU. Run it on a task that
declares eight and the collective is silently reduced to a single device: every rank
after the first is missing, and whatever the attempt measures is not the task. This
method takes no such lock. It hands the session the whole assigned set through
`$GPU_IDS` and tells it to launch the real collective across all of it.

## Device width is the task's own declaration

`scripts/scheduler.py` reads `required_gpus` from each task's `meta.json` and gives the
worker that many consecutive GPUs, physically mounted. A single-rank elementwise kernel
and a TP8/EP8 collective therefore run in the same campaign, side by side, with no
grouping step and no per-campaign GPU width. A task that declares nothing gets
`--gpus-per-task`.

This replaced `geak-kernel-gpu-groups`, which existed only because the scheduler once
took one width for the whole run and tasks had to be partitioned into same-width batches
before it could be called. The requirement was already in `meta.json`.

## What the prompt holds the session to

The same four rules as `geak-kernel-workflow/SKILL.md` ("Four things the steer must
state") — frozen harness, noise band, generalize across the shape sweep, write every
output — plus two that only matter without the Workflow tool:

- **Keep the assigned visibility.** The container exposes exactly `$GPU_IDS`; a session
  that narrows it measures a different task than the one it was given.
- **Reuse the qualified harness.** The profile stage already built and timed it. An
  attempt that spends its budget rebuilding a harness has optimized nothing. The
  authoritative check is `scripts/run_task_entry.py "$TASK_DIR"`, never
  `python3 unittest.py` from inside the task directory, which shadows the stdlib module
  of that name.

## Output

None of its own. Per-kernel results are `scripts/worker.sh`'s
`$WORK_ROOT/state/<kernel>.status.json`; the stage's verdict is
`stages/optimize/finalize.py` (`stages/CONTRACT.md` 3.1).
