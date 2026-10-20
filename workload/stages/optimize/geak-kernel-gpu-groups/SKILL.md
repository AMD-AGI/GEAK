---
name: geak-kernel-gpu-groups
description: Run GEAK kernel optimization when frozen tasks require different GPU counts, keeping single-GPU tasks parallel and genuine collective tasks on their required device group.
---

# Optimize tasks with different GPU requirements

This implements the workload optimize slot. It reuses the existing GEAK task scheduler,
cold verifier and finalizer. Single-GPU tasks can use the standard kernel-workflow prompts
or the [native task prompt](native-task/prompt.sh) for a standalone coding CLI without the
Workflow tool. Multi-GPU tasks use the [collective prompt](collective-method/prompt.sh), which directs
one worker to optimize against the frozen multi-rank entry with its whole device set.
The inner kernel workflow uses a pool lock that selects one GPU, so running it on a
collective would silently reduce an eight-device task to one device. Groups execute
separately, with time reserved for verification and end-to-end measurement.

Read the standard environment from the [stage contract](../../CONTRACT.md), then follow
[the slot's README](../README.md). This implementation is **step 1 only**: run
`python3 $SKILL_ROOT/run_groups.py` to plan the groups and dispatch them. Steps 2 and 3 —
rerunning each task's unit test from cold and writing `optimized.json` and `result.json` —
are identical for every optimize implementation and are described there.
Use `--plan-only` to write `group_plan.json` without starting containers.

Additional `$PARAMS` inputs:

- `task_gpu_requirements`: a map from task directory name to positive GPU count, or
  `task_gpu_requirements_file`: a JSON file containing that map. Without an explicit
  entry, the task's `meta.json.required_gpus` is used. Missing requirements are blocked;
  a collective must never silently become a single-GPU task.
- `single_gpu_method`: `geak-kernel-workflow` (default; requires the runner to expose the
  Workflow tool) or `native-frozen` (native CLI coding against the already-qualified task).
  Both use the same worker, frozen harness and cold verification; no unverified score
  becomes a reported gain.
- `task_name_prefix`: the campaign's unique task-name prefix. The upstream scheduler
  includes task names in global Docker names; this prefix avoids cross-campaign clashes.
- `deadline_utc`: optional ISO timestamp for the existing allocation's deadline.
- `reserve_after_optimization_sec`: minimum time retained after optimization,
  default 21600. `reserve_e2e_sec` protects A/B time specifically (default 14400,
  capped at the minimum reservation). Each group's `verify_wall_sec` allowance
  (default 3600) is additionally reserved; planning uses the larger reservation.
  `verify_wall_sec_by_gpu_count` can override that allowance by GPU count (string keys),
  for example when two long single-GPU tasks need more cold-test time than one collective.
  `wall_overhead_factor` defaults to 1.5.
- `max_wall_sec`: optional additional wall-clock ceiling for all optimization groups.

The usual optimize parameters retain their meaning. GPU groups execute sequentially;
tasks within a group use the existing scheduler's parallel slots. Each group receives a
separate work root. Worker source, credit and attempt history remain there; the canonical
`work/` references those artifacts and `dispatch.json` records each group's scheduler exit
code.

Before launch, conflicting Docker names owned by another work root are refused. On a
wall-clock cutoff, only workers whose `WORK_ROOT` matches the current group are stopped,
No scheduler allocation is created, cancelled or changed. Each group reserves
`verify_wall_sec` so that step 2's cold reruns still fit inside the deadline. KEEP
classification belongs to step 3; only applyback can claim an end-to-end improvement.
