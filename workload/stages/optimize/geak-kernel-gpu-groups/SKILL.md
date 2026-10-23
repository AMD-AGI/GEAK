---
name: geak-kernel-gpu-groups
description: Run GEAK kernel optimization when frozen tasks require different GPU counts, keeping single-GPU tasks parallel and genuine collective tasks on their required device group.
---

# Optimize tasks with different GPU requirements

This implements the workload optimize slot. It reuses the existing GEAK task scheduler;
cold-test judgement and final reporting stay in the slot's README. Single-GPU tasks use the standard kernel-workflow prompts
or the [native task prompt](native-task/prompt.sh) for a standalone coding CLI without the
Workflow tool. Multi-GPU tasks use the [collective prompt](collective-method/prompt.sh), which directs
one worker to optimize against the frozen multi-rank entry with its whole device set.
The inner kernel workflow uses a pool lock that selects one GPU, so running it on a
collective would silently reduce an eight-device task to one device. Groups with disjoint
GPU pools execute concurrently, with time reserved for verification and end-to-end measurement.

Read the standard environment from the [stage contract](../../CONTRACT.md), then follow
[the slot's README](../README.md). This implementation is **step 1 only**: run
`python3 $SKILL_ROOT/run_groups.py` to plan the groups and dispatch them. Steps 2 and 3 —
rerunning each task's unit test from cold and writing `optimized.json` and `result.json` —
are identical for every optimize implementation and are described there.
Use `--plan-only` to write `group_plan.json` without starting containers.

Before dispatch, follow the slot's readiness checks. Choose the method from the runner's
actual capabilities: GEAK's Workflow tool is the default when available; a CLI without
that tool uses `single_gpu_method=native-frozen`. Changing the runner remains a config
choice and never changes scheduling code.

Additional `$PARAMS` inputs:

- `task_limit`: how many extracted tasks this campaign holds in flight at once, default 8.
  It bounds dispatch, not extraction: the head of the roofline order fills the wave and
  every task beyond it is reported as `capacity_deferred` with its rank and reason. Every
  directory must therefore appear in dispatch or in that deferral list, and in final
  reporting. `max_tasks`/`exclude_tasks` are refused when they would omit any delivered
  task; group-local scheduler exclusions only partition the complete set across disjoint
  pools. A task with a missing source directory is a visible delivery defect, not an
  absent task.
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

The usual optimize parameters retain their meaning. `group_plan.json` records each
group's `batch`, physical `gpus`, slots and task waves. Groups in the same batch launch
together on disjoint pools; each existing scheduler fills its own parallel slots. Groups
that cannot each fit a whole slot share no batch. For example, two single-GPU tasks and a
four-GPU collective can run together on eight GPUs; an eight-GPU collective must follow
the single-GPU tasks on that node. Fixed partitions last for the batch; this helper does
not migrate running workers or allocate more machines.

Each GPU-count group keeps a stable work root for resume. Worker source, credit and
attempt history remain there; the canonical `work/` references those artifacts.
`dispatch.json` links a timestamped `dispatch_attempts/` receipt with scheduler start/end
times, elapsed seconds, physical pools and exit codes. Preserve previous attempts when
resuming, and use these receipts to distinguish wall time from summed worker time in
progress reports. A work-root lock refuses a second live dispatcher.

Before launch, conflicting Docker names owned by another work root are refused. On a
wall-clock cutoff, only workers whose `WORK_ROOT` matches a started group are stopped.
No scheduler allocation is created, cancelled or changed. Each group reserves
`verify_wall_sec`; the full cold-test and A/B reservation remains protected through the
last search batch, because step 2 has not run yet. KEEP
classification belongs to step 3; only applyback can claim an end-to-end improvement.
