# Setup and required inputs

Use the tools bundled with this skill:

```bash
export SKILL_ROOT=/absolute/path/to/workload/stages/profile/kernel-extract
export SHARED_TOOLS="$SKILL_ROOT/shared"
export EVAL_DIR=/absolute/path/to/extraction-evidence
export PARSE_PROFILE_STRICT_PHASES=1 KERNEL_ENV_SKIP_ENUM_REAP=1
```

`PARSE_PROFILE_STRICT_PHASES=1` preserves this method's refusal of unknown forward
annotations. Use `kernel_selection.py --strict-launch` for single-anchor evidence;
joint verification already uses strict launch attribution. `KERNEL_ENV_SKIP_ENUM_REAP=1`
keeps GPU wrappers from cleaning up unrelated processes. Export these controls into
the capture/analysis processes, including remote/container commands.

Use Python 3.10+ and Node.js 18+. Real tensor/GPU work requires the serving
environment's PyTorch/backend. The helpers do not install or upgrade packages.

For actual extraction, the user supplies an accessible serving environment,
model and backend, appropriate native/JIT compilers, baseline logs and exports,
compatible raw traces or resources for bounded new capture, and GPU allocation.
Record graph/compile settings, current source/build versions and accepted overlays.
Do not invent these from an old run. A trace-only environment can support
inventory/ranking work but cannot validate new GPU execution.

The GPU wrappers target Linux/Bash/flock and the AMD/ROCm environment
from which the skill was developed. Their idleness heuristic is not proof of
exclusive physical ownership. Map caller allocation to device indices and
verify scheduler/container isolation first. Different logical IDs or separate
containers do not by themselves prove disjoint devices. Other GPU backends need
an equivalent verified allocation/visibility mechanism before GPU work; the
portable skill does not claim new backend support.

The legacy variable names `GEAK_GPU_ALLOWED`, `GEAK_GPU_REQUIRE_IDLE` and related
`GEAK_*` names inside vendored helpers are local configuration keys, not imports
or dependencies on GEAK. Set `GEAK_GPU_ALLOWED` to the caller's actual allocated
set. Preserve an inherited `ROCR_VISIBLE_DEVICES` allocation mask. A set lease
restores the requested `HIP_VISIBLE_DEVICES` only after acquiring all locks.
Do not copy the variable values from a previous run.

Run every GPU command from its task directory under a lease:

```bash
cd "$TASK_DIR"
bash "$SHARED_TOOLS/gpu_lock.sh" "$GPU_ID" python3 unittest.py
# For actual multi-rank tests or serving captures, lease the whole allocated set:
python3 "$SHARED_TOOLS/gpu_lock_set.py" "$GPU_SET" -- <foreground-command>
```

A single-lock comma list is a pool that leases ONE device; a set lock holds every
requested device. Reconcile device IDs with the physical allocation. Serialize captures
and UTs if isolation cannot be established, and let no detached GPU child outlive the
lease.

All participating processes/containers must share the same
`/tmp/team_gpu_locks` lock directory (bind-mount it when necessary), or set
`KERNEL_EXTRACT_LOCK_DIR` explicitly to another writable directory shared by
all participants. Never silently fall back to a task-private lock path when
the shared directory is inaccessible. Existing
serving jobs outside that locking scheme must be accounted for by scheduler
exclusivity or serialization. Never disable idleness checks to bypass contention.
The wrapper also keeps build caches task-local and checks source provenance
using `$SHARED_TOOLS/workspace_sources.py`; it requires no external provenance service.

When no accepted overlay exists, create an empty task-local `baseline_overlay/`;
imports then resolve the installed serving baseline outside the task. If an
accepted overlay exists, stage its configuration while retaining its immutable
external source references. Copying baseline source into the candidate task
does not establish an independent baseline.

Put model artifacts, captures and outputs under `EVAL_DIR`, not inside the skill.
Keep credential handling and remote/container access in the user's existing
tools. Keep run-specific credentials and model data outside the repository.
