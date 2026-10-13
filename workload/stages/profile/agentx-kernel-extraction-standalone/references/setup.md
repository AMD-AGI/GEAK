# Setup and required inputs

The agent reads `SKILL.md` and uses the scripts in this folder directly. No
package installation, orchestration framework, network service or repository
checkout is needed for the skill itself. Keep the entire folder together.

```bash
export SKILL_ROOT=/absolute/path/to/agentx-kernel-extraction-standalone
export EVAL_DIR=/absolute/path/to/new/extraction-evidence
python3 "$SKILL_ROOT/scripts/check_bundle.py"
```

Use Python 3.10+ and Node.js 18+. Python helpers use the standard library until
real tensor/GPU work requires the serving environment's PyTorch and backend.
CPU bundle tests use fakes for tensor interfaces and need no GPU or PyTorch.
The scripts do not install/upgrade packages. Node preflight uses only built-ins
and bundled modules; no npm install is required.

For actual extraction, the user supplies an accessible serving environment,
model and backend, appropriate native/JIT compilers, baseline logs and exports,
compatible raw traces or resources for bounded new capture, and GPU allocation.
Record graph/compile settings, current source/build versions and accepted overlays.
Do not invent these from an old run. A trace-only environment can support
inventory/ranking work but cannot validate new GPU execution.

The bundled GPU wrappers target Linux/Bash/flock and the AMD/ROCm environment
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

All participating processes/containers must share the same
`/tmp/team_gpu_locks` lock directory (bind-mount it when necessary), or set
`KERNEL_EXTRACT_LOCK_DIR` explicitly to another writable directory shared by
all participants. Never silently fall back to a task-private lock path when
the shared directory is inaccessible. Existing
serving jobs outside that locking scheme must be accounted for by scheduler
exclusivity or serialization. Never disable idleness checks to bypass contention.
The wrapper also keeps build caches task-local and checks source provenance
using bundled `workspace_sources.py`; it requires no external provenance service.

When no accepted overlay exists, create an empty task-local `baseline_overlay/`;
imports then resolve the installed serving baseline outside the task. If an
accepted overlay exists, stage its configuration while retaining its immutable
external source references. Copying baseline source into the candidate task
does not establish an independent baseline.

Put model artifacts, captures and outputs under `EVAL_DIR`, not inside the skill.
Keep credential handling and remote/container access in the user's existing
tools. The distributed skill contains neither run-specific credentials nor
model data.
