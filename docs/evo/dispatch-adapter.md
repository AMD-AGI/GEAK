# Connect the native owner to the common dispatcher

`geak_evo.dispatch` accepts the common optimizer's `--task PATH --out PATH`
arguments. It checks the task against a pinned owner binding, calls the admitted
command, and returns the common `schema_version: 1` result. A winner is a real
commit in the dispatcher's clone, containing the measured selected source bytes
and executable modes. Its per-case pairs come from the two final native
measurements; the lane's earlier baseline times are not used in those ratios.

The owner binding JSON has these fields:

| Field | Meaning |
| --- | --- |
| `schema` | `geak-evo-dispatch-owner-v1` |
| `owner_config_ref` | Exact config reference accepted by the admitted command. |
| `task_ref` | Exact native contract reference, equal to the common task's `task_ref` and the owner's native `contract_ref`. |
| `case_map` | Every common lane case name mapped one-to-one to the admitted benchmark case ID. |
| `dispatch_root` | Canonical parent of the task's evaluation directory. |
| `device` | The common task's device mapping, as admitted by the existing owner. Actual physical-device evidence remains in the native measurements. |
| `max_steps` | Existing owner decision bound, no larger than the common task's round bound. |
| `export_seconds` | Positive margin between the owner's original deadline and the command cutoff for result conversion and Git work. |

The common objective is preserved, with only its weight keys translated through
`case_map`. It must equal the admitted command's objective. The adapter computes
the same geometric or time-weighted ratio from complete matched pairs, then
requires `speedup > incumbent_speedup * (1 + min_improvement)` before committing.
The provisional result flags remain false; common Report/Validate and workload
acceptance still follow.

This adapter targets the dispatcher whose `task.budget.deadline_epoch` is the
actual SIGTERM cutoff (introduced in `1125c24b`). It converts that epoch once
using current wall and monotonic time. The original native monotonic deadline
plus export margin must fit that cutoff; historical wall timestamps cannot
extend it. Git work uses the same converted cutoff. The CLI temporarily handles
SIGTERM as `CommandInterrupted`, a dedicated `BaseException` that engineer and
planner turns propagate after recording settlement and elapsed time. Arena
evaluations save their collection receipt before propagating the interruption.
The corrected evaluator source requires a new admitted code closure; existing
frozen evaluator pins do not authorize this prospective version. The admitted
command retains pending work and costs and writes an incomplete result; search
cannot treat the cutoff as a recoverable transport failure and launch another
turn. The CLI restores the prior handler when it returns. Recovery never makes
an elapsed operation-envelope overrun into a successful result.

The clone must belong to the coordinator's ordinary host UID. Its editable paths
must be the exact native source-file closure, and source and protected artifacts
must match the admitted ORIGINAL bytes and modes. Changed incumbent source,
directory-style edit grants, another task reference, or partial case mapping
are refused. The adapter changes only source paths and uses raw Git plumbing
for the commit, so commit bytes do not pass through attribute filters or hooks.

The existing owner entry is wired by [owner-dispatch.patch](owner-dispatch.patch),
against the two exact files listed in [owner-dispatch-base.json](owner-dispatch-base.json).
Apply it to a prospective copy containing `host_driver.py` and
`native_host_adapter.py`, then include those changed files and the cleaned
package in a new authenticated installation manifest and native code closure.
The frozen pilot files remain unchanged. The installed command is the existing
owner invocation, with `--dispatch-owner PATH SHA256` added before the lane
appends `--task` and `--out`; all existing coordinator admission/configuration
arguments remain required.

The patch reuses `owner_main` and `run_host` initialization, retained
`validate_config`/`load_imports`, `NativeCallbacks`, and `Budget`. After normal
search preparation, it calls `dispatch.main` with those services and the actual
episode reader. The command config must match that preparation exactly. There
is no new provider loader, source-adoption service or device manager.

CPU tests exercise the real command/evaluator/export/Git path with synthetic
search and owner receipts, and apply the owner patch to its exact source files
to check argument and service wiring. SIGTERM checks interrupt actual engineer,
planner, and evaluation launch/settlement paths, retain settlement/accounting
records, and return an error without another search step or a winner. These are
source and CPU results; this entry has not been installed or qualified in a live
run.

Numeric support covers admitted native-v2 `task-report-v1` graph measurements
and [generated raw paired reports](evaluation-port.md#generated-paired-results).
Both require their existing admitted owner and actual producer evidence; the
adapter does not manufacture a task contract or native report.
