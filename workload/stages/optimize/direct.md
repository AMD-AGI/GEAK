# Optimize calls

Use the [invocation contract](../CONTRACT.md#direct-module-calls). Kernel implementations
are [Kernel Workflow](geak-kernel-workflow/SKILL.md) and
[native tools / OOB](native-frozen/SKILL.md); [raw agent](subagent-raw/SKILL.md) is another
method. They share the frozen-task, budget and cold-verification contract in [README.md](README.md).
For a serving configuration or implementation experiment use [system OOB](system-oob/SKILL.md).

For a scoped kernel call, prepare separate working copies of the qualified tasks named
in the plan. Give the existing scheduler that task directory and a fresh work root;
keep all configured worker limits and verification reservations. Cold-test the retained
sources before publishing them. `finalize.py --tasks-dir <scoped-tasks> --optimized <report>`
accounts for this invocation, with its `PARAMS` and `OUT`; all supplied tasks must appear.
Leave unselected discoveries and ongoing work visible in campaign state.

The standard recipe always supplies the full configured profile delivery. A smaller
standalone scope is not evidence that the full campaign or a required time budget is
complete. Once a candidate is verified, integration and evaluation may use its immutable
snapshot while other work continues on disjoint resources. Independent measurements
must also retain the configured reference and validation budget.
