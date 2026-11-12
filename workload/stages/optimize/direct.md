# Optimize calls

Use the [invocation contract](../CONTRACT.md#direct-module-calls). Kernel implementations
are [Kernel Workflow](geak-kernel-workflow/SKILL.md),
[kernel dynamic loop](kernel-dynamic-loop/SKILL.md) and [OOB](oob/SKILL.md).
They share the frozen-task, budget and cold-verification contract in [README.md](README.md).
OOB also handles serving-system experiments through direct calls with the plan's system
validation. The scoped kernel instructions below apply to qualified kernel tasks.

Before dispatch, compare `source_tree.py editable <task>` with the qualified UT contract.
A callable's binding adapter is not necessarily its editable computation. Missing delivery
proof or conflicting source permissions returns to UT; a passing test alone cannot admit it.
Budget attempts around observed full-test and method-completion costs. An attempt cap is
not permission to replace the selected method or bypass its verification and integration.

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
