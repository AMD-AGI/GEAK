# Kernel optimization through modular experiments

Start with [SKILL.md](SKILL.md). The controller chooses a question, then a module and
one implementation. Read [PRINCIPLES.md](PRINCIPLES.md) before changing this tree.

## Modules

This is a discovery table, not an execution order. Each module's `README.md` owns its
input/output and acceptance contract. Implementation skills describe how to perform it.

| Module | Responsibility | Initial implementation |
| --- | --- | --- |
| [Setup / Benchmark](stages/setup/README.md) | Establish usable task and measurement inputs | [Task inspection](stages/setup/task-inspect/SKILL.md) |
| [Profile](stages/profile/README.md) | Resolve a bottleneck question | [Native inspection and profiling](stages/profile/native-profile/SKILL.md) |
| [Research](stages/research/README.md) | Find applicable prior evidence or approaches | [Evidence search](stages/research/evidence-search/SKILL.md) |
| [Optimize](stages/optimize/README.md) | Produce an isolated candidate | [Native coding](stages/optimize/native-coding/SKILL.md) |
| [Verify](stages/verify/README.md) | Establish candidate validity and measured performance | [Frozen native UT](stages/verify/frozen-ut/SKILL.md) |
| [Integrate](stages/integrate/README.md) | Test a combination of candidate changes | [Source merge](stages/integrate/source-merge/SKILL.md) |
| [Learn](stages/learn/README.md) | Retain a useful, evidence-linked lesson | [Local note](stages/learn/local-note/SKILL.md) |
| [Deliver](stages/deliver/README.md) | Hand source and evidence back to the parent | [Source handoff](stages/deliver/source-handoff/SKILL.md) |

Add another method at `stages/<module>/<implementation>/SKILL.md`, following the module
contract and [shared call contract](stages/CONTRACT.md). Select its path in the plan;
the controller needs no code change or registry. Changing a method does not change
what counts as passing.

`SKILL.md` owns reasoning; `stages/` owns capabilities and their implementations.
The root `prompt.sh` and `measure.sh` adapt this method to workload's existing worker.
The parent [Optimize contract](../README.md) still owns dispatch, KEEP and cold
verification. Tests under `tests/` exercise these adapters without GPUs or agent calls.
