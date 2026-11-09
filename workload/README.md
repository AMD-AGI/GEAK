# Workload optimization through modular experiments

Start by giving an agent the [root skill](SKILL.md), the task/scenario and a run directory.
The agent follows that control loop: it chooses a question,
plans a bounded experiment, invokes modules and uses their reports to decide the next
step. A module defines a capability and its input/output/acceptance contract; one or
more implementation skills or tools provide the method. Replacing a method does not
replace the criterion. Read [PRINCIPLES.md](PRINCIPLES.md) before changing the workflow.

`Reasoning → Plan → Module(s) → Report → Reasoning`

The [standard recipe](recipes/standard.md) retains the existing five-stage route and
its configured budgets. It is one possible plan. A diagnostic call can return a trace
without extracting UTs; a configuration candidate can be evaluated without a kernel
overlay; a ready candidate can be tested while unrelated work continues on other
resources. Each call declares its scope and leaves evidence, including negative results.

## Modules

This is a discovery table, not a registry or a required sequence. A capability may use
part of an existing implementation. Add another skill or command when a new method is
needed; it need not be named in the controller. All calls follow the
[invocation contract](stages/CONTRACT.md#direct-module-calls).

| Module | Responsibility | Available implementations / instructions |
| --- | --- | --- |
| Setup | Prepare the permitted runtime and resources | [Baseline preparation](stages/baseline/direct.md), using the configured environment and resource tools |
| Baseline | Establish or revalidate a reference | [Adopt](stages/baseline/adopt/SKILL.md), [measure](stages/baseline/measure/SKILL.md) |
| Profile | Diagnose the serving bottleneck | [Profile scope](stages/profile/direct.md), with the selected profiler or native telemetry |
| UT | Define and qualify a computation boundary | [UT scope](stages/profile/direct.md), reusing kernel-extract's tools |
| Roofline | Assess a performance bound or search priority | [Kernel roofline](stages/roofline/direct.md), [CPU fixture](stages/roofline/stub-roofline/SKILL.md) |
| Optimize | Produce an isolated candidate | Kernel: [invocation scope](stages/optimize/direct.md), [Kernel Workflow](stages/optimize/geak-kernel-workflow/SKILL.md), [OOB / native tools](stages/optimize/native-frozen/SKILL.md), [raw agent](stages/optimize/subagent-raw/SKILL.md). System: [OOB](stages/optimize/system-oob/SKILL.md) |
| Apply back | Integrate a candidate and prove it took effect | [Kernel integration](stages/applyback/direct.md), [system runtime](stages/applyback/system-runtime/SKILL.md) |
| Evaluate | Judge a serving comparison | [Native E2E](stages/evaluate/native-e2e/SKILL.md), using the benchmark's own quality and validity receipts |

## Evidence and the performance floor

An experiment names its input state and owns its output directory. `BRIEF.md` carries
constraints and the comparison target; `STATE.md` carries pointers and decisions.
Reuse the modules' native receipts and `REPORT.md` rather than copying their counters
into a second state system. A module can report ongoing work, a blocker, or a negative
finding without inventing a candidate or declaring an optimization success.

### Multi-config scenarios

The standard recipe's diagram shows the combined profile/UT workflow. Multiconfig scenarios
split it into `baseline → profile → ut → roofline → optimize → applyback`: profiling
finishes with a hashed dispatch/shape handoff, and UT generation can run or resume
separately using the fixed templates. See [profile/HANDOFF.md](stages/profile/HANDOFF.md)
and [the UT stage](stages/ut/README.md). Existing five-stage configs remain supported.
A multi-config scenario's applyback judges every CI point the UT covered against that point's
own official CI number (`reference: ci`, see `stages/evaluate/README.md`).

Keep the verified fixed-route result available during migration. Preserve required
search and verification budgets; compare policies at the same starting state and total
resource allowance. Only a qualified serving comparison can replace the reference.
A local kernel win, a working controller and a measured system improvement are different
claims. Reference platforms, target metrics and allowed changes belong to the task.

## Where things live

| Path | Purpose |
| --- | --- |
| `SKILL.md` | The agent's control loop |
| `recipes/standard.md` | The fixed route, commands and compatibility behavior |
| `stages/CONTRACT.md` | Direct module calls and the standard recipe's adapter contract |
| `stages/` | Existing implementation skills and their validation owners; directory names are not a required call order |
| `config/` | Maintained scenarios and shared defaults; campaign-specific variants stay with their campaign |
| `run.py` | Stable executor for the standard recipe and individual configured stages |
| `runners/` | CLI adapters, including their credential and exit-code contract |
| `engine/` | [Isolation, scheduling, recovery and evidence tools](engine/README.md) |
| `docs/` | Illustrations, not executable policy |

The existing [pipeline figure](docs/pipeline.svg) depicts the standard recipe. Native
skills remain byte-identical where the manifests require it. New prompt or selection
policy belongs in Markdown; code is reserved for mechanical execution and validation.
