# GEAK kernel optimization skills

## v4

Select `stages.optimize.skill=kernel-skills` in a workload scenario to run v4 on a qualified
kernel. The worker supplies the frozen UT, assigned GPUs and attempt budget. For a standalone
run, give the lead the task directory (`INPUT`) and allocated GPU IDs (`GPUS`).
[Workload entry](../workload/stages/optimize/kernel-skills/SKILL.md) · [Lead skill](v4/geak-lead/SKILL.md)

The lead selects parent candidates from the measurement archive and dispatches the benchmark,
roofline, investigator, engineer and ship roles. Each round records its decisions in
`decisions/round_<n>.json`. The supplied search strategies are `evo` (default), `hillclimb`
and `avo`. [Search strategies](v4/search/SHELL.md) · [Offline replay](v4/EVIDENCE.md)

## EAT

For a standalone Evolving Agent Teams run, give the EAT skill the input task, a fresh
destination, the allocated GPUs, budget, iteration limit and team size. GEAK prepares
the frozen evaluation and calibrates its noise. The EAT supervisor chooses the team
each iteration, and an independent audit checks the shipped result.
[EAT skill](geak-eat/SKILL.md) · [Supervisor playbook](geak-eat/PLAYBOOK.md)

Both methods use the shared helpers in `bin/` and resolve the workspace and GPU tools
from the sibling `kernel_workflow/` directory in this checkout.
