# GEAK kernel optimization skills

## v4

Select `stages.optimize.skill=kernel-skills` to have workload Brain delegate one qualified
kernel to a supervisor. Each round it chooses roles, combines v4 skills, defines handoffs
and dispatches the team. The worker supplies the frozen UT, assigned GPUs and budget.
[Workload entry](../workload/stages/optimize/kernel-skills/SKILL.md) · [Kernel supervisor](v4/kernel-supervisor/SKILL.md)

For a standalone v4 run, give the [lead](v4/geak-lead/SKILL.md) the task directory
(`INPUT`) and allocated GPU IDs (`GPUS`). Both entry points reuse the candidate archive,
benchmark, roofline, investigator, engineer and ship skills. Each round records its decisions in
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
