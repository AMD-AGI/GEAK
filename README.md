# GEAK + GEAK-EVO

GEAK improves GPU kernels by inspecting code, testing changes, and measuring the
complete operation. GEAK-EVO retains alternative candidates in that workflow:
the planner can develop an earlier candidate and refer to other candidates when
combining ideas. Both use the same entry point, agents, tools, and evaluator.

## Start here

| I want to… | Read |
| --- | --- |
| Install GEAK | [Installation](docs/install/install.md) |
| Improve one kernel | [Kernel workflow](kernel_workflow/README.md) |
| Enable GEAK-EVO | [Run the stock workflow](docs/evo/stock-workflow.md) |
| Improve a serving workload | [Serving workflow](e2e_workflow/README.md) |
| Use GEAK from Hyperloom | [Task and result interface](interface/run_e2e.md) |
| Understand or change GEAK-EVO | [Architecture](docs/evo/architecture.md) |
| Compare GEAK and GEAK-EVO lineages, populations, and results | [Diagrams and code comparison](docs/evo/geak-vs-evo.md) |
| Inspect retained candidates | [Archive guide](docs/evo/archive.md) |
| Use a supplied native task evaluator | [Native measurements](docs/evo/direct-external-command.md) |
| Find existing examples and results | [GEAK guide](docs/reference/geak-guide.md) |

Build and test the original kernel in its GPU environment before optimizing it.
GEAK uses the compiler, libraries, and model weights you provide.

## GEAK-EVO

Set `optimizer: "evo"` in the ordinary kernel workflow arguments. The default
`optimizer: "greedy"` uses GEAK's existing search. GEAK-EVO adds three operations:

1. Select a retained parent and optional donor references during round planning.
2. Develop that parent with the existing engineers and independent verifier.
3. Retain the patch and evidence, including slower or unfinished attempts, for
   later rounds. Promotion still requires the shared improvement checks.

The implementation supports these operations; their availability does not
establish useful diversity, successful recombination, or sustained performance
benefit. Those outcomes require measurements on the intended workload.

The former Python EVO driver and its pilot, group-control, and export tooling
have been retired. The current path is the built-in kernel workflow described
above. See [architecture](docs/evo/architecture.md) for scope and source links.

## Where the code lives

| Directory | Responsibility |
| --- | --- |
| `interface/` | Accept tasks, run workflows, and return results |
| `kernel_workflow/` | Set up, optimize, and validate kernels; retain EVO candidates |
| `e2e_workflow/` | Profile and measure a serving workload |
| `geak/` | Installation support and the shared native measurement helpers |
| `scripts/native_runtime/` | Adapter for a supplied native task evaluator |
| `kb/` | Store and retrieve verified run records |
| `perf_knowledge/` | Hardware guides and reusable skills |
| `docs/`, `examples/`, `ci/` | Guides, examples, and automation |

Workflow code lives in the `src/` folders under `kernel_workflow/` and
`e2e_workflow/`. Edit those files, then run `node scripts/build_workflows.mjs`.
The generated scripts stay in the checkout because the workflow runner loads
one file. CI checks that they match the source.

## Contributing

Keep shared logic in one place and put experiment-specific setup and results
outside the runtime. See [CONTRIBUTING.md](CONTRIBUTING.md).
