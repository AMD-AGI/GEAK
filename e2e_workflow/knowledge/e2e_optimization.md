# Workload optimization — shared decision guide

Use the objective, quality gates, comparison method and original deadlines supplied by this run.
Fixed ISL/OSL output throughput and AgentX trace-replay metrics are different objectives. For AgentX,
retain the declared total-token throughput, interactivity, TTFT and quality definitions; a fast isolated
kernel or a sampled phase does not establish improvement in any of those workload measurements.

When this run supplies both kernel/region and whole-workload objectives, pursue them
concurrently and report them separately in the existing reasoning and reports. Kernel attainment
is not a prerequisite for workload optimization and does not establish workload attainment. Use a
workload fraction only with a separate supported ceiling for the same metric, request work/mix,
resource budget and serving regime; a sum or average of kernel fractions is not that ceiling.
Without it, report measured serving results and the missing denominator as unknown. Select among
kernel, complete-region and available recipe/system actions using workload evidence; high local
attainment does not make avoidable work necessary. Reuse applicable models and observations rather
than adding a calibration stage before useful engineering. Reference targets do not relax numerical
correctness, quality, latency/error gates, protected configuration or the declared evaluation budget.

## Follow cost through the workload

Start with the complete captured inventory, actual callables and producer/consumer dependencies.
Top-N is an index for preparation, not a complete account of optimization opportunities. Preserve
failed extraction, unknown state, graph-hidden work and unrepresented phases as visible gaps.
Keep each cost tied to its source/configuration, rank, stream, phase, shapes and measurement window.
Do not merge same-named or same-shaped calls until layouts, state, semantics and dispatch agree.

Distinguish summed dispatch durations, interval-union activity and workload elapsed time. Concurrent
dispatches contribute separately to the sum. An interval occupied only by one kernel family is not
automatically removable time; dependencies and other ranks can still determine completion. Overlapped
work can also matter through resource contention. Inspect the relevant evidence before choosing a
critical-path explanation. A capture's phase proportions are not the whole replay's weights.

Under a fixed serial model, accelerating fraction `f` of baseline elapsed time by factor `s` saves
`f * (1 - 1/s)` of that time and yields `1 / (1 - f + f/s)` speedup. Use per-case terms when coverage
or acceleration differs. Substituting a share of summed GPU durations gives only a conditional
device-time model. Never rank by share times speedup or call that product an E2E gain.
Do not count a parent region and its member events twice. Estimated improvements, applicable
roofline comparators and actual workload measurements remain distinct evidence.

## Choose the investigation and its executable boundary

Compare plausible recoverable cost, uncertainty, implementation effort and remaining engineering
time. Large operators are useful targets. Groups of small operations, transfers, launch gaps and
changes to dataflow can also be valuable. A roofline describes one formulation and does not prove
that its work, materializations or synchronization are necessary, or that local tuning is exhausted.

Use these as questions, not fixed specialist assignments or mandatory steps:

- **Operation or region:** can an equivalent algorithm, layout, backend or fusion boundary reduce
  work or movement? Read the complete callable and intermediate/state contract. Triton, HIP,
  FlyDSL, CK or assembly are options only where their tools and deployment boundary are qualified.
- **Communication and scheduling:** what do producers, collectives and consumers wait for? Check
  message sizes, routing, rank imbalance and resource sharing. Skew or a collective name alone
  does not prove busy-wait, establish intrinsic transfer time or exclude a kernel rewrite.
- **Serving and capacity:** do KV/cache movement, memory placement, batching, request ownership or
  graph execution explain the workload behavior? Use only actions exposed by this run. For nested
  AgentX, Hyperloom owns enabled serving-recipe actions; the inner kernel path cannot execute
  arbitrary flags or unsupported topology changes. Read the current action menu and its configuration,
  backend, scheduling and memory options before concluding that an action is absent. Record missing
  capabilities rather than pretending a nominated direction has an executor.

Prefill and decode often have different shapes, state and resource limits, and can share a batch.
Identify them from actual observations; neither phase is universally dominant. Check the full
workload objective before excluding a prefill-only or decode-only change.

A new multi-operation boundary needs captured inputs/outputs/state, supported numerical equivalence,
all replaced work inside timing, and a real integration point. Nominate it through the existing
extraction path. If extraction cannot establish the contract, preserve that gap and pursue other
ready work; a larger region is not made executable by naming it a megakernel.

## Integrate, measure and continue

- Bind the replacement to the actual serving callable and verify its loaded/executed path. Keep
  fallback outside qualified coverage. Compatible tensor shapes alone do not prove engagement.
- Prepare JIT code before graph capture using the run's existing warmup path. Keep initialization,
  compilation and benchmark timing separate, within the declared startup and final allowances.
- Include preprocessing, temporary storage, cache keys/lifetimes, copies and synchronization in the
  replaced boundary. Size caches for the actual working set, aliases and lifetimes; pointer equality
  alone does not establish immutable contents or safe reuse. Do not move work outside measurement
  or assume an observed constant is universal.
- Use the supplied same-condition ORIGINAL/current-reference/candidate measurement procedure and
  full quality scope. Preserve all failed legs. A noise allowance or proxy window is not proof of
  reproducibility; do not replace the declared method with a generic request count or shorter replay.
- Preserve exact source, configuration, results and incomplete delivery through the existing archive
  and checkpoint paths. A correct slower realization can retain useful evidence for later development.
  Preservation and closure must not consume the engineering time already assigned to a lineage.
- After an accepted change, reconsider the workload and reuse observations whose applicability still
  holds. Refresh only the evidence needed for the next decision. Stop according to remaining useful
  investigation and the declared budget, not because every untouched row is individually small.
- Resume supported checkpoints without erasing prior failures, changing original clocks or skipping
  required finals. Deliver the best eligible exact bytes and keep better undeliverable results visible.

Check observed fallback messages such as "not found tuned config ... using default config" against
actual shapes, dtype, backend and selected dispatch. They identify an investigation, not proof that
another configuration is faster. Keep helper imports passive/lazy where appropriate so inspection
does not initialize the model, spawn workers or reserve GPUs. Avoid output directories that shadow
importable packages, use isolated compiler caches, and preserve lifecycle ownership.

Use controlled sequential measurements for headline comparisons under the declared method. Disjoint
GPUs can still share host memory bandwidth, PCIe or network resources. Concurrent research is useful,
but its result describes that co-tenancy until appropriately confirmed; interference need not affect
every candidate equally or in a single direction.
