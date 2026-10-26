# Continuing lineages

An opt-in population runs several ordinary `kernel_lane` workflows through the
existing `kernel_workflow` entry point. Each member has its own planner, history,
incumbent, engineering budget and mutable directories. They share an allocated
GPU pool; a GPU is held for a measurement or diagnostic command, not for the
whole lineage's development.

Rounds advance independently across members. Within a member, the existing
engineer/verifier round barrier remains. Later engineer invocations continue
from retained source, history and evidence; this does not require one model
session to stay alive for the whole run.

```text
One ordinary greedy lineage

  incumbent -> parallel engineers -> evaluate -> best eligible next incumbent
       ^                                              |
       +---------------- next round ------------------+

Population

  a: a0 -> a1 -> a2 -> ...     ordinary greedy, no peer imports
  b: b0 -> b1 -> b2 -> ...     retained alternatives, deliberate peer imports
  c: c0 -> c1 -> c2 -> ...     retained alternatives, deliberate peer imports
       |        |        |
       +--------+--------+---- frozen source-bound evaluation requests
                                      |
                              shared GPU command pool
                                      |
                             evidence for exact sources
                                      |
                            retained population best
```

The global best is a deliverable, not a replacement for every local incumbent.
With a declared greedy member, ordinary GEAK's policy is part of the population;
there is no GEAK-specific supervisor, priority or tie-breaking rule. The generic
interface also permits populations without that policy. Diversity comes from actually pursuing
different approaches and retaining their evidence. A larger member count alone
does not establish diversity or an evolutionary advantage.

## Configuration

Keep the usual task, model, native evaluator and deadline arguments. Set
`mode: "optimize"`, `optimizer: "evo"`, an explicit absolute `eval_dir`, and the
GPU pool allocated to this workflow. Then add a lineage list, for example:

```json
{
  "lineages": [
    {"id": "a", "optimizer": "greedy", "imports": []},
    {"id": "b", "optimizer": "evo", "imports": ["a", "c"]},
    {"id": "c", "optimizer": "evo", "imports": ["a", "b"]}
  ]
}
```

Imports name configured peers. Import-enabled EVO members can deliberately
select compatible published sources as parents or donors. Importing does not
change the member's incumbent automatically. Greedy members require no imports
and retain the ordinary planning policy. With no `lineages` argument, the
existing single-workflow behavior is unchanged.

Mutable evaluation state, local knowledge writes and compiler caches are
isolated. Members start with copies of the configured knowledge/state seeds;
remote knowledge reads retain their configured policy and remote writes are
disabled for members. Native measurement prefixes use the stock adapter's
`--state-root` and GPU option, so the dispatcher can derive private state and
replace a pinned measurement lock with the common `--gpu-pool` argument.

The runtime's model concurrency and the number of lineages are separate limits.
Each lineage may have several engineers and inherits its own copy of the declared
search budget; the dispatcher does not divide that budget among members. They
share the runtime's model semaphore and total-call limit. Configure actual
allowances explicitly; 24 lineages can demand substantially more than 24 active
engineers. Shared service does not reproduce a separate GEAK run's resources or
stochastic trajectory.

## Queued evaluations

Use the stock native adapter's capture and `--captured` interfaces described in
[the native runtime guide](../../scripts/native_runtime/README.md). Capture
freezes editable source before the GPU wait. Keep candidate ID, evaluation ID and
the owning tool's process handle together. A result for source A remains a result
for A even if the engineer has subsequently edited source B. Restore A before
returning its score, or evaluate B separately.

Existing background/process tools allow independent work while a measurement
waits. Settle or cancel agent-owned jobs before returning; this interface does
not add a detached service or guarantee survival after an engineer exits. Native
measurement and diagnostics use the same GPU lock namespace. The stock pool
selects a free, idle device; it does not provide a proven FIFO or starvation
bound. Queue waiting remains part of the original wall-clock allowance.

## Checkpoints and delivery

Published checkpoints are source-bound and retained separately from mutable
working directories. The read-only `scripts/query_population.py --root
<eval_dir>/population` query inspects retained evidence without launching a
measurement. Native records are checked through the configured native reader.
Peer patches carry their development base; a patch exported through the normal
final-result interface must instead apply to the original task.

The native population confirms each lineage's best deliverable source through
the caller's original evaluator state, using the same selected GPU for every
confirmation. It then selects the best qualifying result and returns the
ordinary result contract, without an additional measurement of the winner. Search
rankings do not eliminate a lineage from this comparison. Each confirmation's
source, measurement and failure status remain available. A failed confirmation
does not discard successful peers; an unavailable member has no inferred score.

Follow the returned `final_patch` and evaluation IDs. If the deadline prevents
writing the usual top-level patch alias, an accepted result can reference its
already retained immutable member patch. Alias and persistence failures are
reported explicitly. No-gain results keep the ordinary suppressed-export
behavior; a consumer that assumes the conventional alias must check its presence.

In bounded native runs, explicitly allocate `population_final_reserve_s` for
**all** common confirmations, including GPU waiting and preparation. Member
deadlines and search reserves are reduced together, preserving the original
absolute search cutoff. The remaining member reserves must still cover their
validation and cleanup. No search clock is reset and no final allowance is
inferred from an arbitrary multiplier.

The query exposes one best deliverable candidate per lineage in `finalists`.
`best_available_source` also identifies recoverable sources that cannot yet be
exported through the final-result interface. Initial validated sources and
committed local winners are published; slower intermediate candidates remain in
their local archives. Legacy checkpoints can be recovered or imported, but
legacy final export requires an actual child final. They retain the weaker
legacy verifier qualification.

For eligible, comparable retained measurements, selection keeps the maximum
of the actual included greedy member and the other members. Final selection
uses the common confirmations, rather than mixing them with search scores. If
the greedy member has no valid common confirmation, its final comparison is
unavailable. This selection property does not guarantee beating an independent
stochastic GEAK run or make timing noise disappear. Choosing a global best also
does not promise that it wins on every individual case of a multi-case objective.

## Development time and scale

EVO members receive continuing-development guidance: stock iteration, plateau
and target cutoffs become reassessment points, while actual time/resource limits
remain binding. The existing engineer timeout can be increased when the declared
budget supports it. Longer sessions and more members are experimental settings,
not established improvements.

Increase concurrency based on observed usable feedback and delivered gains.
Track GPU completion throughput, queue waiting, unfinished jobs, useful revisions
after feedback and total model cost. More producers can hide CPU/model work;
they cannot increase the GPUs' evaluation service capacity.

This option is implemented at the public `kernel_workflow` entry point. Existing
end-to-end routing that calls `kernel_lane` directly does not enable a population
merely by receiving a `lineages` argument.
