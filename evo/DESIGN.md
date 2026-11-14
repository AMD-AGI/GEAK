# From archive-assisted search to persistent evolutionary lineages

## Status and terminology

The current implementation is PR601 revision `f9a29a6`, based on GEAK main
`a48ac8be`. It adds local archive-backed parent/donor selection to one shared
kernel worker. It does **not** yet implement the multi-lineage design below.
All population diagrams are proposed behavior, not experimental results.

The ordinary kernel lane has one incumbent: the best promoted source so far,
not necessarily the last round's winner. The planner's archive selection is
agent-chosen, not random. A lower-scoring parent can still have a better
descendant, so ranking parents does not rank their future development potential.
Nevertheless, an archive alone does not guarantee sustained development of
several alternatives. A persistent population is the next architectural step.

GEAK main already has relevant building blocks: parallel child workflows in its
kernel bakeoff dispatcher, and per-backend continuation state plus cross-backend
knowledge in E2E deep mode. These are existing main capabilities, not EVO novelty.
The proposal must be compared against those strong controls where applicable.

## Implemented single-lane comparison

```text
GEAK ordinary kernel lane             Current main + EVO kernel lane
---------------------------------    ---------------------------------
Setup / optional author              Same
Analyze / benchmark / profile        Same
Optional research / warm start       Same
             |                                    |
             |                       Capture post-preparation FOUNDER
             |                                    |
        Local round r                         Local round r
             |                                    |
TechLead: choose directions           Same + query archive, choose parents
             |                                    |
Each engineer starts from            Each engineer starts from selected
current canonical                    incumbent/founder/archived source
             |                                    |
Iterative engineering                Same + donor references
             |                                    |
Independent candidate verification   Same, preceded by custody checks
             |                                    |
Round barrier                        Round barrier; record best/development
             |                                    |
Integrate current-round candidates   Same + integrated-patch custody check
             |                                    |
Promote one incumbent / reprofile    Same
Memory / local stall / cleanup       Same; recorded source copies survive
             |                                    |
          Next round                           Next round
             |                                    |
Report / final validation            Same + archive/error summary
```

Both already have specialist/deep engineers, their iterative loops, current-round
integration, hypothesis memory, knowledge and tools. Current EVO adds explicit
retained-source reuse, not those existing capabilities. Donor references are not
proof of recombination. The archive stores reported evidence, not independently
authenticated measurement certificates.

## Proposed population: several complete existing workers

Prefer bounded persistent islands over multiplexing several planners through the
current worker's single mutable canonical/history/profile variables. Execute the
same worker script several times with isolated state. Do not copy its source,
introduce another host or initially refactor all shared phases into a new engine.

```text
Existing standalone dispatcher OR existing E2E parent
          |
          +-- Validate task contract and admitted population resources
          |
          +-- parallel: same kernel_lane.js, separate invocation state
          |       |
          |       +-- A: complete worker invocation, local incumbent/state
          |       +-- B: complete worker invocation, local incumbent/state
          |       +-- C: complete worker invocation, local incumbent/state
          |                 |
          |         Immutable peer checkpoint exchange
          |         Agents request relevant sources/evidence
          |         No writer edits a peer's live workspace
          |
          +-- Collect eligible exact-source finalists
          +-- Common selection measurements, freeze selected source
          +-- Fresh independent final evaluation
          +-- Deliver qualified champion or explicit incomplete/failure
```

The global champion is a delivery candidate, not the compulsory parent of every
island. A successful A must not overwrite B and C. Multiple independent workers
without effective source exchange remain a parallel-search control, not evidence
that crossover or migration helps.

### Host and state placement

```text
Allowed:  dispatcher -> parallel sibling kernel workers
Allowed:  E2E parent -> parallel sibling kernel workers
Avoid:    E2E parent -> population workflow -> kernel workers
                           extra nesting level
```

Reuse the existing host's `parallel` and `workflow` primitives at the parent that
already owns kernel dispatch. An already-nested worker cannot secretly create
another workflow level. Population settings must be forwarded or rejected at
every entry point, never silently ignored. Population size one and EVO-off need
explicit compatibility tests against their preceding behavior.

Each island owns distinct evaluation, canonical, output, cache and `state_dir`
paths. Main's continuation hooks already retain best source and history; reuse
them rather than adding a second general memory database. They are not by
themselves a crash-safe population checkpoint or a lifetime budget ledger.

Persist the island's original founder, archive identity, parent/donor ancestry,
spent credits and deadline across resume. The current archive initializer binds
its first canonical/source: reopening with a different resumed canonical cannot
silently redefine FOUNDER or discard prior entries. Resume needs an explicit
validated archive-reopen/import contract, not another fresh initialization.

Required before implementation validation: resume restores one committed state
generation containing incumbent/development source references, associated history
and lifetime accounting. It does not re-author unless explicitly reseeded. A
timed-out invocation must be stopped or prevented from publishing before another
invocation writes the same island's state. Promise timeout alone is not settlement.

After population final selection, return the existing single-kernel result shape
to the enclosing caller, with any population provenance in a separate optional
field. E2E still overlays one qualified selected implementation and performs its
own serving validation; it must not receive a list where it expects one patch.

The first implementation may repeat preparation in each worker, just as strong
independent-main controls do. Count that cost. Sharing immutable prepared inputs
is a separately validated optimization, not permission to share mutable state or
reuse an unqualified baseline. Repeated source code is unnecessary either way.

### Each island's state and local round

```text
Island B state
  Identity: run / island / task contract / editable source scope
  Local incumbent: exact source + qualification/measurement references
  Development tip: exact unfinished source + explicit unverified status
  Existing memory: question, insights, history, next investigation
  Profile/evidence: exact source and measurement context to which it applies
  Accounting: spent/reserved resources, local progress, stop reason
  Local archive: older sources, reports and source custody records
          |
          v
Read B's state; if exchange is enabled, query completed peer checkpoints
          |
B's TechLead selects a continuation or an explicit reopening/transfer
          |
Prepare private engineer sources; freeze parent/donor identities
          |
Existing engineer loops -> source capture -> existing verification
          |
Existing within-island integration, when eligible
          |
Update B's incumbent against B's incumbent, not the global champion
Retain unfinished development separately; update B's memory/progress
          |
Publish an immutable completed checkpoint; continue B's next round
```

State distinctions are mandatory: local incumbent, unfinished development tip,
archived source and global finalist are not interchangeable. A local improvement
below another island's best is still progress. An unfinished tip may be used for
engineering but cannot become a qualified incumbent or final by being retained.

The local TechLead selects which unfinished source, if any, becomes the continuing
development tip. Other retained branches remain addressable. This is additional
behavior to implement: main's `STATE_DIR/best` is not an unfinished-tip selector.

Before publishing a local promotion, successful materialization must match the
evaluated source. Failed application preserves the previous incumbent and its
score; changed bytes require new qualification. The same identity rule applies
when selecting and materializing the population's final result.

Profiles must match their actual source. Missing or stale profiles stay unknown;
the latest global champion's profile must not silently condition every island.
Restarting a worker must not reset spent credits or turn one lineage into a new
independent experimental replicate.

### Selection, exchange and survival

- Active island identity persists across rounds; its default development path
  stays local. The agent can explicitly reopen an older source or request a donor.
- Initial approaches can be proposed by the existing planning role. When testing
  exchange, controls receive the same initial planning/evidence opportunity.
- An agent can propose retiring/reseeding a stalled island. Record the reason,
  preserved state and consumed resources. Do not replace all islands with global
  top-scoring clones or pretend different IDs prove useful diversity.
- First test fixed, declared per-island budgets. Adaptive resource allocation is
  a separate policy, not an unaccounted advantage of the population treatment.
- Publication is transactional and source-addressed. A packet binds owner/run,
  task/oracle/case/weight contract, source scope and exact source, with originating
  agent findings, limitations, measurement references and qualification status.
- Agents query complete compatible packets and choose whether to request them.
  Imports become local retained objects with provenance; no peer-directory alias,
  mutable path, partial packet or silent fallback to a different source.
- Imported reports are evidence to investigate, not transferable qualification.
  Any descendant must pass the recipient's unchanged verification gates.
- A named donor, source match or improved score alone does not prove its semantic
  contribution. Trace source exposure, implemented contribution and tested path.

Agent-returned JSON and local helper receipts alone do not establish that
publication, import or verification executed. Claims require actual tool-execution
records linked to source-bound artifacts and applicable independent measurements.

### Timing, resources and final selection

```text
Time ------------------------------------------------------------>
A: [local round 1][local round 2][local round 3] ... [A final]
B: [long local round 1][local round 2] ...          [B final]
C: [local round 1][local round 2][local round 3] ... [C final]
          |              |
          +-- immutable completed checkpoints --> agent-requested reads
```

There is no new population-wide round barrier: workers have their own local round
boundaries. The current within-worker engineering/verification barrier remains.
Worker concurrency does not itself implement the separate one-batch-ahead
engineering proposal. Preserve existing deep-round exclusivity inside each
worker; simultaneous workers require separately admitted resources.

Main's existing E2E deep scheduler still has between-burst wave barriers. Reusing
that scheduler preserves them; removing them is a separate scheduling change.
The diagram describes independent local rounds inside sibling worker invocations,
not a claim that every enclosing main scheduler is barrier-free.

Give each worker explicit GPU ownership and model-concurrency allowances. Do not
multiply the same full GPU pool or token allowance by the number of islands while
calling the total unchanged. Keep shared mutable caches isolated. A parked or
failed worker cannot donate its reservation until its processes have settled.

Local stalls compare with local progress. Lack of a new global champion alone
must not terminate improving islands. A population deadline prevents new work
while preserving admitted settlement/final evaluation obligations; it is not an
eight-hour outer watchdog that also consumes preparation and finals. All idle,
helper, transfer, preparation, failure and final costs remain in the ledger.

Compare finalists under the same task contract with a common declared selection
protocol, then freeze the selected exact source and obtain fresh final results.
Do not take the maximum of incomparable self-reported search ratios. Missing
required results stay incomplete; a fallback counts only when actually delivered
and qualified. The best observed member is not a guarantee against an unobserved
independent GEAK run.

Child workers must use `apply_to_original=false`. Final delivery must reproduce
the complete selected source tip from frozen ORIGINAL, including prior bursts and
imports, rather than returning only a delta relative to the latest resumed Git
root. Reconstruct and check that complete artifact before final evaluation; no
concurrent worker is allowed to overwrite the user's original task.

## Illustrative source progression, not measured results

All start from S0. Numbers are invented speedups against one common ORIGINAL.
Rows are aligned for explanation; local round numbers are not global barriers.
Assume the relevant C2 checkpoint is available before B plans its third round.

```text
                 Local round 1       Local round 2       Local round 3
A: tile work     A1 1.12x ----------> A2 1.15x ----------> A3 1.18x
B: pipeline      B1 1.04x ----------> B2 1.09x ----------> B3 1.24x
                                                          ^
C: data movement C1 1.06x ----------> C2 1.11x ---- ref. ---+
                                           \-----------> C3 1.20x

Global leader         A1                  A2                  B3
B survives because its own lineage improves, despite trailing A.
The final selection compares the exact eligible A/B/C sources afresh.
```

The example uses improvements above main's default two-percent local promotion
margin. The donor arrow is a requested reference, not an established contribution.
This monotone example illustrates local-incumbent survival and proposed donor
reuse; existing main deep lanes could also produce this trajectory. It does not
demonstrate unfinished-tip continuation, causal recombination or an EVO advantage.
Those require actual execution evidence.
