# Unified indexer UT

Use for score/selection chains, following the [delivery contract](delivery-contract.md).
Apply the joint boundary when delivering the complete indexer operation. Honor
an explicitly requested score-only/top-k-only or single-phase scope instead of
expanding it. Read supplied deliveries' latest handoffs and verify current serving code;
old GLM prefill/decode tasks can contain optimized candidates, not neutral baselines.

## Boundary and replay

Deliver one task with one `python3 unittest.py`, immutable cases/baseline and
editable claimed implementations across served PREFILL and DECODE phases.
The default scored boundary is actual query/key operands, scales, per-head
weights and metadata through score, masks, ragged/chunk handling, page-table
transforms and top-k to final indices. Include required glue regardless of its
selection share. Upstream plans reused across layers remain inputs; include
projection/RoPE/quantization/cache writes only when the request includes them.

Inspect implicit getters: a KV getter may launch a gather/conversion. A replay
bootstrap can use captured post-gather operands, but final whole-call timing
must reconstruct state and execute deployed getter work. Rebuild paged buffers
from ragged K/scales where possible and verify the native gather reproduces
them. Record a deliberately narrower boundary and cost scope explicitly.

Select the deepest deployed callable containing the whole chain. A score-only
child does not qualify. If composition is necessary, verify its sequence
against serving and do not present the adapter as an installed callable or
bypass selection/binding requirements.

`AUTHORING_STAGE=joint_bootstrap` requests a minimal replay adapter and one
real case through each installed phase entry. Keep compatible saved captures;
return per-phase replay evidence, not full UT PASS. `complete_joint_ut` receives
`JOINT_BOOTSTRAP`/`PRIOR_TASK_DIR` and adds complete cases, editable leaves and
all GPU gates in the same task.

Keep an authoritative oracle per phase. Run selection with `--task-dir` set to
that phase's capture directory: promotion reclaims other process captures
beneath it, so pointing at the whole joint task could delete another phase.
Preserve replayable inputs across retries; improve markers by standalone replay
and reclaim duplicates only after retained/replacement inputs replay successfully.

## Phase declarations and binding

Declare `KERNEL.joint_ut.phases[]` as in the delivery contract. Each phase has
its deployed entry, anchor and mandatory companions; mutually exclusive
shape-dispatch variants must not share a mandatory-member list. Fused operations
with alternative branches use whole-op/per-case dispatch instead.

Run `kernel_selection.py --device-kernel <anchor>` with repeated
`--required-device-kernel <companion>` for each phase. It verifies every member
in every observed invocation and permits deeper candidates only when they
contain the entire chain. Return full per-phase verdicts/references and the
primary phase's top-level callable/verdict; a missing phase or leaf-only result
cannot certify the task.

When entries differ, declare `meta.phase_target_callables` mapping regimes to
callables, with the primary in `meta.target_callable`. `cases.py` calls only
those entries; the overlay chooses their implementation. `h.assert_legs_differ`
checks every entry on both legs. Supply complete `meta.phase_regimes` when
phase graph/compile modes differ; the leg runner rejects missing entries.
Preserve separate-process baseline/candidate isolation and shared timing primitives.

## Integer semantics and editable execution

Check every output row, including large prefill cases (chunk assertions if
needed). Verify valid indices, masks, mapping, cardinality, duplicates, deployed
integer dtype and exact padding sentinel. Invert page maps only over valid
context entries so padding cannot overwrite valid mappings. Preserve permitted
input mutation and keep baseline outputs alive during comparison.

When order is unspecified, compare index sets; permit alternative tied indices
only at an independently validated score threshold. Candidate scores cannot
adjudicate their own ties: a broken constant score could otherwise pass.
Floating-point relative error is not an integer-selection correctness rule.
Retain independent score and top-k checks for diagnosis, while the main check
feeds candidate score output into candidate top-k without golden intermediates.

Test random values and boundaries in each phase. Replay graphs after in-place
input changes to expose stale cached state; restore inputs before measured
calls according to serving's lifecycle. Prove actual editable score, native/JIT
top-k and claimed glue execution per phase. Separate score/top-k bad-source
controls must launch and produce numerical rejection, not an unrelated crash.
Restore uninstrumented source and rerun the normal GPU gates.

## Timing and completion

Time the complete chain with the shared timer in deployment mode; do not sum
independently timed score/top-k medians. Keep diagnostics outside timing unless
serving performs that work. Return both latencies per signature, per-phase
folds and `sum(baseline_ms*calls)/sum(candidate_ms*calls)` on a common basis.
Calls count joint-entry invocations, not constituent launches: chunked prefill
may launch members repeatedly, while caching/speculation/indexer reuse may
reduce calls. Establish counts from the observed census plus source/runtime
proof, not nominal ISL, layer count or configured acceptance length.

Expand from a runnable joint case to the observed manifest, retaining requested
missing cases and their mass. Do not add joint and leaf costs. Completion needs
all declared positive-mass cases on both legs, correctness/deployment/binding
and negative-control evidence, and a hashed report with reconstruction/coverage
limits. A driver invoking two old leaf UTs is not a unified UT or an e2e result.
