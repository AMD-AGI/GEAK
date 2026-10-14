# Unified indexer UT

Use when selected kernels belong to an indexer score/selection chain. Read
the supplied delivery's latest handoff and source boundaries, then verify the
current serving implementation. GLM 0922 decode and prefill joint tasks are
examples, not current source, shapes, weights or validation results.
Use the [delivery contract](delivery-contract.md) for exact return fields,
file/hash transport and preflight against the original queued `KERNEL`.

## Boundary and package

The workflow can invoke `AUTHORING_STAGE=joint_bootstrap` before full extraction.
In this stage, write a minimal case adapter and executable replay, then pass one
real case through each declared installed phase entry. Use existing valid
captures; inspect additional source to resolve concrete replay errors. Leave
full workload weighting, candidate leaf builds and negative controls for
`AUTHORING_STAGE=complete_joint_ut`. Return the requested bootstrap artifact
with per-phase replay results; it is not a complete UT pass. The full stage
receives `JOINT_BOOTSTRAP` and `PRIOR_TASK_DIR` and continues the same task.

Keep a separate authoritative oracle for each phase. Run selection with
`--task-dir` pointing to the corresponding phase capture directory: selection
promotes one oracle and recursively reclaims the other process-local captures
below that directory. Running it against the whole joint task would discard
the other phases. Preserve bootstrap inputs across selection/binding retries;
fix markers using standalone replay and retire duplicates only after the
retained or replacement inputs replay successfully.

Deliver one task directory with one `python3 unittest.py` entry, one immutable
case manifest and baseline, and editable implementations for every claimed
branch. The task contains both served PREFILL and DECODE cases, with separate
per-phase results and one observed-call-weighted aggregate. Preserve the
deployed input/output contracts even if their signatures differ by phase.

The default boundary begins with the actual query/key operands, scales,
per-head weights and live metadata consumed by score, and ends with the final
selected indices. Include score, masks, ragged/chunk handling, page-table
gathers/transforms and top-k work that occurs inside that serving boundary.
An upstream plan computed once and reused across layers remains an input;
record its lifecycle instead of charging it once per layer. Include upstream
projection/RoPE/quantization/cache writes only when the requested boundary
includes them. Do not drop required glue because it is below the Top-5/3% rule.

Inspect implicit state getters as part of the boundary. A KV getter may launch
a gather or conversion rather than just return a tensor. A bootstrap may use
captured post-gather operands to establish replay, but final whole-call timing
must execute the deployed getter work on reconstructed state. Do not replace
it with a cached output while claiming the original callable's full cost.
Reconstruct a paged buffer from captured ragged K/scales when possible and
verify the native gather reproduces them; this does not require another model
capture. Record any deliberately narrower boundary and its cost scope explicitly.

Identify the live joint callable and all member launches, with per-phase
source and dispatch evidence. Prefer the deepest deployable boundary that
contains the **whole chain**. A deeper score-only launcher cannot satisfy a
joint request. Preserve each member's identity/binding proof; one marker at
the outer wrapper is insufficient. If no single deployed callable expresses
the boundary, document a composition adapter and verify it against the actual
serving call sequence. Do not claim such an adapter is an installed callable
or bypass existing selection/overlay checks; a missing framework capability
must be fixed and tested before declaring a deployable handoff.

Declare `KERNEL.joint_ut.phases[]` with a distinct `regime`, anchor
`device_kernel`, `required_device_kernels` (anchor plus companions), and the
structured `seam_candidates` for that phase. Use the deployed phase entry as
`target_callable`; it may differ between paged decode and ragged prefill.
The required members must occur together in each verified invocation. Do not
put mutually exclusive shape-dispatch variants in one mandatory member list.
Ordinary fused MoE extraction retains its existing whole-op/per-case dispatch
contract; it does not need an indexer-style joint declaration merely because
its source contains multiple GEMM variants.
Run `kernel_selection.py --device-kernel <anchor>` with repeated
`--required-device-kernel <companion>` options for each phase's captures.
The tool verifies every required member in every observed joint invocation and
only considers a deeper callable a replacement if it contains the entire chain.
Return each full verdict in `phase_selection_validations[]` with `regime`,
`target_callable`, `seam_candidates` and `selection_validation`, or its supported
`selection_validation_ref` file/hash reference. Keep the primary phase's
callable/verdict or reference in the normal top-level fields for compatibility.
Never abbreviate its phase verdict as `_same_as` the top-level object. The
delivery helper attaches original files and runs the production check.
The production gate checks all declared phases; a passing leaf verdict or a
missing prefill verdict cannot certify a unified task.

For different deployed phase entries, declare `meta.phase_target_callables`
as a regime-to-callable mapping and keep the primary entry in
`meta.target_callable`. `cases.py` selects only among these declared entries;
the overlay decides the implementation. `h.assert_legs_differ` checks every
declared entry on both legs. Use `meta.phase_regimes` with a complete regime
configuration per case regime when deployment graph/compile modes differ;
the shared leg runner applies each phase's measured mode and rejects a missing
phase entry. A module overlay can replace both methods of the same deployed
class without changing the shared timing or correctness code.

Keep the baseline equivalent to the current serving source and freeze its
hashes before candidate validation. Preserve the existing separate-process
baseline/candidate isolation and shared timing primitives. Copying old
`baseline_src`/`kernel_src` packages does not establish compatibility with the
current stack. In particular, 0922 prefill joint's shipped candidate already
contains an optimization; it is not a neutral baseline.

## Correctness and actual execution

The main check consumes final indices produced from this invocation's inputs.
Check validity, mask exclusion, page/ragged mapping, cardinality, duplicate and
padding semantics, and the score threshold at the selection boundary. Use
exact index-set semantics where order is unspecified; handle ties by permitted
equal-score membership rather than requiring one arbitrary reference order.
Judge ties and selection thresholds using independently validated reference
scores. Unchecked candidate scores cannot adjudicate their own output: a
broken score kernel that returns a constant would make every index appear tied.
Do not substitute a floating-point relative-error check for integer semantics.
Check every output row, including large prefill cases; chunk the assertions if
needed to bound memory. Preserve the deployed integer dtype and exact padding
sentinel. Invert page mappings only over valid context entries, so padded page
IDs cannot overwrite valid mappings. Check input mutation against the declared
contract and keep the baseline output alive until candidate comparison ends.

Retain independent score and top-k semantic checks to localize failures. They
do not replace joint checks: candidate score output must flow into candidate
top-k, with no golden/intermediate reference values exposed as candidate inputs.
Use random values and boundary cases in each phase, and replay graphs after
changing tensor contents in place so stale pointer/shape caches fail. Restore
mutated inputs according to the deployed lifecycle before each measured call.

Prove execution of the editable score branch, editable native/JIT top-k branch
and claimed glue in both phases. Run independent bad-source controls that
change score and top-k semantics; each must fail for the expected reason in
the real candidate path. An unrelated import/build crash is not numerical
rejection. Preserve raw outputs and failing evidence; restore uninstrumented
code before the final GPU run.

## Cases, timing and completion

Build a complete case manifest from the current observed workload before
reporting success. Record all positive-mass signatures and any reconstructed
inputs, sampled representatives or uncovered state. Missing requested cases
remain failures/incomplete coverage, never disappear by renormalization.

Time one invocation of the complete chain using the shared timer in deployment
mode. Do not sum isolated score/top-k medians or mix separately measured
speedups as a joint result. Return per-case baseline/candidate latency, per-phase
folds and a combined `sum(baseline_ms*calls)/sum(candidate_ms*calls)` using
one common call basis. Do not add joint and constituent costs to workload
coverage. Keep all correctness, canonicalization and instrumentation outside
the timed region unless the serving chain performs that work.

Here `calls` counts the joint entry, not its constituent launches. A chunked
prefill entry may launch score/top-k several times in one invocation. Reuse a
supplied phase/batch census and establish the joint-entry calls per phase
instance from source plus runtime evidence. Raw request ISL/OSL, nominal layer
count and configured acceptance length do not replace that census when caching,
batching, speculation or indexer reuse changes the calls.

After one runnable joint case proves the intended path, expand the manifest
and gates rather than repeatedly surveying source without executing code.
Completion requires the standard entry to run successfully, all requested
cases measured on both legs, per-phase correctness and deployment checks,
binding/bad-source evidence, and a task-local report with hashes and limitations.
A driver that merely invokes two old leaf UTs is not a unified UT. Kernel UT
completion still does not imply an e2e throughput result or full input-distribution
fidelity when the inputs were reconstructed.
