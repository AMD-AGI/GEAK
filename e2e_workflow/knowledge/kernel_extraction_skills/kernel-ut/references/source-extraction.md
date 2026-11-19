# Source extraction and UT validation

Use with the [UT skill](../SKILL.md) and
[task authoring guide](task-authoring.md). Actual selected-kernel source is
required; an op-only library benchmark is a different deliverable.

## Editable boundary and dependencies

Resolve the live callable and branches on the current stack and served shapes.
Distinguish editable implementations, installed dependencies, unchanged
fallbacks and arch-gated paths. Copying a dispatcher does not make every native
child editable; a verified whole-op seam may still support authored replacement.
Preserve fused-operation semantics and disclose work outside the seam.

Inspect source and exercise representative branches in a clean probe process
inside the serving Docker container, or the equivalent deployed environment,
with its installed packages, resources and accepted overlay. Resolve actual
imported module files there and correlate source dispatch with device launches.
A `sys.modules` delta helps discover dependencies but misses preloaded modules
and unexercised lazy imports. Record original paths, source hashes, image/package
versions, resource dependencies and generation parameters in a task-owned
manifest linked from `meta.kernel_extraction`. Recheck after stack changes.

For marker coverage, resolve both the defining module and the reference the
caller actually holds. `from module import fn`, compatibility import paths and
registered dispatchers can retain a different callable after a module attribute
is patched. Resolve compatibility paths using the caller's import form before
installing markers; mark cached call-site aliases explicitly where needed.
An installed marker with zero calls does not establish that the function is
absent. Reuse existing standalone probe traces when source, inputs and dispatch
remain compatible, but regenerate a machine selection verdict from their raw
events. A prose probe inventory cannot be copied into `candidate_targets_tested`.
Keep the original serving verdict and label supplemental replay evidence and
its case/branch coverage separately; exclude no variant without a disposition.

When a deeper launcher appears to run sibling kernels, inspect the host launch
correlations before changing the extraction boundary. A fused custom CPU op
can give several different GPU launches the same `External id`. That shared id
does not prove that a nested GEMM1 launcher also launched GEMM2. The maintained
selector uses per-launch correlations when present and only falls back to a
fully contained CPU-op External id for device rows without a correlation.
Repair attribution defects in the shared selector with a regression; never
rewrite its verdict's `ok`, member counts or depth fields in a task.

Preserve module boundaries where supported by `candidate_bind`. If bundling is
needed, generate reproducibly: use AST spans for multiline and nested imports,
make renames scope-aware, audit global/import collisions, and preserve external
imports and initialization order. Repair package-relative resources before
import-time constructors consume them. Run `compile()`, import and minimal-call
smokes in the serving environment; parsing alone misses runtime errors.

## Structural state

Cases are generated from seeds at handoff shapes; the UT never captures. Routing,
page tables, cache objects and implicit state read by the callable are reconstructed
from handoff metadata, or from a profile structural fixture when reconstruction cannot
preserve the observed semantics (see [profile capture](../../kernel-profile/references/capture.md)).
Resolve implicit-state getters from the selected callable's actual globals or imports,
rather than a guessed package path, and replay one small real case before building the
full case set. Offline fakes cannot establish compatibility with live object APIs.

## Baseline and candidate execution

The task layout, `meta.target_callable` / `meta.candidate_bind` and the overlay mechanics
are in [task authoring](task-authoring.md). Two rules belong here: cases reach the declared
callable through the ambient overlay, never by importing candidate files directly, and only
`kernel_src/` is editable during optimization.

Keep `assert_legs_differ`, then prove execution below the outer wrapper for
each claimed editable branch. A function-local import, cached alias or registry
can send calls back to the installed library while names, grids and outputs
still match. Use untimed evidence from the actual invoked editable leaf, or a
reversible marker in a diagnostic copy with predeclared cases that must engage.
A locally resolving wrapper or an inspected but uncalled function is insufficient.
Use equivalent entry/launch proof for native or JIT code. Restore instrumentation
and rerun normal identity/correctness checks before freezing the task.

Make this an untimed task check at extraction smoke and after candidate binding
or source changes. For each editable branch, record actual invocation evidence
linked to the candidate source through `_cand_overlay`; fail if that branch
still executes baseline code. Installed dependencies and declared unchanged
fallbacks remain allowed. An import-path difference or correct output alone
must not pass this check. Keep proof artifacts in `meta.kernel_extraction`.

## Cases and dispatch fidelity

Build cases from the observed joint distribution. For each `sig`, retain phase,
operand/state signature, baseline kernels, full grids, calls per phase instance,
branch identity and evidence source. Preserve quantization/scales, preshuffling,
ragged metadata and observed routing where relevant. Target and draft models
need their own evidence; a layer count alone does not establish call frequency.

Use each shape's deployed reference. One anchor's kernel cannot validate a
whole sweep; latency resemblance cannot identify operands or dispatch. Where
Kineto summaries omit grids, read device events in Chrome trace `args.grid`,
separating host launch events from device execution. Compare full dimensions,
not block-count products, and combine launches when one grid is ambiguous.
Resolve `ns::op` through its registration/`torch.ops` when needed and verify the
actual signature and math; do not replace unresolved calls with `torch.mm`.

The unmodified extraction should preserve baseline dispatch and output behavior
on equivalent inputs through the shared harness's separate legs. Expect bitwise identity
for deterministic unchanged code; where it does not hold, measure the real floor (repeated
baseline outputs on one fixed set of inputs) rather than asserting a contract — on this deployment one MoE gemm
is byte-exact across repeats while its `epilog='atomic'` sibling carries a genuine 2-ULP
accumulation-order noise floor, and only a measurement tells them apart. Optimized code may
change names/grids/counts: validate its intended execution independently while preserving
the math, seeded random correctness and deployment graph/compile checks.

Record every requested case and its missing/failure reason with the call/time mass
attached. Diagnostic fallback timings and nonfinite sentinels cannot score.

## Timing and table validation

UT qualification does not time (`--validation-mode correctness`); these rules govern the
performance mode the optimizer runs. After identity matches, compare per-case UT baseline
and deployment latency.
Match graph/compile mode and cache residency; rotate weights/KV buffers only
when deployment evidence calls for it and within the memory budget. Explain
shape-dependent fidelity gaps before projecting gains.

Keep the shared timing primitives, separate-process lifecycle and repetition
budget. When noise calibration is needed, use identical controls on the same
allocation with matching code hashes, inputs and lifecycle, outside scored
cases. Report absolute and relative spread per family/case; distinguish
systematic run shifts from random noise. Exclude nonfinite values from statistics
while retaining missing-case records. Keep identity probes and verbose dispatch
logging outside timed regions; isolate probes that can abort natively by process.

For table overrides, inspect the effective runtime table and actual lookup keys,
including padding and solution parameters. Cached logs can be silent, and
lookup results can be overridden downstream; verify final launches. If overrides
replace a table, premerge with the effective table, preserve all untargeted keys
and values, and probe changed and unchanged branches. Numerical checks remain
required because choices such as split-K can alter reduction order.

<!-- Standalone adaptation, 2026-10-12: local tools and agent-owned stages. -->
