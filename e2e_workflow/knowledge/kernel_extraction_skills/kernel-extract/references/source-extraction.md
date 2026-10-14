# Source extraction and UT validation

Use with the [Kernel Extractor role](../../../../roles/kernel_extractor.md) and
[delivery contract](delivery-contract.md). Op-only library bakeoffs retain their
`extract_op` contract without fabricated source trees.

## Live source and call boundary

Inspect the current serving container, installed packages and accepted overlay.
Identify actual imported files, dispatch branches, resources/build parameters
and their source/image/package hashes. A `sys.modules` delta misses preloaded
and unexercised lazy dependencies. Separate editable implementations, installed
dependencies, unchanged fallbacks and architecture-gated paths; a copied
dispatcher does not make native children editable. Preserve whole fused semantics.

Markers must cover the reference the caller holds, including `from` imports,
compatibility aliases and registered dispatchers. Zero marker calls do not prove
a function is absent. Reuse compatible standalone traces but regenerate machine
verdicts from raw events; prose cannot populate `candidate_targets_tested`.
Keep supplemental replay and original serving evidence separately scoped.

Joint UT verification correlates each member with its own host launch; a shared
parent External ID cannot prove that a leaf contains the complete chain. Use
`--required-device-kernel` for every companion and retain failed verdicts.

Preserve supported module boundaries and import order. If bundling is needed,
use scope-aware transformations, preserve external imports/resources and run
compile, import and minimal-call checks in the deployed environment. Parse-only
success does not prove runtime compatibility.

## Bounded, replayable captures

Trace the effective adapter before launching. `REPEATS=0` does not disable an
outer warmup, and the AgentX client may still replay its configured full duration.
Use an explicitly bounded capture client, skip outer warmup, and set its
`CLIENT_ADAPTER` path: an inherited adapter can override `BENCH_CLIENT`. Probe
counts and throughput never replace the measured baseline's distribution.

For methods, serialize required `self`, metadata and implicit cache state.
Compact referenced pages with an explicit mapping where practical; a full buffer
within budget is a valid first replay. Resolve getters from actual callable
globals/imports, validate required fields, then run one small real case before
expanding capture. `oracle_complete` proves persistence, not replayability.
Keep the old usable capture until its replacement replays; local compaction or
marker improvements alone do not require relaunching the model.

Custom serializers must include hidden tensors in the pre-clone byte estimate
and check actual snapshot bytes. Changing `_snapshot` alone bypasses capture
limits. Test a budget-skipped call followed by another call: a cache keyed only
by `id(args)` can reuse stale state after tuple deallocation. Retain the identity
object or clear the plan at the call boundary. Distinguish capture-added GPU
work from the deployed member-launch sequence.

Shared capture storage deduplicates only exact CPU snapshot content, not argument
names, pointers or version counters. Different layers/weights and reused
activation buffers can change values without those identities changing. Budget
new values conservatively before cloning, and test different tensors, in-place
changes and larger same-name arguments. Preserve skip reasons and failed evidence.

The stock capture module has singleton installation state. Multi-target methods
need separate module instances or a tested adapter with independent phase state,
flushes and budgets. Prove a replay before assuming multi-target capture works.

## Baseline, cases and candidate proof

Use the existing `kernel_src/`, frozen `baseline_overlay`/`baseline_ref/*.orig`,
`cases.py`, oracle and shared harness/leg/overlay helpers. Declare
`meta.target_callable` and `meta.candidate_bind`; cases reach only declared
callables through the ambient overlay, never by importing candidate files.
Only `kernel_src/` remains editable during optimization.

Keep `assert_legs_differ`, then prove actual editable-leaf execution. Local
wrapper paths, correct outputs and matching grids cannot exclude fallback to
installed code via cached aliases, registries or function-local imports. Use
untimed leaf invocation evidence or reversible diagnostic markers, linked to
current candidate source through `_cand_overlay`. Test every claimed editable
branch; explicitly out-of-scope dependencies/fallbacks remain allowed. Restore
instrumentation and rerun normal checks before freezing delivery. Recheck after
binding or source changes, including native/JIT paths.

Each observed signature retains phase, operands/state, baseline kernels, full
grids, calls per phase instance, dispatch and evidence source. Preserve scales,
preshuffling, routing and ragged metadata; target/draft paths need separate
proof. Read Chrome device `args.grid` when summaries omit it, compare full
dimensions rather than block products, and combine launches when grids are
ambiguous. Resolve registered `ns::op` through its real API; substituting
`torch.mm` cannot establish an unresolved op's signature or semantics.

Unchanged extraction should reproduce baseline dispatch and outputs on equivalent
inputs. Expect bitwise identity for deterministic code; otherwise retain the
justified numerical contract. Optimized code may change launch topology, but
still needs independent execution, oracle/random correctness and deployment
graph/compile/input-refresh checks. Record missing cases and their mass; nonfinite
sentinels or diagnostic fallback timings cannot score or justify renormalization.

## Timing and runtime tables

Compare UT baseline latency with matched deployment mode/cache residency. Rotate
weights/KV only when observed behavior calls for it and within memory limits.
Explain shape-dependent fidelity gaps before projecting gains. Keep shared timing,
separate-process lifecycle and repetition budgets; do noise calibration outside
scored cases with matching inputs/code/environment. Report absolute/relative
spread, retain missing-case records and keep diagnostics outside timing.

For table overrides, verify actual lookup keys, padding/solution parameters and
final launches; cached logs or an intermediate lookup may hide downstream
overrides. Premerge replacement tables with the effective table, preserve all
untargeted entries and probe changed/unchanged branches. Numerical checks remain
required, including reduction-order changes such as split-K.
