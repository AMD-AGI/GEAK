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

**Common broken UT: opt imports the original kernel.** An `opt.py`/`opt/` entry
can resolve to a local file yet call an installed implementation via a top-level
import, function-local import, cached alias, registry or native/JIT loader. The
UT then compares the original kernel with itself; editing opt's copied kernel
has no effect. `assert_legs_differ`, a different wrapper path, matching launch
grids and correct outputs do not rule this out.

Before accepting a generated UT:

1. Follow the opt entry through every claimed editable GPU leaf. Make its
   imports, rebinds and build inputs resolve to the delivered candidate source
   under `kernel_src/`, including each joint/dispatch branch. Installed helpers
   outside the requested source scope may remain dependencies; the claimed
   candidate kernel must not silently fall back to the original implementation.
2. Run the real candidate leg through `_cand_overlay` in a fresh process and
   record untimed leaf invocation evidence tied to the current source/build.
   Keep `cases.py` shared and baseline imports independent of opt.
3. Temporarily change the actual candidate GPU leaf to produce a numerically
   wrong result on a covered case. Rebuild/re-JIT as needed and use a fresh
   process so cached modules, binaries or graphs cannot hide the edit. The
   launch must succeed, candidate correctness must fail, and the unchanged
   baseline must still pass. An import/build error or a marker only in the
   Python wrapper is insufficient. If the UT still passes, fix the opt binding
   before claiming extraction success; do not relax the correctness check.
4. Restore the exact clean source, refresh the candidate build and rerun the
   normal UT. Keep the mutation, source hashes, commands and both failure/pass
   logs. Test each claimed editable leaf and repeat after source or binding
   changes; final delivery contains clean code only.

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
