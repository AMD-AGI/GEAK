# Triage — compile, lowering, correctness, runtime and integration failures

**What this chapter decides.** When a candidate fails — it does not compile or lower, it compiles
and then hangs, crashes or returns wrong values, or its integration breaks — this chapter decides
what to inspect first, what to roll back, and which of three buckets the failure belongs to:
**retryable** (keep working), **scoped ceiling** (defer that one task to the user, keep working
elsewhere), or **global blocker** (halt). It also covers the case where the evidence for a claim
does not exist (the missing-documentation protocol). A "correct but slow" candidate is **not** a
triage case — go back to `profile.md` and the three-evidence loop in `climb.md`.

Normal authoring rules live in `../gluon/imports-and-launching.md`, `../gluon/layout-reference.md`,
`../gluon/memory-reference.md`, and `../gluon/matrix-reference.md`. Compiler-behavior insights
(micro-edits the compiler already does, `tl.assume`, `tl.trans`, the scaled-helper dtype trap,
async-as-layout) live in `../tile-programming/compiler-contract.md`. Per-target support lives in
`../hardware/capability-matrix.md`. Lowering failures that are known wrong-code or platform bugs
are in `../pitfalls/platform-known-issues.md`; expressions that silently do not lower as intended
are in `../pitfalls/negative-patterns.md`.

In GEAK, `kernel_workflow` owns the round loop: the deep_engineer reports a global blocker or a
deferred task in its `worker_result` (status + notes) and `report.md` — tech_lead plans around it and
Director arbitrates — rather than halting anything itself. Its own round log (`records.md`) and close
(`close.md`) carry the same record.

## Debug / recovery order

1. Roll back the newest structural layout/memory/matrix change before sweeping launch parameters.
2. Confirm Triton version (method baseline = upstream Triton 3.8.0; 3.6/3.7 differences are
   downgrade notes), backend, and target architecture (gfx950 main line / gfx942 downgrade).
3. Confirm launcher, ABI, and output-feeding path.
4. Confirm layout construction location and `gl.constexpr` arguments.
5. Confirm memory-path dtype, offset unit, fallback, and masks.
6. Confirm broadcast parent layouts and slice lineage.
7. Confirm matrix result layout, operand layouts, instruction shape, accumulator dtype, and store
   layout.
8. Reduce the candidate to one executable subpath; add shared memory / async only after the
   simpler path is correct.
9. Classify the failure: toolchain, repo gate, backend/lowering, API/layout, correctness, or
   performance/integration; return to the matching authoring reference before retrying.

## Symptom -> inspect -> fix

This table is **compile / lowering / layout** time. For a **runtime** failure after a clean
compile — a deadlock / launch-timeout, an illegal access / context poison, or a wrong result
(esp. row/tile-stripe or run-to-run race) — use the async-handoff worksheet
(`## Async-handoff debug worksheet (runtime)` below) first, then return here to classify.

| Symptom | Inspect first | Typical fix |
| --- | --- | --- |
| `arange()` missing `layout` | leftover `tl.arange` or layout-less `gl.arange` | replace selected index path with explicit layout-aware Gluon |
| `cannot convert SliceLayout(...) to tensor` | runtime layout objects inside `@gluon.jit` | move layout construction to host or preserve source-proven constexpr |
| broadcast dimension mismatch | parent layout + slice lineage | regenerate 1D indices from correct parent `SliceLayout` |
| `expected expand_dims input to have a SliceLayout` | slice parent + expand axis | regenerate source index from matching `SliceLayout(axis, parent)` |
| `Layout mismatch in broadcast` | left/right distributed layouts | rebuild one expression around one shared parent layout |
| `expected broadcast left input to be a distributed_type` after `tl.dot` | partial migration boundary | do not feed `tl.dot` results into Gluon ops; keep plain or lower the dot through MFMA |
| helper not executed | host launcher + output feeding | launch the Gluon symbol from the measured wrapper |
| guessed helper import fails | helper naming + source module | locate the real helper; do not invent names (`## Missing Documentation Protocol` below) |
| buffer fallback has no dtype | fallback + pointer element type | typed `gl.full(..., ptr.dtype.element_ty, layout=...)` |
| buffer store dtype mismatch | stored value + destination dtype | cast before store |
| buffer offset verifier failure | offset dtype + layout | use distributed int32/uint32 offsets |
| buffer reads wrong addresses | byte-vs-element offset + scalar base. Buffer offsets are ELEMENT counts; an earlier revision of the memory-path guidance said byte, so a kernel written against it scales the offset twice (2x off on a 2-byte dtype, 4x on a 4-byte one) | drop any `* bytes_per_element` from the offset expression; rebuild pointer math; fall back to `gl.load` when needed |
| `buffer_load_to_shared` lowering fails | offset layout family, rank, dtype, unit, shared layout, consumer. **gfx950:** direct-to-LDS async copy is the main path. **gfx942 downgrade:** async direct-to-LDS supports only 32-bit elements and needs destination `order=[1,0]`; a padded destination fails translation | retry with a source-proven `DistributedLinearLayout` before a build ceiling (`../tile-programming/compiler-contract.md ## Async / buffer path failures are layout-first`); on gfx942 build the swizzled destination first, or fall back to sync staging |
| `convert_layout` input type failure | input is not a distributed tensor | convert only distributed tensors |
| `BlockedLayout size_per_thread` verifier failure | tile size, `threads_per_warp`, `warps_per_cta` | recompute layout from launch contract (wave64) |
| MFMA layout verifier failure | result layout dtype, `instr_shape`, version (4 gfx950 / 3 gfx942) | match target + version expectations |
| CDNA4 INT8 MFMA rejects fp32 accumulator | accumulator dtype | use int32 acc for direct `cdna4.mfma`, or keep plain `tl.dot` |
| JIT boundary mismatch | helper language + decorator | `@gluon.jit` for Gluon device helpers; `@triton.jit` only for separate Triton kernels |
| Gluon import missing | installed Triton + fallback policy | use supported imports; preserve fallback |
| AOT scratch failure | generated metadata + wrapper scratch policy | preserve fallback or redesign; compile success is not full runtime success |
| preshuffled GEMM wrong answers | `DistributedLinearLayout`, unshuffle sequence, K divisibility | preserve reshape/permute/trans order before tuning |
| `builtin.unrealized_conversion_cast` | newest index/layout rewrite | revert newest conversion and shrink the subpath |
| `scaled_upcast` rejects `bf16` though the message expects bf16 | type object vs string argument | pass a Triton dtype object (`tl.bfloat16`), not `"bf16"` (`../tile-programming/compiler-contract.md`) |
| scheduling helper missing (`sched_barrier`) | installed API / local source | classify as version/API blocker; switch direction (`../tile-programming/compiler-contract.md`) |
| `kpack is deprecated starting from gfx950` | stale compile option | set `kpack=1` or remove from new configs |
| scalar return dtype mismatch after a Triton update (`int32` vs `int64`) | wrapper/oracle assumes older scalar dtype | normalize return types at the wrapper boundary or update the oracle |
| multi-scope/multi-region kernel wrong on SOME shapes (large abs / near-1.0 cosine-diff) but passes others | a single combined capability flag drives all regions + a dummy pointer passed for a region that lacks the feature | use PER-region flags; a region without the feature must still get a VALID pointer/value (not a dummy) — a combined flag makes the kernel read garbage from the dummy pointer (e.g. a KV index read as a length), data-dependent so it passes small shapes and fails larger ones |
| transcription reports a layout `UNRECOVERABLE` (`scripts/ttgir_bridge.py recover --arch gfx950`) | which op/layout, and whether the champion TTGIR was dumped with the pipeliner active | probe build first (`kernel_workflow/scripts/kernel_tools/probe.py`), then re-dump the champion at `num_stages=1`; if still unrecoverable, record a **forced divergence** in the ledger or report `structure_suspect` (`transcribe.md`) — never a hand-written substitute |

## Correctness-failure magnitude (before blaming numerics)

Before dismissing a correctness failure as borderline numerics / split-K reduction-order noise,
read its MAGNITUDE: a near-1.0 cosine-diff or large abs error is a real bug (commonly a
control/flag/addressing bug such as the per-region-flag row above), not reduction-order noise —
only a value hovering at the tolerance is genuinely borderline. Also separate data-dependence
from run-to-run: deterministic-but-shape-dependent = a bug; varies run-to-run on the same input =
a race (use `benchmark-hygiene.md ## Determinism race-test (async / barrier / pipeline / layout changes)`).
And know the oracle's gate TYPE — a cosine gate can fail while abs/rel passes (`close.md`).

## First response

Roll back the newest structural change; reduce to one executable subpath; classify the failure;
return to the matching authoring reference. Do not fix verifier failures with arbitrary powers of
two, and do not add target-specific matrix/memory ops before a simpler path is correct.

## Retryable vs scoped-ceiling vs global (continue / defer / halt)

After the symptom-level fix above does not unblock, classify the failure into one of three
buckets and act — so the agent keeps doing the next-priority work instead of stopping:

| Class | Gate (reuse existing thresholds) | Action |
| --- | --- | --- |
| **A. retryable** (agent self-resolves) | perf noise / a win below the GEAK commit gate (`MIN_IMPROVE` = 2 %, or inside a wider measured noise band — `benchmark-hygiene.md`) ; regression that is a **resource cliff** (retry combined with a register-relief layer, `climb.md`) ; **stall** (try a different mechanism at 10/15 steps, stop the layer at 20) ; **crash-loop** same compile/correctness error **<3x** (roll back newest change, re-read source, change edit strategy, re-verify env) ; **soft ceiling** (3 profiles within 1% AND >=3 tuning steps) | retry / fall back along the next-task ladder (next mechanism -> next OPEN layer -> escalate tier); a soft ceiling **stops that layer, not the task** -> move to the next layer. Never pause for the user. |
| **B. scoped ceiling = needs user** (defer + continue in-scope work) | required toolchain/compiler feature absent AND it needs a compiler change while **LLVM co-tuning is NOT sanctioned** ; a **build / version switch** is required (`../hardware/capability-matrix.md ## Build switch permission` = ask) ; **wrong-result** with no in-scope fix (target/dtype unsupported on this build) ; **missing oracle / data / repo gate** for THIS bucket ; an **environment issue scoped to this subtask** the agent cannot fix | record the `needs_user` handoff (below), set the task **`deferred_needs_user`**, and continue any other OPEN work. Return `outcome: partial` with this in `deferred[]`. |
| **C. global blocker** (halt the run) | GPU/driver/env unavailable ; the repo will not build at all ; the oracle is missing entirely ; the harness cannot run ; the task is `blocked / deferred_needs_user` ; the round budget is exhausted | halt; emit the report with done + deferred + open in the deep_engineer's `worker_result` / `report.md` for GEAK's loop. |

A tile retune is none of these: changing the tile re-recovers every layout, so it is a
`resweep_request` (returned in the deep_engineer's result; tech_lead hands it to GEAK's plain tuning), not a retry.

**Profiler layer unavailable (locus-fix-then-escalate).** A profiler-layer degrade is triaged by
CAUSE, not accepted as a silent terminal state. Profiling goes through GEAK's entry
`kernel_workflow/scripts/profile_kernel.sh <gpu_id> <cmd> <out>` under
`kernel_workflow/scripts/gpu_lock.sh` (no inline `HIP_VISIBLE_DEVICES`), whose safe wrapper
(`kernel_workflow/scripts/kernel_tools/rocprofv3_safe.sh`) times out and degrades instead of
hanging; counter-slot overflow behaviour (replay / abort / hang) is version-dependent and is
handled by that timeout + degrade (`profile.md`).
- **Not a failure at all:** an RDNA4 target that is PMC-blind (rocprof-compute unsupported) is a
  **degrade**, and an absent ATT decoder (`ROCPROF_ATT_LIBRARY_PATH` unset or pointing nowhere)
  degrades the ATT layer — record the degrade in `caveats[]` and continue.
- **Locus-fixable** (`profiler_locus_mismatch` / `att_docker_boundary` / `sol_pmc_no_kernel_rows`
  with a container kernel; decoder present) = **A. retryable — run the profiler IN the kernel's
  container FIRST**. GEAK runs the profiler in the kernel's workspace, so a locus mismatch there is
  a GEAK configuration issue to report. Only when the kernel lives in a separate container (optional):
  `scripts/locus.sh` auto-wraps `rocprof-compute`/`rocprofv3`/ATT in `docker exec` when
  `TILE_KERNEL_CONTAINER` is set; `docker cp` artifacts out (`profile.md`, execution locus). A host-side-only
  probe over a container kernel is NOT a valid degrade; only a **co-located** probe that still
  fails is.
- **Genuinely unavailable after the in-locus run** (decoder truly absent + no container-local
  install / needs a host<->container mount / a supervisor hook) = **B. scoped ceiling ->
  `needs_user`** with the exact failing in-locus probe + fix hint.
- **RECURRENCE ESCALATION:** the SAME locus-fixable profiler cause degrading **>=2 consecutive
  rounds MUST raise the B handoff** — the per-round VERIFY loop has lost its ground truth (ATT is
  the per-instruction bubble-ownership source). The
  `../hardware/bound-class-signals.md ## Degraded-mode PMC discriminators (when ATT is unavailable)`
  fallback is sanctioned ONLY after this fix-or-escalate path, never a permanent silent default.

> **Mandatory: a falsifying probe precedes every B classification.** You may not defer a
> lever/feature to `deferred_needs_user` (B) on assumption. Before recording a scoped ceiling you
> MUST run the cheapest falsifying probe and record its exact error or number: *does the op
> compile / legalize?* (`hasattr(<module>, <op>)`; a one-line compile of the op), *does an
> alternative entry work?* (e.g. a different namespace — `cdna3` empty but `cdna4.async_copy`
> present; a different kernel entry), *is a "disabled" flag cosmetic?* (host wrapper default-off
> vs the atom actually not lowering), *what does a microbench say?*. `"unproven"`, `"assumed
> infeasible"`, `"the downstream flag defaults off"`, and `"the cdna3 namespace has no X"` are
> **not** acceptable terminal states — they are hypotheses that a probe must confirm or refute
> first. Only a probe that produces a real error/assert/`hasattr==False` on the target arch
> promotes a failure from A (retryable, keep probing) to B (defer). Stamp the probe command +
> output in the handoff below.

A scoped failure (B) is **rerouted, never a run halt**; only a global blocker (C) stops the run.
The bucket maps onto the existing `failure_class` enum (`records.md ## failure_class enum (trimmed)`):
A ~ `performance_noise` / `correct_but_slower` / `direction_unproven`; B ~
`environment_toolchain_blocker` / `api_or_layout_failure` (no in-scope fix) /
`backend_or_lowering_failure` needing a compiler change; C ~ a box-wide
`environment_toolchain_blocker` or `timeout_budget_exhausted`.

### needs_user handoff (for a scoped ceiling, B)

```text
task (config bucket + bound class):
error_class:                       # one of the B triggers above
what failed (one line) + evidence: # IR/asm/compile/correctness/env signal
falsifying probe:                  # exact command + output that promoted A -> B
user_action_required:              # e.g. sanction LLVM co-tuning / switch build / provide oracle/data / fix env
fallbacks_tried:                   # mechanisms/layers attempted before deferring
best_kept_for_this_bucket:         # the production baseline or best in-scope result held as fallback
resume_entry:                      # re-enter THIS task; what to re-run first after the user resolves it
```

The row is carried into `records.md ## 1d. Residual / Deferred-Task & Resume ledger`. After the
user resolves it, a **resume** run re-optimizes the deferred task and merges/updates the report
(`close.md`).

## Async-handoff debug worksheet (runtime)

Use this when a candidate **deadlocks, crashes the context, or returns wrong values** at runtime
**after a clean compile**. It is the **debug-time dual of the tile-op contract card**
(`../tile-programming/tile-op-contract.md`): `Roles = Scope`, `Storage = Layout`,
`Handoff = Handoff`, plus a debug-only `Lifetime` column. For compile / lowering / layout failures
use `## Symptom -> inspect -> fix` above instead; for "correct but slow" go back to `profile.md` +
the three-evidence loop, not here.

<!-- BEGIN portable core — byte-identical across the 4 tile skills (gluon / cutedsl / flydsl / tilelang). Keep in sync; everything below the END marker is this skill's per-DSL fill. -->

### Worksheet (fill before editing)

For the failing kernel path, write this down **before** changing any code:

| Column | What to record |
| --- | --- |
| **Roles** | the exact threads / lanes / wave / warp / workgroup that ISSUE each async op |
| **Storage** | the live location of each tile at each step (global / shared / registers) |
| **Handoff** | producer, consumer, signal object, arrival/wait count, signal order, and the fence/drain that makes the write visible |
| **Lifetime** | the earliest point each storage slot may be reused / read back / freed |

Roles / Storage / Handoff are the same three axes as the tile-op contract card; **Lifetime** is
the debug-only 4th column — a corollary of Handoff: a slot is free only once every consumer the
Handoff names has passed it.

### Verify the dumped IR/asm against the worksheet

Do **not** rewrite the kernel first. Dump the generated IR/asm and check it against the worksheet:

- role guards in the code match the **Roles** row;
- barrier / fence init precedes the guarded role branches (not nested inside one); on AMD grep
  `s_waitcnt` / `s_barrier`; on NV grep `mbarrier` / phase — do not apply NV mbarrier checks to
  AMD ISA;
- collective ops are **not** narrowed by a lane / warp / workgroup guard;
- wait counts / signal order match the **Handoff** row;
- buffer reuse, store drain, and any free/dealloc happen only after the **Lifetime** row allows.

### Runtime symptom map (skeleton)

Classify first, then use the per-arch drill-down (below the END marker):

| Symptom | Likely area | First check |
| --- | --- | --- |
| hangs / launch timeout | **deadlock** | signal count vs ready producers; barrier/fence placement; who participates in the loop advance |
| illegal access / context poisoned / later calls also fail | **crash** | restart the process; then alloc/dealloc order, OOB, collective issued by the wrong lane set |
| wrong values in row / tile stripes | **sync race or tile-index/ownership** | producer/consumer order; which role owns each stripe; deterministic-vs-run-to-run |
| `NaN` | descriptor / operand / uninitialized accumulator | layout/swizzle, accumulator init |
| finite but patterned-wrong | **stale / partially-visible data** | missing fence, undrained store, buffer reused before Lifetime allows |
| correct but slow | dispatch / resource (NOT a handoff bug) | go to the profiler + three-evidence loop, not this worksheet |

**Change ONE handoff at a time**, re-run correctness, and run the determinism race-test whenever
the change touched async / shared ordering.

<!-- END portable core -->

### gfx950 / Gluon fill

**Dump tools + scan strings.** `kernel_workflow/scripts/kernel_tools/dump_ir.sh` (TTGIR / LLIR /
amdgcn) + `kernel_workflow/scripts/kernel_tools/asm_loop_audit.py`. Grep the asm for:
`buffer_load_to_shared` (async copy issued), `ds_read` / `ds_write` (LDS traffic), `s_waitcnt` /
`s_barrier` (the Handoff signals), `v_mfma*` (DOT), `sched_barrier`.

**Deadlock drill-down (HYBRID — not TileLang-style auto).** Gluon has **no** `num_stages`
auto-pipeliner (in 3.8.0 no pass on the Gluon path consumes `num_stages`). Handoff splits into two
tracks on the card (`../tile-programming/tile-op-contract.md`):

1. **Authored buffer sync (FULL drill-down)** — you write the multi-buffer skeleton (the
   hand-written pipeline order: register prefetch → authored LDS ring → `warp_pipeline_stage`;
   `../tile-programming/pipeline.md`); wrong counts here are the **common** runtime failure mode
   once the pipeline layer lands. On gfx950 the ring is `async_copy` + `commit_group` /
   `wait_group`; **gfx942 downgrade:** sync staging (async direct-to-LDS only for 32-bit elements
   with destination `order=[1,0]`; a padded destination fails translation — build swizzled first),
   so the `commit_group`/`wait_group` checks below apply only where the async path is real.
   - `wait_group(N)` does not match outstanding `commit_group`s for that LDS buffer;
   - `nBuffers` ≠ stage depth (a 3-stage pipeline needs three buffers, not two);
   - independence rule violated — `DOT(k)` consumes the same-slot `LR(k+1)` / `AC(k+2)` before
     `wait_group` retired it
     (`../tile-programming/mental-model.md ## Latency vs throughput (the key distinction)`);
   - runtime buffer index (`smem.index(k % nBuffers)`) — use compile-time literal indices per
     `../tile-programming/pipeline.md ### Hand-built buffering rules (correctness + scheduling footguns)`;
   - `sched_barrier` / membar inside a branch only some lanes take;
   - **silent `buffer_load_to_shared` fallback** — Gluon runs no `CoalesceAsyncCopy` pass; an
     offset layout that is not already pre-coalesced on the LDS-fast dim (`canLoadDirectToLDS`:
     128-bit contiguous run, `vec == contig`) silently lowers to register staging + `ds_write`
     instead of direct-to-LDS. The kernel still runs but `commit_group`/`wait_group` no longer
     match real async traffic → wrong values or apparent hangs. Grep asm for `buffer_load ... lds`
     on the load path; if missing but `ds_write` appears, fix the offset layout coalescing first
     (`../tile-programming/memory-path.md ## Decision signals`).
2. **Compiler-owned interleave (TRIMMED)** — scheduling and register-allocation levers only affect
   hot-loop instruction scheduling; they do **not** use mbarrier/phase. Do **not** port CuTeDSL
   mbarrier/elect_sync deadlock checks here. A build that re-injected plain's pipeliner
   (`gluon_swp` / `patch_reinject` / `patch_async_reinject`) is a labelled `injected` diagnostic
   whose sync the pipeliner owns — it is not a track-1 debugging target and its numbers are never
   a win (`recover.md`).

Apparent hangs with a **layouts-only anchor** (no pipeline layer yet) are usually
compile/lowering — check `## Symptom -> inspect -> fix` first. Wrong values after pipeline edits
usually mean track (1), not missing LLVM sched.

**Crash / poison.** Restart the HIP process after an illegal access before testing the next fix (a
poisoned context makes later unrelated calls fail). Then check OOB GMEM/LDS offsets and an async
copy issued by the wrong lane set.

**Wrong-value (the common gfx950 case).** A row/tile-stripe mismatch that is **deterministic but
shape-dependent** is a tile-index / ownership bug; one that **varies run-to-run** on the same
input is a race — confirm with
`benchmark-hygiene.md ## Determinism race-test (async / barrier / pipeline / layout changes)`
(same input, N>=~40 launches, `maxdiff == 0`). Read the failure MAGNITUDE before blaming numerics
(`## Correctness-failure magnitude (before blaming numerics)` above). `NaN` points at
layout/swizzle or an uninitialized accumulator.

**IR-didn't-take.** If the asm is unchanged after a layout / async edit, the Gluon expression did
not lower as intended -> `../pitfalls/negative-patterns.md`; fix it before re-timing.

### Cross-links

- build-time dual: `../tile-programming/tile-op-contract.md` (`Roles=Scope`, `Storage=Layout`, `Handoff=Handoff`).
- compile / lowering / layout failures + the retryable/scoped/global classification: the sections above.
- race confirmation: `benchmark-hygiene.md ## Determinism race-test (async / barrier / pipeline / layout changes)`.
- correct-but-slow: `profile.md`, `climb.md`.

## Missing Documentation Protocol

Use this when the skill docs, operator-local source, and local Triton source do not prove an API,
layout, or lowering claim.

### Rule

Do not guess. Record the missing evidence and stop treating the claim as fact. (A falsifying probe
— `## Retryable vs scoped-ceiling vs global` above — is the way to turn a missing-evidence claim
into evidence.)

### Missing evidence categories

```text
missing_api
missing_layout_rule
missing_target_support
missing_benchmark_contract
missing_lowering_behavior
missing_runtime_integration
```

### Report template

```text
Claim:
What was checked:
Missing category:
Why existing evidence is insufficient:
Safe fallback:
Suggested doc location:
```

### Safe fallbacks

- keep the path plain Triton;
- use generic `gl.load` / `gl.store` instead of target-specific memory ops;
- stop at layout evidence instead of matrix lowering;
- preserve fallback or prebuilt selection;
- reduce the patch to one source-proven subpath.

### Where to add future evidence (owned files)

- API, layout, and matrix mechanics -> `../gluon/index.md` + the matching
  `../gluon/{layout,memory,matrix}-reference.md`;
- target-family or version details -> `../hardware/capability-matrix.md` /
  `../hardware/planning-constants.md`;
- compiler/lowering symptoms -> this chapter / `../tile-programming/compiler-contract.md`;
- benchmark contract gaps -> `benchmark-hygiene.md`;
- platform-sensitive constraints -> `../pitfalls/platform-known-issues.md`.

## Sources

Merged: `references/failure-triage.md` (whole), `references/debug-async.md` (whole, as
`## Async-handoff debug worksheet (runtime)`; the portable-core BEGIN/END markers kept verbatim),
`references/missing-doc-protocol.md` (whole, as `## Missing Documentation Protocol`).

Rewritten by the contract rules (no content dropped):
- Class A "perf noise / sub-2% win (`phases/harness.md` small-gain)" → a win below the GEAK commit
  gate `MIN_IMPROVE` = 2 % (a wider measured noise band only raises the bar); the harness
  small-gain repeat rule is in `benchmark-hygiene.md`.
- Profiler-unavailable triage: profiler entry restated as GEAK `profile_kernel.sh` under
  `gpu_lock.sh` with the safe-wrapper timeout + degrade; RDNA4 PMC-blind and an absent ATT decoder
  (`ROCPROF_ATT_LIBRARY_PATH`) are degrades, not failures; `scripts/locus.sh` scoped to the optional
  separate-container case.
- Run-mode split (GEAK-embedded vs the pack's own upstream `gluon-direction` agent spine) removed: blockers and
  deferred tasks go into the deep_engineer's GEAK `worker_result`; a `resweep_request` goes to tech_lead.
- `buffer_load_to_shared` row and the async drill-down: gfx950 main path + gfx942 downgrade
  (32-bit only, `order=[1,0]`, padded destination fails translation, swizzled first / sync
  staging).
- Added: `UNRECOVERABLE` row (probe build → re-dump ns=1 → forced divergence / `structure_suspect`,
  no hand substitute); tile retune = `resweep_request`; `injected` re-injection builds are not a
  debugging target; `result: partial` → `outcome: partial` (final-report v3).
- Tool paths: `scripts/dump_ir.sh`, `scripts/asm_loop_audit.py` → `kernel_workflow/scripts/kernel_tools/`.
- Headings demoted one level inside the merged chapter (`## Worksheet (fill before editing)`,
  `## Verify the dumped IR/asm against the worksheet`, `## Runtime symptom map (skeleton)`,
  `## gfx950 / Gluon fill`, `## Cross-links`, and the missing-doc `## Rule` / `## Missing evidence
  categories` / `## Report template` / `## Safe fallbacks` / `## Where to add future evidence
  (owned files)`) — text unchanged. File titles became section headings:
  `# Async-Handoff Debug Worksheet (runtime)` → `## Async-handoff debug worksheet (runtime)`;
  `# Missing Documentation Protocol` → `## Missing Documentation Protocol`;
  `# Failure Triage (compile / lowering / correctness / integration)` → chapter title.
