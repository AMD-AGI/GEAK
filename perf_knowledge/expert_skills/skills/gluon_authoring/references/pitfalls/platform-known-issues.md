# Platform Known Issues (gfx950 / gfx942)

Target- and backend-sensitive constraints that should not live in the core
workflow. Use `../hardware/capability-matrix.md` as the queryable source of truth
for gfx950/gfx942 MFMA, FP8, scaled-op, namespace, and correctness status; keep
this file for general platform-sensitive decision rules. gfx950 (CDNA4) is the main
line and gfx942 (CDNA3) the downgrade; mechanisms are stated for upstream Triton 3.8.0
with older minors as downgrade notes. Kernel-shape and method negatives (admission,
layout, attribution, inline asm) are the sibling page `negative-patterns.md`.

## Map of this page

| theme | section |
| --- | --- |
| scope | `## What belongs here` |
| lowering, evidence, arch and version sensitivity | `## Architecture and backend sensitivity` — evidence layers; gfx950-first generation carry-over; async copy misdiagnosis; scheduling helpers; version- and build-sensitive surfaces; environment and cache; numerics traps |
| benchmark validity | `## Benchmark-sensitive platform issues`, `## gfx942 / CDNA3 hard failures that affect benchmark validity` |
| toolchain pins | `## Toolchain-pin regressions: when the number you measured is not a ceiling` |
| RDNA4 client (GEAK: R9700-only) | `## RDNA4 client PMC availability (gfx1201 ISA; observed on R9700)`, ``## `int64_strides=false` regression on gfx1201 under CUDA graph`` |
| profiling infrastructure | `## Container-locus profiling defects (host ⇄ kernel-container split)` |
| use | `## How to use this file` |

## What belongs here

Only issues that can be stated generically: target-family differences;
architecture-sensitive matrix families; lowering risks tied to backend/compiler
stack; shared-memory constraints; cache/import/artifact behaviors that affect
benchmark validity. Convert one-off kernel stories into generic symptoms,
preconditions, or decision rules first.

## Architecture and backend sensitivity

Areas that may differ by target: wave-size assumptions (both gfx950/gfx942 are
wave64); supported matrix families and instruction shapes; memory-path legality
and profitability; Triton minor-version behavior in lowering.

Backend capability evidence is not a concrete API template. Treat backend /
library / compiler support as a reason to inspect local source and tests, not as
permission to add scheduler, shared-memory, or scaled-matrix features to a first
candidate. For Gluon, separate the layers:

```text
toolchain smoke: can a minimal @gluon.jit kernel compile and run?
repo production gate: does the target repo allow this arch/dtype/path?
backend lowering: does the selected layout/memory/matrix mechanism lower?
operator integration: does the path feed the measured output correctly?
```

Only the last two speak to mechanism viability. The capability boundaries below are grouped
by theme; within each, the gfx950 form comes first and gfx942 is the downgrade.

### Evidence layers: what a probe has and has not shown

- CDNA3 and CDNA4 can share some lowering paths, but CDNA3 evidence does not prove
  CDNA4 scaled behavior.
- Compile success is not correctness. A path can compile through a non-native
  namespace and still produce wrong results. Keep blockers scoped to target,
  dtype, API path, and software version.
- A gfx950-only or scaled source path should be locally probed on gfx950 before
  replacing it with a gfx942 workaround.
- gfx950/CDNA4 evidence must be tracked separately from gfx942 evidence; use
  `local-confirmed` in `../hardware/capability-matrix.md` only for locally-checked,
  scoped facts.
- **A namespace importing is still not the op lowering.** Availability claims of the form "the
  symbol is there" remain not evidence — `warp_specialize` is present in core `gl` on every
  version and aborts the pass manager on CDNA3. Compile a one-op probe on the actual target; just
  make sure the probe satisfies the op's contract, or it proves the wrong thing.

### gfx950 first: what does not carry down to gfx942 (or back up)

- **FP8 spelling differs by generation, and the wrong one is silently upcast.** gfx950 is OCP
  (`float8_e4m3fn` / `e5m2`); the gfx942 downgrade is FNUZ (`fp8e4b8` / `float8_e4m3fnuz`). A
  gfx942-oriented FNUZ format that reaches gfx950 may be upcast (to fp16, with only a warning).
  Verify the actual dtype lowering before interpreting FP8 performance, and take the spelling from
  `perf_knowledge/hardware/data/hw_constants.json` `fp8_dtype` for the `--arch` you pass.
- **A transcription result does not carry across CDNA generations, and the parts that break
  are not the ones a version sweep exercises.** Before quoting a same-generation result on
  another one, re-check at least: (a) the per-generation Gluon namespace on every buffer and
  MFMA builtin, plus the MFMA layout's version field — the newer namespace generally exists on
  older Tritons too, so this looks like a rename, but the accepted `instr_shape` sets differ
  and a changed K dimension pulls the tile and loop structure with it; (b) any arm that failed
  on an LDS/shared-memory limit, since the per-CU budget can differ by more than 2x and such an
  arm may simply start fitting; (c) the depth of any `num_stages`-style sweep (on plain) or ring buffer
  count (on the Gluon path, where `num_stages` is dead), for the same reason — 160 KiB/CU on gfx950
  against 64 KiB on gfx942; (d) counts of constructs that exist to dodge bank conflicts, since the
  bank count changes (64 banks on gfx950, 32 on gfx942). When re-validating layout recovery itself, assert that recovery still succeeds and
  round-trips **within** the new generation and stays stable across its Triton minors — *not*
  that layout digests match the other generation. MFMA layout digests **should** differ across
  generations, so asserting equality manufactures a false "the tool broke here".

### Async copy: available on both generations, and easy to misdiagnose as unavailable

`gl.amd.cdna4.async_copy` lowers from **stock** `gluon_to_ttgir` on both entry points
(`global_load_to_shared`, `buffer_load_to_shared`) at the direct-to-LDS widths each generation
supports — 128 or 32 bits per thread on gfx950, 32 bits only on gfx942 (where it also needs a clean
tiling and a destination `order=[1, 0]`, `../gluon/pipeline/marker-and-version-gates.md ## The gfx942
async-copy width gate — available, and narrow`). The contract is owned by
`../gluon/memory-reference.md ## Async Copy To Shared`; the traps, in the order they bite:

- **A lowering failure is a layout-contract question before it is a capability verdict.** The
  async-copy ops reject a request whose per-thread chunk or thread tiling does not satisfy the
  hardware contract, and they reject it with `failed to translate module to LLVM IR` /
  `unrealized_conversion_cast` — the same words a genuinely absent op produces. A probe that
  varies only the width, or only the dtype, cannot tell the two apart, and concluding "absent on
  this arch" from one is how a working path gets written off. The control that separates them is
  to hold the op, arch and width fixed and vary **only the tiling**: on gfx942 a 32-bit-per-thread
  `buffer_load_to_shared` fails when the threads do not tile the contiguous dimension
  (`4 threads x 1 elem` over a 64-wide dim) and lowers and runs bit-exact when they do
  (`64 x 1`). Same op, same arch, same width. Before recording an arch ceiling, satisfy the
  documented contract first (`../gluon/memory-reference.md ## Async Copy To Shared` states the
  per-thread byte floor and the layout requirement) and re-probe.
- **The per-lane width can be right and still be split across layout repetitions**, which is a
  second way into the same error text and the harder one to see. Each lane must make **one** access
  of a width in `perf_knowledge/hardware/data/hw_constants.json` `direct_to_lds_bit_widths`
  (`[128, 32]` on gfx950, `[32]` on the gfx942 downgrade); a layout that covers *more than the tile* repeats, and every lane then makes two
  accesses instead of one. Measured: a `BLK` covering `[64, 16]` over a `[32, 32]` tile repeated
  twice in N and failed, and the failure was read as the architecture refusing async copy. Check
  the layout's coverage against the tile, not just the arithmetic per lane. With the coverage right,
  both async entry points lower on **stock** `gluon_to_ttgir` with no pass spliced in (next bullet).
- **A coalescing pass legalizes; it does not enable.** `add_coalesce_async_copy` (spliced in by
  `patch_async_reinject.py`, or run by `gluon_swp.py` when it injects the pipeliner) makes an
  off-width or repeated access legal by adding a **bounce** — a cost, and one easily mistaken for an
  intrinsic cost of the async path. Read a bounce in the ISA as "the access was not native", not as
  "async is expensive here".
- **A padded shared destination forced through on an async arm can be a silent miscompile** — the
  fastest arm measured while returning NaN across nearly every element, with the identical sync
  kernel exact. Owned by `../gluon/memory-reference.md ### Shared-layout family + transpose-on-read
  (layout dependency)`: build the async destination swizzled first and check numerics by copy-back
  on any padded-shared async arm before believing its clock.
- **The falsifiable signature that async actually replaced staging is `ds_write == 0`** with a
  matching count of direct-to-LDS loads; sync staging pays a non-zero `ds_write`. Availability is not
  value — the bypass can still lose to sync staging (it measured slower on every minor on gfx942) —
  so A/B it. `scripts/pipeline_examples_cdna4.py` is the runnable check on gfx950
  (`../gluon/pipeline/async-ordering.md ## The async forms — gfx950 only, and the shape differs from A5`).
- **Async-copy offset layouts are version-gated.** Whether `buffer_load_to_shared`
  accepts a given offset layout depends on the Triton minor: an older minor may
  lower only `Blocked` / `Slice` offset layouts and **fail to compile** a
  `DistributedLinearLayout` async offset that a newer minor accepts. Gate the Gluon
  async path to the build that supports the needed offset family; on the unsupported
  minor fall back to a sync `convert_layout`. Probe per build, do not carry the gate
  across builds (`../hardware/capability-matrix.md`).

### Scheduling helpers and hints

- AMD scheduling helpers (`sched_barrier`, `sched_group_barrier`, `set_prio`) are absent from
  `gl.amd.cdna4` and `.cdna3` on every version, even when production kernels contain scheduling
  concepts (some import them behind no-op stubs). Treat the absence as a toolchain ceiling for the
  *inline* spelling and switch direction; the reachable forms are the `warp_pipeline_stage` marker
  and the LLIR level (`../gluon/pipeline/authored-overlap.md`, row B2).
- `warp_pipeline_stage` can be source-available but performance-negative — compile success is
  not a scheduling win, so require quick timing before expanding search. Read that as a gate on
  *this* kernel, not a verdict on the mechanism: it buys nothing on a loop with no LDS staging
  to overlap, or on a memory-bound one, and it is a first-class scheduling model on a loop that
  has both. The mechanism, its gates (Gate 0 is `num_warps >= 8`) and a skeleton are Gluon-only
  and live in `../tile-programming/warp-pipeline.md`; plain Triton reaches the same idea only through the
  automatic ping-pong pass. The two failure modes worth checking first are
  occupancy below 2 waves/SIMD, which leaves nothing to interleave, and a priority assignment
  that gives compute the higher value, which collapses the overlap it was meant to create.

### Version- and build-sensitive surfaces

- **The barrier builtin was renamed, and the rename is either-or.** 3.6.0 exposes only
  `gl.thread_barrier`, 3.7.0 onward only `gl.barrier` (the 3.8.0 spelling); neither keeps an alias for the other, so
  an anchor authored on one minor fails to *compile* on the other with a bare `AttributeError`.
  A one-directional shim is therefore useless half the time — install **both** aliases, each
  only when the target name is missing, so the shim is additive and a no-op on a minor that
  already has it. Dropping that into a `sitecustomize.py` on `PYTHONPATH` applies it at
  interpreter startup, which lets one anchor source sweep every minor without editing the
  anchor or touching site-packages.
- **On the oldest supported minor, a module-scope layout stops being a constexpr once it crosses
  a call boundary.** It gets materialized as an IR value instead, and the failure surfaces as
  `AttributeError: '<SomeLayout>' object has no attribute '_flatten_ir'` — an internal-looking
  message that names no missing capability and points at the call site rather than the layout.
  Two crossings trigger it, both measured: passing the global into a helper that is **not** a
  real builtin (one whose signature has no `_semantic`, e.g. the `zeros` spelling, which carries
  no `layout` parameter on any minor), and **forwarding** the global into a nested `@gluon.jit`
  helper. Neither is exotic — two independent anchors hit one each.
  **Fixes:** thread the layout in as the kernel's own `gl.constexpr` parameter (the body needs no
  other change), or use a genuine builtin such as the `full` spelling. Annotating the global
  itself as `: gl.constexpr` does **not** help — measured, both spellings still fail.
  Later minors accept both crossings, so this only bites when sweeping an anchor authored on a
  newer minor back onto the oldest one. Do not conclude a *language* limit from it: three
  narrower hypotheses (layout-as-constexpr-arg, parented Slice/MFMA layout kinds, interleaved
  constexpr/runtime signatures) were each isolated and all work on the old minor, so probes that
  pass the layout as a parameter will all come back green and hide the real trigger.
- Triton `<3.6` CDNA `AMDMFMALayout.instr_shape` examples may use 2D `[M, N]`;
  `>=3.6` uses 3D `[M, N, K]`. Check local `triton_version.py` before copying.
- Triton 3.5-era Gluon/AOT metadata can be incompatible with later source
  assumptions. Re-test AOT/prebuilt paths after Triton minor-version changes.
- `@triton.autotune(use_cuda_graph=True)` emits a deprecation warning on Triton
  3.7.0 — **not re-checked on 3.8.0**, where it may be warned, removed or unchanged;
  new kernels should not make it part of the stable API contract either way.
- **Fork-only environment variables are not features of your build.** `TRITON_ENABLE_LLIR_SCHED`,
  `TRITON_ENABLE_AMDGCN_AS`, `TRITON_ENABLE_AMDGPU_RA_HINTS` (and the plugin-path variables beside
  them) exist only on the vendor fork lineage; the `TRITON_GLUON_*` re-injection names exist nowhere.
  On a stock 3.8.0 build each is a silent no-op. The list, the evidence, and the upstream route for
  each capability (for matrix/VALU co-execution: `TRITON_HIP_USE_COEXEC_SCHEDULER`, the 3.8.0 coexec
  strategy, on by default for `num_warps <= 4`; for AGPR pinning: `llvm_fn_attrs`) are
  `../tile-programming/non-upstream-reserve.md`.

### Environment, install and cache

- An editable install of a production framework can uninstall/reinstall
  Triton-family packages (`triton`, `amd-triton`, `pytorch-triton-rocm`,
  `triton-rocm`) by ROCm version, silently swapping the build. Preserve the current
  Triton via the framework's opt-out env var (e.g. `AITER_USE_SYSTEM_TRITON=1`) when
  intended, and record the choice + installed Triton identity in benchmark metadata.
- **Failed-compile retry gotcha.** After a compile failure, a stale JIT/artifact
  cache can mask the fix on retry (you re-read the failed artifact) or, worse, serve
  a partially-built one. Clear the candidate `TRITON_CACHE_DIR` between retries of a
  layout/version-sensitive lowering, and re-run a same-code control after the env
  change before trusting the next result (`../method/benchmark-hygiene.md ## Artifact and cache hygiene`).
- High-occupancy hints can be unsafe as well as slow. Avoid broad sweeps that
  combine high `waves_per_eu`, large tiles, and matrix-instruction changes without
  a small correctness gate and crash-isolation plan.

### Numerics traps

- **Masked-pad of a wide-dim region: free padding, but a causal NaN trap.** A masked
  load (`mask + other=0`) zeroes the staged LDS so the matrix op reads 0 — a correct
  pow2 pad with **no host pad-copy** (avoids a large per-call `torch.pad`). Trap
  signature: **NaN at large-seqlen causal** (or any short / small-trip-count loop)
  with a masked-pad kernel — the masked LDS region stale-reads and `0 * inf = NaN`
  (padded rows multiplied against an `-inf`-masked score). Decision rule: masked-pad
  for **full** (non-causal) loops; a **no-pad split** (sync `convert_layout`, or a
  recovered async layout per sub-tile, `../tile-programming/layout-recipes.md`) for
  causal / short loops.

## Benchmark-sensitive platform issues

Implicit cache reuse; artifact path collisions; backend-specific warmup; import
routing that silently changes the loaded implementation; Triton minor-version
changes that alter Gluon JIT / layout / lowering or flip tuned config choices
(`matrix_instr_nonkdim`, `waves_per_eu`, instruction-shape lowering). Always
record enough environment information to reproduce the selected path
(`../method/benchmark-hygiene.md`).

## gfx942 / CDNA3 hard failures that affect benchmark validity

Full capability page: `../hardware/cdna3-gfx942.md`. The ones that break a **sweep/harness**
(not just a single kernel). The heading names gfx942 because that is where they were recorded and
where they bite first; the first applies on gfx950 too, at its own cap:

- **Process-level LDS abort (non-catchable) — both arches, at their own cap.** Allocating more
  LDS than the LDS/CU cap (**160 KiB on CDNA4/gfx950; 64 KiB on the CDNA3/gfx942 downgrade** — so a
  config that aborts on gfx942 may be legal on gfx950, and gfx942 is where it bites first) aborts
  with `local memory exceeds limit` at the **process** level — a `try/except` will **not** catch
  it, so one bad config takes down the whole sweep. The usual triggers are a cshuffle epilogue,
  which needs `2·tile_m·tile_n` LDS, and `NUM_STAGES` whole large tiles. Pre-screen
  `2·tile_m·tile_n ≤ LDS_cap` (cap from `perf_knowledge/hardware/data/hw_constants.json`
  `lds_per_cu_kib` for the `--arch` you pass) and run config sweeps under **subprocess isolation**,
  one process per variant (`../method/benchmark-hygiene.md`).
- **cannot-select MFMA (hard abort, gfx942 only).** The gfx950 shapes v4 `16x16x32` bf16 and
  scaled `mfma_scale_*_f8f6f4` (a4w4 / a8w4 / mxfp8) are cannot-select on gfx942 — a hard compiler
  abort, not a fallback. Confirm with `llvm-mc -mcpu=gfx942` before planning a lever around the
  shape, and defer block-scaled formats to gfx950.
- **External ASM entry crashes vary by entry.** An ASM decode kernel may crash at launch
  (`hipModuleLaunchKernel ... context destroyed`) while the prefill / regular entry runs
  and a persistent variant hangs — switch entry rather than declaring the op un-runnable.

## Toolchain-pin regressions: when the number you measured is not a ceiling

A toolchain re-pin moves the compiler, not just the API surface, and it can take performance
**away** from an unchanged kernel. The failure this produces is not a wrong number — the number
is real — it is a wrong **conclusion**: a kernel closed at a value set by a compiler regression,
recorded as the kernel's ceiling. That verdict then survives the toolchain being fixed.

**Decision rule.** Before recording any at-ceiling / stay-plain / negative verdict, ask whether
the pin under you is the pin the reference numbers were taken on. If it is not, the gap is a
**hypothesis about the toolchain** until separated from the kernel. Three generic signatures,
all observed on a **single** re-pin of one AMD tutorial fork (source: the
`ROCm/gfx950-gluon-tutorials` CHANGELOG), each hitting a *different* kernel class for an
unrelated reason — which is itself the lesson: one re-pin can carry several independent
regressions, so finding one does not clear the others:

| signature | mechanism class | what it hits |
| --- | --- | --- |
| a **new compiler flag** the new pin passes by default, which the old one did not | register-allocation policy change | one configuration only — spills appear on the affected arm while every other arm is untouched, which is what makes it look like that arm's own fault |
| a **known upstream LLVM bug whose fix post-dates the pin** (the change that *exposes* it is in; the change that *fixes* it is not) | codegen defect | one structural family — there, the multi-wave ping-pong GEMMs |
| a **pass whose behaviour narrowed between two tags** (something previously unconditional is now gated behind a predicate) | scheduling / barrier placement | the kernels that depended on the unconditional behaviour — there, the attention family |

**The recorded instance, with the names attached.** The three rows above are stated as signatures
because that is what transfers; the concrete instance is worth carrying too, because these names
are searchable and a reader on a nearby pin may be looking at the same thing. Each row's third
column is the **falsifier** — the one-shot test that proves the pin rather than the kernel owns
the gap:

| signature | the actual cause | falsifier |
| --- | --- | --- |
| new default flag | `amdgpu-use-amdgpu-trackers`, appended unconditionally by the newer Triton for gfx942/gfx950 and absent from the prior tags | stop passing it (the backend's own default) — the affected arm returns to its published value. It costs the *baseline* configuration a few percent, so gate it rather than reverting it |
| LLVM bug, fix post-dates the pin | an unsigned underflow in the register-pressure limit computation: the limit wraps when the reserved-register weight exceeds it, silently disabling pressure checking. Reached only by kernels that restrict AGPR availability (`amdgpu-agpr-alloc=0,0`) — which is exactly what the no-AGPR multi-wave GEMMs set | apply the upstream fix. Note it was **not** a clean cherry-pick there: it restored matrix efficiency on four kernels and cost a fifth roughly 40% of its throughput, so the falsifier and the remedy are different decisions |
| pass behaviour narrowed | the warp-pipeline conversion previously placed the loop-carried wrap-around barrier at the **top** of the loop body unconditionally, so that barrier's `setprio` primed the first cluster's priority every iteration; the newer tag gates that behind a predicate and keeps `setprio` at section ends | rebuild the new tag with the old tag's version of that one pass file, changing nothing else. There it recovered about **9 points** of in-loop matrix efficiency on both attention kernels |

Two things generalize from that list. The three causes live in **three different projects** —
the framework, LLVM, and a single framework pass — so "I found the regression" is not a reason to
stop looking. And the second row is the reminder that **a falsifier is not a fix**: proving the
pin owns the gap can be cheap while making the gap go away is a separate, possibly negative-sum,
decision.

The third is the most dangerous for a closure decision, because it is invisible in every
kernel-side reading: registers, occupancy and the instruction inventory are all unchanged, and
only the in-loop matrix-issue efficiency moves. Identical source measured on the two tags gives
two different efficiencies, so the lower one belongs to the *pin*, not to the kernel — and
rebuilding the newer tag with the older tag's version of that single pass restores it. The
generalizable tell: **an efficiency drop with no corresponding change in any resource reading
is a schedule that was taken away from you**, and no kernel-side lever will win it back.

**What to do rather than absorb it:**

- **Record the pin beside every number** (`../tile-programming/compiler-contract.md ## Toolchain
  identity`). A measurement whose toolchain identity is unknown cannot be compared to a published
  one.
- **When a published reference exists and you are under it by a wide margin at unchanged
  registers/occupancy, suspect the pin before the kernel.** The cheap discriminator is a
  *toolchain-side* control — the same source on the other pin — not another kernel edit.
- **Keep an external control if you have one.** A comparator that does not go through the same
  compiler (a different DSL or a vendor library implementing the same op) is unaffected by that
  compiler's regressions, which is exactly what makes it able to tell you one happened.
- **Never record it as a hardware wall.** A pin regression is a scoped, dated, reversible
  toolchain ceiling; write it as one, with the tag, so it expires when the pin moves.
- **And know that the third signature is actionable, not merely diagnosable.** The four bullets
  above are all about not *misattributing* it; none of them says what may be done. Restoring one
  pass file's prior behaviour and rebuilding — everything else unchanged — is a **sanctioned
  front-end-source change** (`../tile-programming/compiler-contract.md ## Sanctioned tier: changing
  the front end's own source`), and it is the cheapest tier in that family to test because the
  hypothesis is a single file and both versions can be kept side by side as their own gate.
  Three things make it worth naming here rather than leaving as a handoff note: the diagnosis is
  already the hypothesis, so no search is needed; the A/B is a *toolchain-side* control, which is
  the discriminator the second bullet asks for; and a win belongs to the **toolchain**, so it is
  recorded as a scoped dated finding and the kernel's own ceiling is not credited with it.
  Unsanctioned it stays **out of scope** — which is still not a ceiling.

## RDNA4 client PMC availability (gfx1201 ISA; observed on R9700)

**Observed on R9700 (gfx1201):** `rocprofv3 --kernel-trace` reliably captures dispatches
(count > 0), but the **PMC counter path differs from CDNA** — the available counter set
and counter naming/semantics on RDNA4 client parts are not the same as on Instinct
(CDNA), and client-GPU PMC profiling is less complete. CDNA counter names
(`SQ_WAVES` / `VALUInsts` / the `MfmaUtil` / `VALUBusy` family) may be absent or differ.
The ISA caveat may apply to other gfx1201 products, but GEAK's calibrated
product policy and evidence remain R9700-only.

Decision rule (per-box preflight):
- Before trusting a PMC-derived bound class on RDNA4, run `rocprofv3-avail list --pmc` and
  confirm the specific counters you need actually exist on this device. For
  rocprofv3 itself, use `rocprofv3 -L` / `--list-avail`; older profiler
  generations used `--list-basic`, `--list-derived`, or `--list-counters`.
  Treat this as a CLI rename, not an R9700-image defect.
- If the discriminating counters are unavailable, do **not** fabricate them — fall back to
  the no-profiler evidence path: analytical roofline (`../hardware/roofline-models.md`) +
  static `.amdgcn`/`.s` audit (`kernel_workflow/scripts/kernel_tools/asm_loop_audit.py`; the pack's
  `scripts/asm_loop_audit.py` is a shim to it) + floor probe + A/B timing
  at the production boundary (`../method/profile.md ## Profiler-capability preflight`).

> Scope: seen on R9700/gfx1201; the exact available counter set is device+ROCm-version
> dependent — always query `rocprofv3-avail` on the actual box.

## `int64_strides=false` regression on gfx1201 under CUDA graph

**Observed on R9700 (gfx1201):** for a triton attention kernel measured at the
**CUDA-graph** boundary, forcing `int64_strides=false` regressed ~5x (8K causal ~31.5 ms
vs ~6.55 ms with `int64_strides=true`) — the vectorized-load path appeared to fall back
to a non-vectorized path in that `gfx1201 + cuda-graph` configuration.

Decision rule: on gfx1201 attention kernels served under a CUDA graph, keep
`int64_strides=true` unless an A/B on the actual target proves otherwise.

> Scope: single kernel on R9700/gfx1201 at the CUDA-graph boundary. Not established as a
> general RDNA4 rule — A/B this knob on your kernel/boundary, do not treat it as a law.

## Container-locus profiling defects (host ⇄ kernel-container split)

These bite ONLY when the kernel runs in a **separate container** from the host that drives
`capture.sh` (now `kernel_workflow/scripts/kernel_tools/capture.sh`, under GEAK's profiler entry
`kernel_workflow/scripts/profile_kernel.sh`) (`TILE_KERNEL_CONTAINER` set, profilers wrapped via `locus.sh`). On a single-host
setup they do not fire. Each was hit in a real run; the workaround is proven, the root fix is
noted for whoever next has the actual two-container box to verify on — do **not** apply a
host-side `readlink -f` blind, because the container may see the bind-mount at a *different*
absolute path and a wrong "fix" writes profiler output to the wrong place silently instead of
failing loudly.

- **Relative `-d`/`-w` output dir resolves INSIDE the container → every profiler layer silently
  degrades to PMC-blind.** `capture.sh` / `rocprof_compute_probe.sh` hand a relative `$OUT` to
  `locus_run` (= `docker exec [-w $TILE_KERNEL_CONTAINER_WORKDIR]`); with no workdir the container
  CWD is `/`, so rocprofv3 writes to `/exp/.../kt` *inside* the container and the host-side parse
  finds nothing → `balanced` from a missing layer (a confident WRONG class, not "no data").
  **Workaround:** `export TILE_KERNEL_CONTAINER_WORKDIR=<abs work_root>` so relative paths resolve
  to the same bind-mounted path both sides. **Root fix (needs the container to verify):** resolve
  `$OUT` to the *container-visible* absolute path before `locus_run`, or default the workdir to the
  bind-mount root.

- **`locus_run` forwards NO env into `docker exec`.** `HIP_VISIBLE_DEVICES` / `TRITON_*` are not
  passed, so a profiler pass can land on the container's default GPU — a correctness *and* courtesy
  hazard on a shared box where the run owns only some GPUs. **Workaround:** acquire the GPU through
  `kernel_workflow/scripts/gpu_lock.sh <gpu_id> <command...>` (the only GPU lock; it exports the
  locked device and `GEAK_GPU_LOCK_HELD`), and have the app set that device in-process before
  importing torch, so every pass lands right regardless of the wrapper. Never prefix a profiler
  command with an inline `HIP_VISIBLE_DEVICES=`. **Root fix:** `locus_run` should
  `-e HIP_VISIBLE_DEVICES` (and `TRITON_*`) when set.

- **`dump_ir.sh` (`kernel_workflow/scripts/kernel_tools/dump_ir.sh`) runs the app host-side with a host `/tmp` `TRITON_CACHE_DIR`** a container locus
  cannot see → the static-ISA layer is structurally undumpable on a container task. **Workaround:**
  have the app re-enter the container with `TRITON_CACHE_DIR` redirected to a bind-mounted path,
  then copy artifacts back to where `dump_ir.sh` expects them. **Root fix:** route the compile
  through `locus.sh` with a bind-mounted cache.

- **`rocprof-compute profile` runs in-container but `analyze` runs host-side** (ROCm minors differ,
  e.g. 7.1 vs 7.2) → `PermissionError` / missing-package aborts, and a rocprof-compute workload dir
  is not portable across ROCm minors anyway. This is why a `rc_profile.log` path can be cited that
  does not exist host-side (now handled: the probe prints the tail inline or says the log is
  container-side). **Workaround:** a PATH shim that re-execs host-side `rocprof-compute` into the
  container. **Root fix:** run `analyze` through the SAME `locus.sh` as `profile`. Until then SOL is
  optional (`../method/profile.md ### 3.2 Evidence layers: the floor, and the full read`) —
  ATT + static are the required pair and both run through `rocprofv3` alone.

## How to use this file

Use it only after the core flow has answered what the direction is, what the
implementation layer is, what the benchmark boundary is, and why the result is
believed real. If a platform note changes the outcome, feed the result back into
the task contract, benchmark hygiene, dispatch verification, and stop-condition
logic.
