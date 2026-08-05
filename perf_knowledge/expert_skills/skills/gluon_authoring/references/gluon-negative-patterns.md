# Gluon Negative Patterns

Use this file when a Gluon route is plausible but may be the wrong layer for the
measured optimization direction. These are problem shapes and decision rules, not
kernel-specific recipes.

**This file is also the escalation gate's backing**: the Quick Reject checklist
below = the gate's "skip transcription / stay plain" conditions
(`escalation-gate.md`). Per-target support/evidence lives in
`hardware/capability-matrix.md`.

## How To Use This File

Before a non-trivial Gluon probe, ask:

```text
Direction / Triton + ROCm + target arch:
Minimal @gluon.jit smoke result / repo gate result:
Gluon-only mechanism:
Hot loop conversions / dot+reduction+atomic sites affected:
JIT boundary crossed / memory traffic changed:
Expected gain range / stop condition:
```

Reject, downgrade, or narrow the probe when the Gluon route would test too many
unrelated constraints at once. If the minimal smoke fails, classify as a
toolchain blocker before any mechanism conclusion; if only the repo production
gate fails, decide whether one forced-backend probe is worth the risk.

## Quick Reject Checklist (= gate skip-transcription conditions)

Reject or downgrade a Gluon probe (stay plain) when most of these are true:

- the hot path spans multiple `@triton.jit` helpers and tensor values cross the
  JIT boundary;
- converting one small idea requires changing the whole call chain to `@gluon.jit`;
- the inner loop has several dot products with different logical tile shapes;
- dot, reduction, and atomic constraints must all be solved in one patch;
- many constexpr branch combinations and no single measured subpath;
- the candidate requires a large rewrite before any output feeds the benchmark;
- the proposed Gluon version is the same algorithm with no explicit layout,
  memory-path, scheduling, or matrix mechanism;
- `hardware/capability-matrix.md` marks the target/dtype/op path `wrong-result`,
  `version/API-blocker`, or a target-specific blocker;
- the kernel is launch/wrapper dominated and Gluon only changes device-body
  spelling;
- the path is partial migration through `tl.dot`, whose result will not carry a
  Gluon layout for later Gluon ops;
- plain Triton already lowers the hot dot to the desired `tt.dot/#mma` path and
  the remaining bottleneck is config, wrapper, dispatch, or memory traffic;
- the plain backend stages an operand feed in `#ttg.amd_rotating_shared` (visible
  in the dumped TTGIR) **and the `gluon.language` build in front of you exposes no
  rotating-shared constructor** — check before concluding it, the surface moves:
  `ttgir_to_gluon.py` emits `None  # NOT EMITTED: amd_rotating_shared ...` for it
  either way, and that message is a prompt to probe, not proof of a language gap.
  (`amd_wmma` sat behind the same wording and turned out to be constructible as
  `AMDWMMALayout` all along — see `gluon/rdna-wmma-reference.md`.) If the
  constructor really is absent, a faithful transcription is not expressible;
  `PaddedSharedLayout`/`SharedLinearLayout` can approximate the M-major
  staging but only to parity, so a Gluon arm gated on this layout stays plain
  (record it `deferred`, bounded, not a proven wall).

A quick reject routes to `plain-strategies.md` (plain / config / dispatch) or a
smaller Gluon smoke.

## No-Extra-Mechanism Decision

```text
Does plain Triton already lower the hot dot/load/store to the desired path?
Does Gluon remove real memory traffic, layout conversion, launches, or wrapper cost?
Is the Gluon-eligible phase a material fraction of the measured boundary?
Is the expected end-to-end gain above the noise/repeat threshold?
```

If the answers are `yes, no, no/low, no`, stop the Gluon route. Record
`no_extra_gluon_mechanism` and return to the open layer with larger headroom.

## Stage-Gluon Body Route Guardrails

Once escalated, Gluon still needs a mechanism. Keep the plain-Triton comparator
(the target line) while classifying the body bottleneck:

```text
memory API changes bytes or address pressure:
layout changes remove conversion or enable a consumer:
shared memory removes traffic or conversion:
side paths move off the critical path:
epilogue/store layout removes movement:
compiler realization has source-level independence:
```

Reject broadening when the only evidence is "same body in Gluon". If matrix-core
work is absent or not the hot stage, stay on the measured memory/layout/side-path/
reduction/store bottleneck.

## Hot-Loop Layout Conversion

Repeated `convert_layout` inside the innermost loop is a paid operation. It is a
negative signal when the conversion cannot be hoisted to a host-created layout
contract, a phase boundary, a matrix-operand boundary, or a precomputed metadata
path.

Rules: count conversions per loop iteration before tuning launch parameters;
check whether the converted value is loop-invariant; require a hoist point before
treating launch knobs as the fix; keep plain Triton when `tl.dot`/ordinary ops
avoid the movement; only continue with Gluon if the explicit matrix/memory path
removes more work than the conversions add.

### K-Proportional Conversion Cost

```text
fixed host/wrapper overhead / per-K conversion count / per-K scale conversion
K-loop iterations / large-shape comparator result / removable mechanism
```

Fixed launch/wrapper/host-layout costs shrink as a share of larger problems;
hot-loop conversion costs usually scale with `K / BLOCK_K`. If a larger-K or
compute-bound probe does not close the gap, stop the serial Gluon route unless the
next change reduces conversion count, loads in the consumer layout, uses shared
staging, or moves conversion to a phase boundary.

## Fused / Block-Scaled Matrix Kernels

Gluon is a weak body-rewrite candidate when most hold: one body has multiple
matrix paths with different dtype/scale/instruction-shape contracts; the plain
path already lowers the hot dot; source is `dot(a,b)*scale` (scale-before-add,
which MFMA's accumulator cannot express directly); scaled-matrix scale
granularity does not match source granularity (forcing hot-loop scale
replication); the smallest executable rewrite is a serial anchor with repeated
operand `convert_layout`; the removing mechanism (direct-to-LDS staging, compiler
interleaving) is unavailable. Valid next: keep the plain comparator; split the
fused body by feature only if the ABI allows; test one executable subpath with a
mechanism that removes repeated conversion; or record a fused-architecture
negative. Build a phase map first (matrix phases / dtype per phase / scale phases
/ side-load placement / ideal tile per phase / accumulator lifetimes).

## Migration Radius Too Large

Negative signals: matrix operands, online state, masks, and stores all need
different parent layouts; atomic updates constrain layout/ordering in the same
loop as matrix work; a helper returns tensor values to another JIT helper (no
executable partial conversion); correctness depends on several dtype-narrowing
points. Responses: make a layout map before editing; select one executable
subpath; keep the direction plain if the smallest Gluon patch is a whole-kernel
rewrite; record a negative if the migration radius exceeds the expected gain.

## Bandwidth-Ceiling Refinement

Stop device-side compute tuning when measured bandwidth is near the practical
ceiling, the change reduces arithmetic/instruction-count but not memory traffic,
repeats are inside the noise band, or the only gain is `<1-2%`. Exception:
register spill is hidden memory traffic — reducing `waves_per_eu` or changing
launch/config can cut HBM traffic with no visible load/store change; do not
quick-reject that as "no traffic change" until register pressure is considered.
Then: reduce bytes moved, improve locality/reuse, change dispatch/bucketing, or
report a bandwidth-ceiling negative.

## Paged / Indirect Decode

Gluon is unlikely to help when K/V addresses depend on a page-table / gather
chain, address calc is load-dependent (`page_load -> kv_loc -> data_load`), the
per-iteration tile is small with high loop count, plain `tl.dot` already maps to
the desired matrix instruction, and latency is near the access-pattern ceiling.
Buffer ops / async copy / explicit layouts help only when they reduce bytes
moved, change the address chain, improve locality, or remove a measured
conversion. Otherwise keep the direction in plain Triton, split/reduce, dispatch,
or wrapper/boundary work.

One rule for the common case: when the gathered K/V is **L2-resident** (small,
re-read across queries) the bound is **load-latency / memory-level-parallelism, not HBM
bandwidth** — the signal is `MemUnitStalled ~ 0` with the busy counter well under 100%
(`phases/profile.md ## Derived metrics`). Then widening loads (already maxed) and
HBM-BW levers do nothing; only raising occupancy / MLP hides it, and software prefetch
regresses if VGPR/LDS already cap waves (`tile-programming/pipeline.md ## Budget
before deepening`). Async direct-to-LDS also will not apply: scattered per-token offsets
are not pre-coalesced, so the Gluon path (no `CoalesceAsyncCopy`) falls back to register
staging or fails lowering.

## Target-Specific Matrix Blockers (scoped)

If `hardware/capability-matrix.md` marks a target/dtype/op cell `wrong-result` or
`version/API-blocker`, do not spend rounds tuning around it. Example:

```text
target: gfx942 / dtype-op: FP8 Gluon MFMA
status: wrong-result or version/API-blocker
valid next action: plain Triton comparator, config dispatch, or gfx950-local probe
invalid next action: keep tuning the same gfx942 Gluon MFMA path
```

Keep the conclusion scoped to the target/dtype. A gfx942 FP8 blocker is not a
gfx950 blocker unless a gfx950-local probe proves the same failure.

## Slow-Correct Gluon Result

A correct slower Gluon path is evidence, not failure to hide. Before changing more
code, name one overhead: layout conversion; layout padding; memory-path fallback/
mask cost; extra launch/dispatch; shared-memory staging; scheduling-barrier
mismatch; wrapper/artifact selection. If no single removable overhead is visible,
stop the Gluon search for that direction and return to the layer matching the
measured bottleneck.

## gfx942 / CDNA3 negative-result signatures (mechanism + error text)

Documented signatures so a future run recognizes the pattern without re-deriving it.
These are **generic signatures** (condition + mechanism + exact error), not perf logs —
do not read them as absolute ceilings for every shape; A/B on your own kernel.

1. **Explicit tile loses to plain's compiler-managed pipeline (LDS-cap-bound regime).**
   On large-M bf16/int8 GEMM, when the per-CU LDS cap (arch-specific — gfx942 is the tight
   one; read `hw_constants.json` `lds_per_cu_kib`) forces small tiles, even a full
   Route-1 + async + ping-pong explicit Gluon loop can trail the tuned-plain
   compiler-managed pipeline (the compiler's pipeline/RA is hard to beat by hand, and a
   ping-pong turns barrier-sync-bound). Signature: explicit tile at parity-or-below plain
   after the pipeline layer is exhausted. Action: keep plain (measured), record the gap
   decomposition (`escalation-gate.md`), do not keep forcing explicit control.
2. **A hand register double-buffer can BEAT the compiler pipeline — check VGPR/spill
   before deepening.** On some large-M GEMM the winner is a hand-written register
   double-buffer (low VGPR, zero spill, high waves), while Route-1 spills and async does
   not lower. Rule: **before deepening the pipeline, read VGPR / spill from the `.amdgcn`
   KD** — if a register double-buffer fits without spill, try it before more pipeline depth.
3. **Scheduler-limited MFMA-continuity ceiling (asm_loop_audit signature).** Pipeline is
   ON (relaxed `s_waitcnt lgkmcnt(N>0)`, few full-drains, `s_nop=0`, prefetch present) yet
   `asm_loop_audit.py` shows **`MFMA↔VALU transitions = 0` + a long MFMA clump**. That is a
   legitimate *scheduler* ceiling (LLIR_SCHED domain) — not a kernel bug. On attention it
   is where `LLIR_SCHED` asserts; record it as a scheduling ceiling, do not chase it with
   more kernel edits.
4. **Zero-copy V-transpose LLVM assert.** A zero-copy V-transpose layout can fail the
   backend with `TritonAMDGPUInThreadTranspose.cpp:526 PassManager failed`. Signature:
   compile-time PassManager failure tied to the in-thread-transpose pass. Action: fall back
   to an explicit transpose/staging path; record the layout as a lowering ceiling.
5. **cshuffle epilogue LDS process-abort (non-catchable).** A cshuffle epilogue needs
   `2·tile_m·tile_n` LDS; on a large tile this exceeds the per-CU LDS cap (arch-specific —
   read `hw_constants.json`; gfx942 is the tight one) and aborts at the **process
   level** with `local memory exceeds limit` — **not** a catchable Python exception, so a
   config sweep that hits it takes down the whole batch. Action: pre-screen
   `2·tile_m·tile_n ≤ LDS_cap` and run sweeps under subprocess isolation
   (`benchmark-hygiene.md`).
6. **scaled-MFMA cannot-select.** `mfma_scale_*_f8f6f4` is cannot-select on gfx942 (a4w4 /
   a8w4 / mxfp8 have no scaled path) — a genuine `does-not-lower` hardware ceiling, confirm
   with `llvm-mc -mcpu=gfx942` and defer (`hardware/cdna3-gfx942.md`).

## Negative Result Record

```text
Direction / Probe class / Triton + ROCm + target arch:
Smoke result / repo gate or forced-backend result:
bottleneck class / Gluon mechanism tested:
dependent mechanism blocked or unjustified:
Why the path executed / Correctness / Measured boundary:
Result / Dominant overhead / Failure class:
Why not continue / Scaffolding removed / Next valid direction:
```

Keep results in terms of problem shape and mechanism (the `failure_class` enum is
in `experiment-records.md`). Avoid recording kernel-specific constants unless they
are part of the public workload contract.
