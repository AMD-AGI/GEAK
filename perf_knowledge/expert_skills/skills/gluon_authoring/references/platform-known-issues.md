# Platform Known Issues (gfx950 / gfx942)

Target- and backend-sensitive constraints that should not live in the core
workflow. Use `hardware/capability-matrix.md` as the queryable source of truth
for gfx942/gfx950 MFMA, FP8, scaled-op, namespace, and correctness status; keep
this file for general platform-sensitive decision rules.

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

Only the last two speak to mechanism viability. Examples of capability
boundaries:

- CDNA3 and CDNA4 can share some lowering paths, but CDNA3 evidence does not prove
  CDNA4 scaled behavior.
- Compile success is not correctness. A path can compile through a non-native
  namespace and still produce wrong results. Keep blockers scoped to target,
  dtype, API path, and software version.
- A gfx950-only or scaled source path should be locally probed on gfx950 before
  replacing it with a gfx942 workaround.
- gfx950/CDNA4 evidence must be tracked separately from gfx942 evidence; use
  `local-confirmed` in `hardware/capability-matrix.md` only for locally-checked,
  scoped facts.
- AMD scheduling helpers (`sched_barrier`, `sched_group_barrier`) may be absent
  even when production kernels contain scheduling concepts. Treat absence as a
  toolchain ceiling and switch direction.
- `warp_pipeline_stage` can be source-available but performance-negative. Compile
  success is not a scheduling win; require quick timing before expanding search.
- gfx942-oriented FP8 formats (fp8e4b8 / e4m3fnuz) may be upcast on gfx950. Verify
  the actual dtype lowering before interpreting FP8 performance.
- High-occupancy hints can be unsafe as well as slow. Avoid broad sweeps that
  combine high `waves_per_eu`, large tiles, and matrix-instruction changes without
  a small correctness gate and crash-isolation plan.
- An editable install of a production framework can uninstall/reinstall
  Triton-family packages (`triton`, `amd-triton`, `pytorch-triton-rocm`,
  `triton-rocm`) by ROCm version, silently swapping the build. Preserve the current
  Triton via the framework's opt-out env var (e.g. `AITER_USE_SYSTEM_TRITON=1`) when
  intended, and record the choice + installed Triton identity in benchmark metadata.
- Triton 3.5-era Gluon/AOT metadata can be incompatible with later source
  assumptions. Re-test AOT/prebuilt paths after Triton minor-version changes.
- Triton `<3.6` CDNA `AMDMFMALayout.instr_shape` examples may use 2D `[M, N]`;
  `>=3.6` uses 3D `[M, N, K]`. Check local `triton_version.py` before copying.
- **A transcription result does not carry across CDNA generations, and the parts that break
  are not the ones a version sweep exercises.** Before quoting a same-generation result on
  another one, re-check at least: (a) the per-generation Gluon namespace on every buffer and
  MFMA builtin, plus the MFMA layout's version field — the newer namespace generally exists on
  older Tritons too, so this looks like a rename, but the accepted `instr_shape` sets differ
  and a changed K dimension pulls the tile and loop structure with it; (b) any arm that failed
  on an LDS/shared-memory limit, since the per-CU budget can differ by more than 2x and such an
  arm may simply start fitting; (c) the depth of any `num_stages`-style sweep, for the same
  reason; (d) counts of constructs that exist to dodge bank conflicts, since the bank count can
  change. When re-validating layout recovery itself, assert that recovery still succeeds and
  round-trips **within** the new generation and stays stable across its Triton minors — *not*
  that layout digests match the other generation. MFMA layout digests **should** differ across
  generations, so asserting equality manufactures a false "the tool broke here".
- **A namespace importing is not the op lowering, and the gap is silent.** A per-generation
  module can import cleanly on a target it does not support: probed on gfx942, the whole
  `cdna4.async_copy` surface imports, `load_shared_relaxed` from it even compiles and runs, yet
  `buffer_load_to_shared` fails LLVM translation and `global_load_to_shared` fails the pass
  manager, at every vector width and on every Triton minor tested. Availability claims of the
  form "the symbol is there" are therefore not evidence; compile a one-op probe on the actual
  target. The same trap caught `warp_specialize`, which is present in core `gl` on every
  version and still aborts the pass manager on CDNA3.
- **The barrier builtin was renamed, and the rename is either-or.** Some minors expose only
  `gl.thread_barrier`, later ones only `gl.barrier`; neither keeps an alias for the other, so
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
- **Async-copy offset layouts are version-gated.** Whether `buffer_load_to_shared`
  accepts a given offset layout depends on the Triton minor: an older minor may
  lower only `Blocked` / `Slice` offset layouts and **fail to compile** a
  `DistributedLinearLayout` async offset that a newer minor accepts. Gate the Gluon
  async path to the build that supports the needed offset family; on the unsupported
  minor fall back to a sync `convert_layout`. Probe per build, do not carry the gate
  across builds (`hardware/capability-matrix.md`).
- **Failed-compile retry gotcha.** After a compile failure, a stale JIT/artifact
  cache can mask the fix on retry (you re-read the failed artifact) or, worse, serve
  a partially-built one. Clear the candidate `TRITON_CACHE_DIR` between retries of a
  layout/version-sensitive lowering, and re-run a same-code control after the env
  change before trusting the next result (`benchmark-hygiene.md ## JIT prewarm`).
- `@triton.autotune(use_cuda_graph=True)` emits a deprecation warning on Triton
  3.7.0; new kernels should not make it part of the stable API contract.
- **Masked-pad of a wide-dim region: free padding, but a causal NaN trap.** A masked
  load (`mask + other=0`) zeroes the staged LDS so the matrix op reads 0 — a correct
  pow2 pad with **no host pad-copy** (avoids a large per-call `torch.pad`). Trap
  signature: **NaN at large-seqlen causal** (or any short / small-trip-count loop)
  with a masked-pad kernel — the masked LDS region stale-reads and `0 * inf = NaN`
  (padded rows multiplied against an `-inf`-masked score). Decision rule: masked-pad
  for **full** (non-causal) loops; a **no-pad split** (sync `convert_layout`, or a
  recovered async layout per sub-tile, `tile-programming/layout-recipes.md`) for
  causal / short loops.

## `warps_per_cta` is not an independently tunable knob (hand-authored Gluon)

It reads like a config field, so it invites a one-line A/B. It is not one: in a
hand-authored kernel the warp-to-tile mapping is **restated in every layout in the
file**, and changing `mfma_layout.warps_per_cta` alone makes the module incoherent.

Measured, gfx950, triton `3.7.0+amd.rocm7.2.0.git89002410` — editing only
`warps_per_cta=[4, 1] -> [2, 2]` on an MLA forward:

```
llvm/ADT/Sequence.h:275: iota_range::iota_range(T, T, bool):
  Assertion `Begin <= End && "Begin must be less or equal to End."' failed.
```

The process dumps core with **no attribution to a layout or a line**, which reads
like a compiler bug and is not one: that kernel pins **14 hardcoded `warp_bases`
tuples** across its blocked, linear and dot-operand layouts, every one written for a
4-warps-tile-M mapping — one of them the degenerate all-warps-see-everything form
`warp_bases=((0, 0), (0, 0))`, which is only meaningful when warps do **not** tile
the reduced dimension. `[2, 2]` invalidates all of them at once, and the assert is
the pass manager meeting that incoherence.

Consequences, both of which have cost a run:

- **Do not read the crash as "this arch/build cannot do `[2,2]`."** Nothing about
  the target was tested. Grep the source for `warp_bases` first; the count tells you
  the real size of the change.
- **Scope it as a coupled layout rewrite, not a knob.** Every `warp_bases` in the
  file moves together, plus whatever cross-warp exchange the new mapping introduces
  (splitting warps across the reduced dimension means the two dots no longer see the
  same operand rows). That is a multi-round direction; attempting it as a one-lever
  round produces this crash and a false negative. See
  [`entry-modes.md ## The depth contract`](entry-modes.md).

Why it keeps getting attempted: `warps_per_cta=[N, 1]` broadcasts the B operand to
all N warps, so an MLA-shaped kernel reads its whole K and V tile N times out of LDS.
Halving that replication is a real prize — it is just not a one-line one.

## Benchmark-sensitive platform issues

Implicit cache reuse; artifact path collisions; backend-specific warmup; import
routing that silently changes the loaded implementation; Triton minor-version
changes that alter Gluon JIT / layout / lowering or flip tuned config choices
(`matrix_instr_nonkdim`, `waves_per_eu`, instruction-shape lowering). Always
record enough environment information to reproduce the selected path
(`benchmark-hygiene.md`).

## RDNA4 client PMC availability (gfx1201 / R9700 / RX9070)

**Observed on R9700 (gfx1201):** `rocprofv3 --kernel-trace` reliably captures dispatches
(count > 0), but the **PMC counter path differs from CDNA** — the available counter set
and counter naming/semantics on RDNA4 client parts are not the same as on Instinct
(CDNA), and client-GPU PMC profiling is less complete. CDNA counter names
(`SQ_WAVES` / `VALUInsts` / the `MfmaUtil` / `VALUBusy` family) may be absent or differ.

Decision rule (per-box preflight):
- Before trusting a PMC-derived bound class on RDNA4, run `rocprofv3 --list-counters` and
  confirm the specific counters you need actually exist on this device.
- If the discriminating counters are unavailable, do **not** fabricate them — fall back to
  the no-profiler evidence path: analytical roofline (`hardware/roofline-models.md`) +
  static `.amdgcn`/`.s` audit (`../scripts/asm_loop_audit.py`) + floor probe + A/B timing
  at the production boundary (`phases/profile.md ## Profiler-capability preflight`).

> Scope: seen on R9700/gfx1201; the exact available counter set is device+ROCm-version
> dependent — always `--list-counters` on the actual box.

## `int64_strides=false` regression on gfx1201 under CUDA graph

**Observed on R9700 (gfx1201):** for a triton attention kernel measured at the
**CUDA-graph** boundary, forcing `int64_strides=false` regressed ~5x (8K causal ~31.5 ms
vs ~6.55 ms with `int64_strides=true`) — the vectorized-load path appeared to fall back
to a non-vectorized path in that `gfx1201 + cuda-graph` configuration.

Decision rule: on gfx1201 attention kernels served under a CUDA graph, keep
`int64_strides=true` unless an A/B on the actual target proves otherwise.

> Scope: single kernel on R9700/gfx1201 at the CUDA-graph boundary. Not established as a
> general RDNA4 rule — A/B this knob on your kernel/boundary, do not treat it as a law.

## gfx942 / CDNA3 hard failures that affect benchmark validity

Full capability page: `hardware/cdna3-gfx942.md`. The ones that break a **sweep/harness**
(not just a single kernel):

- **Process-level LDS abort (non-catchable).** Allocating more LDS than the LDS/CU cap (**64 KiB on CDNA3/gfx942, 160 KiB on CDNA4/gfx950** — the cap is arch-specific, so a config that aborts on one may be legal on the other)
  (e.g. a cshuffle epilogue's `2·tile_m·tile_n`, or `NUM_STAGES` whole large tiles) aborts
  with `local memory exceeds limit` at the **process** level — a `try/except` will **not**
  catch it, so one bad config takes down the whole sweep. Pre-screen
  `2·tile_m·tile_n ≤ LDS_cap` and run config sweeps under **subprocess isolation**
  (`benchmark-hygiene.md`).
- **cannot-select MFMA (hard abort).** v4 `16x16x32` bf16 and scaled `mfma_scale_*_f8f6f4`
  are cannot-select on gfx942 — a hard compiler abort, not a fallback. Confirm with
  `llvm-mc -mcpu=gfx942` before planning a lever around the shape.
- **External ASM entry crashes vary by entry.** An ASM decode kernel may crash at launch
  (`hipModuleLaunchKernel ... context destroyed`) while the prefill / regular entry runs
  and a persistent variant hangs — switch entry rather than declaring the op un-runnable.

## Container-locus profiling defects (host ⇄ kernel-container split)

These bite ONLY when the kernel runs in a **separate container** from the host that drives
`capture.sh` (`TILE_KERNEL_CONTAINER` set, profilers wrapped via `locus.sh`). On a single-host
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
  hazard on a shared box where the run owns only some GPUs. **Workaround:** pin the device inside
  the app itself (export `HIP_VISIBLE_DEVICES` before importing torch), so every pass lands right
  regardless of the wrapper. **Root fix:** `locus_run` should `-e HIP_VISIBLE_DEVICES` (and
  `TRITON_*`) when set.

- **`dump_ir.sh` runs the app host-side with a host `/tmp` `TRITON_CACHE_DIR`** a container locus
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
  optional (`SKILL.md` §3.2) — ATT + static are the required pair and both run through `rocprofv3`
  alone.

## How to use this file

Use it only after the core flow has answered what the direction is, what the
implementation layer is, what the benchmark boundary is, and why the result is
believed real. If a platform note changes the outcome, feed the result back into
the task contract, benchmark hygiene, dispatch verification, and stop-condition
logic.
