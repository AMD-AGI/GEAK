# Platform Known Issues (gfx950 / gfx942)

Target-, version- and toolchain-sensitive traps: lowering, toolchain pins, harness-breaking aborts.
gfx950 (CDNA4) / Triton 3.8.0 is the main line, gfx942 (CDNA3) the downgrade. Queryable per-target
status: `../hardware/capability-matrix.md`. Kernel-shape and method negatives: `negative-patterns.md`.

RDNA4 is outside this skill's scope. The two RDNA4 / R9700-only rules that
`perf_knowledge/hardware/rdna4_gfx1201/pitfalls.md` points here for (PMC availability with the
`rocprofv3 -L` / `--list-avail` CLI rename, and `int64_strides` under a CUDA graph) are kept at the end of
this page: `## RDNA4 (gfx1201, observed on R9700): kept for GEAK's RDNA4 pages`.

## Architecture and backend sensitivity

### Evidence layers: what a probe has and has not shown

- **Compile success is not correctness, and an import is not a lowering.** `gl.warp_specialize`
  imports from core `gl` and fails the pass manager on CDNA (gfx950 and gfx942; compile-checked on
  Triton 3.8.0).
  Compile a one-op probe on the actual target, satisfying the op's contract.
- gfx942 evidence does not prove gfx950 scaled behaviour (and vice versa); probe gfx950-only paths on
  gfx950 before substituting a gfx942 workaround.

### gfx950 first: what does not carry down to gfx942 (or back up)

- **A FNUZ fp8 dtype reaching gfx950 is silently upcast (to fp16, warning only).** gfx950 is OCP
  (`float8_e4m3fn` / `e5m2`); gfx942 is FNUZ (`fp8e4b8` / `float8_e4m3fnuz`). Take the spelling from
  `perf_knowledge/hardware/data/hw_constants.json` `fp8_dtype` for the `--arch`, and check the
  lowered dtype before reading fp8 perf.
- **Re-check before quoting a result on the other generation:** per-generation namespace and
  accepted MFMA `instr_shape` sets (a changed K drags tile and loop structure); arms that failed on
  LDS (160 KiB/CU gfx950 vs 64 KiB gfx942 — they may now fit); ring depth / plain `num_stages`;
  bank-conflict dodges (64 banks gfx950, 32 gfx942). MFMA layout digests **should** differ across
  generations; assert recovery round-trips within a generation, not digest equality.

### Async copy: available on both generations, and easy to misdiagnose as unavailable

`buffer_load_to_shared` / `global_load_to_shared` lower from stock `gluon_to_ttgir` on both arches:
per-lane width exactly **128 or 32 bit on gfx950, 32 bit on gfx942** (64 bit fails on both; gfx942
also wants a clean tiling and a destination `order` matching the source's fast dim, `[1, 0]` row-major,
`../gluon/pipeline/marker-and-version-gates.md ## The gfx942 async-copy width gate — available, and narrow`).
Contract: `../gluon/memory-reference.md ## Async Copy To Shared`. Verified on Triton 3.8.0 / gfx950
(CPU AOT) unless noted. Traps, in the order they bite:

- **A layout-contract violation fails with the same text as an absent op**: `failed to translate
  module to LLVM IR` / `unrealized_conversion_cast`. Hold op, arch and width fixed and vary **only the
  tiling**: gfx942 32-bit `buffer_load_to_shared` fails with `4 threads x 1 elem` over a 64-wide
  contiguous dim and runs bit-exact with `64 x 1`. Satisfy the contract before recording a ceiling.
- **The layout must cover the contiguous dim exactly; repetition along it fails** (a `BLK` covering
  `[64, 16]` over a `[32, 32]` tile). Repetition along the **outer** dim is fine: `[1,8]x[8,8]x[4,1]`
  fp16 over `[64, 64]` lowers to two `dwordx4 ... lds` per lane; over `[32, 128]` it fails;
  `size_per_thread` `[2, 8]` fails.
- **`add_coalesce_async_copy` legalizes by rewriting your layout, not by enabling it.** (Spliced by
  `patch_async_reinject.py` or run by `gluon_swp.py`.) It inserts a `ttg.convert_layout` on the
  pointer / mask / other tensors and clips the per-lane width to a legal direct-to-LDS size. The data still
  lands direct-to-LDS (`*_load_*_lds`, no data `ds_write`), but the address tensor makes an LDS round trip
  (`ds_write` / `ds_read` + `s_barrier`; compile-checked 3.8.0, gfx950 and gfx942). That bounce in the ISA
  means the source layout was not native. The pass cannot fix a shared order that disagrees with the
  source's fast dim. Gluon's own pipeline does not run it.
- **Padded async destination**: lowers iff interval_bytes >= 64 lanes x per-lane bytes and the
  producer is derived from its `offset_bases` (`[32, 64]` fp16, 8 rows per warp instruction: pad every
  64 elements fails, every 512 lowers). Forced through, it can be a **silent miscompile — fastest arm,
  NaN almost everywhere** — so copy-back check every padded async arm. Home:
  `../gluon/memory-reference.md ### Shared-layout family + transpose-on-read (layout dependency)`.
- **Proof async replaced staging: `ds_write == 0`** plus matching direct-to-LDS load count. It can
  still lose to sync staging (recorded slower on every minor on gfx942; not re-measured here, no gfx942 device) — A/B it;
  `scripts/pipeline_examples_cdna4.py` is the runnable gfx950 check.
- Accepted async **offset** layout families are version-gated (older minors: `Blocked` / `Slice`
  only, `DistributedLinearLayout` fails). Probe per build.

### Scheduling helpers and hints

`sched_barrier` / `sched_group_barrier` / `set_prio` are absent from `gl.amd.cdna3/.cdna4` on Triton
3.8.0 (checked; `../../playbook.md ## Knobs & pitfalls`). `warp_pipeline_stage` compiles; the
recorded wins are on loops with both LDS staging and compute (`../tile-programming/warp-pipeline.md`).
Signal that it has nothing to work with: occupancy < 2 waves/SIMD (nothing to interleave). On a two-stage
memory/compute loop (GEMM-like) the shipped convention (AMD gfx950 Gluon tutorials, aiter) is memory stage
`priority=1`, compute 0; the stated rationale is that a compute-high `setprio` starves the memory wave's
address VALU. On a gfx950 / Triton 3.8.0 two-stage warp-pipelined GEMM the inverted and equal polarities
measured the same as the shipped one (no overlap collapse), so on a two-stage GEMM it is a low-priority
knob. On the four-stage softmax-attention loop the reverse
polarity (matrix stages above memory) measured better only with P's `exp2` pinned in the PV stage; without
the pin neither polarity changed time (gfx950 / 3.8.0):
`../tile-programming/warp-pipeline.md ## Stage priority: the shipped memory-high convention`.

### Version- and build-sensitive surfaces

- 3.6.0 has only `gl.thread_barrier`, 3.7.0+ only `gl.barrier` (no alias: `AttributeError`). On the
  oldest minor a module-scope layout passed through a non-builtin helper or nested `@gluon.jit`
  fails with `AttributeError: '<SomeLayout>' object has no attribute '_flatten_ir'`. Fixes: pass it as a
  `gl.constexpr` kernel parameter, or use a real builtin that takes `layout=` (the `full` spelling, not
  `zeros`); annotating the global itself `: gl.constexpr` does **not** help (both spellings fail on that
  minor; not re-checked here, only 3.8.0 is installed). Neither crossing bites on 3.8.0 (compile-checked:
  a module-scope layout, annotated or not, through a nested `@gluon.jit` helper calling `gl.zeros`).
- `@triton.autotune(use_cuda_graph=True)` warns deprecated on 3.7.0 (not re-checked on 3.8.0), so it
  is a deprecated path to build on. Re-test AOT/prebuilt artefacts after any minor change.
- Fork-only env vars (`TRITON_ENABLE_LLIR_SCHED`, `TRITON_ENABLE_AMDGCN_AS`,
  `TRITON_ENABLE_AMDGPU_RA_HINTS`, `TRITON_GLUON_*`) are silent no-ops on stock 3.8.0
  (`../tile-programming/non-upstream-reserve.md`).

### Environment, install and cache

- **An editable install of a framework can silently swap the Triton package** (`triton`,
  `amd-triton`, `pytorch-triton-rocm`, `triton-rocm`). Opt out (e.g. `AITER_USE_SYSTEM_TRITON=1`)
  and record the installed Triton identity with every number.
- After a compile failure a stale `TRITON_CACHE_DIR` can serve the failed or a partial artefact;
  clear it between retries (`../method/benchmark-hygiene.md ## Artifact and cache hygiene`).

### Numerics traps

- **Masked-pad (`mask` + `other=0` into LDS) is a free pow2 pad for full loops but gives NaN at
  large-seqlen causal / short trip counts**: the masked LDS region stale-reads and padded rows hit
  `-inf` scores, `0 * inf = NaN`. A no-pad split avoids it (sync `convert_layout`, or a per-sub-tile async
  layout, `../tile-programming/layout-recipes.md`) on causal / short loops. GPU-observed.
- **A masked `buffer_load_to_shared(..., mask, other=0.0)` K tail returned wrong GEMM results even without
  warp-pipeline markers** (GPU-observed, gfx950 / Triton 3.8.0, not root-caused): the masked-off lanes appeared
  to leave stale LDS data. Zeroing the tail after the LDS read was correct. Check any masked LDS-destination copy
  by copy-back before relying on `other=` (`../tile-programming/warp-pipeline.md ### A GPU-verified gfx950
  snippet (Triton 3.8.0 and 3.9.0, `num_warps=8`)`).

## Benchmark-sensitive platform issues

Cache reuse, artifact path collisions, import routing that swaps the implementation, and Triton
minor changes that flip tuned choices (`matrix_instr_nonkdim`, `waves_per_eu`) all change the
measured path; record the environment (`../method/benchmark-hygiene.md`).

## gfx942 / CDNA3 hard failures that affect benchmark validity

Capability page: `../hardware/cdna3-gfx942.md`. These break a whole **sweep**, not one kernel:

- **LDS over the per-CU cap (160 KiB gfx950, 64 KiB gfx942)** — on Triton 3.8.0 a catchable
  `OutOfResources ... Hardware limit` at first launch, not a compile error and not a process abort
  (`../hardware/planning-constants.md ## Make an occupancy table falsifiable before you quote it`).
  Triggers: cshuffle epilogue (`2·tile_m·tile_n` LDS), `NUM_STAGES` whole large tiles. Pre-screen
  against `hw_constants.json` `lds_per_cu_kib`.
- **MFMA shapes gfx942 does not have:** bf16 `16x16x32` / `32x32x16` and scaled
  `mfma_scale_*_f8f6f4` (a4w4 / a8w4 / mxfp8). On the Gluon path (Triton 3.8.0) bf16 `16x16x32` is a
  catchable compile error (`no matching matrix core intrinsic`, compile-checked); a cannot-select
  process abort with no fallback was recorded on other paths. Confirm with `llvm-mc -mcpu=gfx942`; defer block-scaled
  formats to gfx950 (`cdna4.mfma_scaled`).
- **External ASM entries crash per entry** (decode: `hipModuleLaunchKernel ... context destroyed`,
  persistent variant hangs, prefill runs) — switch entry before declaring the op un-runnable.

## Toolchain-pin regressions: when the number you measured is not a ceiling

**Before recording an at-ceiling / stay-plain verdict, check the pin matches the one the reference
was taken on; a gap at unchanged registers/occupancy is a toolchain hypothesis first.** One re-pin
of `ROCm/gfx950-gluon-tutorials` (CHANGELOG) carried three independent regressions:

| signature | cause | falsifier |
| --- | --- | --- |
| spills on one config only, other arms untouched | new default flag **`amdgpu-use-amdgpu-trackers`** appended for gfx942/gfx950 | stop passing it — arm returns to published value (baseline loses a few %, so gate it) |
| multi-wave ping-pong GEMMs lose matrix efficiency | LLVM unsigned underflow in the register-pressure limit when reserved weight exceeds it (pressure checks silently off); reached via **`amdgpu-agpr-alloc=0,0`** on that re-pin's toolchain (on the stock Triton 3.8.0 wheel `"0,0"` compiled without an LLVM failure for the GEMMs tried) | apply the upstream fix (restored 4 kernels, badly slowed a 5th: falsifier != fix) |
| attention loses **several points** of in-loop MFMA efficiency, registers/occupancy/inventory unchanged | **warp-pipeline** pass stopped placing the loop wrap-around barrier (and its **`setprio`**) at the loop top unconditionally | rebuild with the old pass file only |

**An efficiency drop with no change in any resource reading points to a schedule the pin took away.**
Record the pin beside every number
(`../tile-programming/compiler-contract.md ## Toolchain identity`); discriminate with the same
source on the other pin or a non-Triton comparator; record it as a dated toolchain ceiling, not a hardware wall. Rebuilding the pass is not reachable on the
stock 3.8.0 wheel.

## Container-locus profiling

With `TILE_KERNEL_CONTAINER` set, host/container path mapping, env forwarding and IR caches go
through `locus.sh` / `dump_ir.sh`; do not reintroduce app-side workarounds or pick a device
outside GEAK's allocation (`../method/profile.md`).

## RDNA4 (gfx1201, observed on R9700): kept for GEAK's RDNA4 pages

Not part of the gfx950 / gfx942 scope above; kept because GEAK's RDNA4 pitfalls page points here.

### PMC availability: per-box preflight

`rocprofv3 --kernel-trace` captured dispatches on R9700, but the PMC counter set and naming on RDNA4 client
parts differ from Instinct (CDNA); CDNA names (`SQ_WAVES`, `VALUInsts`, the `MfmaUtil` / `VALUBusy` family)
may be absent or differ. The exact set depends on the device and the ROCm version.

- Before trusting a PMC-derived bound class on RDNA4, run `rocprofv3-avail list --pmc` on the box and confirm
  the counters you need exist. rocprofv3's own listing is `rocprofv3 -L` / `--list-avail`; older profiler
  generations used `--list-basic`, `--list-derived` or `--list-counters`, which is a CLI rename, not an
  image defect.
- If the discriminating counters are unavailable, do not substitute other counters for them. Fall back to the
  no-profiler evidence path: the analytical roofline (`../hardware/roofline-models.md`), a static `.s` audit
  (`kernel_workflow/scripts/kernel_tools/asm_loop_audit.py`), a floor probe, and A/B timing at the production
  boundary (`../method/profile.md`).

### `int64_strides` under a CUDA graph

On one Triton attention kernel on R9700, timed at the CUDA-graph boundary, `int64_strides=false` regressed
against `int64_strides=true`; the vectorized-load path appeared to fall back to a non-vectorized one in that
configuration. Rule: for gfx1201 attention kernels served under a CUDA graph, keep `int64_strides=true`
unless an A/B on the target shows otherwise. Scope: a single kernel at that boundary; not established as a
general RDNA4 rule. GEAK's product policy and evidence for these rules are R9700-only.
