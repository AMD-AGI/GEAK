# CDNA3 / gfx942 downgrade page (MI300X / MI325X / MI308X)

What changes when a gfx950 design moves to gfx942. Machine data: `hw_constants.json` `arch.gfx942`,
`extract_sku.py --facts --hardware-context $EVAL_DIR/hardware_context.json` (or `--facts MI300X`). ISA queries: `amd_isa.py --arch gfx942 ...` (instruction chapters;
matrix layout needs the Matrix Instruction Calculator, `primary-sources.md`).

### At a glance: gfx950 baseline → gfx942 downgrade

| gfx950 | gfx942 |
| --- | --- |
| async `buffer_load_to_shared` 128-bit + `commit_group` / `wait_group` ring | 32-bit only, via `gl.amd.cdna4.async_copy`; sync register staging is the default ring |
| `ds_read_b64_tr_*` | **none** |
| scaled MFMA (f8f6f4 + E8M0), fp4 | **cannot-select**: no fp4 / mxfp4 / mxfp8 matrix path (hardware ceiling) |
| 160 KiB LDS, 64 banks | **64 KiB**, 32 banks, 512 B min alloc, 16-bit M0 offset |
| bf16/fp16 16x16x32, 32x32x16 | 16x16x16, 32x32x8 only |
| OCP fp8 `float8e4nv` | FNUZ `float8e4b8` (`torch.float8_e4m3fnuz`); Gluon fp8 MFMA on FNUZ with `version=3` |
| 256 CUs, 8 TB/s, 2500 TF fp16 (MI355X) | 304 CUs (MI308X 80), 5.3 TB/s (MI325X 6.0), 1307.4 TF |
| `AMDMFMALayout(version=4)` | `version=3` |

VGPR file, granule and ladder are the same as gfx950 (`planning-constants.md ## VGPR / occupancy
thresholds (CDNA combined budget)`).

## MFMA (matrix core)

- **bf16: v3 `16x16x16` (k_width 4, `v_mfma_f32_16x16x16_bf16`) or `32x32x8`.** `instr_shape=[16,
  16, 32]` bf16 on gfx942 fails the pass manager / cannot-select (compile-verified 3.8.0).
- int8: `16x16x32` / `32x32x16`, int32 accumulator (k_width: `amd_isa.py --arch gfx942 facts`).
- fp8: ISA has `16x16x32` / `32x32x16` (FNUZ). In Gluon, `gl.amd.cdna3.mfma` (the same function as
  `cdna4.mfma`) with `AMDMFMALayout(version=3)` and `float8e4b8` / `float8e5b16` operands lowers to
  `v_mfma_f32_*_fp8_fp8` / `_bf8_bf8` (compile-verified 3.8.0; FNUZ numerics not GPU-checked). OCP
  `float8e4nv` fails with `version=3`; with `version=4` it compiles to the same instruction, which gfx942
  reads as FNUZ, so the numbers are **wrong**; `version=4` + FNUZ fails (an `Unsupported rhs dtype fp8e4b8`
  class of error). Plain `tl.dot(fp8e4nv)` on gfx942 silently upcasts to fp16; `tl.dot(fp8e4b8)` is native
  (**Overrides KB**: `perf_knowledge/languages/triton_amd/isa_verify.md` "If the build refused to lower the dot"; on 3.8.0 it compiles, so look for `v_mfma_f32_*_f16` in an fp8 kernel).
  Convert OCP producers to FNUZ.
- `v_cvt_pk_bf16_f32` is CDNA4-only: bf16 output packing is the `and/or/shift` sequence.
- `v_cvt_off_f32_i4` assembles on gfx942, but `hasattr(rocdl, 'cvt_off_f32_i4') == False`: a
  binding gap, reachable via inline asm.

## LDS / memory path

- **64 KiB LDS is often the binding constraint.** A large explicit multi-buffer (e.g. `128x256`, `BK=64`,
  2 stages) compiles and fails at first launch with `OutOfResources ... Hardware limit: 65536`
  (`planning-constants.md ## Make an occupancy table falsifiable before you quote it`); pre-screen
  `stages * tile bytes <= 64 KiB` before a sweep.
- **Async direct-to-LDS works via the `cdna4` namespace:** `gl.amd.cdna4.async_copy.buffer_load_to_shared`
  + `commit_group` / `wait_group` + `load_shared_relaxed` compile and run; `gl.amd.cdna3` has no
  `async_copy`; the generic `ttg.async_copy_global_to_local` also lowers on gfx942 at 32 bit
  (`global_load_lds_dword`) and fails above it, like the `cdna4` entry points (a width gate, not an op ban).
  `is_async_copy_enabled(gfx942) == False` only means the plain pipeliner does not use it.
- Width {32} only; exact coverage of the fast dim; destination (unswizzled or swizzled) `order` matching the source's fast dim (`[1,0]` row-major); a padded
  destination lowers only with a producer derived from its `offset_bases` (CPU AOT, 3.8.0).
  Narrow DMA often turns `s_waitcnt`-bound - A/B vs sync staging
  (`capability-matrix.md ## Direct-to-LDS granularity (per arch)`).
- `builtin.unrealized_conversion_cast` around v3 + `SwizzledSharedLayout` operands reproduced (compile-only,
  3.6-3.8) only for an async copy whose producer does not exactly cover the destination; transpose-B reads
  themselves lowered -> exact-cover producer, or `PaddedSharedLayout` + `DistributedLinearLayout`
  (`../gluon/memory-reference.md`).
  `compute_efficient_padded_shared_layout` asserts v4-only.
- Conflict-free `ds_read_b128` interval **unverified** on gfx942: `hw_constants.json` says 16 cyc, the
  bandwidth model (64 lanes x 16 B x 4 SIMDs / 128 B/clk) gives 32; calibrate on an ATT trace before reading 32 as a
  conflict (`planning-constants.md ## LDS throughput / conflict (planning)`). Full-conflict stride 128 B;
  bank `(addr/4) % 32`.

## Pipeline / scheduling

- Same as gfx950: no Gluon pass reads `num_stages`; hand-written overlap (register prefetch ->
  authored ring, sync staging by default -> `warp_pipeline_stage` at `num_warps>=8`). Re-injected
  plain pipeliner = diagnostic only, and stages synchronously here.
- Stock `coexec` strategy unavailable (gfx1250-only on 3.8.0); per-compile `llvm_fn_attrs` ILP strategies
  are the available scheduler route (`../tile-programming/llvm-fn-attrs.md`).
- **A CDNA4-developed LLIR schedule pass skips every gfx942 region silently** (observed on an authored /
  fork pass, not present in stock 3.8.0; not re-run here): its cost model lacks
  `16x16x16` / `32x32x8` (unpriced = 0 cycles). Recalibrating needs the shape rows **and** the
  window model: gfx942 co-issue is **12 of 16** (`16x16x16`) and **28 of 32** (`32x32x8`)
  (`isa-mechanisms.md ## Matrix/VALU co-execution — the CDNA-only overlap budget`). This is a
  cost-model boundary, not silicon.
- `TRITON_ALWAYS_COMPILE=1` forces recompiles while iterating.

## Known tooling issues (gfx942)

- `TCC_EA_RDREQ_sum`, `MemUnitBusy` and `L2CacheHit` are absent on gfx942; list the build's counters with
  `rocprofv3 -L`.
- **rocprof-compute on gfx942 / MI325X:** `Incomplete class definition for gfx942. Expecting
  populated vbios but detected None` / `Missing specs fields for gfx942`, and **no `roofline.csv`** (observed on
  MI325X). Version-dependent (source-verified): in rocprof-compute 3.1.0 (ROCm 6.4.1) the spec warning is fatal
  after counter collection, so no `roofline.csv`; from 3.3.0 it is only a warning and profiling continues. Not
  re-run on a gfx942 device.
  SOL / warp-state / memory counters are still valid: read `pmc_perf.csv` with `parse_pmc.py` and
  compute the roofline yourself (`counters` over `datasheet` until `mem_bw_probe.py` gives an
  in-shape ceiling; `roofline-models.md ## Every "% of roofline" carries two bases (the reporting
  contract)`).
- The gfx950 caveats (rocprof-compute >= 3.6.0 for `--roof-only`, the `TCC_HIT` write-request count) are
  different problems; a workaround for one does not address the other.
- A gfx950 kernel usually does not run as-is here: audit `arch == "gfx950"` branches, v4 layouts, `cdna4.*`
  MFMA, 160 KiB-sized tiles, 128-bit async and `mfma_scaled` before porting.
