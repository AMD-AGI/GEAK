# CDNA3 / gfx942 capability page (MI300X / MI308X / MI325X)

**The gfx942 downgrade comparison.** gfx950 (CDNA4, MI355X / MI350X) is the main line of this
skill: every mechanism, recipe and default is written for it first. This page is where gfx942
(CDNA3, MI300X / MI325X / MI308X) is compared against that baseline — for each fact, **what is
unavailable, what changes, and by how much** — and it owns the CDNA3 deltas that the shared pages
(`capability-matrix.md`, `planning-constants.md`, `isa-mechanisms.md`) only point to, so those pages
stay gfx950-primary. gfx942 is still a supported target, not a footnote.

Every value is **calibratable** and every capability is **probe-then-trust** (compile
/ `hasattr` / `llvm-mc` / correctness), never assumed. Machine mirror (GEAK repo root):
`perf_knowledge/hardware/data/hw_constants.json` (`arch.gfx942`) + `perf_knowledge/hardware/data/sku.json`
(MI300X / MI325X / MI308X). SKU peaks/ridge: `amd-cdna3-skus.md`. GEAK's CDNA3 cards carry the
datasheet / microarchitecture side: `perf_knowledge/hardware/cdna3_mi300/{arch,matrix_core,memory_hierarchy,occupancy,xcd_chiplet,isa_notes}.md`
(their 16-VGPR granule is superseded by the measured granule-8 ladder below).

### At a glance: gfx950 baseline → gfx942 downgrade

| gfx950 mechanism (baseline) | on gfx942 |
| --- | --- |
| async `buffer_load_to_shared` at 128-bit (16 B/thread) + `commit_group`/`wait_group` LDS ring | **narrows** to 32-bit (4 B/thread), clean tiling, swizzled destination `order=[1,0]`; often slower than **sync register staging**, which is the default ring there |
| `ds_read_b64_tr_*` LDS→MFMA transpose | **unavailable** |
| scaled MFMA / MXFP (fp8/fp6/fp4 + E8M0) | **unavailable** (cannot-select) — fp4 has no matrix path |
| 160 KiB LDS / CU, 64 banks | **64 KiB** (2.5× less), 32 banks — tiles and stage counts sized for gfx950 do not fit |
| 16×16×32 / 32×32×16 bf16/fp16 MFMA | 16×16×16 / 32×32×8 only (K halves) |
| OCP fp8 (`e4m3fn`) | **FNUZ** (`e4m3fnuz`); Gluon fp8 MFMA blocked — plain `tl.dot` comparator |
| MI355X 2500 TF fp16 / 8.0 TB/s (ridge ~312) | MI300X 1307.4 TF / 5.3 TB/s (ridge ~247); MI325X same compute, 6.0 TB/s (ridge ~218) |
| 256 CUs | 304 (MI300X/MI325X), 80 (MI308X) |

## Silicon constants

| Resource | Value | Note |
| --- | --- | --- |
| Family / wave | CDNA3 / **wave64** | `AMDMFMALayout(version=3)` |
| Active CUs | **304** (MI300X / MI325X) · **80** (MI308X) | `sku.json`; grid should target a CU multiple |
| SIMDs / CU | 4 | |
| HBM | MI300X 192 GB, ~5.3 TB/s · MI325X 256 GB, ~6.0 TB/s (vs gfx950 8.0) | FP16 ridge ~247 ops/B (MI300X), ~218 (MI325X), ~65 (MI308X); datasheet basis — ranks only |
| **LDS / CU** | **64 KiB** (vs gfx950 **160 KiB**) | the dominant CDNA3 planning constraint — see LDS section |
| LDS banks / stride | 32 banks, 4 B | `(addr/4) % 32` |
| LDS min alloc | 512 B | |
| LDS peak read | ~128 B/clk | vs gfx950 ~256 B/clk |
| VGPR budget | 512 / SIMD (**arch + accum combined**) | wave steps (granule 8, cap 8): `<=64->8w <=72->7w <=80->6w <=96->5w <=128->4w <=168->3w <=256->2w <=512->1w >512 spill`. Feed it the **allocated** count `ceil8(ceil4(next_free_vgpr))`, not a printed `.vgpr_count` / `n_regs`; these are **waves/SIMD**, not wg/CU. Full rule + caveats: `thresholds.json confirm.vgpr_wave_cliff`, `planning-constants.md ### Read the field the hardware reads, and round it before you compare` |
| TCP in-flight cap | 32 KiB / CU | |
| LDS offset | 16-bit M0 (vs gfx950 18-bit) | |

## MFMA (matrix core)

Confirm any shape with the stock assembler before building a lever around it:
`echo '<insn>' | /opt/rocm/llvm/bin/llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx942`.

| Fact | Consequence |
| --- | --- |
| **v4 MFMA 16x16x32 bf16 = cannot-select (hard abort)** | use the **v3** `16x16x16` (kw4) or `32x32x8` bf16 shape; a K16 bf16 MFMA is silicon-impossible on gfx942 (assembler-rejected) |
| int8 | v3 `16x16x32` (kw16) or `32x32x16`, int32 accumulator |
| **scaled MFMA `mfma_scale_*_f8f6f4` = cannot-select** | fp4 / mxfp4 / a4w4 / a8w4 / mxfp8 have **no** scaled-MFMA path on gfx942 → hardware ceiling (a genuine `does-not-lower`, not a wrapper gap) |
| `v_cvt_pk_bf16_f32` (pack 2×f32→2×bf16) is CDNA4-only | bf16 output packing cannot be cheapened; the truncation `and/or/shift` sequence is the only path |

Default MFMA shapes (ISA): fp16/bf16 `16x16x16`, `32x32x8`; fp8 / int8 `16x16x32`,
`32x32x16`. In **Gluon**, fp8 MFMA is blocked (`gl.amd.cdna3.mfma` fp8 = API-blocker;
`gl.amd.cdna4.mfma` fp8 on gfx942 = wrong-result) — use **plain `tl.dot`** as the
first-class fp8 GEMM comparator (`capability-matrix.md ## Matrix operation matrix`).

## dtype

| dtype | gfx942 fact |
| --- | --- |
| fp8 | **`torch.float8_e4m3fnuz`** (fp8e4b8), **not** OCP `e4m3fn`. Raw fp8 MFMA can report `Unsupported rhs dtype fp8e4b8` → **dequant-cast** to bf16 for the matmul, or use plain `tl.dot`. |
| int4 → bf16 dequant | `hasattr(rocdl, 'cvt_off_f32_i4') == False` on gfx942 — a **binding gap, not silicon**: `v_cvt_off_f32_i4_e32` assembles on gfx942 too (`llvm-mc -mcpu=gfx942`), so the fast path is still reachable via inline asm / the raw intrinsic; see `../tile-programming/low-precision.md`. |

## LDS / memory path

- **64 KiB LDS/CU is the binding constraint.** An explicit double buffer allocates
  `NUM_STAGES` whole tiles; a large tile (e.g. `128x256` with `BK=64`) blows the 64 KiB
  cap and aborts at the **process level** (`local memory exceeds limit`, not catchable)
  — cap stages, use an 8-wave / smaller-tile / cshuffle variant, and pre-screen
  `2·tile_m·tile_n ≤ LDS_cap` before a config sweep (`../method/benchmark-hygiene.md`).
- **Async direct-to-LDS IS available on gfx942 — via the `cdna4` namespace.** The
  `gl.amd.cdna3` namespace has no `async_copy` submodule, but
  `gl.amd.cdna4.async_copy.buffer_load_to_shared` + `commit_group` / `wait_group` +
  `load_shared_relaxed` **compile and run** on gfx942. The generic
  `ttg.async_copy_global_to_local` is explicitly **illegal** on the gfx942 backend.
  `is_async_copy_enabled(gfx942) == False` only means the **auto-pipeliner** does not use
  async — it does not mean async is unavailable (`../gluon/memory-reference.md ## gfx950
  <-> gfx942 delta`).
- **Direct-to-LDS width = 32-bit (4 B / 2×bf16) per thread** (`supportsDirectToLdsLoadBitWidth`
  CDNA3 = {32}; 128-bit is CDNA4-only). The fast dim must be exactly covered by
  `threads_per_warp * size_per_thread` (no replication), and the destination is a swizzled
  shared layout with `order=[1,0]`; as on gfx950, a **padded** async destination has failed LLVM
  translation, so build the ring swizzled first. Narrow DMA → async often turns
  `s_waitcnt`-bound and net-negative — A/B vs sync register staging, don't assume a win
  (`capability-matrix.md ## Direct-to-LDS granularity (per arch)`).
- Transpose-B / complex dot-operand offset layouts under v3 + `SwizzledSharedLayout` can
  **fail LLVM translation** (`builtin.unrealized_conversion_cast`) → use
  `PaddedSharedLayout` + a source-proven `DistributedLinearLayout` matched to the consumer.
- `compute_efficient_padded_shared_layout` asserts **v4-only** (unavailable on gfx942).
- `ds_read_tr` LDS→MFMA transpose is **CDNA4-only** (none on CDNA3).
- LDS conflict: conflict-free `ds_read_b128` steady interval ~16 cyc; full-conflict stride 128 B.

## Pipeline / scheduling

- No auto software-pipeliner runs in the Gluon lowering, and `num_stages` is consumed by no Gluon
  pass in 3.8.0 (budget parameter / champion record only). The overlap is **hand-written**, in
  the order `../tile-programming/pipeline.md` defines once; on gfx942 the rungs downgrade as
  follows: (1) register-level prefetch — unchanged; (2) the authored LDS ring — **sync register
  staging** by default, 32-bit async (`order=[1,0]`) only where it A/Bs faster, and stage count
  capped by 64 KiB; (3) `warp_pipeline_stage` + scheduling-model choice — available, gated on
  `num_warps>=8` as on gfx950 (`../tile-programming/scheduling-model.md`,
  `../tile-programming/warp-pipeline.md`).
- **Re-injecting plain's pipeliner is the lowest rung**, as on gfx950: a below-parity diagnostic
  that measures the `lost_pipeline` debt, or a last resort when the hand-written ring cannot reach
  parity; its numbers are labelled `injected`, are never a win, and it is never applied to an
  incumbent (already-Gluon) kernel. It needs no rebuild and no installed-file edit, and **no
  Triton environment variable arms it** — it is armed from Python, or by `TRITON_GLUON_SWP=N` for
  the on-disk patch form, a variable read by this pack's `scripts/patch_reinject.py`, not by Triton
  (`../tile-programming/pipeline.md ## Reproduce plain's software pipeline on the Gluon path (the parity-recovery route)`,
  `../method/recover.md`). Probe the *mechanism* (does the IR change?), never a variable name.
  Note `is_async_copy_enabled(gfx942) == False`: a re-injected pipeline on gfx942 stages
  synchronously.
- `TRITON_ALWAYS_COMPILE` is real and upstream: force recompile while iterating.
- `sched_barrier` / `sched_group_barrier` / `iglp_opt`: no user-facing surface on 3.8.0 (and an
  API-blocker on some 3.7.0 builds); the `warp_pipeline_stage` markers emit `s_setprio` +
  `sched_barrier` for you. The canonical statement is
  `../tile-programming/scheduling-model.md` / `../tile-programming/warp-pipeline.md`. The stock
  coexec scheduler strategy is **off by default on gfx942** (as on gfx950; 3.8.0 enables it
  automatically only on gfx1250) — opt in with `TRITON_HIP_USE_COEXEC_SCHEDULER=1` (at
  `num_warps <= 4`) or per compile via `llvm_fn_attrs` (`../tile-programming/llvm-fn-attrs.md`),
  and accept it on an assembly diff. Note gfx942's shapes differ from the gfx950 ones any
  co-execution budget was calibrated on (`isa-mechanisms.md ## Matrix/VALU co-execution — the CDNA-only overlap budget`). The
  source-level absence does not by itself close the LLVM-level route
  (`../tile-programming/llir-codesign.md ## Declaring the interleave`).
- **An LLIR schedule pass developed on CDNA4 will skip every region here, and it looks like a
  hardware wall.** Its per-shape cost model carries the CDNA4 matrix shapes; gfx942's
  `16x16x16` / `32x32x8` are absent, an unpriced shape yields a zero cycle cost, and the region
  is dropped **silently**. This is a **cost-model** boundary, not silicon — the matrix/VALU
  overlap mechanism itself exists on the whole CDNA family
  (`isa-mechanisms.md ## Matrix/VALU co-execution`).
- **Recalibrating such a pass for gfx942 needs two changes, not one.** (a) the shape rows; and
  (b) the **co-execution window model**, because gfx942's authoritative co-issue counts are
  **12 of 16** (`16x16x16`) and **28 of 32** (`32x32x8`) — and `12` is *not* "interval minus the
  constant calibrated on CDNA4's 32-cycle case". Carrying that formula across is the error to
  avoid; the counts are MI-calc-published for CDNA3 (which is the arch the calculator *does*
  cover). Until both are done, record a scoped tool ceiling and fall back to the wave-level
  stage markers or a hand-authored interleave
  (`../tile-programming/llir-codesign.md ## Applicability gate: shapes the tool does not model`).

## External-kernel entry points (when an op ships an ASM/asm-hybrid path)

Signatures only (mechanism, not a specific repo): on this class of box an ASM **decode**
kernel has been seen to crash at launch (`hipModuleLaunchKernel ... context destroyed`),
while the **prefill / regular** ASM entry runs, and a **persistent** variant hangs. If one
entry is un-launchable, **switch entry** rather than declaring the op un-runnable — a
correct alternative entry that passes the oracle is an unblock win. Full signatures:
`../pitfalls/platform-known-issues.md`, `../pitfalls/negative-patterns.md`.

## Cross-arch gate (CDNA4 → CDNA3 warm-start / port)

A gfx950 champion does **not** run as-is on gfx942. Before reusing one, audit its
`get_rocm_arch()` branches for gfx950-only pins and downgrade the anchor to a
**reference** (full transcription / port + re-recover layouts on gfx942):

- `arch == "gfx950"` / v4 layouts / `cdna4.*` MFMA / 160 KiB LDS assumptions;
- tile sizes sized for 160 KiB LDS (won't fit 64 KiB);
- async width / 8-wave / DMA assumptions;
- `mfma_scaled` / scaled paths (cannot-select on gfx942).

A cross-arch pin that leaves the anchor **un-runnable** on gfx942 is an
**unblock win** once any correct candidate passes the oracle
(`../method/close.md`, "Three acceptance bars").

## Known tooling issues (gfx942)

Vendor-tool gaps that are NOT kernel problems — recognize them so you don't misread a
degrade as a kernel bug:

- **rocprof-compute device spec is incomplete for gfx942 / MI325X.** The log shows
  `Incomplete class definition for gfx942. Expecting populated vbios but detected None`
  and `Missing specs fields for gfx942`, and **`roofline.csv` is not produced** — so the
  rocprof-compute *roofline* CLI path is blind on this arch. This does **not** invalidate the
  SOL / warp-state / mem counters, which `pmc`-live still samples correctly.
- **Correct path: read the PMC directly.** `kernel_workflow/scripts/kernel_tools/parse_pmc.py`
  reads `pmc_perf.csv` and derives the SOL/intensity fields without the roofline CLI — this is the
  *sanctioned* gfx942 path, not a workaround. Do not block on the missing `roofline.csv`; the
  roofline is computed from `roofline-models.md` + the measured PMC, not from the vendor CLI, and is
  labelled `numerator_basis = counters`, `denominator_basis = datasheet` until an in-shape probe
  (`mem_bw_probe.py`) replaces the denominator (`roofline-models.md ## Every "% of roofline" carries
  two bases (the reporting contract)`). Profiles are taken through GEAK's
  `kernel_workflow/scripts/profile_kernel.sh` under `gpu_lock.sh`.
- **Contrast with gfx950:** there the empirical roof needs rocprof-compute ≥ 3.6.0 (`--roof-only`
  on gfx95x), and the `FETCH_SIZE` / `TCC_BUBBLE` read bytes under-count — neither caveat is the
  gfx942 one above, so do not carry a gfx942 tooling workaround to gfx950 or vice versa.
- **VGPR occupancy driver comes from the ISA KD, not the roofline tool.** `next_free_vgpr`
  (`.amdhsa_next_free_vgpr`) is parsed by the static ISA audit (`asm_loop_audit.py` prints the
  kernel-descriptor register budget) → `static.vgpr`. This is the occupancy-wall evidence line;
  it is independent of the rocprof-compute spec gap.
