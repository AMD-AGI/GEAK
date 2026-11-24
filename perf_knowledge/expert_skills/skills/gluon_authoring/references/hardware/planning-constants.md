# Planning constants (gfx950, gfx942 downgrade)

Numbers to plug into `roofline-models.md`. Machine forms: `perf_knowledge/hardware/data/`
`hw_constants.json` (per-arch; its gfx950 values match this pack's measured values for the LDS granule,
co-issue costs, MFMA cadence and co-issue fit for the shapes it lists, and the LDS FIFO (`null`); where a
JSON field is a single number or lists fewer shapes than a page's per-shape or per-pass table, the table is
the finer statement), `sku.json` (per-SKU
peaks and MI350X measured ceilings: `extract_sku.py --facts --hardware-context $EVAL_DIR/hardware_context.json`). ISA capacities:
`amd_isa.py --arch gfx950 limits` / `lds`. RDNA / gfx1250 are out of scope for this pack.

## Targets (planning)

| Resource | gfx950 (MI350X / MI355X, CDNA4) | gfx942 (MI300X / MI325X / MI308X, CDNA3, downgrade) |
| --- | --- | --- |
| Wave / matrix | wave64, MFMA + scaled MFMA, `AMDMFMALayout(version=4)` | wave64, MFMA, `AMDMFMALayout(version=3)` |
| CUs | 256 (8 XCD x 32), 4 SIMD/CU | 304 (MI300X/MI325X), 80 (MI308X) |
| HBM | ~8.0 TB/s (both SKUs) | MI300X 5.3, MI325X 6.0 TB/s |
| Matrix peak | MI355X 2500 TF fp16, MI350X 2300 TF - never assume equal | 1307.4 TF fp16 (MI300X = MI325X) |
| LDS / CU | **160 KiB** (163840 B, no reserved region), 64 banks x 4 B | **64 KiB**, 32 banks x 4 B |
| LDS alloc granule | 1280 B (`lds_alloc_granule_bytes`; 320 dwords, LLVM) | 512 B (`lds_alloc_granule_bytes`; not measured on gfx942) |
| LDS peak read | ~256 B/clk | ~128 B/clk |
| L1 (TCP) | 32 KiB/CU; in-flight cap 32 KiB/CU (not observable on MI350X: a streaming read reached the HBM ceiling at fewer bytes in flight per CU) | same |
| L2 / MALL | 32 MiB (4 MiB per XCD) / 256 MiB MALL datasheet (memory-side, **not** L2; MI350X reports 224 MiB — `sku.json mall_mb_device`, used by `hw_budget.py`'s residency check) | 32 MiB / 256 MiB (MI308X L2 16 MiB) |
| VGPR | 512 / SIMD, arch + AGPR **combined** | same |
| Async direct-to-LDS | 128, 96 or 32 bit per lane, plus byte/short forms (encoding table) | 32 bit per lane, plus byte/short forms (encoding table) |
| `ds_read_*_tr` | yes | **none** |
| fp8 | OCP `float8e4nv` | FNUZ `float8e4b8` |
| Transcendental rate | `isa-mechanisms.md ### What fits in the window` | same class |

Full gfx942 delta page: `cdna3-gfx942.md`.

## LDS throughput / conflict (planning)

```text
gfx950: conflict-free ds_read_b128 steady interval ~16 cyc with all 4 SIMDs reading (8 with one wave/CU);
        lane-to-lane stride 32 B = 2-way, 64 B = 4-way, 128 B = 8-way, 256 B = full conflict (timing microbenchmark, one MI350X node, 2026-11)
gfx942: conflict-free ds_read_b128 steady interval UNVERIFIED (table: 16 cyc; the bandwidth model
        64 lanes x 16 B x 4 SIMDs / LDS B/clk with 128 B/clk gives 32); full-conflict stride 128 B
bank index: gfx950 (addr/4)%64 ; gfx942 (addr/4)%32
```

Machine form: `hw_constants.json ds_read_b128_conflict_free_cyc`. gfx950's 16 is GPU-measured (s_memtime
microbench and ATT, all four SIMDs issuing; 2-way 32, 4-way 64) and matches the model 64 lanes x 16 B x 4 SIMDs / 256 B/clk = 4096 B / 256 B/clk.
It is load-dependent: one wave per CU reads a conflict-free b128 every 8 cycles and hides 2-way conflicts, and
SIMD pairs {0,1} / {2,3} share read throughput - compare against a conflict-free baseline under the same load
(`isa-mechanisms.md ## LDS instruction behavior (gfx950)`). The same model gives 32 on gfx942,
which disagrees with `hw_constants.json` (16); neither ATT data nor an ISA statement settles it, so on gfx942
calibrate the conflict-free interval on your own trace before reading 32 as a 2-way conflict.

## VGPR / occupancy thresholds (CDNA combined budget)

```text
arch_vgpr + accum_vgpr share ONE 512-entry file per SIMD. Granule 8, cap 8 waves:
waves/SIMD = min(8, 512 // (8 * ceil(next_free_vgpr / 8)))

<= 64 -> 8 ; <= 72 -> 7 ; <= 80 -> 6 ; <= 96 -> 5 ; <= 128 -> 4
<= 168 -> 3 ; <= 256 -> 2 ; <= 512 -> 1 ; > 512 -> SPILL
```

**The granule is 8 (not 4, not 16) and the ladder has eight rungs; the 3-wave boundary is 168.**
GPU-measured on gfx950 / Triton 3.8.0 (resident workgroups per CU with the VGPR footprint forced by inline asm):
the measured count matched this ladder and LLVM's `; Occupancy` at every point tried, and not granule 16. The
AGPR part counts from `accum_offset` (rounded up to 4) plus `.agpr_count`.
Arbiter: `kernel_workflow/scripts/kernel_tools/amd_occupancy.py` (`waves_by_vgpr`); same ladder in
`hw_constants.json vgpr_wave_steps` (listed through the 2-wave rung; above 256 the kernel runs 1 wave). **Overrides KB** (gfx942 and gfx950):
`perf_knowledge/optimization/occupancy_and_registers.md` uses 16-register granules
(`floor(512 / round_up(VGPR_used, 16))`); verified by a compile-only check of LLVM's own occupancy report:
kernels using 65 / 72 / 73 / 104 VGPRs print `; Occupancy: 7 / 7 / 6 / 4` on both arches, which only granule 8
gives (granule 16 would give 6 at 72). The same basis overrides the granule-16 rule wherever an unlabelled page in
`perf_knowledge/` states it for gfx942 / gfx950 (gfx90a pages are not covered), among them:
`perf_knowledge/hardware/shared/wavefront_simd_vgpr_agpr.md` ("VGPRs allocate in blocks of 16."; there AGPRs
also sit outside the occupancy budget, but on gfx942 / gfx950 they come from the same 512-entry file and count
toward it, row VGPR above), `perf_knowledge/hardware/cdna3_mi300/occupancy.md` ("rounded up to a multiple of
16", "### The 16-VGPR granule trap"), `perf_knowledge/hardware/cdna4_mi350/arch.md` ("512 ×4 B/SIMD,
16-granule"), `perf_knowledge/languages/triton_amd/knobs.md` (`round_up_16(vgpr_used) · waves_per_eu ≤ 512`),
`perf_knowledge/languages/triton_amd/isa_verify.md` (`max_waves = floor(512 / round_up_16(vgpr))`),
`perf_knowledge/languages/hip_cpp/pitfalls.md` (`round_up_16(vgpr_used)`), the "granularity 16" rows of
`perf_knowledge/languages/{hip_cpp/overview,triton_amd/overview}.md`, and the "16-granule" wording in
`perf_knowledge/languages/{triton_amd/overview,asm_mfma/overview,asm_mfma/register_alloc}.md` and
`perf_knowledge/{hardware/cdna3_mi300/arch,index/glossary,operators/elementwise/tuning,optimization/README}.md`. Read `.vgpr_count` / `.amdhsa_next_free_vgpr` as already combined (below).

### Read the field the hardware reads, and round it before you compare

- **Use the kernel descriptor's `.amdhsa_next_free_vgpr`** (arch VGPR + AGPR) as the combined count.
  In LLVM 22 / Triton 3.5-3.8 artifacts AMDHSA **`.vgpr_count` is also combined**; LLVM
  `; NumVgprs:` is architectural-only and `; NumAgprs:` accumulator-only - do not add AGPR to
  `.vgpr_count` again. Front-end `n_regs` and profiler `VGPR` fields are not substitutes (in one sweep, on an
  older toolchain, a front-end register print stayed flat while the descriptor count moved across several
  wave rungs: the compiler was paying in AGPRs; which field it was is not recorded). `waves_per_eu=W` pads
  `.amdhsa_next_free_vgpr` up to the W-wave rung (and so caps a lighter kernel at W waves) while
  `kernel.n_regs` keeps the demand (GPU-measured, gfx950 / 3.8.0).
- **The hardware allocates `ceil8(ceil4(next_free_vgpr))`** (`COMPUTE_PGM_RSRC1[5:0]`,
  `allocated = (granulated + 1) * 8`); measured 0..+7 above the symbolic count. 97 -> 104 = 4
  waves/SIMD, not `512 // 97 = 5`.
- **Round before comparing configurations:** 140 and 144 both allocate 144; 130 and 134 both 136.

Tools: `probe.py measure --dir <ir>` / `asm_loop_audit.py` print the KD block. Verified on compiled
artifacts (LLVM 22, Triton 3.8.0).

## waves/SIMD is not workgroups/CU — convert before spending a round on it

```text
wg_per_CU      = min(waves_per_SIMD_by_VGPR * simd_per_cu // num_warps,
                     LDS_per_CU // (ceil(lds_bytes_per_wg / granule) * granule))
                 # granule: 1280 B gfx950, 512 B gfx942 (hw_constants.json lds_alloc_granule_bytes)
waves_per_SIMD = wg_per_CU * num_warps / simd_per_cu
```

- The two units coincide only at `num_warps == 4` (4 SIMDs/CU); an occupancy table without `num_warps`
  is ambiguous.
- At `num_warps=8` only the VGPR thresholds 64 / 80 / 128 / 256 move `wg/CU` (4 / 3 / 2 / 1);
  129-256 is one plateau.
  Measured: cutting `next_free_vgpr` 240 -> 146 bought **zero** extra workgroups/CU.
- **The `vgpr_count` ceiling rule (the one statement other pages point to):** a CTA of `num_warps >= 8`
  needs 2+ waves per SIMD, so its per-wave budget is at most 256 and a "VGPR <= 256" entry condition cannot
  fail — a spilling build still reports `vgpr_count <= 256` (one read `vgpr_count: 256` with hundreds of
  spill slots). A 4-warp CTA can run at 1 wave/SIMD with a 512 budget, so there the check can fail. Either
  way read `.vgpr_spill_count` in the same row.
- Compute both limiters and take the `min`; which binds changes after structural edits.

## The emitted `; Occupancy: N` is a register-term answer

- **No LDS term:** on 6 dumps it equalled the granule-8 ladder value above exactly. On a kernel with **dynamic
  LDS** (what Triton/Gluon emit; the block prints `; LDSByteSize: 0 bytes/workgroup (compile time
  only)`) it overstated the real limit (one case printed 6 against a real
  `163840 // roundup(38144, 1280) = 163840 // 38400 = 4 wg/CU`, which at 256-thread workgroups (4 waves) is 4 waves/SIMD; GPU-measured residency on other
  gfx950 / Triton 3.8.0 GEMMs was also below the printed value). Hand `min(VGPR-limited, LDS-limited)` is the
  correct value. Static-LDS behaviour
  unverified.
- **It inherits a declared waves-per-EU hint:** `next_free_vgpr` = `max(demand, hint floor)`, so on a
  hinted build both describe the request; a hint can also hold occupancy down.
- **LDS per workgroup for a JIT kernel is `metadata.shared`** (compiled kernel metadata); its ELF
  `.group_segment_fixed_size` reads **0** (= "unlimited" to a calculator). For an AOT binary
  `.group_segment_fixed_size` is the real value. Mixing them can reverse a JIT-vs-AOT comparison.

## Occupancy is a one-sided criterion

An occupancy **loss** may veto a change that spends LDS/VGPR; a computed occupancy **gain** does not
predict a speedup. Measured on CDNA4: a sizeable occupancy gain timed as a tie (and lost on the buckets it should
have won); halving occupancy cost little (and was faster on large-M); a small gain that won isolated
shrank to a tie end to end.
Occupancy serves compile-side feasibility (fits, spills, allocated count, next `wg/CU` threshold);
a timed occupancy change is readable with a pre-registered direction and a floor (this axis's null is a fraction of a percent).

## Tail efficiency is one-sided too

`tail_efficiency` (`roofline-models.md ## Saturation / wave quantization`) can veto a geometry; it
does not by itself rank two configurations. Before attributing an A-vs-B deficit to it, compute it **for both
arms at the same shape** - a measured case had an identical value for the winner and the loser.

## Make an occupancy table falsifiable before you quote it

- **LDS over the cap is a catchable launch-time `OutOfResources`, not a compile error and not a process
  abort (Triton 3.8.0):** `out of resource: shared memory, Required: <N>, Hardware limit: 163840`
  (gfx942: `65536`). The compile succeeds on both arches (AOT `triton.compile` returns a binary with
  `metadata.shared = N`); the first launch compares `metadata.shared` with the device's
  `max_shared_mem` and raises (`triton/compiler/compiler.py`, arch-independent). GPU-verified on gfx950
  for plain `@triton.jit` and Gluon kernels; exactly 163840 B runs. So pre-screen with
  `metadata.shared` / `probe.py measure` and catch `OutOfResources` in a sweep; it is a capacity
  statement, not a compiler bug (capacity failures move when capacity changes; a real lowering defect
  such as `unrealized_conversion_cast` survives a 2.5x LDS cut). LLVM's own `local memory (N) exceeds
  limit (M)` diagnostic applies only to LDS LLVM allocates itself, which a 3.8.0 Triton/Gluon kernel
  does not (its shared pool is an unsized external symbol); if a sweep dies with it, the variant came
  from another stack — isolate variants one per subprocess.
- **Shared-memory dims must be powers of two** - `allocate_shared_memory` with a non-power-of-2 dim (and a
  `.slice` to one) raises the same catchable `CompilationError` as `gl.arange` ("Shape element ... must be a power
  of 2"; compile-checked on Triton 3.6.0, 3.7.0, 3.7.1, 3.8.0, gfx950). A process abort was not reproduced; pad or
  split to powers of two.
- Print compiled shared size next to requested size; do not take the last JIT-cache entry as the
  kernel you just ran.

## Extended planning (attention / fused kernels)

- **AGPR ↔ VGPR:** placement and the default budget split are stated once, in
  `../tile-programming/llvm-fn-attrs.md ### AGPR budget: amdgpu-agpr-alloc=N (intermediate values)`. In short
  (gfx942/gfx950, Triton 3.8.0): the MFMA accumulator destination was `v[..]` in the kernels checked; at
  1 wave/SIMD (4-warp CTA) MFMA destinations moved to AGPRs once demand passed 256 arch VGPRs (compile-verified:
  64x64 / 128x128 fp32 acc at 4 warps -> `v[..]`, `.agpr_count 0`; 256x256 -> 250 AGPRs); at 2+ waves the
  default split varies by kernel and spill copies can use AGPRs well below 256. VALU access to an AGPR costs
  `v_accvgpr_read/write`; read the `v_mfma` destination and `.agpr_count`. RDNA WMMA has **no** AGPR file.
- **MFMA write -> read latency** (completion hazard; required wait states depend on the pass count — 5 / 8 / 12 / 20
  for 2- / 4- / 8- / 16-pass shapes, `amd_isa.py --arch gfx950 hazard`, `bound-class-signals.md`;
  `hw_constants.json mfma_write_read_latency_cyc` is one summary value; `s_nop` in the hot loop): unroll /
  occupancy add independent work; a reorder helps only if independent work already exists. Issue interval and co-execution window are separate
  (`isa-mechanisms.md ## MFMA throughput model`, `## Matrix/VALU co-execution`).
- **A deliberate `s_nop` at a stage head can be a phase lever**, not an exposed hazard: it shifts one
  wave's LDS burst off the other wave's 3-operand VALU inside an MFMA shadow (the contention is ATT-measured on
  gfx950; the `s_nop` remedy was observed on one kernel, not swept); sweep it, it is non-monotone
  (`isa-mechanisms.md ## Instruction-rate facts that flip a lever's sign`).
- **Power / DIDT:** a denser MFMA stream can lower the sustained clock (GPU-measured on gfx950: dense
  MFMA and, further, dense VALU FMA ran below the idle-loop clock); judge a schedule change by
  cycles and the kernel by both cycles and wall time (`../method/benchmark-hygiene.md`).
