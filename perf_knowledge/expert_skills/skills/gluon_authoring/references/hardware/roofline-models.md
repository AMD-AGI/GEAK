# Closed-Form Performance Models (gfx950 planning)

Use these to compute the **ideal budget before editing** and to interpret the
**as-built budget** read from plain Triton (or a Gluon candidate). All constants
come from `planning-constants.md` and are calibratable; see `## Calibration`.
Every example is written for **gfx950 (MI355X / MI350X) first**; gfx942 (MI300X / MI325X)
follows as the downgrade row. Paths that start with `perf_knowledge/`, `kernel_workflow/` or
`e2e_workflow/` are GEAK repo-root paths.

## Every "% of roofline" carries two bases (the reporting contract)

A roofline fraction is a numerator over a denominator, and each of the two can come from a model
or from a measurement. Name both, every time — in a round record, a brief, a closure claim:

| field | values | means |
| --- | --- | --- |
| `numerator_basis` | `model` — analytic bytes / FLOPs (`hw_budget.py --workload` archetype or `--tensors` manifest) · `counters` — measured (PMC / rocprof-compute bytes, ATT cycles) | where the *achieved* quantity came from |
| `denominator_basis` | `datasheet` — the `sku.json` peak · `empirical@<tool>-<version>` — a tool-measured peak (e.g. `rocprof-compute --roof-only` empirical roof), with its version · `in-shape probe` — `mem_bw_probe.py` / a dense MFMA loop at **this** kernel's mix, stride and program count | what ceiling it is a fraction *of* |

How each combination may be used:

- a **`datasheet`** denominator (or a **`model`** numerator) **ranks hypotheses and sets priors
  only**. It never gates a round and never closes a direction;
- only a **`counters`** numerator over an **`empirical@…`** or **`in-shape probe`** denominator may
  gate or close — and a close prefers the in-shape probe (`## Calibration`, `### The denominator
  decides the verdict`);
- a missing leg **degrades the label, not the verdict machinery**: report the lower-basis number
  with its basis and keep going (GEAK's no-hard-verdict ladder), never silently promote it.

`hw_budget.py` prints `[datasheet]` vs `[calibrated]` on its output for exactly this reason; carry
that tag into the record as `denominator_basis`. The method that consumes it is
`../method/budget.md`.

### Where the numbers come from (one source each)

| quantity | single source | reader |
| --- | --- | --- |
| SKU peaks (matrix TF per dtype, HBM TB/s, CUs, L2/MALL, `basis`) | `perf_knowledge/hardware/data/sku.json` | `kernel_workflow/scripts/kernel_tools/_hwdata.py` (`load_json("sku.json")`); `amd-*-skus.md` is its human view |
| per-arch facts (LDS, banks, VGPR file, latencies) | `perf_knowledge/hardware/data/hw_constants.json` | same loader; `planning-constants.md` is the prose |
| discriminator cutoffs (saturation %, SoL lead margin, latency, occupancy bands) | `perf_knowledge/hardware/data/thresholds.json` (each entry `basis`-tagged) | cite the key, not the number — values are recalibrated there |
| workload intensity models | `perf_knowledge/hardware/data/workload_models.json` | `hw_budget.py --workload` |

The tools — `hw_budget.py`, `calc_perf.py`, `mem_bw_probe.py`, `extract_sku.py`, `parse_pmc.py` —
live in `kernel_workflow/scripts/kernel_tools/`; the same file names under this pack's `scripts/`
are shims to them. In the commands below `KT=kernel_workflow/scripts/kernel_tools`. A dtype absent
from a SKU row is **refused**, never filled from fp16 (`### A missing ceiling is not a zero
ceiling`). GEAK's e2e roofline skill (`e2e_workflow/knowledge/analysis_skills/roofline/`) reads the
same `sku.json`.

**rocprof-compute caveats that move a denominator or a numerator:**

- `rocprof-compute profile --roof-only` (the empirical roof → `empirical@rocprof-compute-<ver>`)
  needs **rocprof-compute ≥ 3.6.0 on gfx95x**; older builds produce no usable gfx950 roof. Record
  the version in the basis tag; on gfx942 see `cdna3-gfx942.md ## Known tooling issues (gfx942)`.
- On **gfx950 the `FETCH_SIZE` / `TCC_BUBBLE`-derived read bytes under-count**. A `counters`
  numerator for a gfx950 memory-bound kernel must be cross-checked against the `TCC_EA0_RDREQ*` route
  and the model bytes (`### When the counter routes disagree`) before it gates anything.
- GEAK's own peak-validation checks (an empirical BF16 peak reading ~2× low; any efficiency > 100%
  means a mis-calibrated peak) are in `kernel_workflow/knowledge/profiling_guide.md` ("Cheap checks to
  run from the raw counters BEFORE trusting a label"); the tool walk-through is
  `perf_knowledge/profiling/rocprof_compute_workflow.md` and `perf_knowledge/profiling/roofline_on_mi.md`.

## Roofline / GEMM

```text
TFLOPS        = 2 * M * N * K / (time_us * 1e6)
intensity     = FLOPs / minimum_bytes
                FLOPs = 2*M*N*K
                minimum_bytes ~= M*K*b(A) + K*N*b(B) + M*N*b(C)   # b() = dtype bytes
ridge_point   = peak_compute[dtype] / peak_hbm_bw   # MI355X FP16 ~= 312, MI350X ~= 287 ops/byte
```

**SKU ridge (FP16 matrix peak / peak HBM BW)** — derived from `sku.json` per call
(`sku.json` deliberately stores no ridge: it is a different number for every dtype). Every row
below is `denominator_basis = datasheet`, i.e. a **ranking** input. Full tables in `amd-*-skus.md`;
use `python3 $KT/calc_perf.py roofline --sku <key>`:

| SKU key | arch | FP16 ridge (ops/byte) |
| --- | --- | --- |
| MI355X | gfx950 | ~312 |
| MI350X | gfx950 | ~287 (lower clock basis than MI355X: 2300 vs 2500 TF) |
| MI300X | gfx942 (downgrade) | ~247 |
| MI325X | gfx942 (downgrade) | ~218 (same compute as MI300X, HBM 6.0 TB/s) |
| MI308X | gfx942 (downgrade) | ~65 |
| R9700 | gfx1201 | ~298 |
| AI_MAX_395 | gfx1151 | ~280 (bandwidth basis **measured** 0.212 TB/s, not the 0.256 datasheet) |
| AI_MAX_495 | gfx1151 | ~225 |
| RX9070XT | gfx1200 — not supported by GEAK, reference only | ~305 |
| RX7900XTX | gfx1100 — not supported by GEAK, reference only | ~128 |

Full RDNA/W-series keys: `amd-rdna3-skus.md`, `amd-rdna35-skus.md`, `amd-rdna4-skus.md` + `calc_perf.py --sku`.
The same ridge arithmetic, with the HBM/MALL ladder behind it, is in GEAK's
`perf_knowledge/hardware/shared/hbm_infinity_fabric.md` ("Roofline ridge (why bytes win)") and the
per-generation `perf_knowledge/hardware/{cdna4_mi350,cdna3_mi300}/peak_tables.md`; those cards
quote datasheet peaks, so the same ranking-only rule applies to them.

If `intensity << ridge` -> memory path first. Else -> MFMA continuity / register
/ LDS / epilogue first.

**Attention is not GEMM here:** the `minimum_bytes` above assumes each element is
read once, but attention re-reads K/V **once per Q-tile**, so this GEMM model
**overestimates attention intensity** (and can mis-call the bound class). For
attention use `$KT/calc_perf.py attn-fwd` (real Q+O-once / K-V-reread bytes model,
prints both `%compute-peak` and `%HBM-ceiling`), or the `## Feeds-and-speeds` per-resource
model below for the fused softmax-between-matmul loop.

**Caveat: the roofline class is not the binding resource.** Aggregate intensity
assumes the compute is MFMA; for fused / VALU-heavy kernels
(softmax-between-matmul, dequant) the binding resource can be VALU or
dependency-stall even when intensity says "compute-bound." Confirm the binding
sub-resource from the profiler (`../method/profile.md`) before choosing a layer.

## Feeds-and-speeds (per-resource cycle roofline)

The single-resource roofline (above) only weighs compute vs HBM. For a fused hot
loop (attention / softmax-between-matmul) compute the **per-resource cycle cost**
of one loop iteration from the tile shape `(M, N, d)` and the per-unit
throughputs, then take the **max** — that resource is the bound:

```text
T_mfma = n_mfma_instr * cyc_per_mfma             # cyc_per_mfma: 16 (FP16/BF16), 32 (FP8)  [gfx950]
T_lds  = lds_bytes_read / lds_bytes_per_clk      # lds_bytes_per_clk: 256 (gfx950), 128 (gfx942)
T_exp  = n_transcendental_ops / exp_ops_per_clk  # v_exp/v_rcp/... transcendental issue rate (calibrate)
T_iter = max(T_mfma, T_lds, T_exp, ...)          # the binding resource of the iteration
```

**LDS-operand-reread amplification (the key term).** An `M x N` output is produced
by `ceil(M/mt) * ceil(N/nt)` MFMA instructions (`mt x nt` = the MFMA tile, e.g.
16x16), and **each MFMA re-reads its operand slices from LDS**. So LDS read traffic
scales with the **number of MFMA instructions**, not just the tile's byte size:

```text
lds_bytes_read ~= n_mfma_instr * operand_bytes_per_mfma   # operands re-read per instr
```

A larger tile lowers `T_mfma`'s prologue/epilogue overhead per useful FLOP but
raises `T_lds` with the instruction count — the **quantitative root of the
overlap/occupancy/ILP tri-lemma** (`../tile-programming/slicing.md ## Occupancy
budget (P8)`). It is why a chained-matmul kernel that is MFMA-light can still be
**LDS-traffic-bound**, and why the bound migrates toward LDS as the tile grows.

Use this to **predict the bound before editing**: if `T_lds` or `T_exp` exceeds
`T_mfma`, the lever is LDS-traffic reduction (dedup / transpose-on-read view /
fewer rereads, `../tile-programming/memory-path.md`, `slicing.md`) or
exp-throughput (`../method/profile.md ## Reducing compute-class VALU`), **not** MFMA
continuity. Then confirm against the profiler busy counters
(`../method/profile.md ## Rule: read the busy/throughput counter`).

## Asymmetric scaling (predict the migrating bottleneck)

A cross-generational hardware law: **the matrix engine scales faster than the
other functional units** (LDS bandwidth, the transcendental/exp unit,
per-instruction latencies). So across generations the binding resource of a fused
kernel **migrates** away from MMA — toward LDS traffic, then toward the exp unit
and dependency-chain latency. Design newer-arch kernels pre-emptively for
LDS-traffic + exp/critical-path reduction, not MMA continuity.

| Unit (relative) | gfx942 (CDNA3) | gfx950 (CDNA4) | gfx1250 (CDNA5) |
| --- | --- | --- | --- |
| MFMA throughput | baseline | ~2x (matrix throughput doubled) | higher again + fully-async MMA |
| LDS read BW | 128 B/clk | 256 B/clk | TBD-probe |
| Transcendental (exp) | quarter-rate VALU | quarter-rate VALU (unchanged) | TBD-probe |
| Async / warp-spec | no (occupancy-first) | no (occupancy-first) | named-barrier warpgroups (asynchrony-first) |

**gfx1250 (CDNA5 / MI450 data-center) is a separate ISA fork from RDNA4 — not RDNA,
and not the same target.** RDNA4 client (R9700 / RX9070 XT, gfx1201) is a separate,
occupancy-first WMMA fork with **no** TDM and **no** named-barrier warpgroups; do not
infer gfx1250 behavior from RDNA4 or vice versa. gfx1250 microarch constants are **not
tabulated in this skill** (the `TBD-probe` cells above) — probe on-target before
relying on any gfx1250 number. RDNA5/UDNA (gfx13) is the future client target and may
inherit gfx1250 asynchrony primitives, but that is unconfirmed.

Reading it: MMA grew fastest; LDS BW grew too (so LDS *capacity* + the operand
**reread** term, not raw BW, is the near-term squeeze); the exp unit did **not**
scale, so the MMA:exp ratio widened and softmax/exp + dependency latency become the
bound. This is exactly the gfx950 latency/LDS-bound result, and it sharpens on
gfx1250 (even faster MMA) — where the asynchrony-first levers begin to pay
(warp-specialized correction warpgroup, partial FMA exp-emulation:
`../workloads/attention.md`, `../method/profile.md ## Reducing compute-class VALU`).
Constants + the gfx942 downgrade: `planning-constants.md`; the occupancy-first vs
asynchrony-first model split: `../tile-programming/mental-model.md ## Porting a
technique across architectures`.

## MFMA efficiency (primary compute-bound metric)

```text
MFMA_efficiency = total_mfma_cycles_in_loop / average_iteration_duration   # per SIMD
per_SIMD = per_wave x waves_per_SIMD          # the conversion, and it is easy to skip
```

**Read the second line before comparing anything to the target below.** An instruction trace
reports this **per wave**; the target is **per SIMD**. With one resident wave per SIMD the two
coincide, which is why the distinction stays invisible on intra-wave GEMM — and then a two-wave
kernel (any `warp_pipeline_stage` ping-pong, any four-stage attention) reads **half** its real
figure. A kernel actually near 90% reports ~45%, which looks like a broken schedule and invites a
round of work with nothing to fix; the same factor in the other direction closes a kernel that
still had headroom. Establish the achieved waves/SIMD first, then decide whether the number in
front of you needs the factor applied.

Target: `thresholds.json confirm.mfma_efficiency_target_pct` (calibrated; ~98% at the time of
writing) for a compute-bound hot loop. Cycle-based (less clock/temperature
sensitive than TFLOPS), so it is also the right clock-insensitive A/B discriminator
on a drifting box. Scheduler interleave target: ~4x 16-cyc MFMA per `buffer_load`
(FP16); ~2x 32-cyc MFMA per mem op (BF8).

MFMA efficiency is the right metric **only when MFMA-issue is the binding
sub-resource**. For fused / VALU-between-matmul kernels (softmax, dequant, gating,
...) the profiler may show VALU- or dependency-stall-bound even at modest MFMA
efficiency; optimize that sub-resource instead
(`../method/profile.md ## Bound classification`).

## Low-precision / dequant-VALU (check op-mix before the FLOP roofline)

The arithmetic-intensity roofline (FLOP/byte vs the SKU ridge) models **only** the
matmul FLOPs and HBM bytes. For a **low-precision weight GEMM** (W4A16 / W4A8, or any
dequant-in-loop) the binding cost is frequently the **software dequant VALU** — the
int4/int8 → bf16 unpack that runs *before* each MFMA. This cost has **no axis** on the
FLOP roofline, so a kernel can read "N% of the MFMA roofline" while actually being
**VALU/dequant-bound**, and an MFMA-continuity attack then does nothing.

Rule: for any low-precision/mixed-precision matmul, **run `scripts/asm_loop_audit.py`
and compare VALU% vs MFMA% before trusting FLOP/peak**. If unpack/convert VALU
dominates, treat it as VALU-bound (Layer 7 — cheapen the dequant: a native converter
where the arch has one, else fold the unpack off the dependency path); only apply the
MFMA-continuity roofline once the op-mix shows MFMA dominates. Native int4→bf16
converters are arch-gated (e.g. CDNA4-only on the CDNA line — see
`../tile-programming/low-precision.md`).

## LDS bytes / capacity

```text
A_tile_bytes  = BLOCK_M * BLOCK_K * b(dtype)        # idem B_tile
LDS_bytes     = stages * (A_tile + B_tile + padding/swizzle) + side_path_LDS
constraint    : LDS_bytes <= LDS_per_CU (160 KiB gfx950 / 64 KiB gfx942)
```

Example FP16 256x256x64: A/B tile 32 KiB each -> 64 KiB per K-tile -> ~128 KiB
double-buffered (fits 160 KiB gfx950, not 64 KiB gfx942).

## Register budget

```text
R_acc_per_thread ~= BM * BN * acc_dwords * sharing / (num_warps * 64)
R_total          = R_acc + R_operand + R_prefetch + R_offsets + R_side
                   + R_epilogue + R_compiler_tmp
constraint       : R_total <= register_budget_for_target_occupancy (~512 dwords/thread)
```

Spills (`scratch_load` + `s_waitcnt`) destroy MFMA efficiency; slice M/N to fit.
Tutorial reference: FP16 full tile + prefetch ~512 (at limit); slice-N ~448;
slice-MN ~384.

## Occupancy

```text
wg_per_CU_by_LDS   = floor(LDS_per_CU / LDS_bytes_total)
wg_per_CU_by_waves = floor(max_waves_per_CU / waves_per_WG)
wg_per_CU_by_regs  = floor(reg_file_budget / (R_total * threads))   # reg_file_budget is
                   # the FILE PER SIMD, not the per-wave cap -- they coincide (512) on CDNA
                   # and do NOT on RDNA (1536 file vs 256/wave): `hardware/planning-constants.md`
resident_wg_per_CU = min(by_LDS, by_waves, by_regs, by_SGPR, hw_limit)
active_waves       = resident_wg_per_CU * num_warps
```

Maximize **effective in-flight bytes** and **MFMA duty cycle**, not occupancy
blindly.

**Analytic prediction vs measured attribution.** The `min(by_LDS, by_waves, by_regs)`
above only *predicts* which resource caps occupancy. When a rocprof-compute report is
available, read the **measured** SPI (Workgroup Manager) occupancy limiter as the
ground-truth cross-check: `6.2.7 Insufficient LDS`, `6.2.5 Insufficient VGPRs` (+ SGPR /
Barrier / Waves), alongside `2.1.15 Wavefront Occupancy`. If measured and analytic
disagree, trust the measured limiter and record why. When two limiter fields are nonzero
the larger limits more (HLRS; magnitude is not a linear occupancy delta). Block IDs:
`bound-class-signals.md ## rocprof-compute block-ID cheat-sheet` (probe per build).

### VGPR-bound waves: read the count the hardware reads

The occupancy-binding VGPR count is the kernel descriptor's
`.amdhsa_next_free_vgpr` (= ArchVGPR peak **plus** AGPR, merged via `accum_offset`),
**not** the profiler meta `VGPR` (which often reports ArchVGPR only); the two can
differ substantially. On CDNA, ArchVGPR and AGPR share one physical file, so occupancy
counts them jointly (RDNA has no AGPR file — there is nothing to merge, but the file/cap
split below still bites):

```text
waves_per_SIMD_by_VGPR = min(cap, floor(vgpr_file_per_simd / round_up(next_free_vgpr, granule)))
  CDNA  (gfx942/950, wave64): file  512/SIMD, granule  8, cap  8   # ArchVGPR+AGPR combined
  RDNA  (gfx11xx/120x, wave32): file 1536/SIMD, granule 24, cap 16 # no AGPR file
```

Read `next_free_vgpr` from the `.amdgcn` KD — **not** from a `VGPR` counter and **not** from a
front-end `n_regs` / `.vgpr_count` print. Those report the highest register *number used*; one
measured sweep held `.vgpr_count` at 80/80/80/80/64 while the descriptor moved
257/169/129/97/64, because the compiler was paying in AGPRs. And `next_free_vgpr` is still not
the allocation: the hardware reads `COMPUTE_PGM_RSRC1[5:0]`, i.e.
`ceil8(ceil4(next_free_vgpr))`, measured 0 to +7 above the symbolic count — enough to move a
wave (printed 97 → 5 waves/SIMD; allocated 104 → 4). Round **before** comparing two
configurations (`planning-constants.md ### Read the field the hardware reads, and round it
before you compare`).

When the compiler emitted the `.s` it already wrote `; Occupancy: N` per kernel, arch-correct by
construction, and `$KT/amd_occupancy.py --asm kernel.s` reads it, falling back to the table
above only when it is absent (`$KT/asm_loop_audit.py` prints it in the KD block). **Use it
for the register term; it is not the occupancy.** It carries no LDS term: across 6 dumps at 3
distinct VGPR counts every emitted value equalled `floor(512 / VGPR)` exactly, and on a kernel
whose LDS is allocated **dynamically at launch** — what the tile DSLs emit — it overstated the
real limiter by up to **3x** while printing its own disqualifier (`LDSByteSize: 0
bytes/workgroup (compile time only)`) four lines above the value. Where the emitted number
disagrees with a hand-derived `min(by_LDS, by_regs)` on such a kernel, the hand derivation is
the more correct one; whether a **statically** allocated-LDS kernel makes the emitter include
the term is unverified. Both, with the LDS-per-workgroup field to use instead:
`planning-constants.md ## The emitted ; Occupancy: N is a register-term answer`.

### Occupancy is a lever only for latency/memory-bound work

Raising waves/CU helps **latency/memory-bound** kernels (more in-flight work hides
the wait). It is **inert for throughput-bound** kernels: if a compute unit (VALU or
MFMA) is already the saturated bottleneck, extra resident waves only share that same
saturated unit and add no throughput (a finer-grained tile can even regress).
Confirm the bound is latency/memory — not a saturated compute unit
(`../method/profile.md ## Derived metrics`) — before spending registers/LDS to
raise occupancy.

**And even there the criterion is one-sided: it vetoes, it does not motivate.** Below some point
too few waves hurts; above it, more waves does not buy time. Spend an occupancy *loss* as a reason
to reject a change that buys MLP with LDS or VGPR; do not spend a projected *gain* as a reason to
fund one, and do not pre-register a magnitude derived from one. The measured CDNA4 points behind
this rule are stated once, in `planning-constants.md ## Occupancy is a one-sided criterion`.

GEAK's occupancy cards (`perf_knowledge/hardware/shared/wavefront_simd_vgpr_agpr.md`,
`perf_knowledge/hardware/cdna3_mi300/occupancy.md`, `perf_knowledge/optimization/occupancy_and_registers.md`)
carry the same `min(by_LDS, by_waves, by_regs)` model and the tuning checklist. **Where one of them
quotes a 16-VGPR allocation granule it disagrees with this page:** the arbiter is
`kernel_workflow/scripts/kernel_tools/amd_occupancy.py` (granule 8, cap 8, input
`ceil8(ceil4(next_free_vgpr))`), measured in `planning-constants.md ### Read the field the hardware
reads, and round it before you compare`.

### Reclaimable-VGPR diagnostic (before chasing occupancy by register remap)

Before arguing "lower VGPR -> more waves," check whether `next_free_vgpr` is a
genuine peak-live count or merely over-declared (dead-window) registers. A liveness
pass (e.g. `vgpr_liveness`) reports reclaimable VGPRs: if it is a true peak-live
count, register renaming/remap **cannot** raise occupancy — reaching it needs fewer
simultaneously-live values, which is a source/tiling change (slicing,
`../tile-programming/slicing.md`), not an asm transform.

## Saturation / wave quantization

`saturation = grid_tiles / CUs` says whether the grid fills the device; the **quantization
tail** says how much of the *last* wave is wasted when `grid_tiles` is not a multiple of the
CU count:

```text
waves           = ceil(grid_tiles / CUs)                 # CUs: sku.json row (256 MI355X/MI350X; gfx942 downgrade: 304 MI300X/MI325X, 80 MI308X)
tail_efficiency = grid_tiles / (waves * CUs)             # <<1 on a small grid => quantization tail
```

GEAK's grid-sizing card (`perf_knowledge/optimization/wave_and_grid_sizing.md`) and XCD-locality
cards (`perf_knowledge/hardware/shared/l2_xcd_swizzle.md`, `perf_knowledge/hardware/cdna3_mi300/xcd_chiplet.md`)
cover the grid-to-XCD mapping this formula ignores; tail efficiency is one-sided
(`planning-constants.md ## Tail efficiency is one-sided too`).

When `grid_tiles` sits just over a CU multiple, the last wave is nearly empty
(`tail_efficiency` ≪ 1). Lever: adjust tile size / split-K / persistent kernel so
`grid_tiles` lands near a CU multiple; on a small grid a *smaller* tile that raises
`grid_tiles` past the next CU multiple can win.

**Distinct from the load-imbalance tail.** This is "block count not aligned to CU count"
(all blocks equal work). The load-imbalance tail (LPT remap, ragged split-K,
`../workloads/attention.md`) is "blocks do *unequal* work"; they are separate diagnostics
and separate levers — do not conflate.

## HBM / TCP in-flight model

```text
effective_compute_latency = waves_per_simd * compute_latency
num_req_per_wave          = min(hbm_latency / effective_compute_latency, prefetch_depth)
data_per_request_per_wave = block_size / num_waves_per_workgroup
inflight_bytes_per_CU     = num_req_per_wave * data_per_request_per_wave * active_waves
effective_inflight        = min(inflight_bytes_per_CU, 32 KiB)        # TCP cap (thresholds.json confirm.tcp_inflight_cap_kib)
BW_per_CU                 = effective_inflight / hbm_latency
BW_total                  = BW_per_CU * min(num_workgroups, 256)      # 256 CUs gfx950 (gfx942 downgrade: 304)
effective_pipeline_depth  = min(prefetch_depth,
                                floor(32KiB / (active_waves * data_per_request_per_wave)))
```

`prefetch_depth` is the number of loads in flight ahead of their consumer. On plain Triton it is
`num_stages - 1`; on the Gluon path `num_stages` is consumed by no pass in 3.8.0, so it is the depth
of the ring you **authored** (`../tile-programming/pipeline.md`) — read it from the source / TTGIR,
not from a launch option. The same in-flight argument from the HBM side is GEAK's
`perf_knowledge/hardware/shared/hbm_infinity_fabric.md`.

## Pipeline coverage

```text
stall_hbm = max(0, latency_hbm_to_LDS - work_after_AC)
stall_lds = max(0, latency_LDS_to_reg - work_after_LR)
```

Deepen the pipeline only if `stall_reduction > extra_cost` (extra LDS stages,
prefetch regs, prologue/epilogue, wait complexity).

## LDS throughput (conflict diagnostic)

| Instruction | Conflict-free steady interval | 2-way | 4-way |
| --- | --- | --- | --- |
| `ds_read_b128` (full CU) | 16 cyc | 32 cyc | 64 cyc |
| `ds_read_b64` | 8 cyc | — | — |

Observed `ds_read_b128` interval > 16 cyc (`thresholds.json confirm.ds_read_b128_interval_cyc`)
=> bank conflicts; fix with padding or swizzle (`../tile-programming/layout-recipes.md`). The
per-arch bank geometry (gfx950: 64 banks, 256 B full-conflict stride; gfx942 downgrade: 32 banks,
128 B) is in `planning-constants.md ## LDS throughput / conflict (planning)` and GEAK's
`perf_knowledge/hardware/shared/memory_model_lds_bank.md`.

## Kernel time decomposition

```text
T_CTA = T_prologue
      + num_k_tiles * max(T_mfma, T_lds_residual, T_exp, T_hbm_residual,
                          T_layout_convert, T_spill, T_side)
      + T_epilogue
```

The dominant term inside `max(...)` is the current bound class; the layer loop
attacks it, then reprofiles. `T_mfma` / `T_lds` / `T_exp` come from the
**feeds-and-speeds** model above (with the LDS-operand-reread amplification);
compute all three from the tile shape before editing to predict which resource
binds.

`T_CTA` above is **device-side, kernel-only**. The wall time also carries a host
launch term:

```text
T_wall(eager)      = T_launch_host + T_CTA          # per call, launch exposed
T_wall(CUDA-graph) = T_CTA  (+ amortized capture)   # launch replayed, ~free
```

**Launch-fusion law (mirror of the occupancy law).** (Indexed in the gating-law SoT,
`bound-class-signals.md ## Lever gating laws`.) A change that reduces
`T_launch_host` — fewer kernel launches, fused host dispatch, folding a per-output
recompute into one launch — pays **only while the launch is EXPOSED** (eager,
per-call host dispatch). Under a **CUDA graph** the launch is replayed and already
amortized, so `T_launch_host -> ~0` and the same change is **neutral-to-negative**
on `T_CTA` (a fused per-tile recompute can even *raise* `T_CTA` by repeating work).
So the right tier/structure is **boundary-dependent**: validate launch-reducing
levers at the **production boundary** (`../method/benchmark-hygiene.md`, measurement
basis), and do not carry an eager launch win into a graph-served deployment. This
is the launch-overhead analogue of "occupancy gates latency-removing levers"
(`../method/climb.md`, "2. Reversed-intuition traps — read this once", trap 6).

## Calibration

**Treat every number below as a ranking input, not a close gate.** A roofline ranks hypotheses only
after its boundary, clock, device fill, and workload model have been calibrated for the current run.
In the reporting contract's terms (`## Every "% of roofline" carries two bases (the reporting
contract)`): calibration is what moves a reading from `model`/`datasheet` to `counters` over
`empirical@…`/`in-shape probe`, and only that pair may gate or close.

Use these checks before relying on a gap:

1. **Measure the clock and sustainable rate on this box and at this workload.** Do not divide
   instruction counts by a planning clock or interpret a higher clock as higher useful work.
2. **Probe the ceiling at the current program count and access shape.** A ceiling from a different
   fill, stride, mix, or cache state is not transferable.
3. **Price coupled terms as coupled.** A one-axis ablation is a lower bound until a follow-up
   measurement establishes that the rest of the configuration remains comparable.

Replace planning peaks with measured values from `../method/profile.md` evidence (profiler entry:
`kernel_workflow/scripts/profile_kernel.sh`, run under `kernel_workflow/scripts/gpu_lock.sh`):

- `peak_mfma -> effective_mfma` (from ATT loop cycles, or a dense in-shape MFMA loop);
- `peak_hbm -> effective_hbm` (**both terms**: measured DRAM bytes over a probed
  in-shape ceiling — the four layers below);
- `lds_model -> observed ds_read interval`;
- `register_budget -> observed VGPR/AGPR/spill` (from `.amdgcn` / compile stats);
- `pipeline depth -> observed from .ttgir` (on the Gluon path the authored ring depth;
  `num_stages` is a budget parameter / champion record only, no 3.8.0 Gluon pass reads it).

The calibrated budget (not the raw planning numbers) is the reference the layer
loop optimizes against. Calibrate from **clock-stable** measurements — GEAK's timing
harness (`e2e_workflow/scripts/harness_lib.py`: CUDA events, per-sample sync, read-evict flush,
median) and `../method/benchmark-hygiene.md` for shared / DVFS boxes: values taken under DVFS
drift mis-calibrate both the budget and the bound class. Clock behaviour per part:
GEAK's `perf_knowledge/hardware/cdna4_mi350/clocks_power.md` (gfx942 downgrade:
`perf_knowledge/hardware/cdna3_mi300/clocks_power.md`).

### Zeroth question: does a roofline apply to this kernel at all?

Both terms of the model assume the whole machine is streaming against DRAM. When that is
false the model is not imprecise, it is **inapplicable** — and its gap is then a number
whose size says nothing about how much time is recoverable. Check three things before
computing anything, because each one has a distinct fix and a distinct tell:

| precondition | tell | what it voids |
| --- | --- | --- |
| **the grid fills the machine** (`--grid`) | measured time barely moves while the byte count changes by an order of magnitude | *both* floors — an idle CU is neither computing nor streaming. The fix is parallelism (split-K, finer tiles, a persistent grid), not a better access pattern |
| **the working set clears the memory-side LLC** (`--footprint-mb`) | measured time is *below* the memory floor | the **memory** floor only. The compute floor survives and is then the binding one — do not discard it |
| **the timed region is one kernel** (`--dispatches`) | the profiler shows several dispatches per timed iteration | comparing a per-kernel floor against the region total. Budget each dispatch against its own duration |

Two refinements that catch the same failures a step earlier:

- **Filling every CU once is not saturating memory.** Bandwidth is
  `requests-in-flight x bytes-per-request / latency`, so a CU holding a single workgroup
  caps the first factor no matter how clean its access pattern is. This is measurable, not
  arguable: the in-shape ceiling **rises with program count** over a fixed footprint
  (`mem_bw_probe.py --nprog-sweep`). Quote the ceiling probed at *this* kernel's program
  count, or the gap you compute is one that more parallelism closes, not better locality.
- **Allocation is not traffic.** A workspace or KV cache sized for the general case and then
  used for one sequence can allocate orders of magnitude more than it moves. Count *touched*
  elements; a byte model built from allocation sizes is not conservative, it is wrong.

### The numerator is a manifest, not a formula

An archetype formula (`--workload gemm|attention|moe ...`) describes a kernel that matches
the archetype in **structure**, not just in name. The archetype is usually the assumption
most likely to be wrong, and it fails in ways no amount of shape-variable fitting repairs:

| what the formula cannot say | consequence |
| --- | --- |
| per-tensor dtypes | a low-precision-operand GEMM with a wide output has most of its bytes **in the output**; one `b` is wrong by that ratio |
| this pointer is never dereferenced on the taken path | summing the argument list charges tensors the kernel never reads |
| this operand is re-read once per tile of the other | issued traffic is a multiple of the footprint set by the **tile**, not by the shape |
| this grid dimension is dead | every program is duplicated along it: issued traffic scales, the footprint does not |
| rows are padded to a block boundary | executed work exceeds useful work, and the padded rows still stream operand tiles |
| `q_len == 1` | a decode shape is a gather and a reduction; the quadratic model its name implies is off by roughly the sequence length |
| only the routed experts are touched | charging all of them over-counts by the ratio — and at prefill sizes *every* expert is routed, so the correction is a no-op exactly where it is easiest to test |

State the bytes as a manifest of what the kernel **moves** instead
(`hw_budget.py --tensors "name:dir:dtype:dims[:xN]"`). It returns a **bracket** — the unique
footprint at the low end (perfect reuse), the declared traversals at the high end (no reuse)
— rather than a false point. The truth is wherever the cache hierarchy puts it, which no
model knows; the bracket's job is to make that honest instead of hidden.

### These counters measure fabric traffic, not DRAM traffic

The L2 is the gateway from a compute die onto the fabric, and a **memory-side** last-level
cache sits behind that fabric, in front of DRAM. Every counter in L1 above — `TCC_MISS`,
`FETCH_SIZE`, `WRITE_SIZE`, `TCC_EA0_*` — is sampled at the L2/fabric boundary, so whether a
byte was served by that cache or by HBM is invisible to **all** of them. Therefore:

- a rate derived from them is an *exact* fabric rate and an *upper bound* on the DRAM rate;
  the two coincide only once the footprint clears the memory-side LLC;
- a high %-of-HBM-ceiling can be reached with nothing going to HBM at all, which is why the
  residency precondition above is not optional;
- no counter route can settle the residency question for you — only the footprint can;
- **on gfx950 the `FETCH_SIZE` / `TCC_BUBBLE` read-byte routes additionally under-count**, so
  there the fabric-rate "exact" claim holds only for the `TCC_EA0_*` / `TCC_MISS` routes — never let
  a `FETCH_SIZE`-only number be the `counters` numerator on gfx950.

### When the counter routes disagree

On a pure stream the routes agree to a fraction of a percent, which is what makes them a
usable unit check. On a real kernel they can diverge several-fold, and the divergence has
**two causes that need opposite responses**:

- a **unit or line-size slip** — reproduces on a known-bytes probe, so verify the units on
  this box before anything else;
- a **real access-shape effect** — appears only on the kernel while the same probe stays
  inside tolerance. Then no route is wrong: they measure different things. Whole-line miss
  counting is blind to partial-line stores, write-allocate refills and atomic round-trips,
  so it reads **low** — the direction that fakes *memory does not bind* and sends the next
  round at the wrong axis.

Which route holds is a property of **this kernel's access shape**, not a global constant, so
there is no canonical route to standardise on. Pass them all
(`hw_budget.py --measured-dram-mb a,b,c`) and read the width: a several-fold spread in bytes
is a several-fold spread in the size of the prize, and that width is the finding. If the
resulting byte bracket straddles the ridge, the *regime* is undetermined too — the same
kernel is compute-bound on one route and memory-bound on another, and picking one silently
is how a round opens on the wrong resource.

### Both terms of a memory roofline, and both default the wrong way

`hw_budget.py` with no calibration flags divides an **analytic byte model** by a
**datasheet peak**. Both defaults are optimistic, and they compound: the byte model
assumes bytes nobody measured, the peak assumes a rate no access shape reaches. The
gap they produce is part measurement and part fiction, and the fiction always points
the same way — *there is more headroom than there is*.

Four layers, each measurable, each cross-checkable against the next:

| | what | how |
| --- | --- | --- |
| L1 | **achieved** fabric bandwidth + L2 hit — `numerator_basis = counters` | `--pmc FETCH_SIZE WRITE_SIZE TCC_HIT_sum TCC_MISS_sum` -> `parse_pmc.py` memory block: `TCC_MISS*128 B / dispatch duration`, plus the independent `FETCH_SIZE+WRITE_SIZE` and `TCC_EA0_*` routes. They agree to a fraction of a percent on a pure stream and can diverge **several-fold** on a real kernel — see *When the counter routes disagree* below. Pass every route you measured; the tool carries the width instead of picking a favourite. **gfx950:** the `FETCH_SIZE` route under-counts reads — keep it in the bracket but not alone |
| L2 | **in-shape ceiling**, on whichever axis binds — `denominator_basis = in-shape probe` | *memory:* `mem_bw_probe.py` at YOUR run length / stride / read:write mix / program count — not the datasheet peak, and not the read-only peak either. *Compute:* a dense back-to-back MFMA loop at YOUR dtype -> `--measured-tflops lo,hi`. The datasheet MFMA rate assumes MFMA issued back-to-back with operands already in register, which a kernel that also loads, converts and addresses does not sustain |
| L3 | ideal / gap / %-of-ceiling | `ideal = bytes / ceiling`, `gap = measured - ideal`, `pct = achieved / ceiling`. The ceiling is a range, so all three are ranges, and each is reported with its `numerator_basis` / `denominator_basis` pair |
| L4 | is this gap worth a round? | weigh it by the **served** mix, not by microseconds on one shape (`../method/close.md ## Multi-shape no-regression + dispatch`; in GEAK, Director arbitrates on the served mix) |

```bash
KT=kernel_workflow/scripts/kernel_tools        # GEAK shared tools (pack scripts/ holds shims)
# L1: measured bytes + achieved rate (the numerator; profile under gpu_lock.sh)
python3 $KT/parse_pmc.py <prof_dir> <kernel_substr>     # -> dram_mb, achieved_dram_tb_s, l2_hit_pct
# L2: the ceiling at this access shape (the denominator), as a range
python3 $KT/mem_bw_probe.py --sku MI355X --runlen <run> --stride <stride> --rw-mix <r>:<w>
# L3: both, in the budget -- prints [calibrated] instead of [datasheet]
python3 $KT/hw_budget.py --sku MI355X --workload <w> --shapes ... \
    --measured-ms <ms> --measured-dram-mb <mb>[,<mb>...] --measured-hbm-tb-s <lo>,<hi>
# L3 on a COMPUTE-bound kernel: calibrate the ceiling that actually binds
python3 $KT/hw_budget.py --sku MI355X --workload <w> --shapes ... \
    --measured-ms <ms> --measured-tflops <lo>,<hi>
# L3, general form: when the kernel does not match an archetype, declare what it MOVES,
# and declare the preconditions so the tool can refuse instead of inventing a gap
python3 $KT/hw_budget.py --sku MI355X \
    --tensors "A:r:<dt>:<dims>, B:r:<dt>:<dims>:x<tiles>, C:w:<dt>:<dims>" --flops <n> \
    --grid <wgs> --footprint-mb <unique> --dispatches <n> --measured-ms <ms>
# gfx942 downgrade: --sku MI300X / MI325X (MI325X HBM 6.0 TB/s, same compute as MI300X)
```

**Calibrate the axis that binds, and scope the caveat to it.** Both ceilings are datasheet
numbers by default, so a compute-bound kernel is exposed exactly as much as a memory-bound
one — and it is easy to miss, because a loud refusal on the *memory* axis reads as though
the tool checked everything while the multiple over the *compute* floor sails through
uncaveated. That multiple is the number the round gets sized on. The error has a direction:
a peak that is too high puts the floor too low, so the reading can only ever overstate the
prize (a probed MFMA ceiling has moved a "2.4× on the table" to under 2×). The converse
matters just as much — telling a memory-bound kernel to go probe its MFMA ceiling is noise,
and off-axis warnings are how on-axis ones stop being read.

**A FLOP count is not a compute floor — the engine it issues on is what makes it one.**
`flops / MFMA_peak` is a floor only for FLOPs that go through MFMA. The `c*n` archetypes —
reduction, elementwise, norm, scan, gather — contain no matrix multiply at all, so charging
them at the matrix rate prices the compute floor one to two *orders* too low. It then never
binds, and "memory" falls out as a **default rather than a finding**. That default is the
dangerous part: for a VALU-heavy kernel with modest bytes the real compute floor sits
*above* the memory floor, so the tool reports a byte-removal prize that removing bytes
cannot collect. Declare the engine (`flops_engine` per archetype, or `--flops-engine` on the
manifest path) and price VALU work at a VALU rate. Two hardware facts make this its own
axis rather than a correction factor: MFMA and VALU **co-execute**, and transcendentals run
at a fraction of the VALU rate (`exp_rate_vs_valu`), so a softmax or a norm is priced on
neither the matrix peak nor the plain VALU peak.

**A missing ceiling is not a zero ceiling.** Three states, three different instructions:
zero FLOPs means the compute term *cannot* bind (a conclusion); a rate for this dtype gives
a real floor; **no** rate — the dtype is absent from the SKU table, or the FLOPs are on an
unpriced engine — means the floor is *unknown*, and a memory verdict resting on it is
provisional rather than measured. Never substitute another dtype's rate: that error runs the
dangerous way (too low a peak *lifts* the compute floor, which can flip the binding onto an
engine that was never the constraint), and it is silent. The intensity shortcut that
suppresses this — "AI far below the ridge, so the unpriced term could not bind anyway" —
is only valid on the **same engine as the ridge**: the ridge is `peak_MFMA / peak_HBM`, the
VALU peak is a fraction of the matrix peak, so the real VALU ridge is roughly an order lower
and a kernel comfortably below the MFMA ridge can sit on the compute side of the VALU one.

**The denominator decides the verdict.** Since `ideal = bytes / ceiling` and the measured
time does not move, the *gap* — the thing a round is opened to collect — grows as the
denominator grows. Read one kernel's measured bytes against the three defensible
denominators and they do not disagree by a few percent, they disagree about what to do:

| denominator | where it sits | what it does to the gap |
| --- | --- | --- |
| datasheet peak | highest; reachable by no real access shape | inflates the gap the most — "big prize, keep opening rounds" |
| probed pure-read (or pure-write) peak | below the datasheet peak | still inflates it: a mixed stream cannot reach a pure one |
| **probed in-shape ceiling** at your mix / stride / parallelism | the only one this kernel can reach | the gap you can actually collect — often "close it out" |

Measure the spread on the box you are on; do not carry a number over from another part.
The reachable ceiling has been found *well* below the datasheet peak — enough to multiply
an apparent gap several-fold — and even the read-vs-write asymmetry is per-part: on some
parts writes stream faster than reads, so which pure peak is higher cannot be assumed.

### A ceiling is a range, not a number

Repeat readings of ONE probe configuration drift by a few percent between sessions, and
the ceiling **rises with parallelism** over a fixed footprint. So:

- quote a ceiling as `lo–hi` **with its program count**, and carry the range through
  `ideal` / `gap` / `%-of-ceiling` (`hw_budget.py --measured-hbm-tb-s lo,hi` does);
- **a gap narrower than the range is not a reason to open a round** — it is inside the
  error bar of the number you are chasing.

### A probe that reads too high is worse than no probe

A too-high ceiling makes every gap computed from it look *smaller*, which can close an open
direction. The dangerous case is an over-read that is still *under*
the datasheet peak, so no peak check catches it: a plan that stacks translated copies on an
already-tiled base can claim several times its own footprint as "useful" traffic and report
a plausible-looking ceiling while actually measuring cache. `mem_bw_probe.py` therefore
gates every reading three ways: the **byte accounting** must be injective and fit its
allocation (this is the one that catches the case above), the footprint must be **several
times the memory-side LLC** — not the L2, which is an order of magnitude smaller and is
merely the gateway onto the fabric — and no **shaped** access may read above the linear
stream measured at the same mix and parallelism. A reading that fails any gate is refused,
not returned with a caveat.

Ways a ceiling has gone wrong in practice, all of which these gates catch: a cache-polluted
reading, a strided layout modelled as contiguous, and a pure-read peak quoted for a mixed
stream.

A ceiling read too **low** is the same failure mirrored — it also shrinks the gap. If a
probe reads far below the datasheet peak, decide which it is (a genuinely slow shape, or an
under-saturating probe) before citing it.

### Before you call a kernel memory-bound: floor-probe it

The roofline gap is a **rate** gap. It says nothing about whether any of it is *removable
bytes*, and a naive read ("gap is large, so there are bytes to delete") cannot tell an
**additive** cost from an **already-overlapped** one. Only the floor probe can
(`../method/profile.md ## Rule: floor probe`, `## Bound-the-win probe`):

```text
memory_ideal < non_stream_floor   ->  memory is NOT the binding constraint
                                      the roofline gap is NOT a prize
```

The failure this catches: a brief prices an overfetch at some number of microseconds, the
roofline gap is comfortably larger, and a round opens to delete those bytes. The floor probe
makes both streams cache-resident and the kernel is *still slower* than the memory ideal —
so memory was never the binding constraint, the byte-removal levers were worth only the
overfetch itself, and the real bound was elsewhere (instruction issue, in that case, visible
as a lopsided VALU:MFMA ratio). The mirror case is just as common: a kernel whose byte model
reconciles to its measured DRAM traffic with **zero** removable bytes, because the re-read
the model priced was fully absorbed by cache, leaving a gap that is pure *rate*.

For a **low-precision** kernel, check the op-mix first as well: the FLOP roofline has no
axis for dequant VALU (`## Low-precision / dequant-VALU`), so it can read "N% of the MFMA
roofline" while being VALU-bound.

## GEAK cross-references

The pack page above owns the **method** (two bases, calibration, refusal, floor probe). GEAK's cards
own the **datasheet facts and walk-throughs** it does not restate:

| topic | GEAK card |
| --- | --- |
| roofline concept, bottleneck vocabulary | `perf_knowledge/optimization/roofline_and_bottlenecks.md` |
| placing a kernel on the MI roofline with rocprof-compute | `perf_knowledge/profiling/roofline_on_mi.md`, `perf_knowledge/profiling/rocprof_compute_workflow.md` |
| gfx950 peaks / clocks / memory ladder | `perf_knowledge/hardware/cdna4_mi350/{peak_tables,clocks_power,memory}.md` |
| gfx942 downgrade peaks / clocks / memory ladder | `perf_knowledge/hardware/cdna3_mi300/{peak_tables,clocks_power,memory_hierarchy}.md` |
| HBM / Infinity Fabric / MALL, ridge | `perf_knowledge/hardware/shared/hbm_infinity_fabric.md` |
| MFMA peak formula, per-shape cycles | `perf_knowledge/hardware/shared/matrix_core_mfma_smfmac.md`, `perf_knowledge/hardware/cdna4_mi350/matrix_core_blockscale.md` |

A peak quoted in any of those cards is a `datasheet` denominator: it ranks, it does not gate.
