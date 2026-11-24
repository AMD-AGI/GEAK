# Closed-form performance models (gfx950 planning)

Tools (`KT=kernel_workflow/scripts/kernel_tools`):
`hw_budget.py --brief` (bound, floor, multiple, lever), `extract_sku.py --facts --hardware-context $EVAL_DIR/hardware_context.json` (CUs, L2/MALL,
peaks, ridge), `mem_bw_probe.py`, `parse_pmc.py`. Per-arch constants:
`planning-constants.md`. A dtype absent from a SKU row is never priced at FP16: `hw_budget.py` reports the compute floor UNKNOWN and the bound as provisional.

## Every "% of roofline" carries two bases (the reporting contract)

Name both halves of every fraction: `numerator_basis` = `model` (`hw_budget.py --workload` /
`--tensors`) or `counters` (PMC / ATT); `denominator_basis` = `datasheet` (`sku.json` peak),
`empirical@<tool>-<version>` (e.g. `rocprof-compute --roof-only`), or `in-shape probe`
(`mem_bw_probe.py` / dense MFMA loop at this kernel's mix and program count). A `datasheet`
denominator or `model` numerator **ranks hypotheses only**; `hw_budget.py` prints `[datasheet]` vs
`[calibrated]` for exactly this.

### rocprof-compute caveats that move a numerator or a denominator

- **`rocprof-compute profile --roof-only` needs rocprof-compute >= 3.6.0 on gfx95x**; older builds
  give no usable gfx950 roof. Record the version in the basis tag. gfx942 / MI325X: `roofline.csv`
  is not produced at all (`cdna3-gfx942.md ## Known tooling issues (gfx942)`).
- **gfx950: `TCC_HIT_sum` also counts 64 B write requests**, so L2 hit % on a write-heavy kernel is not
  read reuse. `FETCH_SIZE` (rocprofiler-sdk 1.3.5) matched a known read exactly; still carry two routes
  (`TCC_MISS`, `FETCH_SIZE+WRITE_SIZE`, width-sized `TCC_EA0_*`) and the model bytes
  (`### When the counter routes disagree`).
- An efficiency ratio above 1 (or an empirical BF16 peak far below datasheet) points at a mis-calibrated peak or a byte
  under-count - a sanity rule, not a measurement.
- Counter names are not interchangeable: `parse_pmc.py` sizes L2->EA traffic from `TCC_EA0_RDREQ_{128B,64B,32B}_sum`
  and `TCC_EA0_WRREQ_sum` / `TCC_EA0_WRREQ_64B_sum` (gfx950 writes go out as 64 B requests);
  `TCC_EA0_RDREQ_DRAM_sum` counts DRAM-destined reads only.

## Roofline / GEMM

Ridge = `peak_compute[dtype] / peak_hbm`, derived per dtype per call (`sku.json` stores none):
`extract_sku.py --facts <SKU>` or `hw_budget.py --sku <key> --brief`. FP16 datasheet ridges: MI355X
~312, MI350X ~288; downgrade MI300X ~247, MI325X ~218, MI308X ~65 (rank only).
GEMM minimum bytes = operand reads + block-scale bytes (MX formats) + output writes, each at its own storage
width (`hw_budget.py` adds the scale overhead).
**Attention is not GEMM:** K/V are re-read once per Q-tile, so the GEMM `minimum_bytes` model
overestimates attention intensity; use `hw_budget.py --workload attention` or the per-resource model below.
Intensity names compute vs memory, **not** the binding sub-resource of a fused kernel (VALU, exp,
dependency stall) - confirm with counters (`../method/profile.md`).

## Feeds-and-speeds (per-resource cycle roofline)

```text
T_mfma = n_mfma_instr * cyc_per_mfma        # cycles are PER INSTRUCTION SHAPE (and, for scaled
                                            #   f8f6f4, per operand formats): amd_isa.py --arch gfx950 facts <V_MFMA_...>
T_lds  = lds_bytes_read / lds_bytes_per_clk # ~256 B/clk gfx950 (64 banks), ~128 gfx942 (32 banks)
T_exp  = n_trans_ops * trans_issue_cyc      # see isa-mechanisms.md ### What fits in the window
T_iter = max(T_mfma, T_lds, T_exp, ...)     # the binding resource of one loop iteration
```

LDS read traffic per warp scales with **(warp tile M + warp tile N) x K**, not with the MFMA count: operands
are read once per K step per warp and reused across that warp's MFMA grid, so growing the per-warp tile
*lowers* LDS bytes per MFMA (compile-checked on gfx950 / 3.8.0 plain `tl.dot` hot loops: bytes per MFMA fell as
the per-warp tile grew). The LDS bound gets closer when a tile is split over more warps (each re-reads its
operand slices), with small per-warp tiles, or when operands are re-read for a second matmul. If `T_lds` or
`T_exp` > `T_mfma`, the model points at LDS-traffic / exp reduction rather than MFMA continuity.

## Asymmetric scaling (predict the migrating bottleneck)

gfx942 -> gfx950 doubled MFMA throughput and LDS read BW (128 -> 256 B/clk); the transcendental rate is
not known to have kept pace (gfx950 measured: TRANS 8 cycles per op per SIMD; the gfx942 rate was not measured
here). If it did not, fused kernels (softmax-between-matmul) migrate toward exp / dependency
latency / LDS operand rereads on gfx950; check the exp share on the gfx950 trace rather than assume it. gfx1250 is out of scope.

## MFMA efficiency (primary compute-bound metric)

`MFMA_efficiency = mfma_cycles_in_loop / iteration_duration`. **An instruction trace reports it per
wave; the target (~98%) is per SIMD:
`per_SIMD = per_wave x waves_per_SIMD`.** At 2 waves/SIMD (`warp_pipeline_stage` ping-pong) a kernel
near 90% reads ~45%. Valid only when MFMA issue is the binding sub-resource.

## Low-precision / dequant-VALU (check op-mix before the FLOP roofline)

The FLOP roofline has no axis for operand converts (int4/int8 unpack, fp8 -> bf16 upcast feeding
regular MFMA). Run `asm_loop_audit.py` and compare VALU% vs MFMA% before trusting "% of MFMA peak"
(`optimization-gotchas.md`, row 5; `../tile-programming/low-precision.md`).

## Occupancy

Use `probe.py plan` / `probe.py measure` and `planning-constants.md` (VGPR ladder, read-the-field,
`; Occupancy: N` caveats, wg/CU conversion). Measured cross-check: rocprof-compute SPI limiters
`6.2.5 Insufficient VGPRs` / `6.2.7 Insufficient LDS` + `2.1.15 Wavefront Occupancy`
(`bound-class-signals.md ## rocprof-compute block-ID cheat-sheet`).

### Occupancy as a lever: latency/memory-bound work

More waves are inert when a compute unit is already saturated, and even for latency-bound work the
criterion is one-sided: an occupancy **loss** can veto a change; a projected **gain** does not
justify one (`planning-constants.md ## Occupancy is a one-sided criterion`).

## Saturation / wave quantization

`waves = ceil(grid_tiles / CUs)`, `tail_efficiency = grid_tiles / (waves * CUs)` (CUs: 256
MI355X/MI350X, 304 MI300X/MI325X, 80 MI308X); `saturation = grid_tiles / CUs` is only the coarse
fill check. The formula locates the steps (measured on a gfx950 warp-pipelined GEMM: crossing a CU
multiple cost far more than the added tiles), but `1 - tail_efficiency` overstates the waste — a partly
filled wave ran faster per tile than a full one. Levers (A/B each): an exact-fill tile, a wave split
(full waves on the big tile, the tail on a smaller one), split-K, a persistent grid. Not the
load-imbalance tail (LPT remap). One-sided: `planning-constants.md ## Tail efficiency is one-sided too`.

## HBM / TCP in-flight model

`BW_total = min(inflight_bytes_per_CU, 32 KiB) / hbm_latency * min(num_workgroups, CUs)`, with
`inflight_bytes_per_CU = req_per_wave * bytes_per_req * active_waves` (TCP cap:
`hw_constants.json tcp_inflight_cap_kib`). Below the HBM ceiling achievable bandwidth grows with bytes in
flight per CU; on MI350X a streaming read reached the HBM ceiling well before 32 KiB in flight per CU, so there
the cap is not the binding term (GPU-measured, gfx950). On the Gluon path the prefetch depth is the ring you
**authored** (no 3.8.0 Gluon pass reads `num_stages`).

## LDS throughput (conflict diagnostic)

Conflict-free `ds_read_b128` steady interval 16 cyc on gfx950 with all four SIMDs reading (2-way 32, 4-way 64;
one wave per CU sees 8; gfx942 unverified, `cdna3-gfx942.md`); `ds_read_b64` ~8 cyc (GPU-measured, gfx950).
Observed interval > `hw_constants.json ds_read_b128_conflict_free_cyc` under that load => bank conflicts; under
a lighter or paired-SIMD load compare against a conflict-free baseline measured the same way. Bank
geometry: `planning-constants.md ## LDS throughput / conflict (planning)`.

## Kernel time decomposition

`T_CTA = T_prologue + num_k_tiles * max(T_mfma, T_lds, T_exp, T_hbm_residual, T_convert, T_spill) +
T_epilogue`. **Launch fusion:** the launch-overhead saving is largest while launches are exposed (eager). Under a
CUDA/HIP graph the host launch cost disappears but each captured dispatch still has a measurable device-side cost
on gfx950 (`floor2 - floor1`, `../workloads/gemm-small-m.md`), so fusing small launches can still pay under a
graph — only when that cost exceeds what the fusion adds; a fused per-tile recompute or serialisation (e.g. a
last-arriver reduce) can raise `T_CTA` and lose. Validate at the
production boundary (`../method/benchmark-hygiene.md`).

## Calibration

Datasheet numbers rank; replace them with measurements at this shape on this box: `peak_mfma`
(dense in-shape MFMA loop -> `--measured-tflops lo,hi`), `peak_hbm` (`mem_bw_probe.py` ->
`--measured-hbm-tb-s lo,hi`), LDS (observed `ds_read` interval), registers (`.amdgcn` KD).

### Zeroth question: does a roofline apply to this kernel at all?

| precondition (`hw_budget.py` flag) | tell when violated | voids |
| --- | --- | --- |
| grid fills the machine (`--grid`) | time barely moves while bytes change several-fold | both floors; fix is parallelism |
| working set exceeds the MALL (`--footprint-mb`) | measured time below the memory floor (a re-run served from the MALL can beat a floor built from measured HBM bandwidth) | memory floor only |
| timed region is one kernel (`--dispatches`) | several dispatches per iteration | budget each dispatch separately |

The in-shape HBM ceiling **rises with program count** (`mem_bw_probe.py --nprog-sweep`); count
*touched* bytes, not allocation sizes.

### The numerator is a manifest, not a formula

When the kernel does not match an archetype in structure (per-tensor dtypes, never-dereferenced
pointers, operand re-read per tile, dead grid dim, padded rows, `q_len == 1`, routed experts only),
declare what it moves: `hw_budget.py --tensors "name:dir:dtype:dims[:xN]" --flops <n>
--flops-engine mfma|valu`. It returns a bracket (unique footprint .. declared traversals).

### These counters measure fabric traffic, not DRAM traffic

`TCC_MISS`, `FETCH_SIZE`, `WRITE_SIZE`, `TCC_EA0_*` are sampled at the L2/fabric boundary and
**cannot see the MALL** (memory-side LLC, 256 MiB datasheet; MI350X reports 224 MiB, `sku.json mall_mb_device`): a rate from them is an upper bound on
DRAM rate until the footprint clears the MALL.

### When the counter routes disagree

On a real kernel routes can diverge several-fold. Check units on a known-bytes probe; if it agrees,
the spread is access shape (whole-line miss counting misses partial-line stores / write-allocate /
atomics and reads **low**). Pass every route (`--measured-dram-mb a,b,c`) and report the width; a
bracket straddling the ridge leaves the regime undetermined.

### Both terms of a memory roofline, and both default the wrong way

```bash
python3 $KT/parse_pmc.py <prof_dir> <kernel_substr>              # L1: measured bytes + rate
python3 $KT/mem_bw_probe.py --hardware-context $EVAL_DIR/hardware_context.json --runlen <r> --stride <s> --rw-mix <r>:<w>   # L2: in-shape ceiling
python3 $KT/hw_budget.py --hardware-context $EVAL_DIR/hardware_context.json --workload <w> --shapes ... --measured-ms <ms> \
    --measured-dram-mb <mb>[,<mb>...] --measured-hbm-tb-s <lo>,<hi>      # or --measured-tflops <lo>,<hi>
# no context file: --sku <SKU> names the part (MI300X / MI325X on gfx942)
```

- **A FLOP count is a compute floor only on the engine it issues on:** reduction / norm / scan /
  elementwise have no MFMA; declare `--flops-engine valu` or the memory verdict is a default.
- **A missing ceiling is not a zero ceiling:** no rate for a dtype = floor unknown; never borrow
  another dtype's rate.
- Quote ceilings as `lo-hi` with program count; a gap inside the range is not a finding.

### Before you call a kernel memory-bound: floor-probe it

`memory_ideal < non_stream_floor` (streams made cache-resident, kernel still slower) => memory is not
the binding constraint and the roofline gap is not a prize (`../method/profile.md`).
