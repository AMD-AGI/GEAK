# Bound-class signals (AMD counters → lever)

Discriminators to reason with (no rule engine); the cutoffs are heuristics. Profiles: `profile_kernel.sh`
under `gpu_lock.sh`; parsers `parse_pmc.py`, `hotspot_analyzer.py`, `att_opclass.py`,
`asm_loop_audit.py` (`kernel_workflow/scripts/kernel_tools/CARD.md`). rocprof caveats (`--roof-only`
needs rocprof-compute >= 3.6.0 on gfx950; `TCC_HIT` also counts 64 B write requests there):
`roofline-models.md ### rocprof-compute caveats that move a numerator or a denominator`.

## Bound-class decision tree — the AMD discriminator table

| Dominant signal | Class | Common misread |
| --- | --- | --- |
| `MfmaUtil` low, occupancy adequate (>= 2 waves, no VGPR/LDS limiter) | compute / under-fill (tile growth is the hypothesis to test) | routed to occupancy (`## under-fill vs occupancy (the low-MfmaUtil fork)`) |
| `VALUBusy` high (not `VALUUtilization`) | VALU-bound; split compute / layout-convert / address IOPs (`2.1.1 VALU IOPs` >> `2.1.0 VALU FLOPs` = address math) | `VALUUtilization` ~100% alone |
| `asm_loop_audit` VALU% >> MFMA% on a low-precision GEMM | VALU / software-dequant | "% of MFMA roofline" |
| `VALUUtilization` ~100%, `VALUBusy` and `MfmaUtil` low | dependency stall | "VALU-bound" |
| VALU active threads/wave << 64 or `2.1.12 Branch Util` high | divergence | VALU-bound |
| `SQ_LDS_BANK_CONFLICT / SQ_LDS_IDX_ACTIVE` above a low cutoff (~10%, an uncalibrated heuristic; no tool reads it) (both counters respond on gfx950, rocprofv3 1.3.5), or a `ds_read_b128` interval well above a conflict-free baseline measured under the same SIMD load (the absolute interval moves with how many SIMDs read) | LDS bank conflict — suggested: read the ratio before calling a kernel VALU- or compute-bound | compute-bound |
| LDS ops/MFMA high, conflicts low, sync-removal floor probe flat | LDS issue rate | barrier stall |
| VMEM %-of-peak high (~80% of a measured ceiling, an uncalibrated heuristic; `mem_bw_probe.py` for the ceiling) | memory / bandwidth: cut bytes (more in-flight is inert) | adding prefetch |
| L2 hit low, pipe not saturated, coalescing not low (~60% cutoff, uncalibrated heuristic) | memory / L2 locality: `GROUP_SIZE_M` / XCD remap | |
| L2 hit low **and** coalescing low | access pattern (coalesce / grow tile); remap is the weaker hypothesis | l2-locality |
| L2-Fabric latency high (gfx950: `hw_constants.l2_fabric_latency_high_cyc`; no value for other arches), pipe not saturated | memory / latency: more outstanding loads | bandwidth-bound |
| `Instr Cache Hit Rate` low | ifetch: less unroll | |
| AGPR use with no MFMA, scratch | VGPR -> AGPR spill / register-bound | |
| `tail_efficiency` << 1, small grid | wave quantization (`roofline-models.md ## Saturation / wave quantization`) | load imbalance |

## under-fill vs occupancy (the low-MfmaUtil fork)

Low `MfmaUtil` points two opposite ways: occupancy **adequate** (>= 2 waves AND no VGPR/LDS SPI limiter)
=> the tile is likely too small (growing it, at lower occupancy, is the test); occupancy **inadequate** =>
the capping resource (SPI `6.2.5` / `6.2.7`) is the first suspect. A single crude knob (forcing waves/EU -> spill)
failing does not prove an occupancy wall. Units: `wg/CU` != `waves/SIMD` unless `num_warps == 4`;
occupancy input is `ceil8(ceil4(next_free_vgpr))`
(`planning-constants.md ### Read the field the hardware reads, and round it before you compare`).

## Stall-reason → lever (latency/occupancy sub-classification)

| Dominant stall (ATT / wavefront) | Meaning | Lever |
| --- | --- | --- |
| `s_waitcnt vmcnt` | global latency | async `buffer_load_to_shared`, coalescing, more in-flight |
| `s_waitcnt lgkmcnt` | LDS / SMEM latency or conflicts | swizzle / padding / fewer rereads |
| full-drain `lgkmcnt(0)` + lower `MfmaUtil` at equal VGPR+AGPR and occupancy vs plain, `MemUnitStalled` not dominant | schedule / overlap gap, not bandwidth | hand-written overlap: register prefetch, authored LDS ring, `warp_pipeline_stage`, then the scheduler strategy; which first depends on the tile — register prefetch first paid on small-MFMA loops, the async LDS ring was the parity route on gfx950 GEMM-like loops (`../tile-programming/pipeline.md`) |
| `s_barrier` | waves waiting for each other at CTA barriers (on 3.8.0 the compiler inserts one after each `wait_group` and before each refill of a ring slot) | fewer sync points per iteration (fewer handoffs, deeper ring, larger stages) or less imbalance between waves; race-test any change. `load_shared_relaxed` does not remove the barrier a `wait_group` brings (compile-checked 3.8.0, gfx950; `../tile-programming/pipeline.md ### What the compiler inserts after wait_group: barriers and load_shared_relaxed`) |
| compiler `s_nop` after an MFMA whose result a VALU reads next | exposed MFMA-write -> VALU-read hazard (gfx950: 5 / 8 / 12 / 20 wait states for 2- / 4- / 8- / 16-pass MFMA; independent instructions count toward it) | reorder independent work between them when it exists; in a serial chain create it with unroll or more waves; a deliberate stage-head `s_nop` is a phase lever instead |
| Dependency Wait dominant | waves stopped at `s_waitcnt` (global or LDS) or `s_sleep` = exposed latency, not necessarily bandwidth (C1) | more independent work in flight (prefetch, more waves), or shorten the load -> use chain |
| Issue Wait dominant | waves ready but unable to issue (pipe back-pressure, arbitration loss) = a contended pipe; read it with that pipe's busy counter — a saturated MFMA pipe shows high Issue Wait | more waves did not help there; cut or spread work on the busy pipe (check VALU / MFMA / LDS busy). A latency-bound register chain, where an extra wave did help, read low Issue Wait and Active-dominant |
| low occupancy / idle SIMDs | too few waves (C2) — shows as low occupancy, not as Issue Wait | fill grid / relieve the wave cap |

Definitions: rocprof-compute `gfx950_metrics.yaml` (Dependency + Issue + Active = Wave Cycles). The wait
counters are summed over resident waves, so read them with occupancy, as fractions of `Total Wave Cycles`;
re-read the split after every tile / warp change. The Dependency / Issue rows were checked with
`SQ_WAIT_ANY` / `SQ_WAIT_INST_ANY` plus ATT stall attribution on single-cause microkernels (gfx950,
rocprofv3 1.3.5); "Issue Wait + low busy = too few waves" was not supported there. **Overrides KB**:
`kernel_workflow/knowledge/profiling_guide.md` (Key ratios) reads `Dependency Wait` as a math-unit data
dependency and `Issue Wait` as too few resident waves; use the two rows above instead.

### NV/NCU levers with no AMD hardware counterpart

Named-barrier warpgroups, TMA / `cp.async.bulk` / TMEM, `wgmma` async MMA, `setmaxnreg`, thread-block
clusters / DSMEM have no CDNA3 / CDNA4 hardware counterpart, so recipes built on them (including the
NVIDIA warp-specialized producer/consumer form) do not port as written; wave role splits with ordinary
barriers remain possible in HIP / asm, but are not expressible in Gluon 3.8.0 on gfx950 (no `warp_id` / `lane_id`
role branch; `warp_specialize` hard-errors — `../tile-programming/warp-pipeline.md`). The AMD analog of bulk async copy is the simple `buffer_load_to_shared`
direct-to-LDS path.

## Degraded-mode PMC discriminators (when ATT is unavailable)

Fix the profiler first (`../method/triage.md`). Then: `SQ_WAIT_ANY / SQ_BUSY` (stall-bound; split
latency / over-sync / issue-rate by a floor probe), `SQ_LDS_BANK_CONFLICT / SQ_LDS_IDX_ACTIVE`
(bank conflict), LDS ops per MFMA (issue rate). `parse_pmc.py --print-groups` lists collectable sets;
a group too large for one pass aborts (rc 134).

## rocprof-compute block-ID cheat-sheet (when a rocprof-compute report is available)

- SOL `2.1.x`: `2.1.0 VALU FLOPs`, `2.1.1 VALU IOPs`, `2.1.8 SALU Util`, `2.1.9 VALU Util`,
  `2.1.10 MFMA Util`, `2.1.11 VMEM Util`, `2.1.12 Branch Util`, `2.1.14 IPC`, `2.1.15 Wavefront
  Occupancy`, `2.1.16 LDS BW`, `2.1.17 LDS Bank Conflicts/Access`, `2.1.18-21` vL1D/L2 hit + BW,
  `2.1.22-25` L2-Fabric BW + latency.
- SPI limiters `6.2.x`: `6.2.5 Insufficient VGPRs`, `6.2.7 Insufficient LDS` (+ SGPR / Barrier /
  Waves); the larger of two nonzero fields limits more.
- Per-kernel roofline `4.x`, launch params `7.1.x`. IDs are version-sensitive: probe per build.

## Lever cards (bound-class × applicability)

Timing at the contract boundary is the keep/revert arbiter; a counter only localizes (a rate
counter can stay flat while the kernel speeds up).

### Universal (coarse) vs fine levers

Coarse signals that fire regardless of bound class: a large layout-convert share of hot-loop VALU, a
large address-IOPs share (cutoffs around 15% / 25% are uncalibrated heuristics; no tool reads them). Tile
under-fill counts only when `MfmaUtil` is low (a ~25% cutoff is likewise a heuristic); between that and the compute bar a long VALU run is
VALU-volume-bound, not under-fill.

## Lever gating laws (single source of truth)

- **Occupancy gates latency-removing levers:** removing exposed serial latency pays most reliably at
  low occupancy; with 2+ resident waves it was often neutral-to-negative, with measured exceptions where
  the removed latency sat on the MFMA dependency chain (not automatic either way - A/B).
- **Cutting an idle unit's ops is usually neutral** (its `*Busy` far below saturation), unless they sit on the critical path.
- **Launch-overhead savings from fusion appear only when launches are exposed** (fusion can still pay
  through bytes removed; `roofline-models.md ## Kernel time decomposition`).
- **Efficacy before timing:** confirm the edit landed (ISA / counter moved) before reading TFLOPS.
- **Gluon 3.8.0 has no auto-pipeliner** and no pass reads `num_stages` (compile-checked: identical ISA);
  the overlap is authored.

## Reference-free ATT reading (no gold / vendor kernel at runtime)

References: the analytic per-resource budget (`roofline-models.md ## Feeds-and-speeds`), `full -
floor` from a floor probe, your previous round's ATT, the best point of your own sweep. A vendor
kernel (`mfma_efficiency.py --compare`) is optional; before declaring a hardware wall, diff its ISA
inventory (`asm_loop_audit.py`): if it reaches what you call impossible, the limit is your
formulation. JIT comparator machine code is readable from the JIT cache.

## Rules (necessary-not-sufficient)

- ~100% utilization != bound by that unit: read busy counters and A/B (remove work).
- **Scaled MFMA counters:** `MfmaUtil` / `SQ_VALU_MFMA_BUSY_CYCLES` **do** count scaled MFMA on gfx950 (rocprofv3 1.3.5, Triton 3.8.0); the per-format op counters split by operand format — fp8 x fp4 under `SQ_INSTS_VALU_MFMA_MOPS_F8`, fp4 x fp4 under `SQ_INSTS_VALU_MFMA_MOPS_F6F4` — so read the op counter for the formats you issue. gfx942 fp8
  is non-scaled (regular MFMA counters).
- SALU / scalar-address cost folds into VALU integer-address, not a separate class.
