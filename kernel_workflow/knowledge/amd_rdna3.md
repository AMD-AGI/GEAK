# AMD RDNA3 gfx1100 / Radeon Pro W7900 Reference — DETECT THE BOX FIRST

This file is the hardware reference for the **AMD Radeon Pro W7900** (product target `w7900`,
discrete RDNA 3, `gfx1100`, Navi 31). GEAK uses it when the identity check in §0 reports that
pair. On the W7900, kernels use:

- **wave32** (§2);
- **gfx11 WMMA** with FP16, BF16 and INT8 inputs; INT4 is not validated (§3);
- **64 KiB of LDS per workgroup**, with the occupancy table in §2;
- the **`rocprofv3` kernel trace** for timing, plus the **12-metric safe set** when
  `COUNTER_GUIDANCE=gfx1100-safe-set` (§4);
- the published W7900 peaks in §1 when `ROOFLINE_STATUS=calibrated-w7900`.

Occupancy / LDS constants below are copied from the Gluon skill
`perf_knowledge/expert_skills/skills/gluon_authoring/references/hardware/hw_constants.json`
(`gfx1100`). Roofline peaks come from AMD's published W7900 figures, copied into the
`product: w7900` section of `e2e_workflow/knowledge/analysis_skills/roofline/peaks.md`.

## 0. Detect THIS box first (source of truth > this table)

```bash
# Structured identity: only the exact marketing names "AMD Radeon Pro W7900" and
# "AMD Radeon Pro W7900 Dual Slot" (either "Pro" or "PRO" casing) on gfx1100 give target w7900.
python3 "$SKILL_DIR/../scripts/gpu_identity.py"

# On a multi-GPU host, mask the selected physical device at the ROCr layer before reading any
# other per-agent field. HIP_VISIBLE_DEVICES does not filter rocminfo.
ROCR_VISIBLE_DEVICES=<physical-id> rocminfo
```

- Identity rule: GEAK runs this guidance on `gfx1100` with the structured
  `device_target=w7900`. Use `gfx` for ISA facts and `device_target` for product policy.
- The W7900 has **96 CUs** (48 WGPs). Take the CU count from `rocminfo` and size the grid to it.
  HIP `multiProcessorCount` and torch `multi_processor_count` report **48** on this board, the
  WGP count; use the 96 from `rocminfo` wherever you size by CU.

## 1. Board facts and roofline peaks (`ROOFLINE_STATUS=calibrated-w7900`)

| Field | Radeon Pro W7900 | Source |
|-------|------------------|--------|
| Arch / gfx | RDNA3, `gfx1100` (Navi 31) | `rocminfo` |
| Compute units | 96 | `rocminfo` |
| Memory | 48 GB GDDR6, 864 GB/s | AMD product pages |
| Infinity Cache | 96 MB | AMD product pages |
| L2 cache | 6 MB | HIP `l2CacheSize` |
| Matrix FP16 | ~123 TFLOP/s | AMD product pages |
| Matrix BF16 | not published | AMD lists no BF16 figure |
| Matrix INT8 | ~123 TOPS | AMD product pages |
| Matrix INT4 | 245 TOPS published; not validated, no roofline ceiling | AMD product pages; §3 |
| Vector FP32 | ~61.3 TFLOP/s | AMD product pages |

The AMD product pages are
[Radeon PRO W7900](https://www.amd.com/en/products/graphics/workstations/radeon-pro/w7900.html) and
[Radeon PRO W7900 Dual Slot](https://www.amd.com/en/products/graphics/workstations/radeon-pro/w7900-dual-slot.html),
read on 2026-11-20. Both list 96 CUs, 61.3 TFLOPs FP32 vector, 123 TFLOPs FP16 matrix, 123 TOPs
INT8 matrix, 245 TOPs INT4 matrix, 864 GB/s and 96 MB Infinity Cache. The same values are the
`product: w7900` section of `e2e_workflow/knowledge/analysis_skills/roofline/peaks.md`, without
INT4.

`ROOFLINE_STATUS=calibrated-w7900` means GEAK uses these product-scoped published peaks. They
are datasheet numbers, not peaks measured on this box, so label the denominator you used.

- Memory-bound: `min_time ≈ bytes_moved / 864e9`. An achievable streaming rate is below the
  nameplate; measure a working set larger than the 96 MB Infinity Cache for it. A hot working
  set served by that cache can show effective bandwidth above 864 GB/s; that is cache residency,
  not impossible GDDR bandwidth.
- Compute-bound: `min_time ≈ FLOPs / 123e12` for dense FP16 WMMA, `OPs / 123e12` for INT8 WMMA,
  and `FLOPs / 61.3e12` for FP32 vector math.
- **BF16, FP8 and INT4 kernels:** rate them on the memory axis (864 GB/s) and by measured
  latency. The compute peaks above are for FP16 and INT8 WMMA and FP32 vector math; FP8 data
  runs as FP16 WMMA after dequantization, and INT4 is not validated (§3).
- The FP16 ridge is about **142 FLOP/byte** (`123e12 / 864e9`): an FP16 kernel below it is
  memory-bound on this board.

## 2. RDNA3 fundamentals

- **Wavefront size: 32** for HIP and Triton compute on gfx1100, the default build.
  Shuffle/ballot/any/all operate over 32 lanes. In Triton, `num_warps` counts **wave32** waves:
  `num_warps=4` launches 128 threads.
- **Block sizes**: multiples of **32** (32, 64, 128, 256).
- **LDS**: 64 KiB per workgroup.
- **VGPR**: 256 addressable VGPRs **per wave**, 1536 VGPRs per SIMD, allocation granule 24,
  at most 16 waves per SIMD. Accumulators live in these VGPRs.
- **Occupancy** follows the VGPR table below.

### Occupancy vs VGPRs/wave (compiler-derived, `hw_constants.json` `gfx1100`)

| VGPRs/wave | Max waves/SIMD |
|------------|----------------|
| 96         | 16             |
| 120        | 12             |
| 144        | 10             |
| 168        | 9              |
| 192        | 8              |
| 216        | 7              |
| 240        | 6              |
| 256        | 5              |

Re-derive on a new ROCm with
`perf_knowledge/expert_skills/skills/gluon_authoring/scripts/amd_occupancy.py --compiler-sweep --arch gfx1100`.

## 3. Matrix ISA: gfx11 WMMA

**Guard W7900 matrix code on `gfx1100` and use the builtins, types and layout below: they are
what compiles and computes correctly on this ISA.**

- **WMMA input types: FP16, BF16 and INT8**, dense. FP16/BF16 accumulate into FP32 (or
  FP16/BF16); INT8 accumulates into INT32.
- **INT4 is not validated.** gfx11 has an INT4 WMMA (`v_wmma_i32_16x16x16_iu4`), and AMD
  publishes 245 TOPS for it, but no INT4 matrix kernel has been run and checked on the W7900 with
  the current ROCm path, and the roofline has no INT4 ceiling (§1). W4A16 (GPTQ/AWQ) kernels
  usually dequantize the INT4 weights to FP16 in the operand load and run FP16 WMMA: rate them on
  the memory axis and by measured latency.
- **FP8 models:** keep the weights in compact FP8 storage, dequantize them to FP16/BF16 in the
  operand load (apply the block scale there) and run FP16/BF16 WMMA. Triton lowers an FP8
  `tl.dot` the same way, to FP16 WMMA (`v_wmma_f32_16x16x16_f16`), so FP8 storage saves memory
  traffic and the math is FP16 WMMA.
- **Tile: 16x16x16** per instruction in wave32. HIP builtins carry the `_w32` suffix, for example
  `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c)` with `a`, `b` as 16 x `_Float16` and `c` as
  8 x `float` per lane.
- **gfx11 operand layout (wave32):** load both half-waves with the full A and B data.
  - A (16x16, row-major): lane `l` holds row `l % 16`, all 16 K values. Lanes 16–31 replicate 0–15.
  - B (16x16): lane `l` holds column `l % 16`, all 16 K values. Lanes 16–31 replicate 0–15.
  - C/D (FP32, 8 VGPRs): VGPR `v` of lane `l` holds element `D[2*v + l/16][l % 16]`.
  D is correct only when lanes 16–31 hold the same A and B data as lanes 0–15; with lanes 16–31
  zeroed, half of D is wrong (measured on the W7900).
- In Triton, `tl.dot` on FP16/BF16 lowers to WMMA on gfx1100. Starting points are **unvalidated
  seeds**: sweep `BLOCK_M`/`BLOCK_N` in {32, 64, 128}, `BLOCK_K`, `num_warps` in {2, 4, 8}
  (wave32 units), `num_stages` and `waves_per_eu`, then autotune.
- **WMMA and VALU time are additive:** plan them as serial work within a wave. RDNA3 dual-issues
  some VALU pairs (VOPD).

## 4. Profiling and counters (`COUNTER_GUIDANCE`)

- Profile with `rocprofv3 --kernel-trace`; it captures dispatch durations on gfx1100, and
  `profile_policy.sh` puts it first.
- gfx1100 counters read correctly only at the `profile_standard` amdgpu perf level
  (`STABLE_STD` in amd-smi). The operator sets it and declares it in the handoff; the lane turns
  that into `COUNTER_GUIDANCE`. **Never change the perf level yourself** (no `amd-smi set`, no
  sysfs write): it is the operator's setting, and changing it mid-run also invalidates the
  frozen baseline timing.
- **`COUNTER_GUIDANCE=unavailable`**: profiling is off (the lane's `args.profile`, off by
  default), or the level was not declared as `profile_standard`. Classify
  the bottleneck from kernel-trace durations, per-case latency, dispatch count and the occupancy
  table above.
- **`COUNTER_GUIDANCE=gfx1100-safe-set`**: profiling is on and the level was declared as
  `profile_standard`. Use
  exactly these 12 Metrix metrics, the ones IntelliKit validated on a Radeon Pro W7900
  (AMDResearch/intellikit#203, merged as `1260c08f`; listed under "gfx1100" in
  IntelliKit's `metrix/README.md`):

  | Metrics | What they measure on gfx1100 |
  |---------|------------------------------|
  | `compute.gpu_utilization`, `GPUBusy` | fraction of cycles the GPU was busy |
  | `Wavefronts`, `VALUInsts`, `SALUInsts` | waves launched; vector and scalar ALU instructions |
  | `OccupancyPercent` | achieved wave occupancy |
  | `memory.hbm_read_bandwidth`, `memory.hbm_write_bandwidth`, `memory.hbm_bandwidth_utilization`, `memory.bytes_transferred_hbm` | traffic leaving L2 for the memory fabric |
  | `memory.lds_bank_conflicts`, `memory.lds_bank_conflict_percent` | LDS bank conflicts per access, and % of active LDS cycles with one |

  - **Check the installed Metrix first.** Use the safe set when `metrix list metrics` on the box
    reports `Architecture: gfx1100` with no "backend unavailable" note, and lists
    `memory.lds_bank_conflict_percent` without `memory.l2_hit_rate`: that is the listing of a
    Metrix with that change. Otherwise, follow the `unavailable` path. GEAK's `pyproject.toml`
    pins upstream IntelliKit `2f61453a`, from before #203, so run the check each time.
  - Then collect them after the kernel trace, for example
    `PROFILER_PRIORITY=metrix METRIX_ARGS="-m GPUBusy,OccupancyPercent,memory.hbm_read_bandwidth,memory.hbm_write_bandwidth" bash "$SKILL_DIR/scripts/profile_kernel.sh" ...`.
  - The `memory.hbm_*` metrics count L2-to-fabric requests, and the 96 MB Infinity Cache can
    serve them, so read them as an upper bound on GDDR traffic. Read and write counters come from
    separate passes, so the benchmark must repeat the same dispatches.
  - Infer the L2-bound and matrix-bound classes from latency and the §1 peaks.
- A profile from the kernel trace and the latency table is complete on the W7900, with or without
  counters.
- For library baselines, bake off hipBLASLt, Triton and HIP on this box.

Start cold: use learned cards tagged `platforms: ['gfx1100']`, and tag new wins
`platforms: ['gfx1100']`.

## Critical Rules

1. **NEVER** set `HIP_VISIBLE_DEVICES` inline with profiler commands — always go through `gpu_lock.sh`.
2. Wavefront-level ops operate on **32** threads.
3. Matrix math is gfx11 **WMMA** (`_w32` builtins, A and B replicated across both half-waves).
4. WMMA inputs are FP16, BF16 and INT8; INT4 is not validated (§3); FP8 data is dequantized to
   FP16/BF16 first.
5. Size the grid to the DETECTED CU count (96 on the W7900).
6. With `ROOFLINE_STATUS=calibrated-w7900`, use the §1 peaks: memory, FP16 and INT8 WMMA, FP32
   vector. Rate BF16, FP8 and INT4 kernels on the memory axis.
7. Proceed only with `device_target=w7900` on `gfx1100` (§0).
