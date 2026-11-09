# AMD Ryzen AI / Radeon Client Hardware Reference (RDNA) — DETECT THE BOX FIRST

Hardware facts for AMD client parts. Read this when `rocminfo` reports a `gfx11*` arch.

Currently tabulated: **RDNA 3.5**, `gfx1151` — Radeon 8060S, the integrated GPU in the Strix Halo APU.

## 0. Detect THIS box first

`gfx1151` covers several SKUs — the 8060S (40 CU), 8050S (32 CU) and 8040S (16 CU) report the same
arch string. Read the CU count from `rocminfo` rather than from the arch. Scope the query to the gfx
agent; the CPU agent is listed first and its "Compute Unit" line is the core count, not the GPU's:

```bash
rocminfo | awk '/Name:.*gfx/{f=1} f&&/Compute Unit:/{print $3; exit}'
```

| Part | Arch | CU | Memory | Peak BW | fp8 matrix |
|---|---|---|---|---|---|
| Radeon 8060S | RDNA3.5 / `gfx1151` | 40 | unified, BIOS-sliced | 0.256 TB/s theoretical | none |

Memory is a BIOS-configured slice of system LPDDR5X shared with the CPU, not a fixed VRAM pool. The
cache hierarchy is 2 MiB L2 backed by a 32 MiB Infinity Cache tier in front of DRAM. Single die.

Measured bandwidth depends on the experiment. The shared
[`sku.json`](../../perf_knowledge/hardware/data/sku.json) retains a historical 212 GB/s report
for `AI_MAX_395`; its tool version was not recorded and its source is no longer available.
The [2026-11-06 Radeon 8060S campaign](../../perf_knowledge/hardware/rdna35_gfx1151/peak_tables.md)
measured 230.5–233.3 GB/s reads under the documented conditions. Re-measure on the target box
before using either as an attainable ceiling; neither is an architecture-wide calibration.

## 1. Machine model

| | RDNA3.5 (`gfx1151`) |
|---|---|
| Wavefront | 32 lanes, `__ballot` → 32-bit |
| Matrix ISA | WMMA (`v_wmma_*`) — bf16 / fp16 / iu8 / iu4 |
| fp8 matrix path | none |
| Register file | 1536 VGPR/SIMD, 256 addressable per wave |
| VGPR granule / waves | granule 24, up to 16 waves/SIMD |
| SIMDs | 2 per CU; CUs pair into a WGP |
| LDS | 64 KiB max per workgroup; occupancy-pool scope unverified |
| Memory | LPDDR5X, shared with the CPU |

## 2. Applying these numbers

- **Occupancy** follows from the machine model above — wave32, 1536 VGPR/SIMD, granule 24, cap 16
  waves/SIMD. 249 VGPRs is 5 waves/SIMD here. The per-arch table is
  `perf_knowledge/hardware/data/hw_constants.json`; query it with
  `python3 kernel_workflow/scripts/kernel_tools/amd_occupancy.py --vgpr N --arch gfx1151` (the arch
  is required -- the tool refuses rather than default to a CDNA model).
- **VGPR cost per tile.** At wave32 a given tile occupies twice the VGPR per lane that it would at
  wave64, because half as many lanes share it.
- **The Triton `matrix_instr_nonkdim` and `kpack` knobs are accepted and silently ignored** — they
  do not raise, and they do not change the generated code. gfx1151 maps to `ISAFamily::RDNA3`
  (`TargetFeatures.cpp`), which registers the `BlockedToWMMA` pattern; that pattern's constructor
  takes `nonKDim` and never stores it, and `kPack` is only passed to `BlockedToMFMA`. Autotuning
  over either produces identical configs. (Verified against Triton 3.8.0,
  `third_party/amd/lib/TritonAMDGPUTransforms/AccelerateAMDMatmul.cpp`.)
- **There is no matrix-unit hardware utilization counter on this part**, so a profiler reporting
  nothing for a matrix-utilization metric indicates the counter is absent, not that the matrix unit
  was idle.
- **Backend availability is a probe result, not an arch inference.** aiter, CK and hipBLASLt all
  build for gfx11. Probe the install rather than inferring availability from the arch.
- **No LDS-per-CU number is defined** — the per-workgroup cap is 64 KiB, but the
  occupancy-pool scope is not established here, so `amd_occupancy.lds_per_cu("gfx1151")` returns `None`. `perf_knowledge/hardware/data/hw_constants.json` records only
  `lds_per_wg_kib` for gfx1151; `lds_per_wgp_kib` is deliberately absent (its `_lds_basis` explains
  why the per-WGP scope is not established for RDNA3.5), so LDS occupancy is reported as not priced
  rather than derived. Do not fill it in from gfx1100 / gfx1200.
- **fp8 regimes run on an emulated path.** torch exposes `float8_e4m3fn` as a storage dtype
  independently of the hardware, so an fp8 regime allocates and executes on a part with no fp8 matrix
  unit, at a fraction of the bf16 rate. A measurement taken that way describes the emulation.
- **Roofline denominators** are in
  `e2e_workflow/knowledge/analysis_skills/roofline/peaks.md` (`gfx1151` section).
- **Measured long form.** `$SKILL_DIR/../perf_knowledge/hardware/rdna35_gfx1151/` (start at
  `arch.md`) holds what was measured on a Radeon 8060S — achievable bandwidth, compute peaks by
  shape, clock and power under load, operand-path levers, profiler defects — and the scripts that
  re-take each number. Read it when a number here needs its achievable value or its measurement
  conditions.
- **Memory-utilization readings include the Infinity Cache.** The 32 MiB tier serves the working set
  before DRAM is reached, so a low DRAM-side reading can reflect cache residency. Compare against the
  working-set size before attributing it to bandwidth.
