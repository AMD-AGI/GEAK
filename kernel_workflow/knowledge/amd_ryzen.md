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
| Radeon 8060S | RDNA3.5 / `gfx1151` | 40 | unified, BIOS-sliced | ~0.26 TB/s | none |

Memory is a BIOS-configured slice of system LPDDR5X shared with the CPU, not a fixed VRAM pool. The
cache hierarchy is 2 MiB L2 backed by a 32 MiB Infinity Cache tier in front of DRAM. Single die.

## 1. Machine model

| | RDNA3.5 (`gfx1151`) |
|---|---|
| Wavefront | 32 lanes, `__ballot` → 32-bit |
| Matrix ISA | WMMA (`v_wmma_*`) — bf16 / fp16 / iu8 / iu4 |
| fp8 matrix path | none |
| Register file | 1536 VGPR/SIMD, 256 addressable per wave |
| VGPR granule / waves | granule 24, up to 16 waves/SIMD |
| SIMDs | 2 per CU; CUs pair into a WGP |
| LDS | 128 KB per WGP, 64 KB max per workgroup |
| Memory | LPDDR5X, shared with the CPU |

## 2. Applying these numbers

- **Occupancy** follows from the machine model above — wave32, 1536 VGPR/SIMD, granule 24, cap 16
  waves/SIMD. 249 VGPRs is 5 waves/SIMD here. The per-arch table is
  `perf_knowledge/expert_skills/skills/gluon_authoring/references/hardware/hw_constants.json`; query
  it with `scripts/amd_occupancy.py --vgpr N --arch gfx1151`.
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
- **No LDS-per-CU number is defined** — shared memory is 128 KB per WGP with a 64 KB per-workgroup
  cap, so `amd_occupancy.lds_per_cu("gfx1151")` returns `None`. Read `lds_per_wgp_kib` and
  `lds_per_wg_kib` separately rather than deriving a per-CU figure.
- **fp8 regimes run on an emulated path.** torch exposes `float8_e4m3fn` as a storage dtype
  independently of the hardware, so an fp8 regime allocates and executes on a part with no fp8 matrix
  unit, at a fraction of the bf16 rate. A measurement taken that way describes the emulation.
- **Roofline denominators** are in
  `e2e_workflow/knowledge/analysis_skills/roofline/peaks.md` (`gfx1151` section).
- **Memory-utilization readings include the Infinity Cache.** The 32 MiB tier serves the working set
  before DRAM is reached, so a low DRAM-side reading can reflect cache residency. Compare against the
  working-set size before attributing it to bandwidth.
