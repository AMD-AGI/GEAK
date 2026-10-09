# AMD RDNA4 gfx1201 ISA / R9700 Product Reference — DETECT THE BOX FIRST

The ISA guidance in this file applies to **RDNA 4 client gfx1201**. Product
validation, serving images, and calibrated peaks apply specifically to the
**Radeon AI PRO R9700**. A different gfx1201 product remains uncalibrated. This
is not Instinct CDNA. Do **not** apply
`amd_instinct.md` §2–3 (wave64, MFMA tiles, FNUZ/MX, 512-VGPR combined formula) to this box.

Numeric occupancy / LDS constants below are copied from GEAK's shared hardware data
`perf_knowledge/hardware/data/hw_constants.json` (`gfx1201`). Peak TFLOPS and memory bandwidth for roofline math come from the public
R9700 datasheet in `e2e_workflow/knowledge/analysis_skills/roofline/peaks.md`.

## 0. Detect THIS box first (source of truth > this table)

```bash
# Structured identity: validates one discrete identity and treats only the exact
# R9700 marketing name as product target r9700.
python3 "$SKILL_DIR/../scripts/gpu_identity.py"

# On a multi-GPU/APU host, mask the selected physical device at the ROCr layer
# before reading any additional per-agent field. HIP_VISIBLE_DEVICES does not
# filter rocminfo.
ROCR_VISIBLE_DEVICES=<physical-id> rocminfo
rocm-smi --showmeminfo vram 2>/dev/null | head              # GDDR/HBM capacity
```

- A generic Ryzen iGPU may appear as `AMD Radeon Graphics`; the identity helper
  ignores that agent only when a discrete agent remains. Any genuinely mixed
  discrete identities still fail closed.
- **`gfx125x` is not this card** — that is CDNA5 / MI450-class. Stay on Instinct/CDNA5 docs for those.
- The `gfx` id is what matters for ISA (WMMA vs MFMA, wave size, fp8 dialect). CU/WGP count is what
  matters for grid sizing. Take BOTH from `rocminfo`.
- Treat 640 GB/s as the datasheet ceiling. For an achievable-bandwidth model,
  measure a streaming working set larger than cache on this box; do not assume
  an R9700-specific efficiency factor until that measurement is recorded.

## 1. Card comparison (reference hint — verify on-box)

| Card / SKU (typical) | Arch / gfx | Family | Wave | Matrix ISA | fp8 | MX / block-scale |
|----------------------|------------|--------|------|------------|-----|------------------|
| Radeon AI PRO R9700 | `gfx1201` | RDNA4 | **32** | **WMMA only** | OCP client | **no** |
| RX 9070 XT / other gfx1201 | `gfx1201` | RDNA4 | **32** | **WMMA only** | OCP client | **no** |
`gfx1200` is deliberately unsupported by this workflow until it has independent on-box validation.
Use `gfx` for ISA facts. Use the structured `device_target`: only the exact
`AMD Radeon AI PRO R9700` name selects R9700 images and peaks; other gfx1201
products remain `unknown-device-not-r9700`.

## 2. RDNA4 fundamentals (validated on gfx1201)

- **Wavefront size**: **32** threads (NOT 64 like Instinct CDNA). Shuffle/ballot/any/all operate over
  32 lanes. HIP `__ballot` still returns `unsigned long long`, but only the low 32 lane bits can be
  active; do not infer wave width from the C++ return type. In Triton, `num_warps` is in **wave32**
  units, so the same value launches half as many threads as it does on CDNA wave64.
- **Block sizes**: multiples of **32** (64/128/256 still fine; do not require multiples of 64).
- **LDS**: 64 KiB per workgroup, 128 KiB per WGP, 32 banks. Do not assume CDNA4's 160 KiB/CU.
- **VGPR**: 256 addressable VGPRs **per wave** (this is not the per-SIMD file).
  `llvm-mc -mcpu=gfx1201` accepts `s_alloc_vgpr`; therefore it is wrong to call
  the instruction unavailable. GEAK's HIP/Triton compute paths do not currently
  demonstrate dynamic-VGPR allocation, so workflow occupancy uses the static
  ≤256 model only.
- **No GWS**. **No MFMA+VALU co-issue** (WMMA cannot hide behind VALU the way CDNA MFMA can).
- **Do not** apply the CDNA combined-512-VGPR occupancy formula.

### Occupancy vs VGPRs/wave (compiler-derived steps; ROCm 10 / LLVM 23)

Regenerated on 2026-09-21 with AMD clang 23.0.0git from the pinned ROCm 10
R9700 image using
`kernel_workflow/scripts/kernel_tools/amd_occupancy.py
--compiler-sweep --arch gfx1201`. The breakpoints matched the prior LLVM 22
sweep.

`max_waves_per_simd` = 16. Measured `vgpr_wave_steps` (VGPR count → max waves/SIMD):

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
`python3 kernel_workflow/scripts/kernel_tools/amd_occupancy.py
--compiler-sweep --arch gfx1201` (run under `kernel_workflow/scripts/gpu_lock.sh` only if a GPU is
needed; the sweep itself is compile-only).
Dividing 256 by a kernel's VGPR count (the old CDNA-style file budget) **under-reports** RDNA
occupancy 2–3×.

## 3. Arch-specific: dtype, fp8, WMMA (not MFMA)

**Branch on `gfx`. Wrong matrix ISA or fp8 dialect silently fails correctness or leaves 10× on the table.**

- **Matrix ISA is WMMA**, not MFMA. In Triton, `tl.dot` lowers to WMMA on this family. Do not pick
  CDNA `matrix_instr_nonkdim` / MFMA 16×16×16 / 32×32×8 tiles as if they were RDNA. There is **no**
  `mfma_scaled` / MXFP4 / MXFP6 / MXFP8 block-scale path. A `v_mfma_*` intrinsic or inline-assembly
  port is a build failure on gfx1201; `matrix_instr_nonkdim` is an MFMA-specific knob whose exact
  rejection/ignore behavior depends on the Triton version, not an RDNA tuning axis.
- **fp8 is OCP client**: `torch.float8_e4m3fn` / `torch.float8_e5m2`. **Never** FNUZ
  (`float8_e4m3fnuz`) — that is gfx942.
- Prefer bf16/fp16 for first kernels. Sweep fp8 only after OCP paths are proven on-box.
- **INT4 has native gfx12 WMMA** (`v_wmma_i32_16x16x{16,32}_iu4`, INT32 accumulation). This is
  distinct from the W4A16 deployment path: CDNA also runs W4A16 by unpacking/dequantizing INT4
  weights into fp16/fp8 before MFMA. Claim a native INT4 win only after the emitted ISA contains
  the gfx12 INT4 WMMA instruction.
- Triton **provisional starting hints** (not sourced defaults; measure, then autotune):
  - try `BLOCK_M = 64` alongside other tile heights
  - try `BLOCK_N = 32` alongside 16/64 on **gfx1201**
  - sweep `waves_per_eu`; `6` is an unvalidated seed, not a documented RDNA optimum
  - `num_warps` in wave32 units
- Under **CUDA/HIP graphs**, keep Triton `int64_strides=true` on gfx1201 attention unless an A/B on
  this kernel proves otherwise (`int64_strides=false` has been seen to fall off the vectorized path).

## 4. Peak FLOPS / bandwidth — public R9700 datasheet

Radeon AI PRO R9700 / gfx1201 (Navi 48, 300 W): datasheet GDDR6 pin rate **~640 GB/s**;
dense matrix **191 TFLOP/s FP16/BF16**, **383 TOPS INT8**. Full table, including
FP32/FP8/INT4 and sparse 2:4 rates, is in
`e2e_workflow/knowledge/analysis_skills/roofline/peaks.md`. Do not invent peaks
for a non-R9700 gfx120x SKU, and do not use CDNA MX/`fp4` numbers here.

- **Memory hierarchy:** 8 MiB L2 → 64 MiB Infinity Cache → external GDDR6. A hot working set served
  by cache can report effective bandwidth above the 640 GB/s external-memory pin rate; that is cache
  residency, not impossible GDDR bandwidth. Use a streaming working set larger than cache to measure
  the external-memory roof.
- Memory-bound: `min_time ≈ bytes_moved / 640e9`.
- Compute-bound: `min_time ≈ FLOPs / 191e12` for dense FP16/BF16 WMMA.
- FP16 ridge point is about **298 FLOP/byte**. For comparison, the tabulated MI300X and MI350/355
  ridges are about 247 and 312 FLOP/byte respectively; moving from Instinct to R9700 does not shift
  the balance point in one universal direction.

Report achieved % of **that** ceiling. Size the grid to the **detected** CU/WGP count, not 304.

## 5. Profiling caveats (RDNA4 client)

- `rocprofv3 --kernel-trace` usually captures dispatches. **PMC / SoL counters are not CDNA**:
  `SQ_WAVES`, `VALUInsts`, `MfmaUtil`, `VALUBusy` may be absent or differently named.
- `rocprof-compute` / `omniperf` use CDNA-oriented SoC modules and may abort with `Unsupported arch`
  on gfx1201. Use `rocprofv3` first; only fall through to another tool when it emits real dispatch
  artifacts.
- Before trusting a PMC-derived bound class, run `rocprofv3-avail list --pmc`
  on THIS device. In rocprofv3 itself, use `rocprofv3 -L` /
  `rocprofv3 --list-avail`; older releases called the listing operation
  `--list-basic`, `--list-derived`, or `--list-counters`. This is an option
  rename across profiler generations, not an R9700-image defect.
- If discriminating counters are missing: **do not fail the profile phase**. Classify from
  kernel-trace durations + per-case latency + dispatch count + analytical roofline. Never fabricate
  MFMA%.
- AITER / Navi: several Instinct custom kernels are disabled or fall back to torch/hipBLASLt on
  gfx12 client (`gfx1201`). Do not treat an AITER CDNA win card as a must-win on this box — bake off hipBLASLt / Triton.

Learned Instinct cards (`platforms: ['gfx950']` / `gfx942`) do **not** transfer. Cold-start; new
wins must be tagged `platforms: ['gfx1201']`.

## Critical Rules

1. **NEVER** set `HIP_VISIBLE_DEVICES` inline with profiler commands — always go through `gpu_lock.sh`.
2. Wavefront-level ops operate on **32** threads, not 64.
3. Matrix math is **WMMA**, not MFMA. No MX / scaled MFMA / TDM; use the
   static VGPR occupancy model unless a HIP/Triton path proves dynamic allocation.
4. fp8 is **OCP**, never FNUZ. No MXFP4/6/8.
5. Size the grid to the DETECTED CU/WGP count, not a hard-coded Instinct 304.
6. Missing RDNA4 PMCs are a profiling limitation, not a failed run.
7. If `rocminfo` says `gfx942` / `gfx950`, this file does not apply — use `amd_instinct.md`.
