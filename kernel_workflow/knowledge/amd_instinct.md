# AMD Instinct (MI-series) Hardware Reference — DETECT THE BOX FIRST

This workflow runs on AMD Instinct MI-series accelerators — **CDNA 3** (MI300X / MI300A / MI308X /
MI325X, `gfx942`) and **CDNA 4** (MI350X / MI355X, `gfx950`). They differ in CU count, HBM bandwidth,
peak FLOPS, and — critically for quantized kernels — the **fp8 number format**. Do NOT assume MI300X.

## 0. Detect THIS box first (source of truth > this table)
Always identify the actual accelerator at the start of analysis/profiling, and prefer the detected
values + your measured benchmark over any number written here (this table is a dated hint, like all
reference material in this workflow):

```bash
rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+'          # arch id: gfx942 (CDNA3) | gfx950 (CDNA4)
rocminfo 2>/dev/null | grep -m1 -iE 'Compute Unit'          # CU count on THIS device
rocm-smi --showproductname 2>/dev/null | grep -iE 'MI3'     # marketing name (MI300X/325X/350X/355X/...)
rocm-smi --showmeminfo vram 2>/dev/null | head              # HBM capacity
```
- The `gfx` id is what matters for code paths (fp8 format, MFMA shapes, MX support). The CU count is
  what matters for grid sizing / occupancy. Take BOTH from `rocminfo`, not from the card name.
- For the roofline ceiling, prefer an **empirically achievable** HBM bandwidth (a memory-bound kernel
  typically reaches ~0.7–0.85× nameplate) over the nameplate peak below. When in doubt, MEASURE.

## 1. Card comparison (reference hint — verify on-box)
| Card    | Arch / gfx        | CUs (≈) | HBM cap | HBM BW (≈) | fp8 format | MX (fp4/fp6) |
|---------|-------------------|---------|---------|------------|------------|--------------|
| MI300X  | CDNA3 / `gfx942`  | 304     | 192 GB  | 5.3 TB/s   | **FNUZ**   | no           |
| MI300A  | CDNA3 / `gfx942`  | 228     | 128 GB  | 5.3 TB/s   | **FNUZ**   | no           |
| MI308X  | CDNA3 / `gfx942`  | reduced | 192 GB  | ~5.3 TB/s  | **FNUZ**   | no           |
| MI325X  | CDNA3 / `gfx942`  | 304     | 256 GB  | ~6.0 TB/s  | **FNUZ**   | no           |
| MI350X  | CDNA4 / `gfx950`  | 256     | 288 GB  | ~8 TB/s    | **OCP**    | **yes**      |
| MI355X  | CDNA4 / `gfx950`  | 256     | 288 GB  | ~8 TB/s    | **OCP**    | **yes**      |

CU counts/BW are nameplate and vary by SKU/firmware (MI308X is a reduced-CU variant) — `rocminfo` is
authoritative. CDNA4 (gfx950) is a large generational step up in matrix throughput over CDNA3 and adds
native FP6/FP4 — do NOT carry MI300X compute peaks onto it; look it up or measure.

## 2. CDNA fundamentals (common across gfx942 & gfx950)
- **Wavefront size**: 64 threads (NOT 32 like an NVIDIA warp). `__shfl_xor`/`__ballot`/`__any`/`__all`
  operate over 64 lanes; `__ballot` returns a 64-bit mask.
- **Registers**: up to 256 VGPRs/thread (512 with VGPR pairs), ~106 SGPRs/thread.
- **LDS**: 64 KB per CU, 32 banks × 4 bytes/cycle. Stride 4 B → 32-way conflict; pad to avoid it.
- **L1**: 32 KB/CU. **L2**: large, shared across CUs (256 MB on MI300X-class; varies).
- **Global memory coalescing granularity**: 64 bytes (one cache line).
- **Launch**: max 1024 threads/block; block sizes multiples of 64 (64/128/256 typical).

### Occupancy vs VGPRs/thread (CDNA, 4 SIMDs/CU, max 8 waves/SIMD)
| VGPRs/thread | Max Waves/SIMD | Occupancy |
|-------------|----------------|-----------|
| 24          | 8              | 100%      |
| 28-32       | 7              | 87.5%     |
| 36          | 6              | 75%       |
| 40-48       | 5              | 62.5%     |
| 56-64       | 4              | 50%       |
| 84          | 3              | 37.5%     |
| 128         | 2              | 25%       |
| 256         | 1              | 12.5%     |

Prefer `__launch_bounds__(max_threads, min_waves)` to steer register allocation.

## 3. Arch-specific: dtype, fp8 format, MFMA (gfx942 vs gfx950)
**This is the part you MUST branch on `gfx`** — picking the wrong fp8 format silently fails correctness.
- **gfx942 (CDNA3)** — fp8 is **FNUZ**: `torch.float8_e4m3fnuz` / `torch.float8_e5m2fnuz` (note the
  different bias/range vs OCP). No native MX (fp4/fp6). MFMA tile shapes: 4x4x4, 16x16x16, 32x32x8
  (also 16x16x32 / 32x32x16 for 8-bit). Prefer `matrix_instr_nonkdim=16` for triton GEMM on gfx942.
- **gfx950 (CDNA4)** — fp8 is **OCP**: `torch.float8_e4m3fn` / `torch.float8_e5m2` (standard OCP), and
  it adds native **MXFP4 / MXFP6 / MXFP8** (block-scaled) matrix ops — a major lever for low-precision
  GEMM that does NOT exist on gfx942. New/wider MFMA variants; consult the perf_knowledge cards
  (`quantization/fnuz_vs_ocp.md`, `optimization/mfma_scheduling.md`) when authoring quantized kernels.
- Always match the dtype/tolerance the unittest/oracle encodes; fix the math, never loosen tolerance.

## 4. Peak FLOPS (MI300X concrete; others: detect/measure)
MI300X (gfx942), dense MFMA, as a reference anchor for roofline math:
- FP32 (vector): ~163 TFLOPS · FP16/BF16 (MFMA): ~1.3 PFLOPS · FP8 (MFMA): ~2.6 PFLOPS · INT8: ~2.6 POPS.
MI325X ≈ MI300X compute (same gfx942 core; more/faster HBM). CDNA4 (MI350X/MI355X) is materially higher
and adds FP6/FP4 — treat the MI300X numbers as a LOWER bound there and look up / measure the real peak.
For a roofline estimate: memory-bound `min_time ≈ bytes_moved / achievable_HBM_BW`; compute-bound
`min_time ≈ FLOPs / peak_FLOPS_for_dtype`. Report achieved % per case; measurement is the final word.

## Critical Rules
1. **NEVER** set `HIP_VISIBLE_DEVICES` inline with profiler commands — always go through `gpu_lock.sh`.
2. Branch quantized code on the detected `gfx` (FNUZ on gfx942, OCP + MX on gfx950) — never hard-code one.
3. Wavefront-level ops operate on 64 threads, not 32. `__syncthreads()` is block-level only.
4. Memory coalescing granularity is 64 bytes; LDS has 32 banks (pad to avoid 32-way conflicts).
5. Size the grid to the DETECTED CU count (`rocminfo`), not a hard-coded 304.
6. Prefer `__launch_bounds__(max_threads, min_waves)` to help the compiler with register allocation.
