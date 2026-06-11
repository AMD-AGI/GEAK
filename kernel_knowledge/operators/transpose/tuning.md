---
title: transpose — tuning (LDS bank conflicts, padding vs XOR swizzle, vectorized BW)
kind: technique
operator: transpose
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8]
regimes: [both]
updated: 2026-06-08
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html
  - https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/hardware/lds_bank_conflicts.html
  - https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
---

# transpose — tuning

The entire tuning surface is **LDS bank-conflict avoidance** + **vectorized HBM/LDS access**. There is no
compute to schedule; the kernel is bandwidth-bound and the only way to lose is to serialize LDS.

## 1. The bank-conflict problem (why naive transpose is slow)
LDS on CDNA3 (gfx942) = **32 banks × 4 B**, 128 B/clk; CDNA4 (gfx950) = **64 banks × 4 B**, 256 B/clk.
Bank index = `(byte_addr / 4) mod (#banks)`. A wavefront issues for 64 lanes but accesses are serviced in
**phases** (e.g. `ds_read_b64` → 4 phases of 16 lanes; `ds_read_b128` similar). Within a phase, lanes
hitting the **same bank, different row** serialize.

Naive transpose:
```cpp
__shared__ float tile[32][32];          // BAD: stride 32 = multiple of 32 banks
tile[ty][tx] = A[...];   __syncthreads();
out[...]     = tile[tx][ty];            // column read: all 32 lanes → SAME bank → 32-way conflict
```
The column read maps every lane of the phase to one bank. AMD measures the typical ML LDS read pattern
(vertical, for MFMA staging) as a **4-way conflict in every phase → effective LDS bandwidth cut by 75%**.

## 2. Fix A — padding (`+1`)
```cpp
__shared__ float tile[32][32 + 1];      // pad inner dim by 1 element
```
`tile[i][j]` now lands at `33*i + j` → bank `(33*i + j) mod 32 = (i + j) mod 32`; for fixed `j` (a phase),
varying `i` walks all 32 banks → **conflict-free**. Cost: **12.5–25% extra LDS** (a 64×64 fp16 tile grows
from 8 KB to ~8.25 KB) — significant on the tight 64 KB/CU CDNA3 budget, and **not guaranteed**
conflict-free for every access width / element size (re-profile).

## 3. Fix B — XOR swizzle (preferred on AMD)
Permute the column index instead of padding: `col' = col XOR f(row)` (typically `row & (banks-1)`), so the
write and the transposed read both walk distinct banks **with zero extra LDS**. AMD's explicit
recommendation: *prefer XOR preshuffle over padding (no storage overhead), verify with rocprof.* CK-Tile's
`TileWindow` applies this automatically; in HIP you compute the swizzled offset by hand (see
[[languages/hip_cpp/lds_async.md]] §2). The same swizzle is **required** for direct-to-LDS loads —
removing it in one IREE study caused **201M bank conflicts, −28% TFLOPS**.

| mitigation | extra LDS | conflict-free | effort |
|---|---|---|---|
| none (naive) | 0 | ❌ (4–32 way) | — |
| `+1` padding | +12.5–25% | usually (re-check) | trivial |
| XOR swizzle | **0** | ✅ both R/W | moderate (or free via CK-Tile) |

## 4. Vectorize both ends (128-bit)
- **HBM**: contiguous-row loads/stores must emit `global_load_dwordx4`/`global_store_dwordx4` (`float4`,
  `__restrict__`, 16-B alignment). 4-B per-lane access wastes ~half the bus.
- **LDS**: `ds_read_b128`/`ds_write_b128` (read/write `float4` from LDS) — 16-B LDS access sustains ~80%
  peak (≈20 cyc/64 lanes) vs ~50% for 4-B. Always stage and read the tile in 128-bit chunks.

## 5. CDNA4 hardware transpose (gfx950)
CDNA4 adds **read-with-transpose** DS instructions (`ds_read_b64_tr_b16`, etc.): the LDS crossbar
transposes 16-bit elements **on read**, so you write the tile straight and read it transposed with no
swizzle and no second `__syncthreads`. Lowering is **gfx950+ only** (`amdgpu.ds_read_tr` / ROCDL). On
gfx942 you still use padding or XOR swizzle.

## 6. Grid / occupancy
Aim for **≥1024 workgroups** across 304 CUs; block = multiple of 64 (256 = 4 waves is a good default).
Keep LDS small enough that ≥2 workgroups fit per CU (budget against 64 KB CDNA3 / 160 KB CDNA4).

## Verify
`rocprof-compute`/rocprofv3 → LDS bank-conflict counter ≈ 0 and the kernel is HBM-bound (achieved BW near
`2·bytes/time` at ~5.3 TB/s). ISA (`AMDGCN_ENABLE_DUMP`/`--save-temps`): want `ds_*_b128`,
`global_*_dwordx4`, and on gfx950 `ds_read_*_tr_*`.

## Sources
- 4-way conflict = −75% BW, XOR swizzle preferred over padding, rocprof verify: https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html
- Bank model (32/64 banks, access phases, bank formula): https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/hardware/lds_bank_conflicts.html · https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
- 128-bit LDS/HBM, 50%/80% rule, 201M conflicts/−28%: [[languages/hip_cpp/lds_async.md]] (IREE iree#23765).
- CDNA4 ds_read_tr: AMD CDNA4 ISA reference (DS section) + https://mlir.llvm.org/docs/Dialects/AMDGPU/.
