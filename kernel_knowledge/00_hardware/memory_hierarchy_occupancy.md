# MI300X Memory Hierarchy & Occupancy (CDNA3 / gfx942)

> How registers (VGPR/AGPR), LDS, caches, and HBM constrain a kernel, plus the exact occupancy math and the load/store patterns that keep MI300X fed.
> Companion to `mi300x_cdna3_arch.md` (topology) and `matrix_cores_numerics.md` (MFMA).

---

## 0. The memory ladder, with the numbers that drive decisions

| Level | Capacity | Scope | Approx latency | Bandwidth | Allocation unit |
|---|---|---|---|---|---|
| VGPR (architected) | **512 × 4 B per SIMD** | per wave | register | — | 16 regs |
| AGPR (accumulation) | up to **256 × 4 B per SIMD** | per wave | register (MFMA path) | — | — |
| LDS | **64 KiB / CU** (32 banks × 4 B) | per workgroup | ~20–30 cyc | highest of any GPU tested | 512 B (granularity varies) |
| L1 vector cache (TCP) | **32 KiB / CU**, 128 B line | per CU | tens of cyc | tens of TB/s | — |
| L2 cache | **4 MiB / XCD** (16×256 KiB) | per XCD | — | XCD-local | — |
| Infinity Cache (MALL/L3) | **256 MiB** device-wide | device | **~218 ns** | **~11.9 TB/s** | — |
| HBM3 | **192 GB** | device | +~47 ns TLB miss | **5.3 TB/s** | 4 KiB page |

Cache line = **128 B**. Page = **4 KiB** (use 2 MiB huge pages for >64 MB working sets to extend TLB reach, ~16384 entries/XCD).

---

## 1. The register file: VGPR + AGPR

### 1.1 Physical layout
- The CU has a **512 KiB combined** vector register file, split across its 4 SIMDs.
- Per SIMD/EU a wave can use up to:
  - **512 architected VGPRs** (`v0..v511`, 32-bit each) — usable by all VALU instructions, and
  - **256 AGPRs** (`acc0..acc255`) — accessible **only** to MFMA and the `v_accvgpr_read/write_b32` move instructions.
- AGPRs let matrix kernels hold large FP32 accumulators **without** consuming the architected VGPR budget that limits occupancy. They double effective register storage for matmul.

> Terminology trap: "512" appears twice. **512 VGPRs/EU** = the count used in occupancy math. **512 KiB/CU** = total physical capacity across 4 SIMDs (architected + AGPR). Don't conflate them.

### 1.2 Allocation granularity
VGPRs are allocated in **blocks of 16**. A kernel reporting 170 VGPRs is rounded to **176**. This rounding alone can drop you an occupancy tier — watch boundaries at 64/80/96/128/168/256.

### 1.3 Moving data VGPR↔AGPR
```asm
v_accvgpr_write_b32  acc0, v4     ; VGPR -> AGPR
v_accvgpr_read_b32   v4,  acc0    ; AGPR -> VGPR  (e.g. epilogue, before global_store)
```
The compiler also uses these for cheap spill/fill on CDNA. In a GEMM, keep the C/D accumulator tiles in AGPRs during the K-loop, then `v_accvgpr_read` into VGPRs for the store epilogue (costs ~5%).

### 1.4 Compiler knobs
| Knob | Effect |
|---|---|
| `__launch_bounds__(threads, waves_per_eu)` / `waves_per_eu=N` | tells LLVM to cap VGPRs so N waves fit per EU |
| `amdgpu-mfma-vgpr-form=false` + `amdgpu-agpr-alloc=256` | force accumulators into AGPRs ("AGPR escape hatch") |
| `-Rpass-analysis=kernel-resource-usage` / `.vgpr_count` in ISA dump | read actual VGPR/AGPR/LDS usage |

---

## 2. Occupancy math (the exact procedure)

Occupancy = number of resident waves per SIMD (max **8**) or workgroups per CU. It is the **minimum** of the VGPR limit, the LDS limit, and the hard wave-slot cap.

### 2.1 Inputs
1. **N** = VGPRs per wave (from ISA `.vgpr_count`, rounded up to a multiple of 16).
2. **L** = LDS bytes allocated per workgroup (e.g. `triton_gpu.shared`, or `__shared__` size).
3. **nW** = waves per workgroup = `ceil(threads_per_block / 64)`.

### 2.2 Limits
```
occ_vgpr (waves/SIMD)   = floor(512 / N)              # cap at 8
occ_lds  (workgroups/CU)= floor(65536 / L)            # MI300X LDS = 64 KiB
                                                       # (MI350X/CDNA4 = 160 KiB -> 163840)
wave_slots              = 8 waves/SIMD  = 32 waves/CU  # hard cap
```

### 2.3 Combine (workgroups per CU)
```
wg_from_vgpr = floor( (occ_vgpr * 4) / nW )   # 4 SIMDs per CU
wg_per_CU    = min( wg_from_vgpr, occ_lds, floor(32 / nW) )
waves_per_CU = wg_per_CU * nW                  # final occupancy
```

### 2.4 Worked examples

**Example A — VGPR-limited GEMM.** N=176 VGPR, threads=256 (nW=4), L=32 KiB.
```
occ_vgpr     = floor(512/176)   = 2 waves/SIMD
wg_from_vgpr = floor(2*4 / 4)   = 2 workgroups/CU
occ_lds      = floor(65536/32768)= 2
wg_per_CU    = min(2, 2, 8)     = 2  -> waves_per_CU = 8
```
Drop N to 128 → `occ_vgpr=4`, `wg_from_vgpr=4`, but `occ_lds=2` now binds → still 2 wg/CU. LDS is now the limiter; reduce tile or LDS to gain.

**Example B — LDS-limited softmax/attention.** N=64, threads=512 (nW=8), L=49152 (48 KiB).
```
occ_vgpr     = floor(512/64) = 8
wg_from_vgpr = floor(8*4 / 8) = 4
occ_lds      = floor(65536/49152) = 1   <-- binds
wg_per_CU    = min(4, 1, 4) = 1 -> waves_per_CU = 8
```
Halve LDS (double-buffer smaller, or pack) → `occ_lds=2` → 2 wg/CU → 16 waves/CU.

**Example C — fully occupied bandwidth kernel.** N=48, threads=256 (nW=4), L=8 KiB.
```
occ_vgpr=floor(512/48)=10 -> cap 8; wg_from_vgpr=floor(8*4/4)=8
occ_lds =floor(65536/8192)=8; wave-slot cap floor(32/4)=8
wg_per_CU = min(8,8,8)=8 -> 32 waves/CU (max).
```

> Rule of thumb: **HBM-bound** kernels want ≥4 waves/CU to hide ~218 ns L3 / HBM latency. **MFMA-bound** GEMM often runs 1–2 wg/CU and hides latency with double-buffered LDS + many in-flight loads instead of raw occupancy.

---

## 3. LDS: banks and conflict avoidance

- **64 KiB/CU**, organized as **32 banks of 4 bytes**. Bank = `(byte_address / 4) mod 32`.
- A wave issues LDS in **half-waves of 32 lanes**. A **bank conflict** occurs when ≥2 lanes in the same half-wave hit the **same bank** at **different addresses** (same address = broadcast, no conflict). N-way conflict serializes into N cycles.

### 3.1 The classic conflict: column access of a 32-wide tile
A `float tile[32][32]` accessed as `tile[k][tid]` (row stride 32 = exactly 32 banks) makes every lane in a column map to the **same bank** → 32-way conflict.

**Fix: pad the leading dimension** so the stride is coprime with 32:
```cpp
// BAD: stride 32 -> column access is 32-way conflict
__shared__ float tile[32][32];

// GOOD: pad to 33 -> column lanes spread across all 32 banks
__shared__ float tile[32][33];      // +1 column of padding
float v = tile[k][threadIdx.x];     // conflict-free
```
For 16-bit data, pad so the 32-bank pattern is broken (e.g. `[32][32+4]` for half, or use swizzled layouts). For 128-bit (`_b128`) LDS accesses the conflict granularity follows the 4-byte bank rule across the wider transaction — prefer **vectorized `ds_read_b128`/`ds_write_b128`** to cut instruction count.

### 3.2 Use wide LDS ops
```asm
ds_write_b128 v[0:3], v_data      ; one 16-byte store instead of 4 dwords
ds_read_b128  v_dst, v[0:3]
```
Fewer LDS instructions = less issue pressure and fewer chances to conflict.

### 3.3 MFMA + LDS swizzle
For matmul, stage A/B tiles in LDS with a **swizzled layout** matching the MFMA register/lane mapping so the `ds_read` that feeds MFMA is conflict-free. Composable Kernel and Triton generate these layouts automatically; hand-written kernels should mirror the lane mapping from `matrix_cores_numerics.md`.

---

## 4. Caches: what to exploit, what to avoid

| Cache | Behavior | Optimization |
|---|---|---|
| L1 (TCP, 32 KiB/CU) | write-through, 128 B line, per-CU; ~12-entry VMEM queue | coalesce to 128 B; reuse within a workgroup; don't expect cross-workgroup reuse |
| L2 (4 MiB/XCD) | coalesces all XCD memory traffic; **per-chiplet** | keep a workgroup's working set XCD-local; cross-XCD reuse misses to L3 |
| Infinity Cache (256 MiB) | device-shared L3, ~218 ns, ~11.9 TB/s; coherence point | size hot read-only data (weights, KV blocks) to live here; it absorbs cross-XCD sharing |

- There is **no global L2** — the device-wide cache is the 256 MiB L3. For LLM inference, the 256 MiB MALL can hold meaningful chunks of activations/KV; structuring access for L3 residency cuts HBM traffic.
- L2/L3 cache control via buffer-instruction flags (`glc`/`slc`/`dlc`) lets you bypass or stream around caches for write-once data.

---

## 5. Global loads/stores: coalescing & the load instruction families

### 5.1 Coalescing rules
- Coalesce so the 64 lanes of a wave touch a **contiguous, 128-byte-aligned** region. Best case: each lane reads a 32-bit element and the wave covers 256 B (two cache lines) with `global_load_dwordx?`.
- **Vectorize**: emit `global_load_dwordx4` (16 B/lane) wherever the access pattern allows — fewer instructions, fuller cache lines, especially inside loops.

```cpp
// Encourage dwordx4: use 128-bit vector loads
float4 v = *reinterpret_cast<const float4*>(ptr + 4*tid);   // -> global_load_dwordx4
```

### 5.2 The three vector-memory paths
| Path | ISA | When |
|---|---|---|
| **global_load / global_store** | `global_load_dwordx4`, … | flat 64-bit addressing; default for HIP pointer loads |
| **buffer_load / buffer_store** | `buffer_load_dwordx4` + descriptor (V#) | bounds-checked, supports `glc/slc`, hardware OOB handling → cheaper guards in tiled GEMM |
| **ds_read / ds_write** | `ds_read_b128`, … | LDS access |

### 5.3 Async/direct global→LDS (the big win)
CDNA3 supports **direct global-to-LDS** copies that bypass VGPRs:
```asm
buffer_load_dwordx4 ... lds        ; load straight into LDS, no VGPR staging
global_load_lds_dwordx4 ...        ; flat variant
```
- In Triton/Gluon this is `buffer_load_to_lds`. Using it instead of register-staged copies **saved ~100 VGPR/wave** and moved a reference GEMM from **697 → 1113 TFLOP/s** because freed VGPRs raised occupancy and removed the round-trip.
- Pair with **double buffering**: while MFMA consumes LDS buffer 0, issue async loads filling LDS buffer 1.

```cpp
// Double-buffered K-loop skeleton (conceptual)
load_global_to_lds(buf[0], A_k0, B_k0);     // prologue
for (int k = 1; k < K_tiles; ++k) {
    load_global_to_lds(buf[k & 1], A_k, B_k);   // prefetch next while...
    s_waitcnt(prev);                            // ...consuming previous
    mfma_accumulate(acc, buf[(k-1) & 1]);
}
mfma_accumulate(acc, buf[(K_tiles-1) & 1]);     // epilogue
```
Synchronization uses **`s_waitcnt vmcnt/lgkmcnt`** (count-based, not a fence) — wait only for the specific outstanding loads you need, enabling deep overlap.

---

## 6. Putting it together: an occupancy/bandwidth tuning checklist

| Symptom (from Omniperf) | Likely cause | Fix |
|---|---|---|
| Low waves/CU, high VGPR | register pressure | shrink tile, `waves_per_eu`, AGPR escape hatch, `buffer_load_to_lds` |
| Low waves/CU, high LDS | LDS pressure | smaller tiles, less buffering, pack data, NPS-aware sizing |
| LDS bank-conflict stalls | column access / stride mult. of 32 | pad leading dim (+1/+4), swizzle, `ds_*_b128` |
| Low HBM BW utilization | uncoalesced / scalar loads | `global_load_dwordx4`, 128 B alignment, restructure access |
| Memory-latency-bound | too few waves | raise occupancy to ≥4 waves/CU; prefetch |
| Cross-XCD stalls / atomics slow | data shared across chiplets | keep working set XCD-local; use L3; consider CPX/NPS4 |
| MFMA underfed | LDS read conflicts / no double-buffer | swizzled LDS layout, async copy + double buffer |

**Targets for a tuned FP16/FP8 GEMM (MI300X):** 256×256 tile, 2-stage prefetch, **384–448 VGPR** budget, ≥4 waves/CU, `mfma_16x16` instructions, `buffer_load_to_lds`, conflict-free swizzled LDS.

---

## Sources
1. AMD Instinct MI300X workload optimization (occupancy, VGPR=512/EU, waves_per_eu, occ.sh) — ROCm Documentation: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
2. Optimizing Triton kernels on MI300X (VGPR rounding, LDS padding, occupancy) — ROCm Documentation: https://rocm.docs.amd.com/en/docs-6.1.0/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
3. HIP Hardware implementation (VGPR/AGPR, SIMD, wave slots) — ROCm Documentation: https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
4. AMD Instinct MI Series Accelerator Performance Model — rocprofiler-compute (Omniperf): https://rocm.github.io/rocprofiler-compute/performance_model.html
5. "Testing AMD's Giant MI300X" — Chips and Cheese (measured cache/LDS/HBM latency & bandwidth): https://chipsandcheese.com/p/testing-amds-giant-mi300x
6. Register/occupancy limits on AMD GPUs (HLRS training slides): https://fs.hlrs.de/projects/par/events/2024/GPU-AMD/day1/register_occupancy_limit.pdf
7. AMD CDNA™ 3 Architecture White Paper (LDS/L1/L2/Infinity Cache sizes): https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf
8. AMD Instinct MI300 (CDNA3) ISA Reference Guide (buffer/global/ds instructions, s_waitcnt, v_accvgpr): https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
