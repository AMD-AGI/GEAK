# Memory & Occupancy Optimization — AMD Instinct MI300X (CDNA3 / gfx942)

> Scope: AMD only. Target MI300X (gfx942, CDNA3); gfx950 (CDNA4) notes inline. This file is the memory-hierarchy and occupancy playbook: how to feed the MFMA engines without stalling on HBM/LDS/register traffic. Every technique has a code pattern; a checklist closes the file.

---

## MI300X memory hierarchy (the numbers you tune against)

| Level | Size | Key facts |
|---|---|---|
| **Registers / SIMD** | 512 × 32-bit = **256 VGPR + 256 AGPR** (at 1 wave/SIMD) | VGPR allocated in **groups of 8 dwords** (effective granule ~16 for occupancy math). AGPRs feed MFMA + can be loaded direct from memory. |
| **LDS / CU** | **64 KiB** | 32 banks; **same speed as L1**; allocated in 512-byte aligned blocks. |
| **L1 data cache / CU** | **32 KiB** | **128 B cache line** (vs 16 KiB on MI200). |
| **L2 / XCD** | ~4 MB per XCD (~32 MB total) | XCD-private, coalesces requests, **2048 B/clk**. |
| **HBM3** | 192 GB | ~5.3 TB/s; shared across 8 XCDs. |
| CU count | **304** (8 XCD × 38 CU) | 2048 threads/CU; grids should produce **≥1024 workgroups**. |

The optimization chain: **coalesce global loads to 128 B lines → wide `buffer_load_dwordx4` → stage to LDS (VGPR+`ds_write`, or direct-to-LDS on CDNA4) → XOR-swizzle/pad to avoid bank conflicts → wide `ds_read_b128` to feed MFMA → tune VGPR/AGPR split & `waves_per_eu` for occupancy without spilling.**

---

## 1. Global memory coalescing (128 B cache line)

**Goal:** a wavefront (64 lanes) should touch contiguous addresses landing in the same 128 B cache line(s). Strided/scattered access multiplies HBM transactions.

- Lay out so consecutive lanes read consecutive elements (the contiguous dim of the operand).
- A 64-lane wave reading 4 bytes each = 256 B = exactly 2 cache lines — ideal.
- For row-major A in `C = A·B`, transpose tiling so the K-step reads are coalesced; for col-major B likewise. The common LLM `NT` weight layout (A row-major, B col-major) is coalesce-friendly.

```cpp
// Coalesced: lane i reads element i of a contiguous row tile → 128B-line-aligned bursts
float4 v = *reinterpret_cast<const float4*>(&A[row*K + k + lane*4]);  // 16B/lane, wave = 1KB
```

**Check:** `rocprof-compute` L1/L2 hit rate and "fetch size" per request; low hit + many small fetches = poor coalescing.

---

## 2. Vectorized loads (128-bit)

Use the widest load the alignment allows: `buffer_load_dwordx4` (128-bit) > `dwordx3` (96-bit) > `dwordx2` (64-bit). Wider loads = fewer instructions, fewer address calcs, better HBM burst efficiency.

```cpp
// 128-bit vectorized load (float4 / 8×fp16 / 16×fp8)
using v16b = __attribute__((ext_vector_type(4))) float;
v16b reg = *reinterpret_cast<const v16b*>(ptr);   // requires 16B alignment
```

- In Triton, load `BLOCK_K` such that per-lane load width is 128-bit (`tl.load` of contiguous `float4`-equivalent).
- HipKittens picks `buffer_load_dwordx4`/`dwordx3` by datatype to maximize throughput.
- **Requires 16 B alignment** of the base pointer + per-lane offset, else the compiler downgrades to narrow scalar loads (silent perf loss). Pad row strides to 16 B multiples.

---

## 3. LDS tiling & bank-conflict avoidance

LDS has **32 banks** (4 B wide). A bank conflict occurs when multiple lanes in a wave hit different addresses in the **same bank** in one cycle → serialized. The classic conflict: a tile of width = multiple of 32 elements where column access strides land all lanes on bank 0.

### Two fixes

**(a) Padding** — add 1+ element of padding per row so the stride is coprime with 32:
```cpp
__shared__ float tileA[BM][BK + 1];   // +1 pad column breaks the 32-bank alignment
```
Simple, costs a little LDS, doesn't touch the access math. Constraint: a direct-to-LDS DMA transfer must not cross pad boundaries.

**(b) XOR swizzle** (CUTLASS/CK style) — permute the column index with `col ^ (row & mask)` so both `ds_write` and `ds_read` are conflict-free:
```cpp
int swz_col = col ^ ((row >> s) & ((1<<b)-1));   // XOR swizzle for ds_read_b128 feeding MFMA
tileA[row][swz_col] = val;
```

**Why it matters (measured on MI300X):** in an IREE direct-to-LDS GEMM, *removing* the XOR swizzle introduced **201M LDS bank conflicts** (0 baseline) and a **−27.9% throughput regression (1822 vs 2527 TFLOPS)**. Bank-conflict avoidance is not optional for MFMA feeders.

**Read wide:** feed MFMA with `ds_read_b128` / `ds_read_b96` (HipKittens uses `ds_read_b128/b96`, `ds_write_b64`). Match the swizzle to the read width.

---

## 4. Register pressure & spilling control

Spills go to scratch (HBM-backed) — a single spill in the inner loop is catastrophic. Two levers:

### `__launch_bounds__` (HIP)
```cpp
// (maxThreadsPerBlock, minWavesPerEU) — caps registers so the wanted occupancy is reachable
__global__ void __launch_bounds__(256, 2) gemm(...) { ... }
```
Tells the compiler the max block size + min waves/EU; it then bounds register usage to fit. Too aggressive → spills; too loose → low occupancy.

### `waves_per_eu` (Triton kernarg / compiler hint)
The occupancy math: VGPR allocated in granules of **16**. Usage 170 → rounds to 176; `176 × 3 > 512` → only **2 waves/EU**. Setting `waves_per_eu=3` makes LLVM shrink VGPR usage to fit 3 waves. Sweep 1–4 and check spills.

```python
triton.Config({..., "waves_per_eu": 3}, num_warps=8, num_stages=0)
```

**AGPR vs VGPR:** MFMA accumulators live in AGPRs (up to 256), freeing VGPRs (up to 256) for operands/addresses — so a GEMM can use the full 512 file. Move data with `v_accvgpr_read/write`. Over-spilling into AGPR-as-scratch also hurts. **Check `rocprof-compute` for "VGPR spill" / "scratch" > 0 — that means redesign the tile.**

---

## 5. Double / triple buffering (multi-stage prefetch)

Overlap the load of tile *k+1* with the MFMA compute of tile *k* using ≥2 LDS buffers.

```text
load A0,B0 -> LDS[0]
for k in 0..K/BK:
    prefetch A[k+1],B[k+1] -> LDS[(k+1)%2]   # async/direct-to-LDS, overlaps compute
    mfma(LDS[k%2])                            # compute current
```

**MI300X tradeoff:** each extra buffer = more LDS (of 64 KiB) + more registers → lower occupancy. `num_stages` guidance (CDNA): single GEMM → 0; two fused GEMMs (FlashAttention) → 1. Over-staging spills. Double-buffer is the safe default; triple only if profiling shows load-bound stalls and occupancy still allows it.

---

## 6. Async / direct-to-LDS loads

- **CDNA3 (gfx942):** `buffer_load ... lds` exists for some paths; the common path is `buffer_load → VGPR → ds_write → LDS`. Minimize the VGPR staging footprint and keep `ds_write` wide + swizzled.
- **CDNA4 (gfx950):** `buffer_load ... lds` moves global→LDS in **one instruction, bypassing VGPRs entirely** — eliminates `ds_write`, staging VGPRs, and copy index math. **The XOR swizzle must still be preserved** because subsequent `ds_read` for MFMA needs conflict-free addresses.

Direct-to-LDS frees VGPRs (→ higher occupancy) and removes instructions from the load path — prefer it on gfx950, and use the IREE/HipKittens patterns to keep swizzle intact.

---

## 7. L2 reuse & cache-aware scheduling

L2 is **XCD-private**. The hardware assigns workgroups to XCDs **round-robin**, so the grid launch order determines which tiles share an XCD's L2.

- **Block swizzle (`GROUP_M`)**: reorder tile IDs so tiles sharing operands (same row band of A, or same col band of B) land near each other → L2 hits instead of HBM re-reads. Triton's grouped ordering (`GROUP_SIZE_M`) is exactly this.
- **Persistent kernels** keep weights resident across tiles → maximal L2/register reuse.
- For decode (weights re-read every step), maximizing L2 residency of hot weight tiles is the main bandwidth win.

```python
# L2-friendly tile ordering (GROUP_M swizzle)
num_pid_m = tl.cdiv(M, BM); num_pid_n = tl.cdiv(N, BN)
group_id = pid // (GROUP_M * num_pid_n)
pid_m = group_id * GROUP_M + (pid % GROUP_M)
pid_n = (pid % (GROUP_M * num_pid_n)) // GROUP_M
```

---

## 8. Occupancy tradeoffs

More waves/CU hides latency but each wave needs registers + LDS, and the file/LDS are fixed. The art is the **minimum occupancy that hides your latency**, not max occupancy.

| If profile shows... | Likely cause | Fix |
|---|---|---|
| low MFMA util, high mem stalls | load-bound, too few waves | raise `waves_per_eu`, more buffering, wider loads |
| VGPR/AGPR spills, scratch > 0 | over-staged / huge tile | shrink tile or `num_stages`, `__launch_bounds__` |
| LDS-limited occupancy | tiles too big for 64 KiB | smaller tile, fewer buffers, or fewer blocks/CU |
| bank conflicts in `ds_read` | unswizzled/unpadded LDS | XOR swizzle or pad (see §3) |
| poor L1/L2 hit | bad coalescing / tile order | 128B coalesce, GROUP_M swizzle |

**Concrete sizing example:** each CU handles 2048 threads, block max 1024. Launching **2 blocks/CU** gives each block ~32 KiB LDS → room for larger tiles while keeping 2-deep occupancy.

---

## Memory & occupancy checklist

```
GLOBAL
[ ] Loads coalesced to 128 B cache lines (consecutive lanes → consecutive addrs)
[ ] Widest vector load (buffer_load_dwordx4 / 128-bit); base + stride 16 B aligned
[ ] NT (or coalesce-friendly) operand layout

LDS
[ ] Tile fits in 64 KiB with chosen buffering depth
[ ] Bank conflicts eliminated: XOR swizzle or +1 padding (verify 0 conflicts in profiler)
[ ] Wide ds_read_b128 / ds_write_b64 feeding MFMA, swizzle matched to read width
[ ] Direct-to-LDS on gfx950 (keep swizzle); minimal VGPR staging on gfx942

REGISTERS / OCCUPANCY
[ ] No spills / scratch == 0 (rocprof-compute)
[ ] MFMA accumulators in AGPR; operands/addrs in VGPR (full 512 file usable)
[ ] __launch_bounds__ or waves_per_eu set; swept 1–4; occupancy is "enough to hide latency"
[ ] num_stages per CDNA rule (single GEMM 0, two-GEMM/FA 1)

L2 / SCHEDULING
[ ] GROUP_M block swizzle for L2 reuse
[ ] Persistent kernel for repeated-launch / weight-resident workloads
[ ] Grid ≥ 1024 workgroups (split-K/stream-K if skinny)

VERIFY
[ ] Re-profile: MFMA util up, mem stalls down, AI moved off HBM roof, 0 LDS conflicts, 0 spills
```

---

## Sources

- AMD Instinct MI300 CDNA3 ISA reference (VGPR/AGPR allocation, LDS 512 B blocks, v_accvgpr): <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf>
- AMD MI300X Hot Chips 2024 architecture (64 KiB LDS, 32 KiB L1 / 128 B line, L2 2048 B/clk): <https://hc2024.hotchips.org/assets/program/conference/day1/23_HC2024.AMD.MI300X.ASmith(MI300X).v1.Final.20240817.pdf>
- IREE direct-to-LDS for scaled GEMM (XOR swizzle: 201M conflicts / −27.9% TFLOPS without it): <https://github.com/iree-org/iree/issues/23765>
- AMD MI300X workload optimization (waves_per_eu, VGPR granule occupancy math, OPTIMIZE_EPILOGUE): <https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html>
- HipKittens (ds_read_b128/b96, ds_write_b64, buffer_load_dwordx4 by dtype, CDNA3/4 swizzle): <https://arxiv.org/html/2511.08083v1>
- rocprof-compute pipeline descriptions (VALU/MFMA, LDS, occupancy metrics): <https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/pipeline-descriptions.html>
- FP8 GEMM on MI300X — coalescing to 128 B line, LDS = L1 speed (GPU MODE AMD challenge writeup): <https://github.com/luongthecong123/fp8-quant-matmul>
