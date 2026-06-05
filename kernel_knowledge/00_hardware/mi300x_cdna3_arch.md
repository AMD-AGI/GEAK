# MI300X / CDNA3 (gfx942) Architecture for Kernel Authors

> Target: **AMD Instinct MI300X**, CDNA3, ISA name **gfx942** (also gfx940/gfx941 for MI300A / early steppings; MI325X is also gfx942).
> Successors noted where relevant: **MI325X** (CDNA3, gfx942, 256 GB HBM3e), **MI355X / MI350X** (CDNA4, **gfx950**).
> This file is the orientation map. Memory/occupancy details live in `memory_hierarchy_occupancy.md`; matrix cores in `matrix_cores_numerics.md`.

---

## 0. The one-screen cheat sheet (what a kernel author MUST know)

| Fact | Value | Why it matters for your kernel |
|---|---|---|
| Wavefront size | **64 lanes** (wave64) | All thread-divergence, shuffle, ballot, MFMA layout math is mod 64, not 32. |
| CUs (active) | **304** (8 XCD × 38) | Grid should have **≥ 1024 workgroups** to fill the device + tail. |
| XCDs (chiplets) | **8** per package | The GPU is 8 GPUs glued by Infinity Fabric/Cache. Locality across XCD is NOT free. |
| CUs per XCD | 40 physical / **38 active** | A workgroup lives entirely on one CU on one XCD. |
| SIMDs per CU | **4** (SIMD64) | Occupancy is per-SIMD (per-EU). 1 workgroup spreads its waves across the 4 SIMDs. |
| Wave slots | **8 waves / SIMD → 32 waves / CU** | Hard cap on occupancy regardless of register/LDS room. |
| Peak engine clock | **2100 MHz** (base 1000) | Used in all peak-FLOP/bandwidth math. |
| VGPRs | **512 × 32-bit per SIMD/EU**, alloc granularity 16 | Register pressure is the #1 occupancy killer. |
| AGPRs | up to **256** extra (MFMA accumulators) | Lets matmul keep big accumulators without crushing occupancy. |
| LDS | **64 KiB per CU** (32 banks) | Shared-memory tiling budget; bank conflicts are mod-32. |
| L1 vector cache | **32 KiB per CU**, 128 B line | Coalesce to 128 B; reuse within a workgroup. |
| L2 cache | **4 MiB per XCD** (not global!) | L2 is per-chiplet. Cross-XCD reuse falls through to L3. |
| Infinity Cache (L3/MALL) | **256 MiB**, device-shared, ~218 ns, ~11.9 TB/s | The only large shared cache; the cross-XCD coherence point. |
| HBM3 | **192 GB**, **5.3 TB/s** | Most LLM-inference kernels are HBM-bound → optimize bytes, not FLOPs. |
| Matrix throughput | FP16/BF16 **1307 TFLOP/s**, FP8/INT8 **2615 TFLOP/s** | MFMA is mandatory for any GEMM/attention to be competitive. |
| Partition default | **SPX + NPS1** (one logical GPU) | Unless told otherwise, assume the whole 304-CU / 192 GB device. |

---

## 1. Package topology: the GPU is a chiplet system

The MI300X is **not** a monolithic die. It is a **3.5D** stack: 8 compute chiplets (XCD) hybrid-bonded on top of 4 I/O dies (IOD), surrounded by 8 HBM3 stacks.

```
                MI300X package (153 B transistors, 750 W)
   ┌──────────────────────────────────────────────────────────┐
   │  HBM3   HBM3        HBM3   HBM3                            │
   │   ┌───────────┐      ┌───────────┐                        │
   │   │ XCD0 XCD1 │      │ XCD2 XCD3 │   each XCD: 38 active   │
   │   │  (on IOD0)│      │  (on IOD1)│   CUs + 4 MiB L2        │
   │   ├───────────┤      ├───────────┤                        │
   │   │   IOD0    │◄────►│   IOD1    │  ◄─ Infinity Fabric     │
   │   └───────────┘      └───────────┘     (on-package,       │
   │   ┌───────────┐      ┌───────────┐      ~4.8 TB/s         │
   │   │   IOD2    │◄────►│   IOD3    │      bisection)         │
   │   │ XCD4 XCD5 │      │ XCD6 XCD7 │                         │
   │   └───────────┘      └───────────┘                        │
   │  HBM3   HBM3        HBM3   HBM3      256 MiB Infinity      │
   │                                     Cache spans all IODs  │
   └──────────────────────────────────────────────────────────┘
```

| Die | Count | Node | Contents |
|---|---|---|---|
| XCD (Accelerator Complex Die) | 8 | TSMC N5 | 40 CUs (38 active), 4 ACEs, 4 MiB L2, HW scheduler |
| IOD (I/O Die) | 4 | TSMC N6 | HBM3 memory controllers, Infinity Cache slice, Infinity Fabric, PCIe Gen5, XGMI |
| HBM3 stack | 8 | — | 24 GB each (8-Hi), ~662 GB/s each → 192 GB / 5.3 TB/s aggregate |

**Key consequences for kernels:**
- Each IOD hosts **2 XCDs** and **2 HBM stacks**. Memory closest to your XCD lives on the same IOD.
- L2 is **per-XCD** (4 MiB). There is **no device-wide L2**. The first device-shared level is the **256 MiB Infinity Cache (MALL)** on the IODs.
- Inter-XCD data sharing / atomics go through Infinity Fabric + Infinity Cache: measured global-atomic core-to-core latency ranges **~116–202 ns** depending on whether the two workgroups land on the same or different XCD.

---

## 2. XCD internals

```
              ┌──────────────── XCD (one chiplet) ────────────────┐
              │  HWS (hardware scheduler)                          │
              │  4 × ACE (Asynchronous Compute Engine)            │
              │      └─ dispatch workgroups to CUs                 │
              │  ┌──────────────────────────────────────────────┐ │
              │  │ 40 CUs (38 active)  — each:                   │ │
              │  │   4 × SIMD64  +  4 Matrix Cores              │ │
              │  │   64 KiB LDS, 32 KiB L1, 16 KiB scalar/const │ │
              │  └──────────────────────────────────────────────┘ │
              │  4 MiB shared L2 (16 × 256 KiB slices)             │
              └────────────────────────────────────────────────────┘
```

- **ACEs** are the queue front-ends. Up to 4 ACEs per XCD dispatch independent compute streams → multiple HIP streams / concurrent kernels map naturally to ACEs.
- A workgroup is dispatched to **one CU** and never migrates. Its waves are striped across that CU's 4 SIMDs.

---

## 3. Compute Unit (CU) model

Each CDNA3 CU contains:

| Block | Spec |
|---|---|
| SIMD units | 4 × SIMD64 (64-wide vector ALU pipelines) |
| Vector ALUs | 64 SP lanes per SIMD → 256 FMA lanes / CU / cycle |
| Matrix cores | 4 (one per SIMD); MFMA engine |
| VGPR file | 512 × 32-bit per SIMD (128 KiB/CU architected) |
| AGPR file | up to 256 × 32-bit per SIMD (accumulation, MFMA-only) |
| Scalar regs | ~3.2 KiB / CU |
| LDS | 64 KiB / CU, 32 banks |
| L1 vector cache | 32 KiB / CU, 128-byte line |
| L1 instruction cache | 64 KiB (shared per pair of CUs) |
| L1 constant/scalar cache | 16 KiB |
| Wave slots | 8 per SIMD → 32 per CU |

### 3.1 Execution model (wave64)
- A **wavefront = 64 work-items** executing in lockstep on one SIMD64. Issue is over 4 cycles (16 lanes physically × 4) but the programming model is "64 lanes, one instruction."
- Branch divergence is handled with the 64-bit `EXEC` mask. Cross-lane ops (`ds_swizzle`, `v_permlane`, DPP, `__shfl`) operate over 64 lanes.
- Packed math: INT16/FP16 can run **double-rate** via `pk` instructions; INT32 adds are single-rate.
- Special functions (rsqrt, exp, etc.): 4 ops / SIMD / cycle (transcendental unit).

### 3.2 Per-CU peak throughput (per clock, the basis of all peak FLOPS)
| Computation | FLOPs/clock/CU | MI300X peak (304 CU @ 2.1 GHz) |
|---|---|---|
| Vector FP64 | 128 | 81.7 TFLOP/s |
| Matrix FP64 | 256 | 163.4 TFLOP/s |
| Vector FP32 | 256 | 163.4 TFLOP/s |
| Matrix FP32 | 256 | 163.4 TFLOP/s |
| Vector TF32 (emulated) | 1024 | 653.7 TFLOP/s |
| Matrix FP16 / BF16 | 2048 | 1307.4 TFLOP/s |
| Matrix FP8 / INT8 | 4096 | 2614.9 TFLOP/s |

Peak formula: `FLOPs_per_clock_per_CU × 304 CU × 2.1e9 Hz`. Example FP16: `2048 × 304 × 2.1e9 ≈ 1307 TFLOP/s`.

> Takeaway: vector FP32 and matrix FP32/FP64 are all ~163 TFLOP/s — there is **8×** to be gained by dropping to FP16/BF16 MFMA and **16×** at FP8. For inference, push to the lowest viable precision.

---

## 4. Memory & bandwidth ladder (numbers a kernel author needs)

| Level | Size | Scope | Latency | Bandwidth | Notes |
|---|---|---|---|---|---|
| VGPR/AGPR | 512+256 ×4B ×4 SIMD | per wave | 0 | — | Fastest; occupancy-limiting |
| LDS | 64 KiB | per workgroup (CU) | low (a few dozen cyc) | highest measured of any GPU tested | 32 banks, padding avoids conflicts |
| L1 vector | 32 KiB | per CU | ~tens of cyc | tens of TB/s | 128 B line, write-through |
| L2 | 4 MiB | **per XCD** | — | ~H100-class | coalesces XCD traffic |
| Infinity Cache (L3/MALL) | 256 MiB | **device** | ~218 ns | ~11.9 TB/s (17 theoretical) | cross-XCD coherence point |
| HBM3 | 192 GB | device | + ~47 ns TLB miss | **5.3 TB/s** | 128 × 16-bit channels |

- Cache line **128 bytes**; page size **4 KiB** (use huge pages for big working sets — TLB reach ~64 MB/XCD with 4K pages).
- **HBM-bound reality:** at 5.3 TB/s and 1307 FP16 TFLOP/s, the FP16 roofline ridge is ~**247 FLOP/byte**. Decode-phase LLM kernels (GEMV, attention with small batch, RMSNorm, RoPE, dequant) sit far left of the ridge → they are **bandwidth-bound**; optimize bytes moved, fuse ops, and exploit Infinity Cache residency.

---

## 5. Compute & memory partitioning (SPX / DPX / CPX × NPS1/2/4)

The 8-XCD design can be **spatially partitioned** at driver level via `amd-smi`. This is invisible to a single kernel but changes the device(s) your kernel sees.

### Compute partition modes
| Mode | Logical GPUs | XCDs each | CUs each | HBM each (NPS1) | Use |
|---|---|---|---|---|---|
| **SPX** (default) | 1 | 8 | 304 | 192 GB | one big model / kernel needing the whole device |
| **DPX** | 2 | 4 | 152 | 96 GB | two balanced jobs |
| **CPX** | 8 | 1 | 38 | 24 GB | many small jobs, multi-tenant, inference density |

### Memory partition modes (NUMA-nodes-per-socket)
| Mode | NUMA domains | Pairing | Effect |
|---|---|---|---|
| **NPS1** | 1 | any | unified 192 GB pool, interleaved across all 8 stacks |
| **NPS2** | 2 | DPX | each half owns a memory quadrant |
| **NPS4** | 4 | CPX only | each XCD's traffic stays on its local IOD → lower latency, higher clocks |

**Hard rule:** number of memory partitions must **not exceed** compute partitions → `SPX+NPS4` is **invalid**. Valid: SPX+NPS1, DPX+NPS1/2, CPX+NPS1/4.

**Kernel-author implications:**
- In **CPX/NPS4**, a kernel sees a 38-CU / 24 GB "GPU" with memory local to its XCD → measurably **higher effective bandwidth and clocks** (compute-bound GEMM throughput +10–15% vs SPX in AMD measurements) because cross-XCD traffic is eliminated.
- For a single large model that must span all CUs and >24 GB, you **must** use SPX. Strong-scaling one kernel across all 8 XCDs pays Infinity-Fabric/Cache costs for any cross-XCD data sharing.
- Switching mode requires terminating GPU processes and reloading the amdgpu driver; the device reverts to SPX/NPS1 on reboot.

```bash
# Inspect and set partitioning
amd-smi list
sudo amd-smi set --gpu all --compute-partition CPX
sudo amd-smi set --gpu all --memory-partition  NPS4   # auto-coerces compute partition if needed
```

---

## 6. Grid / occupancy design rules (derived)

1. **Launch ≥ 1024 workgroups** (≥ ~3.4 per CU) so the scheduler hides tails and latency. More is fine; the HWS load-balances across XCDs.
2. **Keep workgroups ≤ 256–1024 threads** (4–16 waves) so they fit a CU; remember occupancy is per-SIMD (512 VGPR / SIMD).
3. **Target 4 waves/CU minimum** for memory-latency hiding on HBM-bound kernels; matmul kernels often run 1–4 waves/CU and rely on MFMA + double-buffered LDS instead.
4. **Avoid cross-XCD dependencies** within a kernel; if you need a device-wide reduction, stage through Infinity Cache and expect ~200 ns sync costs.
5. **Use MFMA** for any matmul-shaped work — see `matrix_cores_numerics.md`. `mfma_16x16` usually beats `mfma_32x32` on MI300X even for large tiles.

---

## 7. Quick gfx942 facts for the compiler / tooling

| Item | Value |
|---|---|
| ISA target | `gfx942` (`--offload-arch=gfx942`) |
| Wave size | 64 (no wave32 on CDNA3) |
| Calculator arch keyword | `cdna3` (or gfx940/941/942, MI300, MI300X, MI325X) |
| Profiler | `rocprofiler-compute` (Omniperf), `rocprof`, `rocprofv3` |
| Sibling CDNA4 target | `gfx950` (MI350X/MI355X) — adds FP6/FP4/MXFP, 160 KiB LDS |

---

## Sources
1. AMD Instinct MI300 Series microarchitecture — ROCm Documentation: https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi300.html
2. AMD CDNA™ 3 Architecture White Paper: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf
3. "AMD Instinct MI300X Generative AI Accelerator and Platform Architecture" — Hot Chips 2024: https://hc2024.hotchips.org/assets/program/conference/day1/23_HC2024.AMD.MI300X.ASmith(MI300X).v1.Final.20240817.pdf
4. Deep dive into the MI300 compute and memory partition modes — ROCm Blogs: https://rocm.blogs.amd.com/software-tools-optimization/compute-memory-modes/README.html
5. AMD Instinct MI300X GPU Partitioning Overview — AMD GPU Driver docs: https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/gpu-partitioning/mi300x/overview.html
6. "Testing AMD's Giant MI300X" — Chips and Cheese (measured latencies/bandwidths): https://chipsandcheese.com/p/testing-amds-giant-mi300x
7. AMD Instinct MI300X Data Sheet (clocks, peak FLOPS): https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf
8. AMD Instinct MI300 (CDNA3) Instruction Set Architecture Reference Guide: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
