---
title: CDNA2 / MI250X / MI210 (gfx90a) — memory hierarchy
kind: hardware
gens: [gfx90a]
dtypes: []
regimes: [both]
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi250.html
  - https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna2-white-paper.pdf
layer: reference
platforms: [gfx90a]
lifecycle: active
bound_type: [hbm_bw, lds_bank]
---

# CDNA2 / MI250X / MI210 (gfx90a) — memory hierarchy

> Cross-gen LDS rules in [../shared/memory_model_lds_bank.md](../shared/memory_model_lds_bank.md);
> HBM/Fabric in [../shared/hbm_infinity_fabric.md](../shared/hbm_infinity_fabric.md). This file is the
> per-GCD CDNA2 ladder and the GCD↔GCD caveat.

## TL;DR
> Per GCD: 64 KiB LDS (32 banks) → 32 KiB L1/CU → **~8 MiB L2 (per GCD)** → **64 GB HBM2e @ 1.6 TB/s**.
> The trap: an MI250X is **two GCDs**, and **L2 is not shared across GCDs** — cross-GCD data crosses a
> 400 GB/s bridge as a multi-GPU transfer, not a cache.

## Concepts

### The ladder (per GCD)
| Level | Capacity | Scope | Bandwidth | Notes |
|---|---|---|---|---|
| VGPR | 512 ×4 B/SIMD | wave | — | 16-granule alloc |
| AGPR | ≤256 ×4 B/SIMD | wave | — | MFMA accumulators |
| LDS | 64 KiB/CU, 32 banks | workgroup | up to 128 B/clk | bank = `(addr/4) mod 32` |
| L1 vector | 32 KiB/CU, 128 B line | CU | — | write-through |
| L2 | ~8 MiB | **per GCD** | — | not shared across the 2 GCDs |
| HBM2e | **64 GB/GCD** (128 GB/OAM) | per GCD | **1.6 TB/s/GCD** (3.2 TB/s OAM) | |
| GCD↔GCD bridge | — | inter-GCD | 400 GB/s bidir | Infinity Fabric, not a cache |

No 256 MiB Infinity Cache (that arrives on CDNA3) — CDNA2's largest cache is the per-GCD L2.

### GCD↔GCD is a multi-GPU boundary
There is **no unified L2 or address space across the two GCDs** by default. Sharing data means a peer
copy / collective over the 400 GB/s (200/dir) bridge. Design multi-GCD work like multi-GPU: partition
so each GCD owns its working set; use RCCL for collectives. (Contrast CDNA3's single-logical-GPU XCD
model with a device-shared 256 MiB L3.)

### Coalescing & loads
Same CDNA family rules: coalesce to **128 B-aligned** `global_load_dwordx4`; prefer `buffer_*` for
bounds-checked tiled GEMM; `ds_read_b128`/`ds_write_b128` for LDS. **Direct global→LDS** exists on
CDNA2 (32 b/lane, like CDNA3). FP16 roofline ridge (per GCD) ≈ 362 TF / 1.6 TB/s ≈ 226 FLOP/byte.

## The levers
1. **Partition work per GCD**; keep each GCD's working set in its own 64 GB / 1.6 TB/s.
2. **Avoid cross-GCD sharing**; if needed, treat it as a multi-GPU transfer (RCCL / peer copy).
3. **Coalesce to 128 B**, pad LDS off 32-bank conflicts, use `ds_*_b128`.
4. **Direct global→LDS + double-buffer** for GEMM staging.

## Pitfalls
- **Assuming a shared L2 across GCDs** — there isn't one.
- **Treating 128 GB as one pool** — it's 2 × 64 GB on two devices.
- **Cross-GCD reuse** silently routed over the 400 GB/s bridge.

## Verify
- `rocminfo` lists two gfx90a devices; `rocm-smi --showmeminfo` shows 64 GB each.
- `rocprof-compute` per-GCD L2/HBM utilization.

## Sources
- AMD Instinct MI250 microarchitecture — ROCm Docs (per-GCD memory, 1.6/3.2 TB/s, GCD↔GCD 400 GB/s):
  https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi250.html
- AMD CDNA2 White Paper (L2 per GCD, LDS, Infinity Fabric):
  https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna2-white-paper.pdf
- AMD Instinct MI250X Data Sheet (64 GB HBM2e/GCD, 1.6 TB/s):
  https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi250x-datasheet.pdf
