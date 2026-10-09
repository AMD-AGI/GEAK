---
title: CDNA4 / MI350X / MI355X (gfx950) — clocks, power & thermals
kind: hardware
gens: [gfx950]
dtypes: []
regimes: [both]
updated: 2026-10-07
sources:
  - https://www.amd.com/en/products/accelerators/instinct/mi350/mi350x.html
  - https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html
  - https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf
  - https://www.servethehome.com/amd-mi350-and-cdna-4-architecture-launched-with-rocm-7/
  - https://www.guru3d.com/story/amd-instinct-mi350-gpus-with-288gb-hbm3e-and-1400w-tdp-announced/
---

# CDNA4 / MI350X / MI355X (gfx950) — clocks, power & thermals

> Peak math uses the clocks here; matrix peaks in [peak_tables.md](peak_tables.md). Per-XCD clock
> variance behaves like CDNA3 — see [../cdna3_mi300/xcd_chiplet.md](../cdna3_mi300/xcd_chiplet.md).

## TL;DR
> Two SKUs of one gfx950 die differ in **peak engine clock and power**: **MI350X air-cooled, 1000 W,
> 2200 MHz**; **MI355X liquid-cooled, 1400 W, 2400 MHz**. The clock difference makes their **datasheet
> peaks differ** (FP16 2.3 vs 2.5 PF, FP32 144.2 vs 157.3 TF — see [peak_tables.md](peak_tables.md));
> the power difference additionally lets MI355X *sustain* more of its peak. N3P process, 185 B
> transistors. As always, **sustained < peak** — compute achieved FLOP/s from measured time.

## Concepts

### Clock & power table
| Param | MI350X | MI355X |
|---|---|---|
| Arch / ISA | CDNA4 / gfx950 | CDNA4 / gfx950 |
| Cooling | air | liquid |
| TDP | **1000 W** | **1400 W** |
| Peak engine clock | **2200 MHz** | **2400 MHz** |
| Dense FP16 / FP8 / FP4 peak | 2.3 / 4.6 / 9.2 PF | 2.5 / 5.0 / 10 PF |
| Process | TSMC **N3P** (XCD) + N6 (IOD) | N3P + N6 |
| Transistors | 185 B | 185 B |
| HBM | 288 GB HBM3E, 8 TB/s | 288 GB HBM3E, 8 TB/s |
| Rack density | up to 10U (air) | 5U (liquid) |

Both have the **same CU design** (256 CU, identical per-CU per-clock matrix rate), so their peaks are
in the ratio of their peak clocks, 2200 : 2400 — MI350X's datasheet peaks are ~8% lower, not equal.
On top of that, MI355X's higher power/cooling sustains a larger fraction of its own peak under heavy
AI load. Peaks per SKU: `perf_knowledge/hardware/data/sku.json` (single source).

### Clock dynamics that bite kernels
- **Peak ≠ sustained.** Like CDNA3, sustained AI-load clock settles below boost; the higher 1400 W
  MI355X envelope keeps clock up longer. Always compute achieved TFLOP/s from wall time.
- **Per-XCD clock variance** (~3–10%, same mechanism as CDNA3): device-wide-synchronized kernels run
  at the slowest XCD; independent grids unaffected beyond load balance.
- **HBM bandwidth (8 TB/s)** is set by the memory data rate, independent of engine clock —
  bandwidth-bound kernels gain nothing from clock headroom, only from fewer bytes.
- **2× matrix throughput/CU** vs CDNA3 means compute-bound GEMM is more sensitive to clock throttling;
  on the 1000 W MI350X watch for power-capped clock under sustained FP8/FP16.

## The levers
1. **Measure achieved FLOP/s** from time; treat peak clock as a ceiling.
2. **Warm up + median-of-≥3** for DVFS lag and XCD variance.
3. **Prefer MI355X (1400 W)** for sustained compute-bound throughput; MI350X (1000 W) for air-cooled
   density.
4. **For bandwidth-bound work**, cut bytes — clock is irrelevant.
5. **CPX/NPS** can localize power/thermal per XCD for many-small-job density.

## Pitfalls
- **Using peak clock in efficiency claims** — overstates utilization.
- **Quoting MI355X peaks for an MI350X (or the reverse)** — the datasheet peaks differ by the
  2200/2400 clock ratio, and the sustained clock under the power cap differs again on top of that.
  Take the product from `scripts/gpu_identity.py` (field `sku`: `mi350x` / `mi355x`) and use its own row.
- **Cold-launch timing** captures pre-ramp clock.

## Verify
- `amd-smi metric --gpu <id>` (sclk/mclk/power/temp/throttle) during the kernel.
- `rocprof-compute` achieved vs theoretical at the measured clock.

## Sources
- AMD CDNA4 Architecture White Paper (clocks, power, 256 CU, N3P):
  https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf
- ServeTheHome, "AMD MI350 and CDNA 4 launched with ROCm 7" (N3P, 185 B transistors, 1000/1400 W):
  https://www.servethehome.com/amd-mi350-and-cdna-4-architecture-launched-with-rocm-7/
- Guru3D, "AMD Instinct MI350 GPUs with 288GB HBM3E and 1400W TDP" (TDP, ~2400 MHz MI355X):
  https://www.guru3d.com/story/amd-instinct-mi350-gpus-with-288gb-hbm3e-and-1400w-tdp-announced/
- AMD Instinct MI350X / MI355X product pages (peak engine clock 2200 / 2400 MHz; per-SKU peaks):
  https://www.amd.com/en/products/accelerators/instinct/mi350/mi350x.html ,
  https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html
- AMD Instinct MI350X GPU datasheet (1000 W air, clocks):
  https://www.koicomputers.com/wp-content/uploads/2025/08/amd-instinct-mi350x-gpu-datasheet.pdf
