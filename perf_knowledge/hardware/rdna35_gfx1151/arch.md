---
title: RDNA3.5 / gfx1151 (Strix Halo) — architecture overview
kind: hardware
gens: [gfx1151]
dtypes: [fp16, bf16, int8, int4]
regimes: [both]
updated: 2026-11-06
sources:
  - ../../../kernel_workflow/knowledge/amd_ryzen.md
  - ../data/hw_constants.json
  - ../../../e2e_workflow/knowledge/analysis_skills/roofline/peaks.md
  - repro/README.md
---

# RDNA3.5 / gfx1151 (Strix Halo) — architecture overview

> Workflow agents should prefer [`kernel_workflow/knowledge/amd_ryzen.md`](../../../kernel_workflow/knowledge/amd_ryzen.md),
> which is the summary; this directory is the measured detail behind it. The machine model,
> occupancy and theoretical peaks live in `amd_ryzen.md`,
> [`hw_constants.json`](../data/hw_constants.json)
> and [`roofline/peaks.md`](../../../e2e_workflow/knowledge/analysis_skills/roofline/peaks.md), and are
> not repeated here. Occupancy is in [occupancy.md](occupancy.md), pitfalls in [pitfalls.md](pitfalls.md).

## TL;DR
> gfx1151 ships as the Radeon 8060S (40 CU), 8050S (32 CU) and 8040S (16 CU); every number in this
> directory was measured on an 8060S. It is a GPU on an APU: it shares LPDDR5X and its power budget
> with the CPU. Measured, not theoretical: DRAM **~232 GB/s** to reads, Infinity Cache **~920 GB/s**,
> WMMA **~36 T/s** fp16 (~27 sustained for seconds on large GEMMs), engine clock **~2.6 GHz at
> 139 W** under load.

## The one-screen cheat sheet
| Fact | Value (Radeon 8060S) | Detail |
|---|---|---|
| Machine model (wave32, WMMA dtypes, no fp8 matrix, VGPR, LDS, SKUs) | see `amd_ryzen.md` §0–1 | — |
| L1 / cache line | 32 KiB / 128 B (`rocminfo`) | — |
| DRAM, measured | 230–233 GB/s read, 220–233 read-read-write (pin 256) | [peak_tables.md](peak_tables.md) |
| Infinity Cache, measured | ~920 GB/s read | [peak_tables.md](peak_tables.md), [memory.md](memory.md) |
| WMMA, measured (best-size median) | fp16 36.2, bf16 35.1, int8 35.2 T/s (theoretical 60) | [peak_tables.md](peak_tables.md) |
| Engine clock under sustained GEMM | 2.54–2.68 GHz, 139 W (nominal 2.9) | [clocks_power.md](clocks_power.md) |
| Noise floor, quiet box | 2σ ≤ 0.4% | [clocks_power.md](clocks_power.md) |
| PMC budget | a 4th counter in one tested set is rejected (error 38), and that pass hangs until killed | [pitfalls.md](pitfalls.md) |

## Concepts
- **An APU, not a discrete card.** CPU load moves GPU numbers through shared DRAM bandwidth and a
  shared power budget ([clocks_power.md](clocks_power.md)).
- **About 60% of theoretical is the measured ceiling** for every WMMA dtype, and GEMMs held for
  seconds sustain less ([peak_tables.md](peak_tables.md)).
- **The Infinity Cache is the main bandwidth lever**, inside a narrow intensity window
  ([memory.md](memory.md)).

## The levers
1. Feed WMMA through 128-bit operand loads and size tiles for the CU count and for decode M
   ([matrix_core.md](matrix_core.md)).
2. Keep the hot working set in the Infinity Cache when the kernel is bandwidth-bound
   ([memory.md](memory.md)).
3. Measure on a quiet host, with batched timing for short kernels ([clocks_power.md](clocks_power.md)).

## Pitfalls
- Scoring against the theoretical peaks alone, including the headroom GEAK's roofline skill reports;
  use the measured bracket in [peak_tables.md](peak_tables.md).
- Reading memory type or bandwidth from `amd-smi`, or assuming `rocprof-compute` runs here; it
  depends on the release ([pitfalls.md](pitfalls.md)).
- Applying these numbers to an 8050S / 8040S; re-measure with [repro/](repro/README.md).

## Verify
```bash
# rocminfo lists the CPU agent first; list the distinct gfx agents.
rocminfo | awk '/^ *Name: *gfx/{print $2}' | sort -u   # gfx1151
```
- The CU count (which SKU) with the query in `amd_ryzen.md` §0.
- [repro/README.md](repro/README.md) lists the script behind each measured number.

## Sources
- Measurements: [repro/](repro/README.md) on a Radeon 8060S, ROCm 7.2.3, 2026-11-06.
- Machine model, occupancy, theoretical peaks: `amd_ryzen.md`, `hw_constants.json` (`gfx1151`),
  `roofline/peaks.md` (linked above).
