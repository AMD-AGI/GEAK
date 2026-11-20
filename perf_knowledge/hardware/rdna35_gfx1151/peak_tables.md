---
title: RDNA3.5 / gfx1151 (Strix Halo) — peak tables (measured)
kind: hardware
gens: [gfx1151]
dtypes: [fp16, bf16, int8]
regimes: [both]
updated: 2026-11-06
sources:
  - ../../../e2e_workflow/knowledge/analysis_skills/roofline/peaks.md
  - repro/ceilings.py
  - repro/bw.py
  - repro/wmma_check.py
---

# RDNA3.5 / gfx1151 — peak tables (measured)

> Theoretical denominators are the `gfx1151` section of
> [`roofline/peaks.md`](../../../e2e_workflow/knowledge/analysis_skills/roofline/peaks.md); this card
> does not restate them. It is what was measured on a Radeon 8060S (40 CU). The 8050S and 8040S share
> the ISA with fewer CUs and were not measured.

## TL;DR
> Measured WMMA tops out at **~36 TFLOP/s fp16, ~35 bf16, ~35 TOP/s int8**, about 60% of theoretical.
> DRAM delivers **~232 GB/s** to reads (91% of the 256 GB/s pin rate) and the 32 MiB Infinity Cache
> **~920 GB/s**. Quote efficiency as a bracket between the theoretical and the measured value.

## Memory & cache peaks
`value @ gfx1151 (Radeon 8060S), ROCm 7.2.3, torch 2.11.0, Triton 3.6.0, 2026-11-06`, three runs each:

| Level | Measured | Theoretical | Source |
|---|---:|---:|---|
| DRAM, read | 230.5–233.3 GB/s | 256 GB/s | `ceilings.py bw`, 89 and 178 MB, cold, per-call timed |
| DRAM, read-read-write | 220–233 GB/s | 256 GB/s | `bw.py` (host-timed, 64–1024 MB, two runs), `noise.py` (256 MB) |
| Infinity Cache (MALL), read | 919–920 GB/s | — | `ceilings.py bw`, 24 MB, warm, timed in batches of 20 calls |

Mixed traffic reaches about the read rate. The read kernel writes one int32 per program, so its
bytes are reads only.

The shared [`sku.json`](../data/sku.json) retains an older 212 GB/s `rocm_bandwidth_test`
report for `AI_MAX_395`, with an unrecorded tool version and a source that is no longer available.
This campaign is separate evidence with the conditions and reproduction scripts above; it does
not replace the catalog default or a TraceLens calibration. Use a measurement from the target
box and matching workload for an attainable ceiling.

## Sustained reality
Dense square GEMM, `value @ gfx1151 (Radeon 8060S), ROCm 7.2.3, torch 2.11.0, Triton 3.6.0,
2026-11-06`. Each cell is the median over three runs of the best implementation in that run
(`torch.mm` = rocBLAS, `torch._int_mm`, or a 24-config Triton `tl.dot` sweep; each timing is a median
of 10 GPU-event-timed calls), with the run-to-run range ([`repro/ceilings.py`](repro/ceilings.py)
`peak`):

| Precision | 1024³ | 2048³ | 3072³ | 4096³ | 8192³ |
|---|---:|---:|---:|---:|---:|
| fp16 → fp32 | 36.0 (36.0–36.2) | 33.5 (32.5–34.0) | **36.2** (36.1–36.5) | 34.8 (34.4–36.2) | 29.1 (28.9–30.3) |
| bf16 → fp32 | 33.6 (31.7–34.4) | 33.5 (33.0–34.2) | **35.1** (34.5–35.3) | 31.0 (30.0–31.9) | 29.1 (29.0–29.8) |
| int8 (int32 accumulate) | 32.1 (31.8–32.3) | 32.5 (32.4–33.8) | 34.4 (34.4–35.3) | 31.7 (31.7–32.6) | **35.2** (35.0–35.4) |

- The best cell is ~60% of the theoretical 60 T for every dtype. Under sustained load the engine
  clock sits at ~2.54–2.68 GHz instead of the 2.9 GHz the theoretical figure assumes
  ([clocks_power.md](clocks_power.md)); that accounts for part of the gap, not all of it.
- Each cell is a short burst (tens of calls). A GEMM that runs for seconds is slower: `torch.mm`
  fp16 8192³ held for 16 s settles at 26.8–27.2 TFLOP/s ([clocks_power.md](clocks_power.md)).
- fp16 / bf16 drop at 8192³ while int8 does not, so the large-shape drop is not a property of the
  matrix unit alone.
- Record any new number in the format above; sustained throughput depends on shape and library.

**Scoring a kernel.** Report a bracket: achieved / theoretical .. achieved / measured at the same
shape and dtype. For example a 2048³ fp16 Triton GEMM at 24.05 TFLOP/s
([`repro/wmma_check.py`](repro/wmma_check.py), median of 40 calls interleaved with `torch.mm`) is
40% .. 72%. Above 100% of the measured cell means the denominator is wrong, not that the kernel is
fast.

**GEAK's roofline skill uses the theoretical row.** It scores gfx1151 against `peaks.md` (60 T)
times `target_eff` 0.90, so a 3072³ fp16 GEMM at the measured 36.2 T still reports
`attainable_speedup` 1.49 and `headroom_class` `moderate` (`roofline_tools.roofline_metrics`, checked
2026-11-06). Read that headroom against the table above before planning on it.

## Sources
- [`repro/ceilings.py`](repro/ceilings.py) (`bw`, `peak`), [`repro/bw.py`](repro/bw.py),
  [`repro/wmma_check.py`](repro/wmma_check.py) — Radeon 8060S, 2026-11-06.
- Theoretical values: [`roofline/peaks.md`](../../../e2e_workflow/knowledge/analysis_skills/roofline/peaks.md).
