---
title: RDNA3.5 / gfx1151 (Strix Halo) — clocks, power & measurement noise
kind: hardware
gens: [gfx1151]
dtypes: []
regimes: [both]
updated: 2026-11-06
sources:
  - repro/ceilings.py
  - repro/noise.py
  - repro/drift.py
  - repro/wmma_check.py
---

# RDNA3.5 / gfx1151 — clocks, power & measurement noise

## TL;DR
> The GPU shares one power budget and one DRAM with the CPU. Under a sustained GEMM the engine clock
> settles at **~2.54–2.68 GHz at 139 W**, below the 2.9 GHz the theoretical peak assumes. On a quiet
> box the noise floor is tiny (2σ ≤ 0.4%) when calls are timed in batches; the large errors come from
> the host and from timing single short calls.

## Concepts
- **Sustained clock.** `[measured]` `torch.mm` fp16 8192³ for 16 s: median engine clock 2.54–2.68 GHz,
  139 W package power, 26.8–27.2 TFLOP/s ([`repro/ceilings.py`](repro/ceilings.py) `clock`, three
  runs). At 2.6 GHz the theoretical 60 TFLOP/s becomes ~54; see [peak_tables.md](peak_tables.md).
- **Two contamination channels.** CPU work takes DRAM bandwidth, and CPU boost takes power the GPU
  clock would use. A GPU-side idle check sees neither.
- **Noise floor.** `[measured]` 256 MB streaming, 20 calls per timed sample, 12 samples on an idle
  box: CV 0.07–0.15%, 2σ 0.1–0.3% ([`repro/noise.py`](repro/noise.py), 2026-11-06; 0.18% / 0.4% on
  2026-09-17). 4 minutes of sustained memory-bound load drift by +0.00–0.10%
  ([`repro/drift.py`](repro/drift.py)). The drift check covers memory-bound load only; a compute-bound
  load runs at the lower clock above.

## The levers
1. Check the host out of band (load average, CPU utilisation, governor, leaked processes)
   immediately before each run.
2. Time short kernels in batches between one pair of GPU events, and compare medians of many
   interleaved rounds.
3. Treat a delta inside 2σ of the floor on your box as no result, in either direction.

## Pitfalls
- **Timing single short calls.** `[measured]` a 2048³ fp16 A/B (~0.6 ms per call, one call per event
  pair, [`repro/wmma_check.py`](repro/wmma_check.py)) spread 16.7% / 34.0% (2026-09-17) and 14.2% /
  37.5% (2026-11-06) round to round, with the cpufreq governor at `powersave`, while batched
  `noise.py` in the same session stayed at 0.15%. Per-call timing includes host launch latency, which
  depends on the CPU clock; the governor was not varied, so it is not shown to be the cause.
- **Leaked processes.** Stale host processes burning ~3.6 cores shifted a benchmark batch by 18%.
  Inside the ROCm container, a self-sustaining chain of `rocm_agent_enumerator` processes
  (re-forking every ~12 s after `rocminfo` hits an allocation failure, 0.7–1.2 GB each) exhausted
  memory on a 30 GB box and pushed replica spread to 45–68%; after clearing it the same arm measured
  0.33% (serving campaigns; [repro/README.md](repro/README.md)). Count them with
  `ps -eo args | grep -c '[r]ocm_agent_enumerator'`; it should be 0–1.
- **Trusting the idle clock reading.** It has read anywhere from 623 MHz to 1.9 GHz on an idle box;
  use the sustained value.

## Verify
- Host side, outside the container: `/proc/loadavg`, CPU utilisation,
  `/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor`, the process count above.
- [`repro/noise.py`](repro/noise.py) on the box you use; [`repro/ceilings.py`](repro/ceilings.py)
  `clock` for the clock your workload runs at.

## Sources
- [`repro/ceilings.py`](repro/ceilings.py), [`repro/noise.py`](repro/noise.py),
  [`repro/drift.py`](repro/drift.py), [`repro/wmma_check.py`](repro/wmma_check.py) — Radeon 8060S,
  2026-11-06.
- Leaked-process figures: gfx1151 serving campaigns (2026-09); no script, see
  [repro/README.md](repro/README.md).
