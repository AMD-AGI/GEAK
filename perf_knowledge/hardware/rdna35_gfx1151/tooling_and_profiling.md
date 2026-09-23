---
title: gfx1151 tooling — profiler support, PMC budget, and three tools that lie
kind: hardware
gens: [gfx1151]
dtypes: [fp32, bf16, fp16, int8, int4]
regimes: [both]
updated: 2026-09-23
sources:
  - on-box rocprofv3 / rocprof-compute / amd-smi, ROCm 7.2.3
  - GEAK/kernel_workflow/knowledge/repro/pmc_slots.sh, pmc_work.py
  - GEAK/kernel_workflow/knowledge/amd_rdna.md §7, §8
---

# gfx1151 tooling and profiling

> Part overview in [arch.md](arch.md); measurement hygiene in [clocks_power.md](clocks_power.md).

## TL;DR
> `rocprofv3` works. Around it, **three tools report wrong or missing data on this part** —
> `amd-smi` calls LPDDR5X "GDDR7", `rocprof-compute` leaves `$max_mclk` unpopulated, and the
> CDNA-scoped roofline skill refuses RDNA outright. Also: the PMC limit is a **hardware-counter
> budget, not a metric count**, and profiling cost is charged **per process start** — ~5.4 s warm
> vs **~85 s cold**, a 16× lever on campaign wall-clock.

## Known defects

| Tool | Symptom | Handling |
|---|---|---|
| `amd-smi static` | reports `VRAM TYPE: GDDR7` | **It is LPDDR5X.** Never read memory type or bandwidth from `amd-smi` here; use `repro/bw.py` — see [memory.md](memory.md) |
| `rocprof-compute` | `$max_mclk` unpopulated | pass `--specs-correction` with the measured value, or every derived bandwidth metric is wrong |
| `rocprof-compute` | output dir named `rdna35_halo` | renamed from `strix_halo`; scripts that hard-code the old name find nothing and silently report no data |
| `perf_knowledge/profiling/kernel_roofline.md` | refuses RDNA | it drives `rocprof-compute --roof-only`, whose roofline mode does not support gfx10/11/12. Not a bug — use the measured path in [peak_tables.md](peak_tables.md) instead |
| `rocprofv3` | — | **works**; this is the supported path |

## The PMC limit is a counter BUDGET, not a metric count
`[measured]` Adding a *fourth* counter is not what fails — exceeding the underlying hardware
counter budget is.

```
SQ_WAVES + GRBM_GUI_ACTIVE + FETCH_SIZE            -> ok
  ... + WRITE_SIZE                                 -> error code 38
```

So a working three-metric set gives no guarantee that a different three-metric set fits. **Derive
your slot budget empirically** (`repro/pmc_slots.sh` does exactly this) rather than reasoning about
how many metrics you asked for, and split into multiple passes when it does not fit.

`GL2C_HIT` / `GL2C_MISS` **are** collectable — one real run measured **GL2 hit 50.37%** — so L2
traffic for a hierarchical roofline is obtainable on this part. Note the raw CDNA `TCP_`/`TCC_`/`TD_`
counter names are *not* available; do not port a CDNA `pmc:` list verbatim.

## Profiling cost is charged per PROCESS START
`[measured]` Per profiling pass: **~5.4 s warm, ~85 s cold**.

The dominant term is **starting the process**, not the number of counters inside it. Consequences:
- **Budget a campaign by the number of process starts**, not by the number of metrics.
- Collect as many counters as the slot budget allows **per launch**, then split — do not launch once
  per metric.
- Keep the process warm across passes where the harness permits; the 16× cold/warm gap dwarfs any
  saving from trimming the counter list.

## The levers
1. **Use `rocprofv3`**; treat `rocprof-compute` output as suspect until `--specs-correction` is set.
2. **Probe the PMC slot budget empirically, per counter set.**
3. **Minimise process starts, maximise counters per start.**
4. **Never take a hardware fact from `amd-smi` on this part** — cross-check against `rocminfo` and
   `repro/bw.py`.

## Pitfalls
- **Believing `GDDR7`.**
- **Deriving bandwidth from `rocprof-compute` without `--specs-correction`.**
- **Hard-coding `strix_halo` as the output directory** — it is `rdna35_halo` now, and the failure is
  silent (no data, no error).
- **Assuming "3 metrics always fits".**
- **Porting a CDNA `pmc:` list.**
- **Budgeting profiling time by metric count** instead of process starts.

## Verify
```bash
rocminfo | awk '/^ *Name: *gfx/{print $2; exit}'   # ground truth for the target
```
- `repro/pmc_slots.sh` — empirical counter-budget probe.
- `repro/pmc_work.py` — a known-workload sanity check for collected counters.

## Sources
- On-box `rocprofv3`, `rocprof-compute`, `amd-smi`, ROCm 7.2.3.
- `GEAK/kernel_workflow/knowledge/repro/pmc_slots.sh`, `pmc_work.py`.
- `GEAK/kernel_workflow/knowledge/amd_rdna.md` §7, §8.
