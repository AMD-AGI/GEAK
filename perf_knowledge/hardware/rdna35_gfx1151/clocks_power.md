---
title: gfx1151 clocks, shared power envelope, and the measurement noise floor
kind: hardware
gens: [gfx1151]
dtypes: [fp32, bf16, fp16, int8, int4]
regimes: [both]
updated: 2026-09-23
sources:
  - GEAK/kernel_workflow/knowledge/repro/noise.py (idle CV over repeated rounds)
  - GEAK/kernel_workflow/knowledge/repro/drift.py
  - GEAK/kernel_workflow/knowledge/amd_rdna.md §1, §7
---

# gfx1151 clocks, power and noise floor

> Part overview in [arch.md](arch.md); profiler defects in
> [tooling_and_profiling.md](tooling_and_profiling.md).

## TL;DR
> This is an **APU**: the CPU and the GPU share one memory system *and* one power envelope, so a
> busy CPU is not neutral background — it moves your GPU number. On a quiet box the noise floor is
> `[measured]` **idle CV 0.18%, 2σ = 0.4%**. Anything under 0.4% is noise. The clock-lock script
> only locks **inside the container**, so an **out-of-band idle check is mandatory** before every
> measurement.

## Concepts

### Shared envelope
| | |
|---|---|
| Engine clock | ~2.9 GHz (the basis of the 59.4 TFLOP/s datasheet math in [peak_tables.md](peak_tables.md)) |
| Memory | LPDDR5X, UMA, **shared with the CPU** |
| Power | **shared CPU+GPU budget** |

Two distinct contamination channels, and they need different checks:
1. **Bandwidth** — a CPU streaming memory takes bandwidth the GPU kernel wanted.
2. **Power** — CPU turbo takes headroom the GPU clock wanted, so the GPU downclocks without any
   memory contention at all.

A GPU-side idle check catches neither. Check the **host**.

### The cpufreq governor is a first-order measurement variable
`[measured]` With the CPU governor left at `powersave`, round-to-round spread on an otherwise
identical A/B reached **16.7% and 34.0%** — two orders of magnitude above the 0.4% noise floor, and
large enough to manufacture or erase any realistic kernel win. Pin the governor (or at minimum
record it, and re-run any result whose spread exceeds the floor).

### The noise floor
`[measured]` `repro/noise.py` on an idle box: **CV 0.18%**, giving a **2σ significance threshold of
0.4%**.

- Report a delta **with** its spread. A "+3%" with a 20% spread is not a result.
- A measured **regression** smaller than 0.4% is also noise — do not reject a kernel on it.
- The floor is for a *quiet* box. It is not valid while anything else is running, which is exactly
  what the out-of-band check below is for.

### Cold vs warm is a bigger effect than most kernel wins
`[measured]` On real decode weights the cold/warm gap is **3–4×**. Warm-up policy is therefore not a
detail: an unwarmed baseline against a warmed candidate fabricates a multi-fold "win". This is the
same defect that produced the retracted 1.98× GEMM claim in [peak_tables.md](peak_tables.md) —
there, on the vendor side.

## The levers
1. **Check host idle out-of-band before every run**, not from inside the container.
2. **Pin the CPU governor**, or treat any run with >0.4% spread as unusable.
3. **Warm both arms identically**, pre-allocate both outputs, interleave the arms, time with GPU
   events, and take a median over many rounds — not a mean over a few.
4. **Compare against the 0.4% floor explicitly** before calling anything a win or a regression.

## Pitfalls
- **Trusting `gpu_lock.sh` alone.** It locks clocks *inside the container*; it neither observes nor
  constrains the host. An out-of-band idle check is still required.
- **Assuming the CPU is neutral.** Two channels, bandwidth and power.
- **Quoting a delta without a spread.**
- **Rejecting a sub-0.4% regression** as a real regression.
- **Comparing a cold arm to a warm arm** — worth 3–4× on real weights.

## Verify
- `repro/noise.py` — re-derive the floor on the box you are actually using; do not inherit 0.4%.
- `repro/drift.py` — detects clock/thermal drift across a long campaign.
- Host-side idle check (load average / CPU utilisation) from **outside** the container, immediately
  before the run.

## Sources
- `GEAK/kernel_workflow/knowledge/repro/noise.py`, `drift.py` — on-box.
- `GEAK/kernel_workflow/knowledge/amd_rdna.md` §1, §7.
