---
title: profiling — measured roofline efficiency for a single kernel
kind: technique
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp4_e2m1, fp6, int8]
updated: 2026-09-20
sources:
  - https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/how-to/profile/mode.html
  - https://github.com/ROCm/rocm-systems/commit/aa5dfb98f96ab080de97259df64ba0d36f6796b4
  - https://rocm.docs.amd.com/en/latest/about/release-notes.html
---

# Measured roofline efficiency for a single kernel

## TL;DR
[`roofline_on_mi.md`](roofline_on_mi.md) tells you how to *draw* a roofline and read a point off it.
This doc is the next step: collapse that picture into **one number per kernel** —

```
attainable   = min(Peak_Compute[dtype], AI_HBM × Peak_BW)
Roofline Eff = Achieved_FLOPs / attainable
```

— so you can rank kernels and say "this one is at 0.31 of its ceiling, that one is at 0.93, stop
tuning the second." It needs a **GPU and a runnable benchmark command**: unlike the analytic
`e2e_workflow` roofline skill, every term here is measured. Budget 3–5 min of microbenchmark on the
first run. Driver: [`../../kernel_workflow/scripts/run_roofline.py`](../../kernel_workflow/scripts/run_roofline.py).

## The recipe
```bash
python3 kernel_workflow/scripts/run_roofline.py \
    --workdir /path/to/kernel --cmd "python bench.py" \
    --name mykernel --compute-peak auto --peaks both
```
Two `rocprof-compute` passes over the same command: `profile --roof-only` for the empirical ceilings
(`roofline.csv`) and `profile` + `analyze` for the kernel's achieved FLOP/s, HBM bytes and AI. The
efficiency is then arithmetic — do it by hand from `analyze -b 4` output if the script is unavailable.

**Which compute peak.** `attainable` is only meaningful against the peak for the dtype the kernel's
MFMA instructions actually use — an FP8 GEMM scored against the FP32 roof reads ~3% efficient and
means nothing. Select by the kernel's MFMA dtype, not by the tensor dtype of its inputs, and record
which peak you used next to the number. gfx950 F6F4 has no separate microbench entry below 3.6.0;
`2 × FP8_empirical` is a defensible stand-in, but label it as an estimate.

## Report both denominators, always
Empirical and datasheet peaks disagree, sometimes by 4×, and **neither one is authoritative on its
own**. Emit both columns:

| | what it is | fails when |
|---|---|---|
| empirical | what the microbench got on *this* box today | microbench bug, or it fails to saturate the unit |
| datasheet | the spec ceiling | a real kernel can't reach it; ignores your clocks/power cap |

An efficiency **above 100% against the empirical peak is not a good kernel** — it is proof the
denominator is wrong. Treat it as a failed measurement, not a result.

## Version requirement: rocprof-compute ≥ 3.6.0 on gfx950
Below 3.6.0 the microbench emits CDNA3 `v_mfma_f32_32x32x8_{f16,bf16}` on a CDNA4 part and scores it
at 16384 FLOP/iter, where `v_mfma_f32_32x32x16_*` does 32768 — so the **BF16/FP16/INT8 empirical
peaks read ~2–4× low** and every BF16 efficiency computed against them reads correspondingly high
(196% against a bad ceiling was the case that surfaced this). 3.6.0 (ROCm 7.13.0) fixes it:
*"Fixed roofline benchmark MFMA FP16/BF16/INT8 peaks for MI 350"*. Measured on gfx950, GFLOP/s:

| dtype | < 3.6.0 | 3.6.0 | ratio |
|---|---|---|---|
| bf16 | 614 663 | 2 435 517 | 3.96× |
| int8 | 1 222 436 | 4 854 023 | 3.97× |
| fp16 | 1 227 048 | 2 162 758 | 1.76× |
| fp4 / F6F4 | *absent* | 9 680 756 | — |
| fp8, fp32, fp64 | — | — | **0.99–1.00×** |

FP8/FP32/FP64 not moving is what makes this a *fix* and not microbenchmark drift. gfx942 is CDNA3 and
was never affected. FP16 is still ~11% under BF16 after the fix because that microbench does not
saturate the unit — the two are equal in hardware, so **BF16 == FP16 remains the standing sanity
check** on any empirical peak table.

**Below 3.6.0 on gfx95x: use the datasheet peaks instead** (gfx950: HBM 8.0 TB/s, BF16/FP16 2.5 PF,
FP8 5 PF, F6F4 10 PF — [`../hardware/cdna4_mi350/peak_tables.md`](../hardware/cdna4_mi350/peak_tables.md)).
As a last-resort fallback when you cannot upgrade, flag any dtype whose empirical peak is not within
~10% of its equal-rate sibling (BF16 vs FP16) and refuse to rank on it.

## Freeze the denominator across an A/B
The microbench is re-run per invocation and **drifts ~17% run-to-run on the pre-3.6.0 HBM roof**, so
two arms profiled separately are scored against two different ceilings and the difference between
them is partly noise in the denominator. Measure the peaks once, then reuse that `roofline.csv` for
every arm of the comparison (`run_roofline.py` does this via `ROOFLINE_MIBENCH_CACHE`). The 3.6.0
bench is much steadier — 1.86% worst case, 0.02% on BF16 — but pinning is still the right default,
and it also buys back the 3–5 min per run.

Changing the peaks does **not** require re-profiling: `Value` and `Peak` are two columns of the same
`analyze` table, so a corrected ceiling can be applied offline. Re-running the profile only mixes the
denominator fix with a fresh sample of timing noise.

## HBM read bytes read 2× low on gfx950 (numerator)
Independent of the peak bug, and it moves the other term. `FETCH_SIZE` was defined using `TCC_BUBBLE`
as a 128B-read counter, but that counter is **0 on gfx950**, so 128B reads were billed as 64B and
reads came out exactly half. Fixed upstream in `aa5dfb98` (in rocprof-compute ≥ 3.6.0), which
switches gfx950 to `TCC_EA0_RDREQ_128B`.

**The same definition lives in `rocprofiler-sdk`**, so `rocprofv3 --pmc FETCH_SIZE` is affected too,
and that side is only fixed in ROCm 10.0.0 — a container can have a patched rocprof-compute and an
unpatched `rocprofv3` at the same time. Tell-tale within one profile: the SoL and L2 panels report
HBM read traffic that differs by ~2×. Cross-check by hand-summing
`TCC_EA0_RDREQ_{32B,64B,128B}` before trusting any `hbm_util` on this part.

## Pitfalls
- Scoring against the empirical peak without checking the tool version — the failure is silent and
  the resulting number looks plausible.
- Reporting >100% efficiency as a result rather than as a broken denominator.
- Comparing two arms scored against two separately-measured microbench runs.
- Taking a single `hbm_util` on gfx950 at face value without the `FETCH_SIZE` check above.

## Verify
`roofline.csv` exists and its BF16 and FP16 compute peaks agree within ~10%; the empirical compute
roof is **both** below the datasheet peak and within a sane fraction of it (a one-sided "not above"
check passes the 614663-vs-2.5e15 bug); the reported efficiency is in `(0, 1]`; the dtype of the peak
used is recorded alongside the number.

## Sources
- `--roof-only`, `roofline.csv`, `analyze` peak columns: ROCm Compute Profiler profile-mode docs.
- MFMA FP16/BF16/INT8 peak fix for MI350: rocprof-compute 3.6.0 changelog (ROCm 7.13.0).
- `FETCH_SIZE` / `TCC_BUBBLE` gfx950 read-byte fix: ROCm/rocm-systems commit `aa5dfb98`; the
  `rocprofiler-sdk` copy of the same definition, fixed in ROCm 10.0.0.
- Peak tables and ridge points: perf_knowledge hardware peak tables (MI300/MI350).
- Measured gfx950 microbench numbers: in-house A/B of rocprof-compute 3.4.0 vs 3.6.0 on MI355X.
