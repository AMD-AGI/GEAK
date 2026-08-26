---
name: tuning-hip
description: Tune hand-written HIP kernels on AMD Instinct — choose launch geometry against the wave64/CU/LDS limits of the target, use rocprofv3 to get ground-truth kernel timings and occupancy, and confirm which kernel actually ran. Also the profiler reference for verifying engagement on any backend.
---

# Tuning HIP kernels

Read `../tuning-core/SKILL.md` first.

Raw HIP is the case with no tuner: no config list, no solution table, no CSV. You choose
launch geometry and compile flags yourself, and the feedback loop is the profiler. That makes
this skill do double duty — **§4 is the profiler reference the other skills point at** for
ground-truth verification, because `rocprofv3` sees the kernel that ran regardless of which
language produced it.

Measured on gfx942 / MI300X, ROCm 7.2.2. Toolchain present in both images: `hipcc`,
`rocprofv3`, `rocprof`, `rocprofv2`, `rocm-smi`, `amd-smi`, `hipconfig`.

## 1. Know the target's constants before choosing geometry

The tunable launch parameters are block size, grid size, LDS per block, and register
pressure. They are constrained by the device, so read the device rather than guessing:

```bash
rocminfo | grep -E "Name:|Compute Unit|Wavefront|LDS"
hipconfig --platform && hipconfig --version
```

The numbers that shape every decision, both parts measured:

| property | gfx942 / MI300X | gfx950 / MI355X | consequence |
| --- | --- | --- | --- |
| wavefront | **64** | **64** | a "warp" is 64 threads, not 32. Block sizes should be multiples of 64 |
| CUs | 304 | 256 | grids below the CU count leave the device partly idle |
| LDS per workgroup | 64 KB | **160 KB** | hard cap on staged tile size |
| max threads/block | 1024 | 1024 | = 16 waves |
| dispatch floor | 42 us | **17 us** | below this you are timing the launch, not the kernel |

The wave64 point is the one that most often survives from other platforms as a bug: a
256-thread block is 4 waves here, not 8, so occupancy and per-thread work both differ from
the same code on a 32-wide device.

The two bold cells are the ones that bite when moving *between these two parts*, because both
change in the direction that makes stale code look fine. A 64 KB LDS assumption on gfx950
does not overflow, it just declines tiles that would have fit; a 42 us floor does not fail,
it just labels fast kernels unworthy of tuning. See §5.

The same arithmetic that prunes a Triton config space prunes a HIP launch space, and for the
same reasons: LDS budget, threads-vs-output-elements, per-thread register state. The
predicate is in `../tuning-core/search_strategy.md`; it is worth applying by hand before you
compile variants.

## 2. Compile flags are part of the search space

```bash
hipcc --offload-arch=gfx942 -O3 kernel.hip -o kernel
```

- **`--offload-arch`** — always set explicitly. A default-arch build can produce a binary that
  runs but is not compiled for your device's MFMA generation.
- **`-mcumode` / launch bounds** — `__launch_bounds__(N)` caps registers per thread so more
  waves fit per CU. This is the HIP-level equivalent of the `waves_per_eu` knob in Triton and
  FlyDSL: it trades per-thread registers for occupancy. It helps latency-bound kernels and
  hurts register-hungry ones — measure, do not assume.
- **`-Rpass-analysis=kernel-resource-usage`** — reports registers, LDS and occupancy at
  compile time. Cheaper than a profiler run for pruning obviously-doomed variants.

## 3. Measure like everything else

HIP gives you no autotuner, which makes it easy to fall into the two errors
`../tuning-core/measurement.md` exists to prevent:

- **Synchronize.** Kernel launches are asynchronous. Timing without `hipDeviceSynchronize()`
  or events measures enqueue cost. The tell is a result above hardware peak — measured
  4363 TFLOPS on a gfx942 device whose peak is ~1300.
- **Repeat.** One timing is not a measurement. On an unchanged kernel, run-to-run spread was
  9.3%; small shapes reached 36% before raising repeat counts.

**The synchronize trap is shape-dependent, which is why it survives review.** Re-measured on
gfx950 with `tuning_benchmark/tools/hip_verify.py`, the same unsynchronised loop over the same
kernel:

| shape | unsynchronised | synchronised | inflation |
| --- | --- | --- | --- |
| 4096³ | 7.378 ms | 7.396 ms | none |
| 256³ | 0.008 ms | 0.019 ms | **2.4×** |

Once the kernel outlasts the enqueue, the queue saturates and the async loop reports the honest
time — so a large shape will not reveal the bug, and a harness validated only on a large shape
ships it. The inflation appears exactly where kernels are short, which is decode, which is
where most serving-side tuning happens. Note also that on gfx950 the inflated figure was still
well under peak (4.2 TFLOPS on a naive kernel), so "check the result against peak" catches this
on a fast kernel and misses it on a slow one. Check by adding the synchronize and seeing whether
the number moves.

Prefer HIP events over wall-clock, and take the median of independent samples. Compare any
claimed gain against the measured spread — `../tuning-triton/SKILL.md` §2 has a worked case
where one variant's +7.7% was real and another's +4.4% was not. And on gfx950, interleave A/B
comparisons rather than running them back to back: `../tuning-core/measurement.md` Rule 6b.

## 4. rocprofv3 — ground truth for what actually ran

This is the profiler section the other skills reference.

```bash
rocprofv3 --kernel-trace --stats -f csv -d ./prof_out -- python3 workload.py
```

Writes `<pid>_kernel_stats.csv`:

```
"Name","Calls","TotalDurationNs","AverageNs","Percentage","MinNs","MaxNs","StdDev"
"Cijk_Ailk_Bljk_BBS_BH_UserArgs_MT256x224x64_MI16x16x1_SN_LDSB1_...",20,...
"void at::native::...distribution_elementwise_grid_stride_kernel<float, 4, ...",...
"__amd_rocclr_fillBufferAligned",1,10183,...
```

Every kernel that executed, by name, with call count and duration. That is the strongest
engagement evidence available, and it is backend-agnostic:

- **Verifying a tuned selection engaged.** The `MT256x224x64` in that trace is the same
  macro-tile the hipBLASLt bench selected as the winner for this shape. Seeing the tuned
  kernel's name in the trace — and the untuned one absent — is proof, where a wall-clock
  improvement is only evidence.
- **Finding the real target.** `Calls` and `Percentage` rank kernels by cost. Tune the top of
  that list, not the kernel you assumed mattered.
- **Catching what you did not intend.** Buffer fills, layout conversions and elementwise
  casts show up here. A transpose that costs 20% of runtime is invisible in a GEMM benchmark
  and obvious in a kernel trace.

Useful companions: `--hip-trace` for API-level calls, `--kernel-include-regex` to narrow a
noisy trace, `--pmc` for hardware counters (occupancy, cache hit rates) when you need to know
*why* a kernel is slow rather than which one is. `--stats` is required for the stats CSV —
unlike older rocprof versions, there are no default kernel stats.

Profile a short run. A full serving workload produces traces too large to read, and the
ranking stabilizes quickly.

## 5. Architecture portability

A HIP kernel compiled for gfx942 does not run on gfx950 and vice versa; `--offload-arch`
makes that explicit, which is the good case — it fails at build time rather than silently.

What does *not* fail loudly is tuned launch geometry. Block sizes, LDS staging and split
factors balanced for 304 CUs are unbalanced on 256. Carry the *kernel* across architectures;
re-tune the *geometry*. Where the arches genuinely diverge — gfx950's microscaled dtypes and
its transposed LDS reads — there is no gfx942 configuration to carry over at all.

FP8 needs its own care, and it is an inversion rather than an extension: gfx942 computes FNUZ
and refuses OCP, gfx950 computes OCP and refuses FNUZ. Identical 8-bit layout, different
exponent bias, so a mismatch corrupts numerics silently rather than failing. Details in
`../tuning-core/correctness_gates.md`.

When a device read fails, raise. A helper that returns 304 CUs or `"gfx942"` as a fallback
turns an outage into a wrong answer that propagates into every launch geometry it touches,
and nothing downstream can distinguish it from a real reading.

## Checklist

- [ ] device constants read from `rocminfo`, not assumed — wave64, CU count, LDS size
- [ ] those reads raise on failure rather than falling back to a plausible constant
- [ ] block sizes are multiples of 64; grid large enough to cover the CUs
- [ ] `--offload-arch` set explicitly
- [ ] resource usage checked at compile time before benchmarking variants
- [ ] timings synchronized, repeated, and compared against the measured spread
- [ ] `rocprofv3 --kernel-trace --stats` confirms the intended kernel ran and ranks the real cost
- [ ] geometry re-tuned per architecture; FP8 dialect matched to the target
