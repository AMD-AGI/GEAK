---
title: RDNA4 / gfx1201 — pitfalls
kind: hardware
gens: [gfx1201]
dtypes: [bf16, fp16, fp8_e4m3]
regimes: [both]
updated: 2026-10-07
sources:
  - ../../expert_skills/skills/gluon_authoring/references/pitfalls/platform-known-issues.md
---

# RDNA4 / gfx1201 — pitfalls

Details and decision rules: Gluon
[`pitfalls/platform-known-issues.md`](../../expert_skills/skills/gluon_authoring/references/pitfalls/platform-known-issues.md)
(RDNA4 PMC + `int64_strides`).

## Profiling
- Kernel-trace usually works. CDNA PMC names (`MfmaUtil`, `SQ_WAVES`, …) may be absent.
- Run `rocprofv3-avail list --pmc` on the actual box before deriving a bound class from counters.
  For rocprofv3's own list use `rocprofv3 -L` / `--list-avail`; older profiler
  generations named the operation `--list-counters`. This is a CLI rename, not
  an R9700-image defect.
- `rocprof-compute` / `omniperf` may abort with `Unsupported arch` on gfx1201. Prefer rocprofv3;
  accept a fallback profiler only when it leaves real dispatch artifacts.
- Missing MFMA% is **not** a failed profile — fall back to durations + latency table.

## Triton / graphs
- gfx1201 attention under CUDA graphs: keep `int64_strides=true` unless A/B proves otherwise
  (observed ~5× regression with `false` on one R9700 kernel).

## Numerics / ISA
- Never FNUZ. Never MX/block-scale MFMA. Never port “interleave VALU into the MFMA shadow.”
- A literal `v_mfma_*` intrinsic/inline-assembly port does not compile for gfx1201. Use WMMA.
- Native INT4 WMMA exists, but `int4_w4a16` can still name a software-dequant path. Verify
  `v_wmma_i32_16x16x{16,32}_iu4` in the emitted ISA before claiming native INT4 execution.
- Provisional Triton FA search hints: include `BLOCK_M=64`, `BLOCK_N=32`, and
  `waves_per_eu=6` in a sweep, but do not treat them as sourced defaults until
  R9700 A/B evidence supports them.
