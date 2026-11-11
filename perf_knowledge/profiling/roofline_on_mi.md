---
title: profiling — building and reading a roofline on MI300X / MI350X
kind: technique
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp4_e2m1, fp6, int8]
updated: 2026-10-16
sources:
  - https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/how-to/profile/mode.html
  - https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi300.html
---

# Roofline on MI300X / MI350X

Use a roofline to compare a kernel's work and traffic with stated capacities.
For matched work `W`, bytes `Q` and time `T`, plot `W/T` against `W/Q`; the simple
model is `min(P, B*W/Q)`. Arithmetic intensity selects the tighter modeled roof,
not the measured bottleneck. Launch cost, dependencies and other resources can
leave a kernel below both roofs.

## Collect and interpret

Use the task's qualified profiler route and inspect its installed `--help`.
For versions supporting the older roof-only interface:

```bash
rocprof-compute profile --name myrun --roof-only -- python bench.py
rocprof-compute analyze -p workloads/myrun/MI300X/ --roofline-data-type FP16
```

`--roof-only` can run calibration as well as application counters. It is not a
cheap counter-only switch; verify that calibration is supported on the actual
version/device. See [the workflow](rocprof_compute_workflow.md) and the
[version-sensitive collection notes](../skills/kernel-hardware-evidence/references/collection.md).
Use a fresh collection directory; existing data may be reused. Analyze retained
captures without treating them as a new measurement.

Read `roofline.csv`, application fields and their definitions. Select the actual
arithmetic dtype/unit; operand storage dtype alone may not identify it. Check
that the plotted field matches the selected dtype. Missing capacity is unknown,
not another dtype's peak. A plot label or successful command does not validate
its arithmetic or collection coverage.

## State the denominator

The [MI300 tables](../hardware/cdna3_mi300/peak_tables.md) and
[MI350 tables](../hardware/cdna4_mi350/peak_tables.md) provide specification context.
Match the actual product, dense/sparse mode, dtype and clock/power conditions.
Label measured calibration as an empirical comparator for its benchmark conditions;
it is neither a universal ceiling nor proof that an operator can reach it. Report
specification fractions when their operands are supported, with those assumptions.

Match `Q` to the memory level and byte meaning: logical footprint, documented
request bytes, or supported physical transfers. Cache reuse can make logical-byte
rates exceed external-memory bandwidth. Include all work/time in the stated
boundary, and keep full-call elapsed time separate from individual or summed
concurrent dispatches. Profiled durations do not replace accepted unprofiled timing.

A high fraction motivates checking avoidable work, traffic or materialization;
it does not end optimization. A low fraction motivates a targeted diagnostic,
not an automatic occupancy verdict. After a rewrite, recompute changed work/traffic
and validate full-operation correctness and latency. Lower FLOP/s can accompany
a useful algorithm that performs less work.

For detailed accounting, use the [roofline reference](../skills/kernel-hardware-evidence/references/roofline.md).
