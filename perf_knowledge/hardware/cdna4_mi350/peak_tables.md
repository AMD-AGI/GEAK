---
title: MI350X and MI355X specification peaks
kind: hardware
gens: [gfx950]
dtypes: [fp64, fp32, bf16, fp16, fp8_e4m3, fp8_e5m2, fp4_e2m1]
regimes: [both]
updated: 2026-10-16
---

# MI350X and MI355X specification peaks

Use the product-specific YAML blocks in the [canonical peak table](../../../e2e_workflow/knowledge/analysis_skills/roofline/peaks.md).
Both products use gfx950, but their published per-GPU compute peaks differ.
Identify the physical product before choosing a denominator.

| Dense specification per GPU | MI350X | MI355X |
|---|---:|---:|
| FP64 matrix/vector | 72.1 TFLOP/s | 78.6 TFLOP/s |
| FP32 matrix/vector | 144.2 TFLOP/s | 157.3 TFLOP/s |
| FP16/BF16 matrix | 2.3 PFLOP/s | 2.5 PFLOP/s |
| OCP FP8 matrix | 4.6 PFLOP/s | 5.0 PFLOP/s |
| MXFP4 matrix | 9.2 PFLOP/s | 10.1 PFLOP/s |
| HBM3E capacity / pin bandwidth | 288 GB / 8 TB/s | 288 GB / 8 TB/s |
| Peak engine clock | 2200 MHz | 2400 MHz |

These are AMD's rounded dense specifications, excluding structured sparsity.
Match dtype, instruction mode, memory interface and measurement boundary; an
arbitrary FP4 implementation does not inherit the MXFP4 rate. The conditional
FP16/BF16 ridge is 287.5 FLOP/byte for MI350X and 312.5 for MI355X.
A specification ratio is not measured utilization or guaranteed headroom. Do not
transfer a sustained fraction from another product or use an end-to-end benchmark
as a kernel compute ceiling. See [clocks and power](clocks_power.md).

## Sources

- [Canonical product YAML and source-check dates](../../../e2e_workflow/knowledge/analysis_skills/roofline/peaks.md)
- [AMD MI350X](https://www.amd.com/en/products/accelerators/instinct/mi350/mi350x.html)
- [AMD MI355X](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html)
