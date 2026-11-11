---
title: Gluon GEMM examples on gfx950
kind: language
gens: [gfx950]
dtypes: [fp16, fp8_e5m2, fp4_e2m1]
regimes: [prefill, training, both]
status: sota
updated: 2026-10-16
---

# Gluon GEMM examples on gfx950

AMD's `gfx950-gluon-tutorials` demonstrates explicit LDS layouts, software
pipelines, register budgeting and MFMA scheduling. Use its revisions as examples
to inspect, not a required sequence or attainable ceiling for another shape.

For the complete v0→v9 progression, follow the [tutorial's worked steps](https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html)
and [versioned examples](https://github.com/ROCm/gfx950-gluon-tutorials).
The sequence introduces LDS layouts and `ds_read_tr`, compares explicit two-/three-stage
pipelines, checks live-register cost, then adds MFMA scheduling and final-loop
cleanup. The register budget belongs to that target/mode; it is not a universal
512-register limit or proof of achieved occupancy.

The tutorial's reported configurations include `256×256×64` FP16 tiles,
`256×256×128` BF8 tiles and `256×256×256` MXFP4 tiles. Low-precision paths require
matching operand encodings, scale layout and accumulation semantics; gfx950
scaled-MFMA support does not transfer to gfx942.

In the tutorial checkout, for builds supporting these switches:

```bash
cd kernels/gemm/a16w16
python bench.py --version 0 --K 8192 --dtype fp16
TRITON_ENABLE_LLIR_SCHED=1 TRITON_ENABLE_AMDGCN_AS=1 \
python bench.py --version 9 --K 8192 --dtype fp16
```

The source reports 1489 TFLOP/s for FP16 at `4096×4096×8192`, 3257 for BF8 at
`4096×4096×16384`, and 5255 for MXFP4 at `4096×4096×32768` on its MI355/ROCm 7.0
configuration. These are published results, not measurements from this cleanup.
Its “MFMA efficiency” percentages need their original work/cycle denominator;
do not relabel them as fractions of a datasheet peak or universal practical limits.

Compare layouts, load-to-use distance, instruction scheduling and resource cost
in the actual lowering. Retain full-operation correctness and unprofiled latency;
large square-ish GEMM results do not determine skinny/decode performance.

## Sources

- [AMD Gluon tutorial](https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html)
- [Tutorial code](https://github.com/ROCm/gfx950-gluon-tutorials)
- [CDNA4 scheduling example](https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html)
