# FlyDSL-capable A16W16 GEMM

This optional case comes from the completed `_gemm_a16_w16_kernel` MI300X
campaign. Unlike the per-token example, its frozen baseline already executes a
real FlyDSL GEMM path when FlyDSL is available, so its optimization opportunity
does not depend on an optional-import failure.

Director validation accepted `avo-v3` at **1.18613x geomean** with correctness
passing and **0.42% CV**:

- c2: 0.1429 ms → 0.1326 ms (1.07768x)
- c32: 1.7594 ms → 1.5511 ms (1.13429x)
- c64: 4.2141 ms → 3.0869 ms (1.36516x)

The final result is hybrid rather than pure FlyDSL: c2 remains on the retuned
FlyDSL path, while c32/c64 route to a retuned Triton kernel. This is intentional
source-level dispatch optimization and is documented here to avoid overstating
the result as a pure FlyDSL backend win.

- `kernel/`: portable true-baseline task package and golden reference
- `reference/final_patch.diff`: cumulative accepted patch through `avo-v3`
- `reference/director_validation.json`: independent validation evidence
- `reference/trajectory.json`: compact campaign lineage
- `handoff.json`: short Team + diverse example configuration

Validate configuration from the repository root:

```bash
python3 examples/avo-examples/run_example.py flydsl-gemm-a16-w16 --dry-run
```

GPU execution requires `AVO_FLYDSL_ENV` to name the FlyDSL environment script.
