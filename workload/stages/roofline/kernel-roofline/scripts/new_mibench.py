#!/usr/bin/env python3
"""Run the rocprof-compute >=3.6.0 roofline microbenchmark standalone, under ROCm 7.2.0.

Why this exists rather than `apt install` of a newer rocprof-compute:

  The bf16 empirical peak this workspace has been fighting is entirely a MICROBENCHMARK bug,
  not a counter-collection bug. In 3.4.0/3.5.0 the gfx950 MFMA benchmark issues the CDNA3
  instructions `v_mfma_f32_32x32x8_f16` / `v_mfma_f32_32x32x8bf16_1k` and scores them at
  16384 FLOP/iter. gfx950 is CDNA4 and has `v_mfma_f32_32x32x16_{f16,bf16}` at 32768
  FLOP/iter. 3.6.0 ("Fixed roofline benchmark MFMA FP16/BF16/INT8 peaks for MI 350") gives
  gfx950 its own bench class using the CDNA4 instructions -- see
  src/roofline/benchmark/gfx9/benchmark_gfx950.py.

  3.5.0 is what makes this separable: it moved the microbench out of the prebuilt
  rocm-amdgpu-bench binaries (`/opt/rocm/bin/roofline-ubuntu22_04-rocm7`) and into Python
  kernels compiled at runtime through HIPRTC. The only native dependencies are
  libamdhip64.so and libhiprtc.so, both of which ROCm 7.2.0 has. rocprofiler-sdk -- the part
  that WOULD force a whole-ROCm upgrade -- is not on this path at all.

  Upgrading only the denominator also keeps every already-published number joinable: the
  counters in geak_out/ and the GLM geak_out5/ leg were collected by the patched 3.4.0 and
  are untouched by this.

cache_sizes is supplied by the caller in-tree (from the SoC spec); the values below are
gfx950's, copied verbatim from the 3.5.0 `cache_sizes` table. They feed only the L1/L2
bandwidth rows, never the MFMA peaks that are the point of this run.

`--bench-src` must point at the `src/` of a rocprof-compute >= 3.6.0 checkout (upstream
development now lives in `ROCm/rocm-systems` under `projects/rocprofiler-compute`; the
standalone `ROCm/rocprofiler-compute` repo is frozen at 3.4.0 on every branch). In that
checkout `src/vendored/` is empty -- pyyaml is vendored at build time -- so a shim module
forwarding to the container's own pyyaml is needed before this will import.

The CSV this writes is exactly what `ROOFLINE_MIBENCH_CACHE` consumes, and what
`repeak.py --peaks-csv` rescores an already-collected leg against.
"""
import argparse
import json
import sys
from pathlib import Path

GFX950_CACHE_SIZES = {"L1": 32 * 1024, "L2": 4 * 1024 * 1024, "MALL": 64 * 1024 * 1024}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench-src", required=True, type=Path,
                    help="src/ of a rocprof-compute >= 3.6.0 checkout.")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sys.path.insert(0, str(a.bench_src.resolve()))

    from roofline.run_benchmark import load_bench
    from utils.hip_interface import hipGetDeviceProperties

    props = hipGetDeviceProperties(a.device)
    arch = props.gcnArchName.split(":", 1)[0]
    print(f"[new_mibench] device={a.device} arch={arch} name={props.name}", flush=True)

    bench = load_bench(a.device, GFX950_CACHE_SIZES)
    print(f"[new_mibench] bench class = {type(bench).__name__}", flush=True)
    print(f"[new_mibench] matrix_ops  = {json.dumps(bench.matrix_ops)}", flush=True)

    metrics = bench.run_benchmark(a.device)
    bench.dump_csv(metrics, a.out)
    print(f"[new_mibench] wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
