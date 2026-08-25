"""Measure one or more GEMM shapes with the discipline described in ../tuning-core/.

This is a reference harness, not a tuner. It exists so that every backend in this skillset
is measured the same way -- synchronized, warmed up, median-of-repeats, with a relative
correctness gate and a peak-plausibility check. Backend-specific tuners are covered in the
per-language skills; this is what you compare their winners against.

    python3 run_case.py --smoke
    python3 run_case.py --regime decode --dtype bf16
    python3 run_case.py --M 4096 --N 4096 --K 4096 --repeats 7

Always pin an idle GPU first:
    rocm-smi --showuse
    export HIP_VISIBLE_DEVICES=<idle gpu>
"""

from __future__ import annotations

import argparse
import statistics
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from shapes import Shape, corpus, smoke  # noqa: E402

# Rough bf16 dense peaks, TFLOPS. Used only as an implausibility check: a measurement
# above peak means a broken harness (usually a missing synchronize), not a fast kernel.
PEAK_TFLOPS = {"gfx942": 1300.0, "gfx950": 2500.0}

DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def arch() -> str:
    try:
        return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    except Exception:
        return "unknown"


def correctness(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    """Return (err_ratio, snr_db).

    err_ratio is the fraction of elements outside tolerance -- the convention used by the
    AMD tuning tools, gated at < 0.05. Absolute error is deliberately NOT reported: it
    grows with K on correct kernels and is useless as a gate. See
    ../tuning-core/correctness_gates.md for the measured demonstration.
    """
    out32, ref32 = out.float(), ref.float()
    mismatched = ~torch.isclose(out32, ref32, rtol=1e-2, atol=1e-2)
    err_ratio = mismatched.sum().item() / out32.numel()
    err_norm = (out32 - ref32).norm()
    snr_db = float("inf") if err_norm == 0 else (20 * torch.log10(ref32.norm() / err_norm)).item()
    return err_ratio, snr_db


def bench(shape: Shape, dtype: torch.dtype, repeats: int, warmup: int, rep: int):
    import triton.testing

    a = torch.randn((shape.M, shape.K), device="cuda", dtype=dtype)
    b = torch.randn((shape.K, shape.N), device="cuda", dtype=dtype)
    fn = lambda: torch.mm(a, b)  # noqa: E731

    out = fn()
    ref = a.float() @ b.float()
    err_ratio, snr_db = correctness(out, ref)

    # median of independent do_bench calls; a single call is one sample and the
    # run-to-run spread on a shared box is several percent.
    samples = [triton.testing.do_bench(fn, warmup=warmup, rep=rep) for _ in range(repeats)]
    ms = statistics.median(samples)
    spread = (max(samples) - min(samples)) / ms * 100 if ms else 0.0
    return ms, spread, err_ratio, snr_db


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--regime", action="append")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--M", type=int)
    p.add_argument("--N", type=int)
    p.add_argument("--K", type=int)
    p.add_argument("--dtype", default="bf16", choices=sorted(DTYPES))
    p.add_argument("--repeats", type=int, default=5, help="independent do_bench samples")
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--rep", type=int, default=100)
    p.add_argument("--gate", type=float, default=0.05, help="max acceptable err_ratio")
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("no GPU visible -- check HIP_VISIBLE_DEVICES and container device flags")

    if args.M and args.N and args.K:
        shapes = [Shape(args.M, args.N, args.K, "custom")]
    elif args.smoke:
        shapes = smoke()
    else:
        shapes = corpus(args.regime)

    a = arch()
    peak = PEAK_TFLOPS.get(a)
    dtype = DTYPES[args.dtype]
    print(f"# arch={a} dtype={args.dtype} repeats={args.repeats} gate=err_ratio<{args.gate}")
    print(f"{'M':>6} {'N':>6} {'K':>6} {'regime':>13} {'ms':>8} {'TFLOPS':>8} "
          f"{'GB/s':>8} {'spread%':>8} {'err':>7} {'SNR':>6}  status")

    failures = 0
    for s in shapes:
        try:
            ms, spread, err_ratio, snr_db = bench(s, dtype, args.repeats, args.warmup, args.rep)
        except Exception as exc:  # OOM or unsupported dtype for this shape
            print(f"{s.M:>6} {s.N:>6} {s.K:>6} {s.regime:>13} {'--':>8} skipped: {type(exc).__name__}")
            continue

        tflops = s.flops / (ms * 1e-3) / 1e12
        bytes_moved = 2 * (s.M * s.K + s.K * s.N + s.M * s.N)
        gbps = bytes_moved / (ms * 1e-3) / 1e9

        status = []
        if err_ratio >= args.gate:
            status.append("FAIL-CORRECTNESS")
            failures += 1
        if peak and tflops > peak:
            status.append("IMPLAUSIBLE>peak")
            failures += 1
        if spread > 10:
            status.append("NOISY")
        # For memory-bound shapes TFLOPS is the wrong yardstick; flag so it is not misread.
        if s.is_memory_bound:
            status.append("mem-bound:judge-GB/s")

        print(f"{s.M:>6} {s.N:>6} {s.K:>6} {s.regime:>13} {ms:>8.3f} {tflops:>8.1f} "
              f"{gbps:>8.1f} {spread:>8.1f} {err_ratio:>7.4f} {snr_db:>6.1f}  {' '.join(status) or 'ok'}")

    if failures:
        sys.exit(f"\n{failures} case(s) failed a gate")


if __name__ == "__main__":
    main()
