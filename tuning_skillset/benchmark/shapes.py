"""GEMM shape corpus for tuning work.

Shapes are grouped by *regime* rather than by size, because the regime — not the raw
dimensions — determines which configs win and which tuning strategy applies. See
``../tuning-core/search_strategy.md`` for the measured evidence that the compute-bound and
decode winners sit at opposite corners of the config space.

Use as a library (``for c in corpus(): ...``) or as a CLI (``python3 shapes.py --regime
decode --format csv``).

The corpus is deliberately generated rather than hard-coded from one model: it is meant to
exercise a tuning *method* across the regime space. When real serving shapes are available,
capture them (see the per-framework skills) and tune those instead — this corpus is the
fallback and the smoke test, not the target.
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass, asdict
from typing import Iterator, Literal

Regime = Literal["square", "tall_skinny", "short_fat", "k_heavy", "decode", "batch_decode"]

# dtypes worth sweeping, both rows confirmed on hardware with
# ``tuning_benchmark/tools/arch_calibrate.py`` -- each entry allocates *and*
# completes a GEMM, which are separate questions (see
# ``../tuning-core/arch_migration.md``).
#
# The FP8 split is an inversion, not a superset: gfx942 computes FNUZ and
# refuses OCP, gfx950 computes OCP and refuses FNUZ. Never move fp8 artifacts
# between the two.
#
# mxfp4/mxfp8 are gfx950-only. Probing them needs care: torch.zeros on an fp4
# tensor raises "fill_cuda not implemented" even where the matrix core is
# present, so a probe written with zeros reports the dtype missing on the only
# part that has it. Use torch.empty.
DTYPES_BY_ARCH = {
    "gfx942": ["bf16", "fp16", "fp8_e4m3_fnuz", "int8"],
    "gfx950": ["bf16", "fp16", "fp8_e4m3", "int8", "mxfp8", "mxfp4"],
}


@dataclass(frozen=True)
class Shape:
    M: int
    N: int
    K: int
    regime: str

    @property
    def flops(self) -> int:
        return 2 * self.M * self.N * self.K

    @property
    def arithmetic_intensity(self) -> float:
        """FLOPs per byte moved, assuming 2-byte operands and no cache reuse.

        Below ~100 the shape is bandwidth-bound on MI300X and should be judged against
        achievable bandwidth, not peak FLOPS.
        """
        bytes_moved = 2 * (self.M * self.K + self.K * self.N + self.M * self.N)
        return self.flops / bytes_moved

    @property
    def is_memory_bound(self) -> bool:
        return self.arithmetic_intensity < 100


def _square() -> Iterator[Shape]:
    """Compute-bound. The classic tuning target; wants the largest viable tile."""
    for n in (1024, 2048, 4096, 8192):
        yield Shape(n, n, n, "square")


def _tall_skinny() -> Iterator[Shape]:
    """Large M, small N. Common in prefill and in projections with narrow output."""
    for m in (4096, 8192, 16384):
        for n in (512, 1024, 2048):
            yield Shape(m, n, 4096, "tall_skinny")


def _short_fat() -> Iterator[Shape]:
    """Small M, large N. The transpose of tall_skinny; different tile preference."""
    for m in (512, 1024, 2048):
        for n in (4096, 8192, 16384):
            yield Shape(m, n, 4096, "short_fat")


def _k_heavy() -> Iterator[Shape]:
    """Deep reduction, narrow output -- down-projection shaped.

    Worth its own regime because split-K becomes the dominant lever here, and because the
    long reduction makes absolute numeric error grow while relative error stays flat (see
    ../tuning-core/correctness_gates.md).
    """
    for k in (8192, 16384, 32768):
        for n in (1024, 2048, 5120):
            yield Shape(1024, n, k, "k_heavy")


def _decode() -> Iterator[Shape]:
    """M=1 GEMV. Pure bandwidth. Peak-FLOPS numbers here are meaningless."""
    for n, k in ((4096, 4096), (8192, 8192), (5120, 17408), (16384, 4096)):
        yield Shape(1, n, k, "decode")


def _batch_decode() -> Iterator[Shape]:
    """Small-M ladder -- the serving regime that matters most and is most often skipped.

    M values follow the bucketing ladder recommended in search_strategy.md: tune these,
    then round live M to the nearest bucket rather than tuning every observed M.
    """
    for m in (2, 4, 8, 16, 32, 64, 128, 256):
        yield Shape(m, 8192, 8192, "batch_decode")


_GENERATORS = {
    "square": _square,
    "tall_skinny": _tall_skinny,
    "short_fat": _short_fat,
    "k_heavy": _k_heavy,
    "decode": _decode,
    "batch_decode": _batch_decode,
}


def corpus(regimes: list[str] | None = None) -> list[Shape]:
    """All shapes, or just the named regimes."""
    names = regimes or list(_GENERATORS)
    unknown = set(names) - set(_GENERATORS)
    if unknown:
        raise ValueError(f"unknown regime(s): {sorted(unknown)}; have {sorted(_GENERATORS)}")
    return [s for n in names for s in _GENERATORS[n]()]


def smoke() -> list[Shape]:
    """One shape per regime -- for validating a harness before committing to a full sweep."""
    return [next(iter(gen())) for gen in _GENERATORS.values()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--regime", action="append", choices=sorted(_GENERATORS),
                   help="restrict to a regime (repeatable); default is all")
    p.add_argument("--smoke", action="store_true", help="one shape per regime")
    p.add_argument("--format", choices=("table", "csv"), default="table")
    p.add_argument("--arch", default="gfx942", choices=sorted(DTYPES_BY_ARCH))
    args = p.parse_args()

    shapes = smoke() if args.smoke else corpus(args.regime)

    if args.format == "csv":
        print("M,N,K,regime")
        for s in shapes:
            print(f"{s.M},{s.N},{s.K},{s.regime}")
        return

    print(f"# arch={args.arch}  dtypes={','.join(DTYPES_BY_ARCH[args.arch])}")
    print(f"{'M':>7} {'N':>7} {'K':>7} {'regime':>13} {'GFLOP':>9} {'AI':>7}  bound")
    for s in shapes:
        bound = "memory" if s.is_memory_bound else "compute"
        print(f"{s.M:>7} {s.N:>7} {s.K:>7} {s.regime:>13} "
              f"{s.flops/1e9:>9.1f} {s.arithmetic_intensity:>7.1f}  {bound}")
    print(f"\n{len(shapes)} shapes")


if __name__ == "__main__":
    main()
