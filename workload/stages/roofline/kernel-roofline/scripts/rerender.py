#!/usr/bin/env python3
"""Re-render the Roofline efficiency block of an existing rocprof leg, offline.

The efficiency block is a pure function of the parsed rocprof output plus the
peak-selection policy, and `run_roofline.py` keeps that rocprof output verbatim
in `<tag>_roofline_raw.txt`. So a change to the *policy* -- the equal-rate-sibling
guard in `run_roofline.py:sanity_check_peak`, a different `--compute-peak`, a
corrected `--hbm-peak-const` -- can be applied to a finished leg without touching
the GPU. No re-profiling means no chance of the numbers moving for an unrelated
reason (contention, a different microbench draw), so the delta isolates the policy
change instead of mixing it with a fresh noise sample.

Everything above the `=== Roofline efficiency` marker (the parsed summary, which
carries the bandwidth numbers) is copied through unchanged.

The peak settings below MUST match the ones the leg was originally rendered with,
except for the single one being changed -- otherwise the re-render moves two
variables and the delta means nothing.

    python3 rerender.py --src <leg_dir> --dst <new_dir> --hbm-peak-const 6851
"""
import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MARKER = "=== Roofline efficiency"

_spec = importlib.util.spec_from_file_location("run_roofline", ROOT / "run_roofline.py")
rr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rr)


def orig_names(text):
    """Kernel names from the existing efficiency block, in order.

    The raw rocprof output truncates long mangled names; the runner un-truncated
    them from pmc_kernel_top.csv, which is not kept. Reusing the names already in
    the rendered file preserves that work -- and the count check below is what
    makes the positional reuse safe.
    """
    body = text.split(MARKER, 1)[1] if MARKER in text else ""
    return [ln[len("kernel: "):].strip()
            for ln in body.splitlines() if ln.startswith("kernel: ")]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="Leg dir holding *_roofline_raw.txt.")
    ap.add_argument("--dst", required=True, type=Path, help="Destination dir for the re-render.")
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument("--hbm-peak-const", type=float, default=None)
    ap.add_argument("--compute-peak", default="auto")
    ap.add_argument("--compute-peak-const", type=float, default=None)
    ap.add_argument("--peaks", choices=["empirical", "datasheet", "both"], default="both")
    args = ap.parse_args()

    SRC, DST = args.src.resolve(), args.dst.resolve()
    ARCH, HBM_CONST, PEAKS, COMPUTE_PEAK = args.arch, args.hbm_peak_const, args.peaks, args.compute_peak
    if not SRC.is_dir():
        print(f"[rerender] --src does not exist: {SRC}", file=sys.stderr)
        return 1
    if DST == SRC:
        print("[rerender] --dst must differ from --src: re-rendering in place destroys "
              "the comparison the re-render exists to make", file=sys.stderr)
        return 1
    DST.mkdir(parents=True, exist_ok=True)
    n_ok = n_sub = 0
    for raw in sorted(SRC.glob("*_roofline_raw.txt")):
        tag = raw.name[: -len("_roofline_raw.txt")]
        rendered = SRC / f"{tag}_roofline.txt"
        if not rendered.is_file():
            print(f"  SKIP {tag}: no rendered file")
            continue
        text = rendered.read_text(errors="replace")
        blocks = rr.parse_kernel_blocks(raw.read_text(errors="replace"))
        names = [b[0] for b in blocks]
        keep = orig_names(text)
        if len(keep) == len(names):
            names = keep
        else:
            print(f"  WARN {tag}: {len(names)} parsed blocks vs {len(keep)} "
                  f"rendered kernels -- keeping raw (truncated) names")
        eff = rr.compute_roofline_efficiency(
            names, [b[1] for b in blocks], [b[2] for b in blocks],
            COMPUTE_PEAK, HBM_CONST, args.compute_peak_const, PEAKS, ARCH)
        head = text.split(MARKER, 1)[0] if MARKER in text else text
        (DST / f"{tag}_roofline.txt").write_text(head + eff.lstrip("\n"))
        # Carry the raw file over so the new dir is self-contained and can be
        # re-rendered again after the next policy change.
        (DST / raw.name).write_text(raw.read_text(errors="replace"))
        n_ok += 1
        n_sub += eff.count(",sub=")
    print(f"re-rendered {n_ok} cases -> {DST}  "
          f"({n_sub} kernel blocks had a peak substituted)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
