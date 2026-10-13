#!/usr/bin/env python3
"""TEMPLATE: run exactly ONE case of ONE unit test under rocprof-compute.

This is a worked example, not a general driver. The body below is bound to the GLM-5.2
UT harness (its `load_ut` / `_make_args` / `_BASE`/`_CAND` / `h.flush_cache` surface);
adapt it to your own UT. What must survive the adaptation is the three ideas in this
docstring -- they are the reason the file exists.

**One shape per command.** `run_roofline.py` profiles a *command* and reports one roofline
block per device kernel, aggregated over the whole run. A command that sweeps shapes
produces one blended block per kernel and no per-(kernel, shape) roofline at all. So the
command has to launch exactly one shape.

**The arm guard belongs here too.** Whichever arm you meant to run, the failure mode is a
wrong number that looks exactly like a right one -- a candidate sneaking into the reference
leg, or a baseline quietly standing in for the candidate and reporting "no regression".
Resolve the callable's `__code__.co_filename` and refuse to run if it does not sit under the
*selected* arm's tree -- `__file__` is empty for a `getattr`'d function and checking it
passes everything. Verify the guard *discriminates* (point it at the other arm and confirm
it rejects), not just that it passes.

`--arm` exists because [combination.md §6](../references/combination.md#6-the-only-experiment-that-adjudicates)
makes the two-implementation natural experiment -- the same kernel, baseline and candidate,
through both legs -- the only construction that settles which engine is wrong. A runner that
can only reach `baseline_src/` cannot perform it.

Two further choices that materially change the AI rocprof reports, both made to match the
TraceLens measured leg rather than to flatter either side:

  * `flush_cache` every iteration. rocprof's `AI vs HBM` counts traffic that actually reached
    HBM, so a cache-warm loop reports a *smaller* byte count and therefore a *higher* AI --
    against a TraceLens byte model that assumes a cold read. Flushing keeps the two byte
    counts talking about the same thing. The flush kernel lands in its own roofline block
    (its own kernel name) and is read separately, never summed into the seam's.
  * Few iterations. rocprof replays the kernel once per counter set; 30 iterations buys no
    accuracy here and costs wall clock linearly.

Usage (this whole string is what gets passed to run_roofline.py --cmd):
    python3 case_runner_template.py --harness <tl_leg.py> \
        --ut-root <dir> --task <task> --case dec_bs3 --arm baseline --iters 5
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import sys
from pathlib import Path


def _tlr(path: Path):
    """Import the TL leg's module for its UT loader + baseline-only guard (no GPU work).

    Importing the *same* module the TraceLens leg uses is the point: it makes the two legs
    share one loader, one seed, and one guard, so a divergence between them cannot come
    from the harness.
    """
    spec = importlib.util.spec_from_file_location("tlr", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"--harness is not an importable module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: Where each arm's source is expected to live, as a path component. The guard is a
#: containment test on the resolved code object, so these must be the real directory names
#: the UT loads from -- not labels.
_ARM_DIRS = {"baseline": ("baseline_src",), "candidate": ("kernel_src", "candidate_src")}


def _arm_callable(UT, arm: str):
    """The raw seam function for `arm`, or None if the UT only exposes wrapped entry points."""
    return getattr(UT, "_BASE" if arm == "baseline" else "_CAND", None)


def _assert_arm(tlr, UT, fn, arm: str) -> None:
    """Refuse to run unless `fn` came out of `arm`'s tree.

    Prefers the TL leg's own guard when it has an arm-aware one, so both legs share a single
    definition of "which arm is this". Falls back to the containment test below -- which is
    the same test, written once more rather than skipped, because a missing guard here is
    silent and a duplicated guard is merely redundant.
    """
    shared = getattr(tlr, "assert_arm", None)
    if shared is not None:
        shared(UT, arm)
        return
    if fn is None:
        raise SystemExit(
            f"cannot verify the {arm} arm: this UT exposes no "
            f"{'_BASE' if arm == 'baseline' else '_CAND'}, and the harness has no "
            f"assert_arm(). Add one before trusting either leg -- an unguarded arm "
            f"reports the other arm's number under this arm's name.")
    # __file__ is empty on a getattr'd function; the provenance lives on the code object.
    src = getattr(getattr(fn, "__code__", None), "co_filename", "") or ""
    if not src:
        raise SystemExit(f"refusing to run: cannot resolve a source file for {fn!r}")
    parts = Path(src).resolve().parts
    want = _ARM_DIRS[arm]
    if not any(p in want for p in parts):
        raise SystemExit(f"refusing to run: --arm {arm} resolves to {src!r}, which is "
                         f"under none of {want}")
    other = [d for a, ds in _ARM_DIRS.items() if a != arm for d in ds]
    hit = [d for d in other if d in parts]
    if hit:
        raise SystemExit(f"refusing to run: --arm {arm} path {src!r} also touches {hit}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--harness", required=True, type=Path,
                    help="The TL leg's module, providing load_ut() and optionally an "
                         "arm-aware assert_arm(UT, arm).")
    ap.add_argument("--ut-root", required=True, type=Path,
                    help="Directory holding the unit-test task packages.")
    ap.add_argument("--task", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--arm", choices=("baseline", "candidate"), default="baseline",
                    help="Which implementation to launch. The guard asserts the SELECTED "
                         "arm, so --arm candidate is not a way to switch the guard off.")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--no-flush", action="store_true",
                    help="skip the LLC flush -- reports the cache-WARM AI, which is a "
                         "different quantity from the TraceLens cold byte model")
    args = ap.parse_args()

    tlr = _tlr(args.harness)
    import torch  # noqa: F401  (after the UT loader has set up sys.path)

    UT = tlr.load_ut(args.ut_root / args.task)
    meta = UT.META if hasattr(UT, "META") else UT.meta
    cases = {c["name"]: (i, c) for i, c in enumerate(meta["cases"])}
    if args.case not in cases:
        print(f"no case {args.case!r} in {args.task}; have {sorted(cases)}", file=sys.stderr)
        return 2
    ci, case = cases[args.case]

    # `_make_args` is not one signature across harnesses -- `(c, gen)`, `(c, gen, ci=0)` and
    # `(c, gen, ci, wts)` all occur within a single workload's task set. Build the call from
    # the signature rather than assume one: guessing costs a TypeError on exactly the task
    # whose signature is the odd one out, which is the task you have not checked yet.
    gen = torch.Generator(device=UT.DEV).manual_seed(UT.SEED + ci)
    params = list(inspect.signature(UT._make_args).parameters)
    extra = {"ci": ci}
    if "wts" in params:
        extra["wts"] = UT._build_weights()
    a = UT._make_args(case, gen, **{k: v for k, v in extra.items() if k in params})

    fn = _arm_callable(UT, args.arm)
    _assert_arm(tlr, UT, fn, args.arm)
    strip = getattr(UT, "_strip", None)
    if strip is not None and fn is not None:
        run = lambda: UT._invoke(fn, strip(a))  # noqa: E731
    else:
        run = (lambda: UT.baseline_call(a)) if args.arm == "baseline" else (lambda: UT.call(a))

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()

    h = UT.h
    for _ in range(args.iters):
        if not args.no_flush:
            h.flush_cache(torch)
        run()
    torch.cuda.synchronize()
    print(f"[case_runner] {args.task}/{args.case} ran {args.iters} iters "
          f"(flush={not args.no_flush})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
