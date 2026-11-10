#!/usr/bin/env python3
"""Establish a kernel's run-to-run noise floor -- and prove the inputs held still first.

Before a correctness tolerance can be defended, two things have to be known about the
installed, unmodified kernel: whether it reproduces its own output on identical inputs,
and if not, by how much. Both are easy to get wrong in the same direction, which is why
this is a script rather than advice.

WHAT GOES WRONG. A `_build_args()` for a MoE seam calls the deployment's own routing
helper. On this deployment `aiter.fused_moe.moe_sorting` returns DIFFERENT
`sorted_expert_ids` and `m_indices` from bit-identical inputs. A probe that rebuilds its
args inside the repeat loop therefore compares two outputs computed under two different
routings and reads the result as "the kernel cannot reproduce itself". The symptom is
loud and specific: relative error concentrated entirely on elements where the reference
is 0, a median relative error over nonzero reference of exactly 0, and bimodality -- the
same shape alternating between bit-identical and wildly different. Measured on a real
MoE gemm, the artifact read as rel_frobenius 0.23-0.47 while the kernel's true self-noise
was 0.

So: phase 1 proves the input construction is deterministic. Only if it is does phase 2
mean anything, and phase 2 then reuses ONE set of args across every repeat.

PACKED PAYLOADS. Where the output is packed sub-byte data (MXFP4 quantized values plus
exponent bytes), `rel_frobenius` and `max_abs_diff` over the packed bytes have no
numerical meaning -- 240 is just a different byte, not a large number. For those the only
defensible criterion is byte-exactness, which `--packed` selects.

ATOMIC EPILOGS. A gemm with `epilog='atomic'` accumulates per-expert partials in
scheduling-dependent order and has a genuine, small noise floor. Measured on the same
deployment: at most 2 bf16 ULP at the output's magnitude, rel_frobenius ~0.004 -- five
times under the UT's 0.02 tolerance. That is a decidable gate, and it is a different
thing from the routing artifact above even though the broken probe reported them as one.

    kernel_self_noise.py --task-dir <dir> --repeats 5 [--only-m 24 60] [--packed] \\
                         --out self_noise.json
"""

import argparse
import importlib.util
import json
import os
import sys


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _compare(torch, a, b, packed):
    """a vs b -> dict. For packed payloads only byte-equality is meaningful."""
    if packed:
        same = bool(torch.equal(a, b))
        return {"criterion": "byte_exact", "byte_exact": same,
                "bytes_differing": int((a != b).sum()) if not same else 0}
    x = a.to(torch.float64)
    y = b.to(torch.float64)
    d = (x - y).abs()
    denom = float(x.norm()) or 1.0
    nzmask = x != 0
    return {
        "criterion": "numeric",
        "bit_identical": bool(torch.equal(a, b)),
        "elements": int(x.numel()),
        "elements_differing": int((d != 0).sum()),
        "max_abs_diff": float(d.max()),
        "rel_frobenius": float((x - y).norm()) / denom,
        # The tell for the routing artifact: if every difference sits where the
        # reference is zero, the two runs agreed wherever they both wrote.
        "elements_differing_where_ref_nonzero": int((d[nzmask] != 0).sum())
        if int(nzmask.sum()) else 0,
        "max_abs_at_ref_absmax_scale": float(x.abs().max()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task-dir", default=None,
                    help="defaults to the working directory, which ut_container.sh "
                         "already sets to the task dir")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--build-repeats", type=int, default=4,
                    help="how many times to rebuild args in phase 1")
    ap.add_argument("--only-m", type=int, nargs="*", default=None)
    ap.add_argument("--packed", action="store_true",
                    help="the output is packed sub-byte data; compare bytes, not values")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    tdir = os.path.abspath(args.task_dir or os.getcwd())
    os.chdir(tdir)
    # The task dir holds a `unittest.py` shadowing the stdlib `unittest` package that
    # torch imports as `unittest.mock`. Mirror unittest.py::_child_setup: keep it off
    # sys.path and load the harness and cases by explicit path.
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != tdir]
    import torch
    h = _load("harness_lib", os.path.join(tdir, "harness_lib.py"))
    cases = _load("cases", os.path.join(tdir, "cases.py"))

    meta = json.load(open(os.path.join(tdir, "meta.json")))
    ms = [int(c["m"]) for c in meta["workload"]["cases"] if "m" in c]
    if args.only_m:
        ms = [m for m in ms if m in args.only_m]
    if not ms:
        print("kernel_self_noise: no cases carry an `m`; this script covers the "
              "shape-by-m task shape only.", file=sys.stderr)
        return 2

    report = {"schema": "agentx-kernel-extraction/kernel_self_noise/1",
              "task": meta.get("task"), "repeats": args.repeats,
              "packed_payload": args.packed, "cases": []}
    rc = 0

    for m in ms:
        entry = {"m": m}

        # ---- phase 1: is the input construction itself deterministic? ----
        builds = [cases._build_args(h, m) for _ in range(args.build_repeats)]
        unstable = []
        base = builds[0]
        for key, v0 in base.items():
            if not torch.is_tensor(v0):
                continue
            for j, b in enumerate(builds[1:], 1):
                v = b.get(key)
                if not torch.is_tensor(v) or v.shape != v0.shape:
                    unstable.append({"operand": key, "build": j, "reason": "shape/type"})
                    break
                if not torch.equal(v0, v):
                    unstable.append({"operand": key, "build": j,
                                     "elements_differing": int((v0 != v).sum())})
                    break
        entry["input_construction_deterministic"] = not unstable
        entry["unstable_operands"] = unstable
        del builds

        if unstable:
            names = sorted({u["operand"] for u in unstable})
            print("m=%-4d INPUT CONSTRUCTION IS NOT DETERMINISTIC: %s differ across "
                  "rebuilds with identical seeds." % (m, ", ".join(names)),
                  file=sys.stderr)
            print("       Any self-noise number taken by rebuilding args per repeat is "
                  "about these operands, not about the kernel. Phase 2 below holds one "
                  "set of args fixed, which is the only comparison that means anything.",
                  file=sys.stderr)
            rc = 1  # not fatal to the measurement, but it must be reported

        # ---- phase 2: with ONE set of args, does the kernel reproduce itself? ----
        a = cases._build_args(h, m)
        runs = []
        for _ in range(args.repeats):
            o = cases.call(a)
            torch.cuda.synchronize()
            runs.append(o.detach().clone())
        entry["pairs"] = [_compare(torch, runs[0], r, args.packed) for r in runs[1:]]

        if args.packed:
            ok = all(p["byte_exact"] for p in entry["pairs"])
            entry["self_noise"] = {"byte_exact_over_repeats": ok}
            print("m=%-4d build_det=%-5s  byte_exact=%s" % (
                m, entry["input_construction_deterministic"], ok))
        else:
            rf = [p["rel_frobenius"] for p in entry["pairs"]]
            ad = [p["max_abs_diff"] for p in entry["pairs"]]
            nz = [p["elements_differing_where_ref_nonzero"] for p in entry["pairs"]]
            entry["self_noise"] = {"rel_frobenius_max": max(rf),
                                   "max_abs_diff_max": max(ad),
                                   "output_absmax": entry["pairs"][0]
                                   ["max_abs_at_ref_absmax_scale"]}
            print("m=%-4d build_det=%-5s  rel_frob<=%.6f  max_abs<=%g  "
                  "differing_where_ref_nonzero=%s" % (
                      m, entry["input_construction_deterministic"], max(rf), max(ad), nz))
        report["cases"].append(entry)
        del a, runs

    if not args.packed:
        floors = [c["self_noise"]["rel_frobenius_max"] for c in report["cases"]]
        report["noise_floor_rel_frobenius"] = max(floors) if floors else None
        print("\nnoise floor (rel_frobenius, over all shapes): %.6f"
              % report["noise_floor_rel_frobenius"])
        print("A correctness tolerance below this floor cannot be met by the installed "
              "kernel itself and so cannot decide anything about a candidate.")
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print("wrote %s" % args.out)
    return rc


if __name__ == "__main__":
    sys.exit(main())
