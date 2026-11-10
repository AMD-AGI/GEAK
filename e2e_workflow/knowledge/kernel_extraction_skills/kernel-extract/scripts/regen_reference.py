#!/usr/bin/env python3
"""Turn a `structure_only` capture into a usable oracle by running the installed kernel.

A `structure_only` capture keeps every operand whose CONTENT selects work -- routing tables,
gather indices, page tables, sequence bookkeeping -- and reduces the bulk floating-point
payload to a seeded descriptor. That makes the capture three orders of magnitude smaller
(on a real GLM-5.2 run, 138 GB of raw captures against ~30 MB of task captures), but it also
means the output recorded at capture time was produced by operands the file no longer holds.
The capture therefore ships with `oracle_status="needs_regen"` and nothing will accept it.

This script closes that: it materializes the operands from their recorded seeds, calls the
installed, unmodified target once per record, and writes a complete `reference_io.pt` whose
operands and output DO correspond. The resulting oracle is a determinism oracle, not a
deployment-value oracle -- say so in `meta.oracle_coverage`. It is still strictly more
discriminating than the alternative it replaces: a capture taken during CUDA-graph capture
records placeholder buffers, and a correctness gate against an all-zero output cannot fail.

The operands are persisted VERBATIM into the output file, not left as descriptors. Rebuilding
them at UT time would reintroduce whatever nondeterminism the rebuild path contains -- on this
deployment `aiter.fused_moe.moe_sorting` returns different routing from bit-identical inputs --
inside the correctness gate, where it is far harder to see than in a probe.

    regen_reference.py captures/reference_io.pt --out captures/reference_io.regen.pt

Exit 0 = a complete oracle was written. 2 = the input was not a structure_only capture, or
the target could not be resolved, or a recomputed output was degenerate.
"""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path


def _resolve(target):
    mod_name, attr = target.split(":", 1)
    obj = importlib.import_module(mod_name)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def _degenerate(torch, out):
    """A recomputed output that is constant tells us the regeneration did not work -- the
    whole point was to replace an oracle that could not reject anything."""
    bad = []

    def walk(o, path):
        if torch.is_tensor(o):
            if o.numel() > 1 and o.is_floating_point() and float(o.detach().float().std()) == 0.0:
                bad.append(path)
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                walk(v, "%s[%d]" % (path, i))
        elif isinstance(o, dict):
            for k, v in o.items():
                walk(v, "%s.%s" % (path, k))

    walk(out, "output")
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference_io")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--target", help="override module:attr (default: as recorded)")
    ap.add_argument("--allow-constant-output", action="store_true",
                    help="accept a constant recomputed output (only for ops whose output "
                         "genuinely is constant; state why in meta.oracle_coverage)")
    args = ap.parse_args(argv)

    here = os.path.dirname(os.path.abspath(args.reference_io))
    task_dir = os.path.dirname(here)
    # The task dir holds a `unittest.py` that shadows the stdlib `unittest` package torch
    # imports as `unittest.mock`. Keep it off sys.path; load helpers by explicit path.
    import importlib.util
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != task_dir]
    import torch

    skill_scripts = str(Path(__file__).resolve().parents[4] / "scripts")
    for name in ("harness_lib", "capture_shapes"):
        for root in (task_dir, skill_scripts):
            path = os.path.join(root, name + ".py")
            if os.path.exists(path) and name not in sys.modules:
                spec = importlib.util.spec_from_file_location(name, path)
                m = importlib.util.module_from_spec(spec)
                sys.modules[name] = m
                spec.loader.exec_module(m)
    h = sys.modules.get("harness_lib")
    cs = sys.modules.get("capture_shapes")
    if cs is None or h is None:
        print("regen_reference: capture_shapes.py / harness_lib.py not found in the shared tools for this "
              "script or in the task dir.", file=sys.stderr)
        return 2

    blob = h.load_reference_io(args.reference_io, map_location="cpu")
    if not isinstance(blob, dict) or blob.get("persist_policy") != "structure_only":
        print("regen_reference: %s is not a structure_only capture (policy=%r). A capture "
              "that already holds its operands needs no regeneration; use it as-is."
              % (args.reference_io, (blob or {}).get("persist_policy")), file=sys.stderr)
        return 2

    target = args.target or blob.get("target")
    try:
        fn = _resolve(target)
    except Exception as exc:
        print("regen_reference: cannot resolve %r in this environment: %s\nRun this inside "
              "the serving container, with the same overlays the capture was taken under."
              % (target, exc), file=sys.stderr)
        return 2

    records, out_records, constant = blob["records"], [], []
    for i, r in enumerate(records):
        a = h.reconstruct_captured(cs.materialize_synth(r["args"], device=args.device),
                                   device=args.device)
        k = h.reconstruct_captured(cs.materialize_synth(r["kwargs"], device=args.device),
                                   device=args.device)
        out = fn(*tuple(a), **(k if isinstance(k, dict) else {}))
        bad = _degenerate(torch, out)
        if bad:
            constant.append((i, bad))
        out_records.append({"sig": r["sig"], "regime": r.get("regime"),
                            "args": cs._snapshot(a), "kwargs": cs._snapshot(k),
                            "output": cs._snapshot(out), "shared_keys": []})
        sys.stderr.write("[regen] record %d/%d: %s\n" % (i + 1, len(records), r["sig"]))

    if constant and not args.allow_constant_output:
        for i, paths in constant:
            print("FATAL  record[%d] recomputed a CONSTANT output at %s" % (i, ", ".join(paths)),
                  file=sys.stderr)
        print("\nregen_reference: refusing to write an oracle that cannot reject anything. "
              "Check that the structural operands are the real ones and that the target "
              "resolves to the deployed implementation.", file=sys.stderr)
        return 2

    torch.save({"target": target, "records": out_records, "shared": {},
                "persist_policy": "structure_only",
                "oracle_status": "regenerated",
                "oracle_coverage": "regenerated_from_installed_kernel",
                "regenerated_from": os.path.abspath(args.reference_io),
                "structural_operands": blob.get("structural_operands") or []}, args.out)
    print("OK  %d records regenerated -> %s\n    This oracle binds the installed kernel's "
          "own output, not the deployed values; record that in meta.oracle_coverage."
          % (len(out_records), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
