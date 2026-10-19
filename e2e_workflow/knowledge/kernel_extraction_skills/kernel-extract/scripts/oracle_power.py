#!/usr/bin/env python3
"""Can this captured oracle reject anything?

A frozen-oracle correctness case and its numerical negative control both compare a
candidate against values recorded by the capture. Neither comparison means anything if
the recorded values are degenerate, and a degenerate capture does not announce itself:
the UT passes, the control "passes", and the report says `max_rel_err: 0.0`, which reads
like a perfect result rather than like an empty one.

The failure mode this exists to catch was observed on a real decode seam. Its capture
hooked python-level calls, but the decode path runs entirely inside a CUDA graph, so the
hooks only fired during graph CAPTURE -- when the tensors are placeholders. Every
floating-point operand and the output were recorded as identically zero. The integer
index operand was recorded with real-looking content, so "is the file all zero" would
have passed it. The negative control then scaled a zero oracle by 0.5, compared 0 against
0, and could not have failed for any candidate whatsoever.

A `structure_only` capture is judged differently, because it is a different claim. It keeps
only the operands whose CONTENT selects work and reduces the bulk floating-point payload to
seeded descriptors, so "this float operand is random" is the policy working, not a defect.
What must then be checked is the other half: that the structural operands are present, are
real, and that the output was actually regenerated against them. An unregenerated
`structure_only` capture is fatal here -- its recorded output came from operands the file no
longer holds.

Three checks, each of which that capture fails:

  1. ZERO VARIANCE. A recorded OUTPUT with std == 0 makes every downstream comparison
     vacuous, as does a record whose float operands are ALL constant at once. A single
     constant operand is only a note: destination buffers are legitimately pre-zeroed,
     and an op that accumulates into `out` records it as a zero operand by contract.
  2. COLLAPSED POOL. A large operand that is almost entirely zero yet holds real
     content at a handful of locations was collapsed by the capture or never populated.
     Whatever gathers from it reads a cache-resident sliver, so neither its values nor
     any timing taken against it are the served ones. The 2.2 GiB paged KV pool in the
     decode capture below held exactly 6 nonzero elements out of 2.3e9.
  3. CONTROL MARGIN. A negative control perturbs an editable leaf and must move the
     result by more than `tol`. How far it actually moves cannot be computed from the
     capture -- it depends on how much of the output that leaf produces -- so this is not
     inferred. Pass the deviation the control was OBSERVED to produce and the gate checks
     it clears `tol` with room to spare. A control whose observed deviation is not
     supplied is reported as unevaluated rather than silently credited.

    oracle_power.py <reference_io.pt> [--tol 0.02] [--control-rel-err 0.43] \
                    [--out report.json]

Exit 0 = the oracle can discriminate. Exit 1 = it cannot; the task's correctness verdict
is UNRESOLVED, not PASS, until the capture is retaken. Exit 2 = could not be evaluated.
"""

import argparse
import json
import os
import sys

# A served KV pool or weight table of this size is densely populated. One that is almost
# entirely zero while holding real content at a handful of locations was collapsed by the
# capture or never written -- the tell that separates "deduplicated pool" from "pool".
POOL_FLOOR = 1 << 20
POOL_DENSITY_FLOOR = 1e-4


def _as_numeric(torch, t):
    """fp8 and other non-arithmetic dtypes -> something std() accepts, losslessly."""
    x = t.detach()
    if "float8" in str(x.dtype):
        x = x.view(torch.uint8)
    return x.to(torch.float64)


def _tensor_stats(torch, t):
    x = _as_numeric(torch, t)
    n = x.numel()
    if n == 0:
        return {"numel": 0, "empty": True}
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "numel": int(n),
        "nonzero": int((x != 0).sum()),
        "distinct": int(torch.unique(x).numel()),
        "std": float(x.std()) if n > 1 else 0.0,
        "absmax": float(x.abs().max()),
        "is_integer": not t.is_floating_point() and "float8" not in str(t.dtype),
    }


def _walk(torch, obj, path, out, synth=None, pending=None):
    if torch.is_tensor(obj):
        out.append((path, _tensor_stats(torch, obj)))
    elif isinstance(obj, dict) and obj.get("__synth__"):
        # A seeded descriptor. Reproducible by construction, so it is persisted, not missing;
        # it just carries no distribution to judge. Counted, never a finding.
        if synth is not None:
            synth.append(path)
    elif isinstance(obj, dict) and obj.get("__needs_regen__"):
        if pending is not None:
            pending.append(path)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _walk(torch, v, "%s.%s" % (path, k), out, synth, pending)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _walk(torch, v, "%s[%d]" % (path, i), out, synth, pending)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference_io", help="captures/reference_io.pt")
    ap.add_argument("--tol", type=float, default=None,
                    help="the tolerance the UT compares against; enables check 3")
    ap.add_argument("--control-rel-err", type=float, default=None,
                    help="the relative deviation the negative control was OBSERVED to "
                         "produce. Not inferable from the capture, so it must be "
                         "measured and passed in.")
    ap.add_argument("--control-margin", type=float, default=10.0,
                    help="how many times tol the observed control deviation must exceed "
                         "before the control counts as discriminating (default 10x)")
    ap.add_argument("--structural-operand", action="append", default=[],
                    help="operand name whose CONTENT selects work (routing, indices, page "
                         "tables). Repeatable; unioned with the capture's own declaration. "
                         "Each must be present and non-degenerate or the gate is fatal.")
    ap.add_argument("--pool-floor", type=int, default=POOL_FLOOR,
                    help="operands at least this large are checked for being collapsed")
    ap.add_argument("--pool-density-floor", type=float, default=POOL_DENSITY_FLOOR)
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    here = os.path.dirname(os.path.abspath(args.reference_io))
    task_dir = os.path.dirname(here)
    # The task dir holds a `unittest.py` that shadows the stdlib `unittest` package that
    # torch imports as `unittest.mock`. Keep it off sys.path and load by explicit path,
    # exactly as unittest.py::_child_setup does.
    import importlib.util
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != task_dir]
    try:
        import torch
    except ImportError:
        print("oracle_power: torch is not importable; run this inside the UT container.",
              file=sys.stderr)
        return 2
    # Prefer the task's own frozen copy (it is the one the gates actually run against); fall
    # back to the skill's, so a capture can be judged before it has been placed in a task.
    hl = next((p for p in (os.path.join(task_dir, "harness_lib.py"),
                           os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "harness_lib.py"))
               if os.path.exists(p)), None)
    if hl is None:
        print("oracle_power: harness_lib.py found neither in %s nor beside this script."
              % task_dir, file=sys.stderr)
        return 2
    spec = importlib.util.spec_from_file_location("harness_lib", hl)
    h = importlib.util.module_from_spec(spec)
    sys.modules["harness_lib"] = h
    spec.loader.exec_module(h)

    blob = h.load_reference_io(args.reference_io, map_location="cpu")
    records = blob["records"] if isinstance(blob, dict) else blob
    shared = (blob.get("shared") or {}) if isinstance(blob, dict) else {}

    policy = (blob.get("persist_policy") if isinstance(blob, dict) else None) or "share_large"
    oracle_status = (blob.get("oracle_status") if isinstance(blob, dict) else None) or "captured"
    declared = list((blob.get("structural_operands") if isinstance(blob, dict) else None)
                    or args.structural_operand or [])
    if args.structural_operand:
        declared = sorted(set(declared) | set(args.structural_operand))

    stats, synth, pending = [], [], []
    _walk(torch, shared, "shared", stats, synth, pending)
    for i, r in enumerate(records):
        _walk(torch, r, "record[%d]" % i, stats, synth, pending)

    findings = []
    notes = []

    # A structure_only capture whose output was never regenerated holds a placeholder where
    # the oracle should be. It is not a weak oracle, it is not an oracle.
    if pending or (policy == "structure_only" and oracle_status != "regenerated"):
        findings.append({
            "check": "oracle_not_regenerated", "path": "blob", "severity": "fatal",
            "detail": ("this is a structure_only capture with oracle_status=%r: the recorded "
                       "output was produced by operands the file no longer holds (%d "
                       "placeholder leaves). Run scripts/regen_reference.py in the serving "
                       "container to compute the output from the materialized operands, and "
                       "record oracle_coverage=regenerated_from_installed_kernel"
                       % (oracle_status, len(pending)))})

    # Structural operands are the half of a structure_only capture that must be REAL. A
    # declared one that is absent means the capture reduced the thing that selects the work:
    # the routing changed, so the UT runs a different distribution of experts/pages than the
    # deployment did, and on this model uniform routing is the SLOW end (2.46x on decode).
    def _named(path, key):
        # `_snapshot` wraps a tensor as {"__tensor__":..., "data": t}, so the operand named
        # `topk_ids` is reached at `record[0].kwargs.topk_ids.data`. Match the segment, not
        # the suffix -- matching the suffix reports every present operand as missing.
        return key in path.replace("[", ".").replace("]", "").split(".")

    for key in declared:
        hits = [(p, s) for p, s in stats if _named(p, key)]
        if not hits:
            if any(_named(p, key) for p in synth):
                findings.append({
                    "check": "structural_operand_synthesized", "path": key,
                    "severity": "fatal",
                    "detail": ("%r is declared structural but was reduced to a random "
                               "descriptor. Random values here change which work runs, not "
                               "just the numbers in it" % key)})
            else:
                findings.append({
                    "check": "structural_operand_missing", "path": key, "severity": "fatal",
                    "detail": "%r is declared structural but appears in no record" % key})
        elif all(s.get("empty") or s["absmax"] == 0.0 for _, s in hits):
            findings.append({
                "check": "structural_operand_degenerate", "path": key, "severity": "fatal",
                "detail": ("%r is declared structural but is identically zero in every "
                           "record, so it selects nothing" % key)})
    for path, s in stats:
        if s.get("empty"):
            continue
        is_output = ".output" in path
        if (s["numel"] >= args.pool_floor and s["absmax"] > 0
                and s["nonzero"] < s["numel"] * args.pool_density_floor):
            findings.append({
                "check": "collapsed_pool", "path": path, "severity": "fatal",
                "detail": ("%d-element operand carries real content at only %d locations "
                           "(%.6f%% nonzero). A pool this empty was collapsed or never "
                           "populated: whatever gathers from it reads a handful of "
                           "cache-resident slots, so neither the values nor the timing "
                           "are the served ones"
                           % (s["numel"], s["nonzero"],
                              100.0 * s["nonzero"] / s["numel"])),
                "stats": s})
        if not s["is_integer"] and s["std"] == 0.0:
            if is_output:
                findings.append({
                    "check": "zero_variance", "path": path, "severity": "fatal",
                    "detail": ("the recorded OUTPUT is constant %s over %d elements -- "
                               "every comparison against it is vacuous"
                               % ("zero" if s["absmax"] == 0 else "value", s["numel"])),
                    "stats": s})
            else:
                # A single constant operand is not by itself a defect: destination
                # buffers are legitimately pre-zeroed, and an op that accumulates into
                # `out` records it as a zero operand by contract. What is diagnostic is
                # EVERY float input of a record being constant at once, which is handled
                # below over the whole record.
                notes.append({"check": "constant_operand", "path": path, "stats": s})

    # Whole-record variant of check 1: one constant operand is routine, but a record in
    # which every float input is constant is a placeholder record, whatever its output
    # happens to hold.
    for i in range(len(records)):
        pre = "record[%d]." % i
        fl = [(p, s) for p, s in stats
              if p.startswith(pre) and ".output" not in p
              and not s.get("empty") and not s["is_integer"] and s["numel"] > 1]
        if fl and all(s["std"] == 0.0 for _, s in fl):
            findings.append({
                "check": "all_float_operands_constant", "path": pre.rstrip("."),
                "severity": "fatal",
                "detail": ("all %d floating-point operands of this record are constant; "
                           "it holds placeholder buffers, not served values" % len(fl))})

    # Check 3. Not inferable from the capture -- the deviation a perturbed leaf produces
    # depends on how much of the output that leaf is responsible for -- so it is checked
    # against a measured number or reported as unevaluated. Never assumed.
    control = {"evaluated": False}
    if args.tol is not None and args.control_rel_err is not None:
        need = args.tol * args.control_margin
        control = {"evaluated": True, "observed_rel_err": args.control_rel_err,
                   "tol": args.tol, "required": need,
                   "discriminating": args.control_rel_err > need}
        if not control["discriminating"]:
            findings.append({
                "check": "control_margin", "path": "negative_control",
                "severity": "fatal",
                "detail": ("the control was observed to move the result by %.4g relative, "
                           "which does not clear %gx tol=%.4g. A control this close to "
                           "the comparison threshold passes on noise, so its pass is not "
                           "evidence that the oracle binds the leaf"
                           % (args.control_rel_err, args.control_margin, need))})
    elif args.tol is not None:
        print("note: --control-rel-err was not given, so the negative control's "
              "discriminating power is UNEVALUATED. Run the control, record the "
              "deviation it produces, and pass it in; a control that was never measured "
              "is not a control that passed.", file=sys.stderr)

    report = {
        "schema": "agentx-kernel-extraction/oracle_power/1",
        "reference_io": os.path.abspath(args.reference_io),
        "n_records": len(records),
        "n_tensors": len(stats),
        "persist_policy": policy,
        "oracle_status": oracle_status,
        "structural_operands": declared,
        "n_synth_leaves": len(synth),
        "tol": args.tol,
        "negative_control": control,
        "findings": findings,
        "notes": notes,
        "can_discriminate": not findings,
        "tensors": [{"path": p, **s} for p, s in stats],
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)

    for f in findings:
        print("FATAL  %-22s %s\n       %s" % (f["check"], f["path"], f["detail"]),
              file=sys.stderr)
    if findings:
        print("\noracle_power: this capture cannot discriminate. The task's correctness "
              "verdict is UNRESOLVED, not PASS. Retake the capture -- for a graphed "
              "phase, during graph REPLAY in steady state, or via a bounded eager "
              "diagnostic leg whose timing is then revalidated in deployment graph mode.",
              file=sys.stderr)
        return 1
    print("OK  %d records / %d tensors: no constant output, no all-placeholder record, "
          "no collapsed pool%s"
          % (len(records), len(stats),
             ", control clears %gx tol" % args.control_margin
             if control["evaluated"] else ", negative control UNEVALUATED"))
    if policy == "structure_only":
        print("    structure_only: %d structural operands verified present and non-"
              "degenerate, %d bulk leaves are seeded draws. This oracle binds the installed "
              "kernel's own output at the served shapes and the served routing -- it does "
              "NOT bind the deployed activation values. Record that in meta.oracle_coverage."
              % (len(declared), len(synth)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
