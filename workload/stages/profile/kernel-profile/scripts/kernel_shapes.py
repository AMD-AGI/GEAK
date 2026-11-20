#!/usr/bin/env python3
"""One point's trace -> each selected kernel's launch shapes, operands and call stack.

A UT lane holds one card and starts no server, yet its case manifest has to carry the
deployed signature of every case: shape, dtype, stride/layout, scalar arguments, phase
and observed call weight. The probe traces already hold all of it when they were armed
with record_shapes and with_stack (serving_probe.py does by default), so it is collected
here once, after selection, instead of by a capture inside each lane.

    python3 kernel_shapes.py --trace-dir <point trace dir> --phase DECODE \
        --census evidence/census/de_1.json --targets selection_targets.json \
        --point-id de_1 --out evidence/shapes/de_1.json

Per kernel, over the census's own instances (same instance definition as
point_census.py, at the census's achieved shape):

  variants[]     every distinct (grid, block) it ran at, launches and us per instance
  samples[]      up to --samples launches per variant, each resolved to
                   host_op      the innermost non-wrapper cpu_op around the launch, with
                                Input Dims / Input type / Input Strides / Concrete Inputs
                   op_chain     enclosing cpu_op names, innermost first
                   py_frames    enclosing Python frames "<file>(<line>): <func>",
                                innermost first (what source_map.py reads)
                   resolved_by  external_id | launch_correlation | unresolved

A Triton kernel is launched by a raw hipModuleLaunchKernel with no External id of its own,
so a launch is resolved through its runtime event's correlation and then by containment
on the launching thread. Values are never read, only structure.

Reads one rank (TP-0 when present): launch shapes are the same on every TP rank.
Stdlib only.
"""
import argparse
import bisect
import glob
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace_scan import scan, DEVICE_CATS, STEP_CATS  # noqa: E402
from point_census import step_spans, instance_windows  # noqa: E402

# Host ops that only wrap or allocate: kept in op_chain, never reported as the launcher.
WRAPPER_OPS = re.compile(r"^aten::(empty|empty_like|empty_strided|zeros|zeros_like|to|"
                         r"_to_copy|contiguous|view|reshape|slice|select|detach|expand|"
                         r"unsqueeze|squeeze|t|transpose|clone|as_strided|copy_|fill_)$")
HOST_CATS = ("cpu_op", "python_function", "cuda_runtime")
# A launch replayed from a captured CUDA graph has the graph's replay() as its innermost
# frame: no launching op and no source frame, only the runner that replayed it.
GRAPH_REPLAY = re.compile(r"(^|/)torch/cuda/graphs\.py\(\d+\): replay$")


def pick_rank_file(trace_dir):
    files = sorted(glob.glob(os.path.join(trace_dir, "*.trace.json*")))
    if not files:
        raise SystemExit("kernel_shapes: no traces in %s" % trace_dir)
    tp0 = [f for f in files if "TP-0." in os.path.basename(f)]
    return (tp0 or files)[0]


def targets_from(path):
    d = json.load(open(path))
    rows = d.get("targets") or d.get("ranking") or []
    return {r.get("id") or r.get("name") for r in rows if r.get("id") or r.get("name")}


def contain(host, queries):
    """host: [(cat, name, ts, dur, pid, tid, args)]; queries: [(tid, ts)] ->
    per query {"ops": [(dur, name, args)], "py": [(dur, name)]}. One pass over host."""
    by_tid = defaultdict(list)
    for i, (tid, ts) in enumerate(queries):
        by_tid[tid].append((ts, i))
    for v in by_tid.values():
        v.sort()
    keys = {t: [ts for ts, _ in v] for t, v in by_tid.items()}
    out = [{"ops": [], "py": []} for _ in queries]
    for cat, name, ts, dur, _pid, tid, args in host:
        ks = keys.get(tid)
        if not ks or cat == "cuda_runtime":
            continue
        lo, hi = bisect.bisect_left(ks, ts), bisect.bisect_right(ks, ts + dur)
        for _ts, idx in by_tid[tid][lo:hi]:
            if cat == "cpu_op":
                out[idx]["ops"].append((dur, name, args))
            else:
                out[idx]["py"].append((dur, name))
    for r in out:
        r["ops"].sort(key=lambda x: x[0])
        r["py"].sort(key=lambda x: x[0])
    return out


def describe(op):
    _dur, name, a = op
    a = a or {}
    return {"op": name, "input_dims": a.get("Input Dims"), "input_types": a.get("Input type"),
            "input_strides": a.get("Input Strides"),
            "concrete_inputs": a.get("Concrete Inputs")}


def shapes(trace_dir, phase, census, wanted, samples=3, draft_shares_name=False,
           max_frames=40):
    path = pick_rank_file(trace_dir)
    s = scan(path, DEVICE_CATS + STEP_CATS + HOST_CATS,
             args_cats=("kernel", "cpu_op", "cuda_runtime"))
    gpu, cpu = step_spans(s.events)
    wins, anchor, err = instance_windows(gpu, cpu, phase, draft_shares_name)
    if err:
        raise SystemExit("kernel_shapes: %s: %s" % (path, err))
    shape = census["achieved_shape"]
    wins = [w for w in wins if w["shape"] == shape][census.get("dropped_first_instances", 0):]
    if not wins:
        raise SystemExit("kernel_shapes: no %s instance at shape %s in %s" % (phase, shape, path))

    dev = [e for e in s.events if e[0] in DEVICE_CATS]
    starts = [e[2] for e in dev]
    host = [e for e in s.events if e[0] in HOST_CATS]
    ops_by_ext, rt_by_corr = defaultdict(list), {}
    for e in host:
        a = e[6] or {}
        if e[0] == "cpu_op" and "External id" in a:
            ops_by_ext[a["External id"]].append(e)
        elif e[0] == "cuda_runtime" and a.get("correlation") is not None:
            rt_by_corr[a["correlation"]] = e
    has_shapes = any(e[0] == "cpu_op" and (e[6] or {}).get("Input Dims") is not None
                     for e in host[:200000])
    has_stack = any(e[0] == "python_function" for e in host)

    variants = defaultdict(lambda: {"launches": 0, "us": 0.0, "samples": []})
    for w in wins:
        lo, hi = bisect.bisect_left(starts, w["start"]), bisect.bisect_left(starts, w["end"])
        for e in dev[lo:hi]:
            if e[1] not in wanted:
                continue
            a = e[6] or {}
            key = (e[1], tuple(a.get("grid") or ()), tuple(a.get("block") or ()))
            v = variants[key]
            v["launches"] += 1
            v["us"] += e[3]
            if len(v["samples"]) < samples:
                v["samples"].append(e)

    queries, slots = [], []
    for key, v in variants.items():
        for e in v["samples"]:
            a = e[6] or {}
            ext = a.get("External id")
            how, q = "unresolved", None
            cands = sorted(ops_by_ext.get(ext) or [], key=lambda c: c[3]) if ext is not None else []
            if cands:
                how, q = "external_id", (cands[0][5], cands[0][2])
            else:
                r = rt_by_corr.get(a.get("correlation"))
                if r is not None:
                    how, q = "launch_correlation", (r[5], r[2])
            slots.append((key, len(queries) if q else None, how))
            if q:
                queries.append(q)
    found = contain(host, queries) if queries else []

    inst = len(wins)
    out = defaultdict(lambda: {"variants": [], "launches_per_instance": 0.0,
                               "us_per_instance": 0.0})
    for key, v in sorted(variants.items(), key=lambda kv: -kv[1]["us"]):
        name, grid, block = key
        recs = []
        for k2, qi, how in slots:
            if k2 != key:
                continue
            r = found[qi] if qi is not None else {"ops": [], "py": []}
            real = [o for o in r["ops"] if not WRAPPER_OPS.match(o[1])]
            frames = [p[1] for p in r["py"]]
            recs.append({"resolved_by": how if (r["ops"] or r["py"]) else "unresolved",
                         "graph_replayed": any(GRAPH_REPLAY.search(f) for f in frames[:3]),
                         "host_op": describe(real[0]) if real else None,
                         "op_chain": [o[1] for o in r["ops"]],
                         "py_frames": [p[1] for p in r["py"]][:max_frames]})
        k = out[name]
        k["variants"].append({"grid": list(grid), "block": list(block),
                              "launches": v["launches"],
                              "launches_per_instance": round(v["launches"] / inst, 4),
                              "us_per_instance": round(v["us"] / inst, 3),
                              "samples": recs})
        k["launches_per_instance"] = round(k["launches_per_instance"] + v["launches"] / inst, 4)
        k["us_per_instance"] = round(k["us_per_instance"] + v["us"] / inst, 3)
    return {
        "contract": "kernel_extract/kernel_shapes/v1",
        "phase": phase, "achieved_shape": shape, "instances": inst, "anchor": anchor,
        "trace": os.path.abspath(path), "scan_method": s.method,
        "record_shapes": has_shapes, "with_stack": has_stack,
        "kernels": dict(out),
        "requested": sorted(wanted),
        "not_launched_here": sorted(wanted - set(out)),
        "graph_replayed_kernels": sorted(
            k for k, v in out.items()
            if all(r["graph_replayed"] for x in v["variants"] for r in x["samples"])
            and any(x["samples"] for x in v["variants"])),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trace-dir", required=True)
    ap.add_argument("--phase", required=True, choices=["PREFILL", "DECODE"])
    ap.add_argument("--census", required=True, help="point_census.py output for this point")
    ap.add_argument("--targets", help="selection_targets.json (targets[].id)")
    ap.add_argument("--kernel", action="append", default=[], help="extra exact kernel name")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--draft-shares-name", action="store_true")
    ap.add_argument("--point-id", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    wanted = set(a.kernel) | (targets_from(a.targets) if a.targets else set())
    if not wanted:
        raise SystemExit("kernel_shapes: no kernels requested (--targets / --kernel)")
    census = json.load(open(a.census))
    out = shapes(a.trace_dir, a.phase, census, wanted, a.samples, a.draft_shares_name)
    out["point_id"] = a.point_id
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    unres = sum(1 for k in out["kernels"].values() for v in k["variants"]
                for r in v["samples"] if r["resolved_by"] == "unresolved")
    print("%s: %d/%d kernels launched, %d instances, record_shapes=%s with_stack=%s, "
          "%d unresolved samples -> %s"
          % (a.point_id, len(out["kernels"]), len(wanted), out["instances"],
             out["record_shapes"], out["with_stack"], unres, a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
