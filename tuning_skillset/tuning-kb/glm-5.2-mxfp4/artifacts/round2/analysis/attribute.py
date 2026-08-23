#!/usr/bin/env python3
"""Attribute GPU kernels back to the Python line that launched them.

A kernel-name profile ranks time but cannot say which line launched
`elementwise_kernel_manual_unroll<..., gpu_kernel_impl_nocast<...>>`. Round 1 established
that a `with_stack` capture must be taken during **prefill**: decode runs inside a HIP graph,
so the decode trace has kernels tagged "Dispatch Task" with no Python frames at all. Prefill
runs eagerly and executes the same layer code.

Linking, on this torch/rocm build:
  * a GPU `kernel` event carries `args["correlation"]`, matching a `cuda_runtime`
    (hipLaunchKernel) event with the same correlation id;
  * that runtime event carries `args["External id"]`, matching the launching `cpu_op`;
  * there is **no** `args["Call stack"]` on the cpu_op here. Instead `with_stack=True` emits a
    full `python_function` event tree. The Python frames are recovered by finding the innermost
    `python_function` on the same thread whose [ts, ts+dur] contains the cpu_op, then walking
    `args["Python parent id"]` outward.

    python3 attribute.py TRACE --match elementwise_kernel_manual_unroll
"""
import argparse
import gzip
import json
import re
from bisect import bisect_right
from collections import defaultdict


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)


GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--match", required=True, help="substring of the kernel name")
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--shapes", action="store_true", help="print launching-op input dims")
    ap.add_argument("--collapse", action="store_true",
                    help="strip nn.Module instance indices so all 78 layers aggregate together")
    args = ap.parse_args()

    ev = load(args.trace)["traceEvents"]

    kernels = [
        e for e in ev
        if e.get("cat") in GPU_CATS and e.get("ph") == "X" and args.match in e.get("name", "")
    ]
    if not kernels:
        print(f"no GPU kernel matching {args.match!r} in this trace")
        return 1

    rt_by_corr, op_by_extid = {}, {}
    pyf_by_id = {}
    per_tid = defaultdict(list)
    for e in ev:
        if e.get("ph") != "X":
            continue
        cat, a = e.get("cat"), (e.get("args") or {})
        if cat == "cuda_runtime" and "correlation" in a:
            rt_by_corr[a["correlation"]] = e
        elif cat == "cpu_op" and "External id" in a:
            op_by_extid.setdefault(a["External id"], e)
        elif cat == "python_function":
            pyf_by_id[a.get("Python id")] = e
            per_tid[(e.get("pid"), e.get("tid"))].append(e)

    for v in per_tid.values():
        v.sort(key=lambda e: e["ts"])
    starts = {k: [e["ts"] for e in v] for k, v in per_tid.items()}

    def frames_for(op):
        """Innermost python_function containing `op`, then out via Python parent id."""
        key = (op.get("pid"), op.get("tid"))
        lst, st = per_tid.get(key, []), starts.get(key, [])
        i = bisect_right(st, op["ts"]) - 1
        end = op["ts"] + op.get("dur", 0)
        inner = None
        while i >= 0:
            e = lst[i]
            if e["ts"] + e.get("dur", 0) >= end:
                inner = e
                break
            i -= 1
        if inner is None:
            return None
        out, seen = [], set()
        cur = inner
        while cur is not None:
            pid = (cur.get("args") or {}).get("Python id")
            if pid in seen:
                break
            seen.add(pid)
            out.append(cur.get("name", "?"))
            cur = pyf_by_id.get((cur.get("args") or {}).get("Python parent id"))
        out.reverse()
        return out

    groups = defaultdict(lambda: {"n": 0, "us": 0.0, "op": None, "dims": set()})
    unattributed = 0
    for k in kernels:
        a = k.get("args") or {}
        rt = rt_by_corr.get(a.get("correlation"))
        op = op_by_extid.get((rt.get("args") or {}).get("External id")) if rt else None
        fr = frames_for(op) if op else None
        if not fr:
            unattributed += 1
            key = ("<unattributed>",)
        else:
            if args.collapse:
                fr = [re.sub(r"^(nn\.Module: \w+?)_\d+$", r"\1_N", f) for f in fr]
            key = tuple(fr)
        g = groups[key]
        g["n"] += 1
        g["us"] += k.get("dur", 0)
        if op is not None:
            g["op"] = op.get("name")
            d = (op.get("args") or {}).get("Input Dims")
            if d:
                g["dims"].add(json.dumps(d))

    tot = sum(k.get("dur", 0) for k in kernels)
    print(f"trace   : {args.trace}")
    print(f"kernel  : *{args.match}*")
    print(f"calls   : {len(kernels)}   total {tot/1000:.2f} ms   "
          f"{tot/len(kernels):.2f} us/call   unattributed {unattributed}")
    print(f"distinct call stacks: {len(groups)}\n")

    for fr, g in sorted(groups.items(), key=lambda kv: -kv[1]["us"])[: args.top]:
        print("=" * 100)
        print(f"{g['n']:5d} calls  {g['us']/1000:8.2f} ms  ({100*g['us']/tot:5.1f}%)  "
              f"{g['us']/g['n']:6.2f} us/call   launching op: {g['op']}")
        for f in list(fr)[-args.frames:]:
            print("      " + f)
        if args.shapes:
            for d in sorted(g["dims"])[:3]:
                print("      dims: " + d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
