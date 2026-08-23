#!/usr/bin/env python3
"""Map GPU kernels back to the CPU aten op (with input shapes) that launched them."""
import gzip, json, re, sys
from collections import defaultdict

def load(p):
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt") as f: return json.load(f)

def short(name):
    n = re.sub(r"<.*>", "", name); n = re.sub(r"\(.*\)", "", n)
    return n.replace("void ", "").strip().split("::")[-1][:70]

def main():
    tr = load(sys.argv[1])
    ev = tr["traceEvents"]
    # correlation id -> kernel
    kern = {}
    runtime = {}       # correlation -> External id
    cpu_ops = defaultdict(list)  # tid -> list of (ts, ts+dur, name, shapes, extid)
    for e in ev:
        if e.get("ph") != "X": continue
        cat = e.get("cat", "")
        a = e.get("args", {}) or {}
        if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            c = a.get("correlation")
            if c is not None: kern[c] = (e["name"], e.get("dur", 0))
        elif cat in ("cuda_runtime", "hip_runtime"):
            c = a.get("correlation")
            if c is not None: runtime[c] = a.get("External id")
        elif cat in ("cpu_op", "user_annotation"):
            cpu_ops[e["tid"]].append(e)

    extid2op = {}
    for tid, lst in cpu_ops.items():
        lst.sort(key=lambda x: (x["ts"], -x.get("dur", 0)))
        for e in lst:
            a = e.get("args", {}) or {}
            xid = a.get("External id")
            if xid is None: continue
            dims = a.get("Input Dims") or a.get("Concrete Inputs")
            prev = extid2op.get(xid)
            # keep the deepest (latest-starting) op for the external id
            if prev is None or e["ts"] >= prev[2]:
                extid2op[xid] = (e["name"], dims, e["ts"])

    agg = defaultdict(lambda: [0.0, 0])
    total = 0.0
    for c, (kname, dur) in kern.items():
        total += dur
        xid = runtime.get(c)
        op = extid2op.get(xid)
        opname = op[0] if op else "?"
        dims = json.dumps(op[1]) if op and op[1] else ""
        k = (short(kname), opname, dims[:90])
        agg[k][0] += dur; agg[k][1] += 1

    print(f"# {sys.argv[1]}\n# total {total/1000:.3f} ms, {len(kern)} kernels")
    topn = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    for (kn, op, d), (v, n) in sorted(agg.items(), key=lambda x: -x[1][0])[:topn]:
        print(f"{100*v/total:6.2f}% {v/1000:9.3f}ms n={n:5d} {v/n:8.2f}us | {kn}\n           <- {op} {d}")

main()
