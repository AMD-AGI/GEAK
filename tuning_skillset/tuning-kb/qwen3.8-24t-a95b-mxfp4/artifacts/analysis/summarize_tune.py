#!/usr/bin/env python3
import json, sys, glob, os
for path in sorted(sys.argv[1:]):
    cur=None; best=[]; done=False; M=N=K=None
    for line in open(path):
        line=line.strip()
        if not line: continue
        try: r=json.loads(line)
        except Exception: continue
        k=r.get("kind")
        if k=="current": cur=r
        elif k=="done": done=True
        elif k=="ok": best.append(r)
        if r.get("M"): M,N,K=r["M"],r["N"],r["K"]
    if not best:
        print(f"{os.path.basename(path)}: no results"); continue
    best=[r for r in best if r["us"]>0]
    best.sort(key=lambda r: r["us"])
    cu = cur["us"] if cur else float("nan")
    if not cur: cur={"lib":"?","sol":"?"}
    print(f"\n{os.path.basename(path)}  M={M} N={N} K={K}  done={done}  n_ok={len(best)}")
    print(f"  current: {cur['lib']}/{cur['sol']}  {cu:.3f} us")
    for r in best[:5]:
        print(f"    {r['lib']:10s} sol={r['sol']:<8} {r['us']:8.3f} us  speedup {cu/r['us']:.3f}x")
