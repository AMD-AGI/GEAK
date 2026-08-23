#!/usr/bin/env python3
"""Pull the headline metrics out of one or more inferencex_result.json files."""
import json, sys, glob, os

rows = []
paths = sys.argv[1:] or sorted(glob.glob("results/*/inferencex_result.json"))
for p in paths:
    if os.path.isdir(p):
        p = os.path.join(p, "inferencex_result.json")
    try:
        d = json.load(open(p))
    except Exception as e:
        print(f"skip {p}: {e}"); continue
    rows.append((os.path.basename(os.path.dirname(p)),
                 d.get("output_throughput"), d.get("total_token_throughput"),
                 d.get("mean_ttft_ms"), d.get("mean_tpot_ms"), d.get("duration")))
w = max((len(r[0]) for r in rows), default=10)
print(f"{'tag':<{w}} {'out tok/s':>10} {'tot tok/s':>10} {'TTFT ms':>10} {'TPOT ms':>9} {'dur s':>8}")
for r in rows:
    print(f"{r[0]:<{w}} {r[1]:>10.3f} {r[2]:>10.1f} {r[3]:>10.1f} {r[4]:>9.3f} {r[5]:>8.1f}")
if len(rows) > 1:
    v = [r[1] for r in rows]
    mean = sum(v)/len(v)
    print(f"\nn={len(v)} mean={mean:.3f} min={min(v):.3f} max={max(v):.3f} "
          f"spread={(max(v)-min(v))/mean*100:.3f}%")
