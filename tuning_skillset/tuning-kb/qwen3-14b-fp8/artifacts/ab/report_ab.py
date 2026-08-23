#!/usr/bin/env python3
"""Report an interleaved across-restart A/B position-matched.

Prints each instance's runs in order (so the within-instance decline is visible),
then compares run i of A against run i of B, which is the only comparison the
0.21% restart floor covers. Also reports whether the two arms' throughput
distributions are disjoint -- with n=6 per arm and a drifting machine, disjoint
ranges are worth more than a difference of means.
"""
import glob, json, os, statistics, sys

label = sys.argv[1]
rows = {}
for d in sorted(glob.glob(f"results/{label}_*")):
    f = os.path.join(d, "inferencex_result.json")
    if not os.path.exists(f):
        continue
    parts = os.path.basename(d).split("_")
    inst, run = parts[1], parts[2]
    rows.setdefault(inst, {})[run] = json.load(open(f))

print(f"{'instance':>9}  " + "  ".join(f"{'run'+str(i):>9}" for i in (1, 2, 3)))
for inst in sorted(rows):
    print(f"{inst:>9}  " + "  ".join(
        f"{rows[inst][r]['output_throughput']:9.3f}" if r in rows[inst] else " " * 9
        for r in ("r1", "r2", "r3")))

print("\nposition-matched A vs B (B = candidate):")
deltas, itls = [], []
for inst_a in sorted(i for i in rows if i.startswith("A")):
    inst_b = "B" + inst_a[1:]
    if inst_b not in rows:
        continue
    for r in ("r1", "r2", "r3"):
        if r not in rows[inst_a] or r not in rows[inst_b]:
            continue
        a, b = rows[inst_a][r], rows[inst_b][r]
        d = (b["output_throughput"] / a["output_throughput"] - 1) * 100
        di = (b["median_itl_ms"] / a["median_itl_ms"] - 1) * 100
        deltas.append(d); itls.append(di)
        print(f"  {inst_a}/{inst_b} {r}: {a['output_throughput']:9.3f} -> "
              f"{b['output_throughput']:9.3f}  {d:+6.2f}%   ITL {di:+6.2f}%")

A = [v["output_throughput"] for i in rows if i.startswith("A") for v in rows[i].values()]
B = [v["output_throughput"] for i in rows if i.startswith("B") for v in rows[i].values()]
print(f"\n  arm A  n={len(A)}  min {min(A):.3f}  median {statistics.median(A):.3f}  max {max(A):.3f}")
print(f"  arm B  n={len(B)}  min {min(B):.3f}  median {statistics.median(B):.3f}  max {max(B):.3f}")
print(f"  disjoint: {min(B) > max(A) or min(A) > max(B)}")
print(f"  position-matched delta: median {statistics.median(deltas):+.2f}%  "
      f"range {min(deltas):+.2f}..{max(deltas):+.2f}%")
print(f"  median-ITL delta:       median {statistics.median(itls):+.2f}%  "
      f"range {min(itls):+.2f}..{max(itls):+.2f}%")
