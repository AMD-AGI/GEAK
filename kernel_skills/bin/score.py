#!/usr/bin/env python3
"""Primary speedup from bin/ab.sh logs: median over reps of the per-rep metric.

Usage: score.py <ab_out_dir> [latency_regex]
  B_<rep>.log carries GEAK_WEIGHTED_SPEEDUP=<x> -> that value per rep (harness pairs in-process).
  Otherwise per-case latencies from A_<rep>.log / B_<rep>.log (regex groups: ms, case) ->
  geomean of base_ms / cand_ms per rep.
Prints JSON {speedup, spread, reps, per_case}; exits 1 when nothing could be scored.
"""
import json, math, re, statistics, sys
from pathlib import Path

out = Path(sys.argv[1])
lat_re = re.compile(sys.argv[2] if len(sys.argv) > 2 else r"GEAK_RESULT_LATENCY_MS=([0-9.eE+-]+)\s+(\S+)")
w_re = re.compile(r"GEAK_WEIGHTED_SPEEDUP=([0-9.eE+-]+)")


def cases(path):
    return {m.group(2): float(m.group(1)) for m in lat_re.finditer(path.read_text(errors="replace"))}


per_rep, rows = [], {}
for b in sorted(out.glob("B_*.log")):
    text = b.read_text(errors="replace")
    w = w_re.findall(text)
    if w:
        per_rep.append(float(w[-1]))
        continue
    a = out / b.name.replace("B_", "A_")
    if not a.exists():
        continue
    base, cand = cases(a), cases(b)
    common = [k for k in base if k in cand and base[k] > 0 and cand[k] > 0]
    if not common or len(common) != len(base):
        continue  # a case missing on either side makes the rep unscorable
    per_rep.append(math.exp(statistics.mean(math.log(base[k] / cand[k]) for k in common)))
    for k in common:
        rows.setdefault(k, []).append((base[k], cand[k]))

if not per_rep:
    print(json.dumps({"speedup": 0.0, "error": "no scorable rep"}))
    sys.exit(1)
med = statistics.median(per_rep)
print(json.dumps({
    "speedup": round(med, 4),
    "spread": round((max(per_rep) - min(per_rep)) / med, 4),
    "reps": len(per_rep),
    "per_case": [{"name": k, "base_ms": statistics.median(x for x, _ in v),
                  "cand_ms": statistics.median(y for _, y in v)} for k, v in rows.items()],
}))
