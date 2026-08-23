#!/usr/bin/env python3
"""Summarise steady-state decode throughput from an SGLang server log.

The decode analogue of analysis/prefill_stats.py, and for the same reason: whole-run throughput
resolves a decode-only change at 1/0.71 of its true size mixed with prefill noise, while the
scheduler prints `gen throughput (token/s)` on every decode batch. Only lines at the full running
batch are counted, and the first few of each wave are dropped -- the first decode batch after a
prefill burst reports a throughput averaged over the prefill it waited on.

  python3 analysis/decode_stats.py <server.log> [tag]
"""
from __future__ import annotations

import re
import statistics
import sys

PAT = re.compile(r"Decode batch, #running-req: (\d+),.*gen throughput \(token/s\): ([0-9.]+)")
FULL_BATCH = 64
DROP_HEAD = 3  # per contiguous run: the batch that straddles the prefill burst


def stats(path: str, full_batch: int = FULL_BATCH) -> dict:
    runs, cur, prev_ok = [], [], False
    for line in open(path):
        m = PAT.search(line)
        ok = bool(m) and int(m.group(1)) == full_batch
        if ok:
            if not prev_ok:
                cur = []
                runs.append(cur)
            cur.append(float(m.group(2)))
        prev_ok = ok
    vals = [v for run in runs for v in run[DROP_HEAD:]]
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "runs": len(runs),
        "median_tok_s": statistics.median(vals),
        "mean_tok_s": statistics.mean(vals),
        "p10": statistics.quantiles(vals, n=10)[0] if len(vals) > 10 else min(vals),
    }


if __name__ == "__main__":
    s = stats(sys.argv[1])
    tag = " ".join(sys.argv[2:])
    if not s["n"]:
        print(f"  {tag}: no full-batch decode lines in {sys.argv[1]}")
    else:
        print(
            f"  {tag}: decode {s['median_tok_s']:.1f} tok/s median "
            f"({s['mean_tok_s']:.1f} mean, p10 {s['p10']:.1f})  n={s['n']} in {s['runs']} waves"
        )
