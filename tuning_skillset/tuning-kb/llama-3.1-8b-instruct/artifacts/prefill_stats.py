#!/usr/bin/env python3
"""Summarise the full-size prefill batches in an SGLang server log.

The scheduler prints one line per prefill batch with `input throughput (token/s)`. Under the
frozen workload every real prefill batch is exactly two 8192-token requests (#new-token: 16384);
the handful of tiny warm-up/health batches are filtered out. Median is reported rather than mean
because the first batch of an instance is always slow (clock ramp).

  python3 analysis/prefill_stats.py <server.log> [arm] [round]
"""
from __future__ import annotations

import re
import statistics
import sys

PAT = re.compile(r"#new-token: (\d+),.*input throughput \(token/s\): ([0-9.]+)")
NTOK = 16384


def stats(path: str) -> dict:
    vals = [
        float(m.group(2))
        for line in open(path)
        if (m := PAT.search(line)) and int(m.group(1)) == NTOK
    ]
    if not vals:
        return {"n": 0}
    med = statistics.median(vals)
    return {
        "n": len(vals),
        "median_tok_s": med,
        "mean_tok_s": statistics.mean(vals),
        "ms_per_batch": NTOK / med * 1000,
        "spread_pct": 100 * (max(vals) - min(vals)) / med,
    }


if __name__ == "__main__":
    s = stats(sys.argv[1])
    tag = " ".join(sys.argv[2:])
    if not s["n"]:
        print(f"  {tag}: no full-size prefill batches in {sys.argv[1]}")
    else:
        print(
            f"  {tag}: prefill {s['median_tok_s']:.0f} tok/s  "
            f"{s['ms_per_batch']:.2f} ms/batch  n={s['n']}  spread {s['spread_pct']:.2f}%"
        )
