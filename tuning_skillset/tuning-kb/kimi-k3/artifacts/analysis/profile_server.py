#!/usr/bin/env python3
"""Capture a per-kernel GPU profile of the running server at the frozen shape.

`reference/tracelens/NOTE.md` says no profile of this model was ever captured, so
the only hot-spot claim in the workspace is the code-lane's own
`kernel_journey.json`.  This produces one.

The server is driven at the benchmark's own shape -- ISL 8192, concurrency 64,
`ignore_eos` so nothing finishes early -- and `/start_profile` is fired once the
batch is fully in decode, with `num_steps` so it auto-stops.  The trace that
matters is rank 0's; the others are symmetric at TP=8.

    python3 analysis/profile_server.py --tag cand_s2 --steps 6

Writes `analysis/profiles/<tag>/` and prints the per-kernel GPU-time table.
"""
import argparse
import glob
import gzip
import json
import os
import threading
import time
from collections import defaultdict

import requests

BASE = "http://127.0.0.1:43113"
ISL = 8192
CONC = 64


def load_thread(stop, tok_per_req):
    """Keep `CONC` requests in flight at the benchmark's prompt length."""
    prompt = "hello " * (ISL // 2)
    body = {
        "model": "/shared_nfs/hyperloom/models/Kimi-K3",
        "prompt": prompt,
        "max_tokens": tok_per_req,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": False,
    }

    def one():
        while not stop.is_set():
            try:
                requests.post(f"{BASE}/v1/completions", json=body, timeout=1800)
            except Exception:
                return

    ts = [threading.Thread(target=one, daemon=True) for _ in range(CONC)]
    for t in ts:
        t.start()
    return ts


def kernel_table(trace_path, topn):
    op = gzip.open if trace_path.endswith(".gz") else open
    with op(trace_path, "rt") as f:
        ev = json.load(f)["traceEvents"]
    tot = defaultdict(float)
    cnt = defaultdict(int)
    for e in ev:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat", "")
        if cat not in ("kernel", "gpu_op", "Kernel"):
            continue
        tot[e["name"]] += e.get("dur", 0.0)
        cnt[e["name"]] += 1
    total = sum(tot.values())
    rows = sorted(tot.items(), key=lambda kv: -kv[1])[:topn]
    return total, [(n, tot[n], cnt[n], 100.0 * tot[n] / total) for n, _ in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--settle", type=int, default=70,
                    help="seconds to let prefill drain before profiling decode")
    ap.add_argument("--topn", type=int, default=40)
    # Kineto on ROCm drops device dispatches it cannot correlate to a host op, and
    # a graph replay's kernels only correlate through hipGraphLaunch -- so asking
    # for GPU alone yields a trace holding nothing but the ungraphed sampler.
    ap.add_argument("--activities", nargs="*", default=["CPU", "GPU"])
    args = ap.parse_args()

    outdir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "profiles", args.tag)
    )
    os.makedirs(outdir, exist_ok=True)

    stop = threading.Event()
    # Long enough that every request is still decoding when the profiler fires.
    load_thread(stop, tok_per_req=4096)
    print(f"[prof] load up: {CONC} x ISL {ISL}; settling {args.settle}s into decode")
    time.sleep(args.settle)

    r = requests.post(
        f"{BASE}/start_profile",
        json={
            "output_dir": outdir,
            "num_steps": args.steps,
            "activities": args.activities,
            "record_shapes": True,
            "profile_id": args.tag,
        },
        timeout=120,
    )
    print(f"[prof] start_profile -> {r.status_code} {r.text[:200]}")

    for _ in range(180):
        time.sleep(2)
        if glob.glob(os.path.join(outdir, "*.trace.json*")):
            time.sleep(20)  # let every rank finish writing
            break
    stop.set()

    files = sorted(glob.glob(os.path.join(outdir, "*.trace.json*")))
    print(f"[prof] {len(files)} trace file(s) in {outdir}")
    if not files:
        return
    rank0 = [f for f in files if "TP-0" in f or "rank-0" in f or "-0." in f] or files
    path = rank0[0]
    print(f"[prof] parsing {os.path.basename(path)}\n")
    total, rows = kernel_table(path, args.topn)
    print(f"{'kernel':<78} {'ms':>9} {'calls':>7} {'%':>6}")
    for n, us, c, pct in rows:
        print(f"{n[:78]:<78} {us/1e3:>9.2f} {c:>7} {pct:>6.2f}")
    print(f"\ntotal GPU time in window: {total/1e3:.2f} ms")

    with open(os.path.join(outdir, "kernel_table.txt"), "w") as f:
        f.write(f"{'kernel':<78} {'ms':>9} {'calls':>7} {'%':>6}\n")
        for n, us, c, pct in rows:
            f.write(f"{n[:78]:<78} {us/1e3:>9.2f} {c:>7} {pct:>6.2f}\n")
        f.write(f"\ntotal GPU time in window: {total/1e3:.2f} ms\n")


if __name__ == "__main__":
    main()
