#!/usr/bin/env python3
"""Profile the live SGLang server's PREFILL path.

The frozen workload is ISL 8192 / OSL 1024 at concurrency 64, i.e. 1,572,864
prefill tokens against 196,608 decode tokens. Measured, prefill is ~36% of the
benchmark's wall clock (78.33 s total vs a 49.9 s pure-decode lower bound), so
it has to be profiled too -- analysis/profile_decode.py only ever saw decode.

Method: idle server, then fire `--reqs` requests of ISL 8192 with max_tokens=1
so essentially all device work is prefill. chunked_prefill_size is 16384, so a
forward pass is two 8192-token sequences. Rank kernels by SUMMED DEVICE
DURATION over the kernel-category events, never by a percentage column.
"""
import argparse, json, os, threading, time, urllib.request, glob, gzip, sys
from collections import defaultdict

P = argparse.ArgumentParser()
P.add_argument("--port", type=int, default=43103)
P.add_argument("--reqs", type=int, default=16)
P.add_argument("--isl", type=int, default=8192)
P.add_argument("--out", default="analysis/prof_prefill")
P.add_argument("--steps", type=int, default=8)
A = P.parse_args()

BASE = f"http://127.0.0.1:{A.port}"
MODEL = os.environ.get("MODEL", "/shared_nfs/hyperloom/models/Qwen3.5-397B-A17B-MXFP4")


def post(path, payload=None, timeout=1800):
    data = json.dumps(payload or {}).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()


import random
rng = random.Random(0)


def prompt_ids():
    return [rng.randint(1000, 200000) for _ in range(A.isl)]


def fire(n):
    """n prefill-only requests, one thread each."""
    ts = []
    for _ in range(n):
        p = prompt_ids()
        t = threading.Thread(target=lambda p=p: post(
            "/v1/completions", {"model": MODEL, "prompt": p, "max_tokens": 1,
                                "temperature": 0.0, "ignore_eos": True}), daemon=True)
        ts.append(t)
    for t in ts:
        t.start()
    return ts


print(f"[prof] warming prefill with {A.reqs} requests")
for t in fire(A.reqs):
    t.join()

outdir = os.path.abspath(A.out)
os.makedirs(outdir, exist_ok=True)
print(f"[prof] capturing {A.steps} prefill forward passes -> {outdir}")
post("/start_profile", {"output_dir": outdir, "num_steps": A.steps,
                        "activities": ["CPU", "GPU"], "profile_by_stage": False})
threads = fire(A.reqs)
for t in threads:
    t.join()
time.sleep(45)
try:
    post("/stop_profile", {})
except Exception:
    pass
time.sleep(20)

files = sorted(glob.glob(os.path.join(outdir, "*.trace.json*")))
print(f"[prof] trace files: {files}")
if not files:
    sys.exit("no trace produced")

tot = defaultdict(float)
cnt = defaultdict(int)
for f in files:
    op = gzip.open if f.endswith(".gz") else open
    with op(f, "rt") as fh:
        tr = json.load(fh)
    for e in tr.get("traceEvents", []):
        if e.get("ph") != "X":
            continue
        if e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        tot[e["name"]] += e.get("dur", 0.0)
        cnt[e["name"]] += 1

rank = sorted(tot.items(), key=lambda kv: -kv[1])
total = sum(tot.values())
out = {"total_us": total, "n_trace_files": len(files),
       "kernels": [{"name": k, "us": v, "calls": cnt[k]} for k, v in rank]}
with open(os.path.join(outdir, "kernel_rank.json"), "w") as fh:
    json.dump(out, fh, indent=1)

print(f"\n{'us':>12} {'%':>7} {'calls':>7} {'us/call':>9}  kernel")
for k, v in rank[:40]:
    print(f"{v:12.1f} {100*v/total:6.2f}% {cnt[k]:7d} {v/cnt[k]:9.2f}  {k[:100]}")
print(f"\ntotal device us across all ranks: {total:.1f}")
