#!/usr/bin/env python3
"""Profile the live SGLang server's decode path.

Drives a small steady-state decode load at the benchmark's concurrency, brackets
a window of it with the server's /start_profile /stop_profile endpoints, then
ranks kernels by SUMMED DEVICE DURATION from the torch trace's kernel events
(not by any percentage column -- see tuning-core/graph_captured_benchmarking.md).
"""
import argparse, json, os, threading, time, urllib.request, glob, gzip, sys
from collections import defaultdict

P = argparse.ArgumentParser()
P.add_argument("--port", type=int, default=43103)
P.add_argument("--conc", type=int, default=64)
P.add_argument("--isl", type=int, default=8192)
P.add_argument("--osl", type=int, default=1024)
P.add_argument("--out", default="analysis/prof")
P.add_argument("--warm-s", type=float, default=25.0, help="decode seconds before capture")
P.add_argument("--steps", type=int, default=8, help="profiler decode steps to record")
A = P.parse_args()

BASE = f"http://127.0.0.1:{A.port}"
MODEL = os.environ.get("MODEL", "/shared_nfs/hyperloom/models/Qwen3.5-397B-A17B-MXFP4")

def post(path, payload=None, timeout=1800):
    data = json.dumps(payload or {}).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()

# Deterministic token ids, same shape as the benchmark's random dataset.
import random
rng = random.Random(0)
def prompt_ids():
    return [rng.randint(1000, 200000) for _ in range(A.isl)]

def worker(stop):
    while not stop.is_set():
        try:
            post("/v1/completions", {"model": MODEL, "prompt": prompt_ids(),
                                     "max_tokens": A.osl, "temperature": 0.0,
                                     "ignore_eos": True, "stream": False})
        except Exception as e:
            if not stop.is_set():
                print("worker error:", e, file=sys.stderr)
            return

stop = threading.Event()
threads = [threading.Thread(target=worker, args=(stop,), daemon=True) for _ in range(A.conc)]
print(f"[prof] launching {A.conc} streams, warming {A.warm_s}s into steady-state decode")
for t in threads: t.start()
time.sleep(A.warm_s)

os.makedirs(A.out, exist_ok=True)
outdir = os.path.abspath(A.out)
print(f"[prof] capturing {A.steps} decode steps -> {outdir}")
post("/start_profile", {"output_dir": outdir, "num_steps": A.steps,
                        "activities": ["CPU", "GPU"], "profile_by_stage": False})
# num_steps auto-stops; give it room to flush the trace.
time.sleep(60)
try:
    post("/stop_profile", {})
except Exception:
    pass
time.sleep(20)
stop.set()
print("[prof] draining workers")
time.sleep(5)

files = sorted(glob.glob(os.path.join(outdir, "*.trace.json*")))
print(f"[prof] trace files: {files}")
if not files:
    sys.exit("no trace produced")

# Rank by summed device duration per kernel name, from the kernel-category
# events only. Host API rows (hipGraphLaunch etc.) live in different pids/cats
# and must not be mixed in.
tot = defaultdict(float); cnt = defaultdict(int)
for f in files:
    op = gzip.open if f.endswith(".gz") else open
    with op(f, "rt") as fh:
        tr = json.load(fh)
    for e in tr.get("traceEvents", []):
        if e.get("ph") != "X":
            continue
        cat = (e.get("cat") or "").lower()
        if cat not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        n = e.get("name", "?")
        tot[n] += e.get("dur", 0.0); cnt[n] += 1

grand = sum(tot.values())
print(f"\n[prof] total device time {grand/1000:.2f} ms over {sum(cnt.values())} dispatches\n")
print("%8s %7s %10s %9s  %s" % ("us", "%", "calls", "us/call", "kernel"))
rank = sorted(tot.items(), key=lambda kv: -kv[1])
for n, d in rank[:45]:
    print("%8.0f %6.2f%% %10d %9.2f  %s" % (d, 100*d/grand, cnt[n], d/cnt[n], n[:110]))

with open(os.path.join(outdir, "kernel_rank.json"), "w") as fh:
    json.dump({"total_us": grand,
               "kernels": [{"name": n, "us": d, "calls": cnt[n]} for n, d in rank]}, fh, indent=1)
print(f"\n[prof] ranking -> {outdir}/kernel_rank.json")
