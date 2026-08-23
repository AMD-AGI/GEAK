#!/usr/bin/env python3
"""Capture a GPU trace of the *steady-state decode* phase of the frozen workload.

Drives 64 concurrent ISL-8192/OSL-1024 requests at the server, waits until prefill
is done and every request is decoding, then asks SGLang to profile N decode steps.
The point is to measure where the 25.5 ms decode step actually goes, rather than
trusting the analytic split in the bundle's reference material.
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.request

PORT = int(os.environ.get("PORT", 43102))
BASE = f"http://127.0.0.1:{PORT}"


def post(path, payload=None, timeout=1800):
    data = json.dumps(payload or {}).encode()
    req = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()


def fire(isl, osl, n, seed):
    """Same shape the benchmark uses: random token ids, ignore_eos, fixed length."""
    import random

    rng = random.Random(seed)
    body = {
        # Token ids straight through /generate avoids any tokenizer variance.
        "input_ids": [rng.randrange(1000, 100000) for _ in range(isl)],
        "sampling_params": {"temperature": 0, "max_new_tokens": osl, "ignore_eos": True},
    }
    try:
        post("/generate", body)
    except Exception as e:  # noqa: BLE001
        print(f"  request failed: {e}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--isl", type=int, default=8192)
    p.add_argument("--osl", type=int, default=1024)
    p.add_argument("--conc", type=int, default=64)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--settle", type=float, default=45.0,
                   help="seconds to wait for all prefills to complete before profiling")
    p.add_argument("--out", default=os.path.abspath("traces"))
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    threads = [
        threading.Thread(target=fire, args=(a.isl, a.osl, 1, i), daemon=True)
        for i in range(a.conc)
    ]
    for t in threads:
        t.start()

    print(f"[profile] {a.conc} requests in flight; settling {a.settle}s so prefill finishes")
    time.sleep(a.settle)

    print(f"[profile] capturing {a.steps} decode steps -> {a.out}")
    print(post("/start_profile", {
        "output_dir": a.out,
        "num_steps": a.steps,
        "activities": ["GPU"],
        "profile_id": "decode",
    }))

    # num_steps auto-stops; give the writer time to flush the trace.
    time.sleep(30)
    print("[profile] trace written; waiting for the driver requests to drain")
    for t in threads:
        t.join(timeout=600)
    print("[profile] done")


if __name__ == "__main__":
    main()
