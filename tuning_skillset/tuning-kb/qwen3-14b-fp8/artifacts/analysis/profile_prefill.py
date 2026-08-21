#!/usr/bin/env python3
"""Capture a GPU trace of the *prefill* phase of the frozen workload.

Decode was profiled first (profile_decode.py) because it is the larger half, but at
1968 tok/s the benchmark spends ~32 s of its ~100 s wall clock outside steady-state
decode, and none of that had been looked at. This starts the profiler *before* the
requests are fired, so the captured steps are the extend/prefill batches rather than
the decode steps.

Fewer requests than the benchmark's 64 on purpose: one wave of prefill batches is
enough to see the kernel mix, and a short capture keeps the trace small enough to
summarise.
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


def fire(isl, osl, seed):
    import random

    rng = random.Random(seed)
    body = {
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
    p.add_argument("--osl", type=int, default=8, help="short: we only want the prefills")
    p.add_argument("--conc", type=int, default=16)
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--out", default=os.path.abspath("traces_prefill"))
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)

    # Arm the profiler first, then generate the work it should see.
    print(f"[profile] arming profiler for {a.steps} forward steps -> {a.out}")
    print(post("/start_profile", {
        "output_dir": a.out,
        "num_steps": a.steps,
        "activities": ["GPU"],
        "profile_id": "prefill",
    }))

    threads = [
        threading.Thread(target=fire, args=(a.isl, a.osl, i), daemon=True)
        for i in range(a.conc)
    ]
    for t in threads:
        t.start()
    print(f"[profile] {a.conc} requests fired")

    time.sleep(40)
    print("[profile] trace should be written; draining")
    for t in threads:
        t.join(timeout=600)
    print("[profile] done")


if __name__ == "__main__":
    main()
