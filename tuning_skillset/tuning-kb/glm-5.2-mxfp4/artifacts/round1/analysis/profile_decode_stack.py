#!/usr/bin/env python3
"""Capture a decode-only torch profile WITH PYTHON STACKS, to attribute a kernel to a line.

The kernel-name-only profiles in this directory answer "where does the time go"; they cannot
answer "which line launched this". `void at::native::vectorized_elementwise_kernel<8,
at::native::AUnaryFunctor<c10::BFloat16, ...>>` is 3.08% of decode at 2 calls per layer per
step, and its name says only "a bf16 tensor op with a scalar operand" -- add/sub/mul/div by a
Python number, launched from somewhere in a 2500-line model file. `with_stack=True` records the
Python frames, and the resulting trace ties each kernel launch back to a file and line.

Everything here goes through `/start_profile`'s POST body, so no environment variable and no
server flag changes: the server being profiled is configuration-identical to the server being
measured.

Two things make this a decode profile rather than a mixed one:
  - the load is a small number of long-output requests, sent and then left to run, so after the
    first few seconds there is no prefill left to capture;
  - the capture waits `--settle` seconds after the last request is accepted before arming.

`with_stack` is expensive (it walks the Python stack on every op), so `--steps` is deliberately
tiny. This profile is for attribution only -- do not read timings off it.

    python3 profile_decode_stack.py --out /tmp/prof_stack
"""
import argparse
import json
import os
import threading
import time
import urllib.request


def post(url, payload, timeout=600):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()


def fire_load(base, n, tokens, prompt_len):
    """Send `n` concurrent long-generation requests; return once they are all accepted."""
    # A fixed nonsense prompt is fine -- we are profiling the decode loop, not the content.
    prompt = "the quick brown fox jumps over the lazy dog. " * (prompt_len // 10)
    errs = []

    def one(i):
        try:
            post(
                f"{base}/generate",
                {
                    "text": prompt,
                    "sampling_params": {
                        "max_new_tokens": tokens,
                        "ignore_eos": True,
                        "temperature": 0,
                    },
                },
                timeout=1800,
            )
        except Exception as e:  # noqa: BLE001
            errs.append(repr(e))

    ts = [threading.Thread(target=one, args=(i,), daemon=True) for i in range(n)]
    for t in ts:
        t.start()
    return ts, errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=43111)
    ap.add_argument("--out", default="/tmp/prof_stack")
    ap.add_argument("--conc", type=int, default=64)
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--prompt-len", type=int, default=8000)
    ap.add_argument("--settle", type=float, default=45.0)
    ap.add_argument("--steps", type=int, default=3)
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    os.makedirs(args.out, exist_ok=True)

    print(f"[load] firing {args.conc} requests x {args.tokens} tokens")
    threads, errs = fire_load(base, args.conc, args.tokens, args.prompt_len)

    print(f"[load] settling {args.settle}s so the batch is past prefill")
    time.sleep(args.settle)

    print(f"[prof] arming with_stack profile, {args.steps} steps -> {args.out}")
    print(
        post(
            f"{base}/start_profile",
            {
                "output_dir": args.out,
                "num_steps": args.steps,
                "activities": ["CPU", "GPU"],
                "with_stack": True,
                "record_shapes": True,
            },
        )
    )
    # /start_profile with num_steps stops itself; poll for the trace to land.
    for _ in range(600):
        time.sleep(1)
        files = [f for f in os.listdir(args.out) if f.endswith(".json.gz") or f.endswith(".json")]
        if files:
            time.sleep(20)  # let every rank finish writing
            break
    print("[prof] files:", sorted(os.listdir(args.out)))
    if errs:
        print("[load] errors:", errs[:3])
    print("[load] leaving requests to drain in the background")


if __name__ == "__main__":
    main()
