#!/usr/bin/env python3
"""Off-contract load driver, for profiling only.

Reproduces the *shape* of the frozen workload (ISL 8192, OSL 1024, concurrency 64) so a
profiler window sees production-shaped prefill and decode batches, but it is NOT the
benchmark: no result json, no throughput claim. Anything measured with this is a probe.

  python3 analysis/probe_load.py --n 64 --conc 64 --osl 1024
"""
import argparse, asyncio, json, random, time

import aiohttp

AP = argparse.ArgumentParser()
AP.add_argument("--url", default="http://127.0.0.1:43101/v1/completions")
AP.add_argument("--model", default="/shared_nfs/hyperloom/models/Llama-3.1-8B-Instruct")
AP.add_argument("--n", type=int, default=64)
AP.add_argument("--conc", type=int, default=64)
AP.add_argument("--isl", type=int, default=8192)
AP.add_argument("--osl", type=int, default=1024)
AP.add_argument("--seed", type=int, default=0)
A = AP.parse_args()

rng = random.Random(A.seed)
# token ids straight through: the completions endpoint accepts a list of ints as prompt,
# which keeps the input length exact without depending on a tokenizer round-trip.
VOCAB = 128000


async def one(sess, sem, i):
    prompt = [rng.randint(10, VOCAB - 1) for _ in range(A.isl)]
    body = {
        "model": A.model,
        "prompt": prompt,
        "max_tokens": A.osl,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": False,
    }
    async with sem:
        t0 = time.perf_counter()
        async with sess.post(A.url, json=body) as r:
            await r.json()
        return time.perf_counter() - t0


async def main():
    sem = asyncio.Semaphore(A.conc)
    to = aiohttp.ClientTimeout(total=3600)
    async with aiohttp.ClientSession(timeout=to) as sess:
        t0 = time.perf_counter()
        lat = await asyncio.gather(*[one(sess, sem, i) for i in range(A.n)])
        dt = time.perf_counter() - t0
    print(
        json.dumps(
            {
                "n": A.n,
                "conc": A.conc,
                "isl": A.isl,
                "osl": A.osl,
                "wall_s": round(dt, 3),
                "out_tok_s": round(A.n * A.osl / dt, 2),
                "mean_e2e_s": round(sum(lat) / len(lat), 3),
            }
        )
    )


asyncio.run(main())
