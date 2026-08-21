#!/usr/bin/env python3
"""Patch 03's guard admits any bs >= 32, but the sweep only measured bs=64.
Check the boundary before shipping it: bs=64 (steady state) and bs=32 (the tail
as requests finish), default config vs the tuned one, interleaved.

Run with BS set by the harness below, not from the environment.
"""
import os
import statistics
import sys

DEFAULT = dict(TILE_SIZE=64, NUM_SEGMENTS_PER_SEQ=16, num_warps=2,
               waves_per_eu=2, num_stages=2)
TUNED = dict(TILE_SIZE=64, NUM_SEGMENTS_PER_SEQ=4, num_warps=4,
             waves_per_eu=2, num_stages=3)
ROUNDS, ITERS = 5, 40


def run_for_bs(bs):
    os.environ["BS"] = str(bs)
    for m in [m for m in list(sys.modules) if m.startswith("bench_unified_attn")]:
        del sys.modules[m]
    sys.path.insert(0, "/work/analysis")
    import bench_unified_attn as B

    t = B.build_inputs()
    # correctness of the tuned arm against the shipped default
    ref = B.run_cfg_once(None, t)
    o = B.run_cfg_once(TUNED, t)
    err = (o.float() - ref.float()).abs().max().item()
    den = max(ref.float().abs().max().item(), 1e-6)
    ok = (err / den) < 0.02

    s = {"default": [], "tuned": []}
    for _ in range(ROUNDS):                       # interleaved
        s["default"].append(B.time_cfg(DEFAULT, t, ITERS))
        s["tuned"].append(B.time_cfg(TUNED, t, ITERS))
    d = statistics.median(s["default"]) * 1000
    u = statistics.median(s["tuned"]) * 1000
    print(f"bs={bs:3d}  default {d:7.2f} us   tuned {u:7.2f} us   "
          f"{d/u:5.3f}x   correctness {'ok' if ok else 'FAIL'}", flush=True)
    return d / u


if __name__ == "__main__":
    for bs in (64, 32):
        run_for_bs(bs)
