"""Graph-captured, clock-drift-tolerant A/B harness for decode-sized kernels.

Two problems the stock `tuning_skillset/benchmark/graph_harness.py` does not solve for a
~15 us kernel, both of which showed up immediately on this box:

1. **Down-clocking.** A graph holding ONE invocation, replayed with a sync after each
   replay, leaves the GPU idle most of the time. Measured qkv_proj at 29 us that way
   against 16 us in the reference trace -- the kernel was not slow, the part was asleep.
   Fix: replicate the op `reps` times inside the graph so a replay is milliseconds of
   back-to-back work, and sync once per replay, not once per op.

2. **The monotonic clock ramp** documented in tuning-core/clocks_and_power.md (~13-17%,
   still climbing 2000 GEMMs in; unpinnable in this container because sysfs is read-only).
   Fix: interleave -- run every candidate once per round for `rounds` rounds and take each
   candidate's *minimum* round median, so all candidates see the same clock history.

Capture validity is guarded the same way the skill prescribes: dirty the output, replay,
verify it was recomputed. `mode` must read "cudagraph" or the number is thrown away.
"""

from __future__ import annotations

import statistics
from typing import Callable

import torch


class CaptureInvalid(RuntimeError):
    pass


def _capture(step: Callable[[], object], reps: int) -> torch.cuda.CUDAGraph:
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            step()
    return g


def bench_one(
    step: Callable[[], object],
    *,
    reps: int = 50,
    warmup: int = 30,
    iters: int = 20,
    dirty: Callable[[], None] | None = None,
    verify: Callable[[], bool] | None = None,
) -> dict:
    """Capture `reps` invocations, time replays, return per-invocation microseconds."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            step()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    g = _capture(step, reps)

    if dirty is not None and verify is not None:
        dirty()
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        if not verify():
            raise CaptureInvalid("graph replay did not recompute a correct result")

    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        g.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0 / reps)  # us per invocation
    del g
    return {
        "mode": "cudagraph",
        "us_median": statistics.median(times),
        "us_min": min(times),
        "us": times,
    }


def race(
    cands: dict[str, Callable[[], object]],
    *,
    rounds: int = 5,
    reps: int = 50,
    warmup: int = 30,
    iters: int = 12,
    guards: dict[str, tuple[Callable[[], None], Callable[[], bool]]] | None = None,
) -> dict[str, dict]:
    """Interleaved A/B: every candidate once per round, `rounds` rounds.

    Returns per-candidate {us: best round median, rounds: [...], mode/err}.
    Taking the minimum over rounds removes the clock ramp: the winner is whichever
    candidate is fastest once the part has woken up, and every candidate gets the
    same number of chances to be measured late.
    """
    guards = guards or {}
    # Pre-capture each candidate's graph once so capture cost is out of the loop.
    graphs: dict[str, torch.cuda.CUDAGraph] = {}
    out: dict[str, dict] = {}
    for name, step in cands.items():
        try:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(warmup):
                    step()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            g = _capture(step, reps)
            if name in guards:
                d, v = guards[name]
                d()
                torch.cuda.synchronize()
                g.replay()
                torch.cuda.synchronize()
                if not v():
                    raise CaptureInvalid("empty/invalid graph")
            graphs[name] = g
            out[name] = {"mode": "cudagraph", "rounds": []}
        except Exception as e:  # noqa: BLE001
            out[name] = {"mode": f"FAILED: {type(e).__name__}: {e}", "rounds": []}

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(rounds):
        for name, g in graphs.items():
            for _ in range(2):
                g.replay()
            torch.cuda.synchronize()
            ts = []
            for _ in range(iters):
                start.record()
                g.replay()
                end.record()
                torch.cuda.synchronize()
                ts.append(start.elapsed_time(end) * 1000.0 / reps)
            out[name]["rounds"].append(statistics.median(ts))

    for name in out:
        r = out[name]["rounds"]
        if r:
            out[name]["us"] = min(r)
            out[name]["us_last"] = r[-1]
            out[name]["spread_pct"] = 100 * (max(r) - min(r)) / min(r)
    for g in graphs.values():
        del g
    graphs.clear()
    return out
