#!/usr/bin/env python3
"""Multi-rank chain timing for whole-layer monokernels (`meta.timing.kind == "rank_chain"`).

`harness_lib.time_op` times one call from one process. A monokernel that exchanges data with its
peer ranks INSIDE the kernel (a whole decoder layer at TP8/EP8: attention, o_proj + all-reduce,
router, MoE + EP exchange, next layer's QKV, one launch) cannot be timed that way: rank 0 alone
waits on its peers forever, and a harness that drives the peers from rank 0's call() puts
host-side distribution, page-table construction and result gathering inside the timed window.
That is how the first Muse Spark target-layer UT measured ~100 ms for a ~0.15 ms launch.

This module is the method of the Muse Spark layer team's frozen benchmark
(muse_layer_opt_1106/bin/bench_worker.py, `target_chain`), itself built on sgl-amd-mono's
test/kernels (test_muse_spark_layer.py `_time_graph`, test_muse_spark_k3_tp8.py):

  * one process per rank, one device each, a gloo group; every rank builds its chain OUTSIDE the
    timed window (weights, KV cache, page tables, peer buffers) through the task's
    `cases.rank_chain(rank, world, group, case)`;
  * a CHAIN of `length` distinct layers, each launch feeding the next, so the weights are as
    cold in L2/MALL as in serving (~60 layers a step) without an eviction pass;
  * every rank captures the chain (`step(length)`) and its bootstrap (`step(short)`) as CUDA
    graphs, the deployment mode; per-launch time = (T[step(length)] - T[step(short)]) /
    (length - short), cuda.Event around `iters` back-to-back replays, after a barrier;
  * every rank runs the same graphs, the same replay counts, in the same order (the in-kernel
    mailboxes pair the ranks), and every rank reports; the bucket's number is the MAX over ranks
    (a step waits for the slowest rank), with the per-rank medians kept in the receipt;
  * the guard / error words are checked after eager validation and after timing: a wait that
    timed out is a failed measurement, never a slow one.

The task supplies, in cases.py:

    def rank_chain(rank, world, group, case) -> chain
        chain.length          int, launches in the full chain (>= 2)
        chain.short           int, launches in the bootstrap-only step (default 0)
        chain.step(n)         enqueue a graph-safe step of n chained launches on the current stream,
                              including any bootstrap launch and the epoch bump; no host sync, no
                              allocation (it is captured)
        chain.check()         raise if a device-side wait failed (guard / epoch error words)
        chain.close()         release peer buffers (optional)

Run as a leg: leg_runner --mode time calls `time_bucket` (in the leg's process, which never
touches a GPU itself); it spawns `world` workers of this file pinned to the leg's visible devices.
"""
import json
import os
import socket
import subprocess
import sys
import tempfile
import time

DEFAULTS = {"world": 8, "reps": 5, "iters": 20, "warmup": 3, "host_probe": 30,
            "init_timeout_s": 600, "worker_timeout_s": 1800}
TIMER = "rank_chain_graph"


def chain_spec(meta):
    """The task's rank_chain timing declaration with defaults, or None for single-process timing."""
    t = (meta or {}).get("timing") or {}
    if t.get("kind") != "rank_chain":
        return None
    spec = dict(DEFAULTS)
    spec.update({k: v for k, v in t.items() if k != "kind"})
    spec["kind"] = "rank_chain"
    spec["world"] = int(spec["world"])
    if spec["world"] < 1:
        raise ValueError("timing.world must be >= 1")
    return spec


def timed_rows(task_dir, bucket=""):
    with open(os.path.join(task_dir, "run", "ut_cases.json")) as f:
        manifest = json.load(f)
    rows = [c for c in manifest.get("cases", []) if c.get("timed")]
    return [c for c in rows if not bucket or c["sig"] == bucket]


def visible_devices(world):
    """The logical device indices the leg may use, `world` of them, in rank order."""
    pinned = os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES")
    devs = [d for d in pinned.split(",") if d.strip()] if pinned else [str(i) for i in range(world)]
    if len(devs) < world:
        raise RuntimeError("rank_chain timing needs %d devices, the leg sees %s" % (world, devs))
    return devs[:world]


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _kill(procs):
    for p in procs:
        if p.poll() is None:
            try:
                os.killpg(p.pid, 9)
            except OSError:
                pass


def time_bucket(task_dir, case, meta, spec=None):
    """Time one manifest row on `world` ranks; returns leg_runner's time-mode row."""
    spec = spec or chain_spec(meta)
    world = spec["world"]
    devs = visible_devices(world)
    port = _free_port()
    out = tempfile.mkdtemp(prefix="rank_chain_")
    procs = []
    try:
        for r in range(world):
            env = dict(os.environ, HIP_VISIBLE_DEVICES=devs[r], CUDA_VISIBLE_DEVICES=devs[r],
                       RANK_CHAIN_SPEC=json.dumps(spec))
            procs.append(subprocess.Popen(
                [sys.executable, os.path.abspath(__file__), "--worker", task_dir, case["sig"],
                 str(r), str(world), str(port), out],
                env=env, stdout=sys.stderr, stderr=sys.stderr, start_new_session=True))
        deadline = time.time() + float(spec["worker_timeout_s"])
        while any(p.poll() is None for p in procs):
            if time.time() > deadline or any(p.poll() not in (None, 0) for p in procs):
                break       # one rank died: the others would wait on it in-kernel until their deadline
            time.sleep(0.2)
        _kill(procs)
        codes = [p.wait() for p in procs]
        path = os.path.join(out, "result.json")
        if any(codes) or not os.path.exists(path):
            raise RuntimeError("rank_chain workers for %s exited %s" % (case["sig"], codes))
        with open(path) as f:
            res = json.load(f)
    finally:
        _kill(procs)
        for f in os.listdir(out) if os.path.isdir(out) else []:
            os.unlink(os.path.join(out, f))
        os.rmdir(out)
    res.update({"sig": case["sig"], "regime": case.get("regime", ""), "m": case.get("m")})
    return res


# ----------------------------------------------------------------------------- the rank workers
def _load(name, path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def summarize(per_rank, spec):
    """Fold the ranks' receipts: the bucket is as slow as its slowest rank."""
    ms = max(r["ms"] for r in per_rank)
    host = max(r["host_ms"] for r in per_rank)
    return {"ms": ms, "wall_ms": None, "timer": TIMER,
            "host_ms": host, "primed": bool(host < ms),
            "per_rank_ms": [r["ms"] for r in per_rank],
            "samples_ms": [r["samples_ms"] for r in per_rank],
            "cache_condition": {"mode": "distinct-layer-chain", "length": per_rank[0]["length"],
                                "short": per_rank[0]["short"], "outside_timing": True,
                                "deployment_calibrated": False},
            "rank_chain": {k: spec[k] for k in ("world", "reps", "iters", "warmup")}}


def _worker(task_dir, sig, rank, world, port, out):
    # the task dir holds a unittest.py that shadows the stdlib module torch imports
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != os.path.abspath(task_dir)]
    import datetime
    import statistics

    import torch
    import torch.distributed as dist

    spec = json.loads(os.environ["RANK_CHAIN_SPEC"])
    torch.cuda.set_device(0)
    dist.init_process_group("gloo", init_method="tcp://127.0.0.1:%d" % port, rank=rank, world_size=world,
                            timeout=datetime.timedelta(seconds=float(spec["init_timeout_s"])))
    cases = _load("cases", os.path.join(task_dir, "cases.py"))
    row = next(c for c in timed_rows(task_dir) if c["sig"] == sig)
    chain = cases.rank_chain(rank, world, dist.group.WORLD, row)
    n_full, n_short = int(chain.length), int(getattr(chain, "short", 0))
    if n_full - n_short < 1:
        raise ValueError("chain.length must exceed chain.short")
    iters, warmup = int(spec["iters"]), int(spec["warmup"])

    def capture(n):
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=st):
            chain.step(n)
        torch.cuda.synchronize()
        return g

    def gtime(g):
        for _ in range(warmup):
            g.replay()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            g.replay()
        e.record()
        e.synchronize()
        return s.elapsed_time(e) / iters

    try:
        chain.step(n_full)                 # eager once: builds / loads every launch before capture
        torch.cuda.synchronize()
        chain.check()
        dist.barrier()
        g_full, g_short = capture(n_full), capture(n_short)
        dist.barrier()
        samples = []
        for _ in range(int(spec["reps"])):
            dist.barrier()
            tf = gtime(g_full)
            dist.barrier()
            ts = gtime(g_short)
            samples.append((tf - ts) / (n_full - n_short))
        chain.check()
        # host cost of issuing the chain, per launch: every rank issues the same replays, so the
        # in-kernel waits pair; enqueue time only (no sync inside the window)
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(int(spec["host_probe"])):
            g_full.replay()
        host = (time.perf_counter() - t0) * 1e3 / int(spec["host_probe"]) / max(1, n_full)
        torch.cuda.synchronize()
        chain.check()
        mine = {"rank": rank, "ms": statistics.median(samples), "samples_ms": samples,
                "host_ms": host, "length": n_full, "short": n_short}
    finally:
        torch.cuda.synchronize()
        close = getattr(chain, "close", None)
        if close:
            close()
    got = [None] * world
    dist.all_gather_object(got, mine)
    if rank == 0:
        res = summarize(sorted(got, key=lambda r: r["rank"]), spec)
        tmp = os.path.join(out, "result.json.tmp")
        with open(tmp, "w") as f:
            json.dump(res, f)
        os.replace(tmp, os.path.join(out, "result.json"))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    if len(sys.argv) == 8 and sys.argv[1] == "--worker":
        _worker(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), sys.argv[7])
        sys.exit(0)
    sys.exit(__doc__)
