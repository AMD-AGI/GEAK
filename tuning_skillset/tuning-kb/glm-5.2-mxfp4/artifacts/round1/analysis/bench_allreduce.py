#!/usr/bin/env python3
"""Compare the three 8-way all-reduce paths at this model's real message sizes.

Why this exists
---------------
The decode profile puts 8.92% of decode GPU time in `cross_device_reduce_2stage` (sglang's
`CustomAllreduce`, the aiter "ca" path) at 13.22 us per call, while *prefill* spends 15.1% in
INT4 quickreduce ("qr"). Two different implementations for the same collective, chosen by size.
That asymmetry is worth checking rather than assuming, because the launch flags already set
`ROCM_QUICK_REDUCE_QUANTIZATION=INT4`, so qr is built and initialised at decode too -- it simply
never gets asked.

Two independent gates keep qr out of decode, and both are source, not configuration:

  1. `parallel_state._resolve_outplace_all_reduce_method` tests `ca` BEFORE `qr`. At the decode
     message size `should_custom_ar` returns True, so `ca` wins and `qr` is never consulted.
  2. Even if the order were flipped, `QuickAllReduce.should_quick_allreduce` applies a minimum
     size. `ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16` defaults to 1, so a bf16 tensor is looked up in
     the fp16 row: `_QR_MIN_SIZE[(float16, 8)][INT4] = 2 MB`. Decode is 768 KB. Blocked.

So answering "should decode use quickreduce?" needs both gates removed, which is a real source
change -- but it is only worth making if qr is actually faster at 768 KB. That is what this
measures, in isolation, against the same two implementations the server uses.

Message sizes come from the model, not from a sweep:
    decode   64 x 6144 bf16 =  768 KB   (concurrency 64, hidden 6144)
    prefill  16384 x 6144 bf16 = 192 MB (one prefill chunk)
plus a scan between them to locate the crossover, if there is one.

Method
------
`torchrun --nproc-per-node 8`. Each rank builds a gloo group (both classes require a non-NCCL
group for their handle exchange) and instantiates `CustomAllreduce` and `QuickAllReduce` exactly
as `parallel_state` does. Every candidate is timed inside a HIP graph of 50 back-to-back calls:
a single 13 us collective is faster than the host can launch it, so an eager loop measures
dispatch, not the kernel -- the same trap that produced a fake 1.31x in the decode split-K
harness. Results are reduced across ranks with max(), since a collective is as slow as its
slowest participant.

Correctness is checked too: quickreduce at INT4 is *lossy by construction*, so the reported
rel-L2 against a torch.distributed reference is a property of the candidate, not a bug.

    torchrun --nproc-per-node 8 analysis/bench_allreduce.py
"""
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, "/sgl-workspace/sglang/python")

from sglang.srt.distributed.device_communicators.custom_all_reduce import (  # noqa: E402
    CustomAllreduce,
)
from sglang.srt.distributed.device_communicators.quick_all_reduce import (  # noqa: E402
    QuickAllReduce,
)

HIDDEN = 6144
# (label, num_tokens). 64 = the decode batch at concurrency 64; 16384 = one prefill chunk.
SHAPES = [
    ("decode 64", 64),
    ("128", 128),
    ("256", 256),
    ("512", 512),
    ("1024", 1024),
    ("2048", 2048),
    ("4096", 4096),
    ("prefill 16384", 16384),
]
ITERS = 50
REPEATS = 7


def capture(fn, warm, inp):
    """Capture ITERS back-to-back calls into one HIP graph. `warm` is the eager-safe variant."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            warm(inp)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    # thread_local capture mode: the NCCL process group runs a watchdog thread, and in the
    # default global capture mode any allocation it makes during capture deadlocks the process.
    with torch.cuda.graph(g, capture_error_mode="thread_local"):
        for _ in range(ITERS):
            fn(inp)
    return g


def eager_time(fn, inp):
    """Median-of-REPEATS us per call, eager. Includes host dispatch -- an upper bound only.

    Used for the rccl reference, which cannot be captured here: graphing an NCCL collective
    alongside the IPC communicators deadlocked reliably (see capture_error_mode above), and rccl
    is only present as a sanity check that the custom paths are worth having at all.
    """
    for _ in range(5):
        fn(inp)
    torch.cuda.synchronize()
    ts = []
    for _ in range(REPEATS):
        dist.barrier()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(ITERS):
            fn(inp)
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000.0 / ITERS)
    ts.sort()
    return ts[len(ts) // 2]


def replay_time(g):
    """Median-of-REPEATS us per call."""
    ts = []
    for _ in range(REPEATS):
        dist.barrier()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000.0 / ITERS)
    ts.sort()
    return ts[len(ts) // 2]


def maxreduce(v):
    t = torch.tensor([v], dtype=torch.float64, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    cpu_group = dist.new_group(list(range(world)), backend="gloo")

    ca = CustomAllreduce(group=cpu_group, device=rank)
    qr = QuickAllReduce(group=cpu_group, device=rank)
    if rank == 0:
        print(f"ca disabled={ca.disabled} max_size={ca.max_size/1024**2:.1f} MB")
        print(
            f"qr disabled={qr.disabled} "
            f"quant={getattr(qr, 'qr_quant_level', None)} "
            f"fp16_kernels={getattr(qr, 'use_fp16_kernels', None)} "
            f"max_size={getattr(qr, 'qr_max_size', 0)/1024**2:.1f} MB"
        )
        print(
            f"{'shape':>16} {'bytes':>10} | {'rccl':>9} {'ca':>9} {'qr':>9} | "
            f"{'gate: ca':>9} {'gate: qr':>9} | {'qr relL2':>9}"
        )

    def rccl(x):
        y = x.clone()
        dist.all_reduce(y)
        return y

    # Phase 1: allocate and capture everything. `ca.capture()` must wrap the graph capture so
    # that `register_graph_buffers` runs on exit -- that is what makes the registered `ca` path
    # (the one the server actually uses inside its decode graph) legal to replay.
    state = []
    with ca.capture():
        for label, ntok in SHAPES:
            inp = torch.randn(ntok, HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.02
            ref = inp.clone()
            dist.all_reduce(ref)
            graphs = {}
            # ca is hard-capped at 16 MB (`_MAX_CAR_SIZE`) and its staging buffer is that size,
            # so above the cap it cannot even be called -- which is exactly why prefill's 192 MB
            # all-reduce falls through to qr in the first place.
            if not ca.disabled and ca.should_custom_ar(inp):
                graphs["ca"] = capture(
                    lambda x: ca._all_reduce_impl(x, registered=True),
                    lambda x: ca._all_reduce_impl(x, registered=False),
                    inp,
                )
            if not qr.disabled:
                graphs["qr"] = capture(qr.quick_all_reduce, qr.quick_all_reduce, inp)
            state.append((label, inp, ref, graphs))

    # Phase 2: replay and time. Correctness is read from an eager call, not from the graph.
    for label, inp, ref, graphs in state:
        nbytes = inp.numel() * inp.element_size()
        t = {k: maxreduce(replay_time(g)) for k, g in graphs.items()}
        t["rccl"] = maxreduce(eager_time(rccl, inp))
        out_qr = qr.quick_all_reduce(inp) if not qr.disabled else None
        rel = (
            ((out_qr.float() - ref.float()).norm() / ref.float().norm()).item()
            if out_qr is not None
            else float("nan")
        )
        gate_ca = ca.should_custom_ar(inp) if not ca.disabled else False
        gate_qr = qr.should_quick_allreduce(inp) if not qr.disabled else False
        if rank == 0:
            print(
                f"{label:>16} {nbytes:>10} | {t.get('rccl', float('nan')):9.2f} "
                f"{t.get('ca', float('nan')):9.2f} {t.get('qr', float('nan')):9.2f} | "
                f"{str(gate_ca):>9} {str(gate_qr):>9} | {rel:9.2e}"
            )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
