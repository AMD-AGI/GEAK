#!/usr/bin/env python3
"""Race quickreduce against the custom allreduce at the sizes this model actually uses.

    torchrun --nproc_per_node=8 /work/analysis/bench_allreduce.py

Needs the server DOWN -- it wants all 8 GPUs.

Why this exists: at concurrency 64 the decode allreduce carries 64 x 8192 x bf16 = 1 MiB and
lands on cross_device_reduce_2stage, 13.4% of decode. Prefill carries 256 MiB and gets
qr_all_reduce instead. The only thing separating them is a table of constants in
quick_all_reduce.py, and ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16 defaults to 1, so the row that
actually applies is _QR_MIN_SIZE[(float16, 8)] = [16, 4, 4, 2] MB -- the *fp16* row, not the
bf16 one, whatever the tensor's real dtype. With ROCM_QUICK_REDUCE_QUANTIZATION=INT8 from the
image that is a 4 MB floor, and 1 MiB is under it.

Measurement follows tuning-core/measurement.md Rule 6b: on gfx950 a back-to-back A/B drifts,
so the two arms are interleaved round by round and compared on medians, never on one shot.
Every rank times itself and rank 0 reports the max across ranks, because a collective is only
as fast as its slowest participant.
"""
import os
import statistics
import sys

import torch
import torch.distributed as dist

from sglang.srt.distributed import init_distributed_environment
from sglang.srt.distributed.parallel_state import (
    get_tensor_model_parallel_group,
    initialize_model_parallel,
)

REPS = 7          # rounds of interleaved A/B; medians over these
ITERS = 200       # calls per timed round
WARMUP = 50

# (label, bytes). The two that matter are decode (1 MiB) and prefill (256 MiB); the rest
# bracket the 4 MB threshold so the shape of the crossover is visible rather than assumed.
SIZES = [
    ("decode bs=64", 64 * 8192 * 2),
    ("bs=128", 128 * 8192 * 2),
    ("bs=256", 256 * 8192 * 2),
    ("2 MiB", 2 * 1024 * 1024),
    ("4 MiB (threshold)", 4 * 1024 * 1024),
    ("8 MiB", 8 * 1024 * 1024),
    ("prefill 16384", 16384 * 8192 * 2),
]


def timeit(fn, iters, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    dist.barrier()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters      # us per call


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world)
    init_distributed_environment(
        world_size=world, rank=rank, local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{os.environ.get('MASTER_PORT', 29500)}",
        backend="nccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=world)
    group = get_tensor_model_parallel_group()
    ca = group.ca_comm       # custom allreduce (cross_device_reduce_*)
    qr = group.qr_comm       # quickreduce

    if rank == 0:
        print(f"# world={world}  ca={'ok' if ca and not ca.disabled else 'DISABLED'}  "
              f"qr={'ok' if qr and not qr.disabled else 'DISABLED'}")
        if qr and not qr.disabled:
            print(f"# qr regime={qr.qr_quant_level.name} cast_bf16_to_fp16={qr.use_fp16_kernels} "
                  f"max_size={qr.qr_max_size/1024/1024:.0f} MiB")
        print(f"# {'size':>18} {'bytes':>10} {'qr us':>9} {'ca us':>9} {'rccl us':>9} "
              f"{'qr/ca':>7} {'qr err':>9}")

    for label, nbytes in SIZES:
        numel = nbytes // 2
        inp = torch.randn(numel, dtype=torch.bfloat16, device=f"cuda:{rank}") / 8
        out = torch.empty_like(inp)

        # what the stack would pick on its own, before any patch
        would_qr = qr.should_quick_allreduce(inp) if qr and not qr.disabled else False

        qr_fn = (lambda: qr.quick_all_reduce(inp, out=out)) if qr and not qr.disabled else None
        ca_fn = (lambda: ca.custom_all_reduce(inp)) if ca and not ca.disabled else None
        rc_ten = inp.clone()
        rc_fn = lambda: dist.all_reduce(rc_ten, group=group.device_group)

        # accuracy of the INT8 arm against an exact reduction, on the real tensor
        err = float("nan")
        if qr_fn is not None:
            ref = inp.float().clone()
            dist.all_reduce(ref, group=group.device_group)
            got = qr.quick_all_reduce(inp).float()
            err = ((got - ref).abs().max() / ref.abs().max().clamp(min=1e-6)).item()

        a, b, c = [], [], []
        for _ in range(REPS):                    # interleaved: drift hits every arm equally
            if qr_fn is not None:
                a.append(timeit(qr_fn, ITERS, warmup=10))
            if ca_fn is not None:
                b.append(timeit(ca_fn, ITERS, warmup=10))
            c.append(timeit(rc_fn, ITERS, warmup=10))

        def worst(vals):
            """median over rounds on this rank, then max over ranks."""
            if not vals:
                return float("nan")
            t = torch.tensor([statistics.median(vals)], device=f"cuda:{rank}")
            dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group.device_group)
            return t.item()

        qus, caus, rcus = worst(a), worst(b), worst(c)
        if rank == 0:
            ratio = caus / qus if qus == qus and qus > 0 else float("nan")
            print(f"  {label:>18} {nbytes:>10} {qus:9.2f} {caus:9.2f} {rcus:9.2f} "
                  f"{ratio:6.3f}x {err:9.2e}   {'(stack picks qr)' if would_qr else '(stack picks ca)'}")
        del inp, out, rc_ten
        torch.cuda.empty_cache()

    dist.barrier()
    if rank == 0:
        print("# done")


if __name__ == "__main__":
    main()
