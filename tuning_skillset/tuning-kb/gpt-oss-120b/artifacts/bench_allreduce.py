#!/usr/bin/env python3
"""Race the available 2-rank all-reduce implementations at the sizes this model
actually uses.

Run: torchrun --nproc_per_node=2 /work/analysis/bench_allreduce.py

Why: after patches 01 and 02, the prefill all-reduce is the largest single item
left in prefill -- ~36% of it, ~14% of wall. The EXTEND profile shows quickreduce
twoshot at ~948 us for the 16384 x 2880 bf16 (94.4 MB) prefill tensor, an implied
99.6 GB/s of link traffic. A cross-device .copy_() tops out at 60 GB/s on this box,
but that is the SDMA path, not what a kernel doing direct peer loads can reach, so
it does not settle whether 948 us is the ceiling. This does: it races the real
implementations at the real sizes.

Arms:
  rccl   torch.distributed.all_reduce (the fallback everything else must beat)
  qr     QuickAllReduce            -- what the server picks at 94.4 MB today
  ca16   CustomAllreduce, stock max_size (16 MB on ROCm) -- declines >16 MB
  ca128  CustomAllreduce, max_size raised to 128 MB -- tests whether the in-tree
         comment "crossover is at 16MB buffer size for ROCm" still holds on MI355X

Sizes: 368,640 B is decode (64 x 2880 bf16), 94.4 MB is prefill (16384 x 2880).
The rest map the crossover so the answer is a curve, not a point.

Discipline: arms are interleaved one round each and the median is taken across
rounds; every arm is checked against the rccl result before its timing counts.
"""
import os
import statistics

import torch
import torch.distributed as dist

MB = 1024 * 1024
HIDDEN = 2880
# (label, n_tokens) -> tensor is n_tokens x 2880 bf16
SHAPES = [
    ("decode      64 tok", 64),
    ("            512 tok", 512),
    ("           2048 tok", 2048),
    ("           4096 tok", 4096),
    ("           8192 tok", 8192),
    ("prefill 16384 tok", 16384),
]
ROUNDS = 5
ITERS = 30


def bench(fn, inp, iters=ITERS):
    for _ in range(5):
        fn(inp)
    torch.cuda.synchronize()
    dist.barrier()
    ts = []
    for _ in range(iters):
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        st.record()
        fn(inp)
        en.record()
        en.synchronize()
        ts.append(st.elapsed_time(en) * 1e3)   # us
    return statistics.median(ts)


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world)
    dev = torch.device(f"cuda:{rank}")
    p0 = rank == 0

    from sglang.srt.distributed.device_communicators.custom_all_reduce import (
        CustomAllreduce,
    )
    from sglang.srt.distributed.device_communicators.quick_all_reduce import (
        QuickAllReduce,
    )

    # Both communicators do their handle exchange over a CPU group and assert the
    # group is not NCCL -- SGLang hands them GroupCoordinator.cpu_group. Same here.
    cpu_grp = dist.new_group(ranks=list(range(world)), backend="gloo")
    qr = QuickAllReduce(group=cpu_grp, device=dev)
    ca16 = CustomAllreduce(group=cpu_grp, device=dev)
    ca128 = CustomAllreduce(group=cpu_grp, device=dev, max_size=128 * MB)
    if p0:
        print(f"qr disabled={qr.disabled}  ca16 max={ca16.max_size/MB:.0f}MB "
              f"disabled={ca16.disabled}  ca128 max={ca128.max_size/MB:.0f}MB "
              f"disabled={ca128.disabled}", flush=True)

    def f_rccl(x):
        y = x.clone()
        dist.all_reduce(y)
        return y

    arms = [
        ("rccl", f_rccl),
        ("qr", lambda x: qr.quick_all_reduce(x)),
        ("ca16", lambda x: ca16.custom_all_reduce(x)),
        ("ca128", lambda x: ca128.custom_all_reduce(x)),
    ]

    if p0:
        print(f"\n{'shape':>20} {'MB':>7} " +
              " ".join(f"{n:>10}" for n, _ in arms) + "   best")
    for label, ntok in SHAPES:
        x = torch.randn(ntok, HIDDEN, dtype=torch.bfloat16, device=dev)
        nbytes = x.numel() * x.element_size()

        # correctness: every arm must match rccl
        ref = f_rccl(x)
        ok = {}
        for name, fn in arms:
            try:
                o = fn(x)
                if o is None:
                    ok[name] = None          # arm declined this size
                else:
                    err = (o.float() - ref.float()).abs().max().item()
                    den = max(ref.float().abs().max().item(), 1e-6)
                    ok[name] = (err / den) < 0.02
            except Exception as ex:
                ok[name] = f"ERR {type(ex).__name__}"
        torch.cuda.synchronize()
        dist.barrier()

        res = {}
        for _ in range(ROUNDS):
            for name, fn in arms:                     # interleaved
                if ok[name] is not True:
                    continue
                res.setdefault(name, []).append(bench(fn, x))
        med = {n: statistics.median(v) for n, v in res.items()}

        if p0:
            cells = []
            for name, _ in arms:
                if ok[name] is None:
                    cells.append(f"{'declined':>10}")
                elif ok[name] is not True:
                    cells.append(f"{str(ok[name])[:10]:>10}")
                else:
                    cells.append(f"{med[name]:9.1f}u")
            best = min(med, key=med.get) if med else "-"
            gbs = nbytes / (med[best] * 1e-6) / 1e9 if med else 0
            print(f"{label:>20} {nbytes/MB:7.2f} " + " ".join(cells) +
                  f"   {best} ({gbs:.0f} GB/s eff)", flush=True)
        del x
        torch.cuda.empty_cache()
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
