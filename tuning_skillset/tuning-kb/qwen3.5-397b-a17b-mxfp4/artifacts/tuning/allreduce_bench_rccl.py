#!/usr/bin/env python3
"""Baseline the 4-GPU all-reduce with plain RCCL, at the prefill message sizes.

The prefill profile shows quickreduce (CodecQ8) taking 801 us for the 16384x4096
bf16 hidden state (134 MB) -- 30.4% of all prefill device time. That number
includes any wait-for-peer skew, so it is not by itself evidence that the
collective is slow. This measures the same sizes with plain torch.distributed
(RCCL) under a barrier, where all ranks are known to arrive together, to get an
achievable-bandwidth reference for the node.

    torchrun --nproc_per_node=4 analysis/allreduce/bench_rccl.py
"""
import os
import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", rank=rank, world_size=world)

TOKENS = [1024, 2048, 4096, 8192, 16384, 32768]
HID = 4096
ITERS = 30

if rank == 0:
    print(f"{'tokens':>7} {'MB':>8} {'us':>9} {'GB/s bus':>9}")

for t in TOKENS:
    x = torch.randn((t, HID), dtype=torch.bfloat16, device=f"cuda:{rank}")
    nbytes = x.numel() * 2
    for _ in range(10):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dist.barrier()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(ITERS):
        dist.all_reduce(x)
    e1.record()
    torch.cuda.synchronize()
    us = e0.elapsed_time(e1) * 1000 / ITERS
    # ring all-reduce bus bandwidth: 2*(n-1)/n * bytes moved per GPU
    busgb = 2 * (world - 1) / world * nbytes / (us * 1e-6) / 1e9
    if rank == 0:
        print(f"{t:7d} {nbytes/1e6:8.1f} {us:9.2f} {busgb:9.1f}")
    dist.barrier()

dist.destroy_process_group()
