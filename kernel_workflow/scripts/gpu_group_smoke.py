#!/usr/bin/env python3
"""Multi-GPU lease smoke: every rank of the leased group joins one all_reduce.

    bash gpu_lock.sh pool:2:<ids> torchrun --nproc_per_node=2 gpu_group_smoke.py
"""

import os

import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    expected_world_size = len(os.environ["GEAK_GPU_GROUP"].split(","))
    if world_size != expected_world_size:
        raise AssertionError(
            f"WORLD_SIZE={world_size}, leased GPU count={expected_world_size}"
        )
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    value = torch.tensor(float(rank), device=f"cuda:{local_rank}")
    dist.all_reduce(value)
    expected = world_size * (world_size - 1) / 2
    if value.item() != expected:
        raise AssertionError(
            f"rank {rank}: all_reduce={value.item()}, expected={expected}"
        )
    dist.barrier()
    if rank == 0:
        print(f"GPU_GROUP_SMOKE_PASS world_size={world_size}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
