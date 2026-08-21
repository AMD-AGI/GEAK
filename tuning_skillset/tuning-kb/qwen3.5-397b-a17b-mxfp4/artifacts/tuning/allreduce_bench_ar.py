import os, torch, torch.distributed as dist
from aiter.dist.parallel_state import (
    init_distributed_environment, ensure_model_parallel_initialized,
    get_tp_group, set_custom_all_reduce)
from aiter.dist.communication_op import tensor_model_parallel_all_reduce

rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
set_custom_all_reduce(True)
init_distributed_environment(world_size=world, rank=rank,
    distributed_init_method=f"tcp://127.0.0.1:{os.environ['AR_PORT']}")
ensure_model_parallel_initialized(world, 1)
group = get_tp_group()
print(f"rank{rank} attrs={[a for a in dir(group) if 'ca' in a or 'custom' in a]}", flush=True)

for M in (32, 64, 96, 128, 256):
    x = torch.randn((M, 4096), dtype=torch.bfloat16, device=f"cuda:{rank}")
    for _ in range(20): tensor_model_parallel_all_reduce(x)
    torch.cuda.synchronize(); dist.barrier()
    g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): tensor_model_parallel_all_reduce(x)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        for _ in range(50): tensor_model_parallel_all_reduce(x)
    for _ in range(3): g.replay()
    torch.cuda.synchronize(); dist.barrier()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record(); [g.replay() for _ in range(10)]; e1.record()
    torch.cuda.synchronize()
    us = e0.elapsed_time(e1) * 1000 / 500
    if rank == 0:
        print(f"M={M:4d}  bytes={M*4096*2/1024:7.0f} KiB   {us:7.3f} us/allreduce", flush=True)
dist.barrier(); dist.destroy_process_group()
