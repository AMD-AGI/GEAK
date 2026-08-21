#!/usr/bin/env python3
"""Measure directed peer-to-peer bandwidth between all 4 GPUs, to find out whether the
TP=4 all-reduce is barrier-bound or fabric-bound.

The prefill profile shows quickreduce (CodecQ8) at 801 us for the 16384x4096 bf16 hidden
state -- 30.4% of prefill device time -- which looks like an obvious target until you know
what the wire can actually do. This copies 134 MB (the exact prefill message size) across
every ordered (src, dst) pair and prints the 4x4 matrix.

    python3 analysis/hw/p2p_bandwidth.py

Result: ~60.5 GB/s per directed xGMI link, ~181 GB/s of egress per GPU with all three peers
active. Against that, RCCL all-reduce (analysis/allreduce/bench_rccl.py) runs at 94% of the
link and aiter's quickreduce-Q8 is already 1.47x RCCL because it compresses. The collective
is fabric-bound, not software-bound; see FINDINGS.md section 5. This measurement, plus the
-3.75% from patches/rejected/allreduce_1stage_crossover_1mib.patch, is why the collective
was closed out as a source of headroom.
"""
import torch, time
N = 134*1024*1024//2   # 134 MB of bf16
res={}
for src in range(4):
    for dst in range(4):
        if src==dst: continue
        a=torch.empty(N,dtype=torch.bfloat16,device=f'cuda:{src}')
        b=torch.empty(N,dtype=torch.bfloat16,device=f'cuda:{dst}')
        torch.cuda.set_device(src)
        for _ in range(5): b.copy_(a)
        torch.cuda.synchronize(src)
        t=time.perf_counter()
        for _ in range(20): b.copy_(a)
        torch.cuda.synchronize(src); torch.cuda.synchronize(dst)
        dt=(time.perf_counter()-t)/20
        res[(src,dst)]=N*2/dt/1e9
        del a,b; torch.cuda.empty_cache()
print("unidirectional P2P GB/s (rows=src, cols=dst)")
print("     " + "".join(f"{d:>9}" for d in range(4)))
for s in range(4):
    print(f"{s:>4} " + "".join(f"{res.get((s,d),0):9.1f}" for d in range(4)))
