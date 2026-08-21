import torch, time, os
import torch.nn.functional as F
from aiter.tuned_gemm import tgemm, get_GEMM_A16W16_config
dev='cuda:0'; dt=torch.bfloat16; torch.cuda.set_device(dev)

def gbench(fn, inner=30, it=40, warm=8):
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(inner): fn()
    for _ in range(warm): g.replay()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(it): g.replay()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/(it*inner)*1e6

SHAPES=[("in_proj",4608,8192),("out_proj",8192,2048),("router_gate",512,8192),
        ("shared_gate_up",512,8192),("shared_down",8192,256),("in_proj_ba",32,8192),
        ("lm_head",31040,8192)]
print(f"{'name':16s} {'M':>6s} {'N':>6s} {'K':>5s} {'libtype':>10s} {'aiter_us':>9s} {'torch_us':>9s} {'gain':>7s}")
for M in (64, 8192, 16384):
    for nm,N,K in SHAPES:
        if nm=="lm_head" and M>64: continue
        x=torch.randn(M,K,device=dev,dtype=dt); w=torch.randn(N,K,device=dev,dtype=dt)
        cfg=get_GEMM_A16W16_config(M=M,N=N,K=K,bias=False,dtype=str(dt),otype=str(dt))
        a=gbench(lambda: tgemm.mm(x,w,None,otype=dt))
        b=gbench(lambda: F.linear(x,w))
        print(f"{nm:16s} {M:6d} {N:6d} {K:5d} {cfg['libtype']:>10s} {a:9.2f} {b:9.2f} {a/b:7.2f}x")
        del x,w; torch.cuda.empty_cache()
