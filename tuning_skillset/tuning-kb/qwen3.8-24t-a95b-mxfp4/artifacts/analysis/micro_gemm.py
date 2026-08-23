import torch, torch.nn.functional as F, time
dev='cuda:0'; dt=torch.bfloat16
def bench(fn, it=200, warm=50):
    for _ in range(warm): fn()
    torch.cuda.synchronize(dev)
    g=torch.cuda.CUDAGraph()
    t=time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(dev); return (time.perf_counter()-t)/it*1e6

for M in (64, 8192, 16384):
    x=torch.randn(M,8192,device=dev,dtype=dt)
    wr=torch.randn(512,8192,device=dev,dtype=dt)   # router gate
    ws=torch.randn(512,8192,device=dev,dtype=dt)   # shared gate_up
    wg=torch.randn(1,8192,device=dev,dtype=dt)     # shared_expert_gate
    wf=torch.cat([wr,ws,wg],0).contiguous()        # fused 1025
    wf2=torch.cat([wr,ws],0).contiguous()          # fused 1024
    a=bench(lambda: F.linear(x,wr)); b=bench(lambda: F.linear(x,ws))
    c=bench(lambda: F.linear(x,wg)); f=bench(lambda: F.linear(x,wf))
    f2=bench(lambda: F.linear(x,wf2))
    print(f"M={M:6d}  router {a:7.2f}  sharedgu {b:7.2f}  sgate {c:7.2f}  SUM {a+b+c:7.2f} | fused1025 {f:7.2f} | fused1024 {f2:7.2f}+sgate = {f2+c:7.2f}")
    del x,wr,ws,wg,wf,wf2; torch.cuda.empty_cache()

# reference shapes
for M,K,N,nm in ((64,8192,4608,'in_proj'),(64,2048,8192,'out_proj'),(64,256,8192,'shared_down')):
    x=torch.randn(M,K,device=dev,dtype=dt); w=torch.randn(N,K,device=dev,dtype=dt)
    print(f"{nm:12s} M{M} K{K} N{N}: {bench(lambda: F.linear(x,w)):7.2f} us")
    del x,w; torch.cuda.empty_cache()
