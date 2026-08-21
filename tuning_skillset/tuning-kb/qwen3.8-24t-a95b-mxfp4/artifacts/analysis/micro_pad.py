import torch, torch.nn.functional as F, time
dev='cuda:0'; dt=torch.bfloat16
def bench(fn, it=300, warm=80):
    for _ in range(warm): fn()
    torch.cuda.synchronize(dev); t=time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(dev); return (time.perf_counter()-t)/it*1e6
for M in (64,16384):
    x=torch.randn(M,8192,device=dev,dtype=dt)
    print(f"--- M={M}")
    for N in (1025,1026,1032,1040,1056,1088,1152,1280,1536):
        w=torch.randn(N,8192,device=dev,dtype=dt)
        print(f"   N={N:5d}: {bench(lambda: F.linear(x,w)):7.2f} us")
        del w; torch.cuda.empty_cache()
    del x; torch.cuda.empty_cache()
