import torch, torch.nn.functional as F, time
dev='cuda:0'; dt=torch.bfloat16; torch.cuda.set_device(dev)
def gbench(fn, inner=50, it=50, warm=10):
    s=torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(inner): fn()
    for _ in range(warm): g.replay()
    torch.cuda.synchronize()
    t=time.perf_counter()
    for _ in range(it): g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter()-t)/(it*inner)*1e6

M=64
x=torch.randn(M,8192,device=dev,dtype=dt)
ws={n: torch.randn(n,8192,device=dev,dtype=dt) for n in (512,1,1025,1040,1024)}
r=gbench(lambda: F.linear(x,ws[512]))
g1=gbench(lambda: F.linear(x,ws[1]))
f25=gbench(lambda: F.linear(x,ws[1025]))
f40=gbench(lambda: F.linear(x,ws[1040]))
f24=gbench(lambda: F.linear(x,ws[1024]))
print(f"M=64  N512 {r:6.2f}  N1 {g1:6.2f}  -> unfused sum {2*r+g1:6.2f}")
print(f"      N1024 {f24:6.2f}  N1025 {f25:6.2f}  N1040 {f40:6.2f}")
# with the split that the real code would do
w=ws[1040]
def fused_split():
    y=F.linear(x,w); return y[:, :512], y[:, 512:1024], y[:, 1024:1025]
print(f"      fused1040+3 slices {gbench(fused_split):6.2f}")
# reference: in_proj / out_proj / shared_down at M=64
for K,N,nm in ((8192,4608,'in_proj'),(2048,8192,'out_proj'),(256,8192,'shared_down')):
    xx=torch.randn(M,K,device=dev,dtype=dt); ww=torch.randn(N,K,device=dev,dtype=dt)
    print(f"      {nm:12s} K{K} N{N}: {gbench(lambda: F.linear(xx,ww)):6.2f} us")
