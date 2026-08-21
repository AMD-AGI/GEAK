import torch, json, re
from collections import defaultdict
from torch.profiler import profile, ProfilerActivity
from aiter.tuned_gemm import tgemm
dev='cuda:0'; dt=torch.bfloat16; torch.cuda.set_device(dev)
def run(nm,M,N,K):
    x=torch.randn(M,K,device=dev,dtype=dt); w=torch.randn(N,K,device=dev,dtype=dt)
    for _ in range(20): tgemm.mm(x,w,None,otype=dt)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(50): tgemm.mm(x,w,None,otype=dt)
        torch.cuda.synchronize()
    p.export_chrome_trace("/tmp/m.json")
    d=defaultdict(lambda:[0.0,0])
    for e in json.load(open("/tmp/m.json"))["traceEvents"]:
        if e.get("ph")=="X" and e.get("cat")=="kernel":
            n=re.sub(r"<.*>","",e["name"]).split("::")[-1]
            d[n][0]+=e["dur"]; d[n][1]+=1
    tot=sum(v[0] for v in d.values())/50
    parts="; ".join(f"{v[0]/50:.2f}us x{v[1]//50} {kk[:52]}" for kk,v in sorted(d.items(),key=lambda x:-x[1][0]))
    bytes_w = N*K*2
    print(f"{nm:22s} {tot:7.2f}us  {bytes_w/1e6/ (tot/1e6) /1e3:7.2f} GB/s-w | {parts}")
    del x,w; torch.cuda.empty_cache()
for N in (32,256,512,1024,1025,1040,1536,2048,4608):
    run(f"M64_N{N}_K8192",64,N,8192)
run("M64_N8192_K2048",64,8192,2048); run("M64_N8192_K256",64,8192,256)
