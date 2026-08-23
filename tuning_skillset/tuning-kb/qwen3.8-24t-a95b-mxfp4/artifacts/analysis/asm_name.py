import torch, aiter
from aiter import gemm_a16w16_asm
from aiter.ops.gemm_op_common import get_padded_m
from torch.profiler import profile, ProfilerActivity
dev="cuda"; dt=torch.bfloat16
M,N,K,sk = 64,4608,8192,2
x=torch.randn(M,K,dtype=dt,device=dev); w=torch.randn(N,K,dtype=dt,device=dev)
out=torch.empty(M,N,dtype=dt,device=dev)
for _ in range(5): gemm_a16w16_asm(x,w,out,None,sk,None,False)
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as p:
    for _ in range(5): gemm_a16w16_asm(x,w,out,None,sk,None,False)
    torch.cuda.synchronize()
names={}
for e in p.events():
    if e.device_type==torch.autograd.DeviceType.CUDA and e.self_device_time_total>0:
        names[e.key]=names.get(e.key,0)+e.self_device_time_total
for k,v in sorted(names.items(),key=lambda x:-x[1]): print(f"{v/5:9.2f}us  {k}")
print()
# padded_M buckets
for gl in (0,1):
    prev=None
    for M_ in [1,2,4,8,16,32,48,56,60,64,65,72,80,96,128,192,256,384,512,1024,2048,4096,8192,16384]:
        pm=get_padded_m(M_,4608,8192,gl)
        print(f"gl={gl} M={M_:6d} -> padded {pm}")
    print()
