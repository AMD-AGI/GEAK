"""Is one N=1024 GEMM cheaper than the two N=512 GEMMs it would replace?

Compares against what the stack does today (torch, twice) and against the best
tuned N=512 row, and prices the .contiguous() the two output slices need.
"""
import json, statistics, torch, torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
from aiter.tuned_gemm import tgemm
dev, dt = "cuda:0", torch.bfloat16
torch.cuda.set_device(dev)

def us(fn, iters):
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(iters): fn()
        torch.cuda.synchronize()
    p.export_chrome_trace("/tmp/cf.json")
    ev = json.load(open("/tmp/cf.json"))["traceEvents"]
    return sum(e["dur"] for e in ev if e.get("ph")=="X" and e.get("cat")=="kernel")/iters

print(f"{'M':>6} {'2x torch N512':>14} {'2x tgemm N512':>14} {'1x tgemm N1024':>15} "
      f"{'+2 contig':>10} {'vs torch':>9} {'vs tuned':>9}")
for M in (64, 8192, 16384):
    x  = torch.randn(M, 8192, device=dev, dtype=dt)/8
    w1 = torch.randn(512, 8192, device=dev, dtype=dt)/8
    w2 = torch.randn(512, 8192, device=dev, dtype=dt)/8
    wf = torch.cat([w1, w2], 0).contiguous()
    n = 512
    def two_torch(): F.linear(x, w1); F.linear(x, w2)
    def two_tuned(): tgemm.mm(x, w1, None, otype=dt); tgemm.mm(x, w2, None, otype=dt)
    def one():       tgemm.mm(x, wf, None, otype=dt)
    def one_c():
        y = tgemm.mm(x, wf, None, otype=dt)
        return y[:, :n].contiguous(), y[:, n:].contiguous()
    # correctness of the fused form
    y = tgemm.mm(x, wf, None, otype=dt)
    r1, r2 = F.linear(x, w1).float(), F.linear(x, w2).float()
    e = max((y[:, :n].float()-r1).abs().max().item(), (y[:, n:].float()-r2).abs().max().item())
    it = max(3, min(40, int(40*64/M))) if M > 512 else 40
    for _ in range(10): two_torch(); two_tuned(); one_c()
    res = {k: statistics.median([us(f, it) for _ in range(5)])
           for k, f in (("tt",two_torch),("tu",two_tuned),("o",one),("oc",one_c))}
    print(f"{M:6d} {res['tt']:14.2f} {res['tu']:14.2f} {res['o']:15.2f} "
          f"{res['oc']:10.2f} {res['tt']/res['oc']:8.3f}x {res['tu']/res['oc']:8.3f}x   err={e:.1e}")
    del x, w1, w2, wf, y, r1, r2; torch.cuda.empty_cache()
