#!/usr/bin/env python3
"""Dump topk_softmax outputs to a file so the stock and patched kernels can be compared
bit-for-bit across the same shape grid.

Used as the correctness evidence for patches/topk_softmax_ldg_width.patch. The patch changes
only the width of the vector loads (and therefore how many lanes cooperate on a row), not
the arithmetic or the reduction order, so the outputs should be *identical*, not merely
close -- and anything less than identical would mean the dispatch is wrong somewhere.

    python3 analysis/topk/dump_topk.py /tmp/topk_stock.pt     # with the stock .so in place
    # rebuild with the patch, then
    python3 analysis/topk/dump_topk.py /tmp/topk_patched.pt
    python3 -c "import torch; a=torch.load('/tmp/topk_stock.pt'); b=torch.load('/tmp/topk_patched.pt'); \
        print(all(x.equal(y) for k in a for x,y in zip(a[k],b[k])))"

Covers M in {1,3,64,65,128,255,256,512,8192,16384} x {bf16,fp16,fp32} at E=512, K=10 --
straddling the num_rows <= 2048 dispatch boundary in both directions and including the
non-power-of-two and sub-wavefront row counts. Result: True, every tensor equal. The
independent reference check against torch.topk+torch.softmax is check_topk.py.
"""
import sys, torch
from aiter.ops.moe_op import topk_softmax
dev="cuda:3"; torch.cuda.set_device(dev); torch.manual_seed(0)
E,K=512,10; out={}
for M in (1,3,64,65,128,255,256,512,8192,16384):
    for dt in (torch.bfloat16, torch.float16, torch.float32):
        g=torch.randn(M,E,dtype=dt,device=dev)
        w=torch.empty(M,K,dtype=torch.float32,device=dev)
        i=torch.empty(M,K,dtype=torch.int32,device=dev)
        t=torch.empty(M,K,dtype=torch.int32,device=dev)
        topk_softmax(w,i,t,g,True)
        out[f"{M}_{dt}"]=(w.cpu(),i.cpu(),t.cpu())
torch.save(out, sys.argv[1])
print("dumped", sys.argv[1])
