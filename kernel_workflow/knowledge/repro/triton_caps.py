import sys

import torch, triton, triton.language as tl, traceback, time
dev='cuda'
print("triton", triton.__version__, "| arch", torch.cuda.get_device_properties(0).gcnArchName)
R={}
def case(name):
    def deco(fn):
        try:
            fn(); R[name]=("PASS","")
        except Exception as e:
            R[name]=("FAIL", f"{type(e).__name__}: {str(e)[:110]}")
        return fn
    return deco

# 1. elementwise (baseline, already known good)
@case("elementwise")
def _():
    @triton.jit
    def k(x,o,n,B:tl.constexpr):
        i=tl.program_id(0)*B+tl.arange(0,B); m=i<n
        tl.store(o+i, tl.load(x+i,mask=m)*2.0, mask=m)
    x=torch.rand(1<<16,device=dev); o=torch.empty_like(x)
    k[(64,)](x,o,x.numel(),B=1024); torch.cuda.synchronize()
    assert torch.allclose(o,x*2)

# 2. reduction (softmax-like row reduce)
@case("reduction/softmax")
def _():
    @triton.jit
    def k(x,o,S,B:tl.constexpr):
        r=tl.program_id(0); i=tl.arange(0,B); m=i<S
        v=tl.load(x+r*S+i,mask=m,other=-float('inf'))
        v=v-tl.max(v,0); e=tl.exp(v); tl.store(o+r*S+i, e/tl.sum(e,0), mask=m)
    x=torch.randn(64,512,device=dev); o=torch.empty_like(x)
    k[(64,)](x,o,512,B=512); torch.cuda.synchronize()
    assert torch.allclose(o, torch.softmax(x,-1), atol=1e-3)

# 3. tl.dot fp16 -- THE critical one (needs WMMA on RDNA)
@case("tl.dot fp16 (WMMA)")
def _():
    @triton.jit
    def k(a,b,c,M,N,K,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
        pm=tl.program_id(0); pn=tl.program_id(1)
        om=pm*BM+tl.arange(0,BM); on=pn*BN+tl.arange(0,BN)
        acc=tl.zeros((BM,BN),dtype=tl.float32)
        for k0 in range(0,K,BK):
            ok=k0+tl.arange(0,BK)
            av=tl.load(a+om[:,None]*K+ok[None,:], mask=(om[:,None]<M)&(ok[None,:]<K), other=0.)
            bv=tl.load(b+ok[:,None]*N+on[None,:], mask=(ok[:,None]<K)&(on[None,:]<N), other=0.)
            acc+=tl.dot(av,bv)
        tl.store(c+om[:,None]*N+on[None,:], acc.to(tl.float16), mask=(om[:,None]<M)&(on[None,:]<N))
    M=N=K=256
    a=torch.randn(M,K,device=dev,dtype=torch.float16); b=torch.randn(K,N,device=dev,dtype=torch.float16)
    c=torch.empty(M,N,device=dev,dtype=torch.float16)
    k[(M//64,N//64)](a,b,c,M,N,K,BM=64,BN=64,BK=32); torch.cuda.synchronize()
    ref=(a.float()@b.float()).half()
    err=(c.float()-ref.float()).abs().max().item()
    assert err<2.0, f"max_err={err}"

# 4. tl.dot bf16
@case("tl.dot bf16")
def _():
    @triton.jit
    def k(a,b,c,M,N,K,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
        pm=tl.program_id(0); pn=tl.program_id(1)
        om=pm*BM+tl.arange(0,BM); on=pn*BN+tl.arange(0,BN)
        acc=tl.zeros((BM,BN),dtype=tl.float32)
        for k0 in range(0,K,BK):
            ok=k0+tl.arange(0,BK)
            av=tl.load(a+om[:,None]*K+ok[None,:]); bv=tl.load(b+ok[:,None]*N+on[None,:])
            acc+=tl.dot(av,bv)
        tl.store(c+om[:,None]*N+on[None,:], acc.to(tl.bfloat16))
    M=N=K=256
    a=torch.randn(M,K,device=dev,dtype=torch.bfloat16); b=torch.randn(K,N,device=dev,dtype=torch.bfloat16)
    c=torch.empty(M,N,device=dev,dtype=torch.bfloat16)
    k[(4,4)](a,b,c,M,N,K,BM=64,BN=64,BK=32); torch.cuda.synchronize()

# 5. autotune (GEAK relies on it heavily)
@case("triton.autotune")
def _():
    @triton.autotune(configs=[triton.Config({'B':256},num_warps=4),
                              triton.Config({'B':1024},num_warps=8)], key=['n'])
    @triton.jit
    def k(x,o,n,B:tl.constexpr):
        i=tl.program_id(0)*B+tl.arange(0,B); m=i<n
        tl.store(o+i, tl.load(x+i,mask=m)+1.0, mask=m)
    x=torch.rand(1<<20,device=dev); o=torch.empty_like(x)
    k[lambda M:(triton.cdiv(x.numel(),M['B']),)](x,o,x.numel()); torch.cuda.synchronize()
    assert torch.allclose(o,x+1)

# 6. atomics
@case("atomics")
def _():
    @triton.jit
    def k(x,o,n,B:tl.constexpr):
        i=tl.program_id(0)*B+tl.arange(0,B); m=i<n
        tl.atomic_add(o, tl.sum(tl.load(x+i,mask=m,other=0.),0))
    x=torch.ones(1<<14,device=dev); o=torch.zeros(1,device=dev)
    k[(16,)](x,o,x.numel(),B=1024); torch.cuda.synchronize()
    assert abs(o.item()-x.numel())<1.0, o.item()

for n,(s,m) in R.items():
    print(f"  {n:24} {s}  {m}")

_p = sum(1 for s,_ in R.values() if s == "PASS")
_f = sum(1 for s,_ in R.values() if s == "FAIL")
print("SUMMARY: %d PASS / %d FAIL" % (_p, _f))

# Exit status has to agree with the summary, or a caller checking $? reads a
# failing capability probe as a green one.
if _f:
    print("FAIL: %d Triton capability case(s) failed on this part" % _f)
    sys.exit(1)
if not _p:
    print("FAIL: no capability case ran at all")
    sys.exit(1)
sys.exit(0)
