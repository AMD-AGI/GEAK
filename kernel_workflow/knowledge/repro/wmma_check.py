import torch, triton, triton.language as tl, time, re
dev='cuda'
@triton.jit
def mm(a,b,c,M,N,K,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pm=tl.program_id(0); pn=tl.program_id(1)
    om=pm*BM+tl.arange(0,BM); on=pn*BN+tl.arange(0,BN)
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    for k0 in range(0,K,BK):
        ok=k0+tl.arange(0,BK)
        av=tl.load(a+om[:,None]*K+ok[None,:]); bv=tl.load(b+ok[:,None]*N+on[None,:])
        acc+=tl.dot(av,bv)
    tl.store(c+om[:,None]*N+on[None,:], acc.to(tl.float16))

M=N=K=2048
a=torch.randn(M,K,device=dev,dtype=torch.float16); b=torch.randn(K,N,device=dev,dtype=torch.float16)
c=torch.empty(M,N,device=dev,dtype=torch.float16)
BM=BN=128; BK=64
grid=(M//BM, N//BN)
h = mm[grid](a,b,c,M,N,K,BM=BM,BN=BN,BK=BK)
torch.cuda.synchronize()

asm = h.asm.get('amdgcn','')
wmma = len(re.findall(r'v_wmma\w*', asm))
mfma = len(re.findall(r'v_mfma\w*', asm))
fma  = len(re.findall(r'v_fma[c_]?_f\d+', asm))
print("ASM: v_wmma=%d  v_mfma=%d  v_fma=%d" % (wmma, mfma, fma))
kinds = sorted(set(re.findall(r'v_wmma\w+', asm)))
print("WMMA opcodes:", kinds[:4])

# correctness
ref=(a.float()@b.float())
err=(c.float()-ref).abs().max().item(); rel=err/ref.abs().max().item()
print("correctness: max_abs_err=%.3f rel=%.2e" % (err, rel))

# throughput
for _ in range(5): mm[grid](a,b,c,M,N,K,BM=BM,BN=BN,BK=BK)
torch.cuda.synchronize(); t0=time.perf_counter()
R=50
for _ in range(R): mm[grid](a,b,c,M,N,K,BM=BM,BN=BN,BK=BK)
torch.cuda.synchronize()
dt=(time.perf_counter()-t0)/R
flops=2*M*N*K
print("GEMM %dx%dx%d: %.3f ms -> %.2f TFLOP/s (fp16)" % (M,N,K,dt*1e3, flops/dt/1e12))
# torch reference for comparison
torch.cuda.synchronize(); t0=time.perf_counter()
for _ in range(R): r=a@b
torch.cuda.synchronize(); dt2=(time.perf_counter()-t0)/R
print("torch (hipBLASLt):     %.3f ms -> %.2f TFLOP/s   [triton/torch = %.2fx]" % (dt2*1e3, flops/dt2/1e12, dt2/dt))
