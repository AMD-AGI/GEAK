import torch, time, statistics
torch.cuda.init()
n = 256*1024*1024//4                      # 256 MB buffers -> DRAM bound, above 32MB L3
x = torch.rand(n, device='cuda'); y = torch.rand(n, device='cuda')
def bw():
    for _ in range(3): o = x + y
    torch.cuda.synchronize()
    t0=time.perf_counter()
    for _ in range(20): o = x + y
    torch.cuda.synchronize()
    return 3*n*4/((time.perf_counter()-t0)/20)/1e9
samples=[]
t_start=time.time()
for i in range(12):
    v=bw(); samples.append(v)
    print("  rep %2d  %6.1f GB/s   t+%4.0fs" % (i, v, time.time()-t_start), flush=True)
m=statistics.mean(samples); sd=statistics.stdev(samples)
print("mean=%.1f  sd=%.2f  CV=%.2f%%  min=%.1f  max=%.1f  spread=%.2f%%"
      % (m, sd, 100*sd/m, min(samples), max(samples), 100*(max(samples)-min(samples))/m))
print("=> 小于 %.1f%% 的差异不可区分于噪声 (2 sigma)" % (2*100*sd/m))
