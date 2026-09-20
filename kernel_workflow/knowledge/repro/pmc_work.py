import torch
a = torch.randn(512, 512, device="cuda", dtype=torch.float16)
b = torch.randn(512, 512, device="cuda", dtype=torch.float16)
for _ in range(5):
    c = a @ b
torch.cuda.synchronize()
print("ok", float(c[0,0]))
