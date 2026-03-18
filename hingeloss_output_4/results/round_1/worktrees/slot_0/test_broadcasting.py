import torch

torch.manual_seed(42)
batch_size = 128
input_size = 128

predictions = torch.randn(batch_size, input_size).cuda()
targets = torch.randint(0, 2, (batch_size,)).float().cuda() * 2 - 1

# PyTorch automatic broadcasting
result1 = predictions * targets
print("PyTorch broadcast shape:", result1.shape)
print("First row:", result1[0, :5])

# My manual broadcasting
targets_broadcast = targets.unsqueeze(1).expand(batch_size, input_size).clone()
result2 = predictions * targets_broadcast
print("\nManual broadcast shape:", result2.shape)
print("First row:", result2[0, :5])

print("\nAre they equal?", torch.allclose(result1, result2))

# Check the hinge loss
loss1 = torch.mean(torch.clamp(1 - result1, min=0))
loss2 = torch.mean(torch.clamp(1 - result2, min=0))
print("\nLoss with PyTorch broadcast:", loss1.item())
print("Loss with manual broadcast:", loss2.item())
