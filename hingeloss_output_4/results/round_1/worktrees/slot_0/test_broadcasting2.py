import torch

torch.manual_seed(42)
batch_size = 4
input_size = 3

predictions = torch.randn(batch_size, input_size).cuda()
targets = torch.randint(0, 2, (batch_size,)).float().cuda() * 2 - 1

print("Predictions shape:", predictions.shape)
print("Predictions:\n", predictions)
print("\nTargets shape:", targets.shape)
print("Targets:", targets)

# PyTorch automatic broadcasting
result1 = predictions * targets
print("\nPyTorch broadcast result:\n", result1)

# My manual broadcasting
targets_broadcast = targets.unsqueeze(1).expand(batch_size, input_size)
print("\nManual broadcast targets:\n", targets_broadcast)
result2 = predictions * targets_broadcast
print("\nManual broadcast result:\n", result2)
