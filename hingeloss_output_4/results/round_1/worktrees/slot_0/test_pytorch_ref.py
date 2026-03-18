import torch
import sys
sys.path.insert(0, '/workspace/GEAK/HingeLoss')
from hingeloss import Model as PyTorchModel

torch.manual_seed(42)
batch_size = 128
input_size = 128

predictions = torch.randn(batch_size, input_size).cuda()
targets = torch.randint(0, 2, (batch_size,)).float().cuda() * 2 - 1

model = PyTorchModel().cuda()
result = model(predictions, targets)

print("PyTorch Model result:", result.item())

# Manual computation
manual_result = torch.mean(torch.clamp(1 - predictions * targets, min=0))
print("Manual computation:", manual_result.item())

# Check if they match
print("Do they match?", torch.allclose(result, manual_result))
