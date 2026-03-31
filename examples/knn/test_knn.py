# Copyright (c) OpenMMLab. All rights reserved.
import sys
import os
from pathlib import Path

# Ensure the test can find the task module when run from the task directory
sys.path.insert(0, str(Path(__file__).parent))


import torch

from knn_wrapper import knn
import time
import os

def test_knn(device):
    new_xyz = torch.tensor([[[-0.0740, 1.3147, -1.3625],
                             [-2.2769, 2.7817, -0.2334],
                             [-0.4003, 2.4666, -0.5116],
                             [-0.0740, 1.3147, -1.3625],
                             [-0.0740, 1.3147, -1.3625]],
                            [[-2.0289, 2.4952, -0.1708],
                             [-2.0668, 6.0278, -0.4875],
                             [0.4066, 1.4211, -0.2947],
                             [-2.0289, 2.4952, -0.1708],
                             [-2.0289, 2.4952, -0.1708]]]).to(device)

    xyz = torch.tensor([[[-0.0740, 1.3147, -1.3625], [0.5555, 1.0399, -1.3634],
                         [-0.4003, 2.4666,
                          -0.5116], [-0.5251, 2.4379, -0.8466],
                         [-0.9691, 1.1418,
                          -1.3733], [-0.2232, 0.9561, -1.3626],
                         [-2.2769, 2.7817, -0.2334],
                         [-0.2822, 1.3192, -1.3645], [0.1533, 1.5024, -1.0432],
                         [0.4917, 1.1529, -1.3496]],
                        [[-2.0289, 2.4952,
                          -0.1708], [-0.7188, 0.9956, -0.5096],
                         [-2.0668, 6.0278, -0.4875], [-1.9304, 3.3092, 0.6610],
                         [0.0949, 1.4332, 0.3140], [-1.2879, 2.0008, -0.7791],
                         [-0.7252, 0.9611, -0.6371], [0.4066, 1.4211, -0.2947],
                         [0.3220, 1.4447, 0.3548], [-0.9744, 2.3856,
                                                    -1.2000]]]).to(device)

    def generate_fake_point_clouds(B=8, N=1024, M=128, D=3, device='cuda'):
        # Use Normal distribution centered at 0
        xyz = torch.randn(B, N, D, device=device) * 1.0  # std=1, mean=0
        new_xyz = torch.randn(B, M, D, device=device) * 1.0
        return xyz, new_xyz

    xyz, new_xyz = generate_fake_point_clouds()

    save_dir = os.path.dirname(os.path.abspath(__file__))
    # torch.save({"tensor": xyz.detach(), "requires_grad": xyz.requires_grad}, os.path.join(save_dir, "xyz.pt"))
    # torch.save({"tensor": new_xyz.detach(), "requires_grad": new_xyz.requires_grad}, os.path.join(save_dir, "new_xyz.pt"))
    
    xyz_data = torch.load(os.path.join(save_dir, "xyz.pt"), map_location=device)
    xyz = xyz_data["tensor"].to(device).requires_grad_(xyz_data["requires_grad"])

    new_xyz_data = torch.load(os.path.join(save_dir, "new_xyz.pt"), map_location=device)
    new_xyz = new_xyz_data["tensor"].to(device).requires_grad_(new_xyz_data["requires_grad"])


    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize() 
    start.record()

    idx = knn(5, xyz, new_xyz)

    end.record()
    torch.cuda.synchronize() 
    elapsed = start.elapsed_time(end)
    print("Perf: "+ str(elapsed) + " ms")

    new_xyz_ = new_xyz.unsqueeze(2).repeat(1, 1, xyz.shape[1], 1)
    xyz_ = xyz.unsqueeze(1).repeat(1, new_xyz.shape[1], 1, 1)
    dist = ((new_xyz_ - xyz_) * (new_xyz_ - xyz_)).sum(-1)
    expected_idx = dist.topk(k=5, dim=2, largest=False)[1].transpose(2, 1)
    
    try:
        assert torch.all(idx == expected_idx)
    except Exception:
        print("Validation failed")
        sys.exit(1)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize() 
    start.record()

    idx = knn(5,
              xyz.transpose(1, 2).contiguous(),
              new_xyz.transpose(1, 2).contiguous(), True)
    
    end.record()
    torch.cuda.synchronize() 
    elapsed = start.elapsed_time(end)
    print("Perf: "+ str(elapsed) + " ms")

    try:
        assert torch.all(idx == expected_idx)
    except Exception:
        print("Validation failed")
        sys.exit(1)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize() 
    start.record()

    idx = knn(5, xyz, xyz)
    
    end.record()
    torch.cuda.synchronize() 
    elapsed = start.elapsed_time(end)
    print("Perf: "+ str(elapsed) + " ms")

    xyz_ = xyz.unsqueeze(2).repeat(1, 1, xyz.shape[1], 1)
    xyz__ = xyz.unsqueeze(1).repeat(1, xyz.shape[1], 1, 1)
    dist = ((xyz_ - xyz__) * (xyz_ - xyz__)).sum(-1)
    expected_idx = dist.topk(k=5, dim=2, largest=False)[1].transpose(2, 1)

    try:
        assert torch.all(idx == expected_idx)
    except Exception:
        print("Validation failed")
        sys.exit(1)

if __name__ == "__main__":

    test_knn('cuda')
