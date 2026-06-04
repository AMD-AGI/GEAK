# Copyright (c) OpenMMLab. All rights reserved.
import sys
import os
from pathlib import Path

# Ensure the test can find the task module when run from the task directory
sys.path.insert(0, str(Path(__file__).parent))


import torch

from ball_query_wrapper import ball_query

import time
import os

def test_ball_query(device):
    new_xyz = torch.tensor(
        [[[-0.0740, 1.3147, -1.3625], [-2.2769, 2.7817, -0.2334],
          [-0.4003, 2.4666, -0.5116], [-0.0740, 1.3147, -1.3625],
          [-0.0740, 1.3147, -1.3625]],
         [[-2.0289, 2.4952, -0.1708], [-2.0668, 6.0278, -0.4875],
          [0.4066, 1.4211, -0.2947], [-2.0289, 2.4952, -0.1708],
          [-2.0289, 2.4952, -0.1708]]],
        device=device)

    xyz = torch.tensor(
        [[[-0.0740, 1.3147, -1.3625], [0.5555, 1.0399, -1.3634],
          [-0.4003, 2.4666, -0.5116], [-0.5251, 2.4379, -0.8466],
          [-0.9691, 1.1418, -1.3733], [-0.2232, 0.9561, -1.3626],
          [-2.2769, 2.7817, -0.2334], [-0.2822, 1.3192, -1.3645],
          [0.1533, 1.5024, -1.0432], [0.4917, 1.1529, -1.3496]],
         [[-2.0289, 2.4952, -0.1708], [-0.7188, 0.9956, -0.5096],
          [-2.0668, 6.0278, -0.4875], [-1.9304, 3.3092, 0.6610],
          [0.0949, 1.4332, 0.3140], [-1.2879, 2.0008, -0.7791],
          [-0.7252, 0.9611, -0.6371], [0.4066, 1.4211, -0.2947],
          [0.3220, 1.4447, 0.3548], [-0.9744, 2.3856, -1.2000]]],
        device=device)

    # B=4
    # M=1024
    # N=128

    # xyz = torch.rand(B, N, 3, device=device) - 0.3 * 9  # scale to [0, 10)
    # new_xyz = torch.rand(B, M, 3, device=device) - 0.3 * 9

    save_dir = os.path.dirname(os.path.abspath(__file__))

    # torch.save({"tensor": xyz.detach(), "requires_grad": xyz.requires_grad}, os.path.join(save_dir, "xyz.pt"))
    # torch.save({"tensor": new_xyz.detach(), "requires_grad": new_xyz.requires_grad}, os.path.join(save_dir, "new_xyz.pt"))
    
    # xyz_data = torch.load(os.path.join(save_dir, "xyz.pt"), map_location=device)
    # xyz = xyz_data["tensor"].to(device).requires_grad_(xyz_data["requires_grad"])

    # new_xyz_data = torch.load(os.path.join(save_dir, "new_xyz.pt"), map_location=device)
    # new_xyz = new_xyz_data["tensor"].to(device).requires_grad_(new_xyz_data["requires_grad"])

    def generate_pointcloud_like_data(B=4, N=16384, M=2048, space_size=20.0, cluster_radius=0.5, device='cuda'):
        """
        Generates synthetic point clouds mimicking real-world distributions.
        - B: batch size
        - N: number of points in xyz
        - M: number of query points
        - space_size: overall spatial extent of the scene
        - cluster_radius: radius within which query points are sampled (denser region)
        """
        # Simulate full 3D scene: uniformly distributed base cloud
        xyz = (torch.rand(B, N, 3, device=device) - 0.5) * space_size  # in range [-10, 10]^3

        # Simulate queries centered around denser regions
        cluster_centers = (torch.rand(B, M, 3, device=device) - 0.5) * space_size
        offsets = (torch.rand(B, M, 3, device=device) - 0.5) * cluster_radius * 2
        new_xyz = cluster_centers + offsets  # Dense neighborhoods

        return xyz.contiguous(), new_xyz.contiguous()

    B, N, M = 4, 16384, 2048
    xyz, new_xyz = generate_pointcloud_like_data(B, N, M, device=device)

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
    
    idx = ball_query(0, 0.2, 5, xyz, new_xyz)
    
    end.record()
    torch.cuda.synchronize() 
    elapsed = start.elapsed_time(end)
    print("Perf: "+ str(elapsed) + " ms")

    expected_idx = torch.tensor(
        [[[0, 0, 0, 0, 0], [6, 6, 6, 6, 6], [2, 2, 2, 2, 2], [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0]],
         [[0, 0, 0, 0, 0], [2, 2, 2, 2, 2], [7, 7, 7, 7, 7], [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0]]],
        device=device)
    

    # torch.save(idx.detach().cpu(), os.path.join(save_dir, 'expected_idx.pt')) 
    expected_idx = torch.load(os.path.join(save_dir, 'expected_idx.pt'), map_location='cpu', weights_only=True)

    try:
        assert torch.all(idx.cpu() == expected_idx)
    except Exception:
        print("Validation failed")
        sys.exit(1)

    # test dilated ball query
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()  # Ensure previous kernels are done
    start.record()

    idx = ball_query(0.2, 0.4, 5, xyz, new_xyz)

    end.record()
    torch.cuda.synchronize()  # Wait for kernel to finish
    elapsed = start.elapsed_time(end)  # in milliseconds
    print("Perf: "+ str(elapsed) + " ms")


    expected_idx = torch.tensor(
        [[[0, 5, 7, 0, 0], [6, 6, 6, 6, 6], [2, 3, 2, 2, 2], [0, 5, 7, 0, 0],
          [0, 5, 7, 0, 0]],
         [[0, 0, 0, 0, 0], [2, 2, 2, 2, 2], [7, 7, 7, 7, 7], [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0]]],
        device=device)
    
    # torch.save(idx.detach().cpu(), os.path.join(save_dir, 'expected_idx_1.pt')) 
    expected_idx = torch.load(os.path.join(save_dir, 'expected_idx_1.pt'), map_location='cpu', weights_only=True)

    try:
        assert torch.all(idx.cpu() == expected_idx)
    except Exception:
        print("Validation failed")
        sys.exit(1)


if __name__ == "__main__":
    test_ball_query("cuda")
