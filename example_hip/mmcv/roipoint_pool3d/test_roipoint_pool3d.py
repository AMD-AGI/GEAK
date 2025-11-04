# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from roipoint_pool3d_wrapper import RoIPointPool3d
import time
import os
import math

def test_roipoint(device, dtype):
    points = torch.tensor(
        [[1, 2, 3.3], [1.2, 2.5, 3.0], [0.8, 2.1, 3.5], [1.6, 2.6, 3.6],
         [0.8, 1.2, 3.9], [-9.2, 21.0, 18.2], [3.8, 7.9, 6.3],
         [4.7, 3.5, -12.2], [3.8, 7.6, -2], [-10.6, -12.9, -20], [-16, -18, 9],
         [-21.3, -52, -5], [0, 0, 0], [6, 7, 8], [-2, -3, -4]],
        dtype=dtype).unsqueeze(0).to(device)
    feats = points.clone()
    rois = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.3],
                          [-10.0, 23.0, 16.0, 10, 20, 20, 0.5]]],
                        dtype=dtype).to(device)


    # Settings
    B = 2       # batch size
    N = 5000    # number of points per batch
    C = 6       # feature dimension
    R = 8       # number of RoIs per batch
    dtype = torch.float
    device = 'cuda'

    # Simulated point cloud: [B, N, 3], coordinates in [-10, 10]
    points = (torch.rand(B, N, 3, dtype=dtype, device=device) * 20) - 10

    # Simulated point-wise features: [B, N, C]
    feats = torch.rand(B, N, C, dtype=dtype, device=device)

    # RoIs: [B, R, 7] → [x, y, z, dx, dy, dz, yaw]
    centers = (torch.rand(B, R, 3, dtype=dtype, device=device) * 20) - 10      # center in [-10, 10]
    sizes = torch.rand(B, R, 3, dtype=dtype, device=device) * 5 + 1            # size in [1, 6]
    yaws = torch.rand(B, R, 1, dtype=dtype, device=device) * 2 * math.pi       # yaw in [0, 2π]
    rois = torch.cat([centers, sizes, yaws], dim=-1)  # shape: [B, R, 7]

    save_dir = os.path.dirname(os.path.abspath(__file__))
    
    # save_tensor = lambda tensor, name: torch.save(
    #     {"tensor": tensor.detach(), "requires_grad": tensor.requires_grad},
    #     os.path.join(save_dir, f"{name}.pt")
    # )

    # save_tensor(points, "points")
    # save_tensor(feats, "feats")
    # save_tensor(rois, "rois")


    load_tensor = lambda name: (
        lambda data: data["tensor"].to(device).requires_grad_(data["requires_grad"])
    )(torch.load(os.path.join(save_dir, f"{name}.pt"), map_location=device, weights_only=True))

    points = load_tensor("points")
    feats = load_tensor("feats")
    rois = load_tensor("rois")


    roipoint_pool3d = RoIPointPool3d(num_sampled_points=4)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize() 
    start.record()
    roi_feat, empty_flag = roipoint_pool3d(points, feats, rois)
    end.record()
    torch.cuda.synchronize() 
    elapsed = start.elapsed_time(end)
    print("Perf: "+ str(elapsed) + " ms")

    
    expected_roi_feat = torch.tensor(
        [[[[1, 2, 3.3, 1, 2, 3.3], [1.2, 2.5, 3, 1.2, 2.5, 3],
           [0.8, 2.1, 3.5, 0.8, 2.1, 3.5], [1.6, 2.6, 3.6, 1.6, 2.6, 3.6]],
          [[-9.2, 21, 18.2, -9.2, 21, 18.2], [-9.2, 21, 18.2, -9.2, 21, 18.2],
           [-9.2, 21, 18.2, -9.2, 21, 18.2], [-9.2, 21, 18.2, -9.2, 21, 18.2]]]
         ],
        dtype=dtype).to(device)
    expected_empty_flag = torch.tensor([[0, 0]]).int().to(device)

    # torch.save(roi_feat.detach().cpu(), os.path.join(save_dir, 'expected_roi_feat.pt')) 
    expected_roi_feat = torch.load(os.path.join(save_dir, 'expected_roi_feat.pt'), map_location='cpu', weights_only=True)

    # torch.save(empty_flag.detach().cpu(), os.path.join(save_dir, 'expected_empty_flag.pt')) 
    expected_empty_flag = torch.load(os.path.join(save_dir, 'expected_empty_flag.pt'), map_location='cpu', weights_only=True)


    try:
        assert torch.allclose(roi_feat.detach().cpu(), expected_roi_feat)
        assert torch.allclose(empty_flag.detach().cpu(), expected_empty_flag)
    except:
        print("Validation failed")

if __name__ == "__main__":

    test_roipoint('cuda', torch.float)
