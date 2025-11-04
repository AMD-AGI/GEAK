# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from roiaware_pool3d_wrapper import RoIAwarePool3d
import time
import os

def generate_fake_roiaware_inputs(num_rois=4, num_pts=5000, device='cuda', dtype=torch.float):
    # Generate rois [num_rois, 7]
    rois = torch.zeros((num_rois, 7), dtype=dtype, device=device)
    rois[:, :3] = torch.rand(num_rois, 3, device=device) * 20  # centers: (x, y, z)
    rois[:, 3:6] = torch.rand(num_rois, 3, device=device) * torch.tensor([10.0, 5.0, 5.0], device=device) + 1.0  # sizes
    rois[:, 6] = (torch.rand(num_rois, device=device) - 0.5) * 2 * np.pi  # yaw

    # Generate pts [num_pts, 3]
    pts = torch.rand(num_pts, 3, dtype=dtype, device=device) * 30  # larger spread
    pts_feature = torch.sin(pts)  # example feature; or just use pts.clone()

    return rois, pts, pts_feature


def test_RoIAwarePool3d(device, dtype):
    roiaware_pool3d_max = RoIAwarePool3d(
        out_size=4, max_pts_per_voxel=128, mode='max')
    roiaware_pool3d_avg = RoIAwarePool3d(
        out_size=4, max_pts_per_voxel=128, mode='avg')
    rois = torch.tensor(
        [[1.0, 2.0, 3.0, 5.0, 4.0, 6.0, -0.3 - np.pi / 2],
         [-10.0, 23.0, 16.0, 20.0, 10.0, 20.0, -0.5 - np.pi / 2]],
        dtype=dtype).to(device)
    # boxes (m, 7) with bottom center in lidar coordinate
    pts = torch.tensor(
        [[1, 2, 3.3], [1.2, 2.5, 3.0], [0.8, 2.1, 3.5], [1.6, 2.6, 3.6],
         [0.8, 1.2, 3.9], [-9.2, 21.0, 18.2], [3.8, 7.9, 6.3],
         [4.7, 3.5, -12.2], [3.8, 7.6, -2], [-10.6, -12.9, -20], [-16, -18, 9],
         [-21.3, -52, -5], [0, 0, 0], [6, 7, 8], [-2, -3, -4]],
        dtype=dtype).to(device)  # points (n, 3) in lidar coordinate
    pts_feature = pts.clone()
    
    rois, pts, pts_feature = generate_fake_roiaware_inputs(num_rois=100, num_pts=20000, device=device, dtype=dtype)
    
    save_dir = os.path.dirname(os.path.abspath(__file__))
    
    # save_tensor = lambda tensor, name: torch.save(
    #     {"tensor": tensor.detach(), "requires_grad": tensor.requires_grad},
    #     os.path.join(save_dir, f"{name}.pt")
    # )

    # save_tensor(rois, "rois")
    # save_tensor(pts, "pts")
    # save_tensor(pts_feature, "pts_feature")


    load_tensor = lambda name: (
        lambda data: data["tensor"].to(device).requires_grad_(data["requires_grad"])
    )(torch.load(os.path.join(save_dir, f"{name}.pt"), map_location=device))

    rois = load_tensor("rois")
    pts = load_tensor("pts")
    pts_feature = load_tensor("pts_feature")



    

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize() 
    start.record()
    pooled_features_max = roiaware_pool3d_max(
        rois=rois, pts=pts, pts_feature=pts_feature)
    end.record()
    torch.cuda.synchronize() 
    elapsed = start.elapsed_time(end)
    print("Perf: "+ str(elapsed) + " ms")

    



    # torch.save(pooled_features_max.detach().cpu(), os.path.join(save_dir, 'pooled_features_max.pt')) 
    pooled_features_max_gt = torch.load(os.path.join(save_dir, 'pooled_features_max.pt'), map_location='cpu', weights_only=True)

    try:
        # import pdb; pdb.set_trace()
        assert pooled_features_max.shape == pooled_features_max_gt.shape
        assert torch.allclose(pooled_features_max.sum(),
                            pooled_features_max_gt.sum().to(device), 1e-3)
    except:
        print("Validation failed")

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize() 
    start.record()
    pooled_features_avg = roiaware_pool3d_avg(
        rois=rois, pts=pts, pts_feature=pts_feature)
    end.record()
    torch.cuda.synchronize() 
    elapsed = start.elapsed_time(end)
    print("Perf: "+ str(elapsed) + " ms")

    # torch.save(pooled_features_avg.detach().cpu(), os.path.join(save_dir, 'pooled_features_avg.pt')) 
    pooled_features_avg_gt = torch.load(os.path.join(save_dir, 'pooled_features_avg.pt'), map_location='cpu', weights_only=True)


    try:
        assert pooled_features_avg.shape == pooled_features_avg_gt.shape
        assert torch.allclose(pooled_features_avg.sum(),
                          pooled_features_avg_gt.sum().to(device), 1e-3)
    except:
        print("Validation failed")

if __name__ == "__main__":

    test_RoIAwarePool3d('cuda', torch.float)
