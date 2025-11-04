# Copyright (c) OpenMMLab. All rights reserved.
import torch

from furthest_point_sample_wrapper import furthest_point_sample, furthest_point_sample_with_dist
import time

def test_fps(device):
    xyz = torch.tensor([[[-0.2748, 1.0020, -1.1674], [0.1015, 1.3952, -1.2681],
                         [-0.8070, 2.4137,
                          -0.5845], [-1.0001, 2.1982, -0.5859],
                         [0.3841, 1.8983, -0.7431]],
                        [[-1.0696, 3.0758,
                          -0.1899], [-0.2559, 3.5521, -0.1402],
                         [0.8164, 4.0081, -0.1839], [-1.1000, 3.0213, -0.8205],
                         [-0.0518, 3.7251, -0.3950]]]).to(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize() 
    start.record()
    
    idx = furthest_point_sample(xyz, 3)

    end.record()
    torch.cuda.synchronize() 
    elapsed = start.elapsed_time(end)
    print("Perf: "+ str(elapsed) + " ms")

    expected_idx = torch.tensor([[0, 2, 4], [0, 2, 1]]).to(device)

    try:
        assert torch.all(idx == expected_idx)
    except:
        print("Validation failed")


def test_fps_with_dist(device):
    xyz = torch.tensor([[[-0.2748, 1.0020, -1.1674], [0.1015, 1.3952, -1.2681],
                         [-0.8070, 2.4137,
                          -0.5845], [-1.0001, 2.1982, -0.5859],
                         [0.3841, 1.8983, -0.7431]],
                        [[-1.0696, 3.0758,
                          -0.1899], [-0.2559, 3.5521, -0.1402],
                         [0.8164, 4.0081, -0.1839], [-1.1000, 3.0213, -0.8205],
                         [-0.0518, 3.7251, -0.3950]]]).to(device)

    expected_idx = torch.tensor([[0, 2, 4], [0, 2, 1]]).to(device)
    xyz_square_dist = ((xyz.unsqueeze(dim=1) -
                        xyz.unsqueeze(dim=2))**2).sum(-1)
    idx = furthest_point_sample_with_dist(xyz_square_dist, 3)
    assert torch.all(idx == expected_idx)

    import numpy as np
    fps_idx = np.load('furthest_point_sample/for_3d_ops/fps_idx.npy')
    features_for_fps_distance = np.load(
        'furthest_point_sample/for_3d_ops/features_for_fps_distance.npy')
    expected_idx = torch.from_numpy(fps_idx).to(device)
    features_for_fps_distance = torch.from_numpy(features_for_fps_distance).to(
        device)
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize() 
    start.record()

    idx = furthest_point_sample_with_dist(features_for_fps_distance, 16)
    
    end.record()
    torch.cuda.synchronize() 
    elapsed = start.elapsed_time(end)
    print("Perf: "+ str(elapsed) + " ms")
    
    try:
        assert torch.all(idx == expected_idx)
    except:
        print("Validation failed")


if __name__ == "__main__":

    test_fps("cuda")
    test_fps_with_dist("cuda")
