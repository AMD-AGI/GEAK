# Copyright (c) OpenMMLab. All rights reserved.
import torch

from three_nn_wrapper import three_nn
import time

import os


known = [[[-1.8373, 3.5605, -0.7867], [0.7615, 2.9420, 0.2314],
          [-0.6503, 3.6637, -1.0622], [-1.8373, 3.5605, -0.7867],
          [-1.8373, 3.5605, -0.7867]],
         [[-1.3399, 1.9991, -0.3698], [-0.0799, 0.9698, -0.8457],
          [0.0858, 2.4721, -0.1928], [-1.3399, 1.9991, -0.3698],
          [-1.3399, 1.9991, -0.3698]]]

unknown = [[[-1.8373, 3.5605, -0.7867], [0.7615, 2.9420, 0.2314],
            [-0.6503, 3.6637, -1.0622], [-1.5237, 2.3976, -0.8097],
            [-0.0722, 3.4017, -0.2880], [0.5198, 3.0661, -0.4605],
            [-2.0185, 3.5019, -0.3236], [0.5098, 3.1020, 0.5799],
            [-1.6137, 3.8443, -0.5269], [0.7341, 2.9626, -0.3189]],
           [[-1.3399, 1.9991, -0.3698], [-0.0799, 0.9698, -0.8457],
            [0.0858, 2.4721, -0.1928], [-0.9022, 1.6560, -1.3090],
            [0.1156, 1.6901, -0.4366], [-0.6477, 2.3576, -0.1563],
            [-0.8482, 1.1466, -1.2704], [-0.8753, 2.0845, -0.3460],
            [-0.5621, 1.4233, -1.2858], [-0.5883, 1.3114, -1.2899]]]

expected_dist = [[[0.0000, 0.0000, 0.0000], [0.0000, 2.0463, 2.8588],
                  [0.0000, 1.2229, 1.2229], [1.2047, 1.2047, 1.2047],
                  [1.0011, 1.0845, 1.8411], [0.7433, 1.4451, 2.4304],
                  [0.5007, 0.5007, 0.5007], [0.4587, 2.0875, 2.7544],
                  [0.4450, 0.4450, 0.4450], [0.5514, 1.7206, 2.6811]],
                 [[0.0000, 0.0000, 0.0000], [0.0000, 1.6464, 1.6952],
                  [0.0000, 1.5125, 1.5125], [1.0915, 1.0915, 1.0915],
                  [0.8197, 0.8511, 1.4894], [0.7433, 0.8082, 0.8082],
                  [0.8955, 1.3340, 1.3340], [0.4730, 0.4730, 0.4730],
                  [0.7949, 1.3325, 1.3325], [0.7566, 1.3727, 1.3727]]]

expected_idx = [[[0, 3, 4], [1, 2, 0], [2, 0, 3], [0, 3, 4], [2, 1, 0],
                 [1, 2, 0], [0, 3, 4], [1, 2, 0], [0, 3, 4], [1, 2, 0]],
                [[0, 3, 4], [1, 2, 0], [2, 0, 3], [0, 3, 4], [2, 1, 0],
                 [2, 0, 3], [1, 0, 3], [0, 3, 4], [1, 0, 3], [1, 0, 3]]]


def generate_fake_point_cloud_data(B=8, N_known=2048, N_unknown=1024, device='cuda', dtype=torch.float32):
    # Random known points in 3D
    known = torch.rand(B, N_known, 3, device=device, dtype=dtype) * 10

    # Random unknown points in similar space
    unknown = torch.rand(B, N_unknown, 3, device=device, dtype=dtype) * 10

    return unknown, known


def test_three_nn(device):
    dtype = torch.float
    known_t = torch.tensor(known, dtype=dtype, device=device)
    unknown_t = torch.tensor(unknown, dtype=dtype, device=device)

    dtype = torch.float
    unknown_t, known_t = generate_fake_point_cloud_data(device=device, dtype=dtype)


    save_dir = os.path.dirname(os.path.abspath(__file__))

    # save_tensor = lambda tensor, name: torch.save(
    #     {"tensor": tensor.detach(), "requires_grad": tensor.requires_grad},
    #     os.path.join(save_dir, f"{name}.pt")
    # )

    # save_tensor(unknown_t, "unknown_t")
    # save_tensor(known_t, "known_t")


    load_tensor = lambda name: (
        lambda data: data["tensor"].to(device).requires_grad_(data["requires_grad"])
    )(torch.load(os.path.join(save_dir, f"{name}.pt"), map_location=device, weights_only=True))

    unknown_t = load_tensor("unknown_t")
    known_t = load_tensor("known_t")


    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize() 
    start.record()

    dist_t, idx_t = three_nn(unknown_t, known_t)
    
    end.record()
    torch.cuda.synchronize() 
    elapsed = start.elapsed_time(end)
    print("Perf: "+ str(elapsed) + " ms")

    # torch.save(dist_t.detach().cpu(), os.path.join(save_dir, 'expected_dist_t.pt')) 
    expected_dist_t = torch.load(os.path.join(save_dir, 'expected_dist_t.pt'), map_location='cpu', weights_only=True)

    # torch.save(idx_t.detach().cpu(), os.path.join(save_dir, 'expected_idx_t.pt')) 
    expected_idx_t = torch.load(os.path.join(save_dir, 'expected_idx_t.pt'), map_location='cpu', weights_only=True)


    # expected_dist_t = torch.tensor(expected_dist, dtype=dtype, device=device)
    # expected_idx_t = torch.tensor(expected_idx, device=device)

    try:
        assert torch.allclose(dist_t.detach().cpu(), expected_dist_t, atol=1e-4, rtol=1e-5)
        assert torch.all(idx_t.detach().cpu() == expected_idx_t)
    except:
        print("Validation failed")

if __name__ == "__main__":

    test_three_nn("cuda", )
