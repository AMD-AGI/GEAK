# Copyright (c) OpenMMLab. All rights reserved.
import torch

from three_interpolate_wrapper import three_interpolate
import time
import os


def generate_large_fake_inputs(B=8, C=64, N=8192, M=2048, dtype=torch.float32, device='cuda'):
    # Simulate random features for each input point
    features = torch.rand(B, C, N, dtype=dtype, device=device)

    # Simulate indices for 3 nearest neighbors from N input points for each of M query points
    idx = torch.randint(0, N, (B, M, 3), dtype=torch.int32, device=device)

    # Create weights that sum to ~1 for interpolation
    raw_weights = torch.rand(B, M, 3, dtype=dtype, device=device)
    weight = raw_weights / raw_weights.sum(dim=-1, keepdim=True)

    return features, idx, weight


def test_three_interpolate(dtype, device):
    features = torch.tensor(
        [[[2.4350, 4.7516, 4.4995, 2.4350, 2.4350, 2.4350],
          [3.1236, 2.6278, 3.0447, 3.1236, 3.1236, 3.1236],
          [2.6732, 2.8677, 2.6436, 2.6732, 2.6732, 2.6732],
          [0.0124, 7.0150, 7.0199, 0.0124, 0.0124, 0.0124],
          [0.3207, 0.0000, 0.3411, 0.3207, 0.3207, 0.3207]],
         [[0.0000, 0.9544, 2.4532, 0.0000, 0.0000, 0.0000],
          [0.5346, 1.9176, 1.4715, 0.5346, 0.5346, 0.5346],
          [0.0000, 0.2744, 2.0842, 0.0000, 0.0000, 0.0000],
          [0.3414, 1.5063, 1.6209, 0.3414, 0.3414, 0.3414],
          [0.5814, 0.0103, 0.0000, 0.5814, 0.5814, 0.5814]]],
        dtype=dtype,
        device=device)

    idx = torch.tensor(
        [[[0, 1, 2], [2, 3, 4], [2, 3, 4], [0, 1, 2], [0, 1, 2], [0, 1, 3]],
         [[0, 2, 3], [1, 3, 4], [2, 1, 4], [0, 2, 4], [0, 2, 4], [0, 1, 2]]],
        device=device).int()

    weight = torch.tensor([[[3.3333e-01, 3.3333e-01, 3.3333e-01],
                            [1.0000e+00, 5.8155e-08, 2.2373e-08],
                            [1.0000e+00, 1.7737e-08, 1.7356e-08],
                            [3.3333e-01, 3.3333e-01, 3.3333e-01],
                            [3.3333e-01, 3.3333e-01, 3.3333e-01],
                            [3.3333e-01, 3.3333e-01, 3.3333e-01]],
                           [[3.3333e-01, 3.3333e-01, 3.3333e-01],
                            [1.0000e+00, 1.3651e-08, 7.7312e-09],
                            [1.0000e+00, 1.7148e-08, 1.4070e-08],
                            [3.3333e-01, 3.3333e-01, 3.3333e-01],
                            [3.3333e-01, 3.3333e-01, 3.3333e-01],
                            [3.3333e-01, 3.3333e-01, 3.3333e-01]]],
                          dtype=dtype,
                          device=device)
    

    save_dir = os.path.dirname(os.path.abspath(__file__))
    

    features, idx, weight = generate_large_fake_inputs(dtype=dtype, device=device)



    # save_tensor = lambda tensor, name: torch.save(
    #     {"tensor": tensor.detach(), "requires_grad": tensor.requires_grad},
    #     os.path.join(save_dir, f"{name}.pt")
    # )

    # save_tensor(features, "features")
    # save_tensor(idx, "idx")
    # save_tensor(weight, "weight")


    load_tensor = lambda name: (
        lambda data: data["tensor"].to(device).requires_grad_(data["requires_grad"])
    )(torch.load(os.path.join(save_dir, f"{name}.pt"), map_location=device, weights_only=True))

    features = load_tensor("features")
    idx = load_tensor("idx")
    weight = load_tensor("weight")


    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize() 
    start.record()
    output = three_interpolate(features, idx, weight)

    end.record()
    torch.cuda.synchronize() 
    elapsed = start.elapsed_time(end)
    print("Perf: "+ str(elapsed) + " ms")


    expected_output = torch.tensor([[[
        3.8953e+00, 4.4995e+00, 4.4995e+00, 3.8953e+00, 3.8953e+00, 3.2072e+00
    ], [
        2.9320e+00, 3.0447e+00, 3.0447e+00, 2.9320e+00, 2.9320e+00, 2.9583e+00
    ], [
        2.7281e+00, 2.6436e+00, 2.6436e+00, 2.7281e+00, 2.7281e+00, 2.7380e+00
    ], [
        4.6824e+00, 7.0199e+00, 7.0199e+00, 4.6824e+00, 4.6824e+00, 2.3466e+00
    ], [
        2.2060e-01, 3.4110e-01, 3.4110e-01, 2.2060e-01, 2.2060e-01, 2.1380e-01
    ]],
                                    [[
                                        8.1773e-01, 9.5440e-01, 2.4532e+00,
                                        8.1773e-01, 8.1773e-01, 1.1359e+00
                                    ],
                                     [
                                         8.4689e-01, 1.9176e+00, 1.4715e+00,
                                         8.4689e-01, 8.4689e-01, 1.3079e+00
                                     ],
                                     [
                                         6.9473e-01, 2.7440e-01, 2.0842e+00,
                                         6.9473e-01, 6.9473e-01, 7.8619e-01
                                     ],
                                     [
                                         7.6789e-01, 1.5063e+00, 1.6209e+00,
                                         7.6789e-01, 7.6789e-01, 1.1562e+00
                                     ],
                                     [
                                         3.8760e-01, 1.0300e-02, 8.3569e-09,
                                         3.8760e-01, 3.8760e-01, 1.9723e-01
                                     ]]],
                                   dtype=dtype,
                                   device=device)


    # torch.save(output.detach().cpu(), os.path.join(save_dir, 'expected_output.pt')) 
    expected_output = torch.load(os.path.join(save_dir, 'expected_output.pt'), map_location='cpu', weights_only=True)


    try:
        assert torch.allclose(output.detach().cpu(), expected_output, 1e-3, 1e-4)
    except:
        print("Validation failed")

if __name__ == "__main__":

    test_three_interpolate(torch.float32, "cuda")
