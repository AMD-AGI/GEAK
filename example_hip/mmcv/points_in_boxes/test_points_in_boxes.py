# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from points_in_boxes_wrapper import points_in_boxes_all, points_in_boxes_part
import time

def test_points_in_boxes_part(device):
    boxes = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.3]],
         [[-10.0, 23.0, 16.0, 10, 20, 20, 0.5]]],
        dtype=torch.float32).to(
            device)  # boxes (b, t, 7) with bottom center in lidar coordinate
    pts = torch.tensor(
        [[[1, 2, 3.3], [1.2, 2.5, 3.0], [0.8, 2.1, 3.5], [1.6, 2.6, 3.6],
          [0.8, 1.2, 3.9], [-9.2, 21.0, 18.2], [3.8, 7.9, 6.3],
          [4.7, 3.5, -12.2]],
         [[3.8, 7.6, -2], [-10.6, -12.9, -20], [-16, -18, 9], [-21.3, -52, -5],
          [0, 0, 0], [6, 7, 8], [-2, -3, -4], [6, 4, 9]]],
        dtype=torch.float32).to(device)  # points (b, m, 3) in lidar coordinate


    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize() 
    start.record()
    
    point_indices = points_in_boxes_part(points=pts, boxes=boxes)
    
    end.record()
    torch.cuda.synchronize() 
    elapsed = start.elapsed_time(end)
    print("Perf: "+ str(elapsed) + " ms")

    expected_point_indices = torch.tensor(
        [[0, 0, 0, 0, 0, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1, -1]],
        dtype=torch.int32).to(device)
    
    try:
        assert point_indices.shape == torch.Size([2, 8])
        assert (point_indices == expected_point_indices).all()
    except:
        print("Validation failed")

    boxes = torch.tensor([[[0.0, 0.0, 0.0, 1.0, 20.0, 1.0, 0.523598]]],
                         dtype=torch.float32).to(device)  # 30 degrees
    pts = torch.tensor(
        [[[4, 6.928, 0], [6.928, 4, 0], [4, -6.928, 0], [6.928, -4, 0],
          [-4, 6.928, 0], [-6.928, 4, 0], [-4, -6.928, 0], [-6.928, -4, 0]]],
        dtype=torch.float32).to(device)
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize() 
    start.record()
    
    point_indices = points_in_boxes_part(points=pts, boxes=boxes)
    
    end.record()
    torch.cuda.synchronize() 
    elapsed = start.elapsed_time(end)
    print("Perf: "+ str(elapsed) + " ms")


    expected_point_indices = torch.tensor([[-1, -1, 0, -1, 0, -1, -1, -1]],
                                          dtype=torch.int32).to(device)
    
    try:
        assert (point_indices == expected_point_indices).all()
    except:
        print("Validation failed")



def test_points_in_boxes_all():

    boxes = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.3],
          [-10.0, 23.0, 16.0, 10, 20, 20, 0.5]]],
        dtype=torch.float32).cuda(
        )  # boxes (m, 7) with bottom center in lidar coordinate
    pts = torch.tensor(
        [[[1, 2, 3.3], [1.2, 2.5, 3.0], [0.8, 2.1, 3.5], [1.6, 2.6, 3.6],
          [0.8, 1.2, 3.9], [-9.2, 21.0, 18.2], [3.8, 7.9, 6.3],
          [4.7, 3.5, -12.2], [3.8, 7.6, -2], [-10.6, -12.9, -20], [
              -16, -18, 9
          ], [-21.3, -52, -5], [0, 0, 0], [6, 7, 8], [-2, -3, -4]]],
        dtype=torch.float32).cuda()  # points (n, 3) in lidar coordinate

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize() 
    start.record()

    point_indices = points_in_boxes_all(points=pts, boxes=boxes)
    
    end.record()
    torch.cuda.synchronize() 
    elapsed = start.elapsed_time(end)
    print("Perf: "+ str(elapsed) + " ms")

    expected_point_indices = torch.tensor(
        [[[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 0], [0, 0],
          [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]],
        dtype=torch.int32).cuda()
    try:
        assert point_indices.shape == torch.Size([1, 15, 2])
        assert (point_indices == expected_point_indices).all()
    except:
        print("Validation failed")

    if torch.cuda.device_count() >= 1:
        pts = pts.to('cuda')
        boxes = boxes.to('cuda')
        expected_point_indices = expected_point_indices.to('cuda')
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize() 
        start.record()

        point_indices = points_in_boxes_all(points=pts, boxes=boxes)
        
        end.record()
        torch.cuda.synchronize() 
        elapsed = start.elapsed_time(end)
        print("Perf: "+ str(elapsed) + " ms")
        
        try:
            assert point_indices.shape == torch.Size([1, 15, 2])
            assert (point_indices == expected_point_indices).all()
        except:
            print("Validation failed")


if __name__ == "__main__":

    test_points_in_boxes_part('cuda')
    test_points_in_boxes_all()
