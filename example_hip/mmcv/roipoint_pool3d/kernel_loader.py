# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.

from torch.utils.cpp_extension import load

roipoint_pool3d_ext = load(name="roipoint_pool3d",
                           extra_include_paths=["roipoint_pool3d/src/include"],
                           sources=["roipoint_pool3d/src/roipoint_pool3d_kernel.hip", "roipoint_pool3d/src/roipoint_pool3d.cpp"],
                           verbose=True)


