# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.

from torch.utils.cpp_extension import load

furthest_point_sample_ext = load(name="furthest_point_sample",
               extra_include_paths=["furthest_point_sample/src/include"],
               sources=["furthest_point_sample/src/furthest_point_sample_cuda.hip", "furthest_point_sample/src/furthest_point_sample.cpp"],
               verbose=True)


