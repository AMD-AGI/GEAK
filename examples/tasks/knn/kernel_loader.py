# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
from torch.utils.cpp_extension import load

knn_ext = load(name="knn",
               sources=["src/knn_cuda.hip", "src/knn.cpp"],
               verbose=True)


