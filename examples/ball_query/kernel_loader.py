# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
from torch.utils.cpp_extension import load

ball_query_ext = load(name="ball_query",
                      sources=["src/ball_query_cuda.hip", "src/ball_query.cpp"],
                      verbose=True)


