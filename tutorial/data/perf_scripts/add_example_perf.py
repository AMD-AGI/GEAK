# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.

import sys
import os

# Add current directory to path for generated kernel import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import generated kernel (will be copied to same dir as this script)
from add_example import add_wrapper

# Import reference kernel from kernels directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KERNELS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'kernels'))
sys.path.insert(0, KERNELS_DIR)
from add_example import add_wrapper as add_wrapper_ref

from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('add_example', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 26):
            size = 2 ** i
            input_tensor = (torch.rand(size, dtype=torch.float32), torch.rand(size, dtype=torch.float32))
            self.input_tensors.append(input_tensor)

    def to_mlu(self, input_tensor):
        return (input_tensor[0].mlu(), input_tensor[1].mlu())

    def call_op(self, input_tensor):
        return add_wrapper(input_tensor[0], input_tensor[1])
    
    def call_op_ref(self, input_tensor):
        return add_wrapper_ref(input_tensor[0], input_tensor[1])

    def get_gbps(self, input_tensor, runtime):
        x = input_tensor[0]
        total_bytes = 3 * x.numel() * x.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        x = input_tensor[0]
        FLOPS = 2 * float(x.size()[0])
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
