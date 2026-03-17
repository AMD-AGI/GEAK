# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.

import sys
import os
import importlib.util

# Add current directory to path for generated kernel import
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Import generated kernel (will be copied to same dir as this script)
from rmsnorm import rmsnorm_triton_wrapper

# Import reference kernel from kernels directory using importlib to avoid cache
KERNELS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "kernels"))
ref_spec = importlib.util.spec_from_file_location(
    "rmsnorm_ref", os.path.join(KERNELS_DIR, "rmsnorm.py")
)
ref_module = importlib.util.module_from_spec(ref_spec)
ref_spec.loader.exec_module(ref_module)
rmsnorm_triton_wrapper_ref = ref_module.rmsnorm_triton_wrapper

from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl


class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__("rmsnorm", dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(4, 14):  # Adjust the range as needed for your testing
            batch_size = 2**i
            M = 128  # Example fixed size, adjust as needed
            K = 256  # Example fixed size, adjust as needed
            x = torch.rand((batch_size, M, K), dtype=torch.float32)
            rms_w = torch.rand((K,), dtype=torch.float32)
            self.input_tensors.append((x, rms_w))

    def to_mlu(self, input_tensor):
        x, rms_w = input_tensor
        return (x.mlu(), rms_w.mlu())

    def call_op(self, input_tensor):
        x, rms_w = input_tensor
        return rmsnorm_triton_wrapper(x, rms_w)

    def call_op_ref(self, input_tensor):
        x, rms_w = input_tensor
        return rmsnorm_triton_wrapper_ref(x, rms_w)

    def get_gbps(self, input_tensor, runtime):
        x, rms_w = input_tensor
        total_bytes = (
            x.numel() + rms_w.numel() + x.numel()
        ) * x.element_size()  # input, weight, and output
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS

    def get_tflops(self, input_tensor, runtime):
        x, rms_w = input_tensor
        # FLOPS: 2 operations per element (square and multiply), plus additional operations for normalization
        FLOPS = 2 * x.numel() + x.size(0) * x.size(1)  # Simplified estimation
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == "__main__":
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
