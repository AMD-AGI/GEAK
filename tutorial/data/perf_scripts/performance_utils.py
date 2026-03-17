# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl

from typing import Callable
import json
import os
import random


def get_random_choice(item_list):
    return random.choice(item_list)


class do_bench_config:
    def __init__(
        self,
        warm_up=25,
        repetition=100,
        grad_to_none=None,
        quantiles=[0.5, 0.8, 0.2],
        return_mode="median",
    ):
        self.warm_up = warm_up
        self.repetition = repetition
        self.grad_to_none = grad_to_none
        self.quantiles = quantiles
        self.return_mode = return_mode


class Performance_Metrics:
    def __init__(self, op_name, dtype=None, is_backward=False, **kwargs):
        self.op_name = op_name
        self.ref_op_name = op_name + "_ref"
        self.dtype = dtype
        if is_backward:
            self.op_name += "backward"
        self.kwargs = kwargs

        self.input_tensors = []
        self.do_bench_config = do_bench_config()

    def get_input_tensors(self):
        raise NotImplementedError("You must implement this method to get input tensors")

    def to_cuda(self, input_tensor):
        raise NotImplementedError("You must implement this method to get input tensors")

    def call_op(self, input_tensor):
        raise NotImplementedError("You must implement this method to call the op")

    def call_op_ref(self, input_tensor):
        raise NotImplementedError(
            "You must implement this method to call the reference op"
        )

    def get_do_bench_config(self, warmup=None, rep=None):
        if warmup != None and rep != None:
            self.do_bench_config = do_bench_config(
                warm_up=warmup,
                repetition=rep,
            )
            return

    def get_runtime(self, op: Callable):
        ms, min_ms, max_ms = triton.testing.do_bench(
            op,
            warmup=self.do_bench_config.warm_up,
            rep=self.do_bench_config.repetition,
            quantiles=self.do_bench_config.quantiles,
            return_mode=self.do_bench_config.return_mode,
        )
        return ms

    def get_gbps(self, input_tensor, runtime):
        raise NotImplementedError(
            "You must implement this method to get the method to calculate GBPS"
        )

    def get_tflops(self, input_tensor, runtime):
        raise NotImplementedError(
            "You must implement this method to get the method to calculate TFLOPS"
        )

    def check_close(self, a, b, rtol=1e-05, atol=1e-08):
        if isinstance(a, (list, tuple)):
            return all(
                self.check_close(x, y, rtol=rtol, atol=atol) for x, y in zip(a, b)
            )
        if isinstance(a, dict):
            return all(
                key in b and self.check_close(a[key], b[key], rtol=rtol, atol=atol)
                for key in a
            )
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            return torch.allclose(a, b, rtol=rtol, atol=atol)
        return a == b

    def get_num_elements(self, input_tensor):
        if isinstance(input_tensor, (list, tuple)):
            return sum(self.get_num_elements(x) for x in input_tensor)
        if isinstance(input_tensor, dict):
            return sum(self.get_num_elements(v) for v in input_tensor.values())
        if isinstance(input_tensor, torch.Tensor):
            return input_tensor.numel()
        return 1

    def run_benchmark(self):
        results = []
        perf = []
        perf_ref = []
        for input_tensor_ in self.input_tensors:
            try:
                input_tensor = self.to_mlu(input_tensor_)
                op = lambda: self.call_op(input_tensor)
                op_ref = lambda: self.call_op_ref(input_tensor)

                # Keep dummy initial calls to converge to optimal triton autotune configs
                output = self.call_op(input_tensor)
                output_ref = self.call_op_ref(input_tensor)

                # The following calls should be using the optimal triton autotune configs
                output = self.call_op(input_tensor)
                output_ref = self.call_op_ref(input_tensor)

                if not self.check_close(output, output_ref, rtol=1e-3, atol=1e-3):
                    print(
                        f"Output mismatch for input size {self.get_num_elements(input_tensor_)}"
                    )
                    continue

                # Randomly choose which operation to run first to avoid bias
                if get_random_choice([0, 1]) == 0:
                    ms = self.get_runtime(op)
                    ms_ref = self.get_runtime(op_ref)
                else:
                    ms_ref = self.get_runtime(op_ref)
                    ms = self.get_runtime(op)

                gbps = self.get_gbps(input_tensor, ms)
                tflops = self.get_tflops(input_tensor, ms)
                result = {
                    "input_size": self.get_num_elements(input_tensor_),
                    "ms": ms,
                    "ms_ref": ms_ref,
                    "GB/s": gbps,
                    "TFLOPS": tflops,
                }
                results.append(result)
                perf.append(ms)
                perf_ref.append(ms_ref)
            except Exception as e:
                print(f"Failed to run benchmark for input tensor. Error: {e}")
            input_tensor = None

        # Calculate average speedup
        speedup = 0.0
        if perf and perf_ref:
            speedup = sum(perf_ref) / sum(perf)

        results.append({"speedup": speedup})

        print(f"```json\n{json.dumps(results, indent=4)}\n```")

        return True, f"```json\n{json.dumps(results, indent=4)}\n```"
