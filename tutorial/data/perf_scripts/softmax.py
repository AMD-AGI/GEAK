import sys
import os
import torch
import triton
import triton.language as tl

# 确保能导入你的性能工具基类
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# from performance_utils import Performance_Metrics, do_bench_config

# 假设你的 softmax 实现已经按照之前的代码定义好
# from softmax_example import softmax as softmax_mlu


class softmax_performance_metrics(Performance_Metrics):
    def __init__(self, dtype=torch.float32, is_backward=False, **kwargs):
        super().__init__(
            "softmax_example", dtype=dtype, is_backward=is_backward, **kwargs
        )

    def get_input_tensors(self):
        """
        构造测试输入。
        Softmax 通常受列数（N）影响较大，因为 N 决定了 BLOCK_SIZE 和 SRAM 占用。
        """
        self.input_tensors = []
        # 固定行数 M=4096，改变列数 N
        M = 4096
        for i in range(10, 15):  # N 从 1024 到 16384
            N = 2**i
            input_tensor = torch.randn((M, N), dtype=self.dtype)
            self.input_tensors.append(input_tensor)

    def to_mlu(self, input_tensor):
        """搬运数据至 MLU"""
        return input_tensor.mlu()

    def call_op(self, input_tensor):
        """调用你实现的 Triton Softmax"""
        return softmax(input_tensor)

    def call_op_ref(self, input_tensor):
        """调用 PyTorch 原生 Softmax 作为参考"""
        return torch.softmax(input_tensor, dim=-1)

    def get_gbps(self, input_tensor, runtime):
        """
        计算有效带宽 (GB/s)。
        对于 Fused Softmax，理论访存量为：读取 MN + 写入 MN。
        """
        x = input_tensor
        # 总访存字节数：读一次 + 写一次
        total_bytes = 2 * x.numel() * x.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS

    def get_tflops(self, input_tensor, runtime):
        """
        计算 TFLOPS（对于 Softmax 来说，通常不是主要瓶颈，但可作为参考）。
        Softmax 操作包含：减法(1)、指数(1)、求和(1)、除法(1)，每个元素约 4~10 FLOPS。
        """
        x = input_tensor
        # 粗略估算每个元素的浮点操作数
        FLOPS_PER_ELEMENT = 8
        total_flops = x.numel() * FLOPS_PER_ELEMENT
        TFLOPS = total_flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == "__main__":
    # 创建性能评估实例
    op_perf = softmax_performance_metrics(dtype=torch.float32)
    op_perf.get_input_tensors()
    # 设置预热和重复次数
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    # 运行并打印结果
    op_perf.run_benchmark()
