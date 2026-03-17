import sys
import os
import torch
import triton

# 假设你的 FusedSwiglu 类在当前脚本中
# from your_module import FusedSwiglu, naive_torch_swiglu


class swiglu_performance_metrics(Performance_Metrics):
    def __init__(self, dtype=torch.bfloat16, is_backward=False, **kwargs):
        super().__init__("fused_swiglu", dtype=dtype, is_backward=is_backward, **kwargs)
        # 固定隐藏层维度
        self.IN_DIM = 4096
        self.OUT_DIM = 11008  # 典型 Llama-7B 比例

    def get_input_tensors(self):
        """
        构造测试输入：改变序列长度 M
        """
        self.input_tensors = []
        # 模拟不同 Token 数量
        for i in range(10, 16):
            M = 2**i
            x = torch.randn((M, self.IN_DIM), dtype=self.dtype)
            w_g = torch.randn((self.IN_DIM, self.OUT_DIM), dtype=self.dtype)
            w_fc = torch.randn((self.IN_DIM, self.OUT_DIM), dtype=self.dtype)
            b_g = torch.randn((self.OUT_DIM,), dtype=self.dtype)
            b_fc = torch.randn((self.OUT_DIM,), dtype=self.dtype)

            # 如果是反向传播，还需要准备 dy
            if self.is_backward:
                dy = torch.randn((M, self.OUT_DIM), dtype=self.dtype)
                self.input_tensors.append((x, w_g, w_fc, b_g, b_fc, dy))
            else:
                self.input_tensors.append((x, w_g, w_fc, b_g, b_fc))

    def to_mlu(self, input_tensor):
        """搬运至加速器，并设置 grad 标志"""
        tensors = [
            t.to("cuda") if isinstance(t, torch.Tensor) else t for t in input_tensor
        ]
        if self.is_backward:
            # 激活 grad
            for t in tensors[:5]:  # x, w_g, w_fc, b_g, b_fc
                t.requires_grad_()
        return tuple(tensors)

    def call_op(self, input_tensor):
        """执行 Triton 融合算子"""
        if self.is_backward:
            x, w_g, w_fc, b_g, b_fc, dy = input_tensor
            # 这里的 is_recompute 参数根据你的 FusedSwiglu 定义传入
            y = FusedSwiglu.apply(x, w_g, w_fc, b_g, b_fc, True, False)
            return torch.autograd.backward(y, dy, retain_graph=True)
        else:
            x, w_g, w_fc, b_g, b_fc = input_tensor
            return FusedSwiglu.apply(x, w_g, w_fc, b_g, b_fc, False, False)

    def call_op_ref(self, input_tensor):
        """执行 PyTorch 原生算子（未融合）"""
        if self.is_backward:
            x, w_g, w_fc, b_g, b_fc, dy = input_tensor
            gate = torch.nn.functional.silu(torch.nn.functional.linear(x, w_g.T, b_g))
            fc = torch.nn.functional.linear(x, w_fc.T, b_fc)
            y = gate * fc
            return torch.autograd.backward(y, dy, retain_graph=True)
        else:
            x, w_g, w_fc, b_g, b_fc = input_tensor
            gate = torch.nn.functional.silu(torch.nn.functional.linear(x, w_g.T, b_g))
            fc = torch.nn.functional.linear(x, w_fc.T, b_fc)
            return gate * fc

    def get_gbps(self, input_tensor, runtime):
        """
        计算有效带宽 (GB/s)
        前向访存：读 x, w_g, w_fc, b_g, b_fc + 写 y
        """
        x, w_g, w_fc, b_g, b_fc = input_tensor[:5]
        element_size = x.element_size()

        if not self.is_backward:
            # 前向理论访存量
            total_bytes = (
                x.numel()
                + w_g.numel()
                + w_fc.numel()
                + b_g.numel()
                + b_fc.numel()
                + (x.shape[0] * w_g.shape[1])
            ) * element_size
        else:
            # 反向访存量极其复杂，此处估算主要 Tensor 读写
            total_bytes = (
                x.numel() * 2 + w_g.numel() * 2 + w_fc.numel() * 2
            ) * element_size

        return total_bytes / (runtime / 1000) / 1e9

    def get_tflops(self, input_tensor, runtime):
        """
        计算算力利用率 (TFLOPS)
        SwiGLU 主要计算量在两个大矩阵乘法上：2 * M * N * K
        """
        x, w_g, w_fc = input_tensor[0], input_tensor[1], input_tensor[2]
        M, K = x.shape
        N = w_g.shape[1]

        # 每个全连接层是 2MNK，这里有两个
        total_flops = 2 * (2 * M * N * K)
        if self.is_backward:
            # 反向传播计算量约为前向的 2 倍
            total_flops *= 3

        return total_flops / (runtime / 1000) / 1e12


if __name__ == "__main__":
    # 测试前向性能
    print("Testing Forward Performance...")
    fwd_perf = swiglu_performance_metrics(is_backward=False)
    fwd_perf.get_input_tensors()
    fwd_perf.get_do_bench_config(warmup=50, rep=100)
    fwd_perf.run_benchmark()

    # 测试反向性能
    print("\nTesting Backward Performance...")
    bwd_perf = swiglu_performance_metrics(is_backward=True)
    bwd_perf.get_input_tensors()
    bwd_perf.get_do_bench_config(warmup=50, rep=100)
    bwd_perf.run_benchmark()
