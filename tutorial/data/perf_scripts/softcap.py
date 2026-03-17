import torch
import triton

# 假设 Softcap 类 (torch.autograd.Function) 已在上下文中定义


class softcap_performance_metrics(Performance_Metrics):
    def __init__(self, dtype=torch.float16, is_backward=False, **kwargs):
        super().__init__("softcap", dtype=dtype, is_backward=is_backward, **kwargs)
        self.softcap_val = 50.0

    def get_input_tensors(self):
        """
        构建测试输入：线性扫描元素数量 (numel)
        从 1MB 到 128MB 的数据量，观察带宽利用率的变化。
        """
        self.input_tensors = []
        # 扫描不同大小的张量，从 2^18 到 2^26 个元素
        for i in range(18, 27):
            numel = 2**i
            x = torch.randn(numel, dtype=self.dtype)

            # 算子输入：(x, softcap_val)
            args = (x, self.softcap_val)

            if self.is_backward:
                # 反向传播需要 dy
                dy = torch.randn_like(x)
                self.input_tensors.append((*args, dy))
            else:
                self.input_tensors.append(args)

    def to_mlu(self, input_tuple):
        """搬运至加速器并处理梯度状态"""
        device = "mlu" if torch.mlu.is_available() else "cuda"
        tensors = [
            t.to(device) if isinstance(t, torch.Tensor) else t for t in input_tuple
        ]

        if self.is_backward:
            # x 需要梯度
            tensors[0].requires_grad_()
        return tuple(tensors)

    def call_op(self, input_tuple):
        """调用 Triton 版 Softcap"""
        if self.is_backward:
            *args, dy = input_tuple
            y = Softcap.apply(*args)
            return y.backward(dy, retain_graph=True)
        else:
            return Softcap.apply(*input_tuple)

    def call_op_ref(self, input_tuple):
        """参考实现：PyTorch 原生 Tanh 组合"""
        x, softcap_val = input_tuple[0], input_tuple[1]
        if self.is_backward:
            dy = input_tuple[2]
            # 模拟原生实现：y = softcap * tanh(x / softcap)
            res = softcap_val * torch.tanh(x.to(torch.float32) / softcap_val).to(
                x.dtype
            )
            return res.backward(dy, retain_graph=True)
        else:
            return softcap_val * torch.tanh(x.to(torch.float32) / softcap_val).to(
                x.dtype
            )

    def get_gbps(self, input_tuple, runtime):
        """
        计算 GB/s。
        对于 Element-wise 算子，这是最重要的衡量指标。
        """
        x = input_tuple[0]
        numel = x.numel()
        element_size = x.element_size()

        if not self.is_backward:
            # 前向：读 x + 写 y
            total_bytes = 2 * numel * element_size
        else:
            # 反向：读 dy, x + 写 dx
            total_bytes = 3 * numel * element_size

        return total_bytes / (runtime / 1000) / 1e9

    def get_tflops(self, input_tuple, runtime):
        """
        计算 TFLOPS。
        虽然是访存密集型，但 tanh 是非线性函数，包含 exp 等昂贵操作。
        """
        x = input_tuple[0]
        numel = x.numel()
        # 粗略估计：x/softcap (1) + tanh (约20-50个指令) + *softcap (1)
        # 这里统一按 20 FLOPS/element 计算
        flops_per_el = 20
        flops = numel * flops_per_el

        if self.is_backward:
            flops *= 2

        return flops / (runtime / 1000) / 1e12


if __name__ == "__main__":
    # 测试前向带宽
    fwd_perf = softcap_performance_metrics(is_backward=False)
    fwd_perf.get_input_tensors()
    fwd_perf.get_do_bench_config(warmup=20, rep=100)
    fwd_perf.run_benchmark()
