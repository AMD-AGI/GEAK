import torch
import triton

# 假设 channel_granul_fp8_quant 函数已在上下文中定义


class fp8_quant_performance_metrics(Performance_Metrics):
    def __init__(
        self,
        dtype=torch.bfloat16,
        float8_dtype=torch.float8_e4m3fn,
        is_backward=False,
        **kwargs,
    ):
        super().__init__("fp8_quant", dtype=dtype, is_backward=is_backward, **kwargs)
        self.float8_dtype = float8_dtype

    def get_input_tensors(self):
        """
        构建测试输入：
        主要考察不同的 M (Sequence Length) 和 N (Hidden Dim) 对性能的影响。
        同时测试两种规约方向：Per-token (Reduce N) 和 Per-channel (Reduce M)。
        """
        self.input_tensors = []
        B = 1  # Batch size 通常在 Grid 中并行
        # 扫描典型的模型维度
        for M in [1024, 2048, 4096]:
            for N in [4096, 8192, 12288]:
                x = torch.randn((B, M, N), dtype=self.dtype)

                # 场景 1: Per-token 量化 (Reduce along N)
                self.input_tensors.append((x, self.float8_dtype, -1))

                # 场景 2: Per-channel 量化 (Reduce along M)
                self.input_tensors.append((x, self.float8_dtype, -2))

    def to_mlu(self, input_tuple):
        """搬运至加速器"""
        device = "mlu" if torch.mlu.is_available() else "cuda"
        # x, dtype, axiswise_dim
        x = input_tuple[0].to(device)
        return (x, input_tuple[1], input_tuple[2])

    def call_op(self, input_tuple):
        """调用 Triton 版量化内核"""
        return channel_granul_fp8_quant(*input_tuple)

    def call_op_ref(self, input_tuple):
        """参考实现：PyTorch 原生逻辑"""
        x, f8_dtype, axiswise_dim = input_tuple
        # 这里仅作性能参考，实际量化逻辑需要对齐内核
        f32_x = x.to(torch.float32)
        amax = torch.amax(torch.abs(f32_x), dim=axiswise_dim, keepdim=True)
        fp8_max = float(torch.finfo(f8_dtype).max)
        scale = fp8_max / torch.clamp(amax, min=1e-12)
        y = torch.clamp(f32_x * scale, -fp8_max, fp8_max).to(f8_dtype)
        return y, 1.0 / scale

    def get_gbps(self, input_tuple, runtime):
        """
        计算 GB/s。
        量化算子访存压力极大。
        """
        x = input_tuple[0]
        B, M, N = x.shape
        element_size = x.element_size()  # BF16/FP16 (2 bytes)
        out_element_size = 1  # FP8 (1 byte)

        # 1. 规约阶段：读 X (一次遍历)
        # 2. 量化阶段：读 X + 写 Y + 写 Scale
        # 注意：Triton 内核中为了两阶段处理，对 X 进行了两次 Load
        total_bytes_read = 2 * (B * M * N * element_size)
        total_bytes_write = B * M * N * out_element_size

        # Scale 的写回
        axiswise_dim = input_tuple[2]
        if axiswise_dim == -1:  # Per-token (B, M, 1)
            total_bytes_write += B * M * 4
        else:  # Per-channel (B, 1, N)
            total_bytes_write += B * N * 4

        return (total_bytes_read + total_bytes_write) / (runtime / 1000) / 1e9

    def get_tflops(self, input_tuple, runtime):
        """
        计算 TFLOPS。
        虽然量化被认为是访存受限，但 Abs、Max、Clamp、Mul 也是有开销的。
        """
        x = input_tuple[0]
        B, M, N = x.shape
        # 每个元素的操作：Abs(1), Max(1), Mul(1), Clamp(2)
        # 规约操作：Max (N 或 M 次)
        flops_per_el = 6
        flops = B * M * N * flops_per_el
        return flops / (runtime / 1000) / 1e12


if __name__ == "__main__":
    perf = fp8_quant_performance_metrics()
    perf.get_input_tensors()
    # 启用 Autotune，测量最稳健的性能
    perf.get_do_bench_config(warmup=50, rep=200)
    perf.run_benchmark()
