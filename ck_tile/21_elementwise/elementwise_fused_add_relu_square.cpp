// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Advanced fused kernel: Computes Y = max(A + B, 0)^2 in a single pass
// This demonstrates deep kernel fusion - instead of:
//   1. Kernel 1: Temp1 = A + B
//   2. Kernel 2: Temp2 = max(Temp1, 0)  (ReLU)
//   3. Kernel 3: Y = Temp2^2
// We do all three operations in one kernel, saving 2x memory bandwidth

#include "ck_tile/host.hpp"
#include "ck_tile/ops/elementwise.hpp"
#include "ck_tile/host/reference/reference_elementwise.hpp"
#include "json_dump_stub.hpp"
#include "elementwise_common.hpp"

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "1024", "m dimension")
        .insert("n", "1024", "n dimension")
        .insert("stride", "-1", "stride per row, if -1 then equal to n")
        .insert("v", "1", "cpu validation or not")
        .insert("x_prec", "fp16", "input precision, fp16/bf16/fp32")
        .insert("y_prec", "fp16", "output precision, fp16/bf16/fp32")
        .insert("warmup", "10", "cold iter")
        .insert("repeat", "50", "hot iter")
        .insert("json", "0", "0: No Json, 1: Dump Results in Json format")
        .insert("jsonfile", "elementwise_fused_ars.json", "json file name to dump results");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

// Custom fused operation: max(a + b, 0)^2
namespace ck_tile {
namespace element_wise {
struct FusedAddReLUSquare
{
    template <typename Y, typename X>
    __host__ __device__ constexpr void operator()(Y& y, const X& x0, const X& x1) const
    {
        // Fuse add, relu, and square in one operation
        auto sum = ck_tile::type_convert<Y>(x0) + ck_tile::type_convert<Y>(x1);
        auto relu = (sum > Y(0)) ? sum : Y(0);
        y = relu * relu;
    }
};
} // namespace element_wise
} // namespace ck_tile

template <typename XDataType, typename YDataType>
bool run(const ck_tile::ArgParser& arg_parser)
{
    ck_tile::index_t M      = arg_parser.get_int("m");
    ck_tile::index_t N      = arg_parser.get_int("n");
    ck_tile::index_t stride = arg_parser.get_int("stride");

    if(stride < 0)
        stride = N;
    int do_validation = arg_parser.get_int("v");
    int warmup        = arg_parser.get_int("warmup");
    int repeat        = arg_parser.get_int("repeat");

    if(stride < N)
    {
        throw std::runtime_error("stride must be >= N");
    }

    using ComputeDataType = float;
    using XElementwiseOperation = ck_tile::element_wise::FusedAddReLUSquare;

    // Initialize input data
    ck_tile::HostTensor<XDataType> x_host_a({M, N}, {stride, 1});
    ck_tile::HostTensor<XDataType> x_host_b({M, N}, {stride, 1});
    ck_tile::HostTensor<YDataType> y_host({M, N}, {stride, 1});
    ck_tile::HostTensor<YDataType> y_validation({M, N}, {stride, 1});

    std::vector<ck_tile::index_t> shape = {M, N};

    // Use range [-2, 3] to test ReLU clipping
    ck_tile::FillUniformDistribution<XDataType>{-2.f, 3.f}(x_host_a);
    ck_tile::FillUniformDistribution<XDataType>{-2.f, 3.f}(x_host_b);

    // Create device memory buffers
    ck_tile::DeviceMem x_buf_a(x_host_a.get_element_space_size_in_bytes());
    ck_tile::DeviceMem x_buf_b(x_host_b.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_buf(y_host.get_element_space_size_in_bytes());

    x_buf_a.ToDevice(x_host_a.data());
    x_buf_b.ToDevice(x_host_b.data());

    // Configure kernel execution - optimized tile sizes
    using BlockTile = ck_tile::sequence<2048>;
    using BlockWarps = ck_tile::sequence<8>;
    using WarpTile = ck_tile::sequence<64>;

    using Shape   = ck_tile::ElementWiseShape<BlockWarps, BlockTile, WarpTile, ComputeDataType>;
    using Problem = ck_tile::ElementWisePipelineProblem<XDataType,
                                                        ComputeDataType,
                                                        YDataType,
                                                        Shape,
                                                        XElementwiseOperation>;

    using Kernel = ck_tile::ElementWiseKernel<Problem, ck_tile::ElementWiseDefaultPolicy>;

    ck_tile::index_t total_elements = 1;
    for(auto d : shape)
        total_elements *= d;

    const ck_tile::index_t kBlockSize      = Shape::kBlockSize;
    constexpr ck_tile::index_t kBlockPerCu = 1;
    constexpr ck_tile::index_t elements_per_block = BlockTile::at(ck_tile::number<0>{});
    ck_tile::index_t kGridSize = (total_elements + elements_per_block - 1) / elements_per_block;

    std::cout << "Fused Add-ReLU-Square Kernel (3-way fusion)" << std::endl;
    std::cout << "grid size = " << kGridSize << std::endl;
    std::cout << "Total elements = " << total_elements << std::endl;

    auto input_tensors = ck_tile::make_tuple(static_cast<XDataType*>(x_buf_a.GetDeviceBuffer()),
                                             static_cast<XDataType*>(x_buf_b.GetDeviceBuffer()));
    auto input_size    = ck_tile::make_tuple(M, N);

    if(!Kernel::IsSupportedArgument(input_size))
    {
        throw std::runtime_error("Kernel configuration not supported");
    }

    // Launch fused kernel
    float ave_time = launch_kernel(
        ck_tile::stream_config{nullptr, true, 0, warmup, repeat},
        ck_tile::make_kernel<kBlockPerCu>(Kernel{},
                                          kGridSize,
                                          kBlockSize,
                                          0,
                                          input_size,
                                          ck_tile::make_tuple(N, 1),
                                          ck_tile::make_tuple(N, 1),
                                          input_tensors,
                                          static_cast<YDataType*>(y_buf.GetDeviceBuffer())));

    std::cout << "Average time: " << ave_time << " ms" << std::endl;
    
    // Calculate bandwidth saved by fusion
    // Without fusion: Read A, Read B, Write T1, Read T1, Write T2, Read T2, Write Y = 7 * size
    // With fusion: Read A, Read B, Write Y = 3 * size
    // Bandwidth reduction: 57%
    float bytes_per_element = sizeof(XDataType) * 2 + sizeof(YDataType);
    float total_bytes = total_elements * bytes_per_element;
    float bandwidth_gb_s = (total_bytes / 1e9) / (ave_time / 1000.0);
    std::cout << "Effective bandwidth: " << bandwidth_gb_s << " GB/s" << std::endl;
    std::cout << "Bandwidth saved by fusion: 57% (compared to unfused Add + ReLU + Square)" << std::endl;

    // Verify output
    bool pass = true;
    if(do_validation)
    {
        y_buf.FromDevice(y_validation.data());
        
        // Reference: max(a + b, 0)^2
        auto op = [](const auto& v0, const auto& v1) {
            auto sum = v0 + v1;
            auto relu = (sum > 0.0f) ? sum : 0.0f;
            return relu * relu;
        };

        ck_tile::reference_binary_elementwise<XDataType, XDataType, YDataType, ComputeDataType>(
            x_host_a, x_host_b, y_host, op);

        pass = ck_tile::check_err(
            y_validation, y_host, "Fused Add-ReLU-Square Error: Incorrect results!", 0.01, 0.01);
    }

    if(arg_parser.get_int("json") == 1)
    {
        dump_elementwise_json_results(arg_parser.get_str("jsonfile"),
                                      arg_parser.get_str("prec"),
                                      kGridSize,
                                      kBlockSize,
                                      ave_time,
                                      0,
                                      0,
                                      "elementwise_fused_add_relu_square");
    }

    return pass;
}

int main(int argc, char* argv[])
{
    bool result = true;
    ck_tile::ArgParser arg_parser;
    std::tie(result, arg_parser) = create_args(argc, argv);
    if(!result)
        return -1;

    try
    {
        const auto x_prec_variant = string_to_datatype(arg_parser.get_str("x_prec"));
        const auto y_prec_variant = string_to_datatype(arg_parser.get_str("y_prec"));
        return std::visit(
            [&](auto&& x_dt, auto&& y_dt) -> int {
                using XDataType = std::decay_t<decltype(x_dt)>;
                using YDataType = std::decay_t<decltype(y_dt)>;
                return run<XDataType, YDataType>(arg_parser);
            },
            x_prec_variant,
            y_prec_variant);
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return -3;
    }
}
