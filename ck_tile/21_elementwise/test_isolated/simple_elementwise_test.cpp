// Simplified elementwise test that works with current CK-tile API
#include "ck_tile/host.hpp"
#include "ck_tile/ops/elementwise.hpp"
#include "ck_tile/host/reference/reference_elementwise.hpp"

#include <iostream>

int main(int argc, char* argv[])
{
    // Parse arguments
    ck_tile::index_t M = 4096;
    ck_tile::index_t N = 4096;
    bool do_validation = true;
    int warmup = 5;
    int repeat = 20;
    
    if(argc > 1) M = std::atoi(argv[1]);
    if(argc > 2) N = std::atoi(argv[2]);
    if(argc > 3) do_validation = (std::atoi(argv[3]) != 0);
    if(argc > 4) warmup = std::atoi(argv[4]);
    if(argc > 5) repeat = std::atoi(argv[5]);
    
    using XDataType = ck_tile::half_t;
    using YDataType = ck_tile::half_t;
    using ComputeDataType = float;
    using XElementwiseOperation = ck_tile::element_wise::Add;
    
    // Initialize host tensors
    ck_tile::index_t stride = N;
    ck_tile::HostTensor<XDataType> x_host_a({M, N}, {stride, 1});
    ck_tile::HostTensor<XDataType> x_host_b({M, N}, {stride, 1});
    ck_tile::HostTensor<YDataType> y_host({M, N}, {stride, 1});
    ck_tile::HostTensor<YDataType> y_validation({M, N}, {stride, 1});
    
    std::vector<ck_tile::index_t> shape = {M, N};
    
    ck_tile::FillUniformDistribution<XDataType>{0.f, 5.f}(x_host_a);
    ck_tile::FillUniformDistribution<XDataType>{0.f, 5.f}(x_host_b);
    
    // Create device buffers
    ck_tile::DeviceMem x_buf_a(x_host_a.get_element_space_size_in_bytes());
    ck_tile::DeviceMem x_buf_b(x_host_b.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_buf(y_host.get_element_space_size_in_bytes());
    
    x_buf_a.ToDevice(x_host_a.data());
    x_buf_b.ToDevice(x_host_b.data());
    
    // Configure kernel
    using BlockTile = ck_tile::sequence<2048>;
    using BlockWarps = ck_tile::sequence<8>;
    using WarpTile = ck_tile::sequence<64>;
    
    using Shape = ck_tile::ElementWiseShape<BlockWarps, BlockTile, WarpTile, XDataType>;
    using Problem = ck_tile::ElementWisePipelineProblem<XDataType,
                                                        ComputeDataType,
                                                        YDataType,
                                                        Shape,
                                                        XElementwiseOperation>;
    using Kernel = ck_tile::ElementWiseKernel<Problem, ck_tile::ElementWiseDefaultPolicy>;
    
    // Compute grid parameters
    ck_tile::index_t total_elements = M * N;
    constexpr ck_tile::index_t kBlockSize = Shape::kBlockSize;
    constexpr ck_tile::index_t kBlockPerCu = 1;
    constexpr ck_tile::index_t elements_per_block = BlockTile::at(ck_tile::number<0>{});
    ck_tile::index_t kGridSize = (total_elements + elements_per_block - 1) / elements_per_block;
    
    std::cout << "Elementwise Add: M=" << M << ", N=" << N << std::endl;
    std::cout << "Grid size: " << kGridSize << ", Block size: " << kBlockSize << std::endl;
    std::cout << "Total elements: " << total_elements << std::endl;
    
    auto input_tensors = ck_tile::make_tuple(
        static_cast<XDataType*>(x_buf_a.GetDeviceBuffer()),
        static_cast<XDataType*>(x_buf_b.GetDeviceBuffer()));
    
    auto input_size = ck_tile::make_tuple(M, N);
    
    // Run kernel
    float ave_time = launch_kernel(
        ck_tile::stream_config{nullptr, true, 0, warmup, repeat},
        ck_tile::make_kernel<kBlockPerCu>(
            Kernel{},
            kGridSize,
            kBlockSize,
            0,
            input_size,
            ck_tile::make_tuple(N, 1),
            ck_tile::make_tuple(N, 1),
            input_tensors,
            static_cast<YDataType*>(y_buf.GetDeviceBuffer())));
    
    std::cout << "Average time: " << ave_time << " ms" << std::endl;
    
    // Validate
    bool pass = true;
    if(do_validation)
    {
        y_buf.FromDevice(y_validation.data());
        auto op = [](const auto& v0, const auto& v1) { return v0 + v1; };
        ck_tile::reference_binary_elementwise<XDataType, XDataType, YDataType, ComputeDataType>(
            x_host_a, x_host_b, y_host, op);
        pass = ck_tile::check_err(y_validation, y_host, "Error", 0.01, 0.01);
        std::cout << (pass ? "PASSED" : "FAILED") << std::endl;
    }
    
    return pass ? 0 : 1;
}
