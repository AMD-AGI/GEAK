// GEAK Test Harness - GEMM+Bias Baseline Kernel
// This file is compiled as a standalone shared library (.so).
// It does NOT depend on CK's build system — only CK headers + HIP runtime.
//
// Build: ./compile.py baseline  (or: make ARCH=gfx942 baseline)

#include <cstdint>
#include <array>
#include <hip/hip_runtime.h>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using Add         = ck::tensor_operation::element_wise::Add;

using ADataType        = F16;
using BDataType        = F16;
using AccDataType      = F32;
using CShuffleDataType = F16;
using DDataType        = F16;
using EDataType        = F16;

using ALayout = Row;
using BLayout = Col;
using DLayout = Row;
using ELayout = Row;

// Memory layout note:
// A(M,K) row-major: element(m,k) at offset m*StrideA + k. StrideA >= K.
// B(K,N) col-major: element(k,n) at offset k + n*StrideB. StrideB >= K.
// D(M,N) row-major: StrideD=0 broadcasts single row across all M rows.
// E(M,N) row-major: element(m,n) at offset m*StrideE + n. StrideE >= N.

using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = Add;

static constexpr auto GemmSpec =
    ck::tensor_operation::device::GemmSpecialization::MNKPadding;

// ============================================================================
// TUNING PARAMETERS — baseline configuration from CK example 03
// ============================================================================
using DeviceOpInstance =
    ck::tensor_operation::device::DeviceGemmMultipleD_Xdl_CShuffle<
        ALayout,                   // ALayout
        BLayout,                   // BLayout
        ck::Tuple<DLayout>,        // DsLayout
        ELayout,                   // ELayout
        ADataType,                 // ADataType
        BDataType,                 // BDataType
        AccDataType,               // AccDataType
        CShuffleDataType,          // CShuffleDataType
        ck::Tuple<DDataType>,      // DsDataType
        EDataType,                 // EDataType
        AElementOp,                // AElementwiseOperation
        BElementOp,                // BElementwiseOperation
        CDEElementOp,              // CDEElementwiseOperation
        GemmSpec,                  // GemmSpecialization
        1,                         // NumGemmKPrefetchStage
        256,                       // BlockSize
        256,                       // MPerBlock
        128,                       // NPerBlock
        32,                        // KPerBlock
        8,                         // AK1
        8,                         // BK1
        16,                        // MPerXDL
        16,                        // NPerXDL
        8,                         // MXdlPerWave
        4,                         // NXdlPerWave
        S<4, 64, 1>,               // ABlockTransferThreadClusterLengths
        S<1, 0, 2>,                // ABlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,                // ABlockTransferSrcAccessOrder
        2,                         // ABlockTransferSrcVectorDim
        8,                         // ABlockTransferSrcScalarPerVector
        8,                         // ABlockTransferDstScalarPerVector_AK1
        1,                         // ABlockLdsExtraM
        S<4, 64, 1>,               // BBlockTransferThreadClusterLengths
        S<1, 0, 2>,                // BBlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,                // BBlockTransferSrcAccessOrder
        2,                         // BBlockTransferSrcVectorDim
        8,                         // BBlockTransferSrcScalarPerVector
        8,                         // BBlockTransferDstScalarPerVector_BK1
        1,                         // BBlockLdsExtraN
        1,                         // CShuffleMXdlPerWavePerShuffle
        1,                         // CShuffleNXdlPerWavePerShuffle
        S<1, 32, 1, 8>,            // CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        4>;                        // CDEBlockTransferScalarPerVector_NPerBlock
// ============================================================================

extern "C" __attribute__((visibility("default")))
float run_kernel(const void* p_a,
                 const void* p_b,
                 const void* p_d0,
                 void* p_e,
                 int64_t M, int64_t N, int64_t K,
                 int64_t StrideA, int64_t StrideB,
                 int64_t StrideD0, int64_t StrideE,
                 bool time_kernel, int warmup, int nrepeat)
{
    DeviceOpInstance device_op;
    auto invoker = device_op.MakeInvoker();

    auto argument = device_op.MakeArgument(
        p_a, p_b,
        std::array<const void*, 1>{p_d0},
        p_e,
        M, N, K,
        StrideA, StrideB,
        std::array<ck::index_t, 1>{static_cast<ck::index_t>(StrideD0)},
        StrideE,
        AElementOp{}, BElementOp{}, CDEElementOp{});

    if(!device_op.IsSupportedArgument(argument))
        return -1.0f;

    StreamConfig config;
    config.stream_id_   = nullptr;
    config.time_kernel_ = time_kernel;
    config.cold_niters_ = warmup;
    config.nrepeat_     = nrepeat;

    float ms = invoker.Run(argument, config);
    (void)hipDeviceSynchronize();
    return ms;
}
