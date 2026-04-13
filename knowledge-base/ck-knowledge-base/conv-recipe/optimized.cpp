// GEAK Test Harness - 2D Convolution Forward Optimized Kernel
// This file is compiled as a standalone shared library (.so).
// The LLM modifies the TUNING PARAMETERS section during optimization.
//
// Build: ./compile.py optimized  (or: make ARCH=gfx942 optimized)
//
// This wrapper computes descriptor arrays (lengths, strides) inline
// rather than using ConvParam + make_*_packed helpers, because ConvParam's
// constructor is not header-only. The computed arrays are equivalent to what
// the helpers produce for GNHWC/GKYXC/GNHWK packed layouts.

#include <cstdint>
#include <array>
#include <hip/hip_runtime.h>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

using F16 = ck::half_t;
using F32 = float;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using InDataType       = F16;
using WeiDataType      = F16;
using AccDataType      = F32;
using CShuffleDataType = F16;
using OutDataType      = F16;

namespace ctl = ck::tensor_layout::convolution;
using InLayout  = ctl::GNHWC;
using WeiLayout = ctl::GKYXC;
using OutLayout = ctl::GNHWK;

using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto ConvSpec =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;
static constexpr auto GemmSpec =
    ck::tensor_operation::device::GemmSpecialization::MNKPadding;

static constexpr ck::index_t NDimSpatial = 2;

// ============================================================================
// TUNING PARAMETERS — modify these to optimize the kernel
// ============================================================================
using DeviceInstance =
    ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<
        NDimSpatial,
        InLayout,
        WeiLayout,
        ck::Tuple<>,               // DsLayout (no D tensors)
        OutLayout,
        InDataType,
        WeiDataType,
        AccDataType,
        CShuffleDataType,
        ck::Tuple<>,               // DsDataType (no D tensors)
        OutDataType,
        InElementOp,
        WeiElementOp,
        OutElementOp,
        ConvSpec,
        GemmSpec,
        1,                         // NumGemmKPrefetchStage
        256,                       // BlockSize
        128,                       // MPerBlock
        256,                       // NPerBlock
        32,                        // KPerBlock
        8,                         // AK1
        8,                         // BK1
        16,                        // MPerXDL
        16,                        // NPerXDL
        4,                         // MXdlPerWave
        8,                         // NXdlPerWave
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
        1,
        1,
        S<1, 32, 1, 8>,
        4>;
// ============================================================================

// Descriptor arrays use canonical CK order:
//   Input:  G, N, C, H, W    (g_n_c_wis)
//   Weight: G, K, C, Y, X    (g_k_c_xs)
//   Output: G, N, K, Ho, Wo  (g_n_k_wos)
// Strides reflect the physical packed layout (GNHWC, GKYXC, GNHWK).

extern "C" __attribute__((visibility("default")))
float run_kernel(const void* p_in,
                 const void* p_wei,
                 void* p_out,
                 int64_t G, int64_t N, int64_t K, int64_t C,
                 int64_t Hi, int64_t Wi,
                 int64_t Y, int64_t X,
                 int64_t stride_h, int64_t stride_w,
                 int64_t dilation_h, int64_t dilation_w,
                 int64_t pad_h, int64_t pad_w,
                 bool time_kernel, int warmup, int nrepeat)
{
    using idx = ck::index_t;

    const idx Ho = (Hi + 2 * pad_h - dilation_h * (Y - 1) - 1) / stride_h + 1;
    const idx Wo = (Wi + 2 * pad_w - dilation_w * (X - 1) - 1) / stride_w + 1;

    // Input GNHWC packed -> canonical G,N,C,H,W
    std::array<idx, 5> a_lengths = {
        static_cast<idx>(G), static_cast<idx>(N), static_cast<idx>(C),
        static_cast<idx>(Hi), static_cast<idx>(Wi)};
    std::array<idx, 5> a_strides = {
        static_cast<idx>(N * Hi * Wi * C),  // G
        static_cast<idx>(Hi * Wi * C),      // N
        static_cast<idx>(1),                // C (innermost in GNHWC)
        static_cast<idx>(Wi * C),           // H
        static_cast<idx>(C)};               // W

    // Weight GKYXC packed -> canonical G,K,C,Y,X
    std::array<idx, 5> b_lengths = {
        static_cast<idx>(G), static_cast<idx>(K), static_cast<idx>(C),
        static_cast<idx>(Y), static_cast<idx>(X)};
    std::array<idx, 5> b_strides = {
        static_cast<idx>(K * Y * X * C),    // G
        static_cast<idx>(Y * X * C),        // K
        static_cast<idx>(1),                // C (innermost in GKYXC)
        static_cast<idx>(X * C),            // Y
        static_cast<idx>(C)};               // X

    // Output GNHWK packed -> canonical G,N,K,Ho,Wo
    std::array<idx, 5> e_lengths = {
        static_cast<idx>(G), static_cast<idx>(N), static_cast<idx>(K),
        Ho, Wo};
    std::array<idx, 5> e_strides = {
        static_cast<idx>(N * Ho * Wo * K),  // G
        static_cast<idx>(Ho * Wo * K),      // N
        static_cast<idx>(1),                // K (innermost in GNHWK)
        static_cast<idx>(Wo * K),           // Ho
        static_cast<idx>(K)};               // Wo

    std::array<idx, NDimSpatial> conv_filter_strides  = {static_cast<idx>(stride_h), static_cast<idx>(stride_w)};
    std::array<idx, NDimSpatial> conv_filter_dilations = {static_cast<idx>(dilation_h), static_cast<idx>(dilation_w)};
    std::array<idx, NDimSpatial> input_left_pads       = {static_cast<idx>(pad_h), static_cast<idx>(pad_w)};
    std::array<idx, NDimSpatial> input_right_pads      = {static_cast<idx>(pad_h), static_cast<idx>(pad_w)};

    DeviceInstance conv;
    auto invoker  = conv.MakeInvoker();
    auto argument = conv.MakeArgument(
        p_in,
        p_wei,
        std::array<const void*, 0>{},
        p_out,
        a_lengths, a_strides,
        b_lengths, b_strides,
        std::array<std::array<idx, NDimSpatial + 3>, 0>{{}},
        std::array<std::array<idx, NDimSpatial + 3>, 0>{{}},
        e_lengths, e_strides,
        conv_filter_strides,
        conv_filter_dilations,
        input_left_pads,
        input_right_pads,
        InElementOp{},
        WeiElementOp{},
        OutElementOp{});

    if(!conv.IsSupportedArgument(argument))
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
