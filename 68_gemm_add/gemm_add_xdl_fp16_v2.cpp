// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// OPTIMIZED VERSION V2: Adjusted tile sizes and wave configuration for LDS bank conflict reduction

#include "common.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using ADataType        = F16;
using BDataType        = F16;
using AccDataType      = F32;
using CShuffleDataType = F32;
using DDataType        = F16;
using EDataType        = F16;

using ALayout = Row;
using BLayout = Col;
using DLayout = Row;
using ELayout = Row;

using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = Add;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

// OPTIMIZED V2: 
// - Changed MPerBlock from 256 to 128 (better balance with NPerBlock=128)
// - Changed KPerBlock from 32 to 64 (reduce LDS round trips)
// - Changed MXdlPerWave from 8 to 4, NXdlPerWave from 4 to 8 (better wave distribution)
// - Adjusted thread cluster for better coalescing: S<8, 32, 1> instead of S<4, 64, 1>
using DeviceOpInstance =
    ck::tensor_operation::device::DeviceGemmMultipleD_Xdl_CShuffle<ALayout,
                                                                   BLayout,
                                                                   ck::Tuple<DLayout>,
                                                                   ELayout,
                                                                   ADataType,
                                                                   BDataType,
                                                                   AccDataType,
                                                                   CShuffleDataType,
                                                                   ck::Tuple<DDataType>,
                                                                   EDataType,
                                                                   AElementOp,
                                                                   BElementOp,
                                                                   CDEElementOp,
                                                                   GemmSpec,
                                                                   1,
                                                                   256,   // BlockSize
                                                                   128,   // MPerBlock: 256->128
                                                                   128,   // NPerBlock (unchanged)
                                                                   64,    // KPerBlock: 32->64
                                                                   8,     // AK1
                                                                   8,     // BK1
                                                                   16,    // MPerXDL
                                                                   16,    // NPerXDL
                                                                   4,     // MXdlPerWave: 8->4
                                                                   8,     // NXdlPerWave: 4->8
                                                                   S<8, 32, 1>,  // ABlockTransferThreadClusterLengths: S<4,64,1>->S<8,32,1>
                                                                   S<1, 0, 2>,
                                                                   S<1, 0, 2>,
                                                                   2,
                                                                   8,
                                                                   8,
                                                                   1,
                                                                   S<8, 32, 1>,  // BBlockTransferThreadClusterLengths: S<4,64,1>->S<8,32,1>
                                                                   S<1, 0, 2>,
                                                                   S<1, 0, 2>,
                                                                   2,
                                                                   8,
                                                                   8,
                                                                   1,
                                                                   1,
                                                                   1,
                                                                   S<1, 32, 1, 8>,
                                                                   4>;

#include "run_gemm_add_example_xdl.inc"

int main(int argc, char* argv[]) { return !run_gemm_add_example(argc, argv); }
