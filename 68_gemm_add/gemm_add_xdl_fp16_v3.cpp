// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// OPTIMIZED VERSION V3: Conservative optimization focusing on LDS access pattern

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

// OPTIMIZED V3: 
// - Changed MPerBlock/NPerBlock to 128/256 (swapped from 256/128) for better balance
// - Adjusted MXdlPerWave/NXdlPerWave to 4/8 (from 8/4) to match new tile ratio
// - Keep thread cluster and vector widths same as baseline
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
                                                                   256,   // NPerBlock: 128->256
                                                                   32,    // KPerBlock (unchanged)
                                                                   8,     // AK1
                                                                   8,     // BK1
                                                                   16,    // MPerXDL
                                                                   16,    // NPerXDL
                                                                   4,     // MXdlPerWave: 8->4
                                                                   8,     // NXdlPerWave: 4->8
                                                                   S<4, 64, 1>,
                                                                   S<1, 0, 2>,
                                                                   S<1, 0, 2>,
                                                                   2,
                                                                   8,
                                                                   8,
                                                                   1,
                                                                   S<4, 64, 1>,
                                                                   S<1, 0, 2>,
                                                                   S<1, 0, 2>,
                                                                   2,
                                                                   8,
                                                                   8,
                                                                   1,
                                                                   1,
                                                                   1,
                                                                   S<1, 32, 1, 8>,
                                                                   8>;  // CDEBlockTransferScalarPerVector: 4->8

#include "run_gemm_add_example_xdl.inc"

int main(int argc, char* argv[]) { return !run_gemm_add_example(argc, argv); }
