// GEAK Test Harness - Softmax Optimized Kernel
// This file is compiled as a standalone shared library (.so).
// The LLM modifies the TUNING PARAMETERS section during optimization.
//
// Build: ./compile.py optimized  (or: make ARCH=gfx950 optimized)

#include <cstdint>
#include <vector>
#include <hip/hip_runtime.h>

#include "ck/ck.hpp"
#include "./device_softmax_impl.hpp" // Target kernel header must point to the local copy of the target kernel.
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

// ============================================================================
// TUNING PARAMETERS — modify these to optimize the kernel
//
// BlockSize:           Number of threads per block (64, 128, 256, 512, 1024)
// ClusterM, ClusterK:  Thread cluster dimensions (must multiply to BlockSize)
// SliceM, SliceK:      Work per thread (elements each thread processes)
// SrcVecDim:           Vectorized load dimension (0=M, 1=K)
// SrcScalarPerVector:  Elements per vectorized load (1, 2, 4, 8)
// OutScalarPerVector:  Elements per vectorized store (1, 2, 4, 8)
// ============================================================================
using KernelInstance =
    ck::tensor_operation::device::DeviceSoftmaxImpl<ck::half_t,  // InDataType
                                                    float,       // AccDataType
                                                    ck::half_t,  // OutDataType
                                                    PassThrough, // InElementwiseOperation
                                                    PassThrough, // AccElementwiseOperation
                                                    3,           // Rank
                                                    1,           // NumReduceDim
                                                    256,         // BlockSize
                                                    8,           // ClusterM
                                                    32,          // ClusterK
                                                    1,           // SliceM
                                                    8,           // SliceK
                                                    1,           // SrcVecDim (0=M, 1=K)
                                                    8,           // SrcScalarPerVector
                                                    8            // OutScalarPerVector
                                                    >;
// ============================================================================

extern "C" __attribute__((visibility("default"))) float run_kernel(const void* in_dev,
                                                                   void* out_dev,
                                                                   const int64_t* lengths,
                                                                   const int64_t* strides,
                                                                   int ndims,
                                                                   const int* reduce_dims,
                                                                   int n_reduce_dims,
                                                                   double alpha,
                                                                   double beta,
                                                                   bool time_kernel,
                                                                   int warmup,
                                                                   int nrepeat)
{
    KernelInstance op;

    std::vector<ck::index_t> ck_lengths(lengths, lengths + ndims);
    std::vector<ck::index_t> ck_strides(strides, strides + ndims);
    std::vector<int> ck_reduce_dims(reduce_dims, reduce_dims + n_reduce_dims);

    auto arg = op.MakeArgumentPointer(ck_lengths,
                                      ck_strides,
                                      ck_reduce_dims,
                                      alpha,
                                      beta,
                                      in_dev,
                                      out_dev,
                                      PassThrough{},
                                      PassThrough{});

    if(!op.IsSupportedArgument(arg.get()))
        return -1.0f;

    StreamConfig config;
    config.stream_id_   = nullptr;
    config.time_kernel_ = time_kernel;
    config.cold_niters_ = warmup;
    config.nrepeat_     = nrepeat;

    auto invoker = op.MakeInvokerPointer();
    float ms     = invoker->Run(arg.get(), config);

    (void)hipDeviceSynchronize();
    return ms;
}