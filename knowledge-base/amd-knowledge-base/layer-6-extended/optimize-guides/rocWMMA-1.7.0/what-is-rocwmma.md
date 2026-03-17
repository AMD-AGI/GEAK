::: {.meta description="C++ library for accelerating mixed precision matrix multiply-accumulate operations
leveraging specialized GPU matrix cores on AMD's latest discrete GPUs
:keywords: rocWMMA, ROCm, library, API, matrix, multiply"}
:::

# What is rocWMMA?

rocWMMA is a C++ header library for accelerating mixed precision matrix multiply-accumulate operations
leveraging specialized GPU matrix cores on AMD\'s latest discrete GPUs. \'roc\' being an AMD-specific
component belonging to the ROCm ecosystem, and WMMA stands for Wavefront Mixed precision Multiply Accumulate.

rocWMMA leverages modern C++ techniques. It is templated for modularity and uses meta-programming paradigms to provide opportunities for customization
and compile-time inferences and optimizations. The API is seamless across supported CDNA and RDNA architectures. It is also portable with the Nvidia
nvcuda::wmma library, allowing those users to easily migrate to the AMD platform.

The API is implemented as GPU device code which empowers users with direct use of GPU matrix cores, right from their kernel code.
Major benefits include kernel-level control which allows authoring flexibility and accessibility to compiler optimization passes in-situ
with other device code. Users can therefore decide when and where kernel run-time launches are required, which is not dictated by the API.

rocWMMA\'s API facilitates the decomposition of matrix multiply-accumulate problems into discretized blocks (also known as fragments) and enables
parallelization of block-wise operations across multiple GPU wavefronts. The programmer\'s perspective is simplified to wavefront handling of fragments,
whereas individual threads are handled internally. This can allow for faster development times and a more seamless experience across multiple architectures.
API functions include data loading and storing, matrix multiply-accumulate and helper transforms that operate on data fragment abstractions. Moreover, data movement
between global and local memory can be done cooperatively amongst the wavefronts in a threadblock to enable data sharing and re-use. Matrix multiply-accumulate
functionality supports mixed precision inputs and outputs with native fixed-precision accumulation.

Supporting code is required for GPU device management and kernel invocation. The kernel code samples and tests provided are built and launched via
the Heterogeneous-Compute Interface for Portability (HIP) ecosystem within ROCm.
