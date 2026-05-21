# HIP — memory (KB) key-parameter patterns

Regex patterns for extracting "key parameters" from HIP kernels and
experience records.

Format (same as Triton): `LABEL   REGEX`, one per line; `#` starts a
comment.

    blockdim        (?:blockDim\.[xyz]|BLOCK_DIM_[XYZ])
    griddim         (?:gridDim\.[xyz])
    shared_mem      (?:__shared__\s+\w+\s+\w+\s*\[)
    ldg             __ldg\s*\(
    hip_ext_load    torch\.utils\.cpp_extension\.load(?:_inline)?\s*\(
    hip_launch      hipLaunchKernelGGL\s*\(
    mfma_intrinsic  __builtin_amdgcn_mfma_\w+
    warp_shuffle    __shfl(?:_sync|_down|_up|_xor)?\s*\(
    warp_ballot     __ballot(?:_sync)?\s*\(
    restrict_ptr    __restrict__\s*\*?
