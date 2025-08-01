
prompt = """
You are an expert Python programmer specializing in NVIDIA Triton kernels, specifically targeting **AMD GPUs using the ROCm environment**.
Your task is to generate a Python code snippet containing a Triton kernel based on the following request:

**Target Platform:** AMD GPU (ROCm)

**Request:**
{instruction}

**CRITICAL FUNCTION INFORMATION:**
Based on analysis, the implementation requires these EXACT function signatures:
{function_signatures}

**Output Requirements:**
1.  **AMD Compatibility:** Generate code compatible with AMD GPUs and ROCm. **DO NOT use CUDA-specific features or functions (e.g., `tl.libdevice`).**
2.  **Complete Code:** Generate a single, complete, and syntactically correct Python code block.
3.  **Triton Kernel:** The core logic must be implemented within a Triton kernel function decorated with `@triton.jit`.
4.  **Imports:** ALWAYS include necessary imports at the beginning:
    ```python
    import torch
    import triton
    import triton.language as tl
    # import math # Only if standard math functions are truly needed outside the kernel
    ```
    Include other imports *only if absolutely necessary*.
5.  **Function Signature (CRITICAL):**
    *   Define EACH function with EXACTLY the signature shown above.
    *   DO NOT change parameter names, counts, or order.
    *   Ensure all parameters in function calls match their function definitions.
    *   **Type Hints:** Use PyTorch tensor type hints (e.g., `x: torch.Tensor`) for tensor arguments. **DO NOT use `tl.pointer`**. Use standard Python types (e.g., `int`, `float`) or `tl.constexpr` for others.
    *   **`constexpr`:** Use `tl.constexpr` **ONLY** for arguments that *must* be known at compile time, typically block sizes (like `BLOCK_SIZE`, `BLOCK_M`) or flags that change the kernel's structure (like `IS_EVEN_K`). Simple numerical values like `eps` or `dropout_p` are usually *not* `constexpr`.
6.  **Data Types:** Be precise with data types inside the kernel (e.g., `tl.float16`, `tl.float32`, `tl.int32`). Ensure type compatibility. Assume input tensors might be `torch.float16` or `torch.float32` unless specified otherwise. Pay attention to potential type promotion/conversion needs (e.g., using `.to(tl.float32)` for accumulations).
7.  **Triton Operations:**
    *   Use Triton language functions correctly (`tl.load`, `tl.store`, `tl.dot`, `tl.arange`, `tl.program_id`, `tl.where`, `tl.atomic_cas`, etc.).
    *   **Pointers & Masks:** Be extremely careful when constructing pointers using offsets and strides. Ensure masks in `tl.load`/`tl.store` are correctly computed and match pointer dimensions. Avoid `ValueError: Mask argument cannot be block type...` or `ValueError: Unsupported ptr type...`.
    *   **`tl.dot`:** Ensure inputs are 2D blocks and have compatible types (e.g., float16, bfloat16). Int32 is generally not supported directly as input.
    *   **`tl.arange`:** Arguments `start` and `end` **must be `tl.constexpr`**.
    *   **Math:** Use functions from `tl.math` where available (e.g., `tl.math.exp`, `tl.math.sqrt`). Check function existence; avoid assuming functions like `tanh` or `log1p` exist if they don't in `tl.math`.
8.  **Triton Version:** Assume Triton version 3.1.0 or later.

**FINAL VERIFICATION:**
Before completing, verify:
1. ALL functions defined in the code have EXACT signatures matching the required function signatures above.
2. ALL function calls exactly match their definitions in terms of parameter counts and names.
3. No functions are called without being defined.
4. No parameters are missing from your implementations.

**Generated AMD ROCm Compatible Triton Kernel Code:**
"""

prompt_rocm = """
You are an expert Python programmer specializing in NVIDIA Triton kernels, specifically targeting **AMD GPUs using the ROCm environment**.
Your task is to generate a Python code snippet containing a Triton kernel based on the following request:

**Target Platform:** AMD GPU (ROCm)

**Request:**
{instruction}

**CRITICAL FUNCTION INFORMATION:**
Based on analysis, the implementation requires these EXACT function signatures:
{function_signatures}

**Output Requirements:**
1.  **AMD Compatibility:** Generate code compatible with AMD GPUs and ROCm. **DO NOT use CUDA-specific features or functions (e.g., `tl.libdevice`).**
2.  **Complete Code:** Generate a single, complete, and syntactically correct Python code block.
3.  **Triton Kernel:** The core logic must be implemented within a Triton kernel function decorated with `@triton.jit`.
4.  **Imports:** ALWAYS include necessary imports at the beginning:
    ```python
    import torch
    import triton
    import triton.language as tl
    # import math # Only if standard math functions are truly needed outside the kernel
    ```
    Include other imports *only if absolutely necessary*.
5.  **Function Signature (CRITICAL):**
    *   Define EACH function with EXACTLY the signature shown above.
    *   DO NOT change parameter names, counts, or order.
    *   Ensure all parameters in function calls match their function definitions.
    *   **Type Hints:** Use PyTorch tensor type hints (e.g., `x: torch.Tensor`) for tensor arguments. **DO NOT use `tl.pointer`**. Use standard Python types (e.g., `int`, `float`) or `tl.constexpr` for others.
    *   **`constexpr`:** Use `tl.constexpr` **ONLY** for arguments that *must* be known at compile time, typically block sizes (like `BLOCK_SIZE`, `BLOCK_M`) or flags that change the kernel's structure (like `IS_EVEN_K`). Simple numerical values like `eps` or `dropout_p` are usually *not* `constexpr`.
6.  **Data Types:** Be precise with data types inside the kernel (e.g., `tl.float16`, `tl.float32`, `tl.int32`). Ensure type compatibility. Assume input tensors might be `torch.float16` or `torch.float32` unless specified otherwise. Pay attention to potential type promotion/conversion needs (e.g., using `.to(tl.float32)` for accumulations).
7.  **Triton Operations:**
    *   Use Triton language functions correctly (`tl.load`, `tl.store`, `tl.dot`, `tl.arange`, `tl.program_id`, `tl.where`, `tl.atomic_cas`, etc.).
    *   **Pointers & Masks:** Be extremely careful when constructing pointers using offsets and strides. Ensure masks in `tl.load`/`tl.store` are correctly computed and match pointer dimensions. Avoid `ValueError: Mask argument cannot be block type...` or `ValueError: Unsupported ptr type...`.
    *   **`tl.dot`:** Ensure inputs are 2D blocks and have compatible types (e.g., float16, bfloat16). Int32 is generally not supported directly as input.
    *   **`tl.arange`:** Arguments `start` and `end` **must be `tl.constexpr`**.
    *   **Math:** Use functions from `tl.math` where available (e.g., `tl.math.exp`, `tl.math.sqrt`). Check function existence; avoid assuming functions like `tanh` or `log1p` exist if they don't in `tl.math`.
8.  **Triton Version:** Assume Triton version 3.2.0 or later.
9.  Maximize performance by exploring the following:
i. Autotuning key parameters BLOCK_SIZE, num_stages, num_warps. 
ii. Better algorithmic implementation (e.g., naive softmax vs online softmax vs fused softmax), better memory access patterns and numerical stability. 
iii. exploring all possible operator fusion strategies within the kernel while adhering to resource constraints.
Primary Autotuning Fields (Mandatory)
1. BLOCK_M, BLOCK_N, BLOCK_K
   * Tile sizes for GEMM or other tensor contractions.
   * Larger blocks improve compute density, but reduce grid-level parallelism.
   * Explore wide range of values like:
     * BLOCK: [32, ..., 128, ..., 2048, ...] 
   * Adjust based on memory reuse and L2 cache locality.
2. num_stages=n
   * Controls pipeline depth for kernel execution.
   * Rules for setting this:
     * 1 if no GEMM.
     * 2 if a single GEMM (e.g., GEMM + ReLU).
     * 1 if two GEMMs are fused (e.g., Flash Attention).
   * Optimize for latency and execution overlap.
3. num_warps
    * Controls number of warps (groups of 64 threads) to launch per block.
    * If it is too low then underutilization -> kernel runs slow.
    * If it is too high then register spill happens and shared memory is overused -> kernel runs slow.
    * You must choose a sweet spot by trying out integer range of 1 to 16.
    * You MUST NOT try the range beyond 16, it is NOT VALID. 
Examples of Autotuning Setup
Here's how Triton kernels should be decorated to allow autotuning:
    * key argument indicates the variables that change and trigger autotune to re-run. This is a must argument and you must not miss this.
    * BLOCK_M refers to the chunk of variable M that will be used for compute by a thread at a time.
    * You must ensure that variables used in the triton.Config should not be passed as arguments to the triton kernel.
For example: the following autotune config receives BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, num_warps, and num_stages as input arguments. Hence the triton kernel must not receive these arguments as inputs in the wrapper function. You must comment/delete any such instances.

NOTE: If you face kernel timeout issues, check if Grid and Program ID Mismatch exists or not for example The kernel is launched with a 1-dimensional (1D) grid, but inside the kernel, it attempts to read program IDs from a 2-dimensional (2D) grid etc.

def grid(args: dict[str, Any]) -> tuple[int]:
    # This creates a 1D grid of size (C * D, )
    return (triton.cdiv(M, args["BLOCK_SIZE_M"]) * triton.cdiv(N, args["BLOCK_SIZE_N"]), )

The grid is calculated as a single integer, creating a 1D grid, however the kernel might try to get two separate program IDs, pid_m and pid_n, as if it were a 2D grid:
pid_m = tl.program_id(0)  # Gets the ID for the first dimension
pid_n = tl.program_id(1)  # Tries to get ID for a non-existent second dimension
"""