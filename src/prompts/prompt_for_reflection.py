prompt = """
You are an expert in writing Triton operators for efficient GPU programming. Analyze the failed test cases and provide insights 
on why the solution failed and how it could be improved. Be specific about the issues found.

**Original problem:**

{problem}

**Attempted solution:**

{solution}

**Test results:**

{test_result}

**Important Instructions:**
- Think before writing the reflection and no more explanation is required after the reflection.
- You should not suggest changes to the name of the function.
- generate the reflection wrapped in a code block with the tag `reflection`, e.g.
"```markdown<your reflections>```"

"""

prompt_exe = """
You are an expert in writing Triton operators for efficient GPU programming. Analyze the failed test cases and provide insights 
on why the solution failed and how it could be improved. Be specific about the issues found.
Runnable test is used to test if the code can be successfully executed.
Correctness test is used to test if the output of the code is correct, i.e. if the code does implement the functionality required in the original problem.

**Original problem:**

{problem}

**Attempted solution:**

{solution}

**Results for runnable test:**

{call_test_result}

**Results for correctness test:**

{exe_test_result}

**Important Instructions:**
- Think before writing the reflection and no more explanation is required after the reflection.
- You should not suggest changes to the name of the function.
- generate the reflection wrapped in a code block with the tag `reflection`, e.g.
"```markdown<your reflections>```"

"""

prompt_ga = """
You are an expert in writing Triton operators for efficient GPU programming. 
Analyze this Triton code and its performance(latency in ms and efficiency in TFLOPS or GB/s), and give a summary about the optimization strategy that the code uses.
Provide insights on how to generate a new code with better performance. 
You can use optimization strategies such as Memory access efficiency, Hardware resource utilization, IR analysis, Assembly analysis, Kernel occupancy, 
TorchInductor with Triton tuning knobs and Auto-tunable kernel configurations and environment variables.

**Original problem:**

{problem}
   
**Triton code:**

{code}

**Test results:**

latency: {latency}"

efficiency(TFLOPS, GB/s): {efficiency}

**Important Instructions:**
- Think before writing the optimization and no more explanation is required after the reflection.
- You should not suggest changes to the name of the function and parameter names, counts, or order.
- generate the reflection wrapped in a code block with the tag `reflection`, e.g.
"```markdown<your reflections>```"

"""

prompt_rocm = """
You are an expert in writing Triton operators for efficient GPU programming. Analyze the failed test cases and provide insights 
on why the solution failed and how it could be improved. Be specific about the issues found.

**Original problem:**

{problem}

**Attempted solution:**

{solution}

**Test results:**

{test_result}

**Important Instructions:**
- Think before writing the reflection and no more explanation is required after the reflection.
- You should not suggest changes to the name of the function.
- generate the reflection wrapped in a code block with the tag `reflection`, e.g.
"```markdown<your reflections>```"

Maximize performance by exploring the following:
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

prompt_exe_rocm = """
You are an expert in writing Triton operators for efficient GPU programming. Analyze the failed test cases and provide insights 
on why the solution failed and how it could be improved. Be specific about the issues found.
Runnable test is used to test if the code can be successfully executed.
Correctness test is used to test if the output of the code is correct, i.e. if the code does implement the functionality required in the original problem.

**Original problem:**

{problem}

**Attempted solution:**

{solution}

**Results for runnable test:**

{call_test_result}

**Results for correctness test:**

{exe_test_result}

**Important Instructions:**
- Think before writing the reflection and no more explanation is required after the reflection.
- You should not suggest changes to the name of the function.
- generate the reflection wrapped in a code block with the tag `reflection`, e.g.
"```markdown<your reflections>```"

Maximize performance by exploring the following:
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

prompt_ga_rocm = """
You are an expert in writing Triton operators for efficient GPU programming. 
Analyze this Triton code and its performance(speedup[vs reference kernel] for e.g. (1.6x) and efficiency in TFLOPS or GB/s), and give a summary about the optimization strategy that the code uses.
Provide insights on how to generate a new code with better performance. 
You can use optimization strategies such as Memory access efficiency, Hardware resource utilization, IR analysis, Assembly analysis, Kernel occupancy, 
TorchInductor with Triton tuning knobs and Auto-tunable kernel configurations and environment variables.

**Original problem:**

{problem}
   
**Triton code:**

{code}

**Test results:**

Speedup: {latency}"

efficiency(TFLOPS, GB/s): {efficiency}

**Important Instructions:**
- Think before writing the optimization and no more explanation is required after the reflection.
- You should not suggest changes to the name of the function and parameter names, counts, or order.
- generate the reflection wrapped in a code block with the tag `reflection`, e.g.
"```markdown<your reflections>```"

Maximize performance by exploring the following:
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