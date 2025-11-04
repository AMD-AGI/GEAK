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

system_prompt = """\nThink before writing the optimization and no more explanation is required after the reflection. 
You should not suggest changes to the name of the function and parameter names, counts, or order.
Output your answer in json format, with the format as follows: {\"reflection\": \"\"}. Please strictly output in JSON format.
generate reflection in the \"reflection\" field."""


prompt_evolve_reflect = """
You are an expert Python programmer specializing in writing and optimizing Triton kernels for AMD GPUs using the ROCm environment.
You are tasked with iteratively improving a codebase.
Given the original problem, metrics information, a history of previous implementations with their test results and reflections, previous top implementations with their test results and current implementation with its test error message, your job is to analyze the error messages and provide insights on why the solution failed and how it could be fixed. Be specific about the issues found.

# Original Problem:
{instruction}

## CRITICAL FUNCTION INFORMATION:
Based on analysis, the implementation requires these EXACT function signatures:
{function_signatures}

## Requirements:
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

# Metrics information
{metrics_info}

# Program Evolution History
{evolution_history}

# Current Program
```python
{current_program}
```

- Test result of the current program: {test_result}
- Reflection on the current program: {reflection}

# Important Instructions:
- Think carefully before writing the reflection.
- Do not include any explanation outside the reflection block.
- Your output must be a reflection wrapped in a code block tagged as reflection, like below:
"```reflection
<your reflection goes here>
```"

"""

"""
text-based crossover and mutation:
crossover1: combine one parent and one inspriration strategy
crossover2: combine one parent and the strategy extracted from 2 insprirations
mutation: identify the potential to improve for the parent
(mutation: the same strategy but another implementation)
"""

prompt_evolve_strategy_optimize = """
You are an expert Python programmer specializing in writing and optimizing Triton kernels for AMD GPUs using the ROCm environment.
You are tasked with iteratively improving a codebase.
You are given the original problem, metrics information, a history of previous implementations with their test results and reflections, previous top implementations with their test results and current implementation with its test error message.
Your task:
- Summarize the optimization strategy used by the current implementation, including key decisions and tuning knobs.
- Analyze its performance characteristics and identify potential weaknesses or bottlenecks.
- Provide detailed insights on how to generate a new implementation with improved performance.

Available Optimization Strategies You May Use:
You are encouraged to draw on the following techniques (as relevant):
- Memory access efficiency (e.g., coalescing, shared memory usage)
- Hardware resource utilization (registers, shared memory, etc.)
- Intermediate Representation (IR) analysis
- Assembly-level analysis
- Kernel occupancy analysis
- TorchInductor integration with Triton tuning knobs
- Auto-tunable kernel configurations
- Environment variable settings (e.g., ROCm tuning flags)
- Any other performance tuning methods applicable to ROCm and AMD GPUs

# Original Problem:
{instruction}

## CRITICAL FUNCTION INFORMATION:
Based on analysis, the implementation requires these EXACT function signatures:
{function_signatures}

## Requirements:
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

# Metrics information
{metrics_info}

# Program Evolution History
{evolution_history}

# Current Program
```python
{current_program}
```

- Test result of the current program: {test_result}
- Reflection on the current program: {reflection}

# Important Instructions:
- Think carefully before writing the reflection.
- Do not include any explanation outside the reflection block.
- Your output must be a reflection wrapped in a code block tagged as reflection, like below:
"```reflection
<your reflection goes here>
```"

"""


prompt_extract_strategy_1 = """
You are an expert Python programmer specializing in writing and optimizing Triton kernels for AMD GPUs using the ROCm environment.
You are given the original problem, metrics information, a history of previous implementations with their test results and reflections, and one additional top implementations with its test result.
Your task is to:
1. Analyze the test results and reflections to determine which implementation performs better.
2. Summarize the reasons why the better implementation outperforms the other.
3. Extract and clearly describe the key strategies or techniques that the better implementation uses to achieve higher performance or accuracy.

# Original Problem:
{instruction}

## CRITICAL FUNCTION INFORMATION:
Based on analysis, the implementation requires these EXACT function signatures:
{function_signatures}

## Requirements:
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

# Metrics information
{metrics_info}

# Previous Implementations
{top_programs}

# Important Instructions:
- Think carefully before writing the reflection.
- Do not include any explanation outside the reflection block.
- Your output must be a reflection wrapped in a code block tagged as reflection, like below:
"```reflection
<your reflection goes here>
```"
"""


prompt_extract_strategy_2 = """
You are an expert Python programmer specializing in writing and optimizing Triton kernels for AMD GPUs using the ROCm environment.
You are given the original problem, metrics information, two previous implementations with their test results and reflections.
Your task is to:
1. Analyze the test results and reflections to determine which implementation performs better.
2. Summarize the reasons why the better implementation outperforms the other.
3. Extract and clearly describe the key strategies or techniques that the better implementation uses to achieve higher performance or accuracy.

# Original Problem:
{instruction}

## CRITICAL FUNCTION INFORMATION:
Based on analysis, the implementation requires these EXACT function signatures:
{function_signatures}

## Requirements:
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

# Metrics information
{metrics_info}

# Previous Implementations
{top_programs}

# Important Instructions:
- Think carefully before writing the reflection.
- Do not include any explanation outside the reflection block.
- Your output must be a reflection wrapped in a code block tagged as reflection, like below:
"```reflection
<your reflection goes here>
```"
"""