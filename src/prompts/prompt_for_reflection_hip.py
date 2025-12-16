# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.

prompt_hip = """
Analyze the failed test cases and provide insights 
on why the solution failed and how it could be improved. Be specific about the issues found.

**Original problem:**

{problem}

**Original code:**

{original_code}

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

prompt_exe_hip = """
Analyze the failed test cases and provide insights 
on why the solution failed and how it could be improved. Be specific about the issues found.
Runnable test is used to test if the code can be successfully executed.
Correctness test is used to test if the output of the code is correct, i.e. if the code does implement the functionality required in the original problem.

**Original problem:**

{problem}

**Original code:**

{original_code}

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

prompt_ga_hip = """ 
Analyze this HIP code and its performance(latency in ms, efficiency in its comparison with original code), and give a summary about the optimization strategy that the code uses.
Provide insights on how to generate a new code for the hip implementation with better performance. 
You can use optimization strategies such as Memory access efficiency, Hardware resource utilization, IR analysis, Assembly analysis, Kernel occupancy, etc. 
**Original problem:**

{problem}

**Original code:**

{original_code}
   
**Attempted solution:**

{code}

**Test results:**

{exe_test_result}
latency: {latency}"
latency ratio to original baseline: {latency_ratio}


**Important Instructions:**
- Think before writing the optimization and no more explanation is required after the reflection.
- You should not suggest changes to the name of the function and parameter names, counts, or order.
- generate the reflection wrapped in a code block with the tag `reflection`, e.g.
"```markdown<your reflections>```"

"""

prompt_ga_hip_gemm = """ 
Analyze this HIP code and its performance(latency in ms, efficiency in its comparison with original code), and give a summary about the heuristic strategy that the code uses.
Provide insights on how to generate a better heuristic for the gemm kernel selection with better performance. 
Consider a better heuristic based on the provided chating history and decision process for the Gemm problem. Note that only heuristic strategy is changeable, do not change the available kernel listed in DeviceOpInstance.
**Original problem:**

{problem}

**Original code:**

{original_code}
   
**Attempted solution:**

{code}

**Test results:**

{exe_test_result}
latency: {latency}"
latency ratio to original baseline: {latency_ratio}


**Important Instructions:**
- Think before writing the optimization and no more explanation is required after the reflection.
- You should not suggest changes to the name of the function and parameter names, counts, or order.
- generate the reflection wrapped in a code block with the tag `reflection`, e.g.
"```markdown<your reflections>```"

"""