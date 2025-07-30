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