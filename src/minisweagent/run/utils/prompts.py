"""LLM prompt templates for run/utils task extraction (see task_parser)."""

# Shared system line for one-shot JSON extraction from task text (parse_task_info + parse_pipeline_params).
JSON_EXTRACTION_SYSTEM_PROMPT = (
    "You are a helpful assistant that extracts structured configuration from the user's task. "
    "Always respond with valid JSON. Don't use tools; you must return the JSON results in one query."
)

# User message templates: call .format(task_content=...)
PARSE_TASK_INFO_USER_TEMPLATE = """Analyze the following optimization task and extract configuration information.

Extract the following information (return null if not found):
1. kernel_name: The name of the kernel/function being optimized (e.g., "gemm", "matmul", "conv2d")
2. kernel_url: The kernel URL or local path if provided
3. kernel_type: Kernel type, strictly one of "hip", "triton", or "other"
4. repo: The repository path mentioned in the task (absolute path or relative path)
5. test_command: The command to run tests or benchmarks
6. metric: The performance metric to measure (e.g., "bandwidth in GB/s", "latency in ms", "throughput")
7. num_parallel: Number of parallel optimization agents to run (integer)
8. gpu_ids: Comma-separated GPU IDs for parallel execution (e.g., "0,1,2,3")
9. output_dir: Directory path where output logs and artifacts should be saved (e.g., "outputs/topk_run", "/workspace/results")
10. model: Model name or identifier to use (e.g., "claude-sonnet-4-20250514", "gpt-4o")
11. config: Path to a YAML configuration file (e.g., "configs/my_setup.yaml", "/path/to/config.yaml")

Return ONLY a valid JSON object with these keys. Example:
{{
  "kernel_name": "matmul",
  "kernel_url": "https://github.com/org/repo/blob/main/kernel.py",
  "kernel_type": "triton",
  "repo": "/path/to/repo",
  "test_command": "python test.py",
  "metric": "Extract throughput in GFLOPS",
  "num_parallel": 4,
  "gpu_ids": "0,1,2,3",
  "output_dir": "outputs/matmul_run",
  "model": null,
  "config": null
}}

If any field cannot be determined from the task, set it to null.

Here is the task content:
{task_content}
"""

PARSE_PIPELINE_PARAMS_USER_TEMPLATE = """Analyze the following task and extract GPU kernel optimization pipeline parameters.

Extract the following (return null if not found or not applicable):
1. kernel_url: The path or URL to the SPECIFIC KERNEL FILE to optimize (e.g., "/path/to/silu.hip", "/workspace/kernels/matmul.py", "https://github.com/org/repo/blob/main/kernel.py"). This is the kernel source file itself, NOT the repository root directory.
2. preprocess_dir: Path to a directory containing existing preprocessing artifacts (e.g., "/path/to/geak_output"). Only set if the user explicitly mentions reusing existing artifacts.
3. heterogeneous: Whether to use heterogeneous mode (diverse optimization strategies across GPUs). Set true if the user mentions "heterogeneous", false if they mention "homogeneous", null if not mentioned.
4. max_rounds: Maximum number of optimization rounds (integer). Only set if explicitly mentioned.
5. start_round: Round number to resume from (integer, 1-based). Only set if explicitly mentioned.
6. pipeline_intent: true if the task describes kernel optimization, performance improvement, GPU kernel work, or profiling. false if it describes general coding tasks like bug fixes, refactoring, or feature additions.

Return ONLY a valid JSON object. Example:
{{{{
  "kernel_url": "/workspace/repo/kernels/silu.hip",
  "preprocess_dir": null,
  "heterogeneous": null,
  "max_rounds": 5,
  "start_round": null,
  "pipeline_intent": true
}}}}

Here is the task content:
{task_content}
"""
