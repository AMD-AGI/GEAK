"""LLM prompt templates for run/utils task extraction (see task_parser)."""

# Shared system line for one-shot JSON extraction from task text (parse_task_info + parse_pipeline_params).
JSON_EXTRACTION_SYSTEM_PROMPT = (
    "You are a helpful assistant that extracts structured configuration from the user's task. "
    "ALWAYS respond with a single, valid JSON object as your entire output -- no preamble, no "
    "markdown, no explanation, no tool calls, no questions. "
    "For file paths and URLs (kernel_url, repo, config, etc.): ONLY extract values that the user "
    "explicitly wrote in the task text. Do NOT guess, infer, or fabricate paths. Return null if "
    "the user did not provide an explicit path. "
    "For other fields (kernel_name, kernel_type, metric, etc.): you may infer from context. "
    "Do not investigate the filesystem; you have only the user's task text and must answer from it."
)

# User message templates: call .format(task_content=...)
PARSE_TASK_INFO_USER_TEMPLATE = """Analyze the following optimization task and extract configuration information.

Extract the following information (return null if not found):
1. kernel_name: The name of the kernel/function being optimized (e.g., "gemm", "matmul", "conv2d")
2. kernel_url: The path or URL to the SPECIFIC KERNEL FILE to optimize. ONLY extract this if
   the user explicitly provides a file path or URL in the task text. Do NOT guess or fabricate
   paths. Return null if the task does not contain an explicit kernel file path/URL.
   If extracted, it MUST end in a file extension (e.g. ``.py``, ``.hip``, ``.cu``, ``.flydsl``).
3. kernel_type: Kernel type, strictly one of "hip", "triton", "pytorch2flydsl", "flydsl", or "other".
   Use "pytorch2flydsl" when the task mentions translating PyTorch code to FlyDSL, converting PyTorch to FlyDSL, or pytorch2flydsl translation.
   Use "flydsl" when the task is about optimizing existing FlyDSL code (not translating from PyTorch).
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
  "kernel_type": "triton",  // one of: "hip", "triton", "pytorch2flydsl", "flydsl", "other"
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
1. kernel_url: The path or URL to the SPECIFIC KERNEL FILE to optimize. ONLY extract if the
   user explicitly provides a file path or URL in the task (e.g., "/path/to/silu.hip",
   "https://github.com/org/repo/blob/main/kernel.py"). Do NOT guess or fabricate paths.
   Return null if not explicitly stated. This is the kernel source file itself, NOT the
   repository root directory.
2. preprocess_dir: Path to a directory containing existing preprocessing artifacts (e.g., "/path/to/geak_output"). Only set if the user explicitly mentions reusing existing artifacts.
3. heterogeneous: Whether to use heterogeneous mode (diverse optimization strategies across GPUs). Set true if the user mentions "heterogeneous", false if they mention "homogeneous", null if not mentioned.
4. max_rounds: Maximum number of optimization rounds (integer). Only set if explicitly mentioned.
5. start_round: Round number to resume from (integer, 1-based). Only set if explicitly mentioned.
6. pipeline_intent: true if the task describes kernel optimization, performance improvement, GPU kernel work, or profiling. false if it describes general coding tasks like bug fixes, refactoring, or feature additions.
7. mode: Wall-clock budget profile -- "quick" or "full". Only set if the user explicitly chooses one.
   - "quick" -> ~1-hour total budget. Triggered by phrases like: "quick mode", "quick run",
     "fast run", "quick optimization", "1 hour", "one hour", "1h", "shorter run", "tight budget",
     "smoke test the optimization", "limit to 60 minutes", "--mode quick", "mode=quick".
   - "full" -> ~2-hour total budget. Triggered by phrases like: "full mode", "full run",
     "thorough run", "long run", "2 hours", "two hours", "2h", "extended run", "deep optimization",
     "--mode full", "mode=full".
   - null if neither is mentioned. Do NOT infer a mode from the kernel size or complexity --
     only set when the user explicitly chooses one.

Return ONLY a valid JSON object. Example:
{{{{
  "kernel_url": "/workspace/repo/kernels/silu.hip",
  "preprocess_dir": null,
  "heterogeneous": null,
  "max_rounds": 5,
  "start_round": null,
  "pipeline_intent": true,
  "mode": "quick"
}}}}

Here is the task content:
{task_content}
"""

EXTRACT_USER_CONSTRAINTS_TEMPLATE = """Analyze the following optimization task and extract mandatory constraints and prescribed optimization directives.

Extract TWO categories:

1. **constraints**: Hard rules that MUST NOT be violated (rejection criteria).
   Look for:
   - Function name constraints ("function name MUST be exactly X", "do NOT rename")
   - Signature constraints ("function signature MUST be identical")
   - Numerical correctness requirements ("output must be numerically identical")
   - Compatibility constraints ("keep all template parameters compatible")
   - Forbidden actions ("do NOT modify the test harness")
   - Any other explicit MUST / MUST NOT / DO NOT rules

2. **directives**: Prescribed optimization strategies that agents SHOULD follow as their primary approach, while retaining freedom to explore additional directions beyond these.
   Look for:
   - Specific optimization strategies ("tune block sizes", "optimize shared memory usage")
   - Memory access guidance ("improve memory coalescing", "vectorize loads/stores")
   - Architecture-specific tuning ("tune for MI355X gfx950 304 CUs")
   - Performance targets ("close efficiency gap toward 75-100% of peak HBM bandwidth")

Do NOT extract:
- Hardware descriptions without an actionable directive
- Model-level or end-to-end profiling numbers (e.g., "89.72 ms across 12288 invocations",
  "4.77% of total GPU compute time", "38.57% of 8.0 TB/s peak HBM bandwidth"). These come
  from full-model benchmarking under different conditions and MUST NOT be used as baselines
  for comparison. GEAK runs its own isolated baseline measurements.
- Workload context descriptions ("LLM inference serving", "decode path")
- File paths or kernel identifiers

IMPORTANT: Performance targets like "close efficiency gap toward 75-100% of peak bandwidth"
are valid directives. But absolute numbers from the user's profiling (durations, invocation
counts, efficiency percentages) are NOT — they reflect a different measurement environment.

Return ONLY a valid JSON object. Example:
{{
  "constraints": [
    "The output function name MUST be EXACTLY: topkGatingSoftmax. Do NOT rename it.",
    "The function signature MUST be IDENTICAL to the original.",
    "Output must be numerically identical to the original."
  ],
  "directives": [
    "Tune block sizes and wave occupancy for MI355X gfx950 (304 CUs).",
    "Optimize shared memory (LDS) usage for expert gating softmax.",
    "Improve memory coalescing for top-k routing output writes."
  ]
}}

If a category has no items, return an empty list for it.

Here is the task content:
{task_content}
"""
