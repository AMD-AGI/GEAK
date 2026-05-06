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
3. kernel_type: Kernel type, strictly one of "hip", "triton", "pytorch2flydsl", "flydsl", or "other".
   Use "pytorch2flydsl" when the task mentions translating PyTorch code to FlyDSL, converting PyTorch to FlyDSL, or pytorch2flydsl translation.
   Use "flydsl" when the task is about optimizing existing FlyDSL code (not translating from PyTorch).
4. repo: The repository path mentioned in the task (absolute path or relative path)
5. test_command: The command to run tests or benchmarks
5b. test_harness: Path or URL to a USER-PROVIDED test/benchmark HARNESS FILE.
   Look for phrases like "Use the test harness at <path>", "test harness:
   <path>", "use harness <path>", "harness file <path>", "with the harness
   at <path>".  This is a SINGLE FILE the user wants the pipeline to use
   verbatim — DIFFERENT from test_command (which is a shell command).
   Set null if the user only described a command but no harness file.
   When BOTH a test_command and a test_harness are mentioned, populate
   both — they're complementary (the harness is the file; the command
   is one way to invoke it).
6. metric: The performance metric to measure (e.g., "bandwidth in GB/s", "latency in ms", "throughput")
7. num_parallel: Number of parallel optimization agents to run (integer).
   If the user did not explicitly state a number but specified a GPU list,
   you MAY leave this null — the pipeline will auto-derive it as
   ``len(gpu_ids)``.  Only set this when the user explicitly says
   "N parallel" / "N workers" etc.
8. gpu_ids: GPU IDs for parallel execution.  Accept any of these forms
   verbatim from the user's prompt:
     - "0,1,2,3"   (comma-separated list)
     - "4-7"       (range, inclusive — e.g., "Use GPUs 4-7" → "4-7")
     - "0,1,4-7"   (mixed)
     - "4"         (single GPU)
   Prefer the FORM the user used (don't expand "4-7" into "4,5,6,7"
   unnecessarily — downstream parsing handles both).
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
  "test_harness": "/path/to/test_kernel_harness.py",
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
3. mode: Execution mode for the optimization run.  One of:
     - "fixed"   — run the same task body across N parallel agents (identical copies).  Set if the user asks for replicated / parallel / best-of-N runs of a single strategy.
     - "planned" — let the planner LLM generate N diverse strategies and run each in its own agent.  Set if the user asks for "diverse strategies", "planner", or wording that suggests multiple distinct approaches.
     - "auto"    — let the pipeline pick fixed vs planned.  This is the default; only set another value if the user is explicit.
   Return null if the user is not explicit about the mode.
   Legacy terms: "heterogeneous" should map to "planned"; "homogeneous" to "fixed".
   NOTE: Do NOT emit "translate" here.  Translation is a separate preprocess phase triggered by ``target_language`` (field 7 below), not a pipeline mode.
4. max_rounds: Maximum number of optimization rounds (integer). Only set if explicitly mentioned.
5. start_round: Round number to resume from (integer, 1-based). Only set if explicitly mentioned.
6. pipeline_intent: true if the task describes kernel optimization, performance improvement, GPU kernel work, or profiling. false if it describes general coding tasks like bug fixes, refactoring, or feature additions.
7. target_language: Target kernel language if the user asks to translate / port / convert the kernel (e.g. "triton", "hip", "cuda").  Only set if the user is explicit about wanting translation.  Null otherwise.  When set, the pipeline runs a translation preprocess phase before optimization.

Return ONLY a valid JSON object. Example:
{{{{
  "kernel_url": "/workspace/repo/kernels/silu.hip",
  "preprocess_dir": null,
  "mode": null,
  "max_rounds": 5,
  "start_round": null,
  "pipeline_intent": true,
  "target_language": null
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
