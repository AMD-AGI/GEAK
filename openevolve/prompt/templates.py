# Modifications Copyright(C)[2025] Advanced Micro Devices, Inc. All rights reserved.
# https://github.com/algorithmicsuperintelligence/openevolve  - Apache License 2.0

"""
Prompt templates for OpenEvolve
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

# Base system message template for evolution
BASE_SYSTEM_TEMPLATE = """You are an expert software developer tasked with iteratively improving a codebase.
Your job is to analyze the current program and suggest improvements based on feedback from previous attempts.
Focus on making targeted changes that will increase the program's performance metrics.
"""

BASE_EVALUATOR_SYSTEM_TEMPLATE = """You are an expert code reviewer.
Your job is to analyze the provided code and evaluate it systematically."""

# User message template for diff-based evolution
DIFF_USER_TEMPLATE = """# Current Program Information
- Current performance metrics: {metrics}
- Areas identified for improvement: {improvement_areas}

{artifacts}

# Program Evolution History
{evolution_history}

# Current Program
```{language}
{current_program}
```

# Task
Suggest improvements to the program that will lead to better performance on the specified metrics for the follwoing inputs shapes:
(64x128), (64x256), (64x512), (64x1024), (64x2048), (64x4096), (64x8192),
(128x64), (128x128), (128x256), (128x512), (128x1024), (128x2048), (128x4096), (128x8192),
(256x64), (256x128), (256x256), (256x512), (256x1024), (256x2048), (256x4096), (256x8192),
(512x64), (512x128), (512x256), (512x512), (512x1024), (512x2048), (512x4096), (512x8192),
(1024x64), (1024x128), (1024x256), (1024x512), (1024x1024),(1024x2048), (1024x4096), (1024x8192),
(2048x64), (2048x128), (2048x256), (2048x512), (2048x1024), (2048x2048),(2048x4096), (2048x8192),
(4096x64), (4096x128), (4096x256), (4096x512), (4096x1024), (4096x2048), (4096x4096), (4096x8192),
(8192x64), (8192x128), (8192x256), (8192x512), (8192x1024), (8192x2048), (8192x4096), (8192x8192).

You MUST use the exact SEARCH/REPLACE diff format shown below to indicate changes:

<<<<<<< SEARCH
# Original code to find and replace (must match exactly)
=======
# New replacement code
>>>>>>> REPLACE

Example of valid diff format:
<<<<<<< SEARCH
for i in range(m):
    for j in range(p):
        for k in range(n):
            C[i, j] += A[i, k] * B[k, j]
=======
# Reorder loops for better memory access pattern
for i in range(m):
    for k in range(n):
        for j in range(p):
            C[i, j] += A[i, k] * B[k, j]
>>>>>>> REPLACE

You can suggest multiple changes. Each SEARCH section must exactly match code in the current program.
Be thoughtful about your changes and explain your reasoning thoroughly.

IMPORTANT: Do not rewrite the entire program - focus on targeted improvements.
"""

# User message template for full rewrite
FULL_REWRITE_USER_TEMPLATE = """# Current Program Information
- Current performance metrics: {metrics}
- Areas identified for improvement: {improvement_areas}

{artifacts}

# Program Evolution History
{evolution_history}

# Current Program
```{language}
{current_program}
```

# Task
Rewrite the program to improve its performance on the specified metrics.
Provide the complete new program code.

IMPORTANT: Make sure your rewritten program maintains the same inputs and outputs
as the original program, but with improved internal implementation.

```{language}
# Your rewritten program here
```
"""

# Template for formatting evolution history
EVOLUTION_HISTORY_TEMPLATE = """## Previous Attempts

{previous_attempts}

## Top Performing Programs

{top_programs}

"""

# Template for formatting a previous attempt
PREVIOUS_ATTEMPT_TEMPLATE = """### Attempt {attempt_number}
- Changes: {changes}
- Performance: {performance}
- The analysis of this program is as follows: {reasoning}
"""

# Template for formatting a top program
TOP_PROGRAM_TEMPLATE = """### Program {program_number} (Score: {score})
```{language}
{program_snippet}
```

The analysis of this program is as follows: \n\n {reasoning}. \n You must learn from this analysis to improve your future attempts.
"""

# Template for formatting inspirations section
INSPIRATIONS_SECTION_TEMPLATE = """## Inspiration Programs

These programs represent diverse approaches and creative solutions that may inspire new ideas:

{inspiration_programs}
"""

# Template for formatting an individual inspiration program
INSPIRATION_PROGRAM_TEMPLATE = """### Inspiration {program_number} (Score: {score}, Type: {program_type})
```{language}
{program_snippet}
```

The analysis of this program is as follows: \n\n {reasoning}. \n You must learn from this analysis to improve your future attempts.
"""

# Template for evaluating a program via an LLM
EVALUATION_TEMPLATE = """Evaluate the following code on a scale of 0.0 to 1.0 for the following metrics:
1. Readability: How easy is the code to read and understand?
2. Maintainability: How easy would the code be to maintain and modify?
3. Efficiency: How efficient is the code in terms of time and space complexity?

For each metric, provide a score between 0.0 and 1.0, where 1.0 is best.

Code to evaluate:
```python
{current_program}
```

Return your evaluation as a JSON object with the following format:
{{
    "readability": [score],
    "maintainability": [score],
    "efficiency": [score],
    "reasoning": "[brief explanation of scores]"
}}
"""

# Multi-file diff user template with baseline profiling support
DIFF_USER_MULTIFILE_TEMPLATE = """# Current Program Information
- Current performance metrics: {metrics}
- Areas identified for improvement: {improvement_areas}

{artifacts}

# Program Evolution History
{evolution_history}

# Program Files
{file_listing}

# File Contents
{file_contents}

# Baseline Hardware Profiling
{baseline_profiling}

# Task
Suggest improvements to the program that will lead to better performance on the specified metrics.
Use the baseline profiling data above to identify bottlenecks and guide your optimization strategy.

## Optimization Strategy Guidance (based on profiling data)
- **Memory-bound kernels**: Focus on coalesced memory access, reducing global loads/stores, using shared memory (LDS), improving cache hit rates, and data prefetching.
- **Compute-bound kernels**: Focus on increasing instruction-level parallelism, using fused multiply-add, reducing register pressure, and optimizing warp/wavefront occupancy.
- **Latency-bound kernels**: Focus on overlapping computation with memory access, reducing synchronization barriers, and improving pipeline utilization.
- **General**: Consider autotuning (BLOCK_SIZE, num_warps, num_stages), data types (fp16/bf16/fp8), and kernel fusion.

You MUST use the exact SEARCH/REPLACE diff format shown below to indicate changes.
For multi-file programs, you MUST include the `file:` prefix to specify which file each change applies to:

<<<<<<< SEARCH file:kernel.py
# Original code to find and replace (must match exactly)
=======
# New replacement code
>>>>>>> REPLACE

Example of valid multi-file diff format:
<<<<<<< SEARCH file:kernel.py
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
=======
    x = tl.load(x_ptr + offsets, mask=mask, eviction_policy="evict_last")
    y = tl.load(y_ptr + offsets, mask=mask, eviction_policy="evict_last")
>>>>>>> REPLACE

You can suggest multiple changes across different files.
Each SEARCH section must exactly match code in the specified file.
Be thoughtful about your changes and explain your reasoning thoroughly.

IMPORTANT: Do not rewrite entire files - focus on targeted improvements.
"""

HINTS = ""

# Default templates dictionary
DEFAULT_TEMPLATES = {
    "system_message": BASE_SYSTEM_TEMPLATE,
    "evaluator_system_message": BASE_EVALUATOR_SYSTEM_TEMPLATE,
    "diff_user": DIFF_USER_TEMPLATE,
    "full_rewrite_user": FULL_REWRITE_USER_TEMPLATE,
    "evolution_history": EVOLUTION_HISTORY_TEMPLATE,
    "previous_attempt": PREVIOUS_ATTEMPT_TEMPLATE,
    "top_program": TOP_PROGRAM_TEMPLATE,
    "inspirations_section": INSPIRATIONS_SECTION_TEMPLATE,
    "inspiration_program": INSPIRATION_PROGRAM_TEMPLATE,
    "evaluation": EVALUATION_TEMPLATE,
    "diff_user_multifile": DIFF_USER_MULTIFILE_TEMPLATE,
    "hints": HINTS,
}


class TemplateManager:
    """Manages templates for prompt generation"""

    def __init__(self, template_dir: Optional[str] = None):
        self.templates = DEFAULT_TEMPLATES.copy()

        # Load templates from directory if provided
        if template_dir and os.path.isdir(template_dir):
            self._load_templates_from_dir(template_dir)

    def _load_templates_from_dir(self, template_dir: str) -> None:
        """Load templates from a directory"""
        import logging
        logger = logging.getLogger(__name__)
        
        for file_path in Path(template_dir).glob("*.txt"):
            template_name = file_path.stem
            with open(file_path, "r") as f:
                content = f.read()
                self.templates[template_name] = content
                msg = f"✅ Loaded template '{template_name}' from {file_path} ({len(content)} chars)"
                print(msg, flush=True)
                logger.info(msg)

    def get_template(self, template_name: str) -> str:
        """Get a template by name"""
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        return self.templates[template_name]

    def add_template(self, template_name: str, template: str) -> None:
        """Add or update a template"""
        self.templates[template_name] = template
