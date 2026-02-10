# Modifications Copyright(C)[2025] Advanced Micro Devices, Inc. All rights reserved.
# https://github.com/algorithmicsuperintelligence/openevolve  - Apache License 2.0

"""
Prompt templates for OpenEvolve
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

# Base system message template for evolution
BASE_SYSTEM_TEMPLATE = """You are an expert GPU kernel optimization engineer specializing in AMD ROCm / HIP / Triton.

Your mission is to find **algorithmically superior** implementations of GPU kernels that
are provably faster than the baseline while remaining functionally correct.

CRITICAL RULES:
1. **Algorithmic changes ONLY**: You must propose changes to the algorithm, data flow,
   memory access patterns, or mathematical formulation.  Examples of valid changes:
   - Kernel fusion (merging two kernels to eliminate intermediate HBM writes)
   - Operator reordering (rearranging computation to keep data in registers/LDS)
   - Strength reduction (replacing exp/div with cheaper approximations)
   - Tiling / blocking strategies (restructuring loops for cache locality)
   - Data layout transformations (AoS to SoA, padding for coalescing)
   - Algorithmic complexity reduction (O(N^2) to O(N log N))
   - Pointer precomputation and loop invariant hoisting
   - FMA exploitation and redundant computation elimination

2. **DO NOT** change autotuning parameters (BLOCK_SIZE, num_warps, num_stages,
   grid dimensions) as your primary optimization strategy.  These are parameter
   sweeps, not algorithmic improvements.  If you must adjust them for a new
   algorithm to work, explain why.

3. **Use hardware profiling data**: The prompt includes detailed Metrix hardware
   metrics (HBM bandwidth utilization, coalescing efficiency, cache hit rates,
   compute busy %, etc.) and GPU specifications.  Your optimization strategy MUST
   be informed by these metrics.  Cite specific metric values in your reasoning.

4. **Maintain correctness**: The optimized kernel must produce the same outputs
   as the baseline (within floating-point tolerance).  Do not sacrifice correctness
   for speed.

5. **Explain your reasoning**: For each change, explain (a) which hardware metric
   motivated it, (b) what algorithmic transformation you are applying, and
   (c) why it should improve performance.
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

# Baseline Hardware Profiling (from Metrix)
{baseline_profiling}

# Task: Algorithmic Optimization
Suggest **algorithmic improvements** to the program that will lead to better
performance on the specified metrics.  Focus on changes to the algorithm,
data flow, memory access patterns, or mathematical formulation.
Use the baseline profiling data above to identify bottlenecks and guide your strategy.

DO NOT simply change autotuning parameters (BLOCK_SIZE, num_warps, etc.).

For every change, explain:
1. Which hardware metric from the profiling data motivated it (cite the value)
2. What algorithmic transformation you are applying
3. Why it should improve performance

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
# Reorder loops for better memory access pattern (improves cache locality)
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

# Baseline Hardware Profiling (from Metrix)
{baseline_profiling}

# Task: Algorithmic Optimization

Study the hardware profiling data above carefully.  Your goal is to propose
**algorithmic changes** (not autotuning) that directly address the identified
bottlenecks and improve the kernel's performance.

## What counts as an algorithmic change:
- Kernel fusion: merge operations to avoid HBM round-trips
- Data layout transformation: AoS -> SoA, padding for coalescing
- Tiling / blocking: restructure loops for L1/L2/LDS reuse
- Strength reduction: replace div/sqrt/exp with approximations
- Pointer precomputation: hoist address math out of inner loops
- FMA exploitation: rewrite `a*b + c` chains to use fused multiply-add
- Redundant computation elimination: factor common sub-expressions
- Algorithmic complexity reduction: O(N^2) -> O(N log N) alternatives
- Software pipelining: overlap loads of iteration N+1 with compute of iteration N
- Write coalescing: accumulate in registers, write full tiles

## What does NOT count (DO NOT DO THESE):
- Changing BLOCK_SIZE, num_warps, num_stages, grid dimensions
- Simply adding `eviction_policy` or `cache_modifier` hints without algorithmic rationale
- Reordering function arguments or renaming variables
- Adding comments without changing logic

## Reasoning requirements:
For EVERY change you propose, you MUST explain:
1. **Which metric** from the profiling data motivated this change (cite the value)
2. **What algorithmic transformation** you are applying
3. **Why it should help**: expected effect on the bottleneck metric

You MUST use the exact SEARCH/REPLACE diff format shown below to indicate changes.
For multi-file programs, you MUST include the `file:` prefix to specify which file each change applies to:

<<<<<<< SEARCH file:kernel.py
# Original code to find and replace (must match exactly)
=======
# New replacement code
>>>>>>> REPLACE

You can suggest multiple changes across different files.
Each SEARCH section must exactly match code in the specified file.

IMPORTANT: Do not rewrite entire files - focus on targeted improvements that address
specific bottlenecks identified in the profiling data.
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
