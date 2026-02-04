"""Kernel ERCS MCP Server.

ERCS = Evaluation, Reflection, Compatibility, Specs

This module provides an MCP server that exposes kernel optimization tools
for Triton/AMD GPU kernel optimization:
- Evaluate: LLM-based kernel quality scoring
- Reflect: Analyze test results and suggest next steps
- Compatibility: Check AMD GPU compatibility
- Specs: Get AMD MI350X hardware reference
"""

import json
import os
from typing import Any

from fastmcp import FastMCP

# Try to import litellm for LLM calls
try:
    from litellm import completion
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

# Try to import anthropic for AMD gateway
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


# Create the MCP server
mcp = FastMCP(
    name="kernel-ercs",
    instructions="Kernel ERCS: Evaluation, Reflection, Compatibility, Specs tools for Triton/AMD GPU kernel optimization"
)


def is_amd_model(model: str) -> bool:
    """Check if the model is an AMD gateway model."""
    model_lower = model.lower()
    return (
        model.startswith("amd/") or
        "claude-sonnet-4" in model_lower or
        "claude-opus-4" in model_lower or
        "gpt-5" in model_lower
    )


def call_amd_gateway(messages: list, model: str, temperature: float = 0.1) -> str:
    """Call AMD LLM gateway directly."""
    if not ANTHROPIC_AVAILABLE:
        raise ImportError("anthropic package not available. Install with: pip install anthropic")
    
    api_key = os.environ.get("AMD_LLM_API_KEY") or os.environ.get("LLM_GATEWAY_KEY")
    if not api_key:
        raise ValueError("AMD_LLM_API_KEY or LLM_GATEWAY_KEY environment variable not set")
    
    model_name = model.removeprefix("amd/")
    
    try:
        user = os.getlogin()
    except OSError:
        user = os.environ.get("USER", "unknown")
    
    client = anthropic.Anthropic(
        api_key="dummy",
        base_url="https://llm-api.amd.com/Anthropic",
        default_headers={
            "Ocp-Apim-Subscription-Key": api_key,
            "user": user,
            "anthropic-version": "2023-10-16",
        },
    )
    
    system_content = ""
    filtered_messages = []
    for msg in messages:
        if msg.get("role") == "system":
            system_content = msg.get("content", "")
        else:
            filtered_messages.append(msg)
    
    response = client.messages.create(
        model=model_name,
        max_tokens=4096,
        system=system_content if system_content else anthropic.NOT_GIVEN,
        messages=filtered_messages,
        temperature=temperature,
    )
    
    return response.content[0].text


def call_llm(messages: list, model: str, temperature: float = 0.1) -> str:
    """Call LLM using appropriate backend."""
    if is_amd_model(model) and ANTHROPIC_AVAILABLE:
        return call_amd_gateway(messages, model, temperature)
    elif LITELLM_AVAILABLE:
        api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        response = completion(
            model=model,
            messages=messages,
            api_key=api_key,
            temperature=temperature,
        )
        return response.choices[0].message.content
    else:
        raise RuntimeError("No LLM backend available. Install anthropic or litellm.")


# Kernel Evaluation Prompt
KERNEL_EVALUATION_PROMPT = """
Evaluate the following Triton kernel on a scale of 0.0 to 1.0 for each of the listed criteria.
For each criteria you must provide:
* A score (float between 0.0 (worst) and 1.0 (best))
* A brief justification (1-2 lines)
* Actionable suggestions, if any

**Target Hardware: AMD MI350X GPU (gfx950)**
- 256 Compute Units
- Wavefront size: 64 threads
- Max 32 waves per CU
- Max workgroup size: 1024 threads
- 288 GB HBM3 memory
- Cache line: 128 bytes

**Evaluation Criteria:**
1. Fusion Intelligence: Does the kernel smartly fuse compatible operations to reduce memory I/O and kernel launches?
2. Block Size Appropriateness: Are the block sizes appropriate for the problem size and AMD hardware? (BLOCK should be multiples of 64, max 1024)
3. Memory Access Efficiency: Does it optimize memory layout, coalesced access (128-byte aligned), and reduce redundant reads/writes?
4. Algorithmic Complexity: Does it fuse multiple for-loops smartly? Are there redundant nested for-loops?
5. Warp/Wavefront Utilization: Does it use thread blocks that fully utilize AMD wavefronts (64 threads)?
6. Software Pipelining: Is num_stages set appropriately? (1 for no GEMM, 2 for single GEMM, 1 for fused double-GEMM)
7. Numerical Stability: Is it numerically safe for large input ranges (e.g., uses max-subtraction in softmax)?
8. Correctness and Portability: Does the kernel handle edge cases? Does it avoid CUDA-only features like tl.libdevice?
9. Optimization Scope: Is the kernel missing techniques from well-known optimization methods?

Evaluate the following Triton kernel:
```python
{kernel_code}
```

Provide scores in JSON format:
{{
    "fusion_intelligence": <float 0.0-1.0>,
    "block_size_appropriateness": <float 0.0-1.0>,
    "memory_access_efficiency": <float 0.0-1.0>,
    "algorithmic_complexity": <float 0.0-1.0>,
    "warp_wavefront_utilization": <float 0.0-1.0>,
    "software_pipelining": <float 0.0-1.0>,
    "numerical_stability": <float 0.0-1.0>,
    "correctness_and_portability": <float 0.0-1.0>,
    "optimization_scope": <float 0.0-1.0>,
    "total_score": <float sum of above>,
    "reasoning": "<Your reasoning and suggestions>",
    "top_issues": ["<issue1>", "<issue2>"],
    "suggested_improvements": ["<improvement1>", "<improvement2>"]
}}

Only output valid JSON, no additional text.
"""


REFLECTION_PROMPT = """
You are an expert in Triton kernels for AMD GPUs (ROCm).
Analyze why this kernel optimization attempt achieved its result and suggest next steps.

# Current Kernel:
```python
{kernel_code}
```

# Test Result:
- Correctness: {correctness_status}
- Speedup vs baseline: {speedup}x
- Test Output: 
{test_output}

# Optimization History:
{history}

# Strategies Already Tried:
{tried_strategies}

## AMD GPU Guidelines:
- AMD wavefront = 64 threads (not 32)
- Block sizes should be multiples of 64, max 1024
- DO NOT use CUDA-only features like tl.libdevice
- num_warps range: [1-16] only

Analyze and provide next steps in JSON:
{{
    "analysis": "<Why this result occurred>",
    "is_improvement": <bool>,
    "bottleneck": "<memory | compute | correctness | launch_overhead | other>",
    "root_cause": "<Specific technical reason>",
    "next_strategy": {{
        "name": "<strategy name>",
        "description": "<What changes to make>",
        "expected_impact": "<Why this might help>",
        "code_hints": "<Specific code patterns to try>"
    }},
    "avoid": ["<strategies that wont help>"],
    "exploration_status": {{
        "strategies_exhausted": <bool>,
        "confidence_in_next": "<high | medium | low>",
        "recommendation": "<continue | stop | try_radically_different>"
    }}
}}

Only output valid JSON.
"""


@mcp.tool()
def evaluate_kernel_quality(
    kernel_code: str,
    model: str = "amd/claude-sonnet-4-20250514"
) -> dict[str, Any]:
    """
    Evaluate Triton kernel quality using LLM analysis.
    
    Analyzes the kernel code and provides scores across 9 quality criteria
    specific to AMD GPU optimization (MI350X/gfx950).
    
    Args:
        kernel_code: The Triton kernel code to evaluate
        model: LLM model to use (default: amd/claude-sonnet-4-20250514)
    
    Returns:
        dict with scores (0.0-1.0 each), total_score, reasoning, and suggestions
    """
    try:
        prompt = KERNEL_EVALUATION_PROMPT.format(kernel_code=kernel_code)
        messages = [{"role": "user", "content": prompt}]
        
        result_text = call_llm(messages, model, temperature=0.1)
        
        # Parse JSON response
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]
        
        result = json.loads(result_text.strip())
        
        # Calculate total if not present
        score_fields = [
            "fusion_intelligence", "block_size_appropriateness",
            "memory_access_efficiency", "algorithmic_complexity",
            "warp_wavefront_utilization", "software_pipelining",
            "numerical_stability", "correctness_and_portability",
            "optimization_scope"
        ]
        
        if "total_score" not in result:
            result["total_score"] = sum(result.get(f, 0.0) for f in score_fields)
        
        result["ready_to_test"] = result["total_score"] >= 6.0
        
        return result
        
    except json.JSONDecodeError as e:
        return {
            "error": f"Failed to parse LLM response: {e}",
            "raw_response": result_text if "result_text" in dir() else None,
            "total_score": 0.0,
            "ready_to_test": False
        }
    except Exception as e:
        return {
            "error": f"Evaluation failed: {e}",
            "total_score": 0.0,
            "ready_to_test": False
        }


@mcp.tool()
def reflect_on_kernel_result(
    kernel_code: str,
    test_output: str,
    speedup: float = 0.0,
    correctness_status: str = "unknown",
    history: str = "",
    tried_strategies: str = "",
    model: str = "amd/claude-sonnet-4-20250514"
) -> dict[str, Any]:
    """
    Analyze kernel test results and get targeted improvement suggestions.
    
    Uses LLM to understand why the kernel achieved its result and what
    specific optimizations to try next.
    
    Args:
        kernel_code: The Triton kernel code that was tested
        test_output: Output from running the test (stdout/stderr)
        speedup: Measured speedup vs baseline (e.g., 1.5 = 50% faster)
        correctness_status: "passed", "failed", or "unknown"
        history: Summary of previous optimization attempts
        tried_strategies: Comma-separated list of strategies already tried
        model: LLM model to use
    
    Returns:
        dict with analysis, bottleneck, next_strategy, and exploration_status
    """
    if not history:
        history = "No previous attempts"
    if not tried_strategies:
        tried_strategies = "None yet"
    
    try:
        prompt = REFLECTION_PROMPT.format(
            kernel_code=kernel_code,
            correctness_status=correctness_status,
            speedup=speedup,
            test_output=test_output[:2000],
            history=history,
            tried_strategies=tried_strategies
        )
        
        messages = [{"role": "user", "content": prompt}]
        result_text = call_llm(messages, model, temperature=0.2)
        
        # Parse JSON
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]
        
        result = json.loads(result_text.strip())
        
        # Ensure required fields
        result.setdefault("analysis", "Analysis not available")
        result.setdefault("bottleneck", "unknown")
        result.setdefault("next_strategy", {"name": "unknown", "description": "No suggestion"})
        result.setdefault("exploration_status", {"recommendation": "continue"})
        
        return result
        
    except json.JSONDecodeError as e:
        return {
            "error": f"Failed to parse response: {e}",
            "analysis": "Reflection failed",
            "bottleneck": "unknown",
            "next_strategy": {"name": "retry", "description": "Try again"},
            "exploration_status": {"recommendation": "continue"}
        }
    except Exception as e:
        return {
            "error": f"Reflection failed: {e}",
            "analysis": str(e),
            "bottleneck": "unknown",
            "next_strategy": {"name": "retry", "description": "Fix error and retry"},
            "exploration_status": {"recommendation": "continue"}
        }


@mcp.tool()
def get_amd_gpu_specs() -> dict[str, Any]:
    """
    Get AMD MI350X GPU specifications for kernel optimization reference.
    
    Returns hardware specs and optimization guidelines for the target GPU.
    
    Returns:
        dict with GPU specifications and optimization tips
    """
    return {
        "gpu": {
            "name": "AMD Instinct MI350X",
            "architecture": "gfx950 (CDNA3+)",
            "compute_units": 256,
            "wavefront_size": 64,
            "max_waves_per_cu": 32,
            "max_workgroup_size": 1024,
            "cache_line_size": 128,
            "vram": "288 GB HBM3",
            "max_gpu_clock": "2400 MHz",
            "memory_clock": "1900 MHz"
        },
        "optimization_tips": {
            "block_sizes": "Use multiples of 64 (wavefront size), max 1024",
            "memory_alignment": "Align to 128 bytes for coalesced access",
            "num_warps": "Range [1-16] only, higher values invalid on AMD",
            "num_stages": "1 for no GEMM, 2 for single GEMM, 1 for fused GEMMs",
            "avoid": ["tl.libdevice", "CUDA-specific intrinsics"],
            "prefer": ["Vectorized loads", "Fused operations", "Online algorithms"]
        },
        "memory_hierarchy": {
            "registers": "256 VGPRs/wave, 1 cycle latency",
            "lds_shared": "64 KB/CU, ~20 cycles",
            "l1_cache": "Per CU, ~50 cycles",
            "l2_cache": "Shared Infinity Cache, ~100 cycles",
            "hbm3": "288 GB, ~300+ cycles"
        }
    }


@mcp.tool()
def check_kernel_compatibility(kernel_code: str) -> dict[str, Any]:
    """
    Quick check for AMD GPU compatibility issues in kernel code.
    
    Scans the kernel for common CUDA-only features and AMD-incompatible patterns.
    
    Args:
        kernel_code: The Triton kernel code to check
    
    Returns:
        dict with compatibility status, issues found, and warnings
    """
    issues = []
    warnings = []
    
    # Check for CUDA-only features
    if "tl.libdevice" in kernel_code:
        issues.append("Uses tl.libdevice (CUDA-only, will fail on ROCm)")
    
    if "cuda" in kernel_code.lower() and "rocm" not in kernel_code.lower():
        warnings.append("Contains cuda references - verify AMD compatibility")
    
    # Check for problematic patterns
    if "num_warps" in kernel_code:
        import re
        warps_match = re.search(r"num_warps\s*[=:]\s*(\d+)", kernel_code)
        if warps_match:
            num_warps = int(warps_match.group(1))
            if num_warps > 16:
                issues.append(f"num_warps={num_warps} exceeds AMD maximum of 16")
    
    # Check block sizes
    import re
    block_matches = re.findall(r"BLOCK[_A-Z]*\s*[=:]\s*(\d+)", kernel_code)
    for block_size in block_matches:
        size = int(block_size)
        if size > 1024:
            issues.append(f"Block size {size} exceeds AMD maximum of 1024")
        elif size % 64 != 0:
            warnings.append(f"Block size {size} not aligned to wavefront (64)")
    
    # Check for program_id usage
    if "tl.program_id(1)" in kernel_code or "tl.program_id(2)" in kernel_code:
        if "grid = (" in kernel_code:
            grid_match = re.search(r"grid\s*=\s*\(([^)]+)\)", kernel_code)
            if grid_match:
                grid_dims = grid_match.group(1).count(",") + 1
                if "tl.program_id(1)" in kernel_code and grid_dims < 2:
                    issues.append("Uses program_id(1) but grid appears to be 1D")
    
    compatible = len(issues) == 0
    
    return {
        "compatible": compatible,
        "issues": issues,
        "warnings": warnings,
        "recommendation": "Ready for testing" if compatible else "Fix issues before testing"
    }


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
