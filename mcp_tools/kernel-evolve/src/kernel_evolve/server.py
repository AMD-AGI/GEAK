"""Kernel Evolve MCP Server.

Evolutionary GPU kernel optimization using LLM-guided mutation and crossover.
This is the OpenEvolve brain as an MCP server.

Tools:
- generate_optimization: LLM generates optimized kernel variant
- mutate_kernel: LLM mutates an existing optimization  
- crossover_kernels: LLM combines two kernel optimizations
- get_optimization_strategies: Get strategies for a bottleneck type
"""

import json
import os
from typing import Any

from fastmcp import FastMCP

# Try to import anthropic for AMD gateway
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Try to import litellm
try:
    from litellm import completion
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False


# Create the MCP server
mcp = FastMCP(
    name="kernel-evolve",
    instructions="Evolutionary GPU kernel optimization using LLM-guided mutation and crossover"
)


def call_llm(messages: list, model: str = "amd/claude-sonnet-4-20250514", temperature: float = 0.7) -> str:
    """Call LLM using AMD gateway or litellm."""
    api_key = os.environ.get("AMD_LLM_API_KEY") or os.environ.get("LLM_GATEWAY_KEY")
    
    if ANTHROPIC_AVAILABLE and api_key:
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
        
        model_name = model.removeprefix("amd/")
        
        system_content = ""
        filtered_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_content = msg.get("content", "")
            else:
                filtered_messages.append(msg)
        
        response = client.messages.create(
            model=model_name,
            max_tokens=8192,
            system=system_content if system_content else anthropic.NOT_GIVEN,
            messages=filtered_messages,
            temperature=temperature,
        )
        
        return response.content[0].text
    
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


# Optimization strategies by bottleneck type
BOTTLENECK_STRATEGIES = {
    "latency": [
        "hip_graph",
        "kernel_fusion", 
        "multi_batch",
        "persistent_kernel",
        "pytorch_replacement"
    ],
    "memory": [
        "vectorized_loads",
        "memory_coalescing",
        "lds_caching",
        "async_prefetch",
        "reduce_traffic"
    ],
    "compute": [
        "mfma_instructions",
        "vectorize_ops",
        "loop_unroll",
        "algorithmic_optimization"
    ],
    "lds": [
        "bank_conflict_padding",
        "warp_shuffle",
        "layout_reorganization"
    ],
    "balanced": [
        "block_size_tuning",
        "num_warps_tuning",
        "general_optimization"
    ]
}


GENERATE_PROMPT = """You are an expert GPU kernel optimization engineer for AMD MI350X GPUs.

Generate an optimized version of this Triton kernel that addresses the identified bottleneck.

## Original Kernel:
```python
{kernel_code}
```

## Bottleneck: {bottleneck}

## Strategy to apply: {strategy}

## AMD GPU Guidelines:
- AMD wavefront = 64 threads (not 32 like NVIDIA)
- Block sizes should be multiples of 64, max 1024
- num_warps range: [1-16] only
- DO NOT use CUDA-only features like tl.libdevice
- Use tl.load/tl.store with masks for boundary handling

Generate a complete, optimized kernel. Output ONLY the Python code, no explanations.
The code should be runnable and include:
1. All necessary imports
2. The optimized @triton.jit kernel
3. A wrapper function named `triton_op` or `run_baseline`

```python
# Your optimized code here
```
"""


MUTATE_PROMPT = """You are an expert GPU kernel optimization engineer for AMD MI350X GPUs.

Mutate this kernel optimization to explore a variation that might perform better.

## Current Kernel:
```python
{kernel_code}
```

## Current Performance:
- Latency: {latency_us} μs
- Speedup vs baseline: {speedup}x

## Mutation Type: {mutation_type}
{mutation_instruction}

## AMD GPU Guidelines:
- AMD wavefront = 64 threads
- Block sizes: multiples of 64, max 1024
- num_warps: [1-16] only
- Avoid tl.libdevice

Make ONE significant change. Output ONLY the complete Python code.

```python
# Your mutated code here
```
"""


CROSSOVER_PROMPT = """You are an expert GPU kernel optimization engineer for AMD MI350X GPUs.

Combine the best aspects of these two kernel optimizations to create a hybrid.

## Parent 1 (speedup: {speedup1}x):
```python
{kernel1}
```

## Parent 2 (speedup: {speedup2}x):
```python
{kernel2}
```

## Instructions:
- Take the most effective technique from each parent
- Ensure they work together correctly
- The hybrid should potentially outperform both parents

## AMD GPU Guidelines:
- AMD wavefront = 64 threads
- Block sizes: multiples of 64, max 1024
- num_warps: [1-16] only

Output ONLY the complete hybrid Python code.

```python
# Your hybrid code here
```
"""


def extract_code(response: str) -> str:
    """Extract Python code from LLM response."""
    if "```python" in response:
        code = response.split("```python")[1].split("```")[0]
    elif "```" in response:
        code = response.split("```")[1].split("```")[0]
    else:
        code = response
    return code.strip()


@mcp.tool()
def generate_optimization(
    kernel_code: str,
    bottleneck: str = "balanced",
    strategy: str = None,
    model: str = "amd/claude-sonnet-4-20250514"
) -> dict[str, Any]:
    """
    Generate an optimized kernel variant using LLM.
    
    The LLM analyzes the kernel and bottleneck to generate an optimization
    that targets the specific performance issue.
    
    Args:
        kernel_code: The original Triton kernel code
        bottleneck: Bottleneck type (latency, memory, compute, lds, balanced)
        strategy: Specific strategy to apply (optional, will auto-select if not provided)
        model: LLM model to use
    
    Returns:
        dict with optimized_code, strategy_used, and metadata
    """
    # Select strategy if not provided
    if not strategy:
        strategies = BOTTLENECK_STRATEGIES.get(bottleneck.lower(), BOTTLENECK_STRATEGIES["balanced"])
        strategy = strategies[0]  # Use first strategy as default
    
    prompt = GENERATE_PROMPT.format(
        kernel_code=kernel_code,
        bottleneck=bottleneck,
        strategy=strategy
    )
    
    messages = [{"role": "user", "content": prompt}]
    
    try:
        response = call_llm(messages, model, temperature=0.7)
        optimized_code = extract_code(response)
        
        return {
            "success": True,
            "optimized_code": optimized_code,
            "strategy_used": strategy,
            "bottleneck": bottleneck,
            "model": model
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "strategy_used": strategy,
            "bottleneck": bottleneck
        }


@mcp.tool()
def mutate_kernel(
    kernel_code: str,
    mutation_type: str = "parameter",
    latency_us: float = 0.0,
    speedup: float = 1.0,
    model: str = "amd/claude-sonnet-4-20250514"
) -> dict[str, Any]:
    """
    Mutate an existing kernel optimization to explore variations.
    
    This implements the mutation operation in evolutionary optimization.
    The LLM makes intelligent changes based on the mutation type.
    
    Args:
        kernel_code: The current kernel code to mutate
        mutation_type: Type of mutation - "parameter", "algorithm", or "hybrid"
        latency_us: Current latency (for context)
        speedup: Current speedup vs baseline (for context)
        model: LLM model to use
    
    Returns:
        dict with mutated_code, mutation_type, and metadata
    """
    mutation_instructions = {
        "parameter": "Change numerical parameters (block sizes, num_warps, unroll factors)",
        "algorithm": "Try a different algorithmic approach while keeping the same goal",
        "hybrid": "Add an additional optimization technique to the existing approach"
    }
    
    instruction = mutation_instructions.get(mutation_type, mutation_instructions["parameter"])
    
    prompt = MUTATE_PROMPT.format(
        kernel_code=kernel_code,
        latency_us=latency_us,
        speedup=speedup,
        mutation_type=mutation_type,
        mutation_instruction=instruction
    )
    
    messages = [{"role": "user", "content": prompt}]
    
    try:
        response = call_llm(messages, model, temperature=0.8)
        mutated_code = extract_code(response)
        
        return {
            "success": True,
            "mutated_code": mutated_code,
            "mutation_type": mutation_type,
            "parent_latency_us": latency_us,
            "parent_speedup": speedup
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "mutation_type": mutation_type
        }


@mcp.tool()
def crossover_kernels(
    kernel1: str,
    kernel2: str,
    speedup1: float = 1.0,
    speedup2: float = 1.0,
    model: str = "amd/claude-sonnet-4-20250514"
) -> dict[str, Any]:
    """
    Combine two kernel optimizations to create a hybrid.
    
    This implements the crossover operation in evolutionary optimization.
    The LLM intelligently combines the best aspects of both parents.
    
    Args:
        kernel1: First parent kernel code
        kernel2: Second parent kernel code
        speedup1: Speedup of first parent
        speedup2: Speedup of second parent
        model: LLM model to use
    
    Returns:
        dict with hybrid_code and metadata
    """
    prompt = CROSSOVER_PROMPT.format(
        kernel1=kernel1,
        kernel2=kernel2,
        speedup1=speedup1,
        speedup2=speedup2
    )
    
    messages = [{"role": "user", "content": prompt}]
    
    try:
        response = call_llm(messages, model, temperature=0.7)
        hybrid_code = extract_code(response)
        
        return {
            "success": True,
            "hybrid_code": hybrid_code,
            "parent1_speedup": speedup1,
            "parent2_speedup": speedup2
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool()
def get_optimization_strategies(
    bottleneck: str
) -> dict[str, Any]:
    """
    Get recommended optimization strategies for a bottleneck type.
    
    Returns a prioritized list of strategies to try for the given bottleneck.
    
    Args:
        bottleneck: Bottleneck type (latency, memory, compute, lds, balanced)
    
    Returns:
        dict with strategies list and descriptions
    """
    strategy_descriptions = {
        "hip_graph": "Use HIP Graph capture to reduce kernel launch overhead",
        "kernel_fusion": "Fuse multiple kernels into one to reduce launches",
        "multi_batch": "Process multiple batches per kernel to amortize launch cost",
        "persistent_kernel": "Keep kernel resident on GPU across multiple calls",
        "pytorch_replacement": "Replace with optimized PyTorch ops if trivial",
        "vectorized_loads": "Use float4/int4 vectorized loads for better bandwidth",
        "memory_coalescing": "Ensure 128-byte aligned coalesced memory access",
        "lds_caching": "Cache frequently accessed data in LDS (shared memory)",
        "async_prefetch": "Use async copy to prefetch data and hide latency",
        "reduce_traffic": "Recompute values instead of loading to reduce traffic",
        "mfma_instructions": "Use MFMA matrix instructions for matrix operations",
        "vectorize_ops": "Vectorize scalar operations for better throughput",
        "loop_unroll": "Unroll loops for better instruction-level parallelism",
        "algorithmic_optimization": "Use more efficient algorithms to reduce FLOPs",
        "bank_conflict_padding": "Add padding to reduce LDS bank conflicts",
        "warp_shuffle": "Use warp-level shuffles instead of LDS where possible",
        "layout_reorganization": "Reorganize data layout for conflict-free access",
        "block_size_tuning": "Tune block sizes for optimal occupancy",
        "num_warps_tuning": "Tune num_warps for optimal resource usage",
        "general_optimization": "Apply general optimization best practices"
    }
    
    bottleneck_lower = bottleneck.lower()
    strategies = BOTTLENECK_STRATEGIES.get(bottleneck_lower, BOTTLENECK_STRATEGIES["balanced"])
    
    return {
        "bottleneck": bottleneck,
        "strategies": strategies,
        "descriptions": {s: strategy_descriptions.get(s, "No description") for s in strategies},
        "priority_order": "Strategies are listed in recommended order of priority"
    }


@mcp.tool()
def suggest_kernel_params(
    kernel_type: str = "elementwise",
    problem_size: int = 1048576
) -> dict[str, Any]:
    """
    Suggest kernel parameters for a given kernel type and problem size.
    
    Returns recommended block sizes, num_warps, and num_stages for AMD GPUs.
    
    Args:
        kernel_type: Type of kernel - "elementwise", "reduction", "matmul", "attention"
        problem_size: Size of the problem (e.g., number of elements)
    
    Returns:
        dict with recommended parameters
    """
    params = {
        "elementwise": {
            "block_size": 1024 if problem_size > 65536 else 256,
            "num_warps": 8,
            "num_stages": 1,
            "notes": "Large blocks for simple ops, single stage since no GEMM"
        },
        "reduction": {
            "block_size": 512,
            "num_warps": 8,
            "num_stages": 1,
            "notes": "Balance between parallelism and reduction efficiency"
        },
        "matmul": {
            "block_m": 128,
            "block_n": 128,
            "block_k": 32,
            "num_warps": 8,
            "num_stages": 2,
            "notes": "Standard GEMM blocking, 2 stages for software pipelining"
        },
        "attention": {
            "block_m": 64,
            "block_n": 64,
            "num_warps": 4,
            "num_stages": 1,
            "notes": "Smaller blocks for attention due to softmax, single stage for fused"
        }
    }
    
    kernel_lower = kernel_type.lower()
    if kernel_lower in params:
        result = params[kernel_lower].copy()
        result["kernel_type"] = kernel_type
        result["problem_size"] = problem_size
        return result
    else:
        return {
            "error": f"Unknown kernel type: {kernel_type}",
            "valid_types": list(params.keys())
        }


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
