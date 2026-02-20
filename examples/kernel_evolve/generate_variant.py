#!/usr/bin/env python3
"""Example: Generate an optimized kernel variant via kernel-evolve MCP.

Usage:
    python examples/kernel_evolve/generate_variant.py

Note: Requires AMD_LLM_API_KEY for the LLM backend.
Without it, the MCP call will fail (shown gracefully).
"""

from minisweagent.tools.mcp_bridge import MCPToolBridge

SAMPLE_KERNEL = '''
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
'''


def main():
    # Create bridge to kernel-evolve MCP
    bridge = MCPToolBridge("kernel-evolve", timeout=120)

    # Get optimization strategies for a memory-bound kernel
    strategies_tool = bridge.tool("get_optimization_strategies")
    result = strategies_tool(bottleneck_type="memory")
    print("=== Strategies for memory-bound kernels ===")
    print(result["output"][:500])

    # Try to generate an optimized variant (requires LLM)
    print("\n=== Generating optimized variant ===")
    generate_tool = bridge.tool("generate_optimization")
    result = generate_tool(
        kernel_code=SAMPLE_KERNEL,
        bottleneck="memory",
        language="triton",
    )
    print(f"Return code: {result['returncode']}")
    print(f"Output: {result['output'][:300]}")


if __name__ == "__main__":
    main()
