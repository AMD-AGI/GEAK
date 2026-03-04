# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.

#!/usr/bin/env python3
"""
Example script demonstrating the RAG filter sub-agent usage.

This shows two ways to use the sub-agent:
1. Standalone usage (requires API key for full test)
2. Integrated in MCP environment
3. Disabled mode (pass-through)

Usage:
    python test_subagent.py              # Basic tests (no API calls)
    python test_subagent.py --full       # Full tests (requires API key)
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from minisweagent.utils.subagent import create_rag_filter_subagent


def test_standalone_subagent(run_api_test: bool = False):
    """Test the sub-agent in standalone mode."""
    print("=" * 80)
    print("Test 1: Standalone RAG Filter Sub-Agent")
    print("=" * 80)
    
    if not run_api_test:
        print("\n⚠️  Skipping API test (use --full to enable)")
        print("   This test requires a valid API key and makes actual LLM calls")
        print("\n✅ Sub-agent module loaded successfully")
        print("\n" + "=" * 80 + "\n")
        return
    
    # Create sub-agent
    subagent = create_rag_filter_subagent(
        model_name="claude-opus-4.5",
        api_key=os.getenv("AMD_LLM_API_KEY"),  # Set in .env or environment
        enabled=True,
    )
    
    # Sample RAG result (simulated)
    sample_rag_result = """
    Chunk 1: HIP (Heterogeneous-compute Interface for Portability) is AMD's GPU programming interface.
    It provides a C++ runtime API and kernel language for parallel computing.
    
    Chunk 2: To optimize HIP kernels, use __launch_bounds__ to control register usage.
    This helps achieve better occupancy on AMD GPUs.
    
    Chunk 3: CUDA is NVIDIA's proprietary GPU programming interface.
    
    Chunk 4: hipMemcpy is used for copying data between host and device memory.
    Always check return codes for error handling.
    
    Chunk 5: For matrix multiplication, use rocBLAS library for optimal performance.
    """
    
    query = "How to optimize HIP kernels?"
    
    print(f"\nQuery: {query}")
    print(f"\nRAW RAG Result ({len(sample_rag_result)} chars):")
    print(sample_rag_result[:200] + "...")
    
    # Process through sub-agent
    filtered_result = subagent.process(sample_rag_result, query=query)
    
    print(f"\nFiltered Result ({len(filtered_result)} chars):")
    print(filtered_result)
    print("\n" + "=" * 80 + "\n")


def test_mcp_environment_integration():
    """Test the sub-agent integrated in MCP environment."""
    print("=" * 80)
    print("Test 2: MCP Environment with Sub-Agent Integration")
    print("=" * 80)
    
    from minisweagent.mcp_integration.mcp_environment import MCPEnabledEnvironment
    
    # Create MCP environment with sub-agent enabled
    env = MCPEnabledEnvironment(
        auto_build_index=False,  # Skip index build for demo
        enable_rag_subagent=True,
        rag_subagent_model="claude-opus-4.5",
        rag_subagent_api_key=os.getenv("RAG_SUBAGENT_API_KEY"),  # Set in .env or environment
    )
    
    print("\n✅ MCP Environment created with RAG sub-agent enabled")
    print(f"   - Sub-agent model: {env.config.rag_subagent_model}")
    print(f"   - Sub-agent enabled: {env.config.enable_rag_subagent}")
    
    # Note: To test actual MCP tool execution with sub-agent processing,
    # you would need to call env.execute("@amd:query {...}") with a real query
    # This would automatically apply the sub-agent to the results
    
    print("\nTo use in real scenario:")
    print('  result = env.execute("@amd:query {\\"topic\\": \\"HIP optimization\\"}")')
    print("  # Result is automatically processed by RAG filter sub-agent")
    print("\n" + "=" * 80 + "\n")


def test_subagent_disabled():
    """Test with sub-agent disabled."""
    print("=" * 80)
    print("Test 3: Sub-Agent Disabled (Pass-through)")
    print("=" * 80)
    
    # Create sub-agent with disabled flag
    subagent = create_rag_filter_subagent(
        enabled=False,
    )
    
    sample_input = "This is some RAG data that should pass through unchanged."
    result = subagent.process(sample_input)
    
    print(f"Input:  {sample_input}")
    print(f"Output: {result}")
    print(f"Pass-through working: {sample_input == result}")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    print("\n🚀 RAG Filter Sub-Agent Test Suite\n")
    
    # Check for --full flag
    run_full_tests = "--full" in sys.argv
    
    if run_full_tests:
        print("🔥 Running FULL test suite (with API calls)\n")
    else:
        print("📋 Running basic test suite (no API calls)")
        print("   Use 'python test_subagent.py --full' for complete tests\n")
    
    # Test 1: Standalone usage
    test_standalone_subagent(run_api_test=run_full_tests)
    
    # Test 2: MCP environment integration
    test_mcp_environment_integration()
    
    # Test 3: Disabled sub-agent
    test_subagent_disabled()
    
    print("✅ All tests completed!")

