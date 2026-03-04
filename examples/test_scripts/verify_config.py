# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.

#!/usr/bin/env python3
"""Quick verification that the MCPEnvironmentConfig works correctly."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from minisweagent.mcp_integration.mcp_environment import MCPEnvironmentConfig, MCPEnabledEnvironment

# Test 1: Config creation
print("Test 1: Creating MCPEnvironmentConfig...")
config = MCPEnvironmentConfig(
    enable_rag_subagent=True,
    rag_subagent_model="claude-opus-4.5",
    rag_subagent_api_key="test-key"
)
print(f"✅ Config created: enable_rag_subagent={config.enable_rag_subagent}")
print(f"   rag_subagent_model={config.rag_subagent_model}")

# Test 2: Environment creation
print("\nTest 2: Creating MCPEnabledEnvironment...")
try:
    env = MCPEnabledEnvironment(
        auto_build_index=False,
        enable_rag_subagent=True,
        rag_subagent_model="claude-opus-4.5",
        rag_subagent_api_key="test-key"
    )
    print(f"✅ Environment created successfully!")
    print(f"   enable_rag_subagent={env.config.enable_rag_subagent}")
    print(f"   rag_subagent_model={env.config.rag_subagent_model}")
    print(f"   Has rag_subagent property: {hasattr(env, 'rag_subagent')}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ All verification tests passed!")

