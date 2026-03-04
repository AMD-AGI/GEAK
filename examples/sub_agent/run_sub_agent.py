# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.

#!/usr/bin/env python3
"""Example: Spawn a child agent for a focused sub-task.

Usage:
    python examples/sub_agent/run_sub_agent.py

Note: Requires AMD_LLM_API_KEY to be set for the LLM model.
This example shows the API; actual execution needs a running model.
"""

from minisweagent.tools.sub_agent_tool import SubAgentTool


def main():
    # SubAgentTool needs a model and environment to work
    tool = SubAgentTool()

    # Without context, it returns an error gracefully
    result = tool(task="Analyze this kernel for memory coalescing issues")
    print(f"Without context: {result}")

    # In production, the agent sets context:
    # tool.set_context(model=my_model, env=my_env)
    # result = tool(
    #     task="Rewrite the kernel to use shared memory tiling",
    #     step_limit=10,
    #     cost_limit=1.0,
    # )
    print("\nTo use with a real model, call tool.set_context(model, env) first.")


if __name__ == "__main__":
    main()
