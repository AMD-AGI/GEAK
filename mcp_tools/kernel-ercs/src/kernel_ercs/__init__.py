"""Kernel ERCS MCP Server.

ERCS = Evaluation, Reflection, Compatibility, Specs

A Model Context Protocol (MCP) server that provides LLM-based kernel optimization
tools for Triton/AMD GPU kernels.

Tools:
- evaluate_kernel_quality: LLM scores kernel on 9 criteria
- reflect_on_kernel_result: LLM analyzes test results, suggests next steps
- check_kernel_compatibility: Quick AMD compatibility check
- get_amd_gpu_specs: AMD MI350X hardware reference
"""

__version__ = "0.1.0"
