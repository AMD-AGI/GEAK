"""Kernel Evolve MCP Server.

Evolutionary GPU kernel optimization using LLM-guided mutation and crossover.

Tools:
- generate_optimization: LLM generates optimized kernel variant
- mutate_kernel: LLM mutates an existing optimization
- crossover_kernels: LLM combines two kernel optimizations
- get_optimization_strategies: Get strategies for a bottleneck type
"""

__version__ = "0.1.0"
