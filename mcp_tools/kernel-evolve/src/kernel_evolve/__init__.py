# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
# SPDX-License-Identifier: Apache-2.0

"""Kernel Evolve MCP Server.

Evolutionary GPU kernel optimization using LLM-guided mutation and crossover.

Tools:
- generate_optimization: LLM generates optimized kernel variant
- mutate_kernel: LLM mutates an existing optimization
- crossover_kernels: LLM combines two kernel optimizations
- get_optimization_strategies: Get strategies for a bottleneck type
"""

__version__ = "0.1.0"
