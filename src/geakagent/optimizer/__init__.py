"""
GEAK Optimizer - Compact kernel optimization framework.

Wraps existing optimizers (OpenEvolve, etc.) with unified interface.
"""

from .core import optimize_kernel, OptimizerType

__all__ = ["optimize_kernel", "OptimizerType"]
