"""
GEAK Optimizer - Compact kernel optimization framework.

Wraps existing optimizers (OpenEvolve, etc.) with unified interface.
"""

from .core import OptimizerType, optimize_kernel

__all__ = ["optimize_kernel", "OptimizerType"]
