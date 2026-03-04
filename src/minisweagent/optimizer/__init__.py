# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.

"""
GEAK Optimizer - Compact kernel optimization framework.

Wraps existing optimizers (OpenEvolve, etc.) with unified interface.
"""

from .core import OptimizerType, optimize_kernel

__all__ = ["optimize_kernel", "OptimizerType"]
