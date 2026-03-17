# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.

"""
GEAK-Agent Tutorial Utilities

Provides self-contained utilities for running the GEAK-Agent tutorial.
No external tb_eval dependency required.

Usage:
    from utils import setup_environment
    TUTORIAL_DIR, SRC_DIR, CORPUS_PATH, TutorialDataloader = setup_environment()
"""

from .display import (
    setup_environment,
    print_header,
    print_section,
    print_config,
    display_kernel_info,
    load_results,
    display_results_summary,
    display_generated_code,
    display_strategy,
)

__all__ = [
    "setup_environment",
    "print_header",
    "print_section",
    "print_config",
    "display_kernel_info",
    "load_results",
    "display_results_summary",
    "display_generated_code",
    "display_strategy",
]
