#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""$KT/hw_context.py: the run hardware context CLI, reachable beside the tools that read it.

Runs <GEAK root>/scripts/hardware_context.py (the repository root's scripts/, not
kernel_workflow/scripts/) with the same arguments. Once per run, no GPU work:

  python3 $KT/hw_context.py --output $EVAL_DIR/hardware_context.json

then pass --hardware-context $EVAL_DIR/hardware_context.json to hw_budget.py, mem_bw_probe.py and
extract_sku.py --facts (or export GEAK_HARDWARE_CONTEXT=$EVAL_DIR/hardware_context.json).
"""
import os
import runpy
import sys

TARGET = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", "..", "..", "scripts", "hardware_context.py"))

if __name__ == "__main__":
    if not os.path.isfile(TARGET):
        sys.exit(f"hw_context.py: {TARGET} is missing from this checkout")
    sys.argv[0] = TARGET
    runpy.run_path(TARGET, run_name="__main__")
