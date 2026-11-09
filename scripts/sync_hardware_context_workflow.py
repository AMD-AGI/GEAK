#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synchronize the pure hardware contract into sandboxed workflow entrypoints."""
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
start = "// BEGIN GENERATED hardware_context_workflow.js\n"
end = "// END GENERATED hardware_context_workflow.js\n"
block = start + (root / "scripts/hardware_context_workflow.js").read_text() + end
bad = []
for rel in ("kernel_workflow/kernel_lane.js", "kernel_workflow/kernel_workflow.js", "e2e_workflow/e2e_workflow.js"):
    path = root / rel
    src = path.read_text()
    if start in src:
        i, j = src.index(start), src.index(end) + len(end)
        updated = src[:i] + block + src[j:]
    else:
        anchor = "const A = args || {};\n" if "const A = args || {};\n" in src else "const A = _rawArgs || {};\n"
        assert anchor in src, rel
        updated = src.replace(anchor, anchor + block, 1)
    if updated != src:
        bad.append(rel)
        if "--check" not in sys.argv:
            path.write_text(updated)
if "--check" in sys.argv and bad:
    raise SystemExit("Stale workflow hardware contract: " + ", ".join(bad))
