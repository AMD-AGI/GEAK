Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

---
tags: ["optimization", "performance", "profiling", "rocprofiler", "analysis"]
priority: "L0-core"
source_url: "https://rocm.docs.amd.com/en/latest/projects/rocprofiler-compute/how-to/analyze/mode.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# Analyze Mode

## Overview

ROCm Compute Profiler provides multiple approaches for engaging with profiling-generated metrics. Your selection of analysis method should consider your familiarity with the profiled application, computing environment, and experience level with the tool.

The documentation describes a spectrum of analysis approaches: "While analyzing with the CLI offers quick and straightforward access to ROCm Compute Profiler metrics from the terminal, Grafana's dashboard GUI adds an extra layer of readability and interactivity you might prefer."

## Available Analysis Methods

The platform supports four distinct analysis approaches:

- **CLI analysis** — Direct terminal-based metric examination
- **Grafana GUI analysis** — Interactive dashboard visualization
- **Standalone GUI analysis** — Independent graphical interface
- **Text-based User Interface (TUI) analysis** — Terminal-based interactive analysis

## Context and Examples

Analysis demonstrations in this documentation section utilize profiling data from the `vcopy.cpp` workload. Performance analysis references the MI200 platform unless otherwise specified.

## Related Documentation

For information about profiling workflows, refer to the Profile mode documentation. Additional context on ROCm Compute Profiler's operational modes is available in the main usage guide.
