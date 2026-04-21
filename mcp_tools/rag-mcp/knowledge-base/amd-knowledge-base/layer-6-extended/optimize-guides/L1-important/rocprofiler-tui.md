---
tags: ["optimization", "performance", "profiling", "rocprofiler", "tui", "analysis"]
priority: "L1-important"
source_url: "https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/how-to/analyze/tui.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# Text-based User Interface (TUI) Analysis

## Overview

ROCm Compute Profiler's analyze mode provides a lightweight Text-based User Interface offering an interactive terminal experience. This feature serves as a more visually engaging alternative to standard CLI analysis, delivering "real-time visual feedback, keyboard shortcuts for common actions, and improved readability with formatted output."

**Note:** TUI is currently in early access. While functional, minor issues may occur, and production workload use is not recommended.

## Launch the TUI Analyzer

1. Execute the analyze command with the `--tui` flag:
   ```
   rocprof-compute analyze --tui
   ```

2. Use the dropdown menu at the top left to select a workload from `rocprof-compute profile` output directories.

3. The center window displays kernel selection options at the top. The first kernel loads by default; select others to view their analysis results.

4. Once results load, expand collapsed sections to view tables, charts, and graphs. Navigate using keyboard shortcuts.

## TUI Analysis Structure

The interface organizes analysis into four hierarchical levels:

1. **Kernel Selection Header with Top Stats** — Enables interactive switching between kernels to examine individual results.

2. **High Level Analysis** — Presents experimental performance metrics reorganized into GPU Speed-of-Light, Compute Throughput, and Memory Throughput sections.

3. **Detailed Block Analysis** — Groups results by metric blocks (similar to CLI output), displaying performance metrics as charts when applicable.

4. **Source Level Analysis** — Shows PC Sampling section. PC sampling requires manual enablement; see the PC sampling guide for setup details.

Follow this top-down structure for thorough performance analysis, beginning with broad overviews and progressively examining specific details.

## Current Limitations

- **PC Sampling:** Not enabled by default; requires manual configuration
- **Filtering:** Advanced kernel and dispatch filtering options are unavailable in current releases
