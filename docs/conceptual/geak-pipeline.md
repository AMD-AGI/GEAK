---
myst:
  html_meta:
    "description": "Learn how the GEAK agent loop works: inputs, parallel optimization, patch evaluation, MCP tools, and output artifacts for GPU kernel tuning on ROCm."
    "keywords": "GEAK, agent loop, pipeline, GPU kernel optimization, ROCm, MCP tools, parallel scaling, patch evaluation, HIP, Triton"
---

# GEAK agent loop

The GEAK pipeline takes a kernel repository and a natural-language prompt as input, runs a parallel agent loop to find the best optimization, and produces a tested, reviewable patch. This topic explains each stage of that loop and the tools available to agents during a run.

```{figure} ../images/GEAK_framework.png
:alt: GEAK agent loop diagram showing inputs (kernel repo and prompt), the parallel agent loop (task plan, optimization, and patch evaluation), MCP tools (knowledge base and kernel profiler), local tools (bash and patch version management), and outputs (best kernel, strategy summary, and updated memory).
:align: center

GEAK agent loop
```

## Inputs

Every GEAK run requires two inputs:

- **Kernel repo**: The Git repository containing the kernel to optimize. GEAK reads the repository structure, locates the target kernel, and builds a codebase context document before any optimization begins.
- **Prompt**: A natural-language task description that tells GEAK what to optimize and what metric to improve (for example, latency, bandwidth, or throughput). The prompt can include the kernel URL, GPU IDs, and a test harness path, or GEAK can discover these automatically during preprocessing.

## The agent loop

Once preprocessing is complete, GEAK enters the agent loop. Multiple agents run in parallel, each in an isolated Git worktree on its own GPU, following the same three-stage cycle.

### Task plan and update

At the start of each cycle the agent reviews the current state: what optimizations have been tried, what the profiler reported, and what the knowledge base suggests. It produces a task plan—a concrete set of code changes to attempt in this iteration—and updates its working memory with observations from prior cycles.

### Optimization

The agent applies its task plan to the kernel source. Optimization strategies vary by kernel type and workload, and include changes such as adjusting tile sizes and block dimensions, rewriting memory access patterns, exploiting hardware-specific instructions, and fusing or splitting operations. Agents consult the AMD knowledge base and kernel profiler during this stage to inform their decisions.

### Evaluate patch

After applying a change, the agent runs the test harness in correctness mode, then in benchmark mode. A patch that fails correctness is discarded. A patch that passes is scored against the baseline metrics captured during preprocessing. The score, along with the strategy used, is recorded in the agent's memory for the next iteration.

### Parallel scaling

GEAK runs multiple agents simultaneously, each exploring a different strategy. This parallel search means a single run covers more of the optimization space than a sequential approach, and failed strategies on one agent don't block others. After each round, GEAK ranks all agents by verified speedup and selects the best patch to carry forward.

## Tools available to agents

Agents have access to two categories of tools during a run.

### MCP tools

MCP (Model Context Protocol) tools run as subprocess servers and are discovered automatically from the `mcp_tools/` directory.

- **Knowledge base**: A curated collection of GPU optimization knowledge covering the ROCm software stack, AMD Instinct architecture, and established optimization patterns. Agents query this through a Retrieval-Augmented Generation (RAG) system, which returns dynamically ranked excerpts relevant to the current kernel and strategy.
- **Kernel profiler**: Integrated ROCm profiling that surfaces hotspots, memory bandwidth utilization, and compute bottlenecks. Agents invoke the profiler to understand where time is being spent before and after applying a change.
- **Cross-session memory**: Insights from past GEAK runs are stored and retrieved across sessions. When GEAK encounters a kernel or workload it has optimized before, relevant past strategies and outcomes are surfaced to agents automatically.

### Local tools

- **Bash command**: Agents execute shell commands to build the kernel, run the harness, inspect files, and interact with the repository.
- **Patch version management**: GEAK tracks every patch applied during a run, allowing agents to compare versions, roll back unsuccessful changes, and build on earlier improvements.

## Outputs

At the end of a run, GEAK produces:

- **Best kernel and speedup**: The winning patch is applied to the repository and committed. The speedup relative to the baseline is reported in `final_report.json`.
- **Strategy summary**: A record of which strategies were explored, which succeeded, and which were discarded. This feeds back into cross-session memory for future runs.
- **Updated memory**: New insights are persisted to the knowledge base so subsequent runs on similar kernels benefit from the current run's findings.

## Related topics

- [What is GEAK?](../what-is-geak.md) — overview of GEAK's capabilities and design goals.
- [Run the agent](../how-to/run-agent.md) — how to invoke `geak` from the command line.
