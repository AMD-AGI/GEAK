---
myst:
  html_meta:
    "description": "GEAK is an agent-driven framework for GPU kernel optimization on ROCm. It profiles, optimizes, and validates HIP, Triton, and FlyDSL kernels using LLM-guided multi-agent search."
    "keywords": "GEAK, GPU kernel optimization, ROCm, HIP, Triton, FlyDSL, LLM agent, AMD Instinct, multi-agent, kernel profiling, MCP, RAG"
---

# What is GEAK?

GEAK (Generating Efficient AI-Centric Kernels) is an agent-driven framework for end-to-end GPU kernel optimization in real codebases. Given a kernel and a test harness, GEAK runs a closed loop of profiling, LLM-guided optimization, and validation, then produces a reviewable patch. It supports HIP, Triton, and FlyDSL kernels and integrates directly with your repository, so no one-off scripts are required.

## Core capabilities

GEAK has three core capabilities: multi-agent search, repository-level workflows, and a curated knowledge base.

### Multi-agent search

GEAK distributes optimization work across multiple agents, each running in an isolated Git worktree and bound to a dedicated GPU. This lets GEAK explore different optimization strategies in parallel rather than sequentially, improving both coverage and robustness.

Each agent runs the same closed loop independently:

1. Profile the kernel to identify bottlenecks.
2. Generate and apply an optimization patch.
3. Validate correctness and measure performance against the baseline.

Agents that produce a verified improvement are ranked, and the best patch is surfaced for review.

GEAK also automatically discovers or generates the test harness it needs. If you provide a harness, GEAK uses it. If you don't, GEAK generates one before optimization begins.

### Repository-level workflows

GEAK works at the repository level, not just on individual files. It can analyze a full repository to understand dependencies, build systems, and surrounding context before optimizing a target kernel. This means optimizations stay consistent with the broader codebase. The evaluation pipeline — baseline capture, correctness checks, and benchmarking — runs the same way every time.

### Knowledge base and tools

GEAK includes a curated AMD knowledge base covering the ROCm software stack, hardware architecture, and established optimization patterns. This knowledge base is exposed to agents through a Model Context Protocol (MCP) based Retrieval-Augmented Generation (RAG) system, so agents can query it dynamically during a run rather than relying on static context.

Additional capabilities include:

- **Kernel profiler**: Integrated profiling via the ROCm toolchain to identify hotspots and memory bottlenecks before optimization begins.
- **In-session memory**: Agents accumulate observations within a single run to avoid repeating failed strategies.
- **Cross-session memory**: Insights from past runs are persisted and retrieved in future runs, so GEAK improves over time on similar kernels and workloads.

## How GEAK works

A GEAK run follows this sequence:

1. **Preprocess**—GEAK resolves the kernel source, builds codebase context, generates or validates a test harness, captures a baseline, and profiles the kernel. The output is a `COMMANDMENT.md` contract that specifies the optimization target and evaluation rules for all agents.
2. **Optimize** — One or more agents receive the contract and begin the profiling, patch, and validation loop. Agents run in parallel on isolated Git worktrees, each on its own GPU.
3. **Select**—After each round, GEAK ranks agents by verified speedup, selects the best patch, and optionally runs additional rounds.
4. **Output**—The winning patch is applied to the repository and committed. A `final_report.json` captures the result.

## Related topics

- [Install GEAK](install/install.md)—set up GEAK and configure a model backend.
- [Run the agent](how-to/run-agent.md)—invoke `geak` from the command line with single-agent and parallel examples.
- [API reference](reference/api-reference.md)—CLI flags, environment variables, and run artifact layout.
