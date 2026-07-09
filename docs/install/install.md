---
myst:
    html_meta:
        "description": "Install GEAK: clone the repository, install the ROCm profiler and a serving backend, and launch Claude Code with the dynamic Workflow feature."
        "keywords": "GEAK, install, ROCm, Claude Code, Workflow, sglang, vLLM, AMD Instinct, setup"
---

# Install GEAK

GEAK is driven by **Claude Code** and runs its deterministic **Workflows** directly from the repository.
There is no package to `pip install` — you clone the repo and point Claude Code at it.

## Prerequisites

- An **AMD Instinct MI GPU** (CDNA — e.g. gfx942 / gfx950; auto-detected).
- **ROCm 6+** with a working user-space (so `rocminfo` / `rocm-smi` work) and a profiler
  (`rocprof-compute`, `rocprofv3`, or `rocprof`). For HIP C++ kernels you also need `hipcc`.
- **Python 3.8+**.
- **Claude Code ≥ 2.1.177** — the workflows use the dynamic **Workflow** (JS orchestration) feature,
  available only from this version. Check with `claude --version`.
- For end-to-end serving optimization: a running-capable backend (`sglang` or `vllm`) and the model
  weights on disk.

See the [compatibility matrix](../compatibility.md) for verified versions.

## Get the repository

```bash
git clone https://github.com/AMD-AGI/GEAK.git
cd GEAK
git checkout GEAK_v4
```

## Launch Claude Code

The workflows spawn many sub-agents and run profiling, benchmark, and build commands on the box, so run
Claude Code with permissions auto-approved. Update first to ensure you have the Workflow feature:

```bash
claude update                                    # update Claude Code to the latest version
IS_SANDBOX=1 claude --dangerously-skip-permissions
```

Then describe what you want in natural language — Claude Code resolves the paths and invokes the
`Workflow` tool for you. See [Run a workflow](../how-to/run-agent.md).

## Related topics

- [Compatibility matrix](../compatibility.md) — verified GPUs, ROCm versions, backends, and dtypes.
- [Run a workflow](../how-to/run-agent.md) — start a single-kernel or end-to-end run.
