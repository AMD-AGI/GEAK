# GEAK Agent

GPU Evolutionary Agent for Kernels - A simple AI agent for GPU kernel optimization.

Based on [mini-swe-agent](https://github.com/SWE-agent/mini-SWE-agent) architecture: the LLM generates bash commands and MCP tool calls to optimize GPU kernels.

## Quick Start

**Run inside the container** (enter with `./scripts/run-docker.sh`). The repo is mounted at `/workspace` so you can edit on the host and run in the container without rebuilding.

```bash
# Enter container (builds image if needed; requires AMD_LLM_API_KEY)
./scripts/run-docker.sh

# Inside container (you land in /workspace)
pytest geak_agent/tests/ -v
python geak_agent/examples/resolve_kernel_url.py /workspace/geak_agent/examples/resolve_kernel_url.py
python geak_agent/examples/profile_kernel.py 'python3 /path/to/kernel.py --profile'
```

**Agent run (inside container):**

```bash
python3 -m geakagent.run.mini -m claude-sonnet-4.5 \
  -t "Optimize kernel at /path/to/kernel.py" \
  --yolo
```

See [docs/RUNTIME_QUICKSTART.md](docs/RUNTIME_QUICKSTART.md) and [docs/RUNTIME_ENV.md](docs/RUNTIME_ENV.md) for runtime options (local vs Docker).

---

## Architecture

- **LLM** generates bash commands and MCP tool calls (profile, evolve, discover).
- **Execute → Observe → Repeat.**

MCPs: automated-test-discovery, kernel-profiler, kernel-evolve, kernel-ercs, metrix-mcp, openevolve-mcp (see `mcp_tools/`).

---

## Installation

For local install (optional; container has everything):

```bash
git clone -b msa https://github.com/AMD-AGI/GEAK-agent.git
cd GEAK-agent
pip install -e .
pip install -e mcp_tools/mcp-client/
# See Dockerfile for full MCP list
```

---

## Project Structure

```
GEAK-agent/
├── src/geakagent/       # Main agent
├── mcp_tools/           # MCP servers & client
├── geak_agent/          # Discovery, examples, resolve_kernel_url
├── examples/            # Example kernels
├── scripts/             # run-docker.sh
└── reference/           # optimization_strategies.py, state.py
```

---

## Reference

- `reference/optimization_strategies.py` - GPU optimization strategies
- `reference/state.py` - State management for optimization runs
