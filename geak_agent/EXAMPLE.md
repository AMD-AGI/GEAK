# geak_agent Usage Example

## Basic Usage

```bash
# Discover tests and benchmarks
python -m geak_agent.cli /path/to/kernel.py --discover-only

# Run with explicit commands
python -m geak_agent.cli /path/to/kernel.py \
  --test "pytest test.py" \
  --bench "python bench.py"
```

## Docker

```bash
# Build and run (auto-builds if needed, execs into running container)
./scripts/run-docker.sh

# Inside container
python3 -m geak_agent.cli ~/path/to/kernel.py
```

## Python API

```python
# Metrix profiling
from geak_agent.mcp_tools.metrix import MetrixTool

tool = MetrixTool(gpu_devices="0")
result = tool.profile_kernel(
    command="python3 kernel.py",
    quick=True
)

# Discovery
from geak_agent.mcp_tools.discovery import discover

result = discover(workspace="/path/to/project")
print(f"Found {len(result.tests)} tests")
```
