---
name: geak-agent-skills
description: Optimize AMD GPU HIP kernels using the GEAK (GPU Enhancement Agent for Kernels) REST API. Provides a CLI tool to configure models, submit HIP kernel files or git repos for optimization, poll task status, and download results. Use when the user mentions GEAK, HIP kernel optimization, GPU kernel performance, or wants to optimize .hip files for AMD GPUs.
---

# GEAK — GPU Kernel Optimization

GEAK (GPU Enhancement Agent for Kernels) is an AI-powered service that optimizes HIP kernels for AMD GPUs. It accepts `.hip` files or git repos, runs an AI agent on GPU to analyze and optimize kernels, then returns improved code.

## Prerequisites

```bash
pip install requests python-dotenv
```

Set environment variables (or create `.env`):

```bash
export GEAK_API_URL="https://your-geak-server.com"
export GEAK_API_KEY="ak-your-api-key"
```

## Quick Start Workflow

Copy `scripts/geak.py` from this skill directory to the project, then follow this workflow:

```
Task Progress:
- [ ] Step 1: Configure LLM model (one-time)
- [ ] Step 2: Submit kernel for optimization
- [ ] Step 3: Wait for completion
- [ ] Step 4: Download and review results
```

### Step 1: Configure Model (One-Time)

```bash
python geak.py config \
    --model_name "openai/claude-opus-4.5" \
    --api_base "http://litellm-service:4000/v1" \
    --api_key "sk-xxx"
```

### Step 2: Optimize

**Single or multiple files:**

```bash
# Single file
python geak.py optimize silu.hip

# Multiple files (kernel + Makefile + headers)
python geak.py optimize silu.hip Makefile kernels.h --step_limit 20

# With custom prompt and GPU count
python geak.py optimize silu.hip \
    --prompt "Optimize for MI300X, focus on memory coalescing" \
    --step_limit 20 --gpu_count 2
```

**Git repository:**

```bash
python geak.py optimize-repo https://github.com/org/hip-kernels.git \
    --branch main \
    --prompt "Optimize all HIP kernels for MI300X. Run tests to verify." \
    --step_limit 30
```

The `optimize` and `optimize-repo` commands automatically create, submit, poll, and download results.

### Step 3: Check Status (if needed)

```bash
python geak.py status <task_id>
```

### Step 4: Download Results (if interrupted)

```bash
python geak.py results <task_id> --output_dir ./my_results
```

## Task Management

```bash
# List all tasks
python geak.py list

# List by status
python geak.py list --status running

# Cancel a task
python geak.py cancel <task_id>
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--step_limit` | 10 | Max agent optimization steps |
| `--gpu_count` | 1 | Number of GPUs for optimization |
| `--prompt` | (none) | Custom optimization instructions |
| `--output_dir` | `geak_output_<id>` | Where to save results |
| `--branch` | main | Git branch (repo mode only) |

## Alternative: MCP Mode

For Cursor IDE users, connect via MCP instead of the CLI script. Add to `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "geak": {
      "url": "https://your-geak-server.com/mcp",
      "headers": {
        "Authorization": "Bearer ak-your-api-key"
      }
    }
  }
}
```

Then use GEAK tools directly in Cursor chat — no script needed.

## Utility Scripts

**scripts/geak.py**: Full CLI tool for GEAK API interaction.

```bash
python scripts/geak.py --help
```

Execute this script directly. It handles all API communication, polling, and file downloads.

## Additional Resources

- For complete API endpoint details, see [api-reference.md](api-reference.md)
