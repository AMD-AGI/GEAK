---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Runtime Environment Configuration

GEAK Agent now supports automatic runtime environment detection and configuration for GPU kernel operations.

## Overview

The runtime environment system:
1. **Detects** local GPU and dependencies (PyTorch, Triton)
2. **Prompts** for Docker if local environment is incomplete
3. **Defaults** to `lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x` Docker image
4. **Manages** Docker containers automatically

---

## Quick Start

### Auto-Detection (Recommended)

The agent will automatically detect your environment and prompt if needed:

```bash
# Run with auto-detection
python3 -m minisweagent.run.mini \
  -m claude-sonnet-4.5 \
  -t "Optimize my kernel" \
  --yolo

# Auto-detection with YOLO mode (no prompts)
python3 -m minisweagent.run.mini \
  -m claude-sonnet-4.5 \
  -t "Optimize my kernel" \
  --yolo  # Will auto-use Docker if local env incomplete
```

### Force Docker

```bash
python3 -m minisweagent.run.mini \
  -m claude-sonnet-4.5 \
  -t "Optimize my kernel" \
  --runtime docker \
  --workspace /path/to/kernels
```

### Force Local

```bash
python3 -m minisweagent.run.mini \
  -m claude-sonnet-4.5 \
  -t "Optimize my kernel" \
  --runtime local
```

### Custom Docker Image

```bash
python3 -m minisweagent.run.mini \
  -m claude-sonnet-4.5 \
  -t "Optimize my kernel" \
  --runtime docker \
  --docker-image rocm/pytorch:latest \
  --workspace /home/user/my_kernels
```

---

## Command-Line Options

### For `minisweagent.run.mini` (Main Agent)

| Option | Description | Default |
|--------|-------------|---------|
| `--runtime TYPE` | Runtime type: `local`, `docker`, or `auto` | `auto` |
| `--docker-image IMG` | Docker image to use | `lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x` |
| `--workspace PATH` | Workspace directory to mount in Docker | Current directory |
| `--no-runtime-check` | Skip runtime environment detection | `false` |

### For `minisweagent.run.mini` (Discovery Pipeline)

Same options as above, plus:

| Option | Description | Default |
|--------|-------------|---------|
| `-r, --runtime TYPE` | Runtime type (short form) | `auto` |

---

## How It Works

### 1. Detection Phase

The agent checks:
- ✅ **GPU availability** - ROCm (`/opt/rocm`) or CUDA (`nvidia-smi`)
- ✅ **PyTorch** - `import torch` and `torch.cuda.is_available()`
- ✅ **Triton** - `import triton`

### 2. Decision Logic

```
Local complete (GPU + torch + triton)?
  ├─ YES → Use local environment ✅
  └─ NO  → Check Docker available?
            ├─ YES → Offer Docker (default) or continue with local
            └─ NO  → Continue with local (limited functionality)
```

### 3. Docker Configuration

When using Docker, the agent automatically:
- Mounts workspace directory as `/workspace`
- Forwards GPU devices (`/dev/kfd`, `/dev/dri`)
- Sets up environment variables
- Manages container lifecycle (start/stop)

---

## Examples

### Example 1: Complete Local Environment

```bash
$ python3 -m minisweagent.run.mini -m claude-sonnet-4.5 -t "Test kernel" --yolo

============================================================
Runtime Environment Detection
============================================================

✅ Current Environment Status:
  • GPU Available: ✅ Yes
  • PyTorch Installed: ✅ Yes
  • Triton Installed: ✅ Yes

✅ Local environment is ready!

┌─ Runtime Environment ──────────────────────────────────┐
│ Runtime: Local                                         │
│ GPU: ✅ Available                                      │
│ PyTorch: ✅ Installed                                  │
│ Triton: ✅ Installed                                   │
│ Status: ✅ Ready                                       │
└────────────────────────────────────────────────────────┘
```

### Example 2: Incomplete Local → Docker Offered

```bash
$ python3 -m minisweagent.run.mini -m claude-sonnet-4.5 -t "Test kernel"

============================================================
Runtime Environment Detection
============================================================

⚠️ Current Environment Status:
  • GPU Available: ❌ No
  • PyTorch Installed: ❌ No
  • Triton Installed: ❌ No

⚠️  Local environment incomplete

Docker is available. Options:
  1. Use Docker (default image: lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x)
  2. Use Docker (custom image)
  3. Continue with local environment (limited functionality)

Select option [1/2/3] (default: 1): 1

✅ Using Docker: lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x

┌─ Runtime Environment ──────────────────────────────────┐
│ Runtime: Docker                                        │
│ Image: lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x    │
│ GPU Devices: /dev/kfd, /dev/dri                       │
│ Status: ✅ Ready for GPU operations                   │
└────────────────────────────────────────────────────────┘
```

### Example 3: YOLO Mode (Auto-Select Docker)

```bash
$ python3 -m minisweagent.run.mini -m claude-sonnet-4.5 -t "Test kernel" --yolo

============================================================
Runtime Environment Detection
============================================================

⚠️ Current Environment Status:
  • GPU Available: ❌ No
  • PyTorch Installed: ❌ No
  • Triton Installed: ❌ No

Using default Docker image: lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x

┌─ Runtime Environment ──────────────────────────────────┐
│ Runtime: Docker                                        │
│ Image: lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x    │
│ GPU Devices: /dev/kfd, /dev/dri                       │
│ Status: ✅ Ready for GPU operations                   │
└────────────────────────────────────────────────────────┘
```

### Example 4: Force Specific Docker Image

```bash
$ python3 -m minisweagent.run.mini \
    -m claude-sonnet-4.5 \
    -t "Optimize kernel" \
    --runtime docker \
    --docker-image rocm/pytorch:rocm6.0_ubuntu22.04_py3.10_pytorch_2.1.1 \
    --workspace /home/user/my_kernels \
    --yolo

┌─ Runtime Environment ──────────────────────────────────┐
│ Runtime: Docker                                        │
│ Image: rocm/pytorch:rocm6.0_ubuntu22.04_py3.10_pyt... │
│ GPU Devices: /dev/kfd, /dev/dri                       │
│ Status: ✅ Ready for GPU operations                   │
└────────────────────────────────────────────────────────┘

Workspace mounted: /home/user/my_kernels → /workspace
```

---

## Docker Images

### Default Image

**Image:** `lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x`

**Includes:**
- ROCm 7.0.0
- PyTorch with ROCm support
- Triton compiler
- Python 3.10+

### Alternative Images

| Image | Use Case |
|-------|----------|
| `rocm/pytorch:latest` | Latest ROCm + PyTorch |
| `rocm/pytorch:rocm6.0_ubuntu22.04_py3.10_pytorch_2.1.1` | Specific ROCm 6.0 |
| `nvcr.io/nvidia/pytorch:23.12-py3` | NVIDIA GPUs (CUDA) |

---

## Programmatic Usage

### In Python Code

```python
from minisweagent.runtime_env import (
    prompt_runtime_environment,
    get_runtime_config_for_agent,
    display_runtime_info,
)

# Auto-detect and prompt
runtime_env = prompt_runtime_environment(auto_confirm=False)

# Display info
display_runtime_info(runtime_env)

# Get agent config
config = get_runtime_config_for_agent(
    runtime_env,
    workspace_path="/path/to/workspace"
)

# Use with agent
if runtime_env.runtime_type == RuntimeType.DOCKER:
    from minisweagent.environments.docker import DockerEnvironment
    env = DockerEnvironment(**config)
else:
    from minisweagent.environments.local import LocalEnvironment
    env = LocalEnvironment(**config)
```

### Force Specific Runtime

```python
from minisweagent.runtime_env import RuntimeEnvironment, RuntimeType

# Force local
runtime_env = RuntimeEnvironment(runtime_type=RuntimeType.LOCAL)

# Force Docker
runtime_env = RuntimeEnvironment(
    runtime_type=RuntimeType.DOCKER,
    docker_image="rocm/pytorch:latest",
    docker_devices=["/dev/kfd", "/dev/dri"],
    has_gpu=True,
    has_triton=True,
    has_torch=True
)
```

---

## Environment Variables

### Existing (from mini-swe-agent)

- `MSWEA_DOCKER_EXECUTABLE` - Docker command (default: `docker`)
- `MSWEA_MINI_CONFIG_PATH` - Config file path
- `MSWEA_VISUAL_MODE_DEFAULT` - Visual mode default

### New (for GEAK)

- `GEAK_DEFAULT_DOCKER_IMAGE` - Override default Docker image (future)
- `GEAK_RUNTIME` - Force runtime type: `local` or `docker` (future)

---

## Troubleshooting

### Issue: Docker Not Found

**Error:**
```
❌ Docker not available!
```

**Solutions:**
1. Install Docker: `sudo apt install docker.io` (Ubuntu)
2. Start Docker: `sudo systemctl start docker`
3. Add user to docker group: `sudo usermod -aG docker $USER` (logout/login)

### Issue: GPU Not Accessible in Docker

**Error:**
```
RuntimeError: No HIP GPUs are available
```

**Solutions:**
1. Check GPU devices exist: `ls -la /dev/kfd /dev/dri`
2. Add devices explicitly:
   ```bash
   --runtime docker \
   --docker-image rocm/pytorch:latest
   ```
3. Verify ROCm installation: `rocminfo`

### Issue: Permission Denied for Docker

**Error:**
```
permission denied while trying to connect to Docker daemon
```

**Solutions:**
1. Add user to docker group: `sudo usermod -aG docker $USER`
2. Logout and login again
3. Or use sudo: `sudo python3 -m minisweagent.run.mini ...`

### Issue: Workspace Not Mounted

**Symptom:** Agent can't find kernel files in Docker

**Solution:** Specify workspace explicitly:
```bash
--workspace /absolute/path/to/kernels
```

---

## Best Practices

### 1. Use Auto-Detection by Default

Let the agent detect and prompt - it's smart enough!

```bash
# Good - let agent decide
python3 -m minisweagent.run.mini -m claude-sonnet-4.5 -t "task"

# Also good - YOLO for CI/CD
python3 -m minisweagent.run.mini -m claude-sonnet-4.5 -t "task" --yolo
```

### 2. Pin Docker Image for Reproducibility

In production/CI, specify exact image:

```bash
--runtime docker \
--docker-image rocm/pytorch:rocm6.0_ubuntu22.04_py3.10_pytorch_2.1.1
```

### 3. Mount Absolute Paths

Always use absolute paths for workspace:

```bash
# Good
--workspace /home/user/project/kernels

# Bad (relative paths may not work in Docker)
--workspace ./kernels
```

### 4. Clean Up Containers

Containers are auto-cleaned (`--rm`), but you can verify:

```bash
# List GEAK containers
docker ps -a | grep minisweagent

# Clean up if needed
docker rm -f $(docker ps -a -q --filter "name=minisweagent")
```

---

## Testing

### Test Runtime Detection

```bash
cd .
python3 -m minisweagent.runtime_env
```

Output shows:
- Current environment status
- Interactive prompt simulation
- Generated config

### Test with Agent

```bash
# Test with simple kernel
python3 -m minisweagent.run.mini \
  -m claude-sonnet-4.5 \
  -t "Check if torch and triton are available" \
  --runtime docker \
  --workspace /path/to/workspace \
  --yolo
```

---

## Integration with Discovery Pipeline

The `minisweagent.run.mini` (discovery pipeline) also supports runtime environments:

```bash
# Auto-detect
python3 -m minisweagent.run.mini /path/to/kernel.py

# Force Docker
python3 -m minisweagent.run.mini /path/to/kernel.py \
  --runtime docker \
  --docker-image rocm/pytorch:latest

# Skip runtime check (use local)
python3 -m minisweagent.run.mini /path/to/kernel.py \
  --no-runtime-check
```

---

## Future Enhancements

- [ ] Support for Singularity containers
- [ ] Pre-warming Docker images
- [ ] Multi-GPU support
- [ ] Remote Docker hosts
- [ ] Container reuse across runs
- [ ] Custom device mappings
- [ ] Environment variable forwarding UI

---

## Summary

The runtime environment system makes GEAK Agent:
- ✅ **Portable** - Works on any machine (with or without GPU)
- ✅ **User-friendly** - Auto-detects and prompts intelligently
- ✅ **Safe** - Defaults to Docker when local env incomplete
- ✅ **Flexible** - Supports custom images and configurations
- ✅ **Automated** - YOLO mode for CI/CD pipelines

**Default behavior:** Auto-detect → Prompt if needed → Default to `lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x`
