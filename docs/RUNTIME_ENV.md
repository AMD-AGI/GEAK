# Runtime Environment Configuration

GEAK Agent supports automatic runtime environment detection and configuration for GPU kernel operations.

## Overview

The runtime environment system:
1. **Detects** local GPU and dependencies (PyTorch, Triton)
2. **Prompts** for Docker if local environment is incomplete
3. **Defaults** to `lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x` Docker image
4. **Manages** Docker containers automatically

See [RUNTIME_QUICKSTART.md](RUNTIME_QUICKSTART.md) for a short reference.

---

## Quick Start

### Auto-Detection (Recommended)

```bash
python3 -m geakagent.run.mini -m claude-sonnet-4.5 -t "Optimize my kernel" --yolo
```

### Force Docker

```bash
python3 -m geakagent.run.mini -m claude-sonnet-4.5 -t "Optimize my kernel" \
  --runtime docker --workspace /path/to/kernels
```

### Force Local

```bash
python3 -m geakagent.run.mini -m claude-sonnet-4.5 -t "Optimize my kernel" --runtime local
```

---

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--runtime TYPE` | Runtime type: `local`, `docker`, or `auto` | `auto` |
| `--docker-image IMG` | Docker image to use | `lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x` |
| `--workspace PATH` | Workspace directory to mount in Docker | Current directory |
| `--no-runtime-check` | Skip runtime environment detection | `false` |

---

## How It Works

1. **Detection:** GPU (ROCm/CUDA), PyTorch, Triton.
2. **Decision:** Local complete → use local; else offer Docker or continue with local.
3. **Docker:** Mounts workspace as `/workspace`, forwards `/dev/kfd`, `/dev/dri`.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Docker not available | `sudo apt install docker.io` |
| Permission denied | `sudo usermod -aG docker $USER` (logout/login) |
| No HIP GPUs | Check `ls /dev/kfd /dev/dri`, run `rocminfo` |
| Workspace not found | Use `--workspace /absolute/path` |

---

## Discovery pipeline

`geak_agent.cli` supports the same runtime options:

```bash
python3 -m geak_agent.cli /path/to/kernel.py --runtime docker
```
