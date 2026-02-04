# Runtime Environment - Quick Reference

**TL;DR:** Agent auto-detects GPU environment and defaults to Docker if needed.

---

## The Basics

### Auto (Default - Recommended)

```bash
python3 -m geakagent.run.mini -m claude-sonnet-4.5 -t "your task" --yolo
```

Agent will:
1. Detect local GPU + dependencies
2. Use local if complete
3. Use Docker (`lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x`) if not
4. Prompt if neither (unless `--yolo`)

### Force Docker

```bash
python3 -m geakagent.run.mini -m claude-sonnet-4.5 -t "your task" \
  --runtime docker \
  --workspace /path/to/kernels \
  --yolo
```

### Force Local

```bash
python3 -m geakagent.run.mini -m claude-sonnet-4.5 -t "your task" \
  --runtime local \
  --yolo
```

### Custom Docker Image

```bash
python3 -m geakagent.run.mini -m claude-sonnet-4.5 -t "your task" \
  --runtime docker \
  --docker-image rocm/pytorch:latest \
  --yolo
```

---

## Command Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--runtime` | `auto`, `local`, `docker` | `auto` | Runtime type |
| `--docker-image` | Image name | `lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x` | Docker image |
| `--workspace` | Directory path | Current dir | Workspace to mount |
| `--no-runtime-check` | Flag | Off | Skip detection |

---

## What Gets Detected

✅ **GPU:** ROCm (`/opt/rocm`) or CUDA (`nvidia-smi`)  
✅ **PyTorch:** `import torch` + `torch.cuda.is_available()`  
✅ **Triton:** `import triton`  
✅ **Docker:** `docker --version`

---

## Decision Tree

```
Complete local (GPU + torch + triton)?
  YES → Use local ✅
  NO  → Docker available?
          YES → Offer Docker (default) or local
          NO  → Use local (limited) ⚠️
```

---

## Examples

### Check What Runtime Will Be Used

```bash
cd .
python3 -m geakagent.runtime_env
```

### Test with Simple Task

```bash
python3 -m geakagent.run.mini -m claude-sonnet-4.5 \
  -t "Run: python3 -c 'import torch; print(torch.cuda.is_available())'" \
  --runtime docker --yolo
```

### Discovery Pipeline with Runtime

```bash
python3 -m geak_agent.cli /path/to/kernel.py \
  --runtime docker \
  --no-confirm
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Docker not available" | Install: `sudo apt install docker.io` |
| "Permission denied" | Add user: `sudo usermod -aG docker $USER` (logout/login) |
| "No HIP GPUs" | Check: `ls /dev/kfd /dev/dri` |
| Workspace not mounted | Use absolute path: `--workspace /full/path` |

---

## Files

- **Module:** `src/geakagent/runtime_env.py`
- **Tests:** `test_runtime_env.sh`
- **Docs:** `RUNTIME_ENV.md` (full guide)
- **Implementation:** `RUNTIME_ENV_IMPLEMENTATION.md` (technical details)

---

## Default Docker Image

```
lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x
```

**Contains:**
- ROCm 7.0.0
- PyTorch + Triton
- SGLang runtime
- Python 3.10+

**Devices Passed:**
- `/dev/kfd` (AMD GPU kernel)
- `/dev/dri` (Direct Rendering)

---

## Status

✅ Implemented  
✅ Tested (6/6 tests passing)  
✅ Documented  
✅ Backward Compatible  
⏳ Pending GPU integration tests

**Ready to use!**
