# Kernel Setup Generator

Generates instruction files and run scripts for new kernel workspaces.

## Usage

```bash
python3 scripts/generate_kernel_setup.py <workspace_dir> \
    --gpu <gpu_num> \
    --agent-num <agent_id> \
    [--kernel-name "Display Name"] \
    [--original-path "path/in/aiter"] \
    [--iterations 50]
```

## Examples

### Basic usage (auto-detect kernel name)
```bash
python3 scripts/generate_kernel_setup.py topk_workspace \
    --gpu 3 \
    --agent-num 3
```

### Full specification
```bash
python3 scripts/generate_kernel_setup.py moe_mxfp4_workspace \
    --gpu 3 \
    --agent-num 8 \
    --kernel-name "MoE MxFP4" \
    --original-path "moe_mxfp4" \
    --iterations 50
```

## Generated Files

The generator creates files **inside the workspace directory**:

```
<workspace_dir>/
├── kernel.py                           # Your kernel (add manually)
├── AGENT_<num>_<NAME>.md              # Generated instruction file
└── run.sh                             # Generated run script (executable)
```

## Workflow

1. **Generate setup files:**
   ```bash
   python3 scripts/generate_kernel_setup.py mykernel_workspace --gpu 3 --agent-num 10
   ```

2. **Add your kernel:**
   ```bash
   cp /path/to/source.py mykernel_workspace/kernel.py
   ```

3. **Reference in Cursor:**
   ```
   @mykernel_workspace/AGENT_10_MYKERNEL.md
   ```

4. **Run optimization:**
   ```bash
   cd mykernel_workspace && ./run.sh
   ```

## Templates

Templates are located in the repository root:
- `KERNEL_TEMPLATE.md` - Instruction file template
- `RUN_SCRIPT_TEMPLATE.sh` - Run script template (checked into git, no API key defaults)

Update templates to change the structure for all future generated files.

**Note:** The RUN_SCRIPT_TEMPLATE.sh does not include a default API key. Users must export `AMD_LLM_API_KEY` before running the generated scripts.
