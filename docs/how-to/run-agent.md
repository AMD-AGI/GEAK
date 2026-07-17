---
myst:
    html_meta:
        "description": "Run a GEAK v4 workflow from Claude Code: end-to-end sglang/vLLM serving-throughput optimization or single-kernel optimization, with depth modes and the accuracy gate."
        "keywords": "GEAK, run workflow, serving throughput, single kernel, Claude Code, Workflow, sglang, vLLM, deep mode, gsm8k"
---

# Run a GEAK workflow

GEAK v4 runs inside Claude Code, orchestrated by deterministic JS Workflows. There is no
`pip install` and no CLI: launch Claude Code and describe the task; it invokes the `Workflow` tool.

## Prerequisites

Before running a workflow, ensure the following are in place.

- **AMD Instinct™ MI GPU** — CDNA (gfx942 / gfx950), auto-detected.
- **ROCm 6+** with `rocminfo` / `rocm-smi`, and a profiler (`rocprof-compute` / `rocprofv3` / `rocprof`).
- **Python 3.8+**.
- **Claude Code ≥ 2.1.177** (dynamic Workflow feature). Check `claude --version`.
- **For E2E:** a running-capable `sglang` or `vllm` and the model weights on disk.

## Get the repo and launch Claude Code

Clone the GEAK repository and launch Claude Code in sandbox mode.

```bash
claude update                          # ensure Claude Code >= 2.1.177
git clone https://github.com/AMD-AGI/GEAK.git && cd GEAK
IS_SANDBOX=1 claude --dangerously-skip-permissions
```

Sandbox mode auto-approves permissions, which the workflows need to run profiling, benchmark, and build
commands.

## Run a workflow (natural language)

The path you give Claude Code (for example, `path_to_GEAK/e2e_workflow`) is mapped to the `workflow_dir` argument that the workflow requires. Replace `path_to_GEAK` with the absolute path to your cloned GEAK repository.

### End-to-end serving throughput (e2e_workflow)

```
use /absolute/path/to/GEAK/e2e_workflow to optimize inference for /models/Qwen3.5-27B-FP8, sglang, ISL/OSL=1024, conc=64, gpus 0,1,2,3
```

Profiles a running server, triages hot kernels by Amdahl (`pct_gpu_time × achievable_speedup`), pulls
levers cheapest-first (config/backend sweep → head GEMM/attention bake-off → editable-kernel milestone
loop), and overlays each accepted change back reversibly, gated on a measured throughput delta.

Output: `e2e_workflow/exp/e2e_<model>_<timestamp>/` — `final_report.md`, `architect_report.md`, `final/`
(overlay + patch + `final_launch.sh`). See [`examples/e2e_workflow/`](../../examples/e2e_workflow/).

### Single kernel (kernel_workflow)

```
use /absolute/path/to/GEAK/kernel_workflow to optimize /absolute/path/to/GEAK/examples/tasks/knn
use /absolute/path/to/GEAK/kernel_workflow to optimize /path/to/silu, budget 8, focus on wrapper overhead
```

Director → TechLead → specialist engineers, multi-round and budget-controlled, each patch independently
verified. Output: `kernel_workflow/exp/team_<kernel>_<timestamp>/`.

**Batch:** run multiple kernels in parallel by launching one non-interactive Claude Code process per
kernel. GPU access is serialized internally using `kernel_workflow/scripts/gpu_lock.sh`, so all
processes can safely share the same GPUs.

```bash
GEAK=/absolute/path/to/GEAK
GPU_IDS="0,1,2,3"

for KERNEL in /path/to/kernel_a /path/to/kernel_b /path/to/kernel_c; do
  IS_SANDBOX=1 claude -p \
    "use $GEAK/kernel_workflow to optimize $KERNEL, gpu_ids $GPU_IDS" \
    --dangerously-skip-permissions \
    > "$KERNEL/claude.log" 2>&1 &
done

wait
```

Each process writes its output to a per-kernel log. The `kernel_workflow` creates its experiment
directory under `kernel_workflow/exp/team_<kernel>_<timestamp>/`. The `-p` flag runs Claude Code
non-interactively (print mode); the process exits when the workflow completes.

## Depth modes (e2e)

Both default off = **default** mode; mutually exclusive, deep wins.

| Mode | Trigger | What runs |
|---|---|---|
| **default** | *(none)* | ConfigSweep + HeadKernel + Milestone. |
| **fast** | "fast mode" | HeadKernel only, parallel, time-boxed (`fast_budget_ms`, 5h). |
| **deep** | "deep mode" | ConfigSweep + HeadKernel, cross-kernel×backend lane pool, many rounds (`deep_head_budget_ms`, 24h). |

```
use /absolute/path/to/GEAK/e2e_workflow, deep mode, to optimize /models/Qwen3.5-27B-FP8 on gpus 0-7
```

## Accuracy gate (quantized kernels)

For FP8 / MXFP4, byte-parity is too strict — switch the e2e gate to task accuracy:

```
... use the gsm8k accuracy gate with limit 200 and tolerance 0.01
```

The Integrator then runs sampled gsm8k (5-shot, greedy, fixed seed) and accepts iff
`cand_em >= baseline_em - tol`.

## See also

- [Install GEAK](../install/install.md) 
- [API reference](../reference/api-reference.md) 
- [Compatibility matrix](../compatibility.md)
