---
myst:
    html_meta:
        "description": "Run the GEAK agent to optimize GPU kernels. Includes single-agent and parallel multi-GPU examples, end-to-end CLI invocations, output structure, and a full CLI flag reference."
        "keywords": "GEAK, run agent, kernel optimization, GPU, CLI, parallel agents, geak CLI, Triton, HIP"
---

# Run the GEAK agent

The `geak` CLI accepts a kernel target and an optional test command, then runs one or more optimization agents against it. This topic shows common invocation patterns, explains parallel multi-GPU runs, and documents every CLI flag.

## Examples

The following examples cover the most common invocation patterns, from a single natural-language prompt to parallel multi-GPU runs.

### Typical kernel optimization (natural-language input)

Pass the full task as a `-t` string. GEAK parses the kernel location, GPU IDs, and harness path from the text.

```bash
geak -t "Optimize the kernel from /path/to/aiter, specifically aiter/ops/triton/topk.py. Use the harness at /path/to/test_topk_harness.py. Use four GPUs with IDs 0-3 simultaneously."
```

### Typical kernel optimization (single agent)

Use explicit flags to specify the kernel and repository when you want precise control over inputs.

```bash
geak --kernel-url /path/to/kernel/file \
  --repo /path/to/kernel/repo \
  --task "Optimize the block_reduce kernel" 
```

### Parallel agents

Pass `--gpu-ids` as a comma-separated list of device indices (`0,1,2,3`). Each parallel agent is bound to one GPU: agent `i` uses `gpu_ids[i]` (0-based). For full isolation, set `--num-parallel` to the same count as the IDs you list; if you supply fewer IDs than agents, some agents share a GPU, and the CLI prints a warning.

```bash
geak --num-parallel 4 \
  --repo /path/to/kernel/repo \
  --kernel-url /path/to/kernel/file \
  --task "Optimize block_reduce. Metric: Extract Bandwidth in GB/s (higher is better)" \
  --gpu-ids 0,1,2,3 
```

### End-to-end examples

The four invocations below show every combination of natural-language (NL) task description versus explicit CLI flags, and with or without a pre-built test harness. All target the same kernel (`topk.py` in [aiter](https://github.com/ROCm/aiter)) on 8 GPUs.

```{note}
In the examples below, suppose `/workspace/GEAK_ARTIFACTS` is the folder where all GEAK-related outputs will be saved. The `test_topk_harness.py` referenced in some examples is a user-created harness file for testing the topk kernel.
```

Task 1 — Without NL, without harness: GEAK discovers the kernel structure, generates its own harness, and runs optimization from CLI flags.

```bash
geak --kernel-url 'https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/topk.py' \
  --gpu-ids '0,1,2,3,4,5,6,7' --yolo \
  --task 'Optimize the topk kernel.' --exit-immediately \
  -o '/workspace/GEAK_ARTIFACTS/topk_wo_task_wo_harness' \
  2>&1 | tee '/workspace/GEAK_ARTIFACTS/topk_wo_task_wo_harness.log'
```

Task 2 — Without NL, with harness: same CLI-flag style, but a pre-existing test harness is supplied via `--test-command`.

```bash
geak --kernel-url 'https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/topk.py' \
  --test-command 'python3 /workspace/GEAK_ARTIFACTS/test_topk_harness.py --correctness && python3 /workspace/GEAK_ARTIFACTS/test_topk_harness.py --full-benchmark' \
  --gpu-ids '0,1,2,3,4,5,6,7' --yolo \
  --task 'Optimize the topk kernel.' --exit-immediately \
  -o '/workspace/GEAK_ARTIFACTS/topk_wo_task_w_harness' \
  2>&1 | tee '/workspace/GEAK_ARTIFACTS/topk_wo_task_w_harness.log'
```

Task 3 — With NL, without harness: everything is expressed in a single natural-language `-t` string. GEAK parses the kernel URL, GPU count, and mode from the text.

```bash
geak -t 'Optimize the topk kernel at https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/topk.py. Use GPUs 0-7.' \
  --yolo --exit-immediately \
  -o '/workspace/GEAK_ARTIFACTS/topk_w_task_wo_harness' \
  2>&1 | tee '/workspace/GEAK_ARTIFACTS/topk_w_task_wo_harness.log'
```

Task 4 — With NL, with harness: natural-language task that also references an external test harness URL.

```bash
geak -t 'Optimize the topk kernel at https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/topk.py, using the harness at https://github.com/AMD-AGI/AIG-Eval/blob/sdubagun/fix-kernel-harness-parity/tasks/geak_eval/topk/test_topk_harness.py. Use GPUs 0-7.' \
  --yolo --exit-immediately \
  -o '/workspace/GEAK_ARTIFACTS/topk_w_task_w_harness' \
  2>&1 | tee '/workspace/GEAK_ARTIFACTS/topk_w_task_w_harness.log'
```

### CLI reference

Options match the Typer `Option` definitions in `main` (same names in `geak` / `mini`).

| Option | Meaning |
|--------|---------|
| `-m`, `--model` | Model name. |
| `--model-class` | For example, `litellm` and `amd_llm`. |
| `-t`, `--task` | Task string. If it equals an existing file path, `geak` reads that file as the task body. |
| `-y`, `--yolo` | Non-interactive / auto-confirm tool execution (sets `agent.mode` to `yolo`). Parallel runs already force `yolo` on each worker; this flag mainly affects single-agent `geak`. |
| `-l`, `--cost-limit` | Agent cost limit (use `0` to disable). |
| `-c`, `--config` | Path to the config file. Overrides the default config file `geak.yaml`. |
| `-o`, `--output` | Trajectory file or output directory. Default is `./optimization_logs/kernel_name_timestamp`. |
| `--exit-immediately` | Sets `agent.confirm_exit` to `False` in config. |
| `--repo` | Repository root for kernel. Even if the kernel code is in a single file, it needs to be in a repository. |
| `--kernel-url` | Kernel source file path or URL. Required unless `kernel target` is supplied another way (for example, parsed from `--task "kernel url is xxx"`). URLs are resolved by `run/preprocess/resolve_kernel_url.py` (clone/checkout under run output). |
| `--num-parallel` | Number of parallel agent runs. |
| `--gpu-ids` | Comma-separated GPU device indices. |
| `--test-command`, `--test_command` | Test command used to verify correctness and performance of the kernel. |

## Outputs

GEAK saves patches and test logs so the optimization progress and results are transparent.

- **Default output base**: `optimization_logs/`
- **Auto-generated run directory**: `optimization_logs/<kernel_name>_<YYYYmmdd_HHMMSS>/`

Typical structure (parallel run):

```bash
optimization_logs/<kernel>_<timestamp>/
├── parallel_0/
│   ├── patch_0.patch
│   ├── patch_0_test.txt
│   └── agent_0.log
├── parallel_1/
│   └── ...
├── best_results.json
└── select_agent.log
```

Structure for triton kernels:

```bash
optimization_logs/<kernel>_<timestamp>/
├── results/round_1/<kernel>-<strategy_0>/
│   ├── patch_0.patch
│   ├── patch_0_test.txt
│   └── task_0.log
├── results/round_1/<kernel>-<strategy_1>/
│   └── ...
├── best_results.json
└── select_agent.log
```

## Related topics

- [Install GEAK](../install/install.md) — set up GEAK and configure a model backend before running.
- [API reference](../reference/api-reference.md) — complete CLI flag reference, environment variables, and artifact layout.
- [GEAK agent loop](../conceptual/geak-pipeline.md) — understand how the optimization pipeline works.