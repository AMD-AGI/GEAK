---
myst:
    html_meta:
        "description": "GEAK v4 reference: Workflow invocation surfaces, e2e_workflow and kernel_workflow arguments, the run_e2e.py integration contract, and run artifact layout."
        "keywords": "GEAK, reference, Workflow, e2e_workflow, kernel_workflow, run_e2e, arguments, handoff, result, artifacts"
---

# Reference

GEAK is invoked through the Claude Code **`Workflow`** tool. In normal use you describe the task in
natural language (see [Run a workflow](../how-to/run-agent.md)) and Claude Code fills in the arguments;
this page documents the arguments and the machine contract for reference and for external integration.

## Invocation surfaces

| Surface | Use |
|---|---|
| Claude Code natural language | Interactive runs — Claude Code resolves paths and calls `Workflow`. |
| `interface/run_e2e.py` | External orchestrators — a version-stable command + two JSON files. |

## `e2e_workflow` arguments

Passed as the `Workflow` tool's `args`. `model_path` and `workflow_dir` are required; the rest have
defaults.

| Argument | Default | Meaning |
|---|---|---|
| `model_path` | — (required) | Model to serve and optimize. |
| `workflow_dir` | — (required) | Absolute path to `e2e_workflow/` (a JS workflow can't read its own path). |
| `backend` | `sglang` | Serving backend: `sglang` \| `vllm` (selects `scripts/adapters/<backend>.sh`). |
| `gpu_ids` | `0` | Comma-separated optimization GPU pool. |
| `tp` | `1` | Serving tensor-parallel size (used for every e2e measurement). |
| `isl` / `osl` / `conc` | `1024` / `1024` / `64` | Workload: input/output length and concurrency. |
| `budget` | `6` | Max editable-kernel optimization tasks. |
| `config_tune` | `true` | Run the config/backend sweep first. |
| `milestone_min_pct` | `5` | Skip editable kernels below this `pct_gpu_time` (Amdahl). |
| `accuracy_gate` | `none` | `gsm8k` switches the quant gate from byte parity to task accuracy. |
| `apply_to_original` | `false` | Emit an apply bundle (overlay + launch script). |
| `kernel_path` | — | Single-kernel pass-through: pass this instead of `model_path` to delegate straight to `kernel_workflow`. |

## `kernel_workflow` arguments

| Argument | Default | Meaning |
|---|---|---|
| `kernel_path` | — (required) | Kernel task directory or source file. |
| `workflow_dir` | — (required) | Absolute path to `kernel_workflow/`. |
| `gpu_ids` | `0` | GPU pool; access is serialized via `scripts/gpu_lock.sh`. |
| `budget` | — | Optimization-round budget. |
| `task` | — | Optional natural-language steer (e.g. focus area). |

## `run_e2e.py` integration contract

```bash
python interface/run_e2e.py <handoff.json> <result.json> [--dry-run]
```

Exit codes: `0` = `ok`/`no_gain`, `1` = crash (`status: error`), `2` = bad usage. `--dry-run` prints the
mapped arguments + prompt and exits without GPU work. As long as `schema_version` stays `1`, the caller
does not change when the workflow evolves internally.

### `handoff.json` (caller → workflow)

Required: `model_path`, `exp_root`. Common optional fields:

| Field | Meaning |
|---|---|
| `framework` | `sglang` \| `vllm`. |
| `tp` / `gpu_ids` | Serving tensor-parallel size / GPU set. |
| `workload.{isl,osl,conc}` | Profile + bench workload. |
| `accepted_flags` / `accepted_env` | Caller's best config; seeds the baseline. |
| `launch_recipe` | Optional launch script/recipe. |
| `bench_protocol.{random_range_ratio,num_prompts,num_warmups,seed}` | Measurement protocol; forwarded so numbers are cross-harness comparable. |

### `result.json` (workflow → caller)

| Field | Meaning |
|---|---|
| `status` | `ok` \| `no_gain` \| `error`. |
| `baseline_throughput_tok_s` / `final_throughput_tok_s` / `throughput_speedup` | The measured result. |
| `output_parity` | `pass` \| `fail` \| `n/a` \| `unknown`. |
| `ttft_ms` / `tpot_ms` | Median latencies. |
| `final_launch_script` / `final_overlay` / `final_patch` | Self-contained deliverables. |
| `report_path` | Human `final_report.md`. |

If the workflow return can't be scraped, `run_e2e.py` rebuilds `result.json` from on-disk artifacts and
sets `recovered_from_disk: true`, so a real win is never lost.

## Run artifacts

Each run writes a timestamped directory under `exp_root` (or `e2e_workflow/exp/`):

```
e2e_<model>_<timestamp>/
├── baseline/            # true baseline bench
├── profile/round_*/     # standardized Top-N per round
├── overlay/             # candidate + accepted reversible overlays
├── final/               # overlay + final_patch.diff + final_launch.sh
├── final_report.md      # complete timeline report
├── architect_report.md  # concise summary
└── director_e2e_validation.json   # independently verified result
```

## Related topics

- [Run a workflow](../how-to/run-agent.md) — invocation examples.
- [GEAK pipeline](../conceptual/geak-pipeline.md) — what each phase does.
