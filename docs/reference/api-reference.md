---
myst:
    html_meta:
        "description": "GEAK v4 reference: the Workflow tool, e2e_workflow and kernel_workflow arguments, helper scripts, the external-orchestrator contract, and run artifacts."
        "keywords": "GEAK, reference, Workflow, e2e_workflow, kernel_workflow, run_e2e, arguments, helper scripts, artifacts"
---

# GEAK API reference

GEAK v4's public surface is not a CLI or a Python package. It is: the Workflow scripts
(`e2e_workflow.js`, `kernel_workflow.js`), the helper scripts they call, and a stable
external-orchestrator contract (`interface/run_e2e.py`). For walkthroughs see
[Run a workflow](../how-to/run-agent.md).

## The `Workflow` tool

Claude Code runs a workflow by calling:

```js
Workflow({ scriptPath: "<absolute path to *.js>", args: { /* see below */ } })
```

`workflow_dir` is always required (a JS workflow can't read its own path). Natural-language prompts
are mapped onto `args`; there is no config file for workflow parameters.

| Workflow | scriptPath | Purpose |
|---|---|---|
| e2e | `e2e_workflow/e2e_workflow.js` | Whole-model sglang/vLLM serving-throughput optimization. |
| single kernel | `kernel_workflow/kernel_workflow.js` | Optimize/author one AMD GPU kernel. |

## `e2e_workflow.js`

`meta.name = "e2e-workflow"`. Phases:
`Setup → Profile → Strategize → ConfigSweep → HeadKernel → Milestone → Finalize → Report → Validate`.

Owns the server, throughput metric, profiling, Amdahl triage, config/backend tuning, hot-kernel
extraction, and reversible reintegration; recursively calls `kernel_workflow.js` per kernel.

### Core args

| Arg | Default | Description |
|---|---|---|
| `model_path` | — (**required** for e2e) | Model weights directory. |
| `kernel_path` | — | Single-kernel pass-through (mutually exclusive with `model_path`); delegates straight to the kernel layer. |
| `workflow_dir` | — (**required**) | This folder (`e2e_workflow/`). |
| `kernel_workflow_dir` | sibling `kernel_workflow/` | Recursive kernel layer location. |
| `exp_root` | sibling `exp/` | Output root. |
| `backend` | `sglang` | Serving adapter: `sglang` \| `vllm` (selects `scripts/adapters/<backend>.sh`). |
| `launch_script` | `""` | Optional custom launch script; else the stack default. |
| `gpu_ids` | `0` | CSV optimization-parallelism pool. |
| `tp` / `serving_tp` | `1` | Serving tensor-parallel size. |
| `serving_gpu` | first `tp` ids | Serving GPU set (distinct from `gpu_ids`). |
| `isl`, `osl`, `conc` | `1024`, `1024`, `64` | Workload; profiling and benchmarking share these. |
| `task` | `""` | Natural-language steer. |
| `apply_to_original` | `false` | If `true`, emit an apply bundle (overlay + launch); never edits site-packages. |
| `phases` | `all` | Run a subset: `{setup, config, head, kernel, final}`. `state` carries cross-phase state. |

### Budget and triage args

| Arg | Default | Description |
|---|---|---|
| `budget` | `6` | Max kernel-optimization tasks (config sweep is free). |
| `kernel_budget` | `6` (`3` in fast) | Budget passed down to each recursive single-kernel run. |
| `min_kernel_tasks` | `4` (capped by `budget`) | Milestone floor. |
| `milestone_min_pct` | `5` | Skip editable kernels below this %GPU time (Amdahl). |
| `config_tune` | `true` | Tier-0 flag/env/backend sweep on/off (runs FIRST). |
| `head_threshold_pct` | `5` | Head-kernel selection threshold. |
| `head_budget` | `3` (≥ GPU count in fast) | Max head bake-offs. |
| `head_author_max` | `2` | Author languages per head (FlyDSL + Triton). |
| `head_protect_pct` | `30` | A dominant head is never silently dropped. |
| `head_corrective_max` | `2` | Corrective re-author retries. |

### Depth modes

Both default `false` → default mode. Mutually exclusive; deep takes precedence. With both off,
the run is byte-identical to an unmodified run (every mode knob is gated).

| Mode | Arg | Phases | Default budget |
|---|---|---|---|
| default | *(none)* | ConfigSweep + HeadKernel + Milestone | — |
| fast | `fast_mode: true` | HeadKernel only (parallel, time-boxed) | `fast_budget_ms` = 5h (`18000000`) |
| deep | `deep_mode: true` | ConfigSweep + HeadKernel (global cross-kernel×backend lane pool) | `deep_head_budget_ms` = 24h (`86400000`) |

Related timing/tuning args: `fast_head_deadline_ms`, `fast_head_workflow_ms`, `deep_wave_budget`,
`deep_max_reseeds`, `deep_converge_streak`, `deep_agent_budget`, `deep_e2e_target`, `deep_backends`.

### Accuracy gate

| Arg | Default | Description |
|---|---|---|
| `accuracy_gate` | `none` | `none` \| `gsm8k`. For quantized kernels, switch the bar to task accuracy. |
| `accuracy_limit` | `200` | Number of gsm8k questions. |
| `accuracy_tol` | `0.01` | Allowed exact-match drop (`cand_em >= baseline_em - tol`). |

### Measurement and misc args

| Arg | Default | Description |
|---|---|---|
| `noise_band_pct` | `0.5` | e2e acceptance band (%). |
| `e2e_repeats` | `2` | Repeats per timed measurement. |
| `ab_finish_retries` | `3` | A/B leg completion retries. |
| `use_expert_skills` | `false` | Consult `perf_knowledge/expert_skills` (advisory priors). OFF = byte-identical. |
| `perf_knowledge_dir` | sibling `perf_knowledge/` | Authoring knowledge base. |
| `time_budget_s`, `initial_extra_server_args`, `initial_extra_env`, `tracelens`, `agent_timeout_ms` | — | Forwarded from the external orchestrator. |

### Example

```js
Workflow({
  scriptPath: "<REPO>/e2e_workflow/e2e_workflow.js",
  args: {
    model_path: "/models/Qwen3.5-27B-FP8",
    workflow_dir: "<REPO>/e2e_workflow",
    backend: "sglang", tp: 4, gpu_ids: "0,1,2,3",
    isl: 1024, osl: 1024, conc: 64,
    budget: 6, config_tune: "true"
  }
})
```

## `kernel_workflow.js`

`meta.name = "kernel-workflow"`. Phases:
`Setup → Author → Analyze → Benchmark → Profile → Optimize → Verify → Merge → Report → Validate`.

Director → TechLead → specialist engineers (algorithm, memory, compute, and host_runtime), multi-round,
budget-controlled, each patch independently verified.

| Arg | Default | Description |
|---|---|---|
| `kernel_path` | — (**required**) | Kernel task directory (or a kernel source to wrap). |
| `workflow_dir` | — (**required**) | This folder (`kernel_workflow/`). |
| `budget` | `6` | Optimization-task budget. |
| `min_improve` | `0.02` | Minimum accepted improvement (2%). |
| `deep_cost` | `2` | Budget cost of a deep-explore engineer. |
| `gpu_ids` | `0` | CSV GPU pool. |
| `task` | `""` | Natural-language steer. |
| `exp_root` | sibling `exp/` | Output root. |
| `eval_dir` | — | Isolated evaluation directory. |
| `apply_to_original` | `false` | Emit apply bundle vs edit-in-place. |
| `mode` | `optimize` | `optimize` \| `author`. |
| `target_language` | `triton` | Author-mode language: `triton` \| `flydsl` \| `hip` \| `ck`. |
| `op_spec` | `{}` | Op specification (author mode). |
| `perf_knowledge_dir` | sibling `perf_knowledge/` | Knowledge base. |
| `update_experience` | `on` | Curate ONE distilled card into `kernel_workflow/knowledge/learned/` after a measured win (every lane run; once centrally per bake-off). `off` skips it. The sink is always this workflow's own `learned/` — an e2e-opened lane never writes into `e2e_workflow/knowledge/learned/`. Cards carry a skill-style discovery header (`name`/`description`/`keywords`/`kernels`/`platforms`/`kernel_class`/`regime`) and `learned/INDEX.md` is regenerated from it by `kernel_workflow/scripts/build_learned_index.js` (`--check` = fail-if-stale). Numbers on a card are relative only (ratios, % of achievable peak) — never wall-clock. |
| `workload_spec_path` | — | Workload-alignment spec; makes the primary metric the time-weighted ratio-of-sums. |
| `agent_timeout_ms` | `3600000` | Per-agent timeout (1h). |
| `agent_retries` | `4` | Agent retry count (min 1). |

**Speedup metric** = `geomean(baseline_ms / optimized_ms)`; with a workload spec it becomes the
time-weighted ratio-of-sums. Author mode writes a fresh baseline against an immutable oracle, then runs
the same optimize loop.

```js
Workflow({
  scriptPath: "<REPO>/kernel_workflow/kernel_workflow.js",
  args: {
    kernel_path: "<REPO>/examples/tasks/knn",
    workflow_dir: "<REPO>/kernel_workflow",
    budget: 8, gpu_ids: "0"
  }
})
```

## Helper scripts

### `e2e_workflow/scripts/bench_e2e.sh`

Backend-agnostic e2e serving-benchmark dispatcher: server lifecycle, health-wait, and cleanup
(or `REUSE_SERVER=1`), warmup + N timed repeats + optional profiling trace, median throughput + spread.
Env-driven; `MODEL` is **required** (no rig default).

Key env vars: `MODEL`, `TP`, `GPU`, `ISL`/`OSL`/`CONC`, `REPEATS` (default 3), `MEM_FRACTION`,
`EXTRA_SERVER_ARGS`, `EXTRA_ENV`, `OVERLAY_PYTHONPATH`, `PROFILE`, `BENCH_CLIENT`, `PORT_BASE`/`PORT_SPAN`,
`NUM_PROMPTS`, `NUM_WARMUPS`, `SEED`, `PROFILE_NUM_STEPS`, `REUSE_SERVER`.

### `e2e_workflow/scripts/op_bench.py`

Single-op multi-backend bake-off and autotune for head kernels (GEMM and attention). Isolated; never touches
a server. Reads `meta.json` (+ optional `reference_io.pt`).

```bash
python op_bench.py --task <op_task_dir> \
  [--backends hipblaslt,tunableop,rocblas,aiter,triton] \
  [--repeats 50] [--warmup 10] [--tol 2e-2] [--triton-autotune] [--seed 0] [--out result.json]
```

### `e2e_workflow/scripts/parse_profile.py`

Standardized profile → per-kernel Top-N (JSON + MD). Merges a torch trace (op names + shapes) with the
rocprofv3 CSV (HW durations); also emits the workload-v1 spec.

```bash
python parse_profile.py --torch-trace <file.json[.gz]> --rocprof-dir <dir> \
  --top 25 --out <prefix> --workload-out <file> --target <name>
```

### `e2e_workflow/scripts/parse_regime.py`

Parse the online serving regime from launch flags + model config; emits a
quant / kv_cache_dtype / compile / cuda_graph / attention_backend descriptor into `meta.json`.

```bash
python parse_regime.py --server-args <...> --model-config <...> --out <file>
```

### `e2e_workflow/scripts/attribute_weights.py`

op_kind-aware weight attribution joining extractor `meta.json` shapes with profiled time; emits
workload-v1 JSON with `weight_source` ∈ {trace, regime, prior}.

```bash
python attribute_weights.py --meta <...> --profile-weights <...> \
  --name-match <...> --min-regime-share 0.0 --out <file>
```

### `e2e_workflow/scripts/gsm8k_eval.py`

Lightweight gsm8k accuracy eval against an OpenAI-compatible `/v1` endpoint (5-shot greedy exact-match).
Prints `GSM8K_EXACT_MATCH=<0..1>`.

```bash
python gsm8k_eval.py --base-url <url> --model <name> \
  [--limit 200] [--fewshot 5] [--max-tokens 1024] [--concurrency 32] [--seed 0] [--out <file>]
```

### `e2e_workflow/scripts/overlay_setup.py` and `capture_shapes.py`

Reversible-overlay tooling (never edits site-packages). `overlay_setup.py` builds a compounding
`sitecustomize`/monkeypatch overlay — subcommands `add-module` and `add-rebind`. `capture_shapes.py`
hooks a live server callable through the overlay to capture real serving shapes + a reference I/O oracle
(`reference_io.pt` + `meta.json`); imported through the overlay, not a standalone CLI.

### `kernel_workflow/scripts/gpu_lock.sh`

GPU lock + per-workspace build isolation. `flock` per GPU id (safe GPU sharing), per-workspace
`TORCH_EXTENSIONS_DIR`, pins `PYTORCH_ROCM_ARCH` to the local gfx, reaps orphaned enumerator processes.

```bash
bash gpu_lock.sh <gpu_id> <command...>
```

### `kernel_workflow/scripts/profile_kernel.sh`

Thin profiling wrapper: warmup + `gpu_lock` + auto-detect best profiler + run + dump RAW output.
Output entry point: `<output_dir>/profile_report.txt`.

```bash
bash profile_kernel.sh <gpu_id> <benchmark_cmd> <output_dir>
```

Env: `PROFILER_PRIORITY` (default `rocprof-compute omniperf rocprofv3 rocprof metrix`), `WARMUP_RUNS` (3),
`RPC_PROFILE_ARGS`, `RPV3_TRACE_ARGS`, `RPROF_ARGS`, `METRIX_ARGS`.

## External-orchestrator contract (`interface/run_e2e.py`)

The only surface an external orchestrator (for example, Hyperloom) touches —
wraps `e2e_workflow.js` arg names behind one command and two JSON files. The
current result schema is version 2; handoff schema 1 remains accepted, while
schema 2 carries same-config alignment metadata.

```bash
python interface/run_e2e.py <handoff.json> <result.json> [--dry-run]
```

Stable `handoff.json` fields: `model_path`, `framework` (→ `backend`), `tp`, `gpu_ids`,
`workload{isl, osl, conc}`, `accepted_flags` / `env`, `exp_root`, `bench_client`, `bench_protocol`,
`inferencex_path`, `raw_baseline_tput`, `orchestrator_best_tput_same_config`.

Env knobs: `GEAK_CLAUDE_MODEL` (`claude-opus-4-8`), `GEAK_CLAUDE_EFFORT` (`ultracode`),
`GEAK_E2E_TIMEOUT_S` (`43200` = 12h), `GEAK_ROOT`, `GEAK_EVAL_DIR`, `INFERENCEX_PATH`.
See [External orchestrator contract](./run-e2e-contract.md) for the full contract.

## Output artifacts

### e2e — `<exp_root>/e2e_<model>_<timestamp>/`

```text
env_report.{md,json}                     # preflight capability report (every phase routes on it)
baseline/bench_summary.json              # TRUE baseline throughput
config/baseline_flags.json               # baseline launch flags
config/sweep_results.json                # Tier-0 sweep results
profile/round_*/profile_topN.{json,md}   # standardized Top-N per round
strategy.md, insight_log.md
kernels/<name>_task/{kernel_src, reference_io.pt, unittest.py, meta.json}   # extracted tasks
kernels/_exp/…                           # recursive single-kernel runs (each verified)
overlay/…                                # candidate + accepted reversible overlays
final/{overlay, final_patch.diff, final_launch.sh}   # deliverable bundle
final_report.md                          # complete run report (Phases tree + artifacts tree)
architect_report.md                      # official verified throughput result
director_e2e_validation.json             # independent re-measurement
```

### single kernel — `<exp_root>/team_<kernel>_<timestamp>/<kernel>/`

```text
COMMANDMENT.md                           # measurement contract
baseline_timing.json, baseline_metrics.json
analysis.json, codebase_context.md, roadmap.md, profiling_summary.md
round_N/engineer_i/{worker_result.json, report.md, best_patch.diff}
round_N/integrate/
insight_log.md, current_best.diff
tech_lead_report.md                      # run report
final_patch.diff                         # winning patch
optimized/                               # applied optimized source
director_validation.json                 # independent re-measurement
```

## Related topics

- [Run a workflow](../how-to/run-agent.md): install and first run.
- [Install GEAK](../install/install.md): prerequisites and environment setup.
- [Compatibility matrix](../compatibility.md): verified hardware/software combinations.
