---
myst:
    html_meta:
        "description": "External integration contract for GEAK v4: the run_e2e.py command, handoff.json and result.json schemas, TraceLens autodiscovery, and measurement alignment with Hyperloom."
        "keywords": "GEAK, run_e2e, external orchestrator, Hyperloom, handoff.json, result.json, integration contract, kernel journey"
---

# GEAK external orchestrator contract (`interface/run_e2e.py`)

`interface/` is the only surface an external orchestrator (for example, Hyperloom)
touches. Everything volatile about the e2e workflow (the `e2e_workflow.js` arg
names, the Claude Code `Workflow` invocation, the `--effort ultracode`
requirement, the SDK-vs-CLI choice) is hidden behind one command and two JSON
files. The result schema is versioned so callers can distinguish contract
changes while the workflow evolves internally.

## Command

Run the external orchestrator entry point as follows.

```bash
python interface/run_e2e.py <handoff.json> <result.json> [--timeout-s N] [--dry-run] \
    [--kernel-fusion | --no-kernel-fusion]
```

- Exit code `0`: `result.json.status` is `ok`, `no_gain` or `timeout`.
- Exit code `1`: a crash; `result.json.status == "error"` with an `error` field.
- Exit code `2`: bad usage or unreadable handoff.
- `--dry-run`: print the mapped `e2e_workflow.js` args and the prompt and
  exit `0` (no GPU work). Use this to validate the mapping in CI. The plan also
  carries a `kernel_fusion` entry: the KernelFusion stage's script and args, or
  `{"enabled": false}` when the stage is off.
- `--kernel-fusion` / `--no-kernel-fusion`: switch the KernelFusion stage on or
  off for this run; overrides the handoff and `GEAK_KERNEL_FUSION` (see
  [KernelFusion stage](#kernelfusion-stage)).

Discovery: the installer should export `GEAK_E2E_RUNNER` pointing at this
file (`$GEAK_ROOT/interface/run_e2e.py`) so the caller has a single
hard-coded handle.

E2E KB recall can use an independent GEAK policy: set
`GEAK_E2E_KB_METRIC_BASIS=e2e_norm_intvty_p50` in the parent process environment,
or `e2e_kb_metric_basis` in the handoff (higher precedence). Explicit policies
exclude other and unlabelled metric bases without changing the benchmark axis,
acceptance rule, or identity ladder. See the
[recall policy contract](../../interface/run_e2e.md#e2e-kb-recall-policy) for
defaults, diagnostics, and overlay compatibility.

The fast-path artifacts live under `<exp_root>/geak_e2e_moe_int4/`
(`baseline/`, `validation/final/`, `final/` bundle, `director_e2e_validation.json`).

## `handoff.json` (caller → workflow)

Pass the following fields in the handoff file.

```jsonc
{
  "schema_version": 2,
  "model_path": "/models/Qwen-Qwen3.5-27B",
  "framework": "sglang",                 // -> backend (sglang|vllm)
  "gpu_type": "MI300X",
  "tp": 8,                               // serving tensor-parallel size (honoured, no TP=1 lock)
  "gpu_ids": "0,1,2,3,4,5,6,7",          // optional; default 0..tp-1
  "workload": { "isl": 1024, "osl": 1024, "conc": 64 },
  "accepted_flags": "--attention-backend triton",  // best config from the caller's search
  "accepted_env": "SGLANG_USE_AITER=1",
  "launch_recipe": "/path/baseline_config.with_envs.yaml",  // optional launch script/recipe
  "raw_baseline_tput": 1485.4,           // caller's pre-change session baseline (audit reference)
  "orchestrator_best_tput_same_config": 1550.8, // caller best measured with accepted_flags/env
  "exp_root": "/work/experiment/geak",   // basename MUST be `geak`; the timestamped run dir is created here
  "bench_client": "auto",                // auto|inferencex|native
  "inferencex_path": "/opt/InferenceX",  // optional; else taken from $INFERENCEX_PATH
  "bench_protocol": {                    // optional; caller's measurement calibration
    "random_range_ratio": 0,             //   fixed(0) vs variable(>0) sequence lengths
    "num_prompts": 192,
    "num_warmups": 8,
    "seed": 0
  }
}
```

Required: `model_path`, `exp_root`. Everything else has a default.

`bench_protocol` is optional and partial-friendly: only the keys present are
applied. Omit it entirely (standalone GEAK, no external orchestrator) and
`bench_e2e.sh` keeps its own defaults unchanged. When the caller supplies it,
those values are the exact knobs the caller's official baseline was measured
with — forwarding them is what makes the workflow's numbers cross-harness
comparable. The `random_range_ratio` convention is `0` for fixed-length and
`>0` for variable-length (lengths sampled in `[(1-ratio)*len, (1+ratio)*len]`);
a silent mismatch between the caller's value and the standalone default causes a
roughly 10-15% calibration gap. Both default to `0` (fixed) so the standalone
and forwarded calibrations agree unless the caller requests variable lengths.

### How handoff maps to the workflow

The mapping is owned by `run_e2e.py:map_args`.

| Handoff field | `e2e_workflow.js` arg | Note |
|---|---|---|
| `model_path` | `model_path` | Required |
| `framework` | `backend` | `sglang` or `vllm` |
| `tp` | `tp` | Serving tensor-parallel (threaded to bench `TP`) |
| `gpu_ids` or `tp` | `gpu_ids` | Defaults to `0..tp-1` |
| `workload.{isl,osl,conc}` | `isl`, `osl`, `conc` | Profile and bench workload |
| `accepted_flags` | `initial_extra_server_args` | Seeds the baseline from caller best config |
| `accepted_env` | `initial_extra_env` | Seeds baseline env |
| `launch_recipe` | `launch_script` | Optional |
| `raw_baseline_tput` | Result audit metadata | Pre-change session baseline; never used as the measurement-alignment signal |
| `orchestrator_best_tput_same_config` | Result alignment metadata | Caller throughput on the accepted config GEAK uses for its baseline |
| `exp_root` | `exp_root` | Run dir root |
| (derived from `exp_root`) | `tracelens` | Auto-discovered upstream TraceLens artifacts; only non-null paths forwarded; key omitted entirely when none found |
| `bench_client`, `inferencex_path` | env `BENCH_CLIENT`, `INFERENCEX_PATH` | Exported so every `bench_e2e.sh` call inherits it (not a JS arg) |
| `bench_protocol.*` | env `RANDOM_RANGE_RATIO`, `NUM_PROMPTS`, `NUM_WARMUPS`, `SEED` | `run_e2e.py:apply_bench_protocol` exports only the provided keys; absent keys keep `bench_e2e.sh` defaults |
| — | `config_tune="false"` | Caller already did config search; not repeated |
| — | `apply_to_original="true"` | `final/final_launch.sh` and overlay are emitted for sweep reuse |

### KernelFusion stage

Before the main workflow, `run_e2e.py` runs the KernelFusion (KF) stage
(`e2e_workflow/kernel_fusion/`): it captures a serving trace, picks fusion
patterns with the `fusion-analysis` skill, writes and unit-tests one fused
kernel per pattern, and applies the passing kernels as one overlay gated by one
e2e A/B (gsm8k accuracy gate, then throughput). When that A/B accepts the
overlay, it becomes the main workflow's `initial_overlay_pythonpath`, so the main
workflow measures its own baseline on the fused stack. A stage that fails, finds
nothing or is switched off leaves the main workflow's args unchanged.

The stage is on by default. Precedence, highest first: the CLI flags, the
`GEAK_KERNEL_FUSION=0|1` environment variable, then the handoff:

```jsonc
{
  "kernel_fusion": false                      // off
  // or a settings object (on unless "enabled": false):
  "kernel_fusion": {
    "prompt": "/path/deployment_notes.txt",   // run-specific prompt(s), comma-separated
    "budget_s": 14400,                        // default: min(12 h, 1/3 of the run)
    "steps": "trace,topk,author,apply",
    "task": "operator constraints for every stage agent"
  }
}
```

The stage budget is taken from the run's timeout; the main workflow gets the
rest. Processes the stage leaves running are stopped before the main workflow
starts. Every settings key also has a `GEAK_KF_*` environment variable; the full
list is in `e2e_workflow/kernel_fusion/README.md`.

### TraceLens prior autodiscovery

Owned by `run_e2e.py:resolve_tracelens_report`. An upstream orchestrator might have
already profiled the same baseline workload with TraceLens and dropped its
artifacts beside the handoff's `geak` directory (under the experiment root, which
is the parent of `geak`). `map_args` resolves them by glob and forwards the
non-null paths to the workflow as `args.tracelens`.

| Key | Glob (relative to the experiment root) | What it is |
|---|---|---|
| `analysis_md` | `kernel-agent/**/tracelens/analysis.md` | Human TraceLens hot-kernel report |
| `kernel_candidates_json` | `kernel-agent/**/kernel_candidates.json` | Machine-readable hot-kernel list |
| `tracelens_report_json` | `kernel-agent/**/tracelens/tracelens_report.json` | Full TraceLens report |
| `trace_file` | `runs/roofline/**/torch_trace` | Roofline torch-trace directory (per-TP-rank `*.pt.trace.json.gz`) |

Resolution prefers the parent of the `geak` segment in `exp_root`; if that path
is not present it falls back to the on-disk grandparent of the handoff file.

When `analysis_md` exists, the Profiler skips its own warm-server trace
collection and builds the standardized Top-N from the TraceLens artifacts. When
`trace_file` also exists, an additional `parse_profile.py` pass recovers real
kernel symbols and reliable per-launch shapes. The System Architect uses
`kernel_candidates.json` as an advisory routing prior without overriding the
measured `%gpu`. A run without TraceLens artifacts is byte-identical to a run with
them: the feature is entirely additive.

## `result.json` (workflow → caller)

The workflow writes the following fields to the result file.

```jsonc
{
  "schema_version": 2,
  "status": "ok | no_gain | error",
  "eval_dir": "/work/experiment/geak/e2e_<model>_<ts>",
  "baseline_throughput_tok_s": 1485.4,
  "final_throughput_tok_s": 1551.4,
  "throughput_speedup": 1.044,
  "output_parity": "pass | fail | n/a | unknown",
  "ttft_ms": 3598.0,
  "tpot_ms": 39.5,
  "final_launch_script": ".../final/final_launch.sh",
  "bench_script": ".../bench_e2e.sh",
  "final_patch": ".../final/final_patch.diff",
  "final_overlay": ".../final/overlay",
  "metric_basis": "aggregate_output_tok_s",
  "bench_client": "inferencex",
  "validated_regimes": [ { "isl": 1024, "osl": 1024, "conc": 64 } ],
  "accepted_kernels": [ /* per-kernel detail */ ],
  "accepted_heads": [ /* head GEMM/attention winners */ ],
  "accepted_config": { "flags": "...", "env": "..." },
  "baseline_basis": {
    "geak_measured_baseline_tok_s": 1551.4,
    "orchestrator_baseline_tok_s": 1485.4,
    "raw_session_baseline_divergence_pct": 4.44,
    "orchestrator_best_tput_same_config": 1550.8,
    "current_best_same_config_divergence_pct": 0.04,
    "measurement_divergence_pct": 0.04
  },
  "baseline_alignment": {
    "status": "aligned | warning | unavailable",
    "primary_metric": "current_best_same_config_divergence_pct",
    "divergence_pct": 0.04,
    "warning_threshold_pct": 3.0,
    "raw_session_divergence_is_measurement_signal": false
  },
  "report_path": ".../final_report.md",
  "kernel_journey_path": ".../kernel_journey.json",
  "recovered_from_disk": true,
  "kernel_fusion": {                     // present only when the KF stage ran
    "status": "accepted | no_gain | analysis_only | error",
    "kf_dir": "/work/experiment/geak/kf_<model>_<ts>",
    "throughput_speedup": 1.018,         // the stage's own A/B
    "final_overlay": ".../overlay/cand_kf_stack",
    "accepted": [ /* fused kernels */ ],
    "rejected": [ /* with stage and reason */ ],
    "main_timeout_s": 28800,             // budget left for the main workflow
    "combined_speedup_vs_original": 1.05 // stage A/B x main-workflow A/B
  }
}
```

Schema v2 renames `baseline_divergence_pct` to
`raw_session_baseline_divergence_pct`. The raw-session metric is audit-only
because it includes accepted upstream configuration gains.

`kernel_fusion.combined_speedup_vs_original` multiplies the stage's A/B
speedup by the main workflow's, which is measured on the fused stack.

`current_best_same_config_divergence_pct` is the primary cross-harness
alignment metric. `measurement_divergence_pct` remains an exact compatibility
alias for existing callers. If `orchestrator_best_tput_same_config` is absent,
the same-config fields are `null` and `baseline_alignment.status` is
`unavailable`; GEAK never treats raw-session divergence as measurement drift.

### When there is no result to report

When no workflow return exists and none can be rebuilt from disk (see Handoff
resilience below), `result.json` reports the failure instead:

```jsonc
{
  "schema_version": 2,
  "status": "timeout | error",
  "error_class": "timeout | llm_unreachable | workflow_parse_error | cli_failed | sdk_import_failed | interrupted | normalize_failed | runner_error",
  "error": "no workflow return within the 32400s budget (furthest phase on disk: search)",
  "phase_reached": "setup | search | finalize | validate"
}
```

- `llm_unreachable`: Claude Code could not reach the model API (connection
  refused or dropped, DNS, no route, proxy or TLS failure). `error` keeps the
  original message, which contains Claude Code's `API Error: ...` reply.
- `phase_reached` is the furthest phase the eval dir shows: `setup` until
  Setup's baseline summary exists, `search` until the final phase fences the
  GPUs (`.final_phase_fence`), then the fenced stage. It is absent when the eval
  dir does not exist.

## Handoff resilience

The workflow return (the JSON object carrying `eval_dir` and `accepted_*`) is the
only value scraped from the agent transcript. A failed scrape previously discarded
the entire run as `workflow_parse_error` even when every artifact
(`director_e2e_validation.json`, the `final/` bundle, the measured gain) was on
disk. `run_e2e.py` removes that fragility in four layers:

1. **Robust capture** — the SDK path accumulates the full transcript (every text
   fragment from every message, including tool-result blocks), not just the last
   assistant text.
2. **Robust extraction** — the parser scans the whole transcript for the last JSON
   object carrying `eval_dir` (tolerates compact single-line, fenced, pretty-printed
   multi-line, and trailing prose).
3. **On-disk sentinel** — on success the parsed return is persisted to
   `<eval_dir>/workflow_return.json`, so any later read never re-scrapes.
4. **Disk recovery** — if capture or extraction still fails (or the run timed out
   after the measured leg), the return is rebuilt from on-disk artifacts:
   `workflow_return.json` if present, else reconstructed from
   `director_e2e_validation.json` with accepted-kernel names recovered from the
   stable `overlay/cand_*` layout. Recovery returns nothing only when no completed
   `eval_dir` exists. Recovered runs set `result.recovered_from_disk = true`.

## `kernel_journey.json`

`run_e2e.py` emits `<eval_dir>/kernel_journey.json` (path echoed in
`result.kernel_journey_path`) so GEAK-authored kernels are visible in the
orchestrator's kernel-journey view. Its per-kernel sub-objects are shaped exactly
as the orchestrator recorder's `record_kernel_{dispatch,backend_result,e2e}`
inputs, so the orchestrator replays them verbatim.

```jsonc
{
  "schema_version": 1,
  "producer": "kernel-agent",
  "eval_dir": ".../e2e_<model>_<ts>",
  "versions": { "geak": { "tool": "geak", "root_dir": "...", "commit": "<sha>", "version": "<sha>" } },
  "kernels": [
    {
      "kernel_id": "int4_w4a16_fused_moe_grouped_gemm",
      "name": "int4_w4a16_fused_moe_grouped_gemm",
      "gpu_pct": 0.57,
      "dispatch":       { "dispatched": true, "backends": ["geak"], "skip_reason": "", "task_group": null },
      "backend_result": { "kernel_id": "...", "run_id": "...", "attempts": [ { "backend": "geak", "attempt_id": "...", "status": "succeeded", "decision": "KEEP", "micro_speedup": 1.6316, "compile_passed": true, "correctness_passed": true, "optimized_path": ".../final_patch.diff", "error": null, "error_type": null } ], "verification": { "micro_speedup": 1.6316, "best_attempt_id": "...", "best_backend": "geak" }, "metadata": { "root_dir": "...", "version": "<sha>" } },
      "e2e":            { "integrated": true, "e2e_gain_pct": 12.21, "validated": true, "decision": "KEEP", "patch_path": ".../final_patch.diff", "target_file": null, "extra_server_args": "--kv-cache-dtype fp8" }
    }
  ]
}
```

On the recovery path, per-kernel `micro_speedup` might be `null` (it only existed
in the scraped return) — it is never fabricated; but when exactly one kernel was
accepted it is credited with the whole measured e2e delta.

## Reusing deliverables for a workload sweep

`final_launch.sh` is self-contained (it bakes `OVERLAY_PYTHONPATH`, accepted
flags and env, `BACKEND`, `TP`) and delegates server launch and bench to
`bench_e2e.sh`. To sweep workloads on the optimized server without rebuilding
the overlay:

1. Start the optimized server once using `final_launch.sh`.
2. For each `(CONC, ISL, OSL)` point, call `bench_e2e.sh` with
   `REUSE_SERVER=1 CONC=.. ISL=.. OSL=..` against the warm server.
3. For any point outside `validated_regimes`, redo a greedy output parity probe
   (the kernels were only validated at the single handoff workload point).

## Measurement alignment

The workflow measures on the same calibration as the caller's official baseline so
`final` and sweep curves are comparable to the caller's raw baseline.

| Knob | Aligned value |
|---|---|
| Primary metric | Aggregate `output_throughput` (output tok/s, not per-GPU) |
| Latency | `ttft_ms` and `tpot_ms` median |
| Dataset | `random`; `random_range_ratio` from `handoff.bench_protocol.random_range_ratio` (caller-driven: `0`=fixed, `>0`=variable), else standalone default `0` (fixed) |
| Workload | Same `ISL/OSL/CONC`; `NUM_PROMPTS` from `bench_protocol.num_prompts`, else `max(CONC*factor, CONC)` |
| Warmups | `NUM_WARMUPS` from `bench_protocol.num_warmups`, else `min(CONC, 8)` |
| Seed | `SEED` from `bench_protocol.seed`, else fixed `0` |
| TP | Same tensor-parallel (TP) size as the caller (no TP=1 lock) |
| Parity | Greedy output diff vs baseline at temperature 0 with fixed seed |
| Bench client | `BENCH_CLIENT=inferencex` uses the same `benchmark_serving.py` as Hyperloom |

The serving stack is always launched by the backend adapter
(`adapters/sglang.sh` or `vllm.sh`). The bench client is selected independently
by `BENCH_CLIENT`:

- `native` (default standalone): each backend's built-in bench
  (`sglang.bench_serving` or vLLM). A small cross-harness difference remains.
- `inferencex`: `adapters/clients/inferencex.sh` redefines `adapter_bench` to
  call Hyperloom's own `benchmark_serving.py`, which is byte-for-byte the same
  client Hyperloom uses.

`run_e2e.py` resolves `handoff.bench_client` (`auto` maps to `inferencex` when
an InferenceX checkout is discoverable using `INFERENCEX_PATH`, else `native`)
and exports `BENCH_CLIENT` and `INFERENCEX_PATH` so every `bench_e2e.sh` the
agents run inherits them.

## Related topics

- [API reference](./api-reference.md): Workflow arguments and run artifacts.
- [Run a workflow](../how-to/run-agent.md): natural-language invocation.
