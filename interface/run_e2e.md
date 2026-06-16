# `interface/run_e2e.py` — external integration contract

`interface/` is the **only** surface an external orchestrator (e.g. Hyperloom)
touches. Everything volatile about the e2e workflow (the `e2e_workflow.js` arg
names, the Claude Code `Workflow` invocation, the `--effort ultracode`
requirement, the SDK-vs-CLI choice) is hidden behind one command and two JSON
files. As long as `schema_version` stays `1`, the caller never changes when
the workflow evolves internally.

## Command

```bash
python interface/run_e2e.py <handoff.json> <result.json> [--dry-run]
```

* Exit code `0` → `result.json.status` is `ok` or `no_gain`.
* Exit code `1` → a crash; `result.json.status == "error"` with an `error` field.
* Exit code `2` → bad usage / unreadable handoff.
* `--dry-run` → print the mapped `e2e_workflow.js` args + the prompt and
  exit `0` (no GPU work). Use this to validate the mapping in CI.

Discovery: the installer should export `PERFSKILLS_E2E_RUNNER` pointing at this
file (`$PERFSKILLS_ROOT/interface/run_e2e.py`) so the caller has a single
hard-coded handle.

## `handoff.json` (caller → workflow)

```jsonc
{
  "schema_version": 1,
  "model_path": "/models/Qwen-Qwen3.5-27B",
  "framework": "sglang",                 // -> backend (sglang|vllm)
  "gpu_type": "MI300X",
  "tp": 8,                               // serving tensor-parallel size (honoured, no TP=1 lock)
  "gpu_ids": "0,1,2,3,4,5,6,7",          // optional; default 0..tp-1
  "workload": { "isl": 1024, "osl": 1024, "conc": 64 },
  "accepted_flags": "--attention-backend triton",  // best config from the caller's search
  "accepted_env": "SGLANG_USE_AITER=1",
  "launch_recipe": "/path/baseline_config.with_envs.yaml",  // optional launch script/recipe
  "raw_baseline_tput": 1485.4,           // caller's official raw baseline (carried for reference)
  "exp_root": "/work/perfskills_exp",    // where the timestamped run dir is created
  "bench_client": "auto",                // auto|inferencex|native — see口径 alignment below
  "inferencex_path": "/opt/InferenceX"   // optional; else taken from $INFERENCEX_PATH
}
```

Required: `model_path`, `exp_root`. Everything else has a default.

### How handoff maps to the workflow (owned by `run_e2e.py:map_args`)

| handoff field | `e2e_workflow.js` arg | note |
|---|---|---|
| `model_path` | `model_path` | required |
| `framework` | `backend` | `sglang` \| `vllm` |
| `tp` | `tp` | serving tensor-parallel (threaded to bench `TP`) |
| `gpu_ids` / `tp` | `gpu_ids` | defaults to `0..tp-1` |
| `workload.{isl,osl,conc}` | `isl` / `osl` / `conc` | profile + bench workload |
| `accepted_flags` | `initial_extra_server_args` | seeds the baseline = caller best config |
| `accepted_env` | `initial_extra_env` | seeds baseline env |
| `launch_recipe` | `launch_script` | optional |
| `exp_root` | `exp_root` | run dir root |
| `bench_client` / `inferencex_path` | env `BENCH_CLIENT` + `INFERENCEX_PATH` | exported so every `bench_e2e.sh` call inherits it (not a JS arg) |
| — | `config_tune="false"` | caller already did config search; never double-run |
| — | `apply_to_original="true"` | so `final/final_launch.sh` + overlay are emitted for sweep reuse |

## `result.json` (workflow → caller)

```jsonc
{
  "schema_version": 1,
  "status": "ok | no_gain | error",
  "eval_dir": "/work/perfskills_exp/e2e_<model>_<ts>",
  "baseline_throughput_tok_s": 1485.4,   // baseline (= caller best config)
  "final_throughput_tok_s": 1551.4,
  "throughput_speedup": 1.044,
  "output_parity": "pass | fail | n/a | unknown",
  "ttft_ms": 3598.0,                     // median, aligned with caller's ttft
  "tpot_ms": 39.5,                       // median, aligned with caller's tpot
  "final_launch_script": ".../final/final_launch.sh",  // self-contained: overlay/flags/env baked in
  "bench_script": ".../bench_e2e.sh",    // supports REUSE_SERVER=1 + CONC/ISL/OSL
  "final_patch": ".../final/final_patch.diff",
  "final_overlay": ".../final/overlay",
  "metric_basis": "aggregate_output_tok_s",   // NOT per-GPU; matches Magpie output_throughput
  "bench_client": "inferencex",               // inferencex => identical client to caller; else native
  "validated_regimes": [ { "isl": 1024, "osl": 1024, "conc": 64 } ],  // redo parity outside these
  "accepted_kernels": [ /* what was optimized + how (per-kernel) */ ],
  "accepted_heads": [ /* head GEMM/attn winners */ ],
  "accepted_config": { "flags": "...", "env": "..." },
  "report_path": ".../final_report.md"   // human report: per-kernel optimizations, changed params, TTFT/TPOT
}
```

## Reusing the deliverables for a workload sweep

`final_launch.sh` is self-contained (it bakes `OVERLAY_PYTHONPATH`, accepted
flags/env, `BACKEND`, `TP`) and delegates server launch + bench to
`bench_e2e.sh`. To sweep workloads on the optimized server without rebuilding
the overlay:

1. Start the optimized server once via `final_launch.sh`.
2. For each `(CONC, ISL, OSL)` point, call `bench_e2e.sh` with
   `REUSE_SERVER=1 CONC=.. ISL=.. OSL=..` against the warm server.
3. For any point outside `validated_regimes`, redo a greedy/temp=0 parity probe
   (the kernels were only validated at the single handoff workload point).

## Measurement-口径 alignment (vs Hyperloom Magpie)

The workflow must measure on the **same口径** as the caller's official baseline so
`final` and sweep curves are comparable to the caller's raw baseline:

| knob | aligned value |
|---|---|
| primary metric | aggregate `output_throughput` (output tok/s, **not** per-GPU) |
| latency | `ttft_ms` / `tpot_ms` median |
| dataset | `random`, `random-range-ratio 1.0` (fixed lengths) |
| workload | same `ISL/OSL/CONC`; `NUM_PROMPTS = max(CONC*factor, CONC)` |
| warmups | `NUM_WARMUPS = min(CONC, 8)` (matches Hyperloom's materialize default) |
| seed | fixed `SEED` |
| TP | same tensor-parallel as the caller (no TP=1 lock) |
| parity | greedy / temp=0 fixed-seed output diff vs baseline |
| **bench client** | `BENCH_CLIENT=inferencex` → the **exact same** `benchmark_serving.py` as Hyperloom |

### Bench-CLIENT adapter (closes the last口径 residual)

The serving stack is always launched by the **backend** adapter
(`adapters/sglang.sh` / `vllm.sh`). The **client** that drives the timed bench is
selected independently by `BENCH_CLIENT`:

* `native` (default standalone) — each backend's built-in bench
  (`sglang.bench_serving` / vLLM). Small cross-harness差异 may remain.
* `inferencex` — `adapters/clients/inferencex.sh` redefines `adapter_bench` to
  call **Hyperloom/Magpie's own** `InferenceX/utils/bench_serving/benchmark_serving.py`
  (`--backend vllm --dataset-name random --request-rate inf --ignore-eos
  --num-warmups $NUM_WARMUPS --percentile-metrics ttft,tpot,itl,e2el`). This is
  byte-for-byte the same client Hyperloom uses, so the only remaining difference
  is `REPEATS`-median vs single-run — not the client.

`run_e2e.py` resolves `handoff.bench_client` (`auto` → `inferencex` when an
InferenceX checkout is discoverable via `INFERENCEX_PATH`, else `native`) and
exports `BENCH_CLIENT` + `INFERENCEX_PATH` so every `bench_e2e.sh` the agents run
inherits it. The profile round (server-side trace) always delegates back to the
backend's native bench. The chosen client is echoed in `result.bench_client`, and
the sweep reuse path carries it forward so sweep points use the same client.
