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
files. As long as `schema_version` stays `1`, the caller never changes when
the workflow evolves internally.

## Command

Run the external orchestrator entry point as follows.

```bash
python interface/run_e2e.py <handoff.json> <result.json> [--dry-run]
```

- Exit code `0`: `result.json.status` is `ok` or `no_gain`.
- Exit code `1`: a crash; `result.json.status == "error"` with an `error` field.
- Exit code `2`: bad usage or unreadable handoff.
- `--dry-run`: print the mapped `e2e_workflow.js` args and the prompt and
  exit `0` (no GPU work). Use this to validate the mapping in CI.

Discovery: the installer should export `GEAK_E2E_RUNNER` pointing at this
file (`$GEAK_ROOT/interface/run_e2e.py`) so the caller has a single
hard-coded handle.

The fast-path artifacts live under `<exp_root>/geak_e2e_moe_int4/`
(`baseline/`, `validation/final/`, `final/` bundle, `director_e2e_validation.json`).

## `handoff.json` (caller → workflow)

Pass the following fields in the handoff file.

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

| handoff field | `e2e_workflow.js` arg | note |
|---|---|---|
| `model_path` | `model_path` | required |
| `framework` | `backend` | `sglang` or `vllm` |
| `tp` | `tp` | serving tensor-parallel (threaded to bench `TP`) |
| `gpu_ids` or `tp` | `gpu_ids` | defaults to `0..tp-1` |
| `workload.{isl,osl,conc}` | `isl`, `osl`, `conc` | profile and bench workload |
| `accepted_flags` | `initial_extra_server_args` | seeds the baseline from caller best config |
| `accepted_env` | `initial_extra_env` | seeds baseline env |
| `launch_recipe` | `launch_script` | optional |
| `exp_root` | `exp_root` | run dir root |
| (derived from `exp_root`) | `tracelens` | auto-discovered upstream TraceLens artifacts; only non-null paths forwarded; key omitted entirely when none found |
| `bench_client`, `inferencex_path` | env `BENCH_CLIENT`, `INFERENCEX_PATH` | exported so every `bench_e2e.sh` call inherits it (not a JS arg) |
| `bench_protocol.*` | env `RANDOM_RANGE_RATIO`, `NUM_PROMPTS`, `NUM_WARMUPS`, `SEED` | `run_e2e.py:apply_bench_protocol` exports only the provided keys; absent keys keep `bench_e2e.sh` defaults |
| — | `config_tune="false"` | caller already did config search; not repeated |
| — | `apply_to_original="true"` | `final/final_launch.sh` and overlay are emitted for sweep reuse |

### TraceLens prior autodiscovery

Owned by `run_e2e.py:resolve_tracelens_report`. An upstream orchestrator might have
already profiled the same baseline workload with TraceLens and dropped its
artifacts beside the handoff's `geak` directory (under the experiment root, which
is the parent of `geak`). `map_args` resolves them by glob and forwards the
non-null paths to the workflow as `args.tracelens`.

| key | glob (relative to the experiment root) | what it is |
|---|---|---|
| `analysis_md` | `kernel-agent/**/tracelens/analysis.md` | human TraceLens hot-kernel report |
| `kernel_candidates_json` | `kernel-agent/**/kernel_candidates.json` | machine-readable hot-kernel list |
| `tracelens_report_json` | `kernel-agent/**/tracelens/tracelens_report.json` | full TraceLens report |
| `trace_file` | `runs/roofline/**/torch_trace` | roofline torch-trace directory (per-TP-rank `*.pt.trace.json.gz`) |

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
  "schema_version": 1,
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
  "report_path": ".../final_report.md",
  "kernel_journey_path": ".../kernel_journey.json",
  "recovered_from_disk": true
}
```

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

| knob | aligned value |
|---|---|
| primary metric | aggregate `output_throughput` (output tok/s, not per-GPU) |
| latency | `ttft_ms` and `tpot_ms` median |
| dataset | `random`; `random_range_ratio` from `handoff.bench_protocol.random_range_ratio` (caller-driven: `0`=fixed, `>0`=variable), else standalone default `0` (fixed) |
| workload | same `ISL/OSL/CONC`; `NUM_PROMPTS` from `bench_protocol.num_prompts`, else `max(CONC*factor, CONC)` |
| warmups | `NUM_WARMUPS` from `bench_protocol.num_warmups`, else `min(CONC, 8)` |
| seed | `SEED` from `bench_protocol.seed`, else fixed `0` |
| TP | same tensor-parallel (TP) size as the caller (no TP=1 lock) |
| parity | greedy output diff vs baseline at temperature 0 with fixed seed |
| bench client | `BENCH_CLIENT=inferencex` uses the same `benchmark_serving.py` as Hyperloom |

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
