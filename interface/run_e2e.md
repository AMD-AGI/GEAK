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

Discovery: the installer should export `GEAK_E2E_RUNNER` pointing at this
file (`$GEAK_ROOT/interface/run_e2e.py`) so the caller has a single
hard-coded handle.

The fast-path artifacts live under `<exp_root>/geak_e2e_moe_int4/`
(`baseline/`, `validation/final/`, `final/` bundle, `director_e2e_validation.json`).

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
  "exp_root": "/work/experiment/geak",   // basename MUST be `geak`; the timestamped run dir is created here
  "bench_client": "auto",                // auto|inferencex|native — see口径 alignment below
  "inferencex_path": "/opt/InferenceX",  // optional; else taken from $INFERENCEX_PATH
  "bench_protocol": {                    // optional; caller's measurement 口径 (see below)
    "random_range_ratio": 0,             //   fixed(0) vs variable(>0) sequence lengths
    "num_prompts": 192,
    "num_warmups": 8,
    "seed": 0
  }
}
```

Required: `model_path`, `exp_root`. Everything else has a default.

`bench_protocol` is optional and **partial-friendly**: only the keys present are
applied. Omit it entirely (standalone GEAK, no external orchestrator) and
`bench_e2e.sh` keeps its own defaults unchanged. When the caller (Hyperloom)
supplies it, those values are the EXACT knobs the caller's official baseline was
measured with — forwarding them is what makes the workflow's numbers
cross-harness comparable. The `random_range_ratio` convention is `0`=fixed-length,
`>0`=variable-length (lengths sampled in `[(1-ratio)*len, (1+ratio)*len]`); a
silent mismatch between the caller's value and the standalone default is otherwise
a ~10-15% 口径 gap. Both default to `0` (fixed) so the standalone and forwarded
口径 agree unless the caller explicitly requests variable lengths.

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
| (derived from `exp_root`) | `tracelens` | auto-discovered upstream TraceLens / kernel-agent artifacts (see below); only non-null paths forwarded; key omitted entirely when none found |
| `bench_client` / `inferencex_path` | env `BENCH_CLIENT` + `INFERENCEX_PATH` | exported so every `bench_e2e.sh` call inherits it (not a JS arg) |
| `bench_protocol.{random_range_ratio,num_prompts,num_warmups,seed}` | env `RANDOM_RANGE_RATIO` / `NUM_PROMPTS` / `NUM_WARMUPS` / `SEED` | `run_e2e.py:apply_bench_protocol` exports ONLY the provided keys, overriding `bench_e2e.sh` standalone defaults; absent ⇒ defaults kept (not a JS arg) |
| — | `config_tune="false"` | caller already did config search; never double-run |
| — | `apply_to_original="true"` | so `final/final_launch.sh` + overlay are emitted for sweep reuse |

### TraceLens prior auto-discovery (owned by `run_e2e.py:resolve_tracelens_report`)

An upstream orchestrator may have already profiled the SAME baseline workload with
TraceLens and dropped its artifacts beside the handoff's `geak` dir (i.e.
under the experiment root = the parent of `geak`). `map_args` resolves them
by glob (each `**` is a randomly-named nested dir) and forwards the **non-null**
paths to the workflow as `args.tracelens`:

| key | glob (relative to the experiment root) | what it is |
|---|---|---|
| `analysis_md` | `kernel-agent/**/tracelens/analysis.md` | human TraceLens hot-kernel report |
| `kernel_candidates_json` | `kernel-agent/**/kernel_candidates.json` | machine-readable hot-kernel list (name/category/source_file/launcher/shapes/bound_type/…) |
| `tracelens_report_json` | `kernel-agent/**/tracelens/tracelens_report.json` | full TraceLens report (same `hot_kernels[]` shape) |
| `trace_file` | `runs/roofline/**/torch_trace` | the roofline torch-trace **directory** (per-TP-rank `*.pt.trace.json.gz`) |

Resolution prefers the parent of the `geak` segment in `exp_root`; if that
path is not present on the box it falls back to the on-disk grandparent of the
handoff file. The same four paths are also surfaced (with nulls) in the human
`tracelens_report` block of the driver prompt.

**How the workflow uses it (entirely additive — a tracelens-less run is byte-identical):**
the Profiler reads `args.tracelens` and, **only when `analysis_md` exists, SKIPS its
own warm-server trace collection** and builds the standardized Top-N from the
TraceLens artifacts; **when `trace_file` also exists it runs an ADDITIONAL
`parse_profile.py` pass** on the rank0 serving trace to recover real kernel
symbols + reliable per-launch shapes and reconcile them (TraceLens `analysis.md`
shapes are treated as a hint and double-checked). The System Architect uses
`kernel_candidates.json` as an advisory routing prior (enriching candidates with
`source_hint`/`launcher_hint`/`bound_type`) without ever overriding the measured
`%gpu`. When `args.tracelens` is absent (or for any post-config reprofile, where
the baseline prior is stale) the workflow profiles/strategizes exactly as before.

## `result.json` (workflow → caller)

```jsonc
{
  "schema_version": 1,
  "status": "ok | no_gain | error",
  "eval_dir": "/work/experiment/geak/e2e_<model>_<ts>",
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
  "report_path": ".../final_report.md",  // human report: per-kernel optimizations, changed params, TTFT/TPOT
  "kernel_journey_path": ".../kernel_journey.json",  // per-kernel journey contract (see below); absent if nothing accepted
  "recovered_from_disk": true             // present+true only when the handoff was rebuilt from on-disk artifacts
}
```

## Handoff resilience (the workflow return is never the single point of failure)

The workflow return (the JSON object carrying `eval_dir` + `accepted_*`) is the
only value scraped from the agent transcript. A failed scrape used to discard
the **entire** run as `workflow_parse_error` even though every artifact
(`director_e2e_validation.json`, the `final/` bundle, the measured gain) is on
disk. `run_e2e.py` now removes that fragility, layered:

1. **Robust capture** — the SDK path accumulates the *full* transcript (every
   text fragment from every message, incl. tool-result blocks), not just the
   last assistant text.
2. **Robust extraction** — the parser scans the whole transcript for the last
   JSON object carrying `eval_dir` (tolerates compact single-line, ```json```
   fences, pretty-printed multi-line, and trailing prose).
3. **On-disk sentinel** — on success the parsed return is persisted to
   `<eval_dir>/workflow_return.json`, so any later read never re-scrapes.
4. **Disk recovery** — if capture/extraction still fails (or the run timed out
   after the measured leg), the return is **rebuilt from on-disk artifacts**:
   `workflow_return.json` if present, else reconstructed from
   `director_e2e_validation.json` (throughput/speedup/parity/overlay/launch +
   `accepted_config` from `serving_config`) with accepted-kernel names recovered
   from the stable `overlay/cand_*` layout. A real win is therefore never lost
   to a lost handoff line. Recovery returns nothing only when no completed
   `eval_dir` exists (the run genuinely produced nothing). Recovered runs set
   `result.recovered_from_disk = true`.

These are general (no model/run-specific assumptions) and key only off the
stable artifact layout the workflow always writes.

## `kernel_journey.json` (per-kernel journey contract → orchestrator)

Because GEAK-e2e is a whole-pipeline e2e optimizer (not a per-kernel backend),
its authored kernels were invisible in the orchestrator's kernel-journey view
(`KERNEL_JOURNEY_SCHEMA.md`), which only saw upstream `tracelens` discovery.
`run_e2e.py` now emits `<eval_dir>/kernel_journey.json` (path echoed in
`result.kernel_journey_path`). It is self-contained and its per-kernel
sub-objects are shaped EXACTLY as the orchestrator recorder's
`record_kernel_{dispatch,backend_result,e2e}` inputs, so the orchestrator
replays them verbatim — all mapping lives here, once.

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

On the recovery path, per-kernel `micro_speedup` may be `null` (it only existed
in the scraped return) — never fabricated; but when exactly one kernel was
accepted it is credited with the whole measured e2e delta (sound attribution).

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
| dataset | `random`; `random-range-ratio` from `handoff.bench_protocol.random_range_ratio` (caller-driven: `0`=fixed, `>0`=variable), else standalone default `0` (fixed) |
| workload | same `ISL/OSL/CONC`; `NUM_PROMPTS` from `bench_protocol.num_prompts`, else `max(CONC*factor, CONC)` |
| warmups | `NUM_WARMUPS` from `bench_protocol.num_warmups`, else `min(CONC, 8)` (the materialize default) |
| seed | `SEED` from `bench_protocol.seed`, else fixed `0` |
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
