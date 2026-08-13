# `interface/run_e2e.py` — external integration contract

`interface/` is the **only** surface an external orchestrator (e.g. Hyperloom)
touches. Everything volatile about the e2e workflow (the `e2e_workflow.js` arg
names, the Claude Code `Workflow` invocation, the `--effort ultracode`
requirement, the SDK-vs-CLI choice) is hidden behind one command and two JSON
files. The result schema is versioned so callers can distinguish contract
changes while the workflow evolves internally.

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
| `raw_baseline_tput` | result audit metadata | pre-change session baseline; never used as the measurement-alignment signal |
| `orchestrator_best_tput_same_config` | result alignment metadata | caller throughput on the accepted config GEAK uses for its baseline |
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
  "schema_version": 2,
  "status": "ok | no_gain | error",
  "eval_dir": "/work/experiment/geak/e2e_<model>_<ts>",
  "baseline_throughput_tok_s": 1485.4,   // baseline leg, measured in the SAME session as the final
  "final_throughput_tok_s": 1551.4,      // hot median, always the same basis as the baseline
  "final_throughput_basis": "hot",
  "throughput_speedup": 1.044,           // ALWAYS equals final/baseline above (see invariant below)
  "output_parity": "pass | fail | n/a | unknown",
  "ttft_ms": 3598.0,                     // median, aligned with caller's ttft
  "tpot_ms": 39.5,                       // median, aligned with caller's tpot
  "final_launch_script": ".../final/final_launch.sh",  // self-contained: overlay/flags/env baked in
  "bench_script": ".../bench_e2e.sh",    // supports REUSE_SERVER=1 + CONC/ISL/OSL
  "final_patch": ".../final/final_patch.diff",   // "" when the run produced no applicable hunk
  "final_overlay": ".../final/overlay",          // "" when the run produced no loadable overlay
  "metric_basis": "aggregate_output_tok_s",   // NOT per-GPU; matches Magpie output_throughput
  "bench_client": "inferencex",               // inferencex => identical client to caller; else native
  "validated_regimes": [ { "isl": 1024, "osl": 1024, "conc": 64 } ],  // redo parity outside these
  "accepted_kernels": [ /* what was optimized + how (per-kernel) */ ],
  "accepted_heads": [ /* head GEMM/attn winners */ ],
  "accepted_config": { "flags": "...", "env": "..." },
  "baseline_basis": {
    "geak_measured_baseline_tok_s": 1551.4,
    "baseline_basis_source": "validation_base_bench_summary", // which leg the denominator came from
    "setup_baseline_tok_s": 1498.2,        // Setup-time baseline, audit only
    "baseline_drift_pct": 3.55,            // how far the box moved between Setup and Validate
    "orchestrator_baseline_tok_s": 1485.4,
    "raw_session_baseline_divergence_pct": 4.44, // audit only; includes accepted config gain
    "orchestrator_best_tput_same_config": 1550.8,
    "current_best_same_config_divergence_pct": 0.04, // primary alignment metric
    "measurement_divergence_pct": 0.04 // backward-compatible alias
  },
  "baseline_alignment": {
    "status": "aligned | warning | warning_recipe_unaligned | unavailable",
    "primary_metric": "current_best_same_config_divergence_pct",
    "divergence_pct": 0.04,
    "warning_threshold_pct": 3.0,
    "raw_session_divergence_is_measurement_signal": false,
    "recipe_aligned_with_orchestrator": true   // false => the two harnesses served different stacks
  },
  "serving_stack": {                       // WHO launched the servers, and what they picked
    "launcher": "magpie | native",
    "launch_script": "/.../benchmarks/vllm_mi355x.sh",  // "" on the native path
    "launch_script_source": "handoff | env | launch_recipe",
    "recipe_aligned_with_orchestrator": true,
    "baseline": {
      "aiter_mentions": 5499,             // near-zero => the accelerated stack never came up
      "kernel_picks": ["Selected AiterFp8BlockScaledMMKernel for Fp8LinearMethod", "..."]
    },
    "validation_base": { "aiter_mentions": 5471, "kernel_picks": ["..."] }
  },
  "validation_evidence": {                 // audit only; never changes status
    "validation_status": "validated_win",
    "speedup_basis": "workflow_return | final_over_baseline",
    "delta_pct": 4.4,
    "noise_band_pct": 1.0,                 // the Director's declared band for this box
    "baseline_spread_pct": 0.2,            // run-to-run scatter of each leg
    "final_spread_pct": 0.3,
    "significance_threshold_pct": 1.0,     // the widest of the three above
    "delta_exceeds_noise": true,
    "spreads_non_overlapping": true,       // null unless BOTH legs reported a spread
    "beats_orchestrator_same_config": true,
    "intermediate_win_not_confirmed": null, // true => Validate did not confirm an accepted A/B
    "validate_final_missing": null          // true => the final number came from a disk A/B
  },
  "report_path": ".../final_report.md",  // human report: per-kernel optimizations, changed params, TTFT/TPOT
  "kernel_journey_path": ".../kernel_journey.json",  // per-kernel journey contract (see below); absent if nothing accepted
  "recovered_from_disk": true             // present+true only when the handoff was rebuilt from on-disk artifacts
}
```

### The reported speedup is always the reported pair

`throughput_speedup` equals `final_throughput_tok_s / baseline_throughput_tok_s`
to within 1e-3, without exception. A consumer may recompute it and will get the
same answer. Three rules keep that true:

* **Same-session pair.** The denominator is the unpatched leg re-measured during
  Validate (`validation/base`), not the Setup baseline. The box drifts by several
  percent within a session, and a Setup denominator reports that drift as
  optimization. `baseline_drift_pct` says how much drift there was, and
  `setup_baseline_tok_s` keeps the old number for audit.
* **One basis.** Both sides are hot medians. Cold rounds never become the
  promoted number: only the first bench of a session runs on a genuinely cold
  box, so a "cold" final measured hours later is a warm round wearing the label,
  and the ratio of the two is mostly cache-fill asymmetry. Cold numbers stay in
  `alignment_metrics` as a diagnostic, where `cold_pairing` says whether the two
  cold rounds are even comparable and `cold_penalty_pct_baseline` /
  `cold_penalty_pct_final` show what each leg paid.
* **The pair has the last word.** If anything upstream reports a speedup the
  published pair contradicts, it is rebuilt from the pair and
  `alignment_metrics.speedup_basis` becomes `final_over_baseline`, with the
  original preserved in `speedup_as_returned`.

A measured Validate verdict is never overridden. When Validate re-runs an
accepted change and does not confirm the gain, that verdict is what ships, with
`validation_evidence.intermediate_win_not_confirmed` recording the disagreement.
Only a **missing** final (the Validate bench crashed, so there is no verdict at
all) falls back to the best accepted intermediate A/B on disk.

`final_patch` and `final_overlay` are empty strings unless the run produced
something loadable — a diff with at least one hunk, an overlay with importable
code. Finalize writes both unconditionally, so their existence proves nothing.

### Choosing a headline out of a candidate pool

When a run dies before Validate, the result is salvaged from the intermediate
A/Bs on disk. Choosing one candidate out of several can manufacture a gain by
itself, because taking a maximum over a noisy pool preferentially selects
whichever candidate drew the most favourable reference leg. The selection rules
mirror what `e2e_workflow.js` requires before it banks a candidate live:

* `accepted` outranks `stack`. The integrator writes `stack` to mean
  "non-negative, engaged, parity-safe — carry it forward to compound, but not a
  standalone win". A stack-only salvage ships as `result_source:
  "disk_stack_provisional"`.
* Candidates are ranked by their own `e2e_delta_pct`, never by absolute
  throughput, which is not comparable across candidates measured at different
  points in the session.
* A soft-gated (sampled-accuracy) candidate whose delta exceeds twice its
  Amdahl ceiling is excluded, exactly as `integAccepted()` excludes it live.
  Byte-exact parity outranks the ceiling and is trusted.
* Parity failures and incomplete A/Bs are skipped.
* Every distinct kernel in the stack is credited, and competing backends of one
  kernel are counted once.

`validation_evidence.recovery` records the pool size, the pick, its gate, its
delta-over-ceiling ratio, and anything excluded, so the choice is auditable.

`raw_session_baseline_divergence_pct` compares GEAK's accepted-config baseline
with the caller's pre-change session baseline. It is audit-only because it
includes configuration gains accepted before GEAK started.

`current_best_same_config_divergence_pct` compares the same accepted
configuration in both harnesses and is the primary alignment metric.
`measurement_divergence_pct` remains an exact compatibility alias for existing
callers. If the handoff omits `orchestrator_best_tput_same_config`, both
same-config fields are `null` and `baseline_alignment.status` is `unavailable`;
GEAK never falls back to the raw-session divergence as a drift signal.

### Same config is not the same stack

Both harnesses can apply the identical flags and environment and still serve
different engines, because the orchestrator launches its server through its own
script and that script — not the transferred config — owns the platform kernel
preset, `--trust-remote-code`, and the gpu-memory-utilization default. When GEAK
launches through its native backend adapter instead, the accelerated kernel
stack can silently fail to come up and the same configuration serves around ten
percent slower. The divergence metric then measures the launch recipe, not the
box or the bench client.

`serving_stack` makes that legible without reading a server log. `launcher`
says who launched (`magpie` = the orchestrator's own script, `native` = GEAK's
adapter), and each leg's `aiter_mentions` / `kernel_picks` record which kernels
the engine actually selected. A near-zero `aiter_mentions` next to a large
negative divergence is the signature of an unaligned recipe, and
`baseline_alignment.status` reports `warning_recipe_unaligned` for exactly that
case so the number is not read as a measurement problem.

Set `BENCH_LAUNCHER=native` in the environment to force the adapter launch; it
outranks every other resolution path and is the escape hatch when the
orchestrator's script cannot run.

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

### Server-LAUNCHER adapter (closes the launch-recipe residual)

The client adapter above aligns *who measures*; this one aligns *what is being
measured*. `BENCH_LAUNCHER=magpie` makes `adapters/launchers/magpie.sh` run the
orchestrator's own launch script for every fresh server, with the authored-kernel
overlay prepended to `PYTHONPATH` (which the orchestrator's own path cannot do),
so recipe parity and overlay application coexist. One adapter serves every
backend, because the scripts share one server-phase contract.

The script itself is resolved most-explicit-first: `handoff.bench_launcher` /
`$BENCH_LAUNCHER` decide the launcher, then the script comes from
`handoff.launch_server_script`, `$MAGPIE_LAUNCH_SCRIPT`,
`$MAGPIE_<BACKEND>_SCRIPT`, or — the case that actually fires — is derived from
`handoff.launch_recipe`. No handoff has ever named the launch script, but every
one names the recipe, and the recipe names both the InferenceX checkout and the
script filename inside its `benchmarks/` directory.
`serving_stack.launch_script_source` records which of those sources won.

Resolution degrades to `native` rather than failing whenever the script cannot
be confirmed usable — recipe unreadable, checkout not present on this box, or
the script's `benchmark_lib.sh` sibling missing. `BENCH_LAUNCHER=native` forces
that degrade explicitly and is the escape hatch.

`MAX_MODEL_LEN` is forwarded to the script on the `magpie` path only, because
the script's own default (4096) has nothing to do with the run and the
orchestrator overrode it by env when it measured the reference. gpu-mem-util is
deliberately *not* forwarded: no handoff carries `mem_fraction`, and the
script's 0.95 default is the recipe being matched. The script writes the server
to `$LOG` and its own trace to `magpie_launch.log` next to it, because the
script's redirect truncates `$LOG` and would otherwise destroy anything the
adapter wrote there.
