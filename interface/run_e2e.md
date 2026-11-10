# `interface/run_e2e.py` — external integration contract

External orchestrators such as Hyperloom use one command and two JSON files.
`run_e2e.py` maps the handoff to the standalone workflow runtime and returns a
versioned result. Runtime dispatch, agent selection, and nested kernel optimizer
defaults are owned by GEAK, so callers can keep the same invocation as the
workflow evolves.

The Python implementation is split by responsibility. `run_e2e.py` owns CLI
handling, environment settings, launch configuration, runtime dispatch and disk
recovery. `e2e/normalization.py` interprets measurement artifacts and builds the
result, while `e2e/reports.py` renders and writes reports from that result. The
entrypoint retains its existing helper imports; it passes recovery, workload
classification and the current alignment threshold to normalization explicitly.

## Command

```bash
python interface/run_e2e.py <handoff.json> <result.json> [--timeout-s SECONDS] [--dry-run]
```

* Exit code `0` → `result.json.status` is `ok`, `no_gain` or `timeout`.
* Exit code `1` → a crash; `result.json.status == "error"` with an `error` field.
* Exit code `2` → bad usage / unreadable handoff.
* `--timeout-s` → state a time budget. When `GEAK_E2E_TIMEOUT_S` also supplies
  a valid budget, the smaller value applies. The default applies when neither
  supplies one.
* `--dry-run` → print the mapped `e2e_workflow.js` args + the prompt and
  exit `0` (no GPU work). Use this to validate the mapping in CI.

Discovery: the installer should export `GEAK_E2E_RUNNER` pointing at this
file (`$GEAK_ROOT/interface/run_e2e.py`) so the caller has a single
hard-coded handle.

### Hyperloom checkout compatibility

Hyperloom's existing installer selects this release through `GEAK_REF`. From
the Hyperloom checkout, pass the branch to the usual installation step:

```bash
GEAK_REF=feature/evo-reviewed-release-20261110 \
  bash src/hyperloom/inference_optimizer/assets/install.sh
```

The installer resolves and installs the selected GEAK checkout and exports
`GEAK_ROOT` and `GEAK_E2E_RUNNER`. An explicitly configured runner must point to
`interface/run_e2e.py` in that same checkout: the script resolves its workflow
files relative to its own location. Retain the complete checkout alongside the
installed Python package. The package remains named `geak`, and the entrypoint
remains `interface/run_e2e.py`. The handoff/result schemas keep their existing
interface.

The standard two-JSON invocation uses the standalone runtime. It selects Claude
unless an explicit backend/profile or the existing unambiguous provider
selection chooses another agent. Omitting `handoff.optimizer` selects `evo` for
the nested kernel worker; `"optimizer": "greedy"` explicitly selects greedy.
This needs no additional Python CLI flags or Hyperloom handoff fields.

Existing workload, benchmark-client, server-configuration, and GPU-placement
inputs still govern the run. Selecting this branch does not establish workload
equivalence or replace Hyperloom's own validation of a returned candidate.

### E2E KB recall policy

GEAK reads the deployment KB after the Setup baseline and before Profile/Strategize.
Normal runs default to `warm_start=on`; `reference` and fast mode offer references
without applying candidates. Resuming phases without Setup skips this read.

To select KB records independently of an orchestrator's metric, export this in the
environment that launches GEAK (or its parent orchestrator):

```bash
export GEAK_E2E_KB_METRIC_BASIS=e2e_norm_intvty_p50
```

The handoff/workflow argument `e2e_kb_metric_basis` takes precedence over this
environment variable. Both accept the metric bases and axis aliases supported by
GEAK, and require an exact recorded basis: other metrics and unlabelled legacy
records are excluded before ranking. Model, workload kind, hardware, TP/EP and
concurrency still use the normal identity ladder. This setting does not change
KB addresses, benchmark axes, acceptance rules, or the metric recorded by new writes.
Without an override, AgentX retains the measurement-derived filter and permits
unlabelled historical records; synthetic reads retain their existing behavior.

For example, a caller can measure `output` while GEAK recalls only p50 candidates.
Those candidates must still pass the current run's local A/B gate to be adopted.
An A/B on another metric does not attest or reject the stored p50 claim. Logs and
`workflow_return.json` → `kb_recall.e2e` report `recall_metric_basis`,
`measurement_metric_basis`, the policy source, and whether matching was strict.
Setting the recall filter alone does not make optimization or acceptance use p50.

For Hyperloom, point `GEAK_ROOT` and `GEAK_E2E_RUNNER` at this GEAK build and
provide `GEAK_KB_STORE_URL` / `GEAK_KB_STORE_TOKEN` to its parent environment.
No Hyperloom change is needed for this recall override. The installer overlays
created by GEAK also expose their real imports through `modules`, so consumers
of the original overlay manifest can load them. A source-tree hash in the manifest
includes nested installer/kernel files in those consumers' overlay fingerprints.

### Agent backend (Claude Code or Codex)

`run_e2e.py` runs the JS workflow through the standalone Node runtime at
`interface/runtime/engine/run_workflow.mjs`. The runtime implements
`agent/parallel/pipeline/phase/workflow` and the host `runProcess` primitive used
by kernel delivery. It dispatches each `agent()` call to the selected CLI.
Claude is the default agent; it uses this same runtime path.

**Two orthogonal axes**, defined in `interface/runtime/engine/registry.json`:
`agents` (how to drive a CLI: `claude` / `codex`) × `models` (an endpoint). A
`profile` pins one `(agent, model)` combo. Adding a third agent is a data change
— a new `agents` entry plus a passing `conformance.mjs --agent <name>`; see
`runtime/SETUP.md` for the R1–R7 bring-up checklist.

| Selection for `run_e2e.py` | Effect |
| --- | --- |
| No explicit selection and no unambiguous provider selection | Runtime with Claude |
| Only `GEAK_AMDKEY` or `OPENAI_API_KEY` credentials configured | Runtime with Codex through provider auto-selection |
| `GEAK_AGENT_PROFILE=codex-gpt56` | runtime, profile's agent+model |
| `GEAK_AGENT_BACKEND=codex` | runtime, agent `codex` |
| `GEAK_MODEL=<name>` | override the model axis (registry `models` key) |

For the Python entrypoint, explicit profile/backend environment settings take
precedence over provider auto-selection. A key only selects Codex while none of
`ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, or
`CLAUDE_CODE_OAUTH_TOKEN` is set, so exporting a gateway key
next to an existing Claude setup retains Claude. `GEAK_AGENT_AUTO=0` disables
provider auto-selection. The existing `--profile`, `--agent`, and `--model`
flags belong to the Node runner; they are not flags for `run_e2e.py`.

Env knobs (all optional):

| Env | Meaning | Default |
| --- | --- | --- |
| `GEAK_AGENT_PROFILE` | registry profile = (agent, model) | unset |
| `GEAK_AGENT_BACKEND` | agent name | Claude when no other selection applies |
| `GEAK_MODEL` | model name (registry `models` key) | unset |
| `GEAK_REGISTRY` | path to a custom registry.json | shipped one |
| `GEAK_NODE_BIN` | node binary for the runtime | `node` |
| `GEAK_CONCURRENCY` | max concurrent agent subprocesses | `min(16, cpus-2)` |
| `GEAK_AGENT_TIMEOUT_MS` | per-agent hard timeout | `14400000` (4 hours) |
| `GEAK_SCHEMA_RETRIES` | in-call structured-output retries | `2` |
| `GEAK_CLAUDE_BIN` / `GEAK_CLAUDE_MODEL` | Claude binary / raw model ID; forwarded through the registry | runner/registry default |
| `GEAK_CLAUDE_SETTINGS` | Claude settings JSON or settings-file path accepted by the CLI | CLI defaults when unset |
| `GEAK_CLAUDE_EFFORT` | Claude effort; valid `low`, `medium`, `high`, `xhigh`, or `max` values are forwarded | CLI default when unset or not a valid effort |
| `GEAK_CLAUDE_CONFIG_DIR` | seeded Claude configuration directory; when nonempty, overrides the child's `CLAUDE_CONFIG_DIR` | inherited `CLAUDE_CONFIG_DIR` or CLI default |
| `GEAK_CODEX_BIN` / `GEAK_CODEX_MODEL` | Codex binary / raw model ID | registry/provider default |
| `GEAK_CODEX_APPROVE` / `GEAK_CODEX_EXTRA_ARGS` | Codex approval argument / extra CLI arguments | registry |
| `GEAK_AMDKEY` | AMD gateway key; also selects codex. `GEAK_`-prefixed so it survives hyperloom's `.env` allowlist — see SETUP.md | inherited |
| `OPENAI_BASE_URL` / `OPENAI_API_KEY` | OpenAI-compatible provider auth (codex); the key also selects codex | inherited |
| `ANTHROPIC_BASE_URL` / `ANTHROPIC_*` | Anthropic provider auth (claude) | inherited |
| `GEAK_AGENT_AUTO` | `0` disables key-based backend selection | `1` |

All real runs require Node on `PATH` and the selected agent CLI with its
credentials. The runtime requires Node 18 or newer; the documented Codex setup
uses Node 20 or newer. Install GEAK's existing Python dependencies and retain
the workflow, role, knowledge, and script directories in the checkout. See
[`runtime/SETUP.md`](runtime/SETUP.md) for installation and agent configuration.

The single-kernel `kernel_workflow.js` has no Python wrapper; run it on the
runtime directly:

```bash
node interface/runtime/engine/run_workflow.mjs kernel_workflow/kernel_workflow.js \
  --agent codex \
  --args '{"kernel_path":"/abs/kernel","workflow_dir":"/abs/kernel_workflow","budget":6}'
```

**Controlled (agent × model) experiments** are built in:

```bash
node interface/runtime/engine/experiment.mjs \
  --script kernel_workflow/kernel_workflow.js \
  --args '{"kernel_path":"/abs/knn","workflow_dir":"/abs/kernel_workflow","budget":6}' \
  --agents claude,codex --models default --repeats 3 --out ./exp_compare
# -> results.jsonl + summary.md/csv (speedup / success-rate / wall / schema-fails; no token/cost)
```

Runtime primitives + config resolution can be smoke-tested with no CLI/network/GPU:
`node interface/runtime/engine/selftest.mjs`. See `runtime/SETUP.md` for the full picture.

The fast-path artifacts live under `<exp_root>/geak_e2e_moe_int4/`
(`baseline/`, `validation/final/`, `final/` bundle, `director_e2e_validation.json`).

## `handoff.json` (caller → workflow)

```jsonc
{
  "schema_version": 2,
  "model_path": "/models/Qwen-Qwen3.5-27B",
  "framework": "sglang",                 // -> backend (sglang|vllm|atom)
  "gpu_type": "MI355X",
  "expected_gfx": "gfx950",              // optional pair; otherwise probed
  "expected_target": "mi355x",           // r9700|r9600d|mi350x|mi355x|unknown; pair with expected_gfx
  "expected_device_name": "AMD Instinct MI355X",
  "expected_physical_cu_count": 256,
  "tp": 8,                               // serving tensor-parallel size (honoured, no TP=1 lock)
  "optimizer": "evo",                    // optional; default evo, explicit greedy remains supported
  "gpu_ids": "0,1,2,3,4,5,6,7",          // optional; default 0..tp-1
  "workload": { "isl": 1024, "osl": 1024, "conc": 64 },
  "accepted_flags": "--attention-backend triton",  // best config from the caller's search
  "accepted_env": "SGLANG_USE_AITER=1",
  "launch_recipe": "/path/baseline_config.with_envs.yaml",  // optional launch script/recipe
  "raw_baseline_tput": 1485.4,           // caller's pre-change session baseline (audit reference)
  "orchestrator_best_tput_same_config": 1550.8, // caller best measured with accepted_flags/env
  "same_config_observed_identity": {     // optional observed upstream server facts
    "backend": "sglang",
    "server_args": { "model_path": "/models/Qwen-Qwen3.5-27B", "tp_size": 8 }
  },
  "exp_root": "/work/experiment/geak",   // basename MUST be `geak`; the timestamped run dir is created here
  "bench_client": "auto",                // auto|inferencex|native|agentx — see measurement alignment below
  "inferencex_path": "/opt/InferenceX",  // optional; else taken from $INFERENCEX_PATH
  "bench_protocol": {                    // optional; measurement settings, with schema-specific overrides below
    "random_range_ratio": 1,
    "num_prompts": 192,
    "num_warmups": 8,
    "seed": 0
  }
}
```

Required: `model_path`, `exp_root`. Everything else has a default.

`optimizer` accepts `evo` or `greedy`; omission selects `evo`. An optional
positive integer `kernel_budget` controls the ordinary nested kernel budget.
Optimizer selection does not select an agent, change the benchmark client, or
implicitly enable every EVO configuration option. See
[`docs/evo/configuration.md`](../docs/evo/configuration.md) for explicit worker
configuration.

With EVO, `evolution_return_development` defaults to `true`. After normal
engineering and evaluation, it permits retaining one concrete unmeasured
development attempt when there is no measured development source to return.
The attempt remains partial and unqualified; correctness, fitness, and promotion
still require the ordinary verifier. Explicit `false` disables this fallback.

The handoff also forwards `evolution_import_stall_sources`,
`record_planner_inputs`, and `case_measurement_view` to the nested kernel worker,
along with `evolution_start`, `evolution_case_evidence_view`,
`evolution_evidence_handoff`, and `evolution_preserve_founder_attempts`. Values
are preserved for worker validation; boolean controls use JSON booleans.
`case_measurement_view` accepts `current_mixed` or `native_consistent` only for
an ordinary workload-weighted task. Importing stall sources requires EVO with
`evolution_start="first_stall"`.

Before a real run selects a backend or architecture policy, `run_e2e.py`
requires structured GPU identity. It uses the explicit `expected_gfx` /
`expected_target` pair when present, otherwise `GEAK_GPU_IDENTITY_JSON`, and
otherwise runs `scripts/gpu_identity.py`. The environment override is a JSON
object with `gfx`, `target`, `marketing_name`, and `physical_cu_count`, for
example:

```bash
export GEAK_GPU_IDENTITY_JSON='{"gfx":"gfx1201","target":"r9700","marketing_name":"AMD Radeon AI PRO R9700","physical_cu_count":64}'
```

Exact `AMD Radeon AI PRO R9600D` detection produces `r9600d` and preserves
the observed CU count (48 on the recorded board). Like `r9700`, this target
requires `gfx1201` in explicit and detected identity pairs. The interface
preserves the product, ISA, device name and CU count in the workflow handoff.

Product identity selects product data, not separate logic. `r9700` and
`r9600d` share the gfx1201 ISA path (vLLM-only serving, CDNA knowledge
isolation) and differ only in the `product:` block of
`e2e_workflow/knowledge/analysis_skills/roofline/peaks.md`; the R9600D block
holds its own datasheet peaks, not R9700's. The default backend is `vllm` for both.
Unknown gfx1201 products remain `unknown` and keep their existing refusal
behavior.

Exact `AMD Instinct MI350X` and `AMD Instinct MI355X` names produce `mi350x`
and `mi355x`, which require `gfx950`. A supplied `expected_device_name`, or a
detected `marketing_name`, that names a different supported product than the
target, or a supported product on a different ISA, is refused.

`--dry-run` is host-only: when neither explicit nor environment identity is
provided it skips rocminfo and reports `gpu_identity_status:
unavailable_dry_run`. A real run never uses that placeholder and still fails
closed.

`bench_protocol` accepts a partial set of measurement settings. For handoffs
without the schema-specific alignment below, supplied keys override the bench
script defaults and omitted keys retain those defaults.

A handoff with `schema_version >= 2` and a `baseline_env_spec` uses warm-server
measurement and validation with `REPLICA_RETRIES=1`. For a synthetic workload,
the adapter additionally sets `NUM_WARMUPS=2*conc`, `SEED=0`, and
`RANDOM_RANGE_RATIO=1`, overriding those individual protocol fields. It retains
an explicit `num_prompts`; otherwise it enables the adaptive prompt-count rule.
AgentX replay keeps its own corpus and warmup settings rather than receiving
these synthetic prompt overrides. The resolved protocol and workload identity
must be checked before comparing results with another harness.

### How handoff maps to the workflow (owned by `run_e2e.py:map_args`)

| handoff field | `e2e_workflow.js` arg | note |
|---|---|---|
| `model_path` | `model_path` | required |
| `framework` | `backend` | `sglang` \| `vllm` \| `atom` |
| `tp` | `tp` | serving tensor-parallel (threaded to bench `TP`) |
| `gpu_ids` / `tp` | `gpu_ids` | defaults to `0..tp-1` |
| `expected_gfx` / `expected_target` | matching `expected_*` args | optional explicit structured identity pair; otherwise probe/env identity |
| `optimizer` | `optimizer` | `evo` when omitted; explicit `greedy` is honored |
| `kernel_budget` | `kernel_budget` | optional positive integer for ordinary nested kernel work |
| `evolution_return_development` | same nested kernel argument | defaults to `true` with EVO; explicit `false` disables the unmeasured-source fallback |
| `evolution_import_stall_sources` / `record_planner_inputs` / `case_measurement_view` | same nested kernel arguments | forwarded for worker validation |
| `workload.{isl,osl,conc}` | `isl` / `osl` / `conc` | profile + bench workload |
| `workload_spec` | AgentX workflow arguments and client environment | `kind: agentx_trace_replay` supplies replay settings and optional observed shape |
| `accepted_flags` | `initial_extra_server_args` | seeds the baseline = caller best config |
| `accepted_env` | `initial_extra_env` | seeds baseline env |
| `launch_recipe` | `launch_script` | optional |
| `baseline_env_spec` | resolved launch configuration/environment | preserves accepted settings, removals, and argument replacement mode |
| `launch_server_script` / `bench_launcher` | server-launcher adapter inputs | explicit launcher/script selection |
| `raw_baseline_tput` | result audit metadata | pre-change session baseline; never used as the measurement-alignment signal |
| `orchestrator_best_tput_same_config` | result alignment metadata | caller throughput on the accepted config GEAK uses for its baseline |
| `same_config_observed_identity` | result identity metadata | optional upstream observed server facts; compared to GEAK's Setup `ServerArgs` / Magpie launch evidence |
| `exp_root` | `exp_root` | run dir root |
| (derived from `exp_root`) | `tracelens` | auto-discovered upstream TraceLens / kernel-agent artifacts (see below); only non-null paths forwarded; key omitted entirely when none found |
| `bench_client` / `inferencex_path` | env `BENCH_CLIENT` + `INFERENCEX_PATH` | exported so every `bench_e2e.sh` call inherits it (not a JS arg) |
| `bench_protocol.{random_range_ratio,num_prompts,num_warmups,seed}` | env `RANDOM_RANGE_RATIO` / `NUM_PROMPTS` / `NUM_WARMUPS` / `SEED` | partial settings, subject to the schema-specific alignment above (not a JS arg) |
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
  "bench_client": "inferencex",               // resolved measurement client; also native or agentx
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
    "current_best_same_config_divergence_pct": 0.04, // Setup seed vs upstream same-config reference
    "measurement_divergence_pct": 0.04 // backward-compatible alias
  },
  "baseline_alignment": {
    // compatibility alias: numeric Setup-seed comparison only
    "status": "aligned | warning | warning_recipe_unaligned | unavailable",
    "primary_metric": "setup_seed_same_config_divergence_pct",
    "divergence_pct": 0.04,
    "warning_threshold_pct": 3.0,
    "raw_session_divergence_is_measurement_signal": false,
    "recipe_aligned_with_orchestrator": true   // false => the two harnesses served different stacks
  },
  "handoff_alignment": {                // authoritative cross-handoff verdict
    "status": "aligned | warning | warning_recipe_unaligned | identity_mismatch | unverified | unavailable",
    "metric_status": "aligned | warning | warning_recipe_unaligned | unavailable",
    "primary_metric": "setup_seed_same_config_divergence_pct",
    "server_identity": {
      "expected": { /* upstream handoff observation, or orchestrator_server_log facts */ },
      "expected_source": "<handoff key> | orchestrator_server_log",
      "observed": { /* GEAK Setup: SGLang ServerArgs, ATOM Engine kwargs, or Magpie identity */ },
      "status": "matched | mismatched | unverified | unavailable",  // worst of Setup and every geak_launched leg
      "mismatches": [{"key": "...", "expected": "...", "observed": "..."}],  // Setup only, when it mismatched
      "geak_launched": [                // servers GEAK launched itself, judged against the orchestrator log
        {"leg": "profile | validation_base | validation_final", "path": "...",
         "status": "matched | mismatched | unverified", "mismatches": [ /* as above */ ]}
      ],
      "geak_launched_reference": "orchestrator_server_log | <handoff key>",
      "orchestrator_reference_problem": "<why no orchestrator log could be trusted>",  // only when set
      "evidence_paths": [".../baseline/server.log", ".../baseline/magpie_launch.log",
                         "<orchestrator server log actually read>", ".../profile/round_0/server.log"]
    }
  },
  "measurement_drift": {                // Setup vs Validate/base, never cross-handoff alignment
    "status": "measured | unavailable",
    "setup_baseline_tok_s": 1498.2,
    "validation_base_tok_s": 1551.4,
    "drift_pct": 3.55,
    "evidence_paths": [".../baseline/bench_summary.json", ".../validation/base/bench_summary.json"]
  },
  "server_identity": { /* same identity block as handoff_alignment.server_identity */ },
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
  "validation_evidence": {                 // audit only, except acceptance (see below)
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
    "validate_final_missing": null,         // true => the final number came from a disk A/B
    "acceptance": {                        // ONLY when the handoff states a KEEP rule; decides "ok"
      "objective": "e2e_norm_intvty_p50", "measured_basis": "e2e_norm_intvty_p50",
      "gain_pct": 4.4, "min_gain_pct": 3.0,
      "guards": { "e2e_norm_intvty_p90": { "reference": 81.9, "candidate": 80.7,
                                           "max_drop_pct": 5.0, "holds": true } },
      "keep": true, "reasons": []
    }
  },
  "report_path": ".../final_report.md",  // human report: per-kernel optimizations, changed params, TTFT/TPOT
  "kernel_journey_path": ".../kernel_journey.json",  // per-kernel journey contract (see below); absent if nothing accepted
  "recovered_from_disk": true,            // present+true only when the handoff was rebuilt from on-disk artifacts
  "tuning_skillset": { /* ADDITIVE; absent unless the tuning phase ran — see below */ }
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

### The caller's KEEP rule decides "ok" when the handoff states one

Without a rule, `status` is `ok` whenever the published final beats the published
baseline. A caller that re-measures and keeps only bigger wins can hand its own
rule in `workload_spec.acceptance`. Hyperloom does this on AgentX sessions graded
on interactivity:

```jsonc
"acceptance": {
  "objective": "e2e_norm_intvty_p50",      // must equal the metric_basis measured
  "min_gain_pct": 3.0,                     // (final - baseline) / baseline * 100 >= this
  "guard_max_drop_pct": {                  // each guard may fall at most this far below the reference
    "e2e_norm_intvty_p90": 5.0,
    "aggregate_output_tok_s": 5.0
  }
}
```

The rule is applied to the published pair. Guards are read from the
`guard_<basis>_median` fields of the two legs' `bench_summary.json`
(`validation/base` or the Setup baseline, and `validation/final`). A guard
missing from either leg fails, so a recovered intermediate win, which has no such
legs, cannot pass. When the rule does not keep the pair, `status` is `no_gain`
and `validation_evidence.acceptance.reasons` says why. The throughput numbers and
the speedup are reported unchanged. Per-kernel gates keep GEAK's own noise band,
so small wins can still stack up to the caller's bar. A block that does not parse
is ignored, with a warning on stderr.

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
* The Amdahl estimate is advisory (the legacy `amdahl_ceiling_pct` field is a
  conditional GPU-time estimate, not an e2e bound). It uses the candidate's
  `affected_gpu_time` coverage when that profile matches the measurement's
  candidate, reference and workload digests, otherwise the head's GPU-time share
  and isolated speedup. A soft-gated (sampled-accuracy) candidate whose delta
  exceeds the estimate by `GEAK_IMPLAUSIBLE_SPEEDUP_MARGIN` (default: twice the
  estimate) gets one independent A/B speedup review on the live path; recovery
  cannot run that review. Until a review confirms the gain, the candidate stays
  eligible and is listed under `awaiting_speedup_review`. Only a review that ran
  and measured a rejection excludes it (`rejected_by_speedup_review`), as
  `integAccepted()` does live. A review result is applied only to the candidate
  whose original `measurement_id` it names.
* Parity failures and incomplete A/Bs are skipped.
* Every distinct kernel in the stack is credited, and competing backends of one
  kernel are counted once.

`validation_evidence.recovery` records the pool size, the pick, its gate, its
delta-over-estimate ratio, the estimate and its source, candidates awaiting
review, and any candidate a measured review rejected, so the choice is auditable.

### `tuning_skillset` (additive)

Emitted only when the standalone tuning phase ran. It is **purely additive**: every key above keeps its
name, type and meaning, and a run without the phase produces a byte-identical `result.json`. A consumer
that does not know about tuning is unaffected.

The tuning gain is **already inside** `throughput_speedup` — the phase runs mid-pipeline and every later
measurement is taken on top of its accepted config. This block **attributes** part of the headline, it
does not add to it; summing the two double-counts.

```jsonc
"tuning_skillset": {
  "phase": "TuningSkillset",
  "ran": true,
  "gate": "accepted | no_win | rejected | incomplete | skipped | not_run",
  "explanation": "prose: what the phase did and what it means for the headline",
  "pre_tune_throughput_tok_s": 1000.0,   // the phase's OWN in-session interleaved A/B,
  "post_tune_throughput_tok_s": 1080.0,  //   not re-derived from the run baseline
  "tuning_delta_pct": 8.0,
  "share_of_total_gain_pct": 40.0,       // null when the run had no net gain to apportion
  "engagement_verified": true,
  "engagement_evidence": "...",          // an accept is withheld without this
  "ops_tuned": [ /* per-op: backend, tuner, shapes, isolated speedup, engaged */ ],
  // Accepted gates only — how the win reaches production:
  "deploy_bundle": ".../tuning/deploy",
  "cache_invalidation": ["rm -rf /tmp/aiter_configs"],
  "live_tree_files": ["aiter/configs/model_configs/..."],  // DATA written inside an installed package
  "apply_overlay": "",                   // non-empty if tuning also needed a routing/dispatch code
                                         //   change to make the artifact bind; merged into final_overlay

  "in_final_bundle": true,
  "reaches_production_via": { "final_patch_includes_tuning": true, "final_launch_runs_deploy": true, ... }
}
```

**Deployment.** A tuning win is usually *data* (a config table a library reads from inside its own
package dir, plus a derived cache that must be dropped or the new rows are silently ignored), so it
cannot travel in `final_overlay`, which is a `PYTHONPATH` overlay for *code*. It ships through the same
two handles you already use: its diff is concatenated into `final_patch`, and `final_launch_script` runs
the bundle's idempotent `deploy.sh` before starting the server. **Reusing `final_launch_script` requires
no extra steps.** Applying `final_patch` by hand also requires the `cache_invalidation` commands.
`in_final_bundle: false` means Finalize could not get the tuning into the bundle — the headline number
will not reproduce from it.

`raw_session_baseline_divergence_pct` compares GEAK's Setup seed baseline
with the caller's pre-change session baseline. It is audit-only because it
includes configuration gains accepted before GEAK started.

`current_best_same_config_divergence_pct` compares the Setup seed and the same
accepted configuration in both harnesses. It is the numeric compatibility
metric for the cross-handoff comparison; `handoff_alignment` is the
authoritative verdict and is `unverified` until actual Setup launch evidence
matches an upstream observed identity. GEAK reads that evidence from
`baseline/server.log` (parsed SGLang `ServerArgs`, ATOM `Engine kwargs` or vLLM
`non-default args`) or
`baseline/magpie_launch.log` (an emitted launch identity). When the handoff names
the orchestrator's own baseline server log
(`baseline_env_spec.measurement_evidence.actual_server_log_path`), GEAK parses it
too, falling back to the newest `attempts/<NNN>/server.log.gz` that holds a complete
launch record when Hyperloom has rotated the live file; attempts that disagree make
it untrusted (`orchestrator_reference_problem`). Those facts judge Setup when the
upstream identity is an opaque digest, and always judge the servers GEAK launched
itself (`geak_launched`: profile round 0, validation base and final), because a
matching Setup says nothing about a later launch. `server_identity.status` is the
worst of all checks (mismatched > unverified > matched); a launch field logged in a
form the parser does not know is recorded under `unparsed` and makes that check
unverified. For vLLM, whose `non-default args` line lists every argument that
differs from its default, every field is compared, including one the parser has
never seen, except the run-scoped `host`, `port`, `model_tag` and the
`profiler_config` of GEAK's own profile servers. Two servers that differ in a
field are `mismatched`; when both sides are complete parsed vLLM records, a field
logged by only one side is also `mismatched`. Partial caller-supplied identities
keep their subset comparison. An item that is not a `key: value` pair is recorded
under `unparsed`. Every evidence path
actually read is recorded in
`server_identity.evidence_paths`.
`measurement_divergence_pct` remains an exact compatibility alias for existing
callers. If the handoff omits `orchestrator_best_tput_same_config`, both
same-config fields are `null` and `handoff_alignment.status` is `unavailable`;
GEAK never falls back to the raw-session divergence as a drift signal.

`measurement_drift` separately compares the Setup baseline with the
Validate/base leg. It is `unavailable`, not zero, when Validate did not
re-measure a base leg. The Validate/base → final pair remains the optimization
metric and the source of the headline `throughput_speedup`; it cannot replace
the Setup-derived handoff verdict.

### Same config is not the same stack

Both harnesses can apply the identical flags and environment and still serve
different engines, because the orchestrator launches its server through its own
script and that script — not the transferred config — owns the platform kernel
preset, `--trust-remote-code`, and the gpu-memory-utilization default. GEAK's
native backend adapter may resolve those settings differently. Matching flags
alone therefore does not establish that both measurements used the same serving
stack.

`serving_stack` makes that legible without reading a server log. `launcher`
says who launched (`magpie` = the orchestrator's own script, `native` = GEAK's
adapter), and each leg's `aiter_mentions` / `kernel_picks` record which kernels
the engine actually selected. A near-zero `aiter_mentions` next to a large
negative divergence is the signature of an unaligned recipe, and
`baseline_alignment.status` reports `warning_recipe_unaligned` for exactly that
case so the number is not read as a measurement problem.

Set `handoff.bench_launcher="native"` to select the backend adapter explicitly.
`BENCH_LAUNCHER=native` supplies that selection when the handoff does not specify
a launcher.

### When there is no result to report

When no workflow return exists and none can be rebuilt from disk (see Handoff
resilience below), `result.json` reports the failure instead:

```jsonc
{
  "schema_version": 2,
  "status": "timeout | error",
  "error_class": "timeout | llm_unreachable | agentx_client_unavailable | workflow_parse_error | cli_failed | sdk_import_failed | interrupted | normalize_failed | runner_error",
  "error": "no workflow return within the 32400s budget (furthest phase on disk: search)",
  "phase_reached": "setup | search | finalize | validate"
}
```

* `llm_unreachable` → Claude Code could not reach the model API (connection
  refused or dropped, DNS, no route, proxy or TLS failure). `error` keeps the
  original message, which contains Claude Code's `API Error: ...` reply.
* `agentx_client_unavailable` → an AgentX handoff's aiperf cannot start, so the
  run stopped before any server launch. `error` names the client and why: it is
  missing, not executable, or in a venv whose interpreter is gone.
* `phase_reached` → the furthest phase the eval dir shows: `setup` until Setup's
  baseline summary exists, `search` until the final phase fences the GPUs
  (`.final_phase_fence`), then the fenced stage. Absent when the eval dir does
  not exist.

## Handoff recovery

The standalone runtime writes its top-level return to
`<eval_dir>/runtime_result.json`. The Python runner reads that file first, then
tries the runtime's `WORKFLOW_RESULT` output, then the workflow's persisted
`<eval_dir>/workflow_return.json`.

When invocation fails or times out, recovery can also reconstruct the return
from this run's retained `director_e2e_validation.json`, serving configuration,
and overlay artifacts. Recovery requires usable artifacts and does not invent
missing measurements. Recovered results set `recovered_from_disk=true`; a
completed final measurement remains authoritative even when it shows no gain.

## Agent call telemetry

The standard runtime writes `<eval_dir>/runtime_metrics.json` alongside its
result. Runtime metrics and backend traces are described in
[`runtime/engine/README.md`](runtime/engine/README.md).

The Claude-home mirroring described below applies when native Claude Workflow
records are available, including legacy prompt-only invocations. Standard
runtime dispatch does not depend on those records. Their absence does not mean
that the runtime made no agent calls.

Native Claude Workflow records live in `$CLAUDE_CONFIG_DIR` when set, otherwise
`~/.claude`, as `projects/<slug>/<session>/workflows/wf_*.json` plus per-agent
transcripts under `subagents/workflows/<runId>/`.

That home is a directory the run does not own and whose lifetime it does not
control. When it is a container overlay, an 18-hour run's entire cost record
dies with the container while `exp_root`, on durable storage, sits there
holding everything except the bill.

So the run mirrors it into its own output:

```
<eval_dir>/llm_trace/
  manifest.json
  projects/<slug>/<session>.jsonl
  projects/<slug>/<session>/workflows/wf_<id>.json
  projects/<slug>/<session>/subagents/workflows/wf_<id>/agent-*.jsonl
```

The layout is not free-form — it reproduces the Claude home from `projects/`
down, because run discovery globs `projects/*/*/workflows/wf_*.json` and derives
the transcript directory *relative to the record it found*. A flat dump would be
unreadable. Read it back as an extra Claude home:

```bash
python3 interface/geak_report.py --eval-dir <eval_dir> --claude-home <eval_dir>/llm_trace
```

| When | What happens |
| --- | --- |
| Every `TaskNotificationMessage`, at most once per `GEAK_TRACE_MIRROR_INTERVAL_S` | Incremental copy of whatever grew. Cheap: unchanged files are skipped by size+mtime. |
| `_emit()` — the guaranteed final flush | Full pass, then the run's report page (`<eval_dir>/report/geak_run_report_<model>.html`) is re-rendered by `interface/geak_report.py`, now that the workflow record exists. Reported in `result.json` as `claude_trace`. |

Selection is an identity match on the record's own `args.eval_dir` /
`args.exp_root`, never a guess by mtime, so a session driving several runs
mirrors each run's transcripts into that run's directory and no other's.

The whole path is best-effort by construction: every failure is recorded in the
manifest or in `result.claude_trace_error` and none of it can raise into the
run. Telemetry must never be the thing that kills an optimization job.

| Env var | Default | Effect |
| --- | --- | --- |
| `GEAK_TRACE_MIRROR_INTERVAL_S` | `900` | Mid-run mirror period. `0` disables the mid-run pass; the final one always runs. |
| `GEAK_TRACE_MIRROR_MAX_MB` | 4096 | Per-pass byte ceiling. An over-budget file is named in the manifest, never truncated — a truncated `wf_*.json` fails to parse and a truncated transcript silently understates a token total. |
| `GEAK_TELEMETRY_WARN` | `1` | `0` silences the startup durability warning. |
| `GEAK_CLAUDE_CONFIG_DIR` | unset | Opt-in: sets `CLAUDE_CONFIG_DIR` for the Claude child process only. Point it at a *seeded* directory — a fresh empty one has no credentials. |

At startup the runner compares the filesystem of the resolved Claude home with
that of `exp_root` and warns on stderr when they differ, since a ledger on a
different device has a different lifetime from the run that produced it. It
warns and continues, always. Note that `CLAUDE_CONFIG_DIR` is read by Claude
Code **at session start**: exporting it after the fact has no effect, which is
precisely why the mirror does not depend on anyone having set it.

## Outcome report (what the run bought)

The call telemetry above answers what a run *cost*. `interface/geak_outcome_report.py`
answers the other half — what each phase *bought* — and `_emit()` writes it at
the end of every run:

```
<eval_dir>/reports/geak_outcome.json     machine-readable
<eval_dir>/reports/geak_outcome.md       the tables
<eval_dir>/reports/SKILL.md              how to rebuild and read both reports
```

It reads only this run's own measured artifacts — `baseline/bench_summary.json`,
`config/sweep_results.json`, `kernels/*/opbench_result.json`,
`tuning/tuning_result.json` — and joins them to the per-phase spend when
the run's own ledger, `reports/trace/llm_calls.jsonl`, is present. Two rules keep it honest:

- **Absent is not zero.** A missing artifact renders as `—`. "We did not measure
  it" and "it contributed nothing" are different claims and conflating them is
  how a phase that spends most of the budget for a measured 0.00 % ceiling stays
  invisible.
- **Phases do not join end to end.** Each measures its own before/after in its
  own server session, so one phase's `after` need not equal the next phase's
  `before`. Those seams are printed, and the compounded speedup is marked an
  estimate whenever one exists. `observed_delta_pct_first_to_last` is the
  measured figure.

It runs on a run whose Claude ledger was lost, since it needs none of it, and it
can be re-run over an archived run at any time:

```bash
python3 interface/geak_outcome_report.py <EVAL_DIR> [--stdout]
```

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

## Measurement alignment with the caller

Comparability depends on the resolved workload, launch configuration, metric,
client, and measurement protocol. A shared branch or matching client name alone
does not establish it.

| Setting | Source |
|---|---|
| primary metric | `E2E_METRIC`; an AgentX `workload_spec.metric_basis` supplies the corresponding metric when the environment has no selection |
| latency | `ttft_ms` / `tpot_ms` median |
| dataset/workload | synthetic `workload.isl/osl/conc`, or the corpus/scenario in an AgentX `workload_spec` |
| synthetic prompt count | explicit `bench_protocol.num_prompts`; otherwise the schema-specific adaptive rule or standalone default |
| warmups, seed, range ratio | `bench_protocol`, subject to the schema-specific overrides documented above |
| TP | same tensor-parallel as the caller (no TP=1 lock) |
| parity | deterministic decoding / temp=0 fixed-seed output comparison with baseline; independent of the kernel optimizer policy |
| bench client | `handoff.bench_client`, resolved and exported as `BENCH_CLIENT` |

### Benchmark client adapter

The serving stack is launched through the selected backend adapter
(`adapters/sglang.sh`, `vllm.sh`, or `atom.sh`). The **client** that drives the timed bench is
selected independently:

* `native` uses the selected serving backend's built-in benchmark client.
* `inferencex` uses `InferenceX/utils/bench_serving/benchmark_serving.py` through
  `adapters/clients/inferencex.sh`. The selected checkout and resolved settings
  still determine comparability with the caller.
* `agentx` replays the declared AgentX workload through its separately installed
  `aiperf` client.

`handoff.bench_client="auto"` selects `agentx` for
`workload_spec.kind="agentx_trace_replay"`; otherwise it selects `inferencex`
when an InferenceX path is available, or `native`. The handoff selection is
resolved before exporting `BENCH_CLIENT`, so an inherited `BENCH_CLIENT` does
not override `auto`. AgentX workload settings carry the corpus, scenario,
durations, concurrency, and warmups; synthetic shape placeholders do not replace
that replay contract. The resolved client is recorded in `result.bench_client`.

### Server-LAUNCHER adapter (closes the launch-recipe residual)

The client adapter above aligns *who measures*; this one aligns *what is being
measured*. `BENCH_LAUNCHER=magpie` makes `adapters/launchers/magpie.sh` run the
orchestrator's own launch script for every fresh server, with the authored-kernel
overlay prepended to `PYTHONPATH` (which the orchestrator's own path cannot do),
so recipe parity and overlay application coexist. One adapter serves every
backend, because the scripts share one server-phase contract.

ATOM additionally keeps a GEAK-owned supervisor around the process group returned
by the Magpie script. ATOM's multiprocessing leader can exit before its rank workers
on SIGTERM; the supervisor drains that external group through SIGKILL when necessary,
preserving the native ATOM adapter's worker-safe teardown behavior.

`handoff.bench_launcher` selects the launcher first, falling back to
`BENCH_LAUNCHER` and then automatic selection. The script is resolved from
`handoff.launch_server_script`, `MAGPIE_LAUNCH_SCRIPT`,
`MAGPIE_<BACKEND>_SCRIPT`, or finally `handoff.launch_recipe`. A recipe can name
both the InferenceX checkout and the script inside its `benchmarks/` directory.
`serving_stack.launch_script_source` records which of those sources won.

Resolution degrades to `native` rather than failing whenever the script cannot
be confirmed usable — recipe unreadable, checkout not present on this box, or
the script's `benchmark_lib.sh` sibling missing. `BENCH_LAUNCHER=native` forces
that degrade explicitly and is the escape hatch.

`MAX_MODEL_LEN` is forwarded to the script on the `magpie` path only, because
the script's own default (4096) has nothing to do with the run and the
orchestrator overrode it by env when it measured the reference. GEAK does not
invent a separate gpu-mem-util value on this path; the selected SGLang/vLLM/ATOM
recipe script and its recorded `EXTRA_<BACKEND>_ARGS` remain authoritative. The script writes the server
to `$LOG` and its own trace to `magpie_launch.log` next to it, because the
script's redirect truncates `$LOG` and would otherwise destroy anything the
adapter wrote there.

GPU pinning on the magpie path is **shape-dependent**. If
`ROCR_VISIBLE_DEVICES` is already set in the launcher's shell (GEAK CI:
`run_local.sh` docker `-e ROCR_VISIBLE_DEVICES=<physical ids>` while `/dev/dri`
is fully passed through), `$GPU` is a *logical* index into that already-sliced
set — the launcher keeps the inherited ROCR and stacks `HIP_VISIBLE_DEVICES=$GPU`
on top (and re-asserts the outer ROCR after `$EXTRA_ENV` so an accepted-env leak
cannot clobber the mask). If no outer ROCR is present (bare Magpie / whole-box
Hyperloom), `$GPU` is *physical*: pin with `ROCR_VISIBLE_DEVICES=$GPU` alone and
clear HIP/CUDA so Magpie can derive the logical HIP range. Unconditionally
rewriting ROCR in the nested-CI shape would re-index the full physical set and
land on cards the job was never given.

Recorded recipe `PATH` entries are existence-checked before replay: missing
directories are dropped, and `PATH` is omitted entirely when nothing remains so
the ambient process `PATH` stands. GEAK's own CI entrypoint
(`ci/node/run_geak_e2e.sh`) exports `BENCH_LAUNCHER="${BENCH_LAUNCHER:-native}"`
and pins the same value into the patched handoff, so a shipped
`baseline_config.with_envs.yaml` cannot silently flip the CI server start path
to magpie.

### Retained measurement evidence

A completed final remeasurement takes precedence over an intermediate claimed win,
including when it shows no gain. A low final is not a missing result and must not
trigger recovery of an earlier higher number. Cold baseline/final diagnostics need
a matched same-session pair; otherwise their pairing limitation stays visible.

The optional `correctness_baseline` result identifies the original serving flags,
environment and overlay retained by the workflow. Empty strings are known empty
settings; `null` means the old state did not retain that value. `known` describes
those stored settings, not proof that correctness passed. The seed flags/environment
are the Director-reported launch settings (or explicit initial inputs); this field
does not independently reconstruct the running server configuration. It does not replace
`baseline_config`, which records the caller's configuration.

The retired `ablation_*` study controls now fail at startup. Remove these keys
from new handoffs; do not reinterpret or resume old study records on the new branch.
