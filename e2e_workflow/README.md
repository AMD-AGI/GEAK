# GEAK-evo end-to-end serving workflow

This JavaScript workflow profiles and optimizes sglang/vLLM/ATOM serving on AMD
Instinct GPUs and provides product-specific gfx1201 support for Radeon AI PRO
R9700 and R9600D (vLLM only).
It runs through the bundled Node runtime and delegates extracted kernel tasks
through the shared `kernel_workflow`. Generic gfx1201 identifies an ISA, not a
supported serving product.

## System and kernel layers

- **System layer** (this directory): owns server setup, profiling, configuration,
  kernel extraction, reversible integration and end-to-end measurement.
- **Kernel layer** (`../kernel_workflow/`): optimizes each extracted task with its
  correctness and performance checks, then returns the final patch and result.
  Omitted `optimizer` selects `evo`; `"optimizer": "greedy"` selects the original
  search policy. The same selection applies to nested kernel workers.

Passing `kernel_path` without `model_path` delegates directly to the kernel layer.
Existing repository paths, the `geak` package and `GEAK_*` settings are retained.

## Selecting useful workload changes

Profiled GPU-time share helps prioritize kernel work. Converting a kernel speedup
into a workload estimate also requires the timing boundary, call distribution and
overlap assumptions. Configuration, cache and scheduling changes can alter those
costs. Integration measures the full serving workload under the same request and
correctness contract. See `knowledge/e2e_optimization.md`.

## Roles → workflow mapping

These roles describe the fixed-load workflow. AgentX uses the extraction route
[described below](#agentx-kernel-extraction-and-ut-delivery).

- **e2e Director** = setup (isolated eval dir + TRUE baseline throughput) + final independent
  throughput validation/arbitration + output-parity gate.
- **System Architect** = strategy: read the standardized Top-N, route by Amdahl into config/kernel/
  host tracks, per-milestone planning + stop rule, and the **persistent cross-run experience library**
  (`knowledge/backend_playbook.md`, grown after every run).
- **Profiler** = warm-server trace (torch + optional rocprofv3) → ONE standardized Top-N artifact via
  `scripts/parse_profile.py` (the "spec" contract).
- **Config Tuner** = Tier-0 flag/env/backend sweep, runs FIRST (default ON), no source rewrite.
- **Kernel Extractor** = capture real shapes + a reference I/O oracle → an IMMUTABLE standalone
  unittest task dir the kernel layer consumes (anti-cheating).
- **e2e Integrator/Validator** = reversible overlay reintegration + e2e throughput gate + final bundle.
- **Kernel squad** = the shared `../kernel_workflow/kernel_workflow.js`, invoked recursively.

## Pipeline

Fixed ISL/OSL workloads:

```
Setup(preflight env-check + baseline throughput) → Baseline Profile(Top-N) → Strategize(Amdahl routing) →
ConfigSweep(flags/env/backends, FIRST) → Re-profile →
LOOP milestone[ plan → per kernel: Extract → recursive kernel_workflow.js → Overlay+e2e gate → ] → Re-profile → grow playbook →
Finalize(overlay+patch+launch bundle) → Architect Report → Director Validation
```
AgentX trace replay instead uses baseline measurement → extraction and UT
validation → head/kernel optimization → E2E validation. It skips the outer
Profile/Strategize, ConfigSweep and Tune phases.

Setup runs a **preflight** (see `knowledge/preflight.md`) — a judgment-guided env self-check (not a
rigid script): it confirms the chosen `backend` stack, the model, GPU visibility; verifies the
caller-established structured product/ISA/CU identity, detects trace
sources, available op backends, and the model's arch class; degrades gracefully and writes
`env_report.{md,json}` that every later phase routes on.
Every accepted change compounds into the carried-forward overlay + config; throughput is always
measured on a warm server, in the same lifecycle as the TRUE baseline it is compared against.

## Pluggable serving backend
The serving stack is NOT baked in. `args.backend` (sglang|vllm|atom, default sglang) selects
`scripts/adapters/<backend>.sh`, which `scripts/bench_e2e.sh` (a backend-agnostic dispatcher: owns
server lifecycle, warmup, timed round(s), summary, free-port allocation) sources. Adding a new
stack = adding one adapter that defines `adapter_launch / adapter_health / adapter_bench`
(+ optional `adapter_default_port`). No role or orchestration change. `MODEL` is **required** — there
is no rig-specific default that could silently bench the wrong target.

ATOM profiling is configured through `--torch-profiler-dir` at server launch and
`ATOM_PROFILER_MORE=1` for shape-bearing traces. It writes a trace below a separate `rank_<N>` (or
`dp<D>_tp<N>`) directory for each worker; `adapter_profile_window` waits for every expected gzip to
finalize and writes `atom_profile_manifest.json`. A missing or incomplete manifest is a failed ATOM
profile rather than permission to continue with a partial rank-0 trace.

## The three backend dimensions (per spec)
A kernel's backend can be changed from three places, in increasing cost (knob names are backend-
specific — see `perf_knowledge/backends/<backend>/` + `perf_knowledge/reference/env_vars.md`, as
reference only; verify every switch by measuring):
1. **launch flags** (`--attention-backend`, `--quantization`, …) — Config Tuner
2. **env vars** (sglang `SGLANG_USE_AITER`, vllm `VLLM_ROCM_USE_AITER`, ATOM
   `ATOM_USE_TRITON_{MLA,MOE,GEMM}` / `ATOM_ENABLE_*_FUSION`, `HIPBLASLT_TUNING_FILE`, …) — Config Tuner
3. **source** (a Triton/**FlyDSL**/HIP/CK/asm reimplementation) — Kernel Extractor + kernel squad,
   overlaid back reversibly (never editing site-packages). For a hot op with **no existing editable
   implementation**, the head track now **authors one from scratch**: the Op Benchmarker DISCOVERs
   existing impls + tunes cheap levers, then emits an `author_plan`; the orchestrator runs the kernel
   layer in **author mode** (`mode=author target_language=flydsl|triton|hip|ck`) to write a fresh
   baseline against the immutable oracle and optimize it, then the Integrator rebinds the op's call site
   to it and gates on e2e. Routing (direct_light tune vs author/rewrite via the kernel layer vs drop) is
   decided by Amdahl headroom + rewrite type. Triton is always a viable author target; **for a dense /
   quantized GEMM (esp. fp8/A4W4) FlyDSL is the preferred author target** (aiter's SOTA GEMM DSL — JIT,
   no build, baseline reuses `flydsl_hgemm`/`flydsl_preshuffle_gemm_a8`; FlyDSL is also one of the
   backends aiter's per-shape DB tune races, so it can also win via the cheap env lever with no author
   step). HIP/CK when the headroom justifies them (`head_author_max`, default 2 = FlyDSL+Triton).

## Invocation

External callers such as Hyperloom keep the standard handoff and result files:

```bash
python interface/run_e2e.py /path/to/handoff.json /path/to/result.json
```

The wrapper performs structured on-box identity detection and invokes the bundled
Node runtime. See the [integration contract](../interface/run_e2e.md) for handoff
fields and result normalization. Omitting `optimizer` selects `evo` without adding
new fields to an existing handoff.

For a direct run, save the following as `e2e-args.json` and replace the paths and
hardware settings. This example targets one MI355X GPU:

```json
{
  "model_path": "/path/to/model",
  "workflow_dir": "/absolute/path/GEAK/e2e_workflow",
  "eval_dir": "/absolute/path/runs/e2e-001",
  "expected_gfx": "gfx950",
  "expected_target": "mi355x",
  "backend": "sglang",
  "gpu_ids": "0",
  "tp": 1,
  "isl": 1024,
  "osl": 1024,
  "conc": 64,
  "budget": 4,
  "kernel_budget": 6
}
```

From the repository root:

```bash
node interface/runtime/engine/run_workflow.mjs \
  e2e_workflow/e2e_workflow.js \
  --agent claude --args-file e2e-args.json \
  --result-file e2e-workflow-result.json
```

The Node result is the raw workflow return; the Python wrapper produces the
normalized integration result. Use a fresh `eval_dir` for each run. See
[runtime setup](../interface/runtime/SETUP.md) for agent/model selection. The
bundled runtime is required even with Claude: kernel delivery uses `runProcess`,
which the native Claude `Workflow` tool does not provide.

Direct model-mode runs require both `expected_gfx` and `expected_target`.
At Setup/Freeze, `expected_target: "unknown"` leaves the product unconstrained;
pin `r9700`, `r9600d`, `mi350x` or `mi355x` to require that product. The gfx1201
serving entry requires an explicit R9700 or R9600D identity before setup. `workflow_dir` must name this
folder; `kernel_workflow_dir` defaults to its sibling `kernel_workflow/`.

Common optional fields in the JSON arguments:

| Field | Purpose |
|---|---|
| `optimizer` | `"evo"` by default; `"greedy"` selects the original kernel search policy. |
| `launch_script` | Use a supplied serving launch script. |
| `budget`, `kernel_budget` | Maximum kernel tasks and the direction budget for each nested task. |
| `milestone_min_pct` | Fixed-load Milestone GPU-time threshold; default 5. AgentX uses its extraction inventory. |
| `config_tune` | Fixed-load configuration sweep; default `"true"`. AgentX skips it. |
| `tuning_skillset` | Fixed-load tuning phase; default `"true"`. AgentX skips it. See below. |
| `tuning_kb` | Consult the per-model tuning KB; default `"true"`. Use `"false"` for blind evaluation. |
| `analysis_skill` | Profile analysis, default `"roofline"`; `"none"` disables it. |
| `use_expert_skills` | Advisory expert priors, default `"false"`; forwarded to kernel workers. |
| `task` | Additional optimization focus. |
| `apply_to_original` | `"true"` requests an apply bundle; integration uses reversible overlays. |

## Standalone AgentX (trace replay, no orchestrator)

`isl`/`osl`/`conc` describe a **synthetic sweep**: every request is the same shape, and the profile and
the bench use it identically. An AgentX submission is not that. It replays a recorded corpus, so the
client owns the request mix, the arrival pattern, its own warmup and the measurement duration.
Standalone it is graded on **InferenceX P90 interactivity** — `p90_intvty_inferencex`, i.e.
`1000 / P90(ITL)` in tok/s/user, decode only, the x-axis of the InferenceX pareto — with total
(input+output) tok/s carried beside it as `guard_total_tok_s_median`. The output-only axis a sweep
uses is a poor fit: on a ~140:1 prefill-to-output corpus it barely moves for a large change in work.

Under Hyperloom this arrives as environment variables that `interface/run_e2e.py` exports.
For a standalone run, `"workload_kind": "agentx_trace_replay"` fills the canonical setup.

Save an `agentx-args.json` with the full argument object, then use the same Node
command with `--args-file agentx-args.json`. For an MI355X TP8 example:

```json
{
  "model_path": "/models/Kimi-K3",
  "workflow_dir": "/absolute/path/GEAK/e2e_workflow",
  "eval_dir": "/absolute/path/runs/agentx-001",
  "expected_gfx": "gfx950",
  "expected_target": "mi355x",
  "backend": "vllm",
  "gpu_ids": "0,1,2,3,4,5,6,7",
  "tp": 8,
  "workload_kind": "agentx_trace_replay",
  "conc": 8
}
```

Note there is no `isl`/`osl` here. Nothing is measured at them on a trace replay — aiperf owns the
sequence lengths — but the roles still read them when they size tiles and synthesize GEMM shapes, so
leaving them at the synthetic `1024/1024` default would aim the whole kernel search orders of
magnitude below the real load. So the run **measures its own shape**: `bench_e2e.sh` reports the
average it actually served as `observed_isl`/`observed_osl` in `bench_summary.json`, and right after
Setup the workflow reads that off the baseline and adopts it, with
`shape_provenance=agentx_measured_this_run`. This is why no corpus average is hardcoded anywhere: the
number moves with the corpus, the tokenizer and the window, and a constant in the source would be
read as fact by every role long after it went stale. All kernel work is scheduled after the baseline,
so the measurement always exists before anything needs it.

Until the baseline lands the shape is openly `1024/1024` with
`shape_provenance=agentx_pending_baseline`, and role prompts mark it a placeholder that must not be
used for kernel sizing. If the baseline reports no shape at all (a client that never returned token
totals), the run continues on that placeholder and says so loudly rather than inventing a number.

To pin a shape measured elsewhere or deliberately target a regime, merge this field
into the argument object. The baseline measurement will not override it:

```json
{
  "workload_spec": {
    "kind": "agentx_trace_replay",
    "corpus": "<yours>",
    "observed_isl": 120000,
    "observed_osl": 800
  }
}
```

This records `shape_provenance=agentx_observed`.

`workload_kind` expands to the graded setup: scenario `inferencex-agentx-mvp`, corpus
`semianalysis_cc_traces_weka_062126`, 393 entries, a 3600s canonical window with 900s inner search
legs, and `p90_intvty_inferencex` as the metric basis. Override any part with the long spelling,
which is the same object Hyperloom puts on its handoff:

```json
{
  "workload_spec": {
    "kind": "agentx_trace_replay",
    "corpus": "semianalysis_cc_traces_weka_062126",
    "num_entries": 393,
    "duration_s": 3600,
    "geak_loop_duration_s": 900,
    "concurrency": 8,
    "metric_basis": "p90_intvty_inferencex",
    "canonical_corpus": "semianalysis_cc_traces_weka_062126",
    "canonical_num_entries": 393,
    "canonical_duration_s": 3600,
    "warmup_requests_per_lane": 10,
    "warmup_grace_period_s": 1800,
    "failed_request_threshold": 0.10,
    "inferencex_path": "/opt/InferenceX",
    "aiperf_bin": "aiperf",
    "profile_warmup_s": 2700,
    "profile_window_s": 20
  }
}
```

Merge `workload_spec` into the full argument object. Its `concurrency` overrides
`conc`; `duration_s` is the canonical validation window and `geak_loop_duration_s`
is the inner search window. Omit `inferencex_path` to use the vendored mapper and
omit observed shape fields to measure them from the baseline.

Three parts of that object do more than they look:

- **The axis.** A declared `metric_basis` wins. Without one, an `E2E_METRIC` already exported in the
  environment decides, since an exported value outranks `bench_env.sh` and is what the bench would
  measure anyway. A declared basis that disagrees with an exported `E2E_METRIC`, or an exported value
  `bench_summarize.py` cannot measure, is refused at start rather than an hour into the baseline. The
  acceptance noise band on a trace replay defaults to 4%, and the band the Director reports can only
  widen it: two same-config baseline replicas differed by 3.86% on P90 interactivity (3.43% on total
  tok/s). `noise_band_pct` sets that floor instead; the TTFT-inclusive `e2e_norm_intvty_p90` moved
  17% on the same pair, so pass one if you grade on it.
- **The reference.** Overriding `corpus`, `num_entries` or `duration_s` does not move what a result is
  stamped canonical against, so a replay of anything else comes back `submission_valid=false`. To
  change the reference itself (a ~256k-context server's `_256k` corpus, say), set `canonical_corpus` /
  `canonical_num_entries` / `canonical_duration_s` too.
- **The mapper.** An `inferencex_path` checkout's `map_aiperf.py` wins over GEAK's vendored copy, and
  upstream's own fallback mapping carries no P90 ITL, so outside Hyperloom's runtime it cannot grade
  the default axis. `agentx_smoke.sh` maps a sample export through whichever copy the run would use
  and blocks when the graded field is missing.

**Check the box first.** Every AgentX prerequisite is invisible until the client has already launched
and warmed a server, so a missing one costs 20+ minutes to discover:

```bash
bash e2e_workflow/scripts/agentx_smoke.sh                    # environment only
MODEL=/models/Kimi-K3 bash e2e_workflow/scripts/agentx_smoke.sh
BASE_URL=http://127.0.0.1:8000 bash e2e_workflow/scripts/agentx_smoke.sh   # + a real short replay
```

It never launches a server. Its most important check is that `aiperf` actually supports `--scenario`:
a stock aiperf cannot replay the corpus at all, but pointed at a server it will still report a
throughput — a "successful" run that measured a different workload.

A handoff run checks the client before anything launches. `run_e2e.py` resolves the `aiperf` the
bench will run (`AIPERF_BIN`, else `aiperf_bin`, else `aiperf` on `PATH`), and when that is an absolute
path that is missing, not executable, or in a venv whose interpreter is gone, it writes
`error_class: agentx_client_unavailable` instead of launching a server. A bare name missing from
`PATH` only warns, since the bench's `PATH` can differ; `GEAK_STRICT_AGENTX_PREFLIGHT=1` refuses it too.

### What the declaration changes

The workflow composes one `bench_env.sh` and the Director writes it beside the copied bench script;
`bench_e2e.sh` and `bench_replica.sh` source it from next to themselves, so every bench in the run —
including the ones role agents issue from their own prompts — selects the trace client and the graded
axis. Each line assigns only an unset-or-empty name, so anything already exported still wins and you can
override one knob on the command line. Every role prompt also gains a workload-identity block stating
what the run measures. **Without a declaration none of this exists**: no file is written, the source is a
no-op, the prompts are byte-identical, and the fixed ISL/OSL path is exactly as before.

### Two things that surprise people

**`isl`/`osl` are not a benchmark here.** A trace corpus spans orders of magnitude in request length,
so there is no single ISL — read the real distribution off the run's own export. They are only the
shape to optimize for: what roles synthesize GEMM inputs and pick tiles against. Nothing is measured
at them, and measured on this stack a synthetic sweep at those lengths came out ~2.76x the
trace-replay throughput, so it is not a cheaper approximation of the real number.

**Search legs are non-canonical by design.** They run the scenario's 900s floor instead of the canonical
3600s, so the client stamps them `submission_valid=false` with the deviations attached. Both legs of an
A/B deviate identically, so the comparison is sound; the number simply cannot be reported as a
submission. Only a `parity`/`validation` leg runs the full window.

Profiling places its window at 75% into the replay (2700s on the canonical leg), because at load start
aiperf is still resolving the corpus and draining per-lane warmups — a window opened there records the
ramp, not the steady state. If the replay ends before the window opens, GEAK captures **no** trace rather
than substituting a synthetic one that would look authentic.

## Modes: default · fast · deep
One pipeline, three depths — selected by the `fast_mode` / `deep_mode` args (both default `false` =
**default** mode; they are mutually exclusive, **deep takes precedence**). Only the HeadKernel depth and
which phases run change; the throughput metric, the e2e gate, and the reversible-overlay contract are
the same. With both off, the standard phase sequence runs. The kernel `optimizer`
is a separate choice and defaults to `evo`.

| Mode | arg | Phases | HeadKernel | Budget (default) | Use when |
|---|---|---|---|---|---|
| **default** | *(none)* | ConfigSweep + HeadKernel + **Milestone** | serial, 1 pass/head, ≤2 authored langs, single e2e gate | — | full pipeline incl. the editable-kernel Milestone loop |
| **fast** | `fast_mode:true` | HeadKernel only (skips ConfigSweep + Milestone) | **parallel** head track (extract/bake/author fan out across GPUs), time-capped | `fast_budget_ms` = 5h | a quick HeadKernel-only win under a wall-clock cap |
| **deep** | `deep_mode:true` | ConfigSweep + HeadKernel (skips Milestone) | **global cross-kernel×backend lane pool** — every (head op × backend) optimizes in parallel, many rounds via STATE_DIR + reseed, cross-pollination (per-op SHARED_KB + run-global GLOBAL_KB), convergence-stop + agent-budget backstop, finalize banks a **combined cross-kernel overlay** | `deep_head_budget_ms` = 24h | the deepest/broadest result: most backends, most rounds, hours available |

To select a depth, add `"fast_mode": true` or `"deep_mode": true` to the JSON
argument object. Optional `fast_budget_ms` and `deep_head_budget_ms` values are
milliseconds; for example, `"deep_head_budget_ms": 64800000` sets an 18-hour cap.

Pick **fast** for a bounded quick pass, **default** for the standard run, **deep** to chase the best
achievable number (it is broader = more backends, deeper = more/faster rounds, parallel = lanes co-opt
spare GPUs while the e2e gate runs on the serving slot, with matched in-window A/B so parallelism never
corrupts a measurement).

## Tuning skillset (`tuning_skillset`, default ON)

A dedicated phase — **after ConfigSweep, before HeadKernel** — that runs the tuning skillset vendored at
`<repo>/perf_knowledge/expert_skills/tuning/`: an independently-validated method for tuning GPU ops on AMD Instinct (per-op
tuners, deploy paths into a live server, and the engagement checks that prove a tuned artifact is
actually what the machine runs).

**It is vendored whole and run whole.** The tree is byte-identical to the standalone repo it is validated
in (37 executable claims in `validate/claims.py`), hash-pinned by
`e2e_workflow/knowledge/tuning_skillset.manifest.sha256`, and consumed by exactly one thin adapter role,
`roles/tuning_specialist.md`. Its method is deliberately **not** paraphrased into `knowledge/`: a
standalone validation is evidence about a specific tree, and scattering it would keep the prose while
losing the guarantee. Fixes go upstream and come back via
`scripts/tuning_skillset_sync.py --sync <dir>`; never edit inside the vendored tree.

**Why standalone, and why before the head track.** The skillset is a complete six-step loop (scope →
baseline → search → correctness gate → deploy → engagement verify). Folded into the head-kernel bake-off
it degenerates into "one more candidate config" and most of it never runs. As its own phase it runs end
to end, and — because it takes its **own** in-session interleaved pre/post A/B — the final report can say
how much of the run's gain came from tuning. Running it first also means the expensive head budget goes
to the ops still hot *after* tuning (an accept triggers a re-profile, exactly as a config win does), and
its accepted env/flags fold into the carried config so every later A/B measures on top of a tuned stack.
Corollary when reading the report: the tuning delta and the head deltas are **not** additive — tuning is
already in the head track's reference leg.

An accept is not taken on trust. The orchestrator withholds it unless engagement was **proven**, both A/B
legs completed, correctness passed, and `post > pre > 0`; otherwise it is downgraded to `no_win` and the
run continues on the pre-tuning config. An unproven tuning artifact would otherwise sit silently in the
reference leg of every downstream measurement — which is the exact failure the skillset exists to catch.

**Scope.** The phase does not author kernels — that is the head/Milestone tracks' job, unchanged. It does
own the dispatch path: which kernel the seam selects, with which parameters, **and the code that makes a
tuned artifact actually bind** (a routing switch, a wrapper that drops the kernel selection). That code
half travels as a reversible overlay (`apply_overlay` → carried `curOverlay` → `final/overlay`), never as
a live-tree source edit. It is not a corner case: the skillset measures a correctly-tuned row deployed
behind a wrapper that ignores the kernel selection running **85.7% slower than doing nothing**, with every
engagement gate passing.

**How the win ships.** The data half of a tuning win is a config table a library reads from inside its
own package dir, often with a derived cache that must be dropped or the new rows are silently ignored.
That cannot travel in `final/overlay`, which is a `PYTHONPATH` mechanism for *code*. So the phase writes
a deploy bundle (`EVAL_DIR/tuning/deploy/`: manifest, git-applyable diff, the files, and an idempotent
`deploy.sh`) and Finalize folds it into the **existing** deliverable handles — the diff is concatenated
into `final_patch.diff`, and `final_launch.sh` runs `deploy.sh` before the server starts. Finalize then
re-checks engagement on the assembled bundle, so a bundle that silently drops the tuning is reported as
broken instead of shipped as complete. Callers reusing `final_launch.sh` need no extra steps.

Results land in `EVAL_DIR/tuning/`, in `final_report.md` **§2b**, in
`wfReturn.tuning_skillset` (`tuning_delta_pct`, `tuning_speedup`, `share_of_total_gain_pct`), and in
`result.json` under a **purely additive** `tuning_skillset` key — no existing key changes name, type or
meaning, and a run without the phase produces a byte-identical `result.json`.
Fast mode skips the phase; deep mode keeps it for fixed-load workloads. AgentX skips
it as described below. Full contract:
`knowledge/tuning_skillset_integration.md`.

## AgentX kernel extraction and UT delivery

An explicit `workload_kind: "agentx_trace_replay"` (or matching workload specification)
activates the [kernel-extract skill](knowledge/kernel_extraction_skills/kernel-extract/SKILL.md).
The extractor follows the complete procedure: measured baseline logs
→ observed workload → sampling points → runtime kernel selection → source extraction
→ standalone GPU UTs. The method comes from the validated standalone skill at `ce1dbe5`. It retains the
five-stage procedure and the additional sampling, coverage, reference-output and
self-noise tools. Existing capture, trace, harness, GPU-lock and delivery helpers
are reused from the repository; their regression tests remain in one place.

UT inputs default to fixed-seed generators at the observed shapes/dtypes/layouts;
reference outputs come from the frozen baseline in an isolated process. UT execution
does not launch serving, profile or capture shapes. Reuse or reliably reconstruct
routing, indices, page tables and other structural metadata; capture only missing
state. Necessary captures default to `structure_only`, with reference regeneration
before validation. An explicit `capture_persist_policy` overrides the fallback
storage policy; non-AgentX capture defaults are unchanged.

GEAK supplies baseline/configuration paths, GPU allocation, accepted overlays and
budgets through the [workflow handoff](knowledge/kernel_extraction_skills/kernel-extract/references/workflow-handoff.md).
The skill completes scheduled UT work before returning a delivery path. The packaging
adapter reads each task's saved declaration and return, preserves failed/deferred
tasks and coverage, and feeds verified tasks into the existing optimization queues.
It checks source/binding/UT evidence without repeating selection or GPU tests; changes
to source, cases or recorded results invalidate reuse. Task-count budgets affect
scheduling, not the selected inventory or its reported coverage.

AgentX follows baseline → B extraction and UT validation → head optimization →
kernel milestones → E2E validation. Outer Profile/Strategize, ConfigSweep and Tune
are skipped. B still uses bounded trace sampling to establish kernel identity and
cost. All three head modes and the milestone loop reuse the prepared task directory
and immutable UT, running it against optimization candidates instead of authoring
another test. Workspace setup retains phase-local oracles, helpers and baseline
resources; head and milestone use the workload-weighted result. Missing or changed
preparation evidence fails the task locally. Current optimization lanes are single-GPU:
completed multi-rank UTs retain their artifacts and are deferred for separate scheduling.

The default selection remains per-phase Top 5 intersected with whole-workload
GPU-time share strictly greater than 3%, without backfilling. Explicit user scope
wins. Head routing uses the prepared task kind; milestones consume the remaining
kernel queue without Architect nomination or the fixed-load 5% filter. Optimization
wins do not trigger another extraction; scheduling uses the original measured order
and weights. Dispatched tasks leave the queue, pending tasks survive resume, and an
empty queue stops dispatch. Fast/deep modes retain their head-only scheduling.
Fixed ISL/OSL workloads keep their existing flow. Each optimization still runs its
existing UT and E2E A/B gate; a UT pass alone does not establish an E2E improvement.

## Roofline-guided routing (`analysis_skill`, default `roofline`)

`pct_gpu_time` locates cost. The pluggable [roofline skill](knowledge/analysis_skills/roofline/SKILL.md)
adds conditional work/traffic models and a second ordering for choosing experiments. Its historical
`attainable_speedup` field is a scenario based on an efficiency prior, not a measured limit.
`expected_e2e_gain_pct = pct_gpu_time × (1 − 1/attainable_speedup)` estimates a reduction in summed
profiled GPU time. It predicts workload latency only if that fraction matches the critical elapsed
interval and other costs and overlap remain unchanged. Measure the workload result separately.
The Architect reports both orderings and states which it followed.

Integration and recovery treat Amdahl projections as diagnostics. The legacy
`implausible_speedup_margin` controls discrepancy reporting and the independent
A/B review for an accuracy-gated gain. The projection alone cannot reject a
candidate or establish a correctness failure. Actual quality, parity, engagement
and complete-measurement requirements still apply. A review that ran and measured
a rejection withholds its candidate; a pending or inconclusive review leaves the
original measured verdict in effect. Investigate model discrepancies using work
completeness, timing scope and quality evidence.

**Roofline is advisory and never prunes a candidate.** Estimated shapes are low-confidence and
display-only; captured shapes with supported models can guide secondary prioritization. Matched
counters raise confidence only for quantities with validated definitions, units and capacities.
Unsupported quantities remain unknown, and missing tools or models do not fail the run.

A high model fraction can motivate hypotheses in `byte_reduction_levers`: avoiding intermediates,
reusing weights/KV, skipping unused experts or changing packing. It does not establish that local
tuning is exhausted or the current work is necessary. Keep the supplied workload and numerical
contract; broader fusion needs the surrounding computation and full-operation measurement.

The test fixture retains this historical example (Qwen3.5-35B-A3B-FP8, gfx950, vLLM TP1,
1k/1k, conc 64). These calculations illustrate ranking behavior; they do not calibrate a universal
roof or establish the model's predictive accuracy. See `scripts/tests/test_roofline_skill.py`.

| kernel | %GPU | model fraction | scenario speedup | modeled GPU-time reduction | measured |
|---|---|---|---|---|---|
| `fused_moe_kernel` | **26.45%** | 88% (memory model) | 1.02× | +0.6% | 1.047× iso, **−0.064% e2e** |
| `kernel_paged_attention_2d` | 8.86% | 18% | 2.8× | +5.7% | **1.56× iso** |

The scenario ranks attention first; `%GPU` ranks MoE first. Isolated speedups and end-to-end deltas
need their own repeat variability before drawing conclusions about workload benefit. Add a skill in
`knowledge/analysis_skills/`; swap with `analysis_skill=<name>`; disable with `analysis_skill=none`.

## Accuracy gate (gsm8k) — OFF by default
By default the e2e gate accepts a kernel on **throughput delta + greedy output parity**
(`accuracy_gate:"none"`). For QUANTIZED kernels (MXFP8/fp8) byte-parity is too strict — a within-tolerance
kernel rounds differently and flips a few borderline greedy argmaxes — so you can switch the bar to
**task accuracy** by adding these fields to the full JSON argument object:

```json
{
  "accuracy_gate": "gsm8k",
  "accuracy_limit": 200,
  "accuracy_tol": 0.01
}
```

- `accuracy_gate:"gsm8k"` → the Integrator serves a fresh TRUE baseline vs the candidate, runs sampled
  gsm8k (5-shot, greedy, fixed seed), and accepts iff `cand_em >= baseline_em - accuracy_tol`.
- `accuracy_limit` = #questions (default **200**; deep uses a larger sample at finalize to de-noise the
  boundary). `accuracy_tol` = allowed exact-match drop (default **0.01**).
- The eval client is `scripts/gsm8k_eval.py` (model-agnostic; queries the OpenAI-compatible endpoint).
- Leaving it unset (`"none"`) changes nothing vs before — the gate stays throughput + parity only.

An accuracy-gated speedup above the Amdahl estimate triggers **one independent A/B review**, not a
correctness rewrite. The existing `implausible_speedup_margin` option remains supported (default
`1.0`, review above twice the estimate; the runner forwards `GEAK_IMPLAUSIBLE_SPEEDUP_MARGIN`). A
confirmation uses fresh reference/candidate servers, at least two timed replicas per leg, and the
same correctness and engagement gates. If the gain reproduces, it is accepted even above the estimate.
Already confirmed evidence is reused only for the same candidate, reference, workload and measurement
protocol identities. The workload identity covers only what decides the served requests, so a profile
and an A/B of the same workload share it; bench/protocol details go in the separate protocol digest.
Missing provenance, a timeout or repeated old A/B artifacts leaves the review
inconclusive. A confirmation that cannot finish before the final reserve is not
started; each timed replica counts as a leg. These outcomes are recorded with the
candidate and leave its original measured verdict in effect. Only a review that
ran and measured a rejection withholds the candidate. No review outcome starts a
kernel rewrite. Final Validate still checks the combined stack.

The estimate uses the candidate's `affected_gpu_time` coverage when its profile matches the current
reference and workload: all affected shapes/phases and removed auxiliary kernels, counted once.
Per-path timings are used when available; applying the head speedup to unmeasured paths is explicitly
labelled an approximation. Stale/missing coverage falls back to the head estimate. The head's original
`pct_gpu_time` is preserved for ranking. Candidate checkpoints include the reviewed integration result,
so timeout recovery respects the same acceptance decision. See [the integrator contract](roles/e2e_integrator.md)
for coverage and measurement-evidence fields.

## Output
Everything lands under `<exp_root>/e2e_<model>_<timestamp>/`:
- `env_report.{md,json}` — the preflight capability report every phase routes on
- `baseline/bench_summary.json`, `env_info.txt`, `config/baseline_flags.json` — the TRUE baseline
- `profile/round_*/profile_topN.{json,md}` — the standardized Top-N each round
- `strategy.md`, `config/sweep_results.json`, `insight_log.md`
- `kernels/<short_name>_task/{kernel_src, reference_io.pt, unittest.py, meta.json}` — extracted tasks
- `kernels/_exp/…` — the recursive single-kernel runs (each with its own verified result)
- `overlay/…` — candidate + accepted reversible overlays
- `tuning/{tuning_report.md, env_audit.txt, claims_report.json, bench_pre/, bench_post/, <op>/}` — the standalone tuning phase + its attributable pre/post A/B
- `final/{overlay, final_patch.diff, final_launch.sh}` — the deliverable bundle
- `architect_report.md`, `director_e2e_validation.json` — the official verified throughput result

## Files
```
e2e_workflow.js   orchestration (deterministic; recursively calls ../kernel_workflow/kernel_workflow.js)
roles/                 director, system_architect, profiler, config_tuner, tuning_specialist, kernel_extractor, op_benchmarker, e2e_integrator
knowledge/             e2e_optimization, profile_parse, preflight (env self-check), backend_playbook + gemm_attention_backends (persistent), sglang_internals, shape_capture
knowledge/analysis_skills/  pluggable profile-analysis skills (INDEX.md + one dir per skill; `roofline` ships by default)
knowledge/tuning_skillset_integration.md  how the VENDORED ../tuning_skillset/ is wired in (+ its manifest)
scripts/               bench_e2e.sh (backend-agnostic dispatcher), adapters/{sglang,vllm,atom}.sh, parse_profile.py (Top-N), op_bench.py, capture_shapes.py, overlay_setup.py
scripts/adapters/clients/  pluggable BENCH CLIENTS (the timed load): agentx.sh (AgentX trace replay via aiperf) + map_aiperf.py vendored beside it so a standalone run needs no InferenceX checkout
scripts/agentx_smoke.sh  AgentX readiness check: run it BEFORE a trace-replay run (never launches a server). See "Standalone AgentX" above.
scripts/tuning_skillset_sync.py  integrity gate on the vendored skillset (--verify / --update / --sync)
scripts/server_teardown.sh  the shared server-kill contract (identity verified at LAUNCH: pid, pgid, /proc start time). Every script that launches a server, including role-authored capture scripts, must source it instead of hand-rolling a kill.
```

## Generality
The script never hard-codes a model or kernel. The workload (isl/osl/conc) drives both profiling and
benchmarking identically; the Profiler's classification + the Architect's Amdahl routing decide what
gets tuned vs rewritten. For a different model, only `model_path` (+ optional `launch_script`)
changes. The persistent `backend_playbook.md` carries learned per-class backend priors across runs.
