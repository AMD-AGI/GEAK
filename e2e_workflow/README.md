# e2e_workflow — End-to-End LLM Inference-Throughput Optimizer (AMD Instinct MI GPUs)

A deterministic **Workflow** (JS-orchestrated multi-agent pipeline) that raises the **sglang/vllm
serving throughput** of an LLM on AMD Instinct MI GPUs. It is a *system layer* built on top of — and recursively
calling — the UNCHANGED single-kernel `kernel_workflow` (`../kernel_workflow/`). The single-kernel workflow's
quality is preserved verbatim; this layer adds everything above the kernel: profiling a running
server, Amdahl triage, config/backend tuning, extracting hot kernels into standalone unittests,
optimizing them with the kernel layer, overlaying them back, and re-validating end-to-end throughput.

## Design: fractal two-altitude
- **System layer** (this dir): owns the server, the throughput metric, profiling, triage, config, and
  reintegration. Roles: e2e Director, System Architect, Profiler, Config Tuner, Kernel Extractor,
  e2e Integrator/Validator.
- **Kernel layer** (`../kernel_workflow/`, UNCHANGED): given a kernel task dir, does the real multi-backend
  optimization with independent verification. The system layer hands it an extracted task dir and
  consumes its verified `final_patch.diff` + geomean. Same contract as a hand-written kernel task.

Because the kernel layer is called as-is, single-kernel optimization effect cannot regress — and the
workflow is **backward compatible**: pass `args.kernel_path` (no `model_path`) and it delegates
straight to the kernel layer (single-kernel pass-through).

## Why a system layer at all (the doctrine)
e2e throughput is **Amdahl-dominated**: only a speedup on a kernel that is a large share of GPU time,
times how often that path runs, moves the headline number. A 5× on a 2%-of-time kernel is invisible.
So the system layer always reasons in `pct_gpu_time × achievable_speedup`, tunes the cheap
landscape-reshaping config knobs FIRST, and gates every kernel change on a measured end-to-end
throughput delta that exceeds the noise band. See `knowledge/e2e_optimization.md`.

## Roles → workflow mapping
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
- **Kernel squad** = the UNCHANGED `../kernel_workflow/kernel_workflow.js`, invoked recursively.

## Pipeline
```
Setup(preflight env-check + baseline throughput) → Baseline Profile(Top-N) → Strategize(Amdahl routing) →
ConfigSweep(flags/env/backends, FIRST) → Re-profile →
LOOP milestone[ plan → per kernel: Extract → recursive kernel_workflow.js → Overlay+e2e gate → ] → Re-profile → grow playbook →
Finalize(overlay+patch+launch bundle) → Architect Report → Director Validation
```
Setup runs a **preflight** (see `knowledge/preflight.md`) — a judgment-guided env self-check (not a
rigid script): it confirms the chosen `backend` stack, the model, GPU visibility; detects gfx, trace
sources, available op backends, and the model's arch class; degrades gracefully and writes
`env_report.{md,json}` that every later phase routes on.
Every accepted change compounds into the carried-forward overlay + config; throughput is always
measured warm, repeated, median, vs the TRUE baseline.

## Pluggable serving backend
The serving stack is NOT baked in. `args.backend` (sglang|vllm, default sglang) selects
`scripts/adapters/<backend>.sh`, which `scripts/bench_e2e.sh` (a backend-agnostic dispatcher: owns
server lifecycle, warmup, repeats, median+spread summary, free-port allocation) sources. Adding a new
stack = adding one adapter that defines `adapter_launch / adapter_health / adapter_bench`
(+ optional `adapter_default_port`). No role or orchestration change. `MODEL` is **required** — there
is no rig-specific default that could silently bench the wrong target.

## The three backend dimensions (per spec)
A kernel's backend can be changed from three places, in increasing cost (knob names are backend-
specific — see `perf_knowledge/backends/<backend>/` + `perf_knowledge/reference/env_vars.md`, as
reference only; verify every switch by measuring):
1. **launch flags** (`--attention-backend`, `--quantization`, …) — Config Tuner
2. **env vars** (sglang `SGLANG_USE_AITER` / vllm `VLLM_ROCM_USE_AITER`, `HIPBLASLT_TUNING_FILE`, …) — Config Tuner
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
Run via the `Workflow` tool. `workflow_dir` must be this folder (a JS workflow can't read its own
path); the kernel layer defaults to the sibling `kernel_workflow/`.
```
Workflow({
  scriptPath: "<E2E_DIR>/e2e_workflow.js",
  args: {
    model_path: "/path/to/model",                    // REQUIRED for e2e mode (no default)
    workflow_dir: "<E2E_DIR>",                       // REQUIRED: this folder
    backend: "sglang",                               // optional: sglang|vllm (selects scripts/adapters/<backend>.sh)
    launch_script: "<...>/launch.sh",                // optional; else the stack's default config
    kernel_workflow_dir: "<...>/workflows",          // optional; default = sibling kernel_workflow/
    budget: 4,            // max kernel-optimization tasks (kernel-layer tasks; config sweep is free)
    kernel_budget: 6,     // budget passed DOWN to each recursive single-kernel run
    milestone_min_pct: 5, // Milestone only optimizes editable kernels with pct_gpu_time >= this (default 5);
                          //   overrides min_kernel_tasks — sub-threshold kernels are skipped (Amdahl)
    config_tune: "true",  // Tier-0 sweep on/off (default ON)
    tuning_skillset: "true", // tuning on/off (default ON). The VENDORED tuning skillset
                          //   (<repo>/perf_knowledge/expert_skills/tuning) is read PER HEAD OP inside the
                          //   HeadKernel track — by quick_tune (the cheap standalone pass) and by
                          //   op_benchmarker Tier A/B — not as a phase of its own.
                          //   "false" injects nothing -> byte-identical.
    quick_tune_minutes: 25, // budget STATED to the cheap standalone pass, per head op (default 25).
                          //   0 disables that rung; the bake-off still reads the skillset.
    tuning_kb: "true",    // consult tuning-kb/ (the per-model ANSWER KEY). Set "false" for blind evals.
                          //   No op budget: tuning follows the head budget. See "Tuning" below +
                          //   knowledge/tuning_skillset_integration.md.
    analysis_skill: "roofline", // profile-analysis skill (default "roofline"; "none" disables).
                          //   Enriches the Top-N with % of the hardware roofline -> attainable speedup ->
                          //   expected e2e gain, so budget goes to the kernel with HEADROOM, not merely
                          //   the biggest one. ADVISORY: annotates + reorders, never prunes a candidate
                          //   and never overrides the measured pct_gpu_time. See "Roofline" below.
    use_expert_skills: "false", // consult perf_knowledge/expert_skills (advisory priors) on/off (default OFF, opt-in);
                          //   set "true" to enable. When OFF (default) nothing is injected -> behavior is
                          //   byte-identical to a run without the feature. Threaded down to the kernel layer too.
    gpu_ids: "0",         // comma-separated
    isl: 1024, osl: 1024, conc: 64,  // workload (profile + bench use the SAME)
    task: "focus on ...", // optional steer
    apply_to_original: "false"       // if "true", emit an apply bundle (overlay + launch), never edits site-packages
  }
})
// Single-kernel pass-through (backward compatible): pass kernel_path instead of model_path.
```

## Modes: default · fast · deep
One pipeline, three depths — selected by the `fast_mode` / `deep_mode` args (both default `false` =
**default** mode; they are mutually exclusive, **deep takes precedence**). Only the HeadKernel depth and
which phases run change; the throughput metric, the e2e gate, and the reversible-overlay contract are
identical. With both off, the run is **byte-identical** to the original (every mode knob is gated).

| Mode | arg | Phases | HeadKernel | Budget (default) | Use when |
|---|---|---|---|---|---|
| **default** | *(none)* | ConfigSweep + HeadKernel + **Milestone** | serial, 1 pass/head, ≤2 authored langs, single e2e gate | — | full pipeline incl. the editable-kernel Milestone loop |
| **fast** | `fast_mode:true` | HeadKernel only (skips ConfigSweep + Milestone) | **parallel** head track (extract/bake/author fan out across GPUs), time-capped | `fast_budget_ms` = 5h | a quick HeadKernel-only win under a wall-clock cap |
| **deep** | `deep_mode:true` | ConfigSweep + HeadKernel (skips Milestone) | **global cross-kernel×backend lane pool** — every (head op × backend) optimizes in parallel, many rounds via STATE_DIR + reseed, cross-pollination (per-op SHARED_KB + run-global GLOBAL_KB), convergence-stop + agent-budget backstop, finalize banks a **combined cross-kernel overlay** | `deep_head_budget_ms` = 24h | the deepest/broadest result: most backends, most rounds, hours available |

```
# default — full pipeline
args: { model_path, workflow_dir, backend:"vllm", tp:4, gpu_ids:"0,1,2,3", isl:1024, osl:1024, conc:64 }
# fast — HeadKernel-only, parallel, time-boxed
args: { ...same..., fast_mode:true, fast_budget_ms: 18000000 }
# deep — exhaustive, multi-backend, parallel (give it all GPUs + hours)
args: { ...same..., deep_mode:true, gpu_ids:"0,1,2,3,4,5,6,7", deep_head_budget_ms: 64800000 }
```
Pick **fast** for a bounded quick pass, **default** for the standard run, **deep** to chase the best
achievable number (it is broader = more backends, deeper = more/faster rounds, parallel = lanes co-opt
spare GPUs while the e2e gate runs on the serving slot, with matched in-window A/B so parallelism never
corrupts a measurement).

## Tuning (`tuning_skillset`, default ON) — inside the head-kernel track

The tuning skillset vendored at `<repo>/perf_knowledge/expert_skills/tuning/` is an
independently-validated method for tuning GPU ops on AMD Instinct: per-op tuners, deploy paths into a
live server, and the engagement checks that prove a tuned artifact is actually what the machine runs.

**It is vendored whole.** The tree is byte-identical to the standalone repo it is validated in (37
executable claims in `validate/claims.py`) and hash-pinned by
`e2e_workflow/knowledge/tuning_skillset.manifest.sha256`. Its method is deliberately **not** paraphrased
into `knowledge/`: a standalone validation is evidence about a specific tree, and scattering it would
keep the prose while losing the guarantee. Fixes go upstream and come back via
`scripts/tuning_skillset_sync.py --sync <dir>`; never edit inside the vendored tree.

**It is read per head op, on the head ladder** — it has no phase of its own. The ladder for each head op
is now three rungs, cheapest first, all scored on the SAME immutable oracle (that op's extracted
unittest):

1. `roles/quick_tune.md` — a bare, bounded standalone pass (`quick_tune_minutes`, default 25). Recalls a
   tuned table from the store if one exists, tries the env/dispatch knobs and a small per-shape sweep,
   proves engagement, and hands `handoff_note` to the next rung. No server, no authoring.
2. `roles/op_benchmarker.md` Tier A/B — the wider cross-backend bake-off and per-backend tune, told what
   rung 1 already timed so it does not re-search it.
3. the recursive `kernel_workflow` lanes — authoring and deep optimization, on top of whatever tuning
   found.

**Why it used to be a standalone phase, and why it is not one now.** It ran between ConfigSweep and
HeadKernel, as one `tuning_specialist` role, taking its own interleaved pre/post server A/B. That bought
attributability — "how much did tuning buy us" as a separate number — and it cost more than it bought:
GEMM and attention are exactly the ops the skillset tunes AND exactly the ops HeadKernel optimizes, so
the two tracks worked the same kernels from opposite ends with no shared context. The head track could
not see which backend tuning had already proven fastest; the tuning track could not see the roofline or
the extracted op oracle the head track was about to build. And the standalone A/B was a full server
pre/post paid before HeadKernel had established anything, on a stack it would then change underneath
itself.

**What that costs, stated plainly.** There is no tuning delta% and no share-of-total-gain any more:
`tuning_delta_pct`, `tuning_speedup` and `share_of_total_gain_pct` are `null`, meaning *not measured*,
never zero. The attribution that replaces them is per op — `ops_tuned[].isolated_speedup`, each measured
on that op's oracle rather than inferred from a server delta. Accepted env/flags/overlay still fold into
the carried config (at the end of the head track, before the post-head re-profile), so every later A/B
measures on top of a tuned stack, and the tuning gain is already inside the head track's reference leg:
the numbers are **not** additive.

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

Results land in `EVAL_DIR/tuning/` (`ops/<op>.json` per op, plus the aggregate `tuning_result.json` the
salvage path reads), in `final_report.md` **§2b**, in `wfReturn.tuning_skillset` (`ops_tuned[]` — the
three headline-delta keys are `null`, see above), and in `result.json` under a **purely additive**
`tuning_skillset` key — no existing key changes name, type or meaning, and a run without tuning produces
a byte-identical `result.json`. Tuning now rides the head track, so it runs wherever HeadKernel runs,
including fast mode. Full contract: `knowledge/tuning_skillset_integration.md`.

## Roofline-guided routing (`analysis_skill`, default `roofline`)

`pct_gpu_time` tells you where the time goes; it does **not** tell you whether that time is
*recoverable*. A kernel already running at the memory or compute ceiling has nothing left to give no
matter how much of the profile it owns. So after the Profiler emits the Top-N it runs one pluggable
**analysis skill** (`knowledge/analysis_skills/<name>/SKILL.md`) which estimates, per kernel:
`roofline_pct` → `attainable_speedup` → `expected_e2e_gain_pct = pct_gpu_time × (1 − 1/attainable)`.
The Architect then reports **both** orderings — by `%GPU` and by expected gain — and says which it
followed; a disagreement between them is the useful signal, so it is never blended into one number.

**Roofline is advisory and can never prune a candidate.** Same doctrine as the TraceLens prior: the
measured `pct_gpu_time` is the judge. Confidence is staged — profile-time shape estimates are `low`
(display only, not ranked on), real extracted shapes are `medium`, rocprofv3 counter measurements
(`FETCH_SIZE`/`WRITE_SIZE`/`MfmaFlops*`) are `high`. Five degradation levels end in "emit nothing and
behave exactly as before", so a missing peak table, an unmodellable op or a bad number cannot fail a run.

**A saturated head is rerouted, not dropped** — the point that matters most for the *largest* kernel.
Being at the wall means it is done with micro-tuning, not done being optimized; it routes to a
**byte-reduction track** (fuse an adjacent op away, stop reading unrouted experts, layout/packing,
lower-precision weights). Those levers must preserve the measurement contract: the user-supplied
workload (`isl`/`osl`/`conc`/batch) is fixed and speculative decoding is not an optimization.

Calibrated on a real run (Qwen3.5-35B-A3B-FP8, gfx950, vLLM TP1, 1k/1k, conc 64) — see
`scripts/tests/test_roofline_skill.py`:

| kernel | %GPU | of roofline | attainable | predicted e2e | measured |
|---|---|---|---|---|---|
| `fused_moe_kernel` | **26.45%** | 88% (memory-bound) | 1.02× | +0.6% | 1.047× iso, **−0.064% e2e** |
| `kernel_paged_attention_2d` | 8.86% | 18% | 2.8× | +5.7% | **1.56× iso** |

Ranking by `%GPU` sends the budget to the MoE, where it was in fact wasted; ranking by headroom sends
it to attention, where the win was. Add a skill by dropping a directory into
`knowledge/analysis_skills/`; swap with `analysis_skill=<name>`; disable with `analysis_skill=none`.

## Accuracy gate (gsm8k) — OFF by default
By default the e2e gate accepts a kernel on **throughput delta + greedy output parity**
(`accuracy_gate:"none"`). For QUANTIZED kernels (MXFP8/fp8) byte-parity is too strict — a within-tolerance
kernel rounds differently and flips a few borderline greedy argmaxes — so you can switch the bar to
**task accuracy**:
```
args: { ...same..., accuracy_gate:"gsm8k", accuracy_limit:200, accuracy_tol:0.01 }
```
- `accuracy_gate:"gsm8k"` → the Integrator serves a fresh TRUE baseline vs the candidate, runs sampled
  gsm8k (5-shot, greedy, fixed seed), and accepts iff `cand_em >= baseline_em - accuracy_tol`.
- `accuracy_limit` = #questions (default **200**; deep uses a larger sample at finalize to de-noise the
  boundary). `accuracy_tol` = allowed exact-match drop (default **0.01**).
- The eval client is `scripts/gsm8k_eval.py` (model-agnostic; queries the OpenAI-compatible endpoint).
- Leaving it unset (`"none"`) changes nothing vs before — the gate stays throughput + parity only.

## Output
Everything lands under `<exp_root>/e2e_<model>_<timestamp>/`:
- `env_report.{md,json}` — the preflight capability report every phase routes on
- `baseline/bench_summary.json`, `env_info.txt`, `config/baseline_flags.json` — the TRUE baseline
- `profile/round_*/profile_topN.{json,md}` — the standardized Top-N each round
- `strategy.md`, `config/sweep_results.json`, `insight_log.md`
- `kernels/<short_name>_task/{kernel_src, reference_io.pt, unittest.py, meta.json}` — extracted tasks
- `kernels/_exp/…` — the recursive single-kernel runs (each with its own verified result)
- `overlay/…` — candidate + accepted reversible overlays
- `tuning/{tuning_report.md, ops/<op>.json, tuning_result.json, deploy/, <op>/}` — per-op tuning from the head track
- `final/{overlay, final_patch.diff, final_launch.sh}` — the deliverable bundle
- `architect_report.md`, `director_e2e_validation.json` — the official verified throughput result

## Files
```
e2e_workflow.js   orchestration (deterministic; recursively calls ../kernel_workflow/kernel_workflow.js)
roles/                 director, system_architect, profiler, config_tuner, quick_tune, kernel_extractor, op_benchmarker, e2e_integrator
knowledge/             e2e_optimization, profile_parse, preflight (env self-check), backend_playbook + gemm_attention_backends (persistent), sglang_internals, shape_capture
knowledge/analysis_skills/  pluggable profile-analysis skills (INDEX.md + one dir per skill; `roofline` ships by default)
knowledge/tuning_skillset_integration.md  how the VENDORED ../tuning_skillset/ is wired in (+ its manifest)
scripts/               bench_e2e.sh (backend-agnostic dispatcher), adapters/{sglang,vllm}.sh, parse_profile.py (Top-N), op_bench.py, capture_shapes.py, overlay_setup.py
scripts/tuning_skillset_sync.py  integrity gate on the vendored skillset (--verify / --update / --sync)
scripts/server_teardown.sh  the shared server-kill contract (identity verified at LAUNCH: pid, pgid, /proc start time). Every script that launches a server, including role-authored capture scripts, must source it instead of hand-rolling a kill.
```

## Generality
The script never hard-codes a model or kernel. The workload (isl/osl/conc) drives both profiling and
benchmarking identically; the Profiler's classification + the Architect's Amdahl routing decide what
gets tuned vs rewritten. For a different model, only `model_path` (+ optional `launch_script`)
changes. The persistent `backend_playbook.md` carries learned per-class backend priors across runs.
