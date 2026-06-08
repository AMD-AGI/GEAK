# team_workflow_e2e — End-to-End LLM Inference-Throughput Optimizer (MI300X)

A deterministic **Workflow** (JS-orchestrated multi-agent pipeline) that raises the **sglang/vllm
serving throughput** of an LLM on AMD MI300X. It is a *system layer* built on top of — and recursively
calling — the UNCHANGED single-kernel `team_workflow` (`../workflows/`). The single-kernel workflow's
quality is preserved verbatim; this layer adds everything above the kernel: profiling a running
server, Amdahl triage, config/backend tuning, extracting hot kernels into standalone unittests,
optimizing them with the kernel layer, overlaying them back, and re-validating end-to-end throughput.

## Design: fractal two-altitude
- **System layer** (this dir): owns the server, the throughput metric, profiling, triage, config, and
  reintegration. Roles: e2e Director, System Architect, Profiler, Config Tuner, Kernel Extractor,
  e2e Integrator/Validator.
- **Kernel layer** (`../workflows/`, UNCHANGED): given a kernel task dir, does the real multi-backend
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
  `scripts/parse_profile.py` (the "规范" contract).
- **Config Tuner** = Tier-0 flag/env/backend sweep, runs FIRST (default ON), no source rewrite.
- **Kernel Extractor** = capture real shapes + a reference I/O oracle → an IMMUTABLE standalone
  unittest task dir the kernel layer consumes (anti-cheating).
- **e2e Integrator/Validator** = reversible overlay reintegration + e2e throughput gate + final bundle.
- **Kernel squad** = the UNCHANGED `../workflows/team_workflow.js`, invoked recursively.

## Pipeline
```
Setup(preflight env-check + baseline throughput) → Baseline Profile(Top-N) → Strategize(Amdahl routing) →
ConfigSweep(flags/env/backends, FIRST) → Re-profile →
LOOP milestone[ plan → per kernel: Extract → recursive team_workflow.js → Overlay+e2e gate → ] → Re-profile → grow playbook →
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
specific — see `kernel_knowledge/02_libraries/<backend>_rocm.md`):
1. **launch flags** (`--attention-backend`, `--quantization`, …) — Config Tuner
2. **env vars** (sglang `SGLANG_USE_AITER` / vllm `VLLM_ROCM_USE_AITER`, `HIPBLASLT_TUNING_FILE`, …) — Config Tuner
3. **source** (a Triton/HIP/CK/asm reimplementation) — Kernel Extractor + kernel squad, overlaid back
   reversibly (never editing site-packages). For a hot op with **no existing editable implementation**,
   the head track now **authors one from scratch**: the Op Benchmarker DISCOVERs existing impls + tunes
   cheap levers, then emits an `author_plan`; the orchestrator runs the kernel layer in **author mode**
   (`mode=author target_language=triton|hip|ck`) to write a fresh baseline against the immutable oracle
   and optimize it, then the Integrator rebinds the op's call site to it and gates on e2e. Routing
   (direct_light tune vs author/rewrite via the kernel layer vs drop) is decided by Amdahl headroom +
   rewrite type. Triton is always a viable author target; HIP/CK when the headroom justifies them
   (`head_author_max`, default 1 = Triton only).

## Invocation
Run via the `Workflow` tool. `workflow_dir` must be this folder (a JS workflow can't read its own
path); the kernel layer defaults to the sibling `workflows/`.
```
Workflow({
  scriptPath: "<E2E_DIR>/team_workflow_e2e.js",
  args: {
    model_path: "/path/to/model",                    // REQUIRED for e2e mode (no default)
    workflow_dir: "<E2E_DIR>",                       // REQUIRED: this folder
    backend: "sglang",                               // optional: sglang|vllm (selects scripts/adapters/<backend>.sh)
    launch_script: "<...>/launch.sh",                // optional; else the stack's default config
    kernel_workflow_dir: "<...>/workflows",          // optional; default = sibling workflows/
    budget: 4,            // max kernel-optimization tasks (kernel-layer tasks; config sweep is free)
    kernel_budget: 6,     // budget passed DOWN to each recursive single-kernel run
    config_tune: "true",  // Tier-0 sweep on/off (default ON)
    gpu_ids: "0",         // comma-separated
    isl: 1024, osl: 1024, conc: 64,  // workload (profile + bench use the SAME)
    task: "focus on ...", // optional steer
    apply_to_original: "false"       // if "true", emit an apply bundle (overlay + launch), never edits site-packages
  }
})
// Single-kernel pass-through (backward compatible): pass kernel_path instead of model_path.
```

## Output
Everything lands under `<exp_root>/e2e_<model>_<timestamp>/`:
- `env_report.{md,json}` — the preflight capability report every phase routes on
- `baseline/bench_summary.json`, `env_info.txt`, `config/baseline_flags.json` — the TRUE baseline
- `profile/round_*/profile_topN.{json,md}` — the standardized Top-N each round
- `strategy.md`, `config/sweep_results.json`, `insight_log.md`
- `kernels/<short_name>_task/{kernel_src, reference_io.pt, unittest.py, meta.json}` — extracted tasks
- `kernels/_exp/…` — the recursive single-kernel runs (each with its own verified result)
- `overlay/…` — candidate + accepted reversible overlays
- `final/{overlay, final_patch.diff, final_launch.sh}` — the deliverable bundle
- `architect_report.md`, `director_e2e_validation.json` — the official verified throughput result

## Files
```
team_workflow_e2e.js   orchestration (deterministic; recursively calls ../workflows/team_workflow.js)
roles/                 director, system_architect, profiler, config_tuner, kernel_extractor, op_benchmarker, e2e_integrator
knowledge/             e2e_optimization, profile_parse, preflight (env self-check), backend_playbook + gemm_attention_backends (persistent), sglang_internals, shape_capture
scripts/               bench_e2e.sh (backend-agnostic dispatcher), adapters/{sglang,vllm}.sh, parse_profile.py (Top-N), op_bench.py, capture_shapes.py, overlay_setup.py
```

## Generality
The script never hard-codes a model or kernel. The workload (isl/osl/conc) drives both profiling and
benchmarking identically; the Profiler's classification + the Architect's Amdahl routing decide what
gets tuned vs rewritten. For a different model, only `model_path` (+ optional `launch_script`)
changes. The persistent `backend_playbook.md` carries learned per-class backend priors across runs.
