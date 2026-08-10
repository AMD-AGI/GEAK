<p align="center">
  <img src="examples/images/logo.png" alt="GEAK v4" width="300">
</p>

<p align="center">
  <a href="https://www.amd.com/en/developer/resources/technical-articles/2026/geak-v4.html"><b>📝 Blog</b></a>
  &nbsp;•&nbsp;
  <a href="https://rocm.docs.amd.com/projects/geak/en/latest/"><b>📚 Documentation</b></a>
</p>

GEAK is an autonomous optimization agent that makes AMD Instinct GPUs run faster, automatically. Built as a
multi-agent system with an evolving knowledge base, it learns from every optimization run and continuously
improves its strategies over time. Point it at a single kernel or a live model-serving stack such as vLLM or
sglang, and GEAK runs the full optimization loop: it finds the bottlenecks, generates and tunes better kernels
across paths such as Triton, FlyDSL, TileLang, and HIP, and validates the speedup on the real system. What
normally takes weeks of expert kernel engineering becomes an automated, repeatable, and self-improving process.

GEAK targets AMD Instinct MI GPUs (CDNA, e.g. gfx942 / gfx950; the on-box card is auto-detected), driven by
Claude Code and orchestrated by deterministic JS Workflows. It ships two workflows, each for a different scenario:

| Workflow | Scope | What it optimizes |
| --- | --- | --- |
| [e2e_workflow](e2e_workflow/) | Whole-model serving | End-to-end sglang / vLLM throughput of a full LLM |
| [kernel_workflow](kernel_workflow/) | Single kernel | Latency / speedup of a single AMD GPU kernel (Triton, HIP, CK, FlyDSL, …) |

Use e2e_workflow to raise a whole model's serving throughput: it triages hot kernels, pulls the cheapest levers
first, and recursively calls kernel_workflow for the kernels worth fixing. Use kernel_workflow on its own to
optimize a single kernel.

---

## Architecture

<p align="center">
  <img src="docs/assets/GEAK_v4_framework.png" alt="GEAK v4 Optimization Pipeline" width="900">
</p>

---

## Getting started

### 1. Prerequisites

- An **AMD Instinct MI GPU** (CDNA, e.g. gfx942 / gfx950), **ROCm 6+**, a profiler (`rocprof-compute` /
  `rocprofv3` / `rocprof`), Python 3.8+.
- For E2E: a running-capable serving backend (`sglang` or `vllm`) and the model weights on disk.

> **⚠️ Build your kernel environment first.** GEAK does **not** install the toolchains your kernels
> need (e.g. PyTorch, Triton, FlyDSL, hipBLASLt) — these differ per kernel. Set up and verify the
> baseline builds/runs before starting a workflow.

### 2. Set up

Installing GEAK does three things: installs the `geak` Python package + deps, clones the GEAK repo, and installs
the Claude Code CLI. By default the repo lands in `./GEAK` under the directory you run the command from (override
the location with `GEAK_HOME`). Pick either method — both end up the same:

**A. One-liner** — run it in the directory where you want GEAK to live:

```bash
pip install "git+https://github.com/AMD-AGI/GEAK"
```

**B. Clone first** — if you'd rather have the checkout up front (e.g. to work on a branch):

```bash
git clone https://github.com/AMD-AGI/GEAK.git
cd GEAK
pip install .
```

### 3. Launch Claude Code in auto mode

**Set up PATH and API access**

You'll need to add `~/.local/bin` to your PATH and configure API access yourself — follow the installer's printed next-steps:

```bash
# Option 1: Anthropic API directly
export ANTHROPIC_API_KEY=sk-ant-...

# Option 2: Standard gateway (x-api-key / bearer)
export ANTHROPIC_BASE_URL=https://your-gateway
export ANTHROPIC_AUTH_TOKEN=your-token
```

> If your gateway authenticates with a **custom header** instead of `x-api-key` / a bearer token
> (e.g. `Ocp-Apim-Subscription-Key`), set it via `ANTHROPIC_CUSTOM_HEADERS`:
> ```bash
> export ANTHROPIC_BASE_URL=https://your-gateway
> export ANTHROPIC_CUSTOM_HEADERS="Your-Header-Name: <your-key>"
> ```

**Launch Claude Code**

The workflows spawn many sub-agents and run profiling / benchmark / build commands on the box, so run
Claude Code with permissions auto-approved (≥ 2.1.177 for the dynamic Workflow feature):

```bash
IS_SANDBOX=1 claude --dangerously-skip-permissions
```

Then just describe what you want in natural language (examples below). Claude Code resolves the paths and
invokes the `Workflow` tool for you.

### Swappable agent backend (Claude Code ↔ codex / cursor)

The workflows are plain JS that normally run on **Claude Code's `Workflow` tool** (which provides the
`agent()` / `parallel()` / `pipeline()` / `workflow()` orchestration primitives). To run the **same
workflows** under a different coding-agent CLI — primarily **codex** or **cursor-agent** — GEAK ships a
**standalone Node runtime** (`interface/runtime/`) that re-implements those primitives itself and
dispatches each `agent()` call to a one-shot backend process. All parallelism and one-level nesting
happen in the runtime, so the agent CLI does **not** need to support parallel or nested subagents.

Two orthogonal axes live in `interface/runtime/registry.json`: **agents** (which CLI: claude / codex /
cursor / qwen / kimi) × **models** (which endpoint). A **profile** pins one `(agent, model)` combo.
Select a backend with `--agent` / `--profile` (or `GEAK_AGENT_BACKEND` / `GEAK_AGENT_PROFILE`); the `.js`
workflows / roles / knowledge are used unmodified. **e2e** always goes through `run_e2e.py` (which
auto-routes to the runtime once a backend is set) and **single kernels** through `run_workflow.mjs`.
The two main backends — **codex** and **cursor** — are documented next.

#### codex backend — self-contained setup

The codex provider is auto-configured from the key you provide (no `config.toml` editing, no provider
selection).

*1. Install the codex CLI* (do not assume it is already present):
```bash
node -v                                  # need Node.js v20+ (install via nvm / pkg manager / nodejs.org)
npm i -g @openai/codex@0.146.1           # pin 0.146.1 — 0.147 breaks with gateways
#   no write access to /usr/local? use a user-level prefix:
#   npm config set prefix "$HOME/.npm-global" && export PATH="$HOME/.npm-global/bin:$PATH"
#   npm i -g @openai/codex@0.146.1
codex --version                          # expect 0.146.1
```

*2. Pick a provider by setting its key* — the runtime auto-selects (first match wins):
```bash
export OPENAI_API_KEY=sk-...                              # -> OpenAI official (api.openai.com, public CA)
# export AMDKEY=<32hex>       SSL_CERT_FILE=/path/ca.pem  # -> AMD gateway (adds Ocp-Apim-Subscription-Key)
# export SAFE_API_KEY=ak-...  SSL_CERT_FILE=/path/ca.pem  # -> SaFE gateway (gpt direct)
# explicit override for any OpenAI-compatible gateway (wins over the above):
#   export OPENAI_BASE_URL=https://your-gateway/v1
```

*3. Run.* codex has **no natural-language mode** (that path needs Claude Code's `Workflow` tool);
drive it through `run_e2e.py` (e2e) or `run_workflow.mjs` (single kernel). The natural-language
`use path_to_GEAK/... to optimize ...` examples further below are **Claude-only**.

```bash
export GEAK_AGENT_BACKEND=codex
# REQUIRED: a model id the chosen provider actually serves (a wrong id 404s at the first
# codex turn). Provider-specific — do NOT reuse one gateway's id on another: an OpenAI
# model (e.g. gpt-5.x) for api.openai.com; gpt-5.6-sol for AMD; gpt-5.6 for SaFE.
export GEAK_CODEX_MODEL=<your-provider-model-id>
# thinking level defaults to max; change with GEAK_CODEX_EFFORT (low|medium|high|xhigh|max):
#   export GEAK_CODEX_EFFORT=xhigh

# --- e2e (whole-model serving throughput): describe the run in a handoff.json ---
cat > handoff.json <<'JSON'
{ "schema_version": 2,
  "model_path": "/models/Qwen3.5-27B-FP8",
  "framework": "sglang", "tp": 1, "gpu_ids": "0",
  "workload": { "isl": 1024, "osl": 1024, "conc": 64 },
  "exp_root": "/abs/work/geak" }
JSON
# required: model_path, exp_root (its basename MUST be `geak`); rest has defaults.
# full schema + fields: interface/run_e2e.md
python interface/run_e2e.py handoff.json result.json     # auto-routes to the codex runtime

# --- single kernel ---
node interface/runtime/run_workflow.mjs kernel_workflow/kernel_workflow.js --agent codex \
  --args '{"kernel_path":"/abs/kernel","workflow_dir":"'"$PWD"'/kernel_workflow","budget":6}'
```

Overrides & troubleshooting: `GEAK_CODEX_AUTOCONFIG=0` disables auto-config (falls back to
`interface/runtime/codex-home/config.toml`); `GEAK_CODEX_EXTRA_ARGS="-c model_provider=..."` pins a provider
manually. 401 → key unset/invalid; 404 model → `GEAK_CODEX_MODEL` unavailable or not Responses-API-capable;
TLS error → intranet gateways need `SSL_CERT_FILE` (public OpenAI does not). claude-via-SaFE needs the
de-stream shim — see [`interface/runtime/SETUP.md`](interface/runtime/SETUP.md).

#### cursor backend — runs on Cursor cloud (NOT via a gateway)

`cursor-agent` uses **Cursor-side models on Cursor's own cloud**, so it needs no gateway, no
`SSL_CERT_FILE`, and no shim — but requests + code leave for Cursor's cloud, and the model is a
Cursor-side id (not one your gateway serves).

```bash
# 1) install cursor-agent (see Cursor docs), then authenticate ONCE:
cursor-agent login                          # or: export CURSOR_API_KEY=...
export GEAK_CURSOR_MODEL=composer-2.5       # a Cursor-side model id (e.g. composer-2.5, sonnet-4-thinking)

# 2) run — single kernel:
node interface/runtime/run_workflow.mjs kernel_workflow/kernel_workflow.js --agent cursor \
  --args '{"kernel_path":"/abs/kernel","workflow_dir":"'"$PWD"'/kernel_workflow","budget":6}'
# e2e:
export GEAK_AGENT_BACKEND=cursor
python interface/run_e2e.py handoff.json result.json
```

> Because cursor uses Cursor-side models + cloud, it **cannot** join a strict "same gateway, same model"
> comparison against codex/claude. More detail: [`interface/runtime/SETUP.md`](interface/runtime/SETUP.md) §B.

**Controlled (agent × model) comparison experiments are built in** — sweep the matrix, N repeats,
fixed task/budget, get a comparison table (speedup / success-rate / wall; no token/cost):

```bash
node interface/runtime/experiment.mjs --script kernel_workflow/kernel_workflow.js \
  --agents claude,codex,cursor --models default --repeats 3 \
  --args '{"kernel_path":"/abs/knn","workflow_dir":"'"$PWD"'/kernel_workflow","budget":6}'
```

Adding a new CLI = a `registry.json` entry (zero code). Full env knobs, the compatibility checklist,
and a no-GPU smoke test (`node interface/runtime/selftest.mjs`) are in
[interface/run_e2e.md](interface/run_e2e.md) and
[interface/runtime/DEV_SUMMARY.md](interface/runtime/DEV_SUMMARY.md).

---

## e2e_workflow — whole-model serving throughput ⭐

`e2e_workflow/` raises the **sglang / vLLM serving throughput** of a whole LLM. It is a *system layer*
that wraps — and recursively calls — the single-kernel kernel_workflow:

1. **Preflight** the env (GPU arch, backend, model).
2. **Profile** a running server on your exact workload.
3. **Triage** hot kernels by **Amdahl** (`pct_gpu_time × achievable_speedup`).
4. **Pull levers cheapest-first** — config/backend sweep → head GEMM/attention bake-off (aiter per-shape
   tune + a kernel *authored* via the recursive kernel layer, FlyDSL-first for GEMM) → editable-kernel
   milestone loop.
5. **Overlay** each accepted change back **reversibly**, gated on a measured warm-server throughput delta
   (interleaved A/B, 0.5% band + engagement proof + output parity).

Every run writes a complete **`final_report.md`** (with a Phases tree + artifacts tree).

### Example

```
use path_to_GEAK/e2e_workflow to optimize inference for /models/Qwen3.5-27B-FP8, sglang, ISL/OSL=1024, conc=64, gpus 0,1,2,3
```

**Output** lands under `e2e_workflow/exp/e2e_<model>_<timestamp>/` — `final_report.md`,
`architect_report.md`, `final/` (overlay + patch + `final_launch.sh`), and per-stage artifacts.

---

## kernel_workflow — single kernel

`kernel_workflow/` optimizes a single AMD GPU kernel — Triton, HIP, CK, FlyDSL, or any AMD GPU source:
Director → TechLead → specialist engineers (algorithm / memory / compute / host_runtime), multi-round and
budget-controlled, with each patch independently verified before it's accepted.

### Example

```
use path_to_GEAK/kernel_workflow to optimize path_to_GEAK/examples/tasks/knn
```

```
use path_to_GEAK/kernel_workflow to optimize /path/to/silu, budget 8, focus on wrapper overhead
```

### Batch (many kernels at once)

Spawn one agent per kernel with isolated GPU assignments; GPU access is serialized via
`scripts/gpu_lock.sh` (flock-based), so kernels can safely share GPUs.

---

## Why Workflows

Control flow — the budget loop, fan-out, verification, and stop conditions — is **deterministic JS** in
`kernel_workflow.js` / `e2e_workflow.js`. LLM agents are called only for judgement (analysis, strategy,
optimization). This makes runs reliable and reproducible.

## Results — single-kernel

12 HIP kernels, measured on AMD MI300X (gfx942) (excluding mla_decode; FAIL counted as 1.0x):

| Method | LLM | Geo Mean |
| ------ | --- | -------- |
| GEAK_v3 (baseline) | Opus 4.8 | 1.90x |
| **kernel_workflow** | Opus 4.8 | **3.68x** |

## Repository layout

```
GEAK/
├── e2e_workflow/        # ⭐ End-to-end LLM serving-throughput optimizer (wraps kernel_workflow/)
│   ├── e2e_workflow.js   # system-layer orchestration (config / head-GEMM / kernel tracks + e2e gate)
│   ├── roles/  knowledge/  scripts/   # adapters/{sglang,vllm}.sh, op_bench.py, parse_profile.py, …
│   └── README.md / PLAN.md
├── kernel_workflow/     # Single-kernel optimizer
│   ├── kernel_workflow.js       # deterministic JS orchestration
│   ├── roles/  knowledge/  scripts/   # gpu_lock.sh, profile_kernel.sh
│   └── README.md
├── perf_knowledge/      # AMD operator × backend SOTA knowledge base (REFERENCE ONLY)
├── examples/            # Example kernel tasks, benchmark comparisons, real e2e runs
└── exp/                 # Experiment outputs (timestamped per run)
```

## Approaches compared

How the workflows in this repo relate to the GEAK_v3 baseline:

|                        | GEAK v3 (baseline)                        | kernel_workflow                                                  | e2e_workflow                                                       |
| ---------------------- | ----------------------------------------- | --------------------------------------------------------------- | ----------------------------------------------------------------- |
| **Target**             | Single kernel                             | Single kernel                                                   | **Whole-model sglang/vLLM serving throughput**                    |
| **Agent backend**      | miniswe                                   | Claude                                                          | Claude                                                            |
| **Architecture**       | Orchestrator + parallel workers           | Hierarchical: Director → TechLead → Engineers → Merge           | e2e Director → System Architect → Profiler / Config Tuner / Kernel Extractor / e2e Integrator (wraps the kernel layer) |
| **Iteration**          | Multi-round                               | Multi-round, budget-controlled                                  | Multi-round, Amdahl-triaged, budget-controlled                    |
| **Orchestration**      | Python                                    | **Deterministic JS** — loop/parallelism/verification in code   | **Deterministic JS**                                              |
| **Verification**       | Orchestrator verifies                     | **Pipelined** — each patch verified by a separate agent        | **Warm-server interleaved A/B** — throughput delta + engagement proof + output parity |
| **Engineer types**     | Generic                                   | **Specialist**: algorithm, memory, compute, host_runtime       | System roles + the specialist kernel squad via the recursive layer |
| **Cross-round memory** | miniswe-memory control                    | **Explicit**: insight blackboard + hypothesis ledger           | **Explicit**: insight blackboard + per-backend knowledge          |
| **Best for**           | Programmatic kernel optimization          | Single-kernel gains with high reliability/reproducibility      | **Raising end-to-end serving throughput of a full model**         |

## License

GEAK is licensed under the Apache License 2.0 — see [LICENSE.md](LICENSE.md).
