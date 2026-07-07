# GEAK_v4 e2e CI harness

Local/self-hosted CI for running the GEAK ("perfskills") end-to-end kernel
optimization workflow on a ROCm GPU box, exactly the way Hyperloom's
`KERNEL_AGENT` phase launches it (`interface/run_e2e.py`).

Everything is driven from `ci/run_local.sh`. A run:

1. resolves the container image + model weights for a model key,
2. **GPU preflight**: a short throwaway container probes the GPU (`rocminfo` +
   torch matmul) and fails fast if it's dead/wedged — before committing to a run,
3. spins up the ROCm/vllm container with GPU passthrough (in the background),
4. installs Claude Code + the Python `claude_agent_sdk` inside it,
5. runs the GEAK e2e workflow for one model into a timestamped folder,
6. while it runs, a **host-side liveness monitor** watches `run.log` and kills a
   wedged/stalled run instead of letting it hang to the wall-clock cap,
7. hard-judges the result on exit code + `result.json` (status **and** a real
   measured baseline).

See [Run topology](#run-topology) for the exact launch order and layering.

## Layout it expects

Paths are **derived from each script's location** — no absolute machine paths
are baked in. The workspace just needs to look like this (siblings of the repo):

```
<workspace>/
  GEAK/                 # this repo (contains ci/, interface/, e2e_workflow/)
  InferenceX/           # bench client (cloned separately)
  geak_runtime/         # per-model handoff.json / recipe / tracelens priors (data repo)
    docker_select.log   # "<framework> (<arch>): <image>" lines used to pick the image
    Qwen-Qwen3-8B/
      handoff.json
      baseline_config.with_envs.yaml
      kernel-agent/tracelens/...
      runs/roofline/torch_trace
```

Model weights live wherever `models.tsv` points (usually outside the workspace,
e.g. `/home/ethany/models/Qwen3-8B`); that dir is same-path bind-mounted too.

## Files

| file | role |
|------|------|
| `lib.sh` | shared config/helpers: path derivation, model registry, image resolver. Sourced by the others; not run directly. |
| `models.tsv` | registry: `<model_key>\t<weights_dir>\t<framework>` |
| `claude_setup.sh` | install + configure Claude Code (global AMD LiteLLM proxy). Re-run every container start (ephemeral fs). |
| `setup_claude.sh` | Step D: install Claude into `$CLAUDE_HOME`, install `claude_agent_sdk`, probe `claude -p`. |
| `gpu_healthcheck.sh` | GPU preflight probe run INSIDE the framework image: `rocminfo` + a tiny torch matmul on GPU 0. Fast, timeout-capped. |
| `run_geak_e2e.sh` | mirror Hyperloom's `run_e2e.py` launch; patches `exp_root`/`model_path`/`launch_recipe`/`inferencex_path` in the handoff. |
| `run_model.sh` | Step E+F: run ONE model into a timestamped dir, then deterministically judge (status + measured baseline). |
| `run_monitor.sh` | host-side liveness monitor: every 5 min feeds `run.log` tail to a `claude -p` arbiter; `docker kill`s a confirmed-stuck run. |
| `monitor_prompt.md` | the monitor's instructions + `VERDICT: CONTINUE\|KILL` output contract. |
| `run_local.sh` | top-level launcher: GPU preflight, background docker GPU run + monitor, or `--dry-run` host-only wiring check. |

## Run topology

**What's a prerequisite vs. installed at runtime, and where it lives:**

| component | installed by CI? | where it lives | when |
|-----------|------------------|----------------|------|
| Docker daemon | no — host prerequisite | host | already running |
| framework image (`rocm/vllm-dev:…`) | pulled, not built | host docker cache | first `docker run` (preflight) |
| Claude Code CLI + `claude_agent_sdk` | **yes, at runtime** | **inside** the container, under `$CLAUDE_HOME` (bind-mounted out) | Step D, every run |
| perfskills / GEAK workflow | no — it *is* the code under test | host checkout, bind-mounted in | launched last, in-container |

There is no "docker install" step: docker is assumed present on the box. The only
thing genuinely *installed* per run is **Claude**, and it happens **inside** the
container. There are effectively **two Claudes**: the in-container **worker** (SDK
path, drives perfskills, full tools) and the host **watchdog** (no tools, reads
log tails, judges liveness) — independent installs/sessions.

**Launch order** (everything sequential except the real run + monitor, which are concurrent):

```
HOST (run_local.sh — orchestrator)
│
├─ 0. lib.sh resolves: model → framework → IMAGE (docker_select.log), weights dir
│
├─ 1. GPU PREFLIGHT ── docker run (throwaway, ≤120s) ──────────────┐  same image +
│       • pulls IMAGE if not cached  (this is the image pull)      │  --device flags
│       • gpu_healthcheck.sh: rocminfo + torch matmul on GPU 0     │  as the real run
│       • FAIL → whole job dies here, seconds in                   │
│     ◄─────────────────────────────────────────────────────────┘
│
├─ 2. docker run --name geak_l1_…  (REAL container, BACKGROUND) ───┐  DOCKER_PID=$!
│   ┌─ INSIDE CONTAINER, sequential (bash -lc, set -e) ────────┐   │
│   │  D. setup_claude.sh   ← CLAUDE INSTALL                   │   │
│   │       install Claude CLI + claude_agent_sdk; probe       │   │
│   │       `claude -p "SETUP OK"` → die if it fails           │   │
│   │            │ (only if D succeeds)                        │   │
│   │            ▼                                             │   │
│   │  E. run_model.sh → run_geak_e2e.sh → run_e2e.py          │   │
│   │       ← PERFSKILLS LAUNCH (GEAK workflow via Claude SDK)  │   │
│   │       patch handoff, serve model, bench, optimize,       │   │
│   │       write result.json                                  │   │
│   │            ▼                                             │   │
│   │  F. deterministic judge: status ok/no_gain AND           │   │
│   │       baseline_throughput_tok_s > 0                      │   │
│   └──────────────────────────────────────────────────────────┘   │
│                                                                   │
├─ 3. run_monitor.sh (HOST, BACKGROUND, PARALLEL to step 2) ────────┤
│       every 300s: tail run.log → claude -p → CONTINUE|KILL        │
│       confirmed KILL (2 votes) → docker kill geak_l1_… → red      │
│                                                                   │
└─ 4. wait "$DOCKER_PID" → RC ──────────────────────────────────────┘
        trap tears down the monitor; monitor_verdict.json ⇒ force RC≥1
        exit RC → GitHub step goes green/red
```

**Three ordering facts that matter:**

1. **Preflight is a gate, not parallel.** It fully completes (and pulls the image)
   before the real run. A dead GPU costs seconds, not hours — and the real
   `docker run` reuses the now-cached image (no re-pull).
2. **In-container D → E → F is strictly sequential and fail-fast.** Claude
   install+probe MUST pass before perfskills launches — perfskills *is* a
   Claude-SDK workflow, so a broken Claude means no GPU/workflow work happens.
3. **The real run (2) and the monitor (3) are the only concurrent pieces** — both
   launched from the host, one watching the other's `run.log`. `wait` yields the
   run's true exit code; the monitor is torn down by the EXIT trap.

**Defense layers (fail red, not false-green):**

| when | layer | catches |
|------|-------|---------|
| before run | GPU preflight (`gpu_healthcheck.sh`) | dead/wedged GPU, docker/device broken |
| during run | Claude watchdog (`run_monitor.sh`) | mid-run stall, GPU wedge, NFS/OOM loop |
| after run | deterministic judge (`run_model.sh` Step F) | false-green `no_gain`, unmeasured baseline, errors |
| absolute | workflow `timeout-minutes` | everything above failing |

## Usage

```bash
cd <workspace>/GEAK

# host-only wiring check (no docker/GPU/Claude): validates handoff -> args mapping
bash ci/run_local.sh Qwen-Qwen3-8B --dry-run

# real GPU smoke run (30-min budget)
bash ci/run_local.sh Qwen-Qwen3-8B --budget 1800

# pin a specific image instead of resolving from docker_select.log
IMAGE=rocm/vllm-dev:some-gfx950-tag bash ci/run_local.sh Qwen-Qwen3-8B
```

Outputs land in `geak_runtime/<model_key>/ci_runs/<timestamp>/`:

- `run.log` — full stdout/stderr of the run
- `result.json` — the workflow result (source of truth for pass/fail)
- `claude/` — `$CLAUDE_HOME` (Claude install + config + logs), persisted
- `claude_tmp/` — container `TMPDIR`, incl. Claude's background-task tree, kept for post-mortem debugging
- `monitor.log` — the liveness monitor's poll-by-poll verdicts (only if the monitor ran)
- `monitor_verdict.json` — present **only** if the monitor killed the run (records the container, reason, kill streak)

**Pass criteria:** exit 0 **and** `result.json.status` in `{ok, no_gain}` **and** a
real measured baseline (`baseline_throughput_tok_s > 0`). Anything else fails:

- a `no_gain`/`ok` with `baseline_throughput_tok_s <= 0` = nothing was actually
  measured (GPU unusable / serving never healthy) → hard fail `gpu_unusable`,
  not a false-green.
- `error_class = workflow_parse_error` is the fingerprint of the `claude -p` CLI
  fallback (Python `claude_agent_sdk` missing in the container).
- a `monitor_verdict.json` (watchdog kill) forces the step red.

## Monitor knobs

The host-side liveness monitor (`run_monitor.sh`) is on by default; tune via env:

| var | default | meaning |
|-----|---------|---------|
| `GEAK_MONITOR` | `1` | set `0` to disable the watchdog entirely |
| `GEAK_MONITOR_INTERVAL_S` | `300` | seconds between polls |
| `GEAK_MONITOR_CONFIRM` | `2` | consecutive KILL votes required before acting (hysteresis) |
| `GEAK_MONITOR_RECHECK_S` | `60` | faster re-poll while confirming a KILL |
| `GEAK_MONITOR_MODEL` | `claude-opus-4-8` | model for the arbiter (host login) |
| `GEAK_MONITOR_TAIL_LINES` | `300` | how much of `run.log` to feed each poll |
| `GPU_HEALTHCHECK_TIMEOUT_S` | `120` | preflight probe cap; `0` skips preflight (CPU-only debugging) |

The monitor no-ops gracefully if `claude` isn't on the host PATH (the run proceeds
unwatched, protected by the workflow's wall-clock cap).

## Environment overrides

| var | default | meaning |
|-----|---------|---------|
| `GEAK_ROOT` | `ci/..` | GEAK repo root |
| `WS` | `GEAK_ROOT/..` | workspace root |
| `INFERENCEX_PATH` | `$WS/InferenceX` | bench client checkout (empty = native bench) |
| `HF_LOGS` | `$WS/geak_runtime` | per-model dataset root |
| `MODELS_TSV` | `ci/models.tsv` | model registry |
| `DOCKER_SELECT` | `$HF_LOGS/docker_select.log` | image selection table |
| `IMAGE` | resolved from `docker_select.log` | override the container image |
| `MODEL_PATH` | from `models.tsv` | override weights dir |
| `PERFSKILLS_E2E_TIMEOUT_S` | `1800` | workflow wall-clock budget (also via `--budget`) |
| `LITELLM_API_KEY` / `LITELLM_BASE_URL` | baked defaults in `claude_setup.sh` | Claude auth via the global LiteLLM proxy |

## Notes

- `NODE_TLS_REJECT_UNAUTHORIZED=0` is a stopgap for the internal corporate
  CA the base image doesn't trust. Preferred long-term fix: bake the corporate
  CA bundle into the image and drop the flag.
- Only the `claude-opus` family is served by the proxy, so the haiku/sonnet
  Claude Code defaults also point at `claude-opus-4-8`.
- The container runs with `--rm`; it's destroyed after each run. The pulled
  image is cached. Everything worth keeping is written to the mounted
  `ci_runs/<timestamp>/` dir.
