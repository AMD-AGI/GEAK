# GEAK_v4 e2e CI harness

Local/self-hosted CI for running the GEAK ("perfskills") end-to-end kernel
optimization workflow on a ROCm GPU box, exactly the way Hyperloom's
`KERNEL_AGENT` phase launches it (`interface/run_e2e.py`).

A single per-node run is driven from `ci/node/run_local.sh` (on the cluster it is
launched by `ci/dispatch/run_matrix.sh` → `ci/dispatch/slurm_job.sh`). A run:

1. resolves the container image + model weights for a model key,
2. **GPU health gate**: a host-side D-state scan (`gpu_dstate_check.sh`) bails if
   the driver is already wedged, then a short throwaway container probes the GPU
   (`rocminfo` + torch matmul) and fails fast if it's dead — before committing,
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
    docker_default.json # fallback image map { framework: { arch: image } } (repo copy: ci/docker_default.json)
    Qwen-Qwen3-8B/
      handoff.json
      baseline_config.with_envs.yaml
      kernel-agent/tracelens/...
      runs/roofline/torch_trace
```

Model weights live wherever `models.tsv` points (usually outside the workspace,
e.g. `/home/ethany/models/Qwen3-8B`); that dir is same-path bind-mounted too.

## Files

The tree is grouped by responsibility:

```
ci/
  lib.sh  models.tsv  README.md   # shared config / registry / docs (root)
  fixtures/                        # committed test fixtures (L0 dry-run)
  dispatch/                        # jump box -> SPUR cluster orchestration
  node/                            # what runs INSIDE a SPUR allocation (per node)
  preflight/                       # gates/prep run before the real workflow
  monitor/                         # host-side liveness watchdog
```

**Shared (root)**

| file | role |
|------|------|
| `lib.sh` | shared config/helpers: path derivation, model registry, image resolver, **SPUR config + weights staging + `tp`/framework from handoff**. Sourced by the others; not run directly. |
| `models.tsv` | **enrollment registry**: `<model_key>\t<hf_repo>\t<tier>` (framework + `tp` come from each `handoff.json`, not here). |

**`dispatch/` — jump box → cluster**

| file | role |
|------|------|
| `run_matrix.sh` | **L1 orchestrator (jump box)**: submit one SPUR job per selected model (`smoke`/`verify`/`probe`/explicit), wait, judge, write a pass/fail matrix; red if any fail. `--print` shows the `sbatch` commands without submitting. `probe`/`--probe` = infra-only harness check that stops at the GEAK e2e doorstep. |
| `slurm_submit.sh` | submit ONE model as a SPUR batch job (allocation derived from `handoff.tp`); prints `<job_id>\t<out_dir>`. |
| `slurm_job.sh` | SPUR **job body** (runs on the compute node): stage weights, forward the granted GPUs, then `node/run_local.sh`. |

**`node/` — per-node run pipeline (inside the allocation)**

| file | role |
|------|------|
| `run_local.sh` | per-node launcher: GPU preflight, background docker GPU run + monitor, `--dry-run` host-only wiring check, or `--probe` (real docker/GPU/weights + Claude, stop before the e2e workflow). Invoked by `dispatch/slurm_job.sh` on the compute node (or directly for local dev). |
| `run_model.sh` | Step E+F: run ONE model into a timestamped dir, then deterministically judge (status + measured baseline). |
| `run_geak_e2e.sh` | mirror Hyperloom's `run_e2e.py` launch; patches `exp_root`/`model_path`/`launch_recipe`/`inferencex_path` in the handoff. |

**`preflight/` — gates/prep before the workflow**

| file | role |
|------|------|
| `gpu_dstate_check.sh` | host-side GPU-wedge pre-check: scans `/proc` (touches no GPU) for tasks stuck in uninterruptible **D-state** in the amdgpu/kfd path. Runs BEFORE the probe so a hung driver fails fast instead of hanging the probe itself. |
| `gpu_healthcheck.sh` | GPU preflight probe run INSIDE the framework image: `rocminfo` + a tiny torch matmul on GPU 0. Fast, timeout-capped (`--kill-after` escalates to SIGKILL). |
| `setup_claude.sh` | Step D: install Claude into `$CLAUDE_HOME`, install `claude_agent_sdk`, probe `claude -p`. |
| `claude_setup.sh` | install + configure Claude Code (global AMD LiteLLM proxy). Re-run every container start (ephemeral fs). |

**`monitor/` — liveness watchdog**

| file | role |
|------|------|
| `run_monitor.sh` | host-side liveness monitor: every 5 min feeds `run.log` tail to a `claude -p` arbiter; `docker kill`s a confirmed-stuck run. |
| `monitor_prompt.md` | the monitor's instructions + `VERDICT: CONTINUE\|KILL` output contract. |

## SPUR / cluster topology

The L1 CI no longer runs on a single GPU box wired to GitHub. Instead:

```
GitHub PR label (l1-ci-smoke | l1-ci-full)
        │
        ▼
self-hosted runner on the JUMP/LOGIN box  (no GPU here)
        │  ci/dispatch/run_matrix.sh smoke|verify|probe
        ▼
sbatch  ──►  SPUR compute node(s)   (partition amd-spur, MI300x)
                 │  ci/dispatch/slurm_job.sh   (one model, tp GPUs, on 1 node)
                 │    • resolve weights from /home/ethany/hf_models (-> /shared_nfs)
                 │    • ci/node/run_local.sh  → docker → Claude → GEAK e2e → judge
                 ▼
        result.json under geak_runtime/<model>/ci_runs/<ts>/
        │
        ▼
run_matrix waits on all jobs, aggregates a pass/fail matrix (red if any fail)
```

- **GPU count is per model**: `handoff.tp` → `sbatch -G <tp> -N 1` (one node, co-located).
- **Account/partition/qos**: `amd-primus` / `amd-spur` / `amd-primus-qos` (override via
  `SPUR_ACCOUNT` / `SPUR_PARTITION` / `SPUR_QOS`). NB the *partition* is `amd-spur`;
  `amd-primus` is the account **and** the qos.
- **Weights** come from a per-model_key catalog `$HF_MODELS_DIR` (`/home/ethany/hf_models`);
  each entry is a dir OR a symlink into shared NFS, e.g.
  `hf_models/Qwen-Qwen3-8B -> /shared_nfs/huggingface_models/Qwen/Qwen3-8B`. Because the
  entries are symlinks, `node/run_local.sh` also bind-mounts `$WEIGHTS_EXTRA_MOUNTS`
  (default `/shared_nfs`, read-only, same-path) so the links — and HF hub-cache blob
  links (`snapshots/<h>/*.safetensors -> ../../blobs/<h>`) — resolve inside the container.
  A model absent from the catalog is downloaded from its `models.tsv` `hf_repo` into
  `$WEIGHTS_CACHE/<model_key>` (default = `$HF_MODELS_DIR`). Verified on a compute node:
  the catalog symlinks + NFS targets are readable and resolve inside a docker bind-mount.
- **Env propagation**: SPUR `sbatch --export=ALL` (default) carries the runner's env —
  including `LITELLM_*` secrets and `WS`/`HF_LOGS`/`INFERENCEX_PATH` — into the job.
- **Shared-FS requirement**: install the self-hosted runner under a shared path (e.g.
  `/home/ethany/...`) so the GEAK checkout it produces is readable by the compute node.
- **Enrollment**: only models with full GEAK material (see `geak_ci_workspace/TABLE.md`)
  are listed in `models.tsv`; the `verify` matrix runs exactly those.

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
├─ 0. lib.sh resolves: model → framework → IMAGE (docker_default.json), weights dir
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
| before probe | D-state pre-check (`gpu_dstate_check.sh`) | already-wedged driver (unkillable D-state tasks) — bail before our probe hangs too |
| before run | GPU preflight (`gpu_healthcheck.sh`) | dead/wedged GPU, docker/device broken |
| during run | Claude watchdog (`run_monitor.sh`) | mid-run stall, GPU wedge, NFS/OOM loop |
| after run | deterministic judge (`run_model.sh` Step F) | false-green `no_gain`, unmeasured baseline, errors |
| absolute | workflow `timeout-minutes` | everything above failing |

## Usage

### On the SPUR cluster (via the jump box)

```bash
cd <workspace>/GEAK

# show the sbatch commands WITHOUT submitting (validate wiring, no cluster use)
bash ci/dispatch/run_matrix.sh smoke  --print
bash ci/dispatch/run_matrix.sh verify --print
bash ci/dispatch/run_matrix.sh probe  --print

# L1 PROBE: fast end-to-end HARNESS check. Real SPUR allocation + docker + GPU
# preflight + image pull + weights mount (+ Claude install), but STOPS at the
# GEAK e2e doorstep — never runs the (hours-long) workflow. Judged on a probe_ok
# marker. 'probe' auto-selects the local-weight models (currently the 3 with NFS
# symlinks). Use this to verify SPUR/docker/weights before spending GPU-hours.
bash ci/dispatch/run_matrix.sh probe
# fastest infra-only probe (skip the Claude install step, too):
GEAK_PROBE_SKIP_CLAUDE=1 bash ci/dispatch/run_matrix.sh probe
# probe one model / an explicit set:
bash ci/dispatch/run_matrix.sh Qwen-Qwen3-8B --probe

# L1 smoke: one SPUR job for the tier==smoke model
bash ci/dispatch/run_matrix.sh smoke  --budget 1800

# L1 verify: one SPUR job per enrolled model, waited on + aggregated
bash ci/dispatch/run_matrix.sh verify --budget 57600

# a single model as one SPUR job (prints "<job_id>\t<out_dir>")
bash ci/dispatch/slurm_submit.sh Qwen-Qwen3-8B --budget 1800
```

### Directly on a GPU node (no SLURM — local dev / debugging)

```bash
cd <workspace>/GEAK

# host-only wiring check (no docker/GPU/Claude): validates handoff -> args mapping
bash ci/node/run_local.sh Qwen-Qwen3-8B --dry-run

# infra probe (real docker/GPU/weights + Claude, stop before the e2e workflow)
bash ci/node/run_local.sh Qwen-Qwen3-8B --probe

# real GPU smoke run (30-min budget)
bash ci/node/run_local.sh Qwen-Qwen3-8B --budget 1800

# pin a specific image instead of resolving from docker_default.json
IMAGE=rocm/vllm-dev:some-gfx950-tag bash ci/node/run_local.sh Qwen-Qwen3-8B
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
| `GEAK_SKIP_DSTATE_CHECK` | `0` | set `1` to skip the D-state wedge pre-check |
| `GEAK_DSTATE_SAMPLE_GAP_S` | `3` | seconds between the two D-state samples (sustained-detection window) |

The monitor no-ops gracefully if `claude` isn't on the host PATH (the run proceeds
unwatched, protected by the workflow's wall-clock cap).

## Tunables — one file

All timeouts / caps / intervals / toggles have their defaults in **`ci/config.sh`** — the single place to change them. It's sourced by `ci/lib.sh` (which every script sources or inherits from) and only `export`s values, so edits there apply everywhere (jump box, SPUR job, and container). Every entry is `${VAR:-default}`, so a one-off env override (CI secret, `--budget`, `FOO=... ci/...`) still wins over the file. The tables below list the same variables with their `config.sh` defaults.

## Environment overrides

| var | default | meaning |
|-----|---------|---------|
| `GEAK_ROOT` | `ci/..` | GEAK repo root |
| `WS` | `GEAK_ROOT/..` | workspace root |
| `INFERENCEX_PATH` | `$WS/InferenceX` | bench client checkout (empty = native bench) |
| `HF_LOGS` | `$WS/geak_runtime` | per-model dataset root |
| `MODELS_TSV` | `ci/models.tsv` | enrollment registry |
| `DOCKER_DEFAULT` | `ci/docker_default.json` (fallback `$HF_LOGS/docker_default.json`) | image selection map |
| `IMAGE` | resolved from `docker_default.json` | override the container image |
| `MODEL_PATH` | resolved on node (see below) | override weights dir |
| `PERFSKILLS_E2E_TIMEOUT_S` | `1800` | workflow wall-clock budget (also via `--budget`) |
| `LITELLM_API_KEY` / `LITELLM_BASE_URL` | **required** (no default; from CI secrets or local export) | Claude auth via the global LiteLLM proxy. `claude_setup.sh` errors if unset — nothing is hardcoded. |

### SPUR / weights overrides

| var | default | meaning |
|-----|---------|---------|
| `SPUR_PARTITION` | `amd-spur` | sbatch `-p` (the only partition on this cluster) |
| `SPUR_ACCOUNT` / `SPUR_QOS` | derived from `SPUR_ACCOUNT_FALLBACK` (i.e. `amd-hyperloom` / `amd-hyperloom-qos`) | sbatch `-A` / `--qos`. Used **only** when auto-select is off (`SPUR_AUTOSELECT=0`) or for `--print` display; with auto-select on they're overwritten per job by `pick_account()`. Export both (with `SPUR_AUTOSELECT=0`) to force a specific pool. |
| `SPUR_AUTOSELECT` | `1` | `slurm_submit.sh`: before submitting, probe `SPUR_ACCOUNT_CANDIDATES` in order and submit **this** job to the first account/QoS that can place its GPU footprint (`tp`) now — done **per model**, so a `tp=1` job can land on an account a `tp=8` job can't (the cluster has one partition + many idle nodes; the real limit is the per-QoS `QOSGrpNodeLimit`). `0` disables and uses `SPUR_ACCOUNT`/`SPUR_QOS` as-is. |
| `SPUR_ACCOUNT_CANDIDATES` | `amd-hyperloom:amd-hyperloom-qos amd-general:amd-general-qos amd-primus:amd-primus-qos` | space-separated `account:qos` priority list; probed in order, first that can place the job **now** wins. |
| `SPUR_ACCOUNT_FALLBACK` | `amd-hyperloom:amd-hyperloom-qos` | where to submit (and pend) when **no** candidate can place the job now. Independent of candidate order, so an account can be both first-choice and the pend-here fallback. |
| `SPUR_PROBE_WAIT_S` / `SPUR_PROBE_POLL_S` | `24` / `3` | the auto-select probe is a 1-node/**tp-GPU** job (matches the heaviest model), submitted then scancelled; this is how long to watch it before deeming a QoS full, and the poll interval. |
| _(pending policy)_ | n/a | `run_matrix.sh` waits on PENDING jobs **indefinitely** (there is no pending timeout). It returns only when **no** job is pending or running; a job that crashes/cancels is logged and the rest are still waited on. A *held* requeue (`JobHoldMaxRequeue`) is auto-released once, but jobs are never scancelled for pending too long — cancel a long-pending job by hand on the cluster. The GitHub `timeout-minutes` is the only outer backstop. |
| `SPUR_CPUS_PER_GPU` | `8` | cpus-per-task = `tp * this` |
| `SPUR_TIME_HEADROOM_S` | `7200` | added to the GEAK budget for the sbatch `-t` wall clock |
| `SPUR_PROBE_TIME` | `1:00:00` | fixed sbatch `-t` wall clock for `--probe` jobs (image pull + Claude, no e2e) |
| `GEAK_HARD_TIMEOUT_S` | `budget + SPUR_TIME_HEADROOM_S - GEAK_KILL_BUFFER_S - <pre-run elapsed>` | `run_local.sh` watchdog `docker kill`s the container after this many seconds — a clean, supervised cut ~`GEAK_KILL_BUFFER_S` BEFORE SLURM's `-t` (writes `timeout.json`, judged FAIL). The pre-run elapsed (D-state check + cold image pull + GPU preflight) is subtracted so the cut stays before SLURM's wall clock even after a slow cold-node pull. Prevents orphaned GPU containers on timeout. |
| `GEAK_KILL_BUFFER_S` | `300` | how far ahead of the SPUR wall clock the watchdog fires (kill before SLURM's untrappable SIGKILL) |
| `GEAK_PROBE_SKIP_CLAUDE` | `0` | `1` = skip the Claude install step in `--probe` (fastest infra-only check) |
| `HF_MODELS_DIR` | `/home/ethany/hf_models` | per-model_key weights catalog (dirs or symlinks into NFS) |
| `WEIGHTS_EXTRA_MOUNTS` | `/shared_nfs` | colon-separated NFS roots bind-mounted (ro, same-path) so catalog symlinks resolve in-container |
| `WEIGHTS_CACHE` | `$HF_MODELS_DIR` | writable dir weights are downloaded into (keyed by model_key) |
| `GEAK_ALLOW_DOWNLOAD` | `0` | `1` = allow `stage_weights` to auto-download from HF when local weights are absent. Default `0` = CI runs local-weight models only and fails fast otherwise (no silent multi-GB pulls). |
| `GEAK_MATRIX_POLL_S` | `60` | `run_matrix.sh` queue poll interval |
| `SPUR_DRYRUN` | `0` | `1` = print sbatch commands instead of submitting (same as `--print`) |

## Notes

- `NODE_TLS_REJECT_UNAUTHORIZED=0` is a stopgap for the internal corporate
  CA the base image doesn't trust. Preferred long-term fix: bake the corporate
  CA bundle into the image and drop the flag.
- Only the `claude-opus` family is served by the proxy, so the haiku/sonnet
  Claude Code defaults also point at `claude-opus-4-8`.
- The container runs with `--rm`; it's destroyed after each run. The pulled
  image is cached. Everything worth keeping is written to the mounted
  `ci_runs/<timestamp>/` dir.
- The container runs as **root** and executes GEAK Python from the runner's
  checkout (bind-mounted). To stop root-owned `__pycache__/*.pyc` from landing in
  the checkout — which the unprivileged runner then can't delete, failing the next
  `actions/checkout` clean with `EACCES: permission denied, unlink ...pyc` — the
  container is run with `PYTHONDONTWRITEBYTECODE=1` and `PYTHONPYCACHEPREFIX`
  pointed at the run's tmp dir. If you ever hit that EACCES again (e.g. from an
  older run), the offending dir is root-owned: rename it aside from a dir you own
  (`mv .../scripts/__pycache__ ~/.ci_pycache_trash/`) or remove it with root via a
  compute-node container; the jump box has neither docker nor passwordless sudo.
