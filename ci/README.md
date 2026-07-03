# GEAK_v4 e2e CI harness

Local/self-hosted CI for running the GEAK ("perfskills") end-to-end kernel
optimization workflow on a ROCm GPU box, exactly the way Hyperloom's
`KERNEL_AGENT` phase launches it (`interface/run_e2e.py`).

Everything is driven from `ci/run_local.sh`. A run:

1. resolves the container image + model weights for a model key,
2. spins up the ROCm/vllm container with GPU passthrough,
3. installs Claude Code + the Python `claude_agent_sdk` inside it,
4. runs the GEAK e2e workflow for one model into a timestamped folder,
5. hard-judges the result on `result.json.status`.

## Layout it expects

Paths are **derived from each script's location** — no absolute machine paths
are baked in. The workspace just needs to look like this (siblings of the repo):

```
<workspace>/
  GEAK/                 # this repo (contains ci/, interface/, e2e_workflow/)
  InferenceX/           # bench client (cloned separately)
  huggingface_logs/     # per-model handoff.json / recipe / tracelens priors (cloned separately)
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
| `run_geak_e2e.sh` | mirror Hyperloom's `run_e2e.py` launch; patches `exp_root`/`model_path`/`launch_recipe`/`inferencex_path` in the handoff. |
| `run_model.sh` | Step E+F: run ONE model into a timestamped dir, then deterministically judge. |
| `run_local.sh` | top-level launcher: docker GPU run, or `--dry-run` host-only wiring check. |

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

Outputs land in `huggingface_logs/<model_key>/ci_runs/<timestamp>/`:

- `run.log` — full stdout/stderr of the run
- `result.json` — the workflow result (source of truth for pass/fail)
- `claude/` — `$CLAUDE_HOME` (Claude install + config + logs), persisted
- `claude_tmp/` — container `TMPDIR`, incl. Claude's background-task tree, kept for post-mortem debugging

**Pass criteria:** exit 0 and `result.json.status` in `{ok, no_gain}`. Anything
else fails. `error_class = workflow_parse_error` is the fingerprint of the
`claude -p` CLI fallback (Python `claude_agent_sdk` missing in the container).

## Environment overrides

| var | default | meaning |
|-----|---------|---------|
| `GEAK_ROOT` | `ci/..` | GEAK repo root |
| `WS` | `GEAK_ROOT/..` | workspace root |
| `INFERENCEX_PATH` | `$WS/InferenceX` | bench client checkout (empty = native bench) |
| `HF_LOGS` | `$WS/huggingface_logs` | per-model dataset root |
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
