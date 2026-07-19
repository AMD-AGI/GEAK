# Onboarding — adding a new model or docker image to L1 CI

Step-by-step for the two most common changes: **enrolling a new model** and
**adding/updating a container image**. For the overall architecture and knobs
see [`README.md`](./README.md).

All paths below are relative to the workspace root (`$WS`, default
`/home/ethany/geak_ci_workspace`), where the layout is:

```
$WS/
  GEAK/                 # this repo (ci/, interface/, e2e_workflow/)
  InferenceX/           # bench client (cloned separately)
  geak_runtime/         # per-model data root ($HF_LOGS)
    <model_key>/
      handoff.json                     # REQUIRED — properties + task (schema_version 1)
      baseline_config.with_envs.yaml   # REQUIRED — launch recipe
      kernel-agent/**/tracelens/analysis.md            # TraceLens priors (optional but recommended)
      kernel-agent/**/kernel_candidates.json
      kernel-agent/**/tracelens/tracelens_report.json
      runs/roofline/**/torch_trace                     # roofline torch trace dir
      geak/                            # exp_root — CREATED AT RUN TIME (eval dirs land here)
```

Model **weights** live in the catalog `$HF_MODELS_DIR` (default
`/home/ethany/hf_models`), keyed by `<model_key>` — NOT under `geak_runtime/`.

---

## A. Enrolling a new model

### 1. Stage the weights (CI never auto-downloads)
Put the weights at `$HF_MODELS_DIR/<model_key>` — either a real directory or a
symlink into shared NFS, e.g.:

```bash
ln -s /shared_nfs/huggingface_models/Qwen/Qwen3-8B /home/ethany/hf_models/Qwen-Qwen3-8B
```

CI resolves weights by `<model_key>` and **refuses to download** by default (a
stray enrollment must not silently pull 100+GB and burn a GPU alloc). To allow a
one-off HF download instead of pre-staging, set `GEAK_ALLOW_DOWNLOAD=1` and give a
real `hf_repo` in `models.tsv` (see step 3).

### 2. Drop the per-model GEAK material under `geak_runtime/<model_key>/`
- **`handoff.json`** (required, `schema_version: 1`) — the single source of truth
  for `framework` and `tp` (they are NOT in `models.tsv`). Required non-empty keys:
  `model_path` and `exp_root` (both are hard-asserted; `exp_root` is rewritten at
  run time so any non-empty placeholder is fine). Functional fields used by the run:

  ```json
  {
    "schema_version": 1,
    "model_path": "<placeholder or real path; overridden by staged weights>",
    "framework": "vllm",                 // vllm | sglang  -> picks the image
    "gpu_type": "mi300x",
    "tp": 1,                              // tensor-parallel = GPUs to allocate on SPUR
    "workload": { "isl": 1024, "osl": 1024, "conc": 64 },
    "accepted_flags": "--kv-cache-dtype fp8 ...",
    "accepted_env": "VLLM_USE_AITER=1 ...",
    "launch_recipe": "<overridden to the local baseline_config.with_envs.yaml>",
    "raw_baseline_tput": 0,
    "exp_root": "<placeholder; overridden to geak_runtime/<model_key>/geak>",
    "bench_client": "auto",
    "inferencex_path": "<overridden to $WS/InferenceX>",
    "gpu_ids": "0",
    "bench_protocol": { "random_range_ratio": 1.0, "num_prompts": 320, "num_warmups": 8 }
  }
  ```

  At run time `run_geak_e2e.sh` localizes the handoff into `handoff.patched.json`,
  overwriting `exp_root`, `launch_recipe`, `inferencex_path`, and `model_path`
  (only if `MODEL_PATH` is set) with local values; everything else is used as-is.

- **`baseline_config.with_envs.yaml`** (required) — the vLLM/sglang launch recipe.

- **TraceLens priors** (recommended — enables the profiler's fast-path and roofline
  enrichment; without them the run just re-collects a torch trace):
  - `kernel-agent/**/tracelens/analysis.md`
  - `kernel-agent/**/kernel_candidates.json`
  - `kernel-agent/**/tracelens/tracelens_report.json`
  - `runs/roofline/**/torch_trace` (a directory holding the rank0 serving trace)

  > **CRITICAL layout rule:** these priors live in the **model dir**
  > (`geak_runtime/<model_key>/kernel-agent`, `.../runs`), NOT inside `geak/`.
  > `run_model.sh` sets `exp_root = <model_key>/geak`, and
  > `run_e2e.py:resolve_tracelens_report()` strips the trailing `geak` to search
  > one level up (the model dir). If you nest the priors under `geak/` or rename
  > `exp_root` to anything but `geak`, discovery silently misses them and the run
  > falls back to `source: torch-trace`.

### 3. Add a row to [`models.tsv`](./models.tsv)
TAB-separated: `<model_key>\t<hf_repo>\t<tier>`

```
my-org-my-model	my-org/my-model	verify
```

- `model_key` — must match the `geak_runtime/<model_key>/` folder name.
- `hf_repo` — HF repo id for optional download, or `-` if weights are pre-staged.
- `tier` — `smoke` (the single L1-smoke model, also in the full matrix) or
  `verify` (full `l1-ci-full` matrix only).

### 4. Validate before spending GPU time
```bash
# From the repo root. See the exact sbatch commands without submitting:
ci/dispatch/run_matrix.sh <model_key> --print

# Host-only wiring check (no docker/GPU/Claude):
SPUR_DRYRUN=1 ci/node/run_local.sh <model_key> --dry-run

# Infra-only probe (real SPUR/docker/GPU/weights + Claude, stops before the e2e):
ci/dispatch/run_matrix.sh probe        # runs the probe tier over locally-weighted models
```
A model only joins the `probe`/auto-run sets once its weights are present locally
(`weights_present`). Confirm the TraceLens priors resolve by checking the first
run's `profile_topN.md` shows `source: tracelens` (or `tracelens+trace`).

---

## B. Adding or updating a container image

Images are chosen from [`docker_default.json`](./docker_default.json), a nested map
`{ "<framework>": { "<arch>": "<image>" } }`. `arch` is auto-detected on the
compute node as **`MI300`** (gfx942/gfx90a) or **`MI355`** (gfx950), overridable
with `GEAK_GPU_ARCH`.

```json
{
  "vllm":   { "MI300": "…/vllm-openai-rocm:<tag>",  "MI355": "…/vllm-openai-rocm:<tag>" },
  "sglang": { "MI300": "…/sglang:<tag>",            "MI355": "…/sglang:<tag>" }
}
```

To add/update an image:
1. Edit `ci/docker_default.json` — add a new `<framework>` block, or update the
   `MI300`/`MI355` tag for an existing one. (This repo file is authoritative; there
   is no workspace fallback.)
2. A model picks its image by its `handoff.json` `framework` + the node's arch.

Overrides (no file edit needed):
- `IMAGE=<repo:tag>` — force one exact image for the run.
- `DOCKER_DEFAULT=<path>` — point at an alternate map file.

Notes:
- Prefer the AMD `primussafe` **`profilerfix`** images: the vLLM one is built for
  both gfx942 and gfx950 and fixes the stock `vllm/vllm-openai-rocm:v0.21.0` torch
  eager SIGSEGV on gfx950.
- New/large images pull cold on first use; the run-time watchdog subtracts the
  cold-pull time from its hard-timeout so it still cuts before SLURM's wall clock.
