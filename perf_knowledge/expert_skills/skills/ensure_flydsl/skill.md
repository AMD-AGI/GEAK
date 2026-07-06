---
id: ensure_flydsl
title: Ensure FlyDSL is importable (version-gated reuse, else build the pinned source)
kind: expert_skill
authors:
- geak
scope: dependency          # a build/dependency skill: makes flydsl usable; invoked BY other flydsl skills, not profile-matched on its own
match:
  needs: flydsl
  arch_class:
  - '*'
  gens:
  - gfx942
  - gfx950
  dtypes:
  - int4_w4a16
  - fp8_e4m3_fnuz
role: dependency_provider
supersedes: []
---

# Ensure FlyDSL is importable (single source of truth for the FlyDSL build)

This skill makes `import flydsl, kernels.moe_gemm_2stage` resolve from ONE consistent tree, so the
FlyDSL optimization skills (`flydsl_rewrite_quantized_moe` for authoring, `apply_flydsl_moe_to_vllm`
for the e2e apply-back) can run. It is a **dependency skill**: other skills / the System Architect
invoke it; it is not selected from the profile Top-N on its own.

## When to run
As **step 0** of any FlyDSL skill, the first time flydsl is about to be authored or applied. The
Architect routes flydsl as a build-on-demand candidate in Strategize (see `roles/system_architect.md`
§0a) but does NOT build there; the build happens here, once, right before first use. Idempotent — safe
to invoke from every flydsl skill; later calls short-circuit.

## Procedure
Run the bundled executor (do NOT hand-roll clone/build — it carries the version gate, the ROCm-image
fixes, the concurrency lock, and the env file):

```bash
bash "$(dirname "$0")/ensure_flydsl.sh"    # or: perf_knowledge/expert_skills/skills/ensure_flydsl/ensure_flydsl.sh
source "${FLYDSL_ROOT:-/opt/flydsl/FlyDSL}/flydsl_env.sh"
python3 -c "import flydsl, kernels.moe_gemm_2stage as k; print(flydsl.__file__, k.__file__)"   # must resolve under ONE tree
```

## What the executor guarantees (`ensure_flydsl.sh`)
- **Version-only reuse gate**: if an ambient flydsl+kernels is importable with `__version__ >=
  FLYDSL_MIN_VERSION` (default `0.2.2`), REUSE it — never overwrites a newer flydsl, never pip-installs
  system-wide (portable / does not disturb other workloads). The pinned commit is NOT used to gate reuse.
- **Else clone + build the PIN** (`a35627a2…`) into container-internal `/opt/flydsl` (overlay fs, not a
  bind-mounted host dir), guarded by an `flock` so concurrent Setup/author retries can't corrupt one tree.
- **ROCm-image compat fixes applied before the build**: hip cmake symlink (cmake refs
  `libamdhip64.so.7.2.70201` but the runtime is `7.2.53211`) and auto-install `patchelf` (build.sh's
  `CopyFlyPythonSources` needs it) — the two failures that otherwise kill the source build late.
- **Writes `flydsl_env.sh`** (`FLYDSL_ROOT` / `PYTHONPATH` / `FLYDSL_SHIM_DIR` / `VLLM_USE_FLYDSL_MOE`)
  for the workflow + vLLM + subagents to source.
- Exit 0 ⇒ flydsl importable + env written. Non-zero ⇒ install failed (loud); the calling skill must
  drop flydsl and record why (do not silently fall back to triton).

## Overrides
- `FLYDSL_ROOT` — build/checkout location (default `/opt/flydsl/FlyDSL`).
- `FLYDSL_MIN_VERSION` — reuse floor (default `0.2.2`).
- `FLYDSL_BUILD_LOCK` — lock file path.
