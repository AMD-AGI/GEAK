# Preflight — Environment Self-Check (guidance, not a script)

> This is a **judgment guide**, not a fixed `doctor.sh`. A rigid preflight script has one failure
> mode the moment reality differs from its assumptions: it aborts, and a green/red bit tells you
> nothing about *how* to proceed. Instead, the **Director (PHASE=setup)** runs these checks itself
> with Bash/Read, **interprets** each result, **degrades gracefully** where it safely can, and only
> hard-stops on the few things that genuinely make a run meaningless. Treat every command below as a
> probe whose *output you reason about* — not a gate that must return 0.

## Operating principles
- **Probe → interpret → decide.** Each check yields one of: `ok` (proceed), `degrade` (proceed with a
  recorded limitation), or `block` (cannot produce a trustworthy throughput number → stop and report
  what's missing and how to fix it). Most checks are `degrade`, not `block`.
- **Write findings, don't just exit.** Record everything to `EVAL_DIR/env_report.md` (+ a compact
  `EVAL_DIR/env_report.json` the later phases can read). A run that proceeded *with known limitations*
  is far more useful than an opaque abort.
- **Never edit the environment to make a check pass.** Don't pip-install, don't change site-packages,
  don't download weights. If something required is missing, that's a `block` with a clear remedy — the
  user fixes it, not the workflow.
- **Adapt the plan to what you find.** Capability detected here flows downstream: no rocprofv3 →
  Profiler runs torch-trace only; aiter absent → drop aiter from candidate backends; gfx unknown →
  widen tuning search instead of trusting gfx942 priors.

## What's a `block` vs a `degrade`
| Condition | Verdict | Why |
|---|---|---|
| `MODEL` empty / path missing / not loadable | **block** | nothing to serve |
| chosen `BACKEND` import/CLI absent (no sglang / no `vllm`) | **block** (or switch backend if the other is present and the task allows) | can't launch the server |
| requested GPU id not visible | **block** | benchmarks would run on the wrong/no device |
| port busy | **degrade** | dispatcher auto-allocates a free port; just record it |
| rocprofv3 absent | **degrade** | Profiler falls back to torch-trace (shapes kept, HW durations approximate) |
| `amd-smi`/`rocminfo` absent | **degrade** | record gfx as "unknown"; widen tuning, don't trust gfx942 priors |
| aiter / CK profiler / hipblaslt-bench absent | **degrade** | remove those rungs from the backend ladder; note it |
| baseline bench spread > ~5% | **degrade→re-measure** | noisy box; re-run, raise the noise band, or pin clocks |

## Probes (run, then reason about the output)

**1. Backend resolve.** Decide `BACKEND` (arg, else default `sglang`). Confirm the stack is actually
importable/callable — don't trust the name:
```bash
# sglang:
python3 -c "import sglang; print('sglang', sglang.__version__)"   # block if this fails
# vllm:
python3 -c "import vllm; print('vllm', vllm.__version__)" && vllm --help >/dev/null
```
Confirm the matching adapter exists: `ls "$SKILL_DIR/scripts/adapters/${BACKEND}.sh"`. If the chosen
backend is absent but the other is present, note it and (only if the task is backend-agnostic) switch.

**2. Model.** `MODEL` must be set and resolvable. For a local path, check it exists and has a
`config.json`; for an HF id, note that first launch will download (and may be slow / need auth).
```bash
[ -n "$MODEL" ] || echo "BLOCK: MODEL unset"
[ -e "$MODEL" ] && ls "$MODEL"/config.json 2>/dev/null
```
Read `config.json` for the **architecture class** (dense / MoE / hybrid-mamba / MLA) and dtype — this
is the capability signal the Architect uses instead of guessing from kernel names. Record it.

**3. GPU visibility & arch.** Confirm the requested `gpu_ids` are actually present:
```bash
amd-smi list 2>/dev/null || rocm-smi --showid 2>/dev/null || rocminfo 2>/dev/null | grep -m1 gfx
```
Record gfx (e.g. `gfx942`). Unknown → `degrade` (don't apply gfx942-specific priors blindly).

**4. Profiler capability (degrade-friendly).** Prefer rocprofv3 for authoritative HW durations, but
never hard-require it:
```bash
command -v rocprofv3 || command -v rocprof || echo "no rocprof — torch-trace only"
```
Record which trace sources are available; the Profiler reads this from `env_report.json`.

**5. Tuning/backends present (shapes the ladder).** Probe the optional rungs; missing ones are simply
removed from the candidate list, not errors:
```bash
python3 -c "import aiter; print('aiter ok')" 2>/dev/null || echo "no aiter"
# FlyDSL (aiter's GEMM/attn DSL — SOTA author target for dense/quantized GEMM). It is NOT a top-level
# module: probe via aiter.ops.flydsl.is_flydsl_available() (a function), NOT `import flydsl` /
# `aiter.flydsl` (those raise ImportError even when FlyDSL is installed → false "no flydsl").
python3 -c "import aiter.ops.flydsl as f; print('flydsl', f.installed_flydsl_version if f.is_flydsl_available() else 'unavailable')" 2>/dev/null || echo "no flydsl"
command -v hipblaslt-bench || echo "no hipblaslt-bench (offline GEMM tune unavailable)"
command -v ckProfiler   || echo "no ckProfiler (CK instance sweep unavailable)"
```
When `is_flydsl_available()` is true, **flydsl MUST appear in `available_backends`** (it is reachable
via the aiter per-shape DB tune `libtype=flydsl` AND as a Tier-C author target). Do not infer its
absence from an `import flydsl` failure — only `is_flydsl_available()` is authoritative.

**Record WHY each optional backend is absent + HOW to provision it (don't just drop it).** For every
backend NOT in `available_backends`, write an `absent_backends[<name>] = {probe, remedy}` entry to
`env_report.json` so later phases can surface an ACTIONABLE hint instead of silently dropping a
strategy-mandated lever (the workflow itself never installs anything — the remedy is for the operator).
FlyDSL absence in particular has **two independent layers** — say which one is missing:
- The probe `python3 -c "import aiter.ops.flydsl as f; f.is_flydsl_available()"` can fail at the
  `import aiter.ops.flydsl` step (`ModuleNotFoundError: No module named 'aiter.ops.flydsl'`) → the
  installed **`amd_aiter` build does not ship the `aiter/ops/flydsl/` wrapper** (the layer holding
  `is_flydsl_available`, `flydsl_preshuffle_gemm_a8`, etc.). This is the usual blocker.
- Or it imports but `is_flydsl_available()` returns False → the separate **top-level `flydsl` package**
  (`importlib.util.find_spec("flydsl")`) is missing.
`pip install flydsl` only fixes the SECOND layer; if the first is missing, `aiter.ops.flydsl` stays a
`ModuleNotFoundError` even after it. A correct remedy therefore names BOTH parts, e.g.:
```json
"absent_backends": {
  "flydsl": {
    "probe": "import aiter.ops.flydsl -> ModuleNotFoundError (wrapper missing); and/or is_flydsl_available()==False",
    "remedy": "FlyDSL needs BOTH (1) the top-level `flydsl` pip pkg (`pip install 'flydsl>=0.1.5'`, on PyPI) AND (2) aiter's `aiter/ops/flydsl/` wrapper, which ships only in a flydsl-enabled `amd_aiter` build. On this image (1) alone is insufficient: `aiter.ops.flydsl` is still ModuleNotFoundError after `pip install flydsl`. Install a flydsl-enabled `amd_aiter` build (or restore `aiter/ops/flydsl/` from a matching aiter source), then re-run. The workflow will NOT install it for you.",
    "mandated_by": "fp8/quantized GEMM head Tier-C author (op_benchmarker default orders flydsl first)"
  }
}
```

**6. Tooling.** `curl`, `python3`, free disk under `EXP_ROOT`. Missing `curl` → adapters that health-
check via curl must be adjusted (note it); low disk → `block` (traces + overlays need room).

**7. Smoke the measurement path (the real test).** The only check that proves the stack works
end-to-end is a tiny warm bench. Do ONE short run via the dispatcher and confirm it prints an
`E2E_SUMMARY` line with a sane number:
```bash
OUT_DIR="$EVAL_DIR/preflight_smoke" BACKEND="$BACKEND" MODEL="$MODEL" GPU="<first gpu>" \
ISL=128 OSL=32 CONC=4 NUM_PROMPTS=8 REPEATS=1 PROFILE=0 \
  bash "$EVAL_DIR/bench_e2e.sh" 2>&1 | tee "$EVAL_DIR/logs/preflight_smoke.log"
```
If this fails, read the server log it points to and diagnose (wrong flag for this image, OOM →
lower `MEM_FRACTION`, missing `--trust-remote-code`, etc.). Capture any `EXTRA_SERVER_ARGS` the image
needs so the real baseline uses them. This is also where vllm CLI drift is caught (see
`scripts/adapters/vllm.sh`).

## Output (always write, even on block)
Write `EVAL_DIR/env_report.md` (human) and `EVAL_DIR/env_report.json` (machine), e.g.:
```json
{
  "backend": "sglang", "backend_version": "0.5.11",
  "model": "/path", "model_arch_class": "hybrid_mamba_moe", "model_dtype": "bf16",
  "gfx": "gfx942", "gpu_ids": ["0"],
  "trace_sources": ["torch"],            // add "rocprofv3" if present
  "available_backends": ["aiter","hipblaslt","triton","flydsl"], // include "flydsl" iff aiter.ops.flydsl.is_flydsl_available(); aiter/ck/flydsl removed only if absent
  "absent_backends": {                    // one entry per OPTIONAL backend NOT available, with an actionable remedy (see probe 5)
    "flydsl": {"probe": "import aiter.ops.flydsl -> ModuleNotFoundError", "remedy": "needs both `pip install 'flydsl>=0.1.5'` AND a flydsl-enabled amd_aiter build (ships aiter/ops/flydsl/); pip flydsl alone is insufficient on this image", "mandated_by": "fp8 GEMM head author"}
  },
  "port": 31037,                          // the auto-allocated port, if any
  "limitations": ["rocprofv3 absent: HW durations approximate; ranking from torch trace"],
  "verdict": "ok|degrade|block",
  "blockers": []                          // populated only on block, each with a remedy
}
```
Downstream phases read `env_report.json`: the Profiler picks its trace sources from `trace_sources`,
the Architect routes using `model_arch_class` + `available_backends`, the bake-off ladder uses
`available_backends`, and tuning priors are gated on `gfx`. The **Op Benchmarker gates its `author_plan`
on `available_backends`** (a backend in `absent_backends` is NOT emitted as an author lane — it is
emitted as a `backend_absent` advisory), and the **report renders `absent_backends` as a BACKEND_ABSENT
(env-provisioning) section** so a mandated-but-missing lever is never silently dropped.

> Bottom line: preflight's job is not to pass or fail — it's to **hand the rest of the run an accurate
> picture of this machine** so every later decision is made against reality instead of assumptions.
> When something is missing, prefer a recorded limitation over an abort; reserve `block` for the
> handful of conditions that make the throughput number meaningless.
