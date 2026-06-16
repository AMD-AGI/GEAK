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
command -v hipblaslt-bench || echo "no hipblaslt-bench (offline GEMM tune unavailable)"
command -v ckProfiler   || echo "no ckProfiler (CK instance sweep unavailable)"
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
  "available_backends": ["hipblaslt","triton"],   // aiter/ck removed if absent
  "port": 31037,                          // the auto-allocated port, if any
  "limitations": ["rocprofv3 absent: HW durations approximate; ranking from torch trace"],
  "verdict": "ok|degrade|block",
  "blockers": []                          // populated only on block, each with a remedy
}
```
Downstream phases read `env_report.json`: the Profiler picks its trace sources from `trace_sources`,
the Architect routes using `model_arch_class` + `available_backends`, the bake-off ladder uses
`available_backends`, and tuning priors are gated on `gfx`.

> Bottom line: preflight's job is not to pass or fail — it's to **hand the rest of the run an accurate
> picture of this machine** so every later decision is made against reality instead of assumptions.
> When something is missing, prefer a recorded limitation over an abort; reserve `block` for the
> handful of conditions that make the throughput number meaningless.
