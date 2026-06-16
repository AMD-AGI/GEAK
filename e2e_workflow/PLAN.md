# e2e_workflow — Design & Build Plan

## Goal
Extend the single-kernel `kernel_workflow` into an end-to-end LLM inference-throughput optimizer for
sglang/vllm on AMD Instinct MI GPUs, **without changing the single-kernel workflow** and while staying backward
compatible with single-kernel optimization. Spec: `../instruct_e2e.md`.

## Locked design decisions (2026-06-03, via interactive Q&A)
1. **Topology = fractal two-altitude.** A new system layer wraps the UNCHANGED kernel layer; kernel
   optimization recursively calls `../kernel_workflow/kernel_workflow.js`. Guarantees single-kernel effect
   can't regress.
2. **Independent System Architect** owns e2e strategy + Amdahl budgeting + the persistent
   cross-run experience library (`knowledge/backend_playbook.md`).
3. **Dedicated Kernel Extractor** captures shapes + a reference I/O oracle → an immutable standalone
   unittest task dir (anti-cheating; the kernel layer consumes it unchanged).
4. **Dedicated Config Tuner**, runs FIRST, default ON — flag/env/source-backend sweep; reshapes the
   profile so we re-profile after it.
5. **Dedicated e2e Integrator/Validator** does reversible overlay reintegration + the e2e throughput
   gate, on a milestone cadence with a warm server and an Amdahl gate.
6. **Profile = mixed Top-N** (not forced per-stage), but each entry carries shapes; a same-named
   kernel in prefill vs decode = different shape regimes → optimization may produce regime-specific
   variants.
7. **Patch form = runtime monkeypatch / PYTHONPATH overlay** (reversible, never edits site-packages).
8. **Env isolation = PYTHONPATH overlay** (copy only the touched package subtree).

## Roles (system layer)
| role | phases | owns |
|---|---|---|
| e2e Director | setup, validate | isolated env, TRUE baseline throughput, final independent re-measure + parity + arbitration |
| System Architect | strategize, plan_milestone, update_experience, report | Amdahl routing, budget, stop rule, persistent playbook |
| Profiler | baseline, reprofile | warm trace → standardized Top-N (`parse_profile.py`) |
| Config Tuner | sweep | Tier-0 flags/env/backends, one axis at a time |
| Kernel Extractor | extract | shapes + oracle → immutable unittest task dir |
| e2e Integrator/Validator | integrate, finalize | reversible overlay, e2e gate, deliverable bundle |
| (kernel squad) | — | UNCHANGED `../kernel_workflow/kernel_workflow.js`, called recursively |

## Pipeline (deterministic, in `e2e_workflow.js`)
`Setup → Baseline Profile → Strategize → [ConfigSweep → Re-profile → Re-strategize] →
LOOP milestone[ plan → per-kernel(Extract → recursive kernel layer → Overlay+e2e gate) → Re-profile → grow playbook ] →
Finalize → Report → Validate`.

- **Budget** = number of kernel-optimization tasks (config sweep is free). `noImprove<2` early-stop.
- Each accepted change compounds into the carried-forward overlay + config.
- Throughput measured warm, repeated, median, vs TRUE baseline; every kernel gated on e2e delta >
  noise band AND output parity.

## Measurement discipline
Warm server always; ≥3 repeats, median + spread; profile and bench share ISL/OSL/conc; output parity
(greedy/temp=0, fixed seed) on every numeric-changing step; accept only deltas above the noise band.

## Build status
- [x] `scripts/parse_profile.py` — standardized Top-N (validated on the real 27B trace: found
      hipBLASLt GEMM at ~79% gpu time).
- [x] `scripts/bench_e2e.sh` — warm-server throughput bench (median + spread, overlay/flags/env, reuse mode).
      VALIDATED: reproduces baseline 1532.3 tok/s, 0.02% spread over 2 repeats (orig ~1533).
- [x] `scripts/capture_shapes.py` — hook-based shape + reference I/O oracle capture.
- [x] `scripts/overlay_setup.py` — reversible sitecustomize/monkeypatch overlay builder (manifest-driven).
      VALIDATED against real sglang: add-rebind (silu_and_mul→sentinel) and add-module (sys.modules
      injection + parent attr bind) both work; `import sglang` still resolves; no package shadowing.
- [x] knowledge: `e2e_optimization.md`, `profile_parse.md`, `backend_playbook.md` (persistent),
      `sglang_internals.md`, `shape_capture.md`.
- [x] roles: director, system_architect, profiler, config_tuner, kernel_extractor, e2e_integrator.
- [x] `e2e_workflow.js` orchestration (+ single-kernel pass-through). node --check OK.
- [~] Validate on Qwen3.5-27B end-to-end. Component-level + critical-path validation DONE (see below).
      Full pipeline run requires the Workflow tool (recursive kernel-layer dispatch), which the build
      agent cannot invoke directly — to be run by the user via the Workflow tool.

## Validation results (2026-06-03)
- **overlay_setup.py** — re-validated after the package-shadowing rewrite: add-rebind + add-module both
  work on real sglang; overlay dir contains no `sglang/` so shadowing is structurally impossible.
- **bench_e2e.sh** — end-to-end run reproduced the TRUE baseline at 1532.3 tok/s (0.02% spread). The
  tight spread means the 3% noise band is conservative and even ~1-2% deltas are detectable.
- **parse_profile.py** — found hipBLASLt GEMM at ~79% gpu time on the real 27B trace.
- **Bottleneck map** (ISL/OSL=1024 conc=64): hipBLASLt GEMM (~79%, library → Config Tuner: Tensile DB /
  aiter GEMM), then mamba/gated-delta Triton kernels (editable → kernel squad), CK attention, aiter
  rmsnorm. aiter attn + aiter tuned GEMM are ALREADY the default in this image (seen in server logs).
- **Open:** the cheapest real win is the Tier-0 config sweep; first editable kernel-squad target is the
  gated-delta linear-attn Triton path. A small-budget full run via the Workflow tool is the next step.
