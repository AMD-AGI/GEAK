# GEAK v4 Compatibility Matrix

## Scope

Verified hardware, software, runtime, and backend combinations for **GEAK v4** (a Claude Code +
JS-Workflow GPU optimizer; no pip package, no CLI). Only tested configurations are listed.

---

## 1. Runtime — Claude Code

| Component | Version / Requirement | Status | Notes |
|---|---|---|---|
| Claude Code | **≥ 2.1.177** | Required | The workflows use the **dynamic Workflow** (JS orchestration) feature, available only from this version. Check with `claude --version`. |
| Launch mode | `IS_SANDBOX=1 claude --dangerously-skip-permissions` | Required | Workflows spawn sub-agents and run profiling / benchmark / build commands on the box, so permissions must be auto-approved. |
| Default model | `claude-opus-4-8` | Verified | Default used by the external-orchestrator entry point (`interface/run_e2e.py`). |
| Effort | `ultracode` | Verified | Default effort for `interface/run_e2e.py`. |

---

## 2. Invocation Mode

| Mode | How | Status |
|---|---|---|
| Natural language → `Workflow` tool | Describe the task to Claude Code; it maps the prompt onto `Workflow({ scriptPath, args })` | Verified |
| Direct `Workflow` call (e2e) | `scriptPath: "<repo>/e2e_workflow/e2e_workflow.js"` | Verified |
| Direct `Workflow` call (single kernel) | `scriptPath: "<repo>/kernel_workflow/kernel_workflow.js"` | Verified |
| Batch (many kernels) | `scriptPath: "<repo>/kernel_workflow/kernel_workflow_bmk.js"` | Verified |
| External orchestrator (Hyperloom) | `python interface/run_e2e.py <handoff.json> <result.json>` | Verified |

---

## 3. Operating System

| OS | Status |
|---|---|
| Ubuntu | Verified |

---

## 4. Python

| Python version | Status | Notes |
|---|---|---|
| 3.8+ | Required minimum | Stated in README prerequisites. |
| 3.12 | Verified | Compiled artifacts in the tree are cpython-312. |

---

## 5. GPU Hardware

The on-box card is auto-detected (`rocminfo` / `rocm_agent_enumerator`); `PYTORCH_ROCM_ARCH` is pinned
to the local `gfx` at build time.

| Architecture | gfx target | Example cards | Status |
|---|---|---|---|
| CDNA3 | gfx942 | MI300X, MI308X | Verified |
| CDNA4 | gfx950 | MI355X | Verified |


---

## 6. ROCm Stack

| Component | Version / Requirement | Status |
|---|---|---|
| ROCm | 7.2.x | Verified |
| ROCm | 7.1.x | Verified |
| ROCm | 7.0.x | Verified |
| ROCm | 6.4.x | Verified |

---

## 7. Profilers

The profiler is auto-detected via a degrade ladder
(`PROFILER_PRIORITY="rocprof-compute rocprofv3 rocprof"`).

| Profiler | Status |
|---|---|
| `rocprof-compute` | Verified (preferred) |
| `rocprofv3` | Verified |
| `rocprof` | Verified |

---

## 8. Serving Backends (e2e_workflow)

The serving stack is not baked in; `args.backend` selects `scripts/adapters/<backend>.sh`.

| Backend | Default port | Status |
|---|---|---|
| sglang | 30000 | Verified (default) |
| vllm | 8000 | Verified |

| Bench client | Status | Notes |
|---|---|---|
| Backend-native (`bench_e2e.sh`) | Verified | Default dispatcher. |
| inferencex | Verified | Opt-in via `BENCH_CLIENT=inferencex` (Hyperloom / Magpie parity); needs `$INFERENCEX_PATH`. |

---

## 9. Kernel Languages (kernel_workflow)

`target_language` for author mode; also the languages the head-kernel bake-off can win with.

| Kernel language | Status | Notes |
|---|---|---|
| Triton | Verified | Always a viable author target. |
| FlyDSL | Verified | Preferred author target for dense / quantized GEMM (aiter's SOTA GEMM DSL, JIT, no build). Probed via `aiter.ops.flydsl.is_flydsl_available()`. |
| HIP | Verified | Used when headroom justifies it. |
| CK (Composable Kernel) | Verified | Used when headroom justifies it; FP8 GEMM tuning. |
| cutlass_port, gluon, mojo, tilelang, rocwmma, asm_mfma, hipkittens | Reference (perf_knowledge) | Documented, not part of the default author ladder. |

---

## 10. Op-backend Bake-off (op_bench.py head kernels)

| Backend | Status |
|---|---|
| hipblaslt | Verified |
| rocblas / TunableOp | Verified |
| aiter | Verified |
| triton (with `--triton-autotune`) | Verified |

---

## 11. Precision / Data Types

| Data type | Status | Notes |
|---|---|---|
| FP16 / BF16 | Verified | General kernel optimization. |
| FP8 | Verified | Head-GEMM tuning + author target; gsm8k accuracy gate recommended for quantized kernels. |
| FP4 / MXFP | Verified | Quantized GEMM (A4W4); accuracy gate recommended. |

---

## 12. Accuracy Gate (e2e_workflow)

| Gate | Requirement | Status |
|---|---|---|
| `none` (default) | Throughput delta + greedy output parity | Verified |
| `gsm8k` | Sampled gsm8k (5-shot, greedy, fixed seed) via `scripts/gsm8k_eval.py` against an OpenAI-compatible `/v1` endpoint | Verified |

---

## Notes

- The on-box GPU arch, backend, profiler, and op backends are re-checked at runtime by the Setup
  **preflight**, which writes `env_report.{md,json}`.
- To report a verified configuration not listed here, please open a pull request.
