---
myst:
    html_meta:
        "description": "Supported hardware, software, runtime, and backend combinations for GEAK v4: AMD Instinct GPUs, ROCm, Claude Code, serving backends, kernel languages, and data types."
        "keywords": "GEAK, compatibility, ROCm, AMD Instinct, MI300X, MI355X, sglang, vLLM, Triton, HIP, CK, FlyDSL, Claude Code"
---

# GEAK compatibility matrix

Supported hardware, software, runtime, and backend combinations for GEAK v4 (a Claude Code +
JS-Workflow GPU optimizer; no pip package, no CLI). Only tested configurations are listed.

## Runtime — Claude Code

| Component | Version or requirement | Status | Notes |
|---|---|---|---|
| Claude Code | ≥ 2.1.177 | Required | The workflows use the dynamic Workflow (JS orchestration) feature, available only from this version. Check with `claude --version`. |
| Launch mode | `IS_SANDBOX=1 claude --dangerously-skip-permissions` | Required | Workflows spawn sub-agents and run profiling, benchmark, and build commands on the box, so permissions must be auto-approved. |
| Default model | `claude-opus-4-8` | Supported | Default used by the external-orchestrator entry point (`interface/run_e2e.py`). |
| Effort | `ultracode` | Supported | Default effort for `interface/run_e2e.py`. |

## Invocation mode

| Mode | How | Status |
|---|---|---|
| Natural language → `Workflow` tool | Describe the task to Claude Code; it maps the prompt onto `Workflow({ scriptPath, args })` | Supported |
| Direct `Workflow` call (e2e) | `scriptPath: "<repo>/e2e_workflow/e2e_workflow.js"` | Supported |
| Direct `Workflow` call (single kernel) | `scriptPath: "<repo>/kernel_workflow/kernel_workflow.js"` | Supported |
| External orchestrator (Hyperloom) | `python interface/run_e2e.py <handoff.json> <result.json>` | Supported |

## Operating system

| OS | Status |
|---|---|
| Ubuntu 22.04 | Supported |
| Ubuntu 24.04 | Supported |

## Python

| Python version | Status | Notes |
|---|---|---|
| 3.8+ | Required minimum | Stated in the README prerequisites. |
| 3.12 | Supported | Compiled artifacts in the tree are cpython-312. |

## GPU hardware

The on-box card is auto-detected (`rocminfo` / `rocm_agent_enumerator`); `PYTORCH_ROCM_ARCH` is pinned
to the local `gfx` at build time.

| Architecture | gfx target | Example cards | Status |
|---|---|---|---|
| CDNA3 | gfx942 | MI300X | Supported |
| CDNA4 | gfx950 | MI355X | Supported |

## ROCm stack

| Component | Version / requirement | Status |
|---|---|---|
| ROCm | 7.2.x | Supported |
| ROCm | 7.1.x | Supported |
| ROCm | 7.0.x | Supported |
| ROCm | 6.4.x | Supported |

## Profilers

The profiler is auto-detected using a degrade ladder
(`PROFILER_PRIORITY="rocprof-compute rocprofv3 rocprof metrix"`).

| Profiler | Status |
|---|---|
| `rocprof-compute` | Supported (preferred) |
| `rocprofv3` | Supported |
| `rocprof` | Supported |
| `metrix` 0.1.0 | Supported |

## Serving backends (e2e_workflow)

The serving stack is not baked in; `args.backend` selects `scripts/adapters/<backend>.sh`.

| Backend | Status |
|---|---|
| sglang | Supported |
| vllm | Supported |

| Bench client | Status | Notes |
|---|---|---|
| Backend-native (`bench_e2e.sh`) | Supported | Default dispatcher. |
| inferencex | Supported | Opt-in using `BENCH_CLIENT=inferencex` (Hyperloom or Magpie parity); needs `$INFERENCEX_PATH`. |

## Kernel languages (kernel_workflow)

`target_language` for author mode; also the languages the head-kernel bake-off can win with.

| Kernel language | Status | Notes |
|---|---|---|
| Triton | Supported | Always a viable author target. |
| FlyDSL | Supported | Preferred author target for dense / quantized GEMM (aiter's SOTA GEMM DSL, JIT, no build). Probed using `aiter.ops.flydsl.is_flydsl_available()`. |
| HIP | Supported | GEAK chooses HIP based on its e2e experience. |
| CK (Composable Kernel) | Supported | GEAK chooses CK based on its e2e experience; FP8 GEMM tuning. |

## Precision and data types

| Data type | Status | Notes |
|---|---|---|
| FP16 / BF16 | Supported | General kernel optimization. |
| FP8 | Supported | Head-GEMM tuning + author target; gsm8k accuracy gate recommended for quantized kernels. |
| FP4 / MXFP | Supported | Quantized GEMM (A4W4); accuracy gate recommended. |

## Accuracy gate (e2e_workflow)

| Gate | Requirement | Status |
|---|---|---|
| `none` (default) | Throughput delta + greedy output parity | Supported |
| `gsm8k` | Sampled gsm8k (5-shot, greedy, fixed seed) using `scripts/gsm8k_eval.py` against an OpenAI-compatible `/v1` endpoint | Supported |

## Notes

- The on-box GPU arch, backend, profiler, and op backends are re-checked at runtime by the Setup
  preflight, which writes `env_report.{md,json}`.
- Only tested configurations are listed. To report a verified configuration not listed here, open a pull request.
