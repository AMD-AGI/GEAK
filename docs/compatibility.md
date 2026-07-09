---
myst:
    html_meta:
        "description": "Supported hardware, software, and runtime configurations for GEAK v4: AMD Instinct GPUs, ROCm versions, Claude Code, serving backends, kernel languages, and data types."
        "keywords": "GEAK, compatibility, ROCm, AMD Instinct, MI300X, MI355X, sglang, vLLM, Triton, HIP, CK, FlyDSL, Claude Code"
---

# GEAK compatibility matrix

Hardware, software, and runtime configurations supported by GEAK v4.

## GPU hardware

| GPU model | Architecture |
|---|---|
| MI300X / MI308X | gfx942 (CDNA3) |
| MI355X | gfx950 (CDNA4) |

The on-box card is auto-detected.

## ROCm stack

| Component | Requirement |
|---|---|
| ROCm | 6+ (user-space with `rocminfo` / `rocm-smi`) |
| Profiler | one of `rocprof-compute`, `rocprofv3`, `rocprof` |
| HIP compiler | `hipcc` (for HIP C++ kernels) |

## Runtime

| Component | Requirement |
|---|---|
| Claude Code | ≥ 2.1.177 (dynamic Workflow feature) |
| Python | 3.8+ |

## Serving backends (e2e_workflow)

| Backend | Status |
|---|---|
| SGLang | Supported |
| vLLM | Supported |

Backends are pluggable via `e2e_workflow/scripts/adapters/<backend>.sh`.

## Kernel languages (kernel_workflow)

| Language | Status |
|---|---|
| Triton | Supported |
| HIP | Supported |
| CK | Supported |
| FlyDSL | Supported |

## Data types

| Data type | Status |
|---|---|
| FP32 / FP16 / BF16 | Supported |
| FP8 | Supported |
| FP4 | Supported |

## Related topics

- [Install GEAK](install/install.md) — prerequisites and setup.
- [What is GEAK?](what-is-geak.md) — overview and supported workflows.
