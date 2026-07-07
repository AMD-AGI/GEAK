---
myst:
    html_meta:
        "description": "Verified hardware, software, and runtime combinations for GEAK. Covers ROCm versions, AMD Instinct GPUs, Python versions, kernel languages, and core Python dependencies."
        "keywords": "GEAK, compatibility, ROCm, AMD Instinct, MI300X, MI355X, Python, Triton, HIP, FlyDSL"
---

# GEAK compatibility matrix

This topic lists the hardware, software, and runtime configurations that have been verified with GEAK. Use it to confirm that your GPU, ROCm version, Python version, and kernel language are supported before installing. Only verified and tested configurations are listed. Untested versions are intentionally omitted.

---

## GEAK release

These GEAK releases are tracked with their status.

| Release tag | Commit SHA | Release date | Status |
|---|---|---|---|
| `v3.2.2` | `9e14d4a` | 2026-06-23 | Latest |
| `v3.2.1` | `c0a1f93` | 2026-06-15 | Latest |
| `v3.2.0` | `d9a80f7` | 2026-05-21 | Stable |
| `v3.1.0` | `1501039` | 2026-04-20 | Stable |
| `v3.0.0` | `bc2d6d5` | 2026-04-01 | Stable |
| `v2.0.0` | `8c58fe9` | 2026-01-13 | Legacy |
| `v1.0.0` | `536178b` | 2025-08-01 | Deprecated |

---

## Host and installation mode

These installation modes are verified.

| Install mode | How | Status |
|---|---|---|
| Docker install | `AMD_LLM_API_KEY=<KEY> bash scripts/run-docker.sh` | Verified |
| Local install (make) | `make install` | Verified |
| Local full install | `make install-full` (core + dev + langchain + swe-rex) | Verified |
| Editable install (developer) | `make install-dev` or `pip install -e .` | Verified |
| Pip wheel / source | `pip install mini-swe-agent` | Verified |
| Docker editable | `scripts/run-docker.sh --editable` (mounts host repo) | Verified |

---

## Operating system

These operating systems are verified.

| OS | Status |
|---|---|
| Ubuntu 22.04.5 LTS | Verified |

---

## Python

These Python versions are verified.

| Python version | Status | Notes |
|---|---|---|
| 3.10 | Verified | Minimum required (`requires-python = ">=3.10"`) |
| 3.11 | Verified | Used in CI (pytest, lint, preprocess tests) |

---

## GPU hardware

These GPU models are verified.

| GPU model | Architecture | Status |
|---|---|---|
| MI300X | gfx942 (CDNA3) | Verified |
| MI308X | gfx942 (CDNA3) | Verified |
| MI355X | gfx950 (CDNA4) | Verified |
| RDNA4 | gfx1201 | Verified |

---

## ROCm stack

These ROCm versions are verified.

| Component | Version or requirement | Status |
|---|---|---|
| ROCm | 7.2.x | Verified |
| ROCm | 7.1.x | Verified |
| ROCm | 7.0.x | Verified |
| ROCm | 6.4.x | Verified |

---

## Kernel languages

These kernel languages are verified.

| Kernel language | Status |
|---|---|
| HIP | Verified |
| Triton | Verified |
| FlyDSL | Verified |
| PyTorch-to-FlyDSL translation | Verified |
| CK | Support FP8 GEMM tuning |

---

## Frameworks and target workloads

These frameworks and workloads are verified.

| Framework or workload | Status |
|---|---|
| SGLang | Verified |
| vLLM | Verified |

---

## Precision and data types

These precision formats are verified.

| Data type | Status | Notes |
|---|---|---|
| FP32 | Verified | General kernel optimization |
| FP16 | Verified | General kernel optimization |
| BF16 | Verified | General kernel optimization |
| FP8 | Verified | General kernel optimization |
| FP4 | Verified | General kernel optimization |

---

## Core Python dependencies

These Python packages are used, but some are optional.

| Package | Version constraint | Required | Notes |
|---|---|---|---|
| `litellm` | >= 1.75.5 | Core | LLM routing |
| `openai` | != 1.100.0, != 1.100.1 | Core | Excluded broken releases |
| `anthropic` | — | Core | |
| `google-genai` | — | Core | |
| `fastmcp` | >= 2.0.0 | Core | MCP tool server runtime |
| `mcp[cli]` | >= 1.2.0 | Core | MCP CLI |
| `metrix` | Pinned commit (`bcbfa02`) | Core | AMD GPU profiling (IntelliKit) |
| `langchain` | >= 0.3.0 | Optional (`[langchain]`) | RAG hybrid retrieval |
| `faiss-cpu` | >= 1.7.4 | Optional (`[langchain]`) | Vector similarity search |
| `sentence-transformers` | >= 2.2.0 | Optional (`[langchain]`) | Embedding models |
| `swe-rex` | >= 1.4.0 | Optional (`[full]`) | SWE-agent runtime |

Install extras:

```bash
pip install -e '.[langchain]'   # RAG support
pip install -e '.[full]'        # Everything (dev + langchain + swe-rex)
```

---

## Notes

- Only tested configurations are listed. Untested versions are intentionally omitted.
- To report a verified configuration not listed here, open a pull request.

## Related topics

- [Install GEAK](install/install.md)—installation instructions for verified configurations.
- [What is GEAK?](what-is-geak.md)—overview of GEAK's capabilities and supported kernel types.
- [Release notes](release-notes.md)—per-version changelog and feature history.
