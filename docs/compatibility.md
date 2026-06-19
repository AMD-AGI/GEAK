# GEAK Compatibility Matrix

## Scope

This page lists verified hardware, software, runtime, and workflow combinations for GEAK.
Only verified and tested configurations are listed. Untested versions are intentionally omitted.

---

## 1. GEAK Release

| Release tag | Commit SHA | Release date | Status |
|---|---|---|---|
| `v3.2.1` | `c0a1f93` | 2026-06-15 | **Latest** |
| `v3.2.0` | `d9a80f7` | 2026-05-21 | Stable |
| `v3.1.0` | `1501039` | 2026-04-20 | Stable |
| `v3.0.0` | `bc2d6d5` | 2026-04-01 | Stable |
| `v2.0.0` | `8c58fe9` | 2026-01-13 | Legacy |
| `v1.0.0` | `536178b` | 2025-08-01 | Deprecated |

---

## 2. Host / Installation Mode

| Install mode | How | Status |
|---|---|---|
| Docker install | `AMD_LLM_API_KEY=<KEY> bash scripts/run-docker.sh` | Verified |
| Local install (make) | `make install` | Verified |
| Local full install | `make install-full` (core + dev + langchain + swe-rex) | Verified |
| Editable install (developer) | `make install-dev` or `pip install -e .` | Verified |
| Pip wheel / source | `pip install mini-swe-agent` | Verified |
| Docker editable | `scripts/run-docker.sh --editable` (mounts host repo) | Verified |

---

## 3. Operating System

| OS | Status |
|---|---|
| Ubuntu | Verified |

---

## 4. Python

| Python version | Status | Notes |
|---|---|---|
| 3.10 | Verified | Minimum required (`requires-python = ">=3.10"`) |
| 3.11 | Verified | Used in CI (pytest, lint, preprocess tests) |

---

## 5. GPU Hardware

| GPU model | Architecture | Status |
|---|---|---|
| MI300X | gfx942 (CDNA3) | Verified |
| MI308X | gfx942 (CDNA3) | Verified |
| MI355X | gfx950 (CDNA4) | Verified |
| RDNA4 | gfx1201 | Verified |

---

## 6. ROCm Stack

| Component | Version / Requirement | Status |
|---|---|---|
| ROCm | 7.2.x | Verified |
| ROCm | 7.1.x | Verified |
| ROCm | 7.0.x | Verified |
| ROCm | 6.4.x | Verified |

---

## 7. Kernel Languages

| Kernel language | Status |
|---|---|
| HIP | Verified |
| Triton | Verified |
| FlyDSL | Verified |
| PyTorch-to-FlyDSL translation | Verified |
| CK | Support FP8 GEMM tuning |

---

## 8. Frameworks / Target Workloads

| Framework / Workload | Status |
|---|---|
| SGLang | Verified |
| vLLM | Verified |

---

## 9. Precision / Data Types

| Data type | Status | Notes |
|---|---|---|
| FP32 | Verified | General kernel optimization |
| FP16 | Verified | General kernel optimization |
| BF16 | Verified | General kernel optimization |
| FP8 | Verified | General kernel optimization |
| FP4 | Verified | General kernel optimization |

---

## 10. Core Python Dependencies

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

- Only verified and tested configurations are listed. Untested versions are intentionally omitted.
- To report a verified configuration not listed here, please open a pull request.
