---
title: Repo index — pinned sources
kind: reference
updated: 2026-06-08
---

# Repo index — pinned sources

Single place for the `repo@commit` / canonical-URL pins cited across perf_knowledge. Cards cite inline; this
consolidates the most-used ones. Grow as cards are added (P2–P4).

## On-box (verified locally; pin the installed version)
| repo | pin (on-box) | path used | notes |
|---|---|---|---|
| ROCm/aiter | `a6bb499375849eec45d68c5ccaebc8865fd422c0` (v0.1.12.post1-150) | `aiter/tuned_gemm.py`, `gradlib/`, `aiter/ops/flydsl/`, `aiter/configs/` | central kernel engine; dense-GEMM live path |
| flydsl (pip) | `0.1.5` | `aiter/ops/flydsl/*` | MLIR-Python DSL (FLIR→ROCDL) |
| sglang | 0.5.11 | serving stack | attention backend selection |

## Upstream repos
- ROCm/aiter — https://github.com/ROCm/aiter
- ROCm/rocm-libraries (Composable Kernel now lives here) — https://github.com/ROCm/rocm-libraries (projects/composablekernel)
- ROCm/composable_kernel (DEPRECATED mirror) — https://github.com/ROCm/composable_kernel
- ROCm/hipBLASLt — https://github.com/ROCm/hipBLASLt
- ROCm/mori — https://github.com/ROCm/mori
- ROCm/rocWMMA — https://github.com/ROCm/rocWMMA
- Dao-AILab/flash-attention — https://github.com/dao-ailab/flash-attention
- tile-ai/tilelang — https://github.com/tile-ai/tilelang
- HazyResearch/HipKittens — https://arxiv.org/html/2511.08083v1
- AMD-AGI/GEAK — https://github.com/AMD-AGI/GEAK (FlyDSL authoring docs ingested into `languages/flydsl/authoring_*` + `debugging.md` @ `c0a1f937` from `src/minisweagent/skills/flydsl/docs/`; re-sync on upstream change)
- sgl-project/sglang — https://github.com/sgl-project/sglang
- vllm-project/vllm — https://github.com/vllm-project/vllm
- deepseek-ai/DeepEP — https://github.com/deepseek-ai/DeepEP

## AMD primary docs (canonical)
- CDNA3 ISA — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
- CDNA4 ISA — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf
- CDNA4 whitepaper — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf
- MI300X workload optimization — https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- Matrix Core CDNA3/4 blog — https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
- rocprof-compute (omniperf) — https://rocm.docs.amd.com/projects/omniperf/en/amd-staging/what-is-rocprof-compute.html

## Sources
- Pins recorded from the on-box installs and the cards' inline citations (per sourcing_rules.md).
