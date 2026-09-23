---
title: Repo index — pinned sources
kind: reference
updated: 2026-09-21
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
- ROCm/aiter — https://github.com/ROCm/aiter (upstream ingest pins below are distinct from the
  on-box `a6bb4993` pin above; re-sync each snapshot when its source changes):
  - `b0ced008`: `.claude/skills/flydsl-kernel-code-cleanup/SKILL.md`, `requirements.txt`
    (`flydsl==0.3.2`) and `aiter/ops/flydsl/kernels/` →
    `languages/flydsl/authoring_api_migration.md`.
  - `04c7b808`: `.claude/skills/review-pr/{SKILL.md,rules.md}` → `workflows/review_kernels.md`
    plus amendments to `profiling/benchmarking_methodology.md`, `quantization/fnuz_vs_ocp.md`,
    `optimization/lds_and_bank_conflicts.md` and `languages/flydsl/authoring_api_migration.md`.
  - `04c7b808`: `.claude/skills/aiter-op-test/SKILL.md` → amendments only to
    `workflows/optimize_single_kernel.md`, `profiling/benchmarking_methodology.md`,
    `operators/layout_shuffle/overview.md` and `workflows/review_kernels.md`; the overlapping
    material already lives in `expert_skills/tuning/tuning-core/correctness_gates.md` and
    `expert_skills/tuning/benchmark/README.md`.
- ROCm/rocm-libraries (Composable Kernel now lives here) — https://github.com/ROCm/rocm-libraries (projects/composablekernel)
- ROCm/composable_kernel (DEPRECATED mirror) — https://github.com/ROCm/composable_kernel
- ROCm/hipBLASLt — https://github.com/ROCm/hipBLASLt
- ROCm/mori — https://github.com/ROCm/mori
- ROCm/rocWMMA — https://github.com/ROCm/rocWMMA
- Dao-AILab/flash-attention — https://github.com/dao-ailab/flash-attention
- tile-ai/tilelang — https://github.com/tile-ai/tilelang
- HazyResearch/HipKittens — https://arxiv.org/html/2511.08083v1
- AMD-AGI/GEAK — https://github.com/AMD-AGI/GEAK (ingested @ `c0a1f937` from `src/minisweagent/skills/flydsl/docs/` into **`languages/flydsl/authoring_tile_programming.md`, `authoring_optimization.md`, `authoring_gemm_levers.md`, `debugging.md`** only; re-sync on upstream change. The other `authoring_*` files come from different upstreams — see the two entries below and the 2026-08-12 changelog entry for `authoring_attention_levers.md`)
- ROCm/FlyDSL — https://github.com/ROCm/FlyDSL (the DSL itself; `docs/api_stability.md` + `.claude/skills/api-stability/` @ `da731e68` ingested into `languages/flydsl/api_stability.md`. The policy is versioned with the source — re-sync on upstream change)
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
