# Sourcing rules — every claim must be grounded

perf_knowledge is only useful if its claims are trustworthy and current. Three hard rules:

## 1. Every file ends with `## Sources`
Cite **primary** sources, in priority order:
1. **AMD CDNA ISA reference guides** (cite the chapter/section, e.g. "CDNA4 ISA §7.2.1 MFMA with Block
   Exponent Scaling").
2. **AMD ROCm docs & ROCm/GPUOpen blogs** (full URL).
3. **GitHub source** — `org/repo@<commit-or-tag>:path/to/file` (commit pin matters; APIs/perf drift).
4. **arXiv / peer-reviewed papers** (id + title).
5. **Vendor / partner benchmarks** (clearly labeled as vendor-reported).
Secondary blogs are allowed only to *corroborate* a primary source, never as the sole basis.

## 2. Performance numbers are measured + version-tagged
Format: `value @ hardware, ROCm <ver>, <lib>@<commit/ver>, <date>`.
- Prefer **measured** numbers (median of ≥3 warm repeats; note spread). Mark vendor numbers as such.
- **Never** present theoretical peak as achievable. Context: MI300X commonly sustains only ~45% of
  peak FLOPs across FP8/BF16/FP16 (arXiv 2510.27583) — so SOTA cards quote *achieved*, not peak.
- Solution/kernel indices (hipBLASLt solidx, aiter tuned configs) are **ROCm/aiter-build-specific** —
  always note the build; never ship a hand-copied tuned table as portable.

## 3. Code-level corroboration when a library is on-box
When the backend is installed locally (e.g. `/sgl-workspace/aiter`, composable_kernel, hipBLASLt),
verify the claim against the **actual source** (entrypoint, dispatch path, tuning knob) and cite the
`repo@commit:path`. Cross-check the on-box version against the public repo. Use an `Explore`/
`general-purpose` subagent to read source excerpts; do not paste large dumps into the docs.

## Maintenance
- `reference/repo_index.md` lists every cited `repo@commit` once (single source of truth for pins).
- `reference/rocm_version_matrix.md` records which ROCm/lib versions each perf number was taken on.
- When a number is re-measured on a new stack, **append** (don't overwrite) with the new date so the
  trend is visible.

## Canonical primary-source set (seed; grow in repo_index.md)
- AMD CDNA3 ISA: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
- AMD CDNA4 ISA: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf
- AMD CDNA4 whitepaper: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf
- ROCm MI300X workload optimization: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- Matrix Core programming (CDNA3/4): https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
- Optimizing Triton kernels: https://rocm.docs.amd.com/en/docs-6.1.1/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
- CK-Tile FlashAttention: https://rocm.blogs.amd.com/software-tools-optimization/ck-tile-flash/README.html
- rocprof-compute (omniperf): https://rocm.docs.amd.com/projects/omniperf/en/amd-staging/what-is-rocprof-compute.html
- ROCm/aiter: https://github.com/ROCm/aiter
- MI300X ≈45% of peak (reality check): https://arxiv.org/pdf/2510.27583

## Sources
- This file is a policy doc; the list above is its own evidence base.
