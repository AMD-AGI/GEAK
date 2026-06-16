---
title: hipBLASLt TensileLite — kernel generation & the solution DB
kind: backend
backend: hipblaslt
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
status: competitive
updated: 2026-06-05
sources:
  - https://rocm.blogs.amd.com/artificial-intelligence/hipblaslt-tensilelite-tuning/README.html
  - https://github.com/ROCm/hipBLASLt
---

# hipBLASLt TensileLite

## TL;DR
TensileLite is the **kernel-generation layer under hipBLASLt** (a hipBLASLt-internal fork of Tensile). Its
kernels are the `Cijk_*` solutions; a per-arch **logic DB** maps each problem to ranked solutions. Use
TensileLite when **no pooled solution** is fast enough for a hot, fixed shape — you generate *new* kernels
and merge them into the library logic (heavier: a rebuild). For most workloads, offline tuning over the
existing pool ([offline_tuning.md](offline_tuning.md)) is enough; TensileLite is the escalation.

## Concepts

| Concept | Meaning |
|---|---|
| **Solution** | one generated kernel: tile (MT0×MT1), depth-U (K-tile), MFMA instr, global-split-K (GSU), workgroup, pipeline/scheduling. |
| **Solution index** | integer ID in the built library for an arch. **Not stable across ROCm versions/archs.** |
| **Logic YAML** | per-arch tuning DB: (transpose, dtype, M/N/K) → ranked solutions. gfx942 lives under `library/src/.../Tensile/Logic/asm_full/aquavanjaram/gfx942/`. |
| **Heuristic** | at runtime `AlgoGetHeuristic` consults the logic DB + size to rank candidates. |

### The "no tuned config" fallback (why TensileLite/offline tuning matter)
`hipblasLtMatmulAlgoGetHeuristic` looks up the gfx942 logic DB by (transA, transB, dtypes, M/N/K bucket).
If the **exact shape isn't in the DB**, it falls back to nearest/"Equality"/generic logic — often a
reasonable-but-not-optimal kernel, the classic "config not found" case that leaves **10–40%** on the table
for odd LLM shapes. Remedy: offline tuning (pick best pooled solution) or TensileLite (make a new one).

## Kernel-generation workflow
1. Dump the real shapes (Stage 1 of [offline_tuning.md](offline_tuning.md)).
2. Run `find_exact.py` / the TensileLite `Tensile` driver with a **problem YAML** + a **tuning logic YAML**
   describing the search space (tile sizes, MFMA, depthU, GSU, …). `AlgoMethod: "all"` is the fixed value.
3. Build gfx942-only logic for speed (< ~2h):
   ```bash
   ./install.sh -idc --logic-yaml-filter "gfx942/*/*" -a gfx942 -j 256 --build_dir build
   # -i install, -d deps, -c clients (gives hipblaslt-bench)
   ```
4. Merge the winning solutions into the library logic (e.g. `.../gfx942/Equality/`) and rebuild, or export
   to an override file for runtime.

## The levers (TensileLite search space)
- **Tile**: MT0×MT1 (workgroup macro-tile), DepthU (K per iteration).
- **MFMA**: instruction shape — prefer `mfma_16x16` on CDNA3/4.
- **GSU / global-split-K**: parallelize K for skinny/decode shapes.
- **Scheduling/pipeline**: prefetch depth, LDS/registers.
- **Workspace**: bound with `HIPBLASLT_TUNING_USER_MAX_WORKSPACE`.

## When to use vs offline tuning
| Situation | Use |
|---|---|
| Hot shape, a pooled solution is "good enough" | offline tuning (no rebuild) |
| Hot, fixed shape with **no** adequate pooled solution | TensileLite kernel-gen (rebuild) |
| Many shapes, want automation | QuickTune / Primus over the pool, escalate stragglers to TensileLite |

## Pitfalls
- Heavy: requires a hipBLASLt rebuild; only worth it for fixed, persistent shapes.
- Generated solution indices are version/arch-locked.
- gfx942-only logic filter is essential to keep build time sane.

## Cross-links
[offline_tuning.md](offline_tuning.md) · [api.md](api.md) · [env.md](env.md) · [when_wins.md](when_wins.md).

## Sources
- Customizing Kernels with hipBLASLt TensileLite GEMM Tuning (advanced guide): https://rocm.blogs.amd.com/artificial-intelligence/hipblaslt-tensilelite-tuning/README.html
- GEMM Tuning within hipBLASLt — Part 1 (find_exact.py / recompilation): https://rocm.blogs.amd.com/software-tools-optimization/hipblaslt-offline-tuning-part1/README.html
- hipBLASLt repo (`tensilelite/`, Logic YAML): https://github.com/ROCm/hipBLASLt
