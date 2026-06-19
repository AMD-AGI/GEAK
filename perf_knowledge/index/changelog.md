# Changelog — perf_knowledge build progress

## P0 (2026-06-08) — scaffold
- Directory skeleton (index, 10–99).
- index: README, conventions, sourcing_rules, taxonomy, sota_matrix (seed), sota_registry.yaml
  (schema + seed), decision_trees, glossary, this changelog.
- Templates: _templates/sota_card_template.md, operator_overview_template.md.
- Sample operators (style references): dense_gemm (overview + backends: flydsl, triton, hipblaslt, aiter),
  attention_prefill_fmha (overview + backends: ck_tile).

## Planned
- P1: hardware (CDNA1–4) + languages (top-10) + backends (top-10).
- P2: operators Cartesian SOTA cards (the bulk, ~50 operators × ~10 backends).
- P3: optimization + quantization + profiling + workflows.
- P4: case_studies + generate/validate sota_registry.yaml from cards + wire into e2e_workflow.

## P1 (2026-06-08) — foundations done (~104 files)
- hardware/ (27): shared matrix-core/memory/numerics + CDNA1/2/3/4 deep (incl. CDNA4 block-scaled MFMA, MXFP, FP6@FP4 rate).
- languages/ (42): triton_amd, flydsl (on-box source), hip_cpp, composable_kernel(ck_tile), asm_mfma, tilelang, rocwmma, hipkittens, mojo, cutlass_port(na).
- backends/ (35): aiter(on-box pin a6bb4993), hipblaslt, composable_kernel_lib, rocblas_tunableop, flash_attention_rocm, mori_rccl, miopen, pytorch_inductor, sglang_kernels, vllm_kernels.
- reference/repo_index.md seeded; old-base links repointed (perf_knowledge self-contained); all files have frontmatter + ## Sources.

## P2 quant operators (2026-06-08) — quantization op family done (33 files)
- operators/quant_dequant_fp8 (9): overview/tuning/numerics/fusion + backends triton, hip, asm, aiter, vllm_kernels.
  e4m3/e5m2, per-tensor/per-token/per-block dynamic+static quant+dequant; FNUZ(CDNA3)↔OCP(CDNA4) split; 224 ROCm cap.
- operators/quant_int8 (8): overview/tuning/numerics/fusion + backends triton, hip, aiter, vllm_kernels.
  SmoothQuant W8A8 per-token×per-channel, symmetric+azp, INT32 accumulate; CDNA1/2 fallback.
- operators/quant_fp4_mxfp (8): overview/tuning/numerics/fusion + backends triton, hip, ck, aiter.
  MXFP4/MXFP6 32-elem E8M0 block scale; CDNA4-only HW (FP6@FP4 rate, scaled MFMA); gfx942 simulation/FP4BMM crash noted.
- operators/kv_cache_quant (8): overview/tuning/numerics/fusion + backends triton, hip, aiter, vllm_kernels.
  fp8/int8 paged KV store/load quant (slot_mapping), fused QK-norm+RoPE+write+quant; per-tensor/block scales.
- numerics.md is the depth focus per op; all grounded on on-box ROCm/aiter@a6bb49937 + vllm-project/vllm csrc + OCP MX spec.
- Cross-linked [[...]] to scaled_quant_gemm, fused_norm_quant, attention_decode_paged, paged_kv_copy, rope/mrope, hardware/cdna4_mi350.

## P2 (2026-06-08) — operator Cartesian cards (~420 files)
- 54 operators × {overview,tuning,numerics,fusion} + 204 backend SOTA cards.
- Families: GEMM(7), attention core(5)+advanced(5), norm/act/pos(9), MoE(4)+collectives(4), quant(4), embedding/sampling(3), elementwise/reduction(5), conv(3), data-movement(5).
- Every card sourced; heavy on-box corroboration (aiter@a6bb4993 MLA/paged-attn/gated-delta/sampling/quant/shuffle, vLLM csrc, mori).
- Normalized backend card filenames to canonical taxonomy ids (ck/fa_rocm/mori/rccl); 0 id↔filename mismatches.

## P3 (2026-06-08) — cross-cutting sections (41 files)
- optimization/ (12): occupancy, LDS/bank-conflicts, MFMA scheduling, pipelining, vectorization, XCD/L2, grid sizing, autotuning, roofline, fusion, numerics.
- quantization/ (10): formats, FNUZ-vs-OCP, MXFP block-scaling, scaling strategies, Quark calibration, accuracy gates, HW support matrix, KV quant, deployment recipes.
- profiling/ (10): rocprofv3/rocprof-compute/rocprof-sys, counters, bottleneck flow, roofline, traces, benchmarking methodology, engagement verification, pitfalls.
- kernel_workflow/ (9): single-kernel ladder, e2e model flow, GEMM tuning recipe, attention backend selection, kernel integration, GEAK authoring, backend choice, bring-up checklist.

## P4 (2026-06-08) — index + case studies + audit
- index/sota_matrix.md + sota_registry.yaml AUTO-GENERATED from card frontmatter (index/_gen_registry.py): 54 ops, 204 entries, YAML-valid.
- case_studies/ (by_model + by_kernel) from validated wins.
- Full-tree link audit: 0 broken relative links (excl. templates' placeholders).

## Landscape survey (2026-06-09)
- landscape/ (7 docs, ~2.1k lines, ~200 sources): 6 parallel research sweeps + synthesis README.
- multi_backend_libraries, authoring_dsls, ai_kernel_agents, autotuning_frameworks, amd_sota_2026, serving_stack_registries.
- Headline borrows: registry v3 (ranked list keyed by (op,gen,dtype,regime) + variant-qualified backends + dispatch seam + tuned_config + exclusions); reward-hacking-proof correctness gate; Triton→Gluon→TileLang/HipKittens authoring ladder; CDNA ping-pong scheduling prior; Gluon near-peak GEMM (hipBLASLt no longer the bar); add Gluon+HipKittens columns to matrix.

## SOTA refresh (2026-06-09) — Gluon + HipKittens + number refresh + CDNA prior
- Added **Gluon** (Triton low-level dialect): languages/gluon/ (overview, programming_model, gemm_cookbook) + backend cards dense_gemm/scaled_quant_gemm/quant_fp4_mxfp (FP16 1489@98.75%, BF8 3257@99.72%, MXFP4 5255@92.41% — AMD-measured MI355X). Added `gluon` to taxonomy.
- Added **HipKittens** (arXiv 2511.08083) backend cards: dense_gemm, scaled_quant_gemm, attention_prefill_fmha, gqa_mqa_attention, rmsnorm, rope (academic SOTA; honest losses kept e.g. AITER MHA bwd seq8192).
- Matrix generator: authoring langs (gluon/hipkittens/…) now column-grouped after core 6. Matrix+registry regenerated: 225 cards, 225 entries, YAML-valid; gluon/hipkittens columns live in GEMM/attention/norm families.
- **CDNA wave-scheduling prior** added to optimization/mfma_scheduling.md: wave-specialization underperforms on CDNA3/4 → 8-wave ping-pong / 4-wave interleave (HipKittens; adopted in AMD CDNA4 GEMM blogs).
- Number refresh (34 files): GEMM (hipBLASLt 2750/3130 bar beaten by 8-wave 3204), attention (concrete ROCM_AITER_FA 3.6-4.4x, MLA 1.33-1.52x + gfx942/gfx950 nuance), MoE (FlyDSL 1.6x, Kimi +162%, MoRI 2.56x BW), collectives (3-way adaptive + QuickReduce 3x), norm/quant (PTPC-FP8 2.5x), hardware (MLPerf v6.0 MI355X parity-to-win vs B300).

## FlyDSL authoring docs ingested (2026-06-17)
- Ingested the FlyDSL authoring knowledge from the GEAK FlyDSL **skill** (`AMD-AGI/GEAK@c0a1f937:src/minisweagent/skills/flydsl/docs/`) into perf_knowledge as **reference** (NOT as a skill; the skill's SKILL.md router was deliberately not ingested):
  - `languages/flydsl/authoring_tile_programming.md` — write a first correct `@flyc.kernel` (CuTe tile model, 4 patterns, MFMA ref).
  - `languages/flydsl/authoring_optimization.md` — structure-first optimization workflow (fusion→LDS→MFMA-loop→tuning).
  - `languages/flydsl/authoring_gemm_levers.md` — GEMM authoring levers (tiling/LDS/swizzle/epilogue).
  - `languages/flydsl/debugging.md` — correctness/NaN/zeros/mismatch/compile/hang triage (fills a prior gap).
- Complementary to the existing `languages/flydsl/` (which covers *using* aiter's built-in flydsl GEMM library). Frontmatter records `source_commit` for one-way re-sync; skill-orchestration phrasing neutralized; each file ends with `## Sources`. Cross-linked from overview.md deep-dive map + the dense_gemm flydsl card. sources_index regenerated (630 docs).
