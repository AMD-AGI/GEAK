# Term index — the word you would search for → the heading that answers it

Pointers only; the target wins if they disagree. Validated by
`python3 scripts/check_term_index.py --pack <pack>` (file and heading).

| you searched for | answered at |
| --- | --- |
| `fit`, `fitted curve`, `regression`, `residuals`, `extrapolate`, `leverage` | `../method/benchmark-hygiene.md ## Check the instrument before you read the scale off it` |
| `wave quantization`, `tail_efficiency`, `grid_tiles`, `last wave` | `roofline-models.md ## Saturation / wave quantization` |
| `tail efficiency` as an A/B explanation | `planning-constants.md ## Tail efficiency is one-sided too` |
| `occupancy gain`, `free up VGPR`, `will more waves help` | `planning-constants.md ## Occupancy is a one-sided criterion` |
| `waves/SIMD`, `wg/CU`, `workgroups per CU` | `planning-constants.md ## waves/SIMD is not workgroups/CU — convert before spending a round on it` |
| `next_free_vgpr`, `.vgpr_count`, `VGPR granule`, `waves per SIMD ladder` | `planning-constants.md ## VGPR / occupancy thresholds (CDNA combined budget)` |
| `; Occupancy:`, `LDSByteSize: 0`, `metadata.shared`, `group_segment_fixed_size` | `planning-constants.md ## The emitted ; Occupancy: N is a register-term answer` |
| `OutOfResources`, `Hardware limit: 163840`, `LDS budget`, `power of 2 shared` | `planning-constants.md ## Make an occupancy table falsifiable before you quote it` |
| `lds_per_cu_kib`, `LDS banks`, `160 KiB` | `planning-constants.md ## Targets (planning)` |
| `AGPR`, `v_accvgpr`, `accumulator in a[..]` | `planning-constants.md ## Extended planning (attention / fused kernels)` |
| `ridge`, `ops/byte`, `arithmetic intensity` | `roofline-models.md ## Roofline / GEMM` |
| `FETCH_SIZE`, `TCC_HIT` write requests, `--roof-only`, `rocprof-compute version` | `roofline-models.md ### rocprof-compute caveats that move a numerator or a denominator` |
| `MALL`, `TCC_EA0`, `DRAM bytes vs fabric` | `roofline-models.md ### These counters measure fabric traffic, not DRAM traffic` |
| `MFMA cycles`, `issue interval`, `PASS` | `isa-mechanisms.md ## MFMA throughput model (planning cycles)` |
| `co-issue`, `co-execution window`, `VALU under MFMA` | `isa-mechanisms.md ## Matrix/VALU co-execution — the CDNA-only overlap budget` |
| `v_exp_f32 rate`, `TRANS`, `v_permlane`, `v_pk_` | `isa-mechanisms.md ### What fits in the window, and at what price (gfx950, 32-cyc shape)` |
| `ds_write stall`, `LDS FIFO`, `ds_read_tr latency` | `isa-mechanisms.md ## LDS instruction behavior (gfx950)` |
| `no matching matrix core intrinsic`, `instr_shape`, `fp8 MFMA shape` | `capability-matrix.md ## Matrix operation matrix` |
| `128 or 32`, `direct-to-LDS width`, `async copy width` | `capability-matrix.md ## Direct-to-LDS granularity (per arch)` |
| `amdgpu-agpr-alloc`, `sched-strategy`, `coexec`, `warp_predicate` | `capability-matrix.md ## Memory and scheduling matrix` |
| `SQ_INSTS_VALU_MFMA_MOPS_F8` / `_F6F4`, scaled-MFMA counters | `bound-class-signals.md ## Rules (necessary-not-sufficient)` |
| `6.2.5`, `6.2.7`, `2.1.15`, `rocprof-compute block` | `bound-class-signals.md ## rocprof-compute block-ID cheat-sheet (when a rocprof-compute report is available)` |
| `VOP3 opcode`, `errata`, `DS_READ_TR pseudo-code` | `primary-sources.md ## Errata and trust levels` |
| `roofline.csv missing`, `Missing specs fields for gfx942` | `cdna3-gfx942.md ## Known tooling issues (gfx942)` |
| `which lever`, `what should I try` | `bound-class-signals.md`; availability per arch in `capability-matrix.md` |
