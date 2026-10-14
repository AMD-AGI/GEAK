# GEMM development decision cards

This is the actionable layer. It does not dump regex matches and ask the reader to infer a recommendation. Every curated card states **when it applies, what to try, why, alternatives, evidence strength and limits**. The raw, reproducible source observations remain in [`gemm_source_evidence.md`](gemm_source_evidence.md).

Source baseline: AITER `a6bb499375849eec45d68c5ccaebc8865fd422c0` · 30 curated decision card(s) citing 271 of 1630 source-evidence records · 12 FlyDSL shipped-seed groups · 41 backend-selection groups · 306 Triton seed groups (on [`gemm_triton_seeds.md`](gemm_triton_seeds.md)).

How to read this page:

- Steps follow the order a kernel is written in, from its math contract to validation. A card sits under the step of its first axis — the question it answers — and every other step its axes touch links back to it: choosing split-K in step 3 is listed again in step 8, where the partials are combined, and step 9, where the combine's host state is written.
- Within a step, constraint cards come first. Read every constraint card before costing a candidate: architecture gates are in step 2 and build legality in step 3.
- A card's *Same question in other implementations* block shows how Triton, Gluon, CK or ASM answer the same question in AITER, with their own source lines. Transfer the intent, not the spelling: use the target language's docs to write it.
- Evidence is `file:line`, or `file:line-end` when the cited construct spans lines; open the range to read the whole mechanism.
- The two shipped tables after the cards are what AITER's tuner selected — FlyDSL configurations per bucket, and which backend it chose per bucket. They seed a search; they never end one.

Evidence levels:

- `source_observed`: implementation precedent only; add a candidate and measure it.
- `shipped_config` / shipped selection: parameter seeds and backend choices recorded in AITER's shipped tables; alternatives and benchmark results are not attached, so vary and measure locally.
- Background references point at always-on docs under `perf_knowledge/` (hardware facts, how-to), never at measurements.
- Measured guidance is deliberately not copied here: it stays in learned cards/expert skills behind their existing feature switches.

## Design traits

A card whose match `requires` a trait is **deferred** by `_select_candidates.py` until you pass `--trait <name>`. A trait describes your design, which the selector cannot see, so deferred means "applies once your kernel does this", not "does not apply".

- `lds_staging` — The design stages at least one operand tile through LDS: it is stored to LDS and the MFMA fragments are read back from it.
- `reused_weight` — B, the weight, is reused across calls, so a one-time host-side layout transform such as a preshuffle is paid once per weight rather than per call.
- `split_k_hot_loop` — The K loop is hand-scheduled with ROCm scheduling directives over LDS-staged operands, as the FlyDSL split-K HGEMM hot loop is.

## Coverage by architecture

What this corpus holds for each target, so a plan knows how much it is standing on. A card counts for a gfx when its match does not exclude it. *Arch-specific source* counts gates that name the gfx (or a prefix of it, such as `gfx95`) plus records whose code sits under such a gate. ASM kernels are the rows of AITER's per-arch kernel inventories (`hsa/<gfx>/<kind>/*.csv`). Counts, not rankings.

| gfx | cards that can apply | arch-specific source records | ASM kernels by kind | FlyDSL seed groups | backend-selection groups | Triton seed groups |
|---|---|---|---|---|---|---|
| `gfx950` | 29 of 30 | 6 | bf16gemm 24, f4gemm 35, fp8gemm_blockscale 6, i8gemm 8 | 12 | 24 | 176 |
| `gfx942` | 29 of 30 | 19 | bf16gemm 22, fp8gemm_blockscale 6, i8gemm 9 | 0 | 17 | 47 |
| `gfx1250` | 29 of 30 | 0 | — | 0 | 0 | 83 |

## Development order

| step | axes | cards that answer it | cards from other steps that bear on it |
|---|---|---|---|
| 1. Interface and math semantics | — | — | — |
| 2. Architecture capability and implementation path | `architecture_capability`, `implementation_variant` | `flydsl-gfx942-path-gates` (constraint), `flydsl-small-m-hgemm-family` (performance_candidate), `gemm-shape-regime-routing` (performance_candidate), `gemm-default-backend-routing` (performance_candidate) | `flydsl-a8-legality`, `flydsl-half-mfma-call-forms`, `flydsl-a8-mfma-selection`, `flydsl-async-copy-by-family`, `gemm-blockscale-shipped-bucket-structure`, `flydsl-preshuffle-b-layout-contract`, `gemm-skinny-decode-wave-split`, `flydsl-host-launch-contract` |
| 3. Grid, workgroup tile, waves and split-K | `workgroup_tile`, `work_partition`, `execution_geometry`, `configuration_constraints` | `flydsl-hgemm-tiling-validity` (constraint), `flydsl-a8-legality` (constraint), `gemm-workgroup-xcd-mapping` (performance_candidate), `gemm-split-k-when-grid-underfills` (performance_candidate), `flydsl-hgemm-config-space` (performance_candidate), `flydsl-a8-preshuffle-config-space` (performance_candidate), `gemm-blockscale-shipped-bucket-structure` (performance_candidate), `gemm-asm-tile-inventory-and-selection` (performance_candidate), `gemm-skinny-decode-wave-split` (performance_candidate) | `flydsl-a8-mfma-selection`, `gemm-blockscale-k-loop-scale-application`, `flydsl-cshuffle-epilogue`, `flydsl-xor16-lds-layout-consistency`, `flydsl-preshuffle-b-layout-contract`, `flydsl-splitk-hot-loop-schedule-bundle`, `flydsl-a8-interleaved-schedule`, `flydsl-splitk-reduction-contract`, `flydsl-small-m-hgemm-family`, `gemm-shape-regime-routing`, `flydsl-host-launch-contract`, `flydsl-test-parity-gate`, `flydsl-lds-pipeline-depth` |
| 4. Layout and address computation | `operand_layout` | `flydsl-mfma-fragment-lane-layout` (semantic), `flydsl-xor16-lds-layout-consistency` (performance_candidate), `flydsl-preshuffle-b-layout-contract` (performance_candidate) | `flydsl-a8-data-movement-helpers` |
| 5. Global, LDS and register data movement | `memory_access` | `flydsl-async-copy-by-family` (performance_candidate), `flydsl-a8-data-movement-helpers` (performance_candidate), `gemm-streamed-operand-cache-policy` (performance_candidate) | `flydsl-gfx942-path-gates`, `gemm-workgroup-xcd-mapping`, `flydsl-splitk-hot-loop-schedule-bundle`, `flydsl-a8-interleaved-schedule`, `gemm-skinny-decode-wave-split` |
| 6. MFMA / dot compute | `compute_instruction` | `flydsl-a8-scale-granularity-contract` (constraint), `flydsl-a8-mfma-selection` (semantic), `gemm-blockscale-k-loop-scale-application` (semantic), `flydsl-half-mfma-call-forms` (performance_candidate) | `flydsl-gfx942-path-gates`, `flydsl-splitk-hot-loop-schedule-bundle`, `flydsl-a8-interleaved-schedule`, `flydsl-mfma-fragment-lane-layout` |
| 7. Pipeline and synchronization | `pipeline_schedule` | `flydsl-splitk-hot-loop-schedule-bundle` (performance_candidate), `flydsl-a8-interleaved-schedule` (performance_candidate), `flydsl-lds-pipeline-depth` (performance_candidate) | `flydsl-hgemm-tiling-validity`, `flydsl-async-copy-by-family`, `flydsl-a8-preshuffle-config-space`, `flydsl-a8-data-movement-helpers` |
| 8. Reduction, epilogue and write-back | `epilogue` | `flydsl-splitk-reduction-contract` (constraint), `flydsl-cshuffle-epilogue` (performance_candidate) | `flydsl-a8-scale-granularity-contract`, `gemm-split-k-when-grid-underfills`, `flydsl-a8-preshuffle-config-space`, `flydsl-mfma-fragment-lane-layout`, `flydsl-test-parity-gate` |
| 9. Host configuration, ABI and launch | `runtime_contract` | `flydsl-host-launch-contract` (constraint) | `flydsl-a8-scale-granularity-contract`, `gemm-blockscale-k-loop-scale-application`, `gemm-split-k-when-grid-underfills`, `flydsl-preshuffle-b-layout-contract`, `flydsl-splitk-reduction-contract`, `gemm-default-backend-routing` |
| 10. Correctness validation and tuning | `validation_contract`, `tunable_surface` | `flydsl-test-parity-gate` (constraint) | `flydsl-hgemm-config-space`, `flydsl-a8-preshuffle-config-space`, `gemm-blockscale-shipped-bucket-structure`, `gemm-streamed-operand-cache-policy` |

## 1. Interface and math semantics

Fixed by the task rather than chosen: `math_contract` and `dtype` in `comparison_context` above, read from the task's meta.json, with the operator's numerics.md for parity. Its consequences are placed where they are implemented — scale application in step 6, the argument and layout contract in step 9.

No card answers this step yet.

## 2. Architecture capability and implementation path

- `architecture_capability` — Which implementation paths and instruction/data formats exist on the target architecture?
- `implementation_variant` — Which kernel family or opaque backend implementation is selected for this scenario?

Also bears on this step: `flydsl-a8-legality` (step 3), `flydsl-half-mfma-call-forms` (step 6), `flydsl-a8-mfma-selection` (step 6), `flydsl-async-copy-by-family` (step 5), `gemm-blockscale-shipped-bucket-structure` (step 3), `flydsl-preshuffle-b-layout-contract` (step 4), `gemm-skinny-decode-wave-split` (step 3), `flydsl-host-launch-contract` (step 9).

### Which FlyDSL GEMM paths does gfx942 (MI300X) actually give you, and which does it silently switch or refuse?

**Card:** `flydsl-gfx942-path-gates` · **type:** `constraint` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `architecture_capability`, `compute_instruction`, `memory_access`, `implementation_variant`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `scaled_quant_gemm`, `grouped_gemm_moe`, `batched_gemm`, `splitk_streamk_gemm`, `skinny_gemv_decode`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`
- `gfx`: `gfx942`

#### Use when

- operator_family=gemm
- target_language=flydsl
- target architecture is gfx942 — read this before costing any other card on this page

#### Try

- Read the architecture gates before the tuning tables. Whether a path exists on your box outranks every fact about how to tune it, and several gates in this library key on gfx942 alone.
- Do not fund async copy for the split-K HGEMM family on gfx942. That family computes its async-copy flag as `architecture is not gfx942`, rejects any request that disagrees, and its kernel hard-codes async off on this branch — there is no knob to flip. This does NOT carry over to the A8 (fp8/int8) preshuffle family, which keeps `use_async_copy` as a real knob on gfx942 with a 4-byte A copy instead of 16 — see flydsl-async-copy-by-family.
- Do not fund the small-M kernel family on gfx942. Its compile entry point raises, and its config registry returns nothing. Port the design if you want it; you cannot call it.
- Expect the HGEMM to use the m16n16k16 fragment with 4-byte DMA and two MFMA steps per warp-K, not the m16n16k32 / 16-byte / one-step bundle the other architectures get. The choice is made for you by one architecture test — see flydsl-half-mfma-call-forms.
- Expect the A8 family on gfx942 to take its hand-written gfx942 schedule branch, K=32 fp8 MFMAs instead of the K=128 scaled MFMA, and tile_k=64 tiles that the gfx950 catalog omits — see flydsl-a8-mfma-selection and flydsl-a8-interleaved-schedule.
- Expect no shipped FlyDSL seed: every gfx942 fp8 row in AITER's tuned databases went to CK or ASM. Budget a FlyDSL kernel on this box as authoring work, not as a lookup — see gemm-shape-regime-routing.
- On gfx950 these gates open: the wider fragment, the 16-byte DMA, async copy and the small-M family are all in play. Other architectures are not excluded by these gates, but AITER ships no FlyDSL configuration for them.

#### Why this is a candidate

- Six gfx942 tests exist in the FlyDSL GEMM sources and they do different things: one switches the HGEMM fragment/DMA/MFMA bundle, one fixes HGEMM async copy library-wide, two shut the small-M family, one narrows the A8 async A-copy width and one selects the A8 gfx942 schedule.
- This is the cheapest knowledge on the page. A gate costs nothing to check and a whole round to discover, and none of it is visible from the tuning tables, which record what was selected on the architectures where the path was open.

#### Keep as alternatives

- On gfx942, spend the search on what remains genuinely open — the tile geometry, split_k, B staging and the epilogue — rather than on the gated paths.
- If a gated path is the real ceiling for your shape, vendor and port it deliberately as a structural change, and budget it as one.

#### Evidence

- `src_c4c19187bcb7933c` — `arch_gate` `gfx942` at `aiter/ops/flydsl/gemm_kernels.py:41`
- `src_a0f1066a6a2d6667` — `config_validity` `Current kernel fixes async_copy from the active GPU architecture;` at `aiter/ops/flydsl/gemm_kernels.py:123-126`
- `src_e689c0cbe1b4f8e8` — `arch_gate` `gfx942` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:128-137`
- `src_7eee3f6769348b4c` — `arch_gate` `gfx942` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:411-412`
- `src_e04716b6d760605b` — `arch_gate` `gfx942` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:331-332`
- `src_42146c2d073d655e` — `arch_gate` `gfx942` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:243`
- `src_fbe3baca81526883` — `arch_gate` `gfx942` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:255`
- `sel_5aecc3631bf688e9` — backend selected by AITER's tuner in `a8w8_bpreshuffle_tuned_gemm` on `gfx942`/80 CUs, `M<=16`: ck 204 (rows)
- `sel_438e88e4bc86460c` — backend selected by AITER's tuner in `a8w8_bpreshuffle_tuned_gemm` on `gfx942`/80 CUs, `M>4096`: ck 391 · asm 4 (rows)
- `sel_f1384cc1069c9823` — backend selected by AITER's tuner in `a8w8_blockscale_bpreshuffle_tuned_gemm` on `gfx942`/80 CUs, `M>4096`: asm 32 · ck 6 (rows)

#### Background (always-on reference docs, not measurements)

- [`hardware/cdna3_mi300/memory_hierarchy.md`](../hardware/cdna3_mi300/memory_hierarchy.md)
- [`languages/flydsl/knobs.md:71-75`](../languages/flydsl/knobs.md)
- [`languages/flydsl/kernel_families.md:19-32`](../languages/flydsl/kernel_families.md)

#### Limits

- These are the gates that name an architecture literally. A path can also be effectively closed by something this card does not see — a missing dtype, an LDS budget, a compiler version.
- Gates are per family: a gate in one family's source says nothing about another family's, which is exactly how the async-copy statement used to be over-read.
- Gates move between releases. Every citation carries its file and line; re-read them against the checkout you are building on before trusting the list.
- That a path is open says nothing about whether it is fast.

### What does AITER do differently when M is a handful of rows, and which knobs does that open?

**Card:** `flydsl-small-m-hgemm-family` · **type:** `performance_candidate` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `implementation_variant`, `architecture_capability`, `workgroup_tile`, `work_partition`, `execution_geometry`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `skinny_gemv_decode`
- `target_languages`: `flydsl`
- `exclude_gfx`: `gfx942`
- `dtypes`: `bf16`
- `shape`: `m_max=16`

#### Use when

- operator_family=gemm
- target_language=flydsl
- M is small — the shipped routing threshold is M < 17
- dtype is bf16

#### Try

- Check the architecture first. gfx942 is excluded by the source: the compile entry point raises and the config registry returns an empty set before it looks at the shape. gfx950 is where the registry is exercised; other architectures are not excluded by the gate but have no shipped configuration. On gfx942 read this card as a porting reference.
- Treat small M as a different kernel family rather than the generic kernel with a smaller tile: AITER routes it to a separate implementation selected by an explicit kernel_family value.
- In that family tile_m is fixed at 16 with block_m_warps=1 and 2 stages — the M dimension stops being a search axis, and the search moves entirely onto N and K.
- Search tile_n far wider than the generic family allows — the shipped set runs (32, 64, 96, 128, 160, 192, 224, 256, 384, 512, 768, 1024) — with tile_k in (32, 64, 96, 128, 160, 192, 256) and split_k up to 32.
- Consider the axes this family adds and the generic one does not: n_tile_repeat, persistent_n_tiles, waves_per_eu and a b_to_lds unroll factor.
- Respect the combination rules: n_tile_repeat > 1 only without B-in-LDS and only for the two shipped shapes; persistent_n_tiles > 1 only with B-in-LDS, tile_n >= 128, block_n_warps >= 2 and no more persistent tiles than N/tile_n.
- Give the generic family a real arm, not a token one: all 86 FlyDSL rows AITER ships for gfx950 bf16 at M<=16 are generic-family kernels (mostly tile_m=32 with split_k 4-16), none from this family. The database does not say whether this family was in that race.

#### Why this is a candidate

- A skinny-M GEMM is bounded by how much of the machine the few M tiles can occupy, and every axis this family adds — repeat, persistent tiles, waves per EU — is an occupancy axis rather than a tiling one.
- The routing threshold, the fixed tile_m and the option sets are stated as named constants in the source, so the boundary between the two families is readable rather than folklore.

#### Keep as alternatives

- Stay on the generic family with a small tile_m and split_k > 1 when a second kernel cannot be written or maintained; that is what AITER shipped for these buckets and is the cheaper first move.
- Port the HIP skinny kernels' structure instead — rows spread over CUs, K split across the lanes of a wave, no combine (gemm-skinny-decode-wave-split). That is what AITER's default route runs for an untuned bf16 decode case, on both architectures.
- Keep the generic default configuration as the control arm regardless of which family is chosen.

#### Evidence

- `src_38699a8ce57faf0c` — `kernel_family` `KERNEL_FAMILY_HGEMM, hgemm` at `aiter/ops/flydsl/kernels/hgemm_dispatch.py:8`
- `src_8ffea30406cf0355` — `kernel_family` `KERNEL_FAMILY_SMALL_M, small_m` at `aiter/ops/flydsl/kernels/hgemm_dispatch.py:9`
- `src_bca963564a69909c` — `kernel_family` `KERNEL_FAMILY_SMALL_M` at `aiter/ops/flydsl/kernels/hgemm_dispatch.py:56-71`
- `src_b446c07f43b11542` — `config_limit` `SMALL_M_KERNEL_MAX, 17` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:69`
- `src_034b3cc6e3fd6aa8` — `tile_shape` `TILE_M, 16` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:70`
- `src_390ae8578e28ce34` — `config_space` `SMALL_M_TILE_N_OPTIONS` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:83-96`
- `src_60717171485c58f5` — `config_space` `SMALL_M_TILE_K_OPTIONS` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:81`
- `src_ee5e839c6374f38f` — `config_limit` `SMALL_M_MAX_SPLIT_K, 32` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:82`
- `src_a7c0de9870934fc0` — `config_space` `SMALL_M_N_TILE_REPEAT_OPTIONS` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:102`
- `src_28936b493eeb6f6f` — `config_space` `SMALL_M_PERSISTENT_N_TILE_OPTIONS` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:103`
- `src_1dcdf9c7b6794c6e` — `config_space` `SMALL_M_NON_B_TO_LDS_WAVES_PER_EU_OPTIONS` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:97`
- `src_44b5602aacc7e7df` — `config_space` `SMALL_M_B_TO_LDS_UNROLL_OPTIONS` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:101`
- `src_cceaeb92fb86b540` — `config_validity` `_validate_small_m_registry_config` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:172-228`
- `src_bc6d04baf2182ba4` — `config_limit` `MAX_LDS_BYTES, 163840` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:76`
- `src_7eee3f6769348b4c` — `arch_gate` `gfx942` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:411-412`
- `src_e04716b6d760605b` — `arch_gate` `gfx942` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:331-332`
- `cfg_973fe63f897809df` — FlyDSL `hgemm` `abf16_wbf16_bf16` selections in `bf16_tuned_gemm` on `gfx950`/256 CUs, `M<=16`: 86 shipped row(s)

#### Background (always-on reference docs, not measurements)

- [`languages/flydsl/knobs.md:77-80`](../languages/flydsl/knobs.md)
- [`languages/flydsl/patterns.md:78-82`](../languages/flydsl/patterns.md)
- [`operators/skinny_gemv_decode/backends/flydsl.md:19-28`](../operators/skinny_gemv_decode/backends/flydsl.md)

#### Limits

- On gfx942 none of this is callable; see the first action. The rest of the card describes the gfx950 path.
- That AITER maintains a separate family for small M is a design precedent, not a measurement that it beats a well-tuned generic configuration on your shape.
- The option sets are what the shipped registry enumerates; they are candidates to search, and nothing here says which member wins.
- The family is bf16-only and its LDS budget is its own; do not carry its constants into the generic kernel.

### For each shape bucket, which backend did AITER's own tuner ship, and what does that imply for a FlyDSL kernel?

**Card:** `gemm-shape-regime-routing` · **type:** `performance_candidate` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `implementation_variant`, `workgroup_tile`, `work_partition`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `scaled_quant_gemm`, `grouped_gemm_moe`, `batched_gemm`, `splitk_streamk_gemm`, `skinny_gemv_decode`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`

#### Use when

- target_language=flydsl
- a workload with one or more (M, N, K) cases, with gfx and dtype known

#### Try

- Look up each case's bucket in the backend-selection table below before choosing directions. It says which implementation AITER shipped for comparable shapes on that gfx and CU count — what a FlyDSL kernel competes with in production, even when the task's frozen baseline is Triton.
- gfx950 bf16: FlyDSL was selected for 86 of 165 rows at M<=16 and 52 of 174 at M 65-256, but for only 7 of 186 rows above M=1024, where ASM took 164. Large-M bf16 is where FlyDSL has no shipped precedent: plan structural work (wave schedule, tile, workgroup mapping), not a config lookup.
- gfx950 fp8 per-token with preshuffled B: FlyDSL was selected in every bucket, most often at the extremes (72 of 111 rows at M<=16, 77 of 133 above M=4096) — seed from the FlyDSL A8 groups.
- fp8 128x128 block scale on either architecture, and every gfx942 fp8 database: no FlyDSL row exists — CK, CK-Tile and ASM own them. A FlyDSL kernel there is authored from the A8 precedents plus a block-scale K loop, with no shipped FlyDSL seed (see flydsl-a8-scale-granularity-contract and gemm-blockscale-k-loop-scale-application); which configuration the Triton baseline actually runs per case — often one fallback tile for every M — is gemm-blockscale-shipped-bucket-structure.
- Route small-M bf16 on gfx950 through both the generic HGEMM (what AITER shipped) and the small-M family (flydsl-small-m-hgemm-family); route quantized operands to the A8 family.
- Treat buckets as separate problems: when a workload spans buckets (decode plus prefill), expect different winners per case and keep a per-case dispatch rather than one global configuration.
- For a case no database row covers, read AITER's default route instead (gemm-default-backend-routing): the HIP skinny kernels for small-M bf16/fp16, hipblaslt (gfx942) or ASM (elsewhere) for B-preshuffled bf16, CK for fp8 — that is what a FlyDSL kernel competes with there.

#### Why this is a candidate

- The tuned databases are the only place AITER records which implementation it chose for which shape regime.
- A frozen Triton baseline says nothing about where FlyDSL already wins in production and where it never has; this table does.

#### Keep as alternatives

- Ignore the shipped selection and search blind — the control arm, and the only option for a bucket the database does not cover.

#### Evidence

- `src_38699a8ce57faf0c` — `kernel_family` `KERNEL_FAMILY_HGEMM, hgemm` at `aiter/ops/flydsl/kernels/hgemm_dispatch.py:8`
- `src_bca963564a69909c` — `kernel_family` `KERNEL_FAMILY_SMALL_M` at `aiter/ops/flydsl/kernels/hgemm_dispatch.py:56-71`
- `src_b446c07f43b11542` — `config_limit` `SMALL_M_KERNEL_MAX, 17` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:69`
- `src_f854f40de1f28585` — `scale_operand` `_needs_per_token_scale` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:403`
- `sel_81411b8982b03045` — backend selected by AITER's tuner in `bf16_tuned_gemm` on `gfx950`/256 CUs, `M<=16`: flydsl 86 · asm 45 · triton 34 (rows)
- `sel_4ff2330404509760` — backend selected by AITER's tuner in `bf16_tuned_gemm` on `gfx950`/256 CUs, `M65-256`: asm 88 · flydsl 52 · triton 34 (rows)
- `sel_0bf8dd0fd028cea8` — backend selected by AITER's tuner in `bf16_tuned_gemm` on `gfx950`/256 CUs, `M1025-4096`: asm 62 · triton 6 · flydsl 4 (rows)
- `sel_c472ec443f7b30a0` — backend selected by AITER's tuner in `bf16_tuned_gemm` on `gfx950`/256 CUs, `M>4096`: asm 102 · triton 9 · flydsl 3 (rows)
- `sel_7c8d8e410633ad1e` — backend selected by AITER's tuner in `a8w8_bpreshuffle_tuned_gemm` on `gfx950`/256 CUs, `M<=16`: flydsl 72 · ck 37 · cktile 2 (rows)
- `sel_1b5489e6d41507a3` — backend selected by AITER's tuner in `a8w8_bpreshuffle_tuned_gemm` on `gfx950`/256 CUs, `M>4096`: flydsl 77 · cktile 36 · ck 20 (rows)
- `sel_8b543c4558078855` — backend selected by AITER's tuner in `a8w8_blockscale_tuned_gemm` on `gfx950`/256 CUs, `M>4096`: ck 178 · cktile 147 (rows)
- `sel_438e88e4bc86460c` — backend selected by AITER's tuner in `a8w8_bpreshuffle_tuned_gemm` on `gfx942`/80 CUs, `M>4096`: ck 391 · asm 4 (rows)

#### Background (always-on reference docs, not measurements)

- [`operators/dense_gemm/tuning.md`](../operators/dense_gemm/tuning.md)
- [`operators/splitk_streamk_gemm/tuning.md:16-19`](../operators/splitk_streamk_gemm/tuning.md)
- [`languages/flydsl/patterns.md:41-53`](../languages/flydsl/patterns.md)

#### Limits

- Counts are rows, not unique shapes: a shape shipped in two model databases counts twice.
- The tuner's box, competing set and toolchain are not recorded; a selection says where AITER chose a backend, never by how much.
- The table reflects AITER at the pinned commit; re-extract after an AITER bump.

### Which backend does AITER's own dispatch pick when no tuned row exists, and how does that differ between gfx942 and gfx950?

**Card:** `gemm-default-backend-routing` · **type:** `performance_candidate` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `implementation_variant`, `runtime_contract`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `scaled_quant_gemm`, `skinny_gemv_decode`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`

#### Use when

- target_language=flydsl
- a case whose production path, or whose frozen baseline, is AITER's dispatch rather than a fixed kernel

#### Try

- bf16/fp16 (`tuned_gemm.get_GEMM_A16W16_config`): a tuned row wins first, and a `flydsl` row is served only when `is_flydsl_available()` and its kernel name parses — otherwise the row is dropped. With no row: gfx12 goes to torch; a B-preshuffled request goes to hipblaslt on gfx942 and, elsewhere, to ASM when the input is bf16, the output bf16 or fp32, and N and K are multiples of 64 (it asserts otherwise); an unshuffled request goes to the HIP skinny kernels when `is_skinny_default_shape` holds, else to torch.
- The skinny route owns bf16/fp16 decode by default: K % 8 == 0 and M = 1 with N <= 2*CU and K <= 9216, M <= 4 with N <= CU and K <= 9216, M <= 8 with N <= CU and K <= 5120, or M <= 16 with N <= CU and K <= 256 (gemm-skinny-decode-wave-split).
- fp8 128x128 block scale (`gemm_op_a8w8.gemm_a8w8_blockscale`): a B-preshuffled call goes to ASM only on gfx950 with m >= 16, k >= 512 and bf16 output, and asserts otherwise; an unshuffled call goes to CK or CK-Tile by tuned row, CK by default. The B-preshuffled block-scale entry (`gemm_a8w8_blockscale_bpreshuffle`) picks CK-Tile, CK or ASM by tuned row — that is where gfx942 reaches its ASM block-scale kernels.
- fp8 per-token with preshuffled B (`gemm_a8w8_bpreshuffle`, fp8 weights only at the pinned commit): CK, CK-Tile or FlyDSL by tuned row (`gemm_a8w8_bpreshuffle_flydsl`, only when FlyDSL is available), CK with no row.
- Read these as what a FlyDSL kernel competes with for a case AITER never tuned; where it did tune, the backend-selection table (gemm-shape-regime-routing) says what it picked.

#### Why this is a candidate

- The default route is code, not a table: it is what production runs for every shape the tuned databases do not cover, and it differs by architecture.
- It names FlyDSL as a routed backend for bf16 and for per-token fp8, and never for block scale.

#### Keep as alternatives

- Ignore AITER's dispatch and compare only against the task's frozen baseline — the control arm.

#### Evidence

- `src_2577a978dcbebafb` — `kernel_family` `is_skinny_default_shape` at `aiter/tuned_gemm.py:76-97`
- `src_c9b676e9661f7f1e` — `kernel_family` `flydsl` at `aiter/tuned_gemm.py:133-141`
- `src_57c3a751ec4ad78b` — `kernel_family` `asm` at `aiter/tuned_gemm.py:172`
- `src_07dceb286dc43c55` — `kernel_family` `skinny` at `aiter/tuned_gemm.py:182`
- `src_44e19b6d045bd0ec` — `kernel_family` `gfx950_a8w8_blockscale_ASM` at `aiter/ops/gemm_op_a8w8.py:668`
- `src_9dd4c645f2e22c23` — `kernel_family` `ck` at `aiter/ops/gemm_op_a8w8.py:677-682`
- `src_ae977d7a08feed5e` — `kernel_family` `cktile` at `aiter/ops/gemm_op_a8w8.py:679-682`
- `src_08f91e081b402af2` — `kernel_family` `gemm_a8w8_blockscale_ck` at `aiter/ops/gemm_op_a8w8.py:683`
- `src_f4e4d3f780fda60a` — `kernel_family` `cktile` at `aiter/ops/gemm_op_a8w8.py:734-743`
- `src_24f53ab3d301476f` — `kernel_family` `asm` at `aiter/ops/gemm_op_a8w8.py:738-743`
- `src_bcbb91da2b9d2eb8` — `kernel_family` `flydsl` at `aiter/ops/gemm_op_a8w8.py:629-630`
- `src_df8a47f1c59069da` — `kernel_family` `gemm_a8w8_bpreshuffle_ck` at `aiter/ops/gemm_op_a8w8.py:632`

#### Background (always-on reference docs, not measurements)

- [`languages/flydsl/patterns.md:41-53`](../languages/flydsl/patterns.md)
- [`operators/skinny_gemv_decode/tuning.md:16-33`](../operators/skinny_gemv_decode/tuning.md)

#### Limits

- The routing conditions are AITER's at the pinned commit; its dispatch changes between releases, so re-read `tuned_gemm.py` and `ops/gemm_op_a8w8.py` after an AITER bump.
- A route says which backend runs, not how fast it is; hipblaslt and torch routes are libraries this corpus does not index.

## 3. Grid, workgroup tile, waves and split-K

- `workgroup_tile` — How much M/N/K work does one workgroup own?
- `work_partition` — How is work partitioned or persisted across tiles and the K dimension?
- `execution_geometry` — How many waves/warps and workitems cooperate on the tile?
- `configuration_constraints` — Which combinations are legal with respect to divisibility, LDS/register budgets and fixed parameters?

Also bears on this step: `flydsl-a8-mfma-selection` (step 6), `gemm-blockscale-k-loop-scale-application` (step 6), `flydsl-cshuffle-epilogue` (step 8), `flydsl-xor16-lds-layout-consistency` (step 4), `flydsl-preshuffle-b-layout-contract` (step 4), `flydsl-splitk-hot-loop-schedule-bundle` (step 7), `flydsl-a8-interleaved-schedule` (step 7), `flydsl-splitk-reduction-contract` (step 8), `flydsl-small-m-hgemm-family` (step 2), `gemm-shape-regime-routing` (step 2), `flydsl-host-launch-contract` (step 9), `flydsl-test-parity-gate` (step 10), `flydsl-lds-pipeline-depth` (step 7).

### Which FlyDSL HGEMM configurations will fail to build at all, before any of them can be slow?

**Card:** `flydsl-hgemm-tiling-validity` · **type:** `constraint` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `configuration_constraints`, `workgroup_tile`, `pipeline_schedule`, `execution_geometry`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `splitk_streamk_gemm`, `skinny_gemv_decode`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`
- `dtypes`: `f16`, `bf16`

#### Use when

- operator_family=gemm
- target_language=flydsl
- candidate configurations are being generated automatically or swept

#### Try

- Filter candidates before compiling them: N must satisfy N >= tile_n and N % tile_n == 0, and K/split_k must satisfy K/split_k >= tile_k and (K/split_k) % tile_k == 0.
- Require tile_k >= 32 and tile_k % 32 == 0, tile_m % (block_m_warps*16) == 0 and tile_n % (block_n_warps*16) == 0.
- Require each of tile_m*tile_k, tile_n*tile_k and tile_m*tile_n to be divisible by 8*64*block_m_warps*block_n_warps, the per-block vectorised load width.
- Estimate LDS as a_lds = max(stages*tile_m*tile_k*2, tile_m*tile_n*2) when B is not staged through LDS; when it is, the estimate is not a_lds plus something, it becomes align_up(a_lds,16) + stages*tile_n*tile_k*2. Drop candidates over the block budget, which the library takes from `addressable_lds_bytes_for_gfx`: 163840 bytes on gfx950 and 65536 on the other CDNA targets.
- Do not spend a round on stages, pack_n or c_to_lds in the generic family: the kernel compiles a fixed 2-stage pipeline, pack_n=1 and c_to_lds off, and rejects anything else. They read like tuning knobs in the signature and are not.
- Do not combine b_preshuffle=True with b_to_lds=True; the kernel rejects it.
- Check the split-K reduction counter capacity separately — see flydsl-splitk-reduction-contract.

#### Why this is a candidate

- AITER performs all of these checks in host code before the kernel is built, so they are the stated contract rather than an inference from a crash.
- An optimizer that cannot separate 'illegal' from 'slow' spends its budget re-discovering the first, and a budgeted search has few rounds to spend.

#### Keep as alternatives

- Let the compiler reject the configuration and catch the exception — acceptable only when the search is small, since each rejection still costs a compile.

#### Evidence

- `src_9ae627bec0096537` — `config_validity` `_validate_hgemm_tiling` at `aiter/ops/flydsl/gemm_kernels.py:330-456`
- `src_d3f3cb90295b0c40` — `config_validity` `_estimate_hgemm_lds_bytes` at `aiter/ops/flydsl/gemm_kernels.py:249-268`
- `src_5c7e74ddb8a071de` — `config_validity` `_check_split_k_counter_capacity` at `aiter/ops/flydsl/gemm_kernels.py:695-707`
- `src_7c7ddd401ff4aaeb` — `config_limit` `SPLIT_K_COUNTER_MAX_LEN, 128` at `aiter/ops/flydsl/gemm_kernels.py:37`
- `src_c139f07a63569327` — `config_limit` `HGEMM_EXTRA_BLOCK_K_LOOPS_MIN, 2` at `aiter/ops/flydsl/gemm_kernels.py:77`
- `src_539c4b7bf95dca07` — `config_limit` `HGEMM_EXTRA_BLOCK_K_LOOPS_MAX, 8` at `aiter/ops/flydsl/gemm_kernels.py:78`
- `src_0f8216bb2c53105f` — `config_validity` `Invalid tile_k={tile_k}; latest kernel requires tile_k >= 32` at `aiter/ops/flydsl/gemm_kernels.py:358-360`
- `src_4404f2040e4191a0` — `config_validity` `Invalid tile_k={tile_k}; latest kernel requires tile_k % 32 == 0` at `aiter/ops/flydsl/gemm_kernels.py:362-364`
- `src_d76e1615be7ab856` — `config_validity` `Invalid tile combination: tile_m * tile_k must be divisible by` at `aiter/ops/flydsl/gemm_kernels.py:415-418`
- `src_cbd1aeb93c0eb00c` — `config_validity` `Current kernel only supports stage={FIXED_STAGE}; got stage={stage}` at `aiter/ops/flydsl/gemm_kernels.py:119-121`
- `src_c01de8a05cd112e2` — `config_validity` `Current kernel only supports `pack_n=1`; got pack_n={pack_n}` at `aiter/ops/flydsl/gemm_kernels.py:372-374`
- `src_d0f75ad1bcdf6730` — `config_validity` `Current kernel only supports c_to_lds={FIXED_C_TO_LDS};` at `aiter/ops/flydsl/gemm_kernels.py:128-131`
- `src_e248a6669d5daaaf` — `config_validity` `Current kernel requires b_to_lds=False when b_preshuffle=True` at `aiter/ops/flydsl/gemm_kernels.py:161-163`
- `src_ce6c8804ca232bf6` — `config_validity` `addressable_lds_bytes_for_gfx` at `aiter/ops/flydsl/utils.py:14-22`
- `src_c694123000988e0b` — `arch_gate` `gfx950` at `aiter/ops/flydsl/utils.py:18-19`

#### Background (always-on reference docs, not measurements)

- [`languages/flydsl/authoring_optimization.md:75-93`](../languages/flydsl/authoring_optimization.md)
- [`languages/flydsl/knobs.md:20-40`](../languages/flydsl/knobs.md)
- [`languages/flydsl/patterns.md:21-39`](../languages/flydsl/patterns.md)

#### Limits

- The LDS formula is the library's own estimate for this kernel shape; a rewritten staging scheme changes it and the estimate stops applying.
- Passing every check means the configuration builds, not that it is fast, and not that it is correct in a modified kernel.

### Which FlyDSL A8 / FP4 preshuffle-GEMM configurations are rejected before they can be slow?

**Card:** `flydsl-a8-legality` · **type:** `constraint` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `configuration_constraints`, `architecture_capability`, `workgroup_tile`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `scaled_quant_gemm`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`
- `dtypes`: `fp8_e4m3fnuz`, `fp8_e4m3fn`, `int8`, `int4`, `fp4_e2m1`, `mxfp4`

#### Use when

- target_language=flydsl
- quantized GEMM built on the A8 preshuffle family (fp8, int8/int4, or fp4 on gfx950)
- candidate configurations are being generated or swept

#### Try

- Filter before compiling: tile_k*elem_bytes must be a multiple of 64; tile_m*tile_k*elem_bytes (halved for fp4) must divide evenly over the fixed 256-thread block; and each thread's share of the A tile must be a multiple of the 16-byte load.
- Accept only in_dtype in {fp8, int8, int4, fp16, bf16, fp4} and out_dtype in {fp16, bf16}; there is no fp32 output.
- On the scaled-MFMA path (gfx95x with non-integer, non-half inputs) require tile_k to be a multiple of 128 — see flydsl-a8-mfma-selection.
- Treat FP4 as gfx950-only: it raises on other architectures, fp8 activations with fp4 weights are not supported, and FP4 with tile_k=128 supports only lds_stage=2.
- For int8, confirm the installed FlyDSL exposes `mfma_i32_16x16x32i8` (or the `_i8` spelling); its absence raises instead of falling back.

#### Why this is a candidate

- Every item is a check and a raise in the compile entry point, so it is the stated contract rather than an inference from a crash.
- The fixed 256-thread block and the 16-byte A load make several tile shapes that look plausible from the HGEMM space illegal here.

#### Keep as alternatives

- Let the compiler reject configurations only when the search is small; each rejection still costs a compile.

#### Evidence

- `src_e1aea77c2c7f9600` — `config_validity` `tile_k_bytes must be divisible by 64, got tile_k_bytes={tile_k_bytes}` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:198-201`
- `src_6a9c3bef7e7f1bd0` — `config_validity` `tile_m*tile_k*elem_bytes/a_elem_vec_pack must be divisible by` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:232-235`
- `src_d2906b5ee9f2c4b6` — `config_validity` `bytes_per_thread_a ({bytes_per_thread_a}) must be divisible by {a_load…` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:240-242`
- `src_3479ce3bb9436d9d` — `config_validity` `in_dtype must be one of (` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:165-168`
- `src_65bcb0094b2ef2f3` — `config_validity` `out_dtype must be` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:170`
- `src_bc08df5300de01f3` — `config_validity` `tile_k must be divisible by 128 for mfma_scale_x128, got tile_k={tile_…` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:855-857`
- `src_7e2a5d7fc5c13904` — `config_validity` `FP4 GEMM requires gfx950, got {get_hip_arch()}` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:1706`
- `src_68781327203b224d` — `config_validity` `fp8-A not yet supported with MXFP4 kernel (op_sel_a overflow)` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:1702-1704`
- `src_f5c5e87d02b8a8e5` — `config_validity` `FP4 tile_k=128 currently only supports lds_stage=2` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:211`
- `src_c9b3df8a49dab9d2` — `config_validity` `FP4 requires tile_k=128 or tile_k >= {64 * 2 * a_elem_vec_pack}` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:205-209`
- `src_f9f2544162138ea2` — `mfma_intrinsic` `mfma_i32_16x16x32i8` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:215-217`

#### Background (always-on reference docs, not measurements)

- [`languages/flydsl/knobs.md:82-87`](../languages/flydsl/knobs.md)
- [`operators/scaled_quant_gemm/backends/flydsl.md:60-78`](../operators/scaled_quant_gemm/backends/flydsl.md)

#### Limits

- These are the checks in AITER's vendored kernel at the pinned commit; an authored kernel with its own staging has its own contract.
- Passing every check means the configuration builds, not that it is fast.

### How should a FlyDSL GEMM map workgroup IDs onto output tiles and XCDs, and where does AITER already do it?

**Card:** `gemm-workgroup-xcd-mapping` · **type:** `performance_candidate` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `work_partition`, `memory_access`, `execution_geometry`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `scaled_quant_gemm`, `grouped_gemm_moe`, `batched_gemm`, `splitk_streamk_gemm`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`

#### Use when

- target_language=flydsl
- a multi-XCD part (MI300X / MI355X: 8 XCDs, each with its own L2)
- the grid launches enough workgroups that their placement across XCDs matters

#### Try

- Know the starting point: no workgroup remap was found in the dense FlyDSL HGEMM or A8 sources scanned here (only the grouped-GEMM kernel has one), so consecutive workgroups follow the hardware's round-robin across XCDs and neighbours that share an A or B panel land on different L2s (background).
- Port the remap as a block-id permutation in the kernel prologue, the way the FlyDSL grouped-GEMM kernel does behind `xcd_swizzle`: with 8 XCDs, wgid = (linear_id % 8) * (num_wgs / 8) + linear_id / 8, then rasterise M in groups of `xcd_swizzle` rows before decoding the (m, n) tile.
- Or follow Triton's shared helpers: `remap_xcd` / `remap_xcd_chunked` regroup program IDs per XCD, then `pid_grid(..., GROUP_SIZE_M)` walks M in groups so consecutive programs reuse one B panel.
- Keep the unmapped kernel as the control arm (`xcd_swizzle=0` is the grouped kernel's default), gate the remap by grid size, and prove every tile is still visited exactly once when the grid is not a multiple of the XCD count.

#### Why this is a candidate

- Two implementations in AITER already solve this and state the formula in source; the dense FlyDSL kernels are the ones that do not.
- Whether it pays depends on how much panel reuse crosses XCDs for the shape, which is why it is a candidate with a control arm and not a default.

#### Keep as alternatives

- No remap — the default dispatch order.
- Grouped-M rasterisation alone (`GROUP_SIZE_M`) without an XCD remap.

#### Same question in other implementations

- **triton** — `remap_xcd` / `remap_xcd_chunked` plus `pid_grid` with `GROUP_SIZE_M` in the shared pid helper, called from the GEMM kernels with `NUM_XCDS`. Evidence: `src_1b8ded5e4825e5e1` (`aiter/ops/triton/utils/_triton/pid_preprocessing.py:28-54`), `src_a0815635126a1ea6` (`aiter/ops/triton/utils/_triton/pid_preprocessing.py:10-24`), `src_d760ac9f0e4bff02` (`aiter/ops/triton/utils/_triton/pid_preprocessing.py:58-80`), `src_334e95f3f9406108` (`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16.py:105`), `src_de6d336c2ee7601b` (`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16.py:76`)

#### Evidence

- `src_69a5695f3c33fd50` — `grid_mapping` `xcd_swizzle` at `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py:132`
- `src_3a83a99245462451` — `grid_mapping` `xcd_swizzle` at `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py:523-558`
- `src_923af36a5869aea6` — `grid_mapping` `_wgs_per_xcd` at `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py:544`

#### Background (always-on reference docs, not measurements)

- [`optimization/xcd_l2_locality.md:36-49`](../optimization/xcd_l2_locality.md)
- [`hardware/shared/l2_xcd_swizzle.md`](../hardware/shared/l2_xcd_swizzle.md)

#### Limits

- The grouped kernel's tile decode after the remap is MoE-specific; the remap ports, the decode has to be rewritten for a dense grid.
- Measured effects of XCD remapping live in learned cards and expert skills behind their switches; this card carries none.

### When does a GEMM case need split-K to fill the machine, and what does choosing it commit the rest of the kernel to?

**Card:** `gemm-split-k-when-grid-underfills` · **type:** `performance_candidate` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `work_partition`, `workgroup_tile`, `execution_geometry`, `epilogue`, `runtime_contract`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `scaled_quant_gemm`, `grouped_gemm_moe`, `splitk_streamk_gemm`, `skinny_gemv_decode`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`

#### Use when

- target_language=flydsl
- a case whose output tile count ceil(M/tile_m)*ceil(N/tile_n) is well below the CU count — 304 on MI300X, 256 on MI350X/MI355X — the usual state of decode and skinny-M shapes
- K is deep enough to split: every split still runs several tile_k iterations (the HGEMM derivation asks for 2 to 8 block-K loops per split for its non-base split factors)

#### Try

- Count output tiles against CUs before tuning anything else in the case. A grid that is a fraction of the CU count leaves the rest of the machine idle, and no schedule or layout change later in the kernel recovers it: this is a partition decision, made before the pipeline work.
- Raise split_k until the workgroup count approaches the CU count, taking values from the kernel's legal set — for the FlyDSL HGEMM, split_k divides K and K/split_k is a multiple of tile_k (flydsl-hgemm-config-space). Past that point a larger split adds combine traffic without adding parallelism.
- Count rounds, not workgroups: AITER's ASM launchers score each (tile, split) by ceil(tiles*split / CUs) and keep the fewest rounds, preferring fewer idle CUs in the last round (gemm-asm-tile-inventory-and-selection). One workgroup past a CU multiple costs a whole extra round, so step the split down or the tile up rather than overshoot.
- Choose the combine together with the split, because split-K is not a knob on its own: the FlyDSL HGEMM initialises the output from split 0 and atomically adds the others (flydsl-splitk-reduction-contract), Triton writes fp32 partials and runs a separate reduce kernel, ASM multiplies the launch grid by the split. An authored kernel has to pick one and supply its host-side state.
- Re-check numerics with the split: a different accumulation order — and, in FlyDSL's atomic combine, adding partials already rounded to the output dtype — changes the result, so gate against the task's oracle at its stated tolerance.
- Seed from where AITER shipped split-K: FlyDSL bf16 on gfx950 selects split_k > 1 in 83 of 86 rows at M<=16 and 38 of 45 at M 17-64, and split_k = 1 in every row above M = 256; Triton's gfx942 fp8 block-scale configs vary NUM_KSPLIT (1 to 16) only in the buckets up to M<=512 and fix it at 1 from M<=1024 on.

#### Why this is a candidate

- This is the grid-fill rule of the background docs — compute the output tiles and split K when they are far below the CU count. The FlyDSL derivation checks divisibility only, so the occupancy half of the decision is written nowhere in the kernel.
- Both backends' shipped selections put split-K where that rule predicts it, in the small-M buckets, and turn it off for large M.

#### Keep as alternatives

- Keep split_k = 1 and shrink tile_m / tile_n to raise the tile count: nothing to combine, but less MFMA work per workgroup (background).
- Stream-K or a persistent grid sized to the CU count, which also balances a tile count just off a CU multiple — no FlyDSL precedent in AITER.
- split_k = 1 as the control arm.

#### Same question in other implementations

- **triton** — NUM_KSPLIT partitions K inside the kernel and SPLITK_BLOCK_SIZE sizes each partition; a separate reduce kernel sums the fp32 partials. The block-scale kernel carries the same two parameters. Evidence: `src_0f50a57ed622a4f4` (`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16.py:77`), `src_8f1fdafc50973349` (`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16.py:209`), `src_4fbd4a0e0992c880` (`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8w8_blockscale.py:80`), `src_a6edf5790cd2f1f1` (`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8w8_blockscale.py:81`)
- **asm** — the launcher scores every (kernel, split) of the architecture by rounds = ceil(tiles*split / CUs), keeps the fewest rounds, and multiplies the launch grid by the split; the bf16 launcher caps split at min(CUs / output tiles, 16, K / sub_k). Evidence: `src_4044f249c79abf90` (`csrc/py_itfs_cu/asm_gemm_a16w16.cu:250`), `src_b07a8dc43c913b9d` (`csrc/py_itfs_cu/asm_a8w8_blockscale_bpreshuffle.cu:353`), `src_3528bd18e9f8f895` (`csrc/py_itfs_cu/asm_a8w8_blockscale_bpreshuffle.cu:119`), `src_787ebf8727d0c7e1` (`csrc/py_itfs_cu/asm_gemm_a16w16.cu:138`)
- **hip** — the decode skinny kernels split K across the lanes of a wave instead of across workgroups: weight rows are spread over CUs x YTILE (`mindiv`) and each lane's partial is reduced with cross-lane shuffles, so nothing is combined in memory. Evidence: `src_4afbdde799d96232` (`csrc/kernels/custom_kernels.cu:1738`), `src_0b8de37c2b304d4a` (`csrc/kernels/custom_kernels.cu:350`)

#### Evidence

- `src_642476c9b693d5f1` — `config_space` `_hgemm_split_k_options` at `aiter/ops/flydsl/gemm_kernels.py:229-246`
- `src_58d4bf39e7ca8f03` — `config_validity` `Invalid split-K: K={k} must be divisible by split_k={split_k}` at `aiter/ops/flydsl/gemm_kernels.py:397-399`
- `src_fce6be7eb6b247d6` — `config_space` `HGEMM_BASE_SPLIT_K_OPTIONS` at `aiter/ops/flydsl/gemm_kernels.py:75`
- `src_5c7e74ddb8a071de` — `config_validity` `_check_split_k_counter_capacity` at `aiter/ops/flydsl/gemm_kernels.py:695-707`
- `cfg_973fe63f897809df` — FlyDSL `hgemm` `abf16_wbf16_bf16` selections in `bf16_tuned_gemm` on `gfx950`/256 CUs, `M<=16`: 86 shipped row(s)
- `cfg_ac0c03746ab634f6` — FlyDSL `hgemm` `abf16_wbf16_bf16` selections in `bf16_tuned_gemm` on `gfx950`/256 CUs, `M17-64`: 45 shipped row(s)
- `cfg_ae017b25b6473300` — FlyDSL `hgemm` `abf16_wbf16_bf16` selections in `bf16_tuned_gemm` on `gfx950`/256 CUs, `M257-1024`: 29 shipped row(s)
- `cfg_d2642e1347a53ea1` — Triton seed `G E M M   A 8 W 8 _ B L O C K S C A L E` on `gfx942`, `M_LEQ_8` (see [`gemm_triton_seeds.md`](gemm_triton_seeds.md))
- `cfg_c23460f537b313ac` — Triton seed `G E M M   A 8 W 8 _ B L O C K S C A L E` on `gfx942`, `M_LEQ_16` (see [`gemm_triton_seeds.md`](gemm_triton_seeds.md))
- `cfg_da4a49e152c50056` — Triton seed `G E M M   A 8 W 8 _ B L O C K S C A L E` on `gfx942`, `M_LEQ_1024` (see [`gemm_triton_seeds.md`](gemm_triton_seeds.md))

#### Background (always-on reference docs, not measurements)

- [`operators/splitk_streamk_gemm/tuning.md:16-37`](../operators/splitk_streamk_gemm/tuning.md)
- [`optimization/wave_and_grid_sizing.md:49-61`](../optimization/wave_and_grid_sizing.md)
- [`languages/flydsl/knobs.md:52-60`](../languages/flydsl/knobs.md)
- [`languages/flydsl/patterns.md:68-76`](../languages/flydsl/patterns.md)

#### Limits

- Whether split-K pays for a shape is a measurement; the tile count says when it is worth trying, not that it wins.
- The CU counts are hardware facts from the background docs, not from this source.
- A split that fills the grid can still be illegal for its combine: the FlyDSL HGEMM counters allow at most 128 output tiles when split_k > 1.

### Which tile, warp and split-K values will the FlyDSL HGEMM family actually accept, and how does a shape narrow them?

**Card:** `flydsl-hgemm-config-space` · **type:** `performance_candidate` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `workgroup_tile`, `work_partition`, `execution_geometry`, `tunable_surface`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `splitk_streamk_gemm`, `skinny_gemv_decode`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`
- `dtypes`: `f16`, `bf16`

#### Use when

- operator_family=gemm
- target_language=flydsl
- a configuration is being chosen or searched for a known M, N, K

#### Try

- Search tile_m within (16, 32, 48, 64, 80, 96, 112, 128, 160, 256), tile_n within (64, 128, 160, 192, 256) and tile_k within (64, 96, 128, 160, 256) rather than treating the 128x128x64 default as the space.
- Search the warp layout and B staging as one knob: the registry enumerates exactly five (block_m_warps, block_n_warps, b_to_lds) variants — (1,2,F), (1,4,F), (2,2,F), (1,4,T), (2,2,T).
- Cap tile_m at max(96, align_up(2*M, 16)): a tile_m far above M pays for rows that do not exist, and the shipped derivation refuses those candidates outright.
- Take split_k from 1..32, keeping only values that divide K and leave K/split_k divisible by tile_k; every candidate must clear that filter, including 1, 2, 4, 8 and 16. Those five need nothing further; any other divisor also has to leave between 2 and 8 block-K loops per split.
- Whether split_k > 1 is worth generating at all is the grid-fill decision in gemm-split-k-when-grid-underfills: the derivation here checks divisibility, never occupancy.
- Seed from what AITER shipped for your bucket before sweeping (FlyDSL tuned-DB table below): on gfx950 bf16 every shipped FlyDSL selection used either (2,2,b_to_lds=True) or (1,4,b_to_lds=False); the M<=64 buckets are dominated by tile_m=32 with split_k 4-16; every bucket above M=256 uses split_k=1, and M 257-1024 favours the non-power-of-two tile_m=96.

#### Why this is a candidate

- AITER states the space as module-level option tuples and two derivation helpers, not as one default; the default is one point inside it.
- The derivations encode the shape-dependence directly — tile_m is bounded by M, and split_k by the divisibility of K — so the narrowing rule is readable rather than guessed.
- This card says which candidates are legal and worth generating. It does not say which one wins; that is what the benchmark is for.

#### Keep as alternatives

- Keep the library default 128x128x64 with split_k=1 as the control arm to measure against.
- Fix the tile and search only split_k when N or K constrain the tiling to one legal shape.

#### Evidence

- `src_426b8960fb14a04f` — `config_space` `HGEMM_TILE_M_OPTIONS` at `aiter/ops/flydsl/gemm_kernels.py:74`
- `src_62ce0ec6b152ecaa` — `config_space` `HGEMM_TILE_N_OPTIONS` at `aiter/ops/flydsl/gemm_kernels.py:72`
- `src_3c72689d786ca8be` — `config_space` `HGEMM_TILE_K_OPTIONS` at `aiter/ops/flydsl/gemm_kernels.py:73`
- `src_fce6be7eb6b247d6` — `config_space` `HGEMM_BASE_SPLIT_K_OPTIONS` at `aiter/ops/flydsl/gemm_kernels.py:75`
- `src_96320a5bab17a8f5` — `config_limit` `HGEMM_MAX_SPLIT_K, 32` at `aiter/ops/flydsl/gemm_kernels.py:76`
- `src_562e9940a52a202f` — `config_space` `_hgemm_tile_m_options` at `aiter/ops/flydsl/gemm_kernels.py:222-226`
- `src_642476c9b693d5f1` — `config_space` `_hgemm_split_k_options` at `aiter/ops/flydsl/gemm_kernels.py:229-246`
- `src_27d99a67a1d27ccc` — `config_space` `KERNEL_CONFIG_VARIANTS` at `aiter/ops/flydsl/gemm_kernels.py:79-105`
- `cfg_973fe63f897809df` — FlyDSL `hgemm` `abf16_wbf16_bf16` selections in `bf16_tuned_gemm` on `gfx950`/256 CUs, `M<=16`: 86 shipped row(s)
- `cfg_ac0c03746ab634f6` — FlyDSL `hgemm` `abf16_wbf16_bf16` selections in `bf16_tuned_gemm` on `gfx950`/256 CUs, `M17-64`: 45 shipped row(s)
- `cfg_147eed9b10acbc4e` — FlyDSL `hgemm` `abf16_wbf16_bf16` selections in `bf16_tuned_gemm` on `gfx950`/256 CUs, `M65-256`: 52 shipped row(s)
- `cfg_ae017b25b6473300` — FlyDSL `hgemm` `abf16_wbf16_bf16` selections in `bf16_tuned_gemm` on `gfx950`/256 CUs, `M257-1024`: 29 shipped row(s)

#### Background (always-on reference docs, not measurements)

- [`operators/splitk_streamk_gemm/tuning.md:16-19`](../operators/splitk_streamk_gemm/tuning.md)
- [`languages/flydsl/knobs.md:42-50`](../languages/flydsl/knobs.md)
- [`languages/flydsl/authoring_gemm_levers.md:103-117`](../languages/flydsl/authoring_gemm_levers.md)
- [`optimization/occupancy_and_registers.md:54-67`](../optimization/occupancy_and_registers.md)

#### Limits

- These are the values the library is willing to compile, not values anybody measured as best; every candidate still has to be timed on the target box.
- Shipped selections come from AITER's tuner on its own box and competing set; they seed a search, they do not end one.
- The option tuples are the generic HGEMM family's. The small-M family has its own, wider, N space — see flydsl-small-m-hgemm-family.
- A candidate that is inside the space can still be rejected at compile time by the tiling and LDS rules — see flydsl-hgemm-tiling-validity.

### Which tile, LDS-stage, epilogue, async and occupancy combinations will the FlyDSL A8 (fp8/int8) preshuffle GEMM build, and how do they differ by architecture?

**Card:** `flydsl-a8-preshuffle-config-space` · **type:** `performance_candidate` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `workgroup_tile`, `tunable_surface`, `pipeline_schedule`, `epilogue`, `execution_geometry`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `scaled_quant_gemm`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`
- `dtypes`: `fp8_e4m3fnuz`, `fp8_e4m3fn`, `int8`

#### Use when

- target_language=flydsl
- fp8 or int8 GEMM with a preshuffled B (the A8 preshuffle family, or an authored kernel modelled on it)
- a configuration is being chosen or searched

#### Try

- Search tiles from the family's own catalogs, not from the HGEMM tuples: a shared lds_stage=2 catalog spanning tile_m 16-256 (tile_k up to 512 for small M), a smaller separate catalog for lds_stage=1, the tile_k=64 tiles 64x256x64 and 128x128x64 only on gfx942, and 256x256x128 only on gfx950.
- Treat lds_stage, use_cshuffle_epilog, use_async_copy and waves_per_eu as a four-way sweep: the tuner crosses lds_stage in {1,2}, cshuffle in {0,1}, async in {0,1} and waves_per_eu in {0..4}.
- Drop waves_per_eu values above the register estimate before compiling, as the tuner does with `_estimate_max_wpe`: tile_m*tile_n/256 accumulator VGPRs per thread for the 256-thread block, times 1.5 for pipeline buffers, against the 512-VGPR file.
- Estimate LDS with the family's own formula: only the A tile is staged (tile_m*tile_k*elem_bytes, halved for fp4); a CShuffle epilogue adds a 2*tile_m*tile_n staging buffer that shares the ping/pong space; lds_stage=2 keeps two buffers. Compare with `max_lds_bytes_for_tune()`, the per-arch budget.
- Start from the defaults the tuner falls back to — gfx942 128x128x128, gfx950 128x256x256, both lds_stage=2 and waves_per_eu=2 — plus the small-M defaults 16x64x512 and 32x64x512.
- Then seed from what AITER shipped for your bucket (FlyDSL tuned-DB table below): on gfx950 fp8 the M<=16 selections are mostly 16x64x512 and the M>1024 selections mostly 128x256x128 or 256x128x128, while lds_stage, CShuffle, async copy and waves_per_eu each take both values inside the same bucket — those four stay search axes.

#### Why this is a candidate

- AITER states the A8 space as module-level catalogs, a combo sweep and two estimators in its tune module, so the space is readable rather than guessed.
- The catalogs differ by architecture in ways a copied config will not reveal: a gfx942 tile_k=64 tile does not exist on gfx950.

#### Keep as alternatives

- Keep the family's per-arch default kernel as the control arm.
- Use the HGEMM family when the operands are not quantized; the two families share neither knobs nor catalogs.

#### Evidence

- `src_f6f28f20adae28cd` — `config_space` `_base_tiles_lds2_common` at `aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py:188-218`
- `src_af9ebdf13bdc947e` — `config_space` `_base_tiles_lds2_942_extra` at `aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py:221-224`
- `src_273e37ba3a8cefa8` — `config_space` `_base_tiles_lds2_950_extra` at `aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py:227-229`
- `src_6bd4c9e3168467b4` — `config_space` `_base_tiles_lds1` at `aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py:232-241`
- `src_055d381c0e94f538` — `config_space` `_LDS_STAGES` at `aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py:246`
- `src_84d3653b2aaf49f3` — `config_space` `_CSHUFFLE_VALS` at `aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py:247`
- `src_623470961bcb8c24` — `config_space` `_ASYNC_COPY_VALS` at `aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py:248`
- `src_32599cca99565a9f` — `config_space` `_WAVES_PER_EU` at `aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py:249`
- `src_ff07751a8278d3bb` — `config_validity` `_estimate_max_wpe` at `aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py:267-279`
- `src_c701f7377a04c988` — `config_validity` `preshuffle_gemm_estimated_lds_bytes` at `aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py:119-157`
- `src_771ddc6685eb37b6` — `config_validity` `max_lds_bytes_for_tune` at `aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py:177-179`
- `src_bdc2de724bf3f124` — `config_space` `kernels_list_942` at `aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py:298-300`
- `src_89bc3925dd87e7aa` — `config_space` `kernels_list_950` at `aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py:301-303`
- `src_865f0e9972c72d26` — `config_space` `default_kernels_dict_942` at `aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py:306-314`
- `src_63fcc4e0eff52ae1` — `config_space` `default_kernels_dict_950` at `aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py:316-321`
- `cfg_909edfc582fce0bd` — FlyDSL `a8_preshuffle` `af8_wf8_b16` selections in `a8w8_bpreshuffle_tuned_gemm` on `gfx950`/256 CUs, `M<=16`: 72 shipped row(s)
- `cfg_2f03fe974e5d2b5c` — FlyDSL `a8_preshuffle` `af8_wf8_b16` selections in `a8w8_bpreshuffle_tuned_gemm` on `gfx950`/256 CUs, `M17-64`: 42 shipped row(s)
- `cfg_496162f129968cb9` — FlyDSL `a8_preshuffle` `af8_wf8_b16` selections in `a8w8_bpreshuffle_tuned_gemm` on `gfx950`/256 CUs, `M1025-4096`: 50 shipped row(s)
- `cfg_9636311ea3a3ce0d` — FlyDSL `a8_preshuffle` `af8_wf8_b16` selections in `a8w8_bpreshuffle_tuned_gemm` on `gfx950`/256 CUs, `M>4096`: 77 shipped row(s)

#### Background (always-on reference docs, not measurements)

- [`languages/flydsl/knobs.md:82-87`](../languages/flydsl/knobs.md)
- [`operators/scaled_quant_gemm/backends/flydsl.md:60-78`](../operators/scaled_quant_gemm/backends/flydsl.md)

#### Limits

- These are the configurations the tuner will build and the ones it shipped, not a ranking; every candidate still needs the target box.
- The catalogs belong to AITER's vendored FlyDSL at the pinned commit; an authored kernel with a different staging scheme needs its own estimate.
- Shipped FlyDSL selections exist only for gfx950 per-token fp8; there is none for gfx942 and none for block-scaled fp8.

### Which configuration does the Triton block-scale baseline run for each case, and what does AITER ship instead where it tuned small and large M separately?

**Card:** `gemm-blockscale-shipped-bucket-structure` · **type:** `performance_candidate` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `workgroup_tile`, `work_partition`, `execution_geometry`, `implementation_variant`, `tunable_surface`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `scaled_quant_gemm`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`
- `dtypes`: `fp8_e4m3fnuz_blockscale`, `fp8_e4m3fn_blockscale`

#### Use when

- target_language=flydsl
- an fp8 128x128 block-scale workload measured against AITER's Triton kernel, especially one whose cases span decode and prefill (for example M=8 and M=32768 in one kernel)

#### Try

- Find the baseline's configuration for every case before planning, in the task's frozen baseline or through the lookup: a per-(N, K) file if AITER tuned that shape, else the generic file, then M_LEQ / M_GEQ buckets, then `any`. gfx942 ships block-scale files for nine (N, K) shapes; the generic file's only entry is `any` — a 128x128x128 tile, NUM_KSPLIT = 1, 4 warps — so an untuned shape runs that one configuration at every M.
- Price the fallback per case: at decode M=8 with N=4096 that tile launches 32 workgroups on a 304-CU MI300X, a grid-fill gap a FlyDSL decode kernel can close (gemm-split-k-when-grid-underfills); at large M the same tile sits inside the range AITER ships for the shapes it did tune.
- Where AITER did tune a shape, small and large M are different structures: on gfx942 the M<=32 buckets ship BLOCK_SIZE_M mostly 4 to 16, BLOCK_SIZE_N mostly 16 and mostly one or two warps, and vary NUM_KSPLIT from 1 to 16; from M<=1024 on NUM_KSPLIT is 1, with BLOCK_SIZE_M 64 to 256, BLOCK_SIZE_N mostly 128 and mostly four to eight warps.
- Hold the scale block fixed across regimes: BLOCK_SIZE_K = 128 and the 16x16 MFMA (matrix_instr_nonkdim = 16) are shared by every gfx942 bucket, so decode and prefill differ in tile, split and warps, not in how K is blocked (gemm-blockscale-k-loop-scale-application).
- Mirror the tuned shapes rather than the fallback: one entry point that dispatches on M to a kernel specialised for each regime, since the tuned tiles differ by more than an order of magnitude in BLOCK_SIZE_M.
- Know the production competitors per regime too: AITER's tuned block-scale B-preshuffle database on gfx942 selects CK in 16 of 18 rows at M<=16 and ASM in 32 of 38 above M=4096.

#### Why this is a candidate

- In a Triton-to-FlyDSL block-scale task the frozen baseline is this Triton kernel with whatever the lookup returned, so the per-case target is readable before any kernel is written.
- The shipped tables are the only record of which tile and split AITER chose per regime once it did tune a shape.

#### Keep as alternatives

- One configuration for every case — simpler, and the control arm; expect it to trail on the regime it was not chosen for.

#### Evidence

- `src_2a228a58ade64092` — `config_lookup` `get_gemm_config` at `aiter/ops/triton/utils/gemm_config_utils.py:142-177`
- `src_318b927528cb6cda` — `tunable_param` `BLOCK_SIZE_M` at `aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8w8_blockscale.py:76`
- `src_caaad70b7145147d` — `tunable_param` `BLOCK_SIZE_N` at `aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8w8_blockscale.py:77`
- `src_4fbd4a0e0992c880` — `tunable_param` `NUM_KSPLIT` at `aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8w8_blockscale.py:80`
- `src_a6edf5790cd2f1f1` — `tunable_param` `SPLITK_BLOCK_SIZE` at `aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8w8_blockscale.py:81`
- `cfg_d6c5b6bf7798a092` — Triton seed `G E M M   A 8 W 8 _ B L O C K S C A L E` on `gfx942`, `M_LEQ_4` (see [`gemm_triton_seeds.md`](gemm_triton_seeds.md))
- `cfg_d2642e1347a53ea1` — Triton seed `G E M M   A 8 W 8 _ B L O C K S C A L E` on `gfx942`, `M_LEQ_8` (see [`gemm_triton_seeds.md`](gemm_triton_seeds.md))
- `cfg_c23460f537b313ac` — Triton seed `G E M M   A 8 W 8 _ B L O C K S C A L E` on `gfx942`, `M_LEQ_16` (see [`gemm_triton_seeds.md`](gemm_triton_seeds.md))
- `cfg_89a57b3ca5ee84c7` — Triton seed `G E M M   A 8 W 8 _ B L O C K S C A L E` on `gfx942`, `M_LEQ_32` (see [`gemm_triton_seeds.md`](gemm_triton_seeds.md))
- `cfg_658519ce57ddcf17` — Triton seed `G E M M   A 8 W 8 _ B L O C K S C A L E` on `gfx942`, `M_LEQ_512` (see [`gemm_triton_seeds.md`](gemm_triton_seeds.md))
- `cfg_da4a49e152c50056` — Triton seed `G E M M   A 8 W 8 _ B L O C K S C A L E` on `gfx942`, `M_LEQ_1024` (see [`gemm_triton_seeds.md`](gemm_triton_seeds.md))
- `cfg_1311810498769365` — Triton seed `G E M M   A 8 W 8 _ B L O C K S C A L E` on `gfx942`, `M_LEQ_2048` (see [`gemm_triton_seeds.md`](gemm_triton_seeds.md))
- `cfg_3eff28954f18b5fe` — Triton seed `G E M M   A 8 W 8 _ B L O C K S C A L E` on `gfx942`, `any` (see [`gemm_triton_seeds.md`](gemm_triton_seeds.md))
- `sel_f7e394faa23e37e6` — backend selected by AITER's tuner in `a8w8_blockscale_bpreshuffle_tuned_gemm` on `gfx942`/80 CUs, `M<=16`: ck 16 · asm 2 (rows)
- `sel_f1384cc1069c9823` — backend selected by AITER's tuner in `a8w8_blockscale_bpreshuffle_tuned_gemm` on `gfx942`/80 CUs, `M>4096`: asm 32 · ck 6 (rows)

#### Background (always-on reference docs, not measurements)

- [`optimization/wave_and_grid_sizing.md:49-61`](../optimization/wave_and_grid_sizing.md)

#### Limits

- Shipped configurations are selections without timings; the tuner's box and competing set are not recorded.
- These are Triton knob names (every group is on gemm_triton_seeds.md): carry over tile shape, split and warp count as intent, not as spellings.
- The bucket groups pool AITER's nine tuned (N, K) files; a task shape outside them has no shipped small-M configuration at all.

### Which tiles and split-K capabilities does AITER's hand-written ASM ship per architecture, and how does its launcher choose among them for a shape?

**Card:** `gemm-asm-tile-inventory-and-selection` · **type:** `performance_candidate` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `workgroup_tile`, `work_partition`, `execution_geometry`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `scaled_quant_gemm`, `splitk_streamk_gemm`, `skinny_gemv_decode`
- `target_languages`: `flydsl`

#### Use when

- target_language=flydsl
- a tile and split are being chosen per case, and the vendor's own choices for the same dtype and architecture are worth knowing

#### Try

- Read the per-architecture inventory as the vendor's tile set: fp8 128x128 block scale is the same six kernels on gfx942 and gfx950 — tile_n = 128, tile_m in 32, 48, 64, 80, 96, 128, all split-K capable and B-preshuffled. bf16 is 22 kernels on gfx942 (tile_n = 64, tile_m 32 to 160, 12 split-K capable) and 24 on gfx950 (adding two 256x256 kernels). fp4 exists only on gfx950 (35 kernels, 2 split-K capable).
- Choose tile and split the way the launcher does, per case rather than once: for every kernel of the architecture (the block-scale launcher also requires tile_n to divide N), and every split from 1 to 16 it supports, count the rounds `ceil(ceil(M/tile_m) * ceil(N/tile_n) * split / num_cu)`; keep the fewest rounds, and among equal rounds the one with fewer idle CUs in the last round or the larger `tile_m * tile_n / (tile_m + tile_n)`.
- The bf16 launcher also caps the split at `min(num_cu / output_tiles, 16, K / sub_k)` — split only up to what fills the machine.
- Under block scale keep tile_n a search axis, with 128 as ASM's point: every block-scale ASM kernel uses tile_n = 128 on both architectures (one N tile, one W scale per K block), while Triton's gfx942 block-scale configs fix GROUP_N = 128 but ship BLOCK_SIZE_N 16 or 64 at M <= 8 and mostly 128 at M 1025-2048 (gemm-blockscale-k-loop-scale-application).

#### Why this is a candidate

- The inventories state what exists per architecture; the launchers state the rule that picks from it. Both are the vendor's grid-fill heuristic written as code, and `num_cu` is read at run time, so the same rule serves the 304-CU MI300X and the 256-CU MI355X.

#### Keep as alternatives

- A single fixed tile for every case — the control arm, and what an untuned Triton baseline does.

#### Evidence

- `src_3528bd18e9f8f895` — `grid_fill` `local_round, tg_num, num_cu` at `csrc/py_itfs_cu/asm_a8w8_blockscale_bpreshuffle.cu:119`
- `src_c2ed9024f2039d1f` — `grid_fill` `local_round, tg_num, num_cu` at `csrc/py_itfs_cu/asm_gemm_a16w16.cu:145`
- `src_787ebf8727d0c7e1` — `grid_fill` `max_splitk, std::min(std::min(static_cast<int>(num_cu / pure_tg_num), …` at `csrc/py_itfs_cu/asm_gemm_a16w16.cu:138`
- `src_0558cdca6ff86ab9` — `grid_fill` `local_round, tg_num, num_cu` at `csrc/py_itfs_cu/asm_gemm_a8w8.cu:98`
- `src_fa4eb9ba8bba7192` — `grid_fill` `local_round, tg_num, num_cu` at `csrc/py_itfs_cu/asm_gemm_a4w4.cu:124`
- `src_3865c44e30f36de9` — `asm_tile` `tile_m=128, tile_n=128, split_k_capable=1, bpreshuffle=1` at `hsa/gfx942/fp8gemm_blockscale/fp8gemm_bf16_blockscale.csv:2`
- `src_fcf5d73d12271d55` — `asm_tile` `tile_m=32, tile_n=128, split_k_capable=1, bpreshuffle=1` at `hsa/gfx942/fp8gemm_blockscale/fp8gemm_bf16_blockscale.csv:3`
- `src_c792a003b578b049` — `asm_tile` `tile_m=48, tile_n=128, split_k_capable=1, bpreshuffle=1` at `hsa/gfx942/fp8gemm_blockscale/fp8gemm_bf16_blockscale.csv:4`
- `src_acee84984c040f98` — `asm_tile` `tile_m=64, tile_n=128, split_k_capable=1, bpreshuffle=1` at `hsa/gfx942/fp8gemm_blockscale/fp8gemm_bf16_blockscale.csv:5`
- `src_fe82852692f7eaca` — `asm_tile` `tile_m=80, tile_n=128, split_k_capable=1, bpreshuffle=1` at `hsa/gfx942/fp8gemm_blockscale/fp8gemm_bf16_blockscale.csv:6`
- `src_8475d71b5a90595e` — `asm_tile` `tile_m=96, tile_n=128, split_k_capable=1, bpreshuffle=1` at `hsa/gfx942/fp8gemm_blockscale/fp8gemm_bf16_blockscale.csv:7`
- `src_8866d3dbeb612f3a` — `asm_tile` `tile_m=128, tile_n=128, split_k_capable=1, bpreshuffle=1` at `hsa/gfx950/fp8gemm_blockscale/fp8gemm_bf16_blockscale.csv:2`
- `src_a4937eaa160077d1` — `asm_tile` `tile_m=32, tile_n=128, split_k_capable=1, bpreshuffle=1` at `hsa/gfx950/fp8gemm_blockscale/fp8gemm_bf16_blockscale.csv:3`
- `cfg_d2642e1347a53ea1` — Triton seed `G E M M   A 8 W 8 _ B L O C K S C A L E` on `gfx942`, `M_LEQ_8` (see [`gemm_triton_seeds.md`](gemm_triton_seeds.md))
- `cfg_1311810498769365` — Triton seed `G E M M   A 8 W 8 _ B L O C K S C A L E` on `gfx942`, `M_LEQ_2048` (see [`gemm_triton_seeds.md`](gemm_triton_seeds.md))

#### Background (always-on reference docs, not measurements)

- [`optimization/wave_and_grid_sizing.md:49-61`](../optimization/wave_and_grid_sizing.md)
- [`operators/splitk_streamk_gemm/tuning.md:16-37`](../operators/splitk_streamk_gemm/tuning.md)

#### Limits

- The inventory lists what AITER ships, not which kernel wins a shape; the launcher's rule is a heuristic, and the task's benchmark decides.
- ASM tiles are binaries: the tile shapes transfer as intent; the MFMA schedule inside them is not visible.

### How does AITER's hand-written HIP skinny GEMM organise work for M <= 16, and what can a FlyDSL decode kernel take from it?

**Card:** `gemm-skinny-decode-wave-split` · **type:** `performance_candidate` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `work_partition`, `execution_geometry`, `memory_access`, `implementation_variant`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `skinny_gemv_decode`, `scaled_quant_gemm`
- `target_languages`: `flydsl`
- `regimes`: `decode`

#### Use when

- target_language=flydsl
- decode: M is a handful of rows (AITER's skinny route covers M <= 16), so an MFMA tile along M is mostly padding

#### Try

- Stage the whole activation in LDS when it fits and keep it there for the workgroup's lifetime: the `_sml_` variant reserves 64 KB — one workgroup per CU on gfx942's 64 KB LDS, as the source comment intends, while gfx950's 160 KB would fit two — and is chosen when K*M <= 32K elements for bf16/fp16 (64K for the fp8 variant); up to 1.2x of that the `_hf_` variant, beyond it `_big_`.
- Split the output rows over every CU and the K dimension over the lanes: each wave owns YTILE rows of the weight, waves per workgroup are sized with `mindiv(rows, CuCount * YTILE, WvPrGrp)`, and each lane takes A_CHUNK elements of K per step.
- Reduce the lanes' partial dot products inside the wave with cross-lane shuffles (`__shfl_xor`, `__shfl`) — no second kernel and no atomics.
- Treat the knobs as a family, not constants: THRDS, YTILE, WvPrGrp, A_CHUNK and UNRL are template parameters of every variant.
- Compare with the FlyDSL small-M HGEMM (flydsl-small-m-hgemm-family), which answers the same regime with 16-row MFMA tiles: choose the structure for a FlyDSL decode kernel by measurement, not by precedent.

#### Why this is a candidate

- It is AITER's default bf16/fp16 decode kernel (gemm-default-backend-routing), and its launcher is the grid-fill rule for decode written in HIP.

#### Keep as alternatives

- An MFMA tile with split-K across workgroups (gemm-split-k-when-grid-underfills) — what the FlyDSL HGEMM and the Triton baselines do at decode.

#### Evidence

- `src_b8e76e6d00fd7ca3` — `kernel_family` `K_in * N_in <= 64 * 1024` at `csrc/kernels/custom_kernels.cu:2229`
- `src_abfa46c903cb04ea` — `kernel_family` `K_in * N_in <= 32 * 1024` at `csrc/kernels/custom_kernels.cu:1736`
- `src_abb2ef768886f8dc` — `kernel_family` `K_in * N_in <= 32 * 1024 * 1.2` at `csrc/kernels/custom_kernels.cu:1742`
- `src_4afbdde799d96232` — `grid_fill` `__wvPrGrp, mindiv(M_in, CuCount * _YTILEs, _WvPrGrp)` at `csrc/kernels/custom_kernels.cu:1738`
- `src_e32202e7a829f4ab` — `grid_fill` `__wvPrGrp, mindiv(M_in, CuCount * _YTILEs, _WvPrGrp)` at `csrc/kernels/custom_kernels.cu:2231`
- `src_0b8de37c2b304d4a` — `lane_reduction` `__shfl_xor` at `csrc/kernels/custom_kernels.cu:350`
- `src_c99341de831ed9d8` — `lane_reduction` `__shfl` at `csrc/kernels/custom_kernels.cu:1951`
- `src_f9a2106c23412f25` — `lds_alloc` `__shared__` at `csrc/kernels/custom_kernels.cu:713`
- `src_d33d9c364eba88bd` — `waves` `WvPrGrp* THRDS` at `csrc/kernels/custom_kernels.cu:688`
- `src_d78de056f0337d85` — `tunable_param` `typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK, int…` at `csrc/kernels/custom_kernels.cu:687`
- `src_5147e7cddd350323` — `tunable_param` `typename scalar_t, typename fp8_t, int THRDS, int YTILE, int WvPrGrp, …` at `csrc/kernels/custom_kernels.cu:1775`
- `src_2577a978dcbebafb` — `kernel_family` `is_skinny_default_shape` at `aiter/tuned_gemm.py:76-97`

#### Background (always-on reference docs, not measurements)

- [`operators/skinny_gemv_decode/tuning.md:16-33`](../operators/skinny_gemv_decode/tuning.md)
- [`languages/flydsl/knobs.md:77-80`](../languages/flydsl/knobs.md)

#### Limits

- The HIP family is fp16/bf16 plus an fp8 variant that multiplies by two scalar (per-tensor) scales at the end; a block-scale decode kernel still needs the per-K-block scale loop.
- Its structure is a precedent for a FlyDSL decode kernel, not a drop-in: FlyDSL has no skinny kernel in AITER to copy.

## 4. Layout and address computation

- `operand_layout` — How are operands transformed before and inside LDS?

Also bears on this step: `flydsl-a8-data-movement-helpers` (step 5).

### Which lane holds which row and column of a 16x16 MFMA fragment in AITER's FlyDSL kernels, and what does that fix for loads, in-register scaling and stores?

**Card:** `flydsl-mfma-fragment-lane-layout` · **type:** `semantic` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `operand_layout`, `epilogue`, `compute_instruction`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `scaled_quant_gemm`, `grouped_gemm_moe`, `splitk_streamk_gemm`, `skinny_gemv_decode`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`

#### Use when

- target_language=flydsl
- a kernel loads MFMA fragments, scales an accumulator per row or per column, or writes it back

#### Try

- Load fragments by lane: in the HGEMM, lane l reads A row `l % 16` at K offset `(l // 16) * frag_values * mfma_steps` (`ldmatrix_a_m_idx`, `ldmatrix_a_k_vec_idx`), and B mirrors it along N (`ldmatrix_b_n_idx`, `ldmatrix_b_k_vec_idx`).
- Read the accumulator as lane l holding four consecutive rows starting at `(l // 16) * 4`, in column `l % 16`: the row epilogue's row offset is `lane_div_16 * 4 + ii` for ii in 0..3, the HGEMM store uses `stmatrix_c_m_vec_idx = w_tid // WMMA_N * WMMA_C_FRAG_VALUES` and `stmatrix_c_n_idx = w_tid % WMMA_N`, and the A8 kernels name the same 4x16 lane grid `make_layout((4, 16), (16, 1))`.
- Scale in registers with that mapping: a lane multiplies its four rows by their row scales and its one column by that column's scale — the in-loop block-scale multiply (gemm-blockscale-k-loop-scale-application) and a per-token epilogue both need it.
- Derive loads, scales and stores from one mapping: a transposed reading on any side permutes the output without failing to compile.

#### Why this is a candidate

- Every FlyDSL GEMM in AITER writes this mapping as index arithmetic, so it can be read off the source instead of being rediscovered by debugging a permuted result.

#### Keep as alternatives

- Stage the accumulator through LDS with a C-shuffle epilogue (flydsl-cshuffle-epilogue) and index it row-major there, paying LDS for a simpler mapping.

#### Evidence

- `src_f01ef3793a7bd87c` — `fragment_layout` `ldmatrix_a_m_idx, w_tid % WMMA_M` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:288`
- `src_7e41979f22eae1f9` — `fragment_layout` `ldmatrix_a_k_vec_idx, w_tid // WMMA_M * WMMA_A_FRAG_VALUES * MFMA_PER_…` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:289`
- `src_44f9bf23c53fc7bc` — `fragment_layout` `ldmatrix_b_n_idx, w_tid % WMMA_N` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:290`
- `src_8ba4dcf6e9d68f6e` — `fragment_layout` `ldmatrix_b_k_vec_idx, w_tid // WMMA_N * WMMA_B_FRAG_VALUES * MFMA_PER_…` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:291`
- `src_91899d8d1f4341f6` — `fragment_layout` `stmatrix_c_m_vec_idx, w_tid // WMMA_N * WMMA_C_FRAG_VALUES` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:813`
- `src_074f27a8a5c89fcf` — `fragment_layout` `stmatrix_c_n_idx, w_tid % WMMA_N` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:814`
- `src_5cae13453d752cfb` — `fragment_layout` `lane_div_16_mul4, lane_div_16 * 4` at `aiter/ops/flydsl/kernels/mfma_epilogues.py:73`
- `src_1743dd27416aa823` — `layout_epilogue` `default_epilog` at `aiter/ops/flydsl/kernels/mfma_epilogues.py:50-82`
- `src_29a6126027710513` — `fragment_layout` `layout_lane16, fx.make_layout((4, 16), (16, 1))` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:426`

#### Background (always-on reference docs, not measurements)

- [`languages/flydsl/authoring_attention_levers.md:49-62`](../languages/flydsl/authoring_attention_levers.md)
- [`languages/flydsl/authoring_tile_programming.md:201-277`](../languages/flydsl/authoring_tile_programming.md)

#### Limits

- The mapping is for the 16x16 MFMA atom these kernels use; a 32x32 instruction has a different lane layout.
- K per lane depends on the instruction (four bf16 or eight fp8 elements per lane for the 16x16 forms), which the frag-value constants carry.

### How should a FlyDSL GEMM apply an XOR16 LDS layout without making stores and loads disagree?

**Card:** `flydsl-xor16-lds-layout-consistency` · **type:** `performance_candidate` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `operand_layout`, `configuration_constraints`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `scaled_quant_gemm`, `grouped_gemm_moe`, `batched_gemm`, `splitk_streamk_gemm`, `skinny_gemv_decode`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`
- `requires`: `lds_staging`

#### Use when

- operator_family=gemm
- target_language=flydsl
- an operand is staged through LDS (in the A8 family only A is; B is read preshuffled from global memory)
- XOR16 is included as a bank-conflict candidate

#### Try

- Define the byte-column transform once as col_in_bytes XOR ((row modulo k_blocks16) times 16).
- Apply the identical transform at every site: the global-to-LDS store and the LDS-to-fragment load, for each operand that goes through LDS.
- Re-derive the swizzle period per architecture: gfx942 LDS has 32 banks in 64 KiB, gfx950 has 64 banks in 160 KiB, so a stride that fully aliases on one may only partially conflict on the other (background).

#### Why this is a candidate

- The AITER implementation calls one helper at every write and read site; copying only one side changes addressing semantics rather than performance.
- The cited sites are A store, A fragment load, B store and B fragment load — the set that has to move together.

#### Keep as alternatives

- Keep the unswizzled layout as the control arm.
- Pad the LDS row instead of swizzling when LDS headroom allows — the FlyDSL grouped-GEMM kernels pad K by 8 elements unless they use the 128-bit LDS layout.
- Benchmark a different padding or swizzle scheme when the tile geometry changes.

#### Same question in other implementations

- **gluon** — `gl.SwizzledSharedLayout(...)` declared per staged operand (and per block-scale buffer); the compiler applies the same permutation at store and load. Evidence: `src_99b2357d48e96781` (`aiter/ops/triton/gluon/gemm_a8w8.py:123-125`), `src_4e349fde44ffedd3` (`aiter/ops/triton/gluon/gemm_a8w8.py:126-128`), `src_0526e1f18b1c8460` (`aiter/ops/triton/gluon/gemm_a8w8_blockscale.py:128-130`)

#### Evidence

- `src_2e329cfe46a52fb9` — `lds_swizzle` `swizzle_xor16` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:27-28`
- `src_d1f233838e74f022` — `lds_swizzle` `swizzle_xor16` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:455-457`
- `src_6f222b94149ab249` — `lds_swizzle` `swizzle_xor16` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:530`
- `src_d2c808ad3cd9ce7c` — `lds_swizzle` `swizzle_xor16` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:625-627`
- `src_0438c6e24a955ea2` — `lds_swizzle` `swizzle_xor16` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:675`
- `src_03338c20f236373f` — `lds_pad` `pad_k, 0` at `aiter/ops/flydsl/kernels/moe_gemm_2stage.py:206`

#### Background (always-on reference docs, not measurements)

- [`languages/flydsl/authoring_optimization.md:75-93`](../languages/flydsl/authoring_optimization.md)
- [`hardware/shared/memory_model_lds_bank.md`](../hardware/shared/memory_model_lds_bank.md)
- [`optimization/lds_and_bank_conflicts.md`](../optimization/lds_and_bank_conflicts.md)
- [`languages/flydsl/authoring_gemm_levers.md:133-148`](../languages/flydsl/authoring_gemm_levers.md)

#### Limits

- The citations prove address consistency, not that XOR16 reduces bank conflicts on every tile.
- Retain the unswizzled implementation and choose by target-box measurement.

### What does enabling B preshuffle on a FlyDSL HGEMM actually require?

**Card:** `flydsl-preshuffle-b-layout-contract` · **type:** `performance_candidate` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `operand_layout`, `runtime_contract`, `implementation_variant`, `configuration_constraints`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `splitk_streamk_gemm`, `skinny_gemv_decode`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`
- `requires`: `reused_weight`

#### Use when

- operator_family=gemm
- target_language=flydsl
- B preshuffle is being considered as a candidate
- the regime of B is known or can be read from the caller: a weight reused across calls, or per-call data

#### Try

- Put the preshuffled B path on the candidate list on the generic family instead of deferring it: b_preshuffle is the default of the public entry point (flydsl_hgemm) and of the internal compile helper.
- Do not read that default as the deployment state: AITER's tuned database keys on `bpreshuffle`, and every FlyDSL bf16 selection it ships for gfx950 has b_preshuffle=False — those deployments pass unshuffled weights. Which layout you may assume is set by the caller's weight contract, not by the entry-point default.
- Treat b_preshuffle as a change of B's storage layout, not as a boolean tuning flag: the kernel expects B already permuted into the family's shuffle layout, and the host side must either shuffle it or be told it is pre-shuffled.
- Do not decline preshuffle on the grounds that no offline shuffle is available. The library performs the permutation itself with shuffle_weight(b, layout=_get_flydsl_shuffle_layout(pack_n)), exposes auto_shuffle_b to apply it at call time, and skips it when the tensor already carries the is_shuffled marker — the shuffle, the layout selector and the once-per-tensor skip are all readable from the entry point, so a copy of the kernel that cannot import them can still reimplement the same three pieces.
- Hoist the permutation out of the measured region rather than counting it against the win: key a cached shuffled copy on the identity of the source tensor so it is paid once per distinct B and reused by every later launch. is_shuffled is that idea in the library; a cache that also detects in-place mutation of the source is the same idea made safe.
- Scope the work item to the host layer before assigning it. The permutation happens on the host side of the entry point, before the launcher is called — it is not reachable from the kernel body. A direction that hands an engineer only the kernel source will come back 'not attempted, the host file is outside my lane', which is a decomposition failure and not a finding about preshuffle. The owner of this direction needs the configuration/launch module and permission to add a host-side module beside it.
- Make the cache invalidation-safe or do not cache: a freed tensor's address can be handed to a later allocation, so identity alone can alias two different matrices. Pair the address with something that changes when the contents do, and treat a mismatch as a re-shuffle rather than a stale hit.
- Settle the accounting question first, then act on the answer instead of parking the lever because the question exists. Read the benchmark loop: if it builds one B and calls the kernel repeatedly against it, with warmup iterations before the timed ones, then a once-per-tensor shuffle lands in warmup by construction and the steady-state cost is the honest number for that workload. Cache it, and write one line in the report saying B is treated as a reused operand and the permutation is amortised — a declared accounting choice is a complete answer, whereas deferring the lever leaves the fastest available configuration untested and reports nothing.
- Where B is genuinely per-call data the same reading gives the opposite instruction: do not cache, charge the shuffle to every launch, and expect the unshuffled path to win. The rule is that the regime decides, and the regime is readable from the caller.
- Keep b_to_lds=false whenever b_preshuffle=true; the current kernel rejects the combination.
- Keep b_preshuffle=false on the small-M family, which rejects it outright.
- Retune the tile shortlist on the preshuffled tree rather than carrying over the configuration that won without it; the two paths feed the MFMA differently and their best tiles need not agree.
- The A8 (fp8/int8) family has no such choice: it always consumes B preshuffled, so its host side must supply the shuffled weight (and the matching scale layout) — see flydsl-a8-preshuffle-config-space.

#### Why this is a candidate

- The generic HGEMM path supports preshuffled B and its dispatch default is even b_preshuffle=true, so this is a supported layout rather than a forbidden one — but the guard rails around it are all layout guard rails.
- The two rejections in the source are specific and different in kind: an unsupported combination (with b_to_lds) and an unsupported family (small-M). Neither is a statement about speed.
- The reason to name the amortisation machinery explicitly is that the layout requirement reads like a blocker and is not one. An author who stops at 'the kernel wants B permuted' concludes the harness cannot supply it; the entry point shows the library supplying it three different ways in the same nine lines.

#### Keep as alternatives

- Use the unshuffled B path as the correctness and performance control.
- Shuffle B once at load time outside the timed region when the deployment allows it, and measure both accounting rules explicitly.
- If B genuinely changes every call, the amortisation argument does not apply and the unshuffled path is the right answer — say which of the two regimes you are in before choosing.

#### Same question in other implementations

- **asm** — the bf16 assembly GEMM takes a `preshuffle` flag in its launcher and selects a preshuffled object. Evidence: `src_2616db93f0f6b2b7` (`csrc/py_itfs_cu/asm_gemm_a16w16.cu:105`)
- **ck** — separate B-preshuffle device-op templates (`*_BPreshuffle`) with their own tuned databases. Evidence: `src_24c2dc21486cdfe7` (`csrc/ck_gemm_a8w8_blockscale_bpreshuffle/include/gemm_a8w8_blockscale_bpreshuffle_common.cuh:92`), `src_6352032268991afb` (`csrc/ck_gemm_a8w8_blockscale_bpreshuffle/include/gemm_a8w8_blockscale_bpreshuffle_common.cuh:93`)

#### Evidence

- `src_93c0f2962c52fdfb` — `layout_preshuffle` `b_preshuffle` at `aiter/ops/flydsl/kernels/hgemm_dispatch.py:32`
- `src_4c74ced4671a1e5b` — `layout_preshuffle` `b_preshuffle` at `aiter/ops/flydsl/gemm_kernels.py:162`
- `src_3ee379ecc88d0ad1` — `layout_preshuffle` `b_preshuffle` at `aiter/ops/flydsl/gemm_kernels.py:767-768`
- `src_7b5432b56d9acd63` — `layout_preshuffle` `b_preshuffle` at `aiter/ops/flydsl/gemm_kernels.py:768`
- `src_520242be3bd7400e` — `layout_preshuffle` `b_preshuffle` at `aiter/ops/flydsl/gemm_kernels.py:886`
- `src_e7db1b030bf2e5c6` — `layout_preshuffle` `shuffle_weight` at `aiter/ops/flydsl/gemm_kernels.py:883`
- `src_886a3fb0a447c9a7` — `layout_preshuffle` `b_preshuffle` at `aiter/ops/flydsl/gemm_kernels.py:729`
- `src_2aa8afff0a4748b0` — `layout_preshuffle` `b_preshuffle` at `aiter/ops/flydsl/gemm_kernels.py:863`
- `src_e248a6669d5daaaf` — `config_validity` `Current kernel requires b_to_lds=False when b_preshuffle=True` at `aiter/ops/flydsl/gemm_kernels.py:161-163`
- `cfg_973fe63f897809df` — FlyDSL `hgemm` `abf16_wbf16_bf16` selections in `bf16_tuned_gemm` on `gfx950`/256 CUs, `M<=16`: 86 shipped row(s)
- `cfg_147eed9b10acbc4e` — FlyDSL `hgemm` `abf16_wbf16_bf16` selections in `bf16_tuned_gemm` on `gfx950`/256 CUs, `M65-256`: 52 shipped row(s)

#### Background (always-on reference docs, not measurements)

- [`languages/flydsl/knobs.md:62-69`](../languages/flydsl/knobs.md)
- [`languages/flydsl/patterns.md:55-66`](../languages/flydsl/patterns.md)

#### Limits

- A shipped default is a precedent, not a benchmark. That the library chose b_preshuffle=true says the path is first class and worth trying early; it does not say it wins on your shape, and the unshuffled control still has to be run.
- The amortisation only holds where one B is reused across launches. Under an accounting rule that charges the first shuffle to a single call, the path can lose; state which rule you are reporting.
- A benchmark that reuses one pre-shuffled B across iterations measures the steady state, not the first call; state which one you are reporting.
- The card is grounded in the vendored library's contract. A kernel copy that has been edited may have moved these checks.

## 5. Global, LDS and register data movement

- `memory_access` — Which access width and cache policy does the data path use?

Also bears on this step: `flydsl-gfx942-path-gates` (step 2), `gemm-workgroup-xcd-mapping` (step 3), `flydsl-splitk-hot-loop-schedule-bundle` (step 7), `flydsl-a8-interleaved-schedule` (step 7), `gemm-skinny-decode-wave-split` (step 3).

### Is global-to-LDS async copy a knob, a fixed fact, or unavailable — per FlyDSL kernel family and architecture?

**Card:** `flydsl-async-copy-by-family` · **type:** `performance_candidate` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `memory_access`, `architecture_capability`, `pipeline_schedule`, `implementation_variant`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `scaled_quant_gemm`, `grouped_gemm_moe`, `batched_gemm`, `splitk_streamk_gemm`, `skinny_gemv_decode`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`
- `requires`: `lds_staging`

#### Use when

- target_language=flydsl
- an operand tile is staged through LDS

#### Try

- Split-K HGEMM: not a knob. The flag is computed from the architecture (off on gfx942, on elsewhere) and a request that disagrees raises. Every FlyDSL bf16 selection AITER ships for gfx950 has async_copy on.
- A8 preshuffle (fp8/int8): a real knob on both architectures. The A tile is copied 4 bytes per lane on gfx942 and 16 bytes elsewhere, and the tuner sweeps async on and off in both catalogs.
- Search it per M bucket rather than fixing it: in AITER's gfx950 fp8 selections, async copy is on in 17 of 72 rows at M<=16 but in 39 of 50 at M 1025-4096 and 65 of 77 above 4096.
- When async copy is on, drop the ds_write group from any hand schedule: A lands in LDS without passing through VGPRs.
- For an authored kernel, size the copy to the hardware: direct global-to-LDS moves up to 32 bits per lane on CDNA3 and up to 128 bits on CDNA4 (background), which is why the gfx942 A8 path copies 4 bytes.

#### Why this is a candidate

- The same word — async copy — names an architecture fact in one family and a searched knob in the other; reading one family's gate as the other's rule removes a live candidate.
- The copy width is written in the A8 compile entry point, and the sweep values are module constants in its tune catalog.

#### Keep as alternatives

- Synchronous global-to-register-to-LDS staging, which is the gfx942 HGEMM path and the async=0 arm of the A8 sweep.

#### Evidence

- `src_c4c19187bcb7933c` — `arch_gate` `gfx942` at `aiter/ops/flydsl/gemm_kernels.py:41`
- `src_a0f1066a6a2d6667` — `config_validity` `Current kernel fixes async_copy from the active GPU architecture;` at `aiter/ops/flydsl/gemm_kernels.py:123-126`
- `src_8f59fe9a6d65aaf3` — `async_copy` `ASYNC_COPY` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:132`
- `src_14bff946c5d70663` — `async_copy` `use_async_copy` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:136`
- `src_42146c2d073d655e` — `arch_gate` `gfx942` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:243`
- `src_623470961bcb8c24` — `config_space` `_ASYNC_COPY_VALS` at `aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py:248`
- `src_6dd333649bf42b12` — `scheduling` `sched_dswr` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:1291`
- `cfg_909edfc582fce0bd` — FlyDSL `a8_preshuffle` `af8_wf8_b16` selections in `a8w8_bpreshuffle_tuned_gemm` on `gfx950`/256 CUs, `M<=16`: 72 shipped row(s)
- `cfg_496162f129968cb9` — FlyDSL `a8_preshuffle` `af8_wf8_b16` selections in `a8w8_bpreshuffle_tuned_gemm` on `gfx950`/256 CUs, `M1025-4096`: 50 shipped row(s)
- `cfg_9636311ea3a3ce0d` — FlyDSL `a8_preshuffle` `af8_wf8_b16` selections in `a8w8_bpreshuffle_tuned_gemm` on `gfx950`/256 CUs, `M>4096`: 77 shipped row(s)
- `cfg_973fe63f897809df` — FlyDSL `hgemm` `abf16_wbf16_bf16` selections in `bf16_tuned_gemm` on `gfx950`/256 CUs, `M<=16`: 86 shipped row(s)

#### Background (always-on reference docs, not measurements)

- [`hardware/cdna3_mi300/memory_hierarchy.md`](../hardware/cdna3_mi300/memory_hierarchy.md)
- [`hardware/cdna4_mi350/memory.md:51-55`](../hardware/cdna4_mi350/memory.md)
- [`languages/flydsl/deep.md:65-73`](../languages/flydsl/deep.md)
- [`languages/flydsl/knobs.md:71-75`](../languages/flydsl/knobs.md)
- [`languages/flydsl/authoring_gemm_levers.md:119-131`](../languages/flydsl/authoring_gemm_levers.md)
- [`languages/flydsl/authoring_tile_programming.md:396-426`](../languages/flydsl/authoring_tile_programming.md)

#### Limits

- Selection shares show where AITER's tuner chose async copy, not by how much it helped.
- The HGEMM rule is tied to the current kernel; an authored HGEMM can stage however the hardware allows.

### Which global-to-LDS and LDS-to-register helpers do the FlyDSL A8 and MoE GEMMs share, and which access widths do they fix?

**Card:** `flydsl-a8-data-movement-helpers` · **type:** `performance_candidate` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `memory_access`, `operand_layout`, `pipeline_schedule`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `scaled_quant_gemm`, `grouped_gemm_moe`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`
- `requires`: `lds_staging`

#### Use when

- target_language=flydsl
- an operand is staged through LDS and the other is read preshuffled from global memory, as in the A8 and MoE kernels

#### Try

- Move A at one width end to end: the MoE kernels take the largest of 16, 8 or 4 bytes that divides each thread's share of the tile (fp16/bf16 must divide by 16), load that many bytes from global (`buffer_copy_gmem16_dwordx4` for 16, raw dword loads for 8 and 4) and store them with the matching XOR16 helper — `lds_store_16b_xor16`, `lds_store_8b_xor16` or `lds_store_4b_xor16`. The A8 GEMM uses the same 16-byte global copy and writes LDS inline, or skips the register hop with async copy (flydsl-async-copy-by-family).
- Read A back from LDS one 8-byte K-pack per K32 MFMA step through the same XOR16 swizzle the store used (`lds_load_pack_k32`: an 8-byte load, or a 16-byte load split in halves); the GEMM and MoE kernels wrap it in a local 64-K variant, `lds_load_packs_k64`.
- Never stage B: read it preshuffled straight from global into MFMA packs (`load_b_pack_k32`, `load_b_packs_k64`, `load_b_tile`) in the layout `make_preshuffle_b_layout` defines — K-packs of 8 or 16 bytes over 1- or 2-byte elements; it raises on anything else. The scales have their own preshuffled layout (`make_preshuffle_scale_layout`).
- Reuse the layer rather than rewriting it: the 2-stage MoE kernels import the same pipeline module and write their B loads the same way.

#### Why this is a candidate

- These helpers are the data-movement half of the A8 family, named by the widths they fix; a FlyDSL fp8 kernel built on them inherits a movement path AITER already ships.

#### Keep as alternatives

- Stage B through LDS as well (the HGEMM's `b_to_lds`), paying LDS capacity for an unshuffled weight.

#### Evidence

- `src_54e4ca3a8185528d` — `access_width` `buffer_copy_gmem16_dwordx4` at `aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py:467-489`
- `src_f419311ce8a02db3` — `access_width` `lds_store_16b_xor16` at `aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py:492-516`
- `src_c77d2ddf58185db3` — `access_width` `lds_store_8b_xor16` at `aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py:519-543`
- `src_f735e442dfbc314e` — `access_width` `lds_store_4b_xor16` at `aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py:546-570`
- `src_498bdd55bb2b600e` — `access_width` `lds_load_pack_k32` at `aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py:573-604`
- `src_aa04a6d4ea345675` — `access_width` `load_b_pack_k32` at `aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py:346-444`
- `src_2a3468398e382ec9` — `access_width` `load_b_packs_k64` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:525-550`
- `src_53b921a28e9cd675` — `access_width` `load_b_tile` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:552-581`
- `src_36c3ee38d337b470` — `layout_preshuffle` `make_preshuffle_b_layout` at `aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py:159-223`
- `src_11e30e2c665aafa7` — `layout_preshuffle` `make_preshuffle_scale_layout` at `aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py:96-148`
- `src_1c0548a53c4cf22e` — `config_validity` `kpack_bytes must be 8 or 16, got {kpack_bytes!r}` at `aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py:175`
- `src_2bc956feedfd346d` — `config_validity` `elem_bytes must be 1 or 2, got {elem_bytes!r}` at `aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py:183`
- `src_14d7a32aaedf7288` — `access_width` `load_b_tile` at `aiter/ops/flydsl/kernels/moe_gemm_2stage.py:639-692`
- `src_6d43092520bc779c` — `config_validity` `[fp16] bytes_per_thread_x ({bytes_per_thread_x}) must be divisible by …` at `aiter/ops/flydsl/kernels/moe_gemm_2stage.py:442-444`
- `src_0e09838af93de4ba` — `config_validity` `bytes_per_thread_x ({bytes_per_thread_x}) must be divisible by 4 to us…` at `aiter/ops/flydsl/kernels/moe_gemm_2stage.py:454-456`

#### Background (always-on reference docs, not measurements)

- [`languages/flydsl/authoring_gemm_levers.md:119-131`](../languages/flydsl/authoring_gemm_levers.md)
- [`optimization/vectorization_and_coalescing.md:33-47`](../optimization/vectorization_and_coalescing.md)

#### Limits

- The helpers are the A8/MoE family's; the HGEMM stages its operands with its own code, and a block-scale kernel adds scale loads these helpers do not cover.
- Which width is fastest for a tile is a measurement; the helpers state which widths are legal.

### Which operand loads and output stores should carry a streaming (non-temporal) cache policy, and which should keep the default?

**Card:** `gemm-streamed-operand-cache-policy` · **type:** `performance_candidate` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `memory_access`, `tunable_surface`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `scaled_quant_gemm`, `grouped_gemm_moe`, `batched_gemm`, `splitk_streamk_gemm`, `skinny_gemv_decode`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`

#### Use when

- target_language=flydsl
- one operand is read once per call while the other is reused across tiles — typically the weight B at decode, where every workgroup streams its own slice of W while the few rows of A are shared — or an output is written once and not read back by this kernel

#### Try

- Make the cache policy a per-operand knob on the global loads, not a kernel-wide switch: FlyDSL's `buffer_ops.buffer_load` takes `cache_modifier`, and AITER's FlyDSL MoE GEMM exposes it for its B loads only, as `b_nt` (default 0 in one of its kernels and 2 in the other), while its other loads pass `cache_modifier=0`.
- Search the streamed operand's load policy — the default against the streaming setting — and leave the reused operand's loads at the default.
- Store a write-once output non-temporally when nothing re-reads it soon: the FlyDSL MoE kernel writes its output with `llvm.StoreOp(..., nontemporal=True)`.

#### Why this is a candidate

- AITER's Triton a16w16 GEMM makes the same choice per operand: `cache_modifier` is a kernel parameter applied to the B loads only, never to A. Of the 306 shipped Triton GEMM config groups, 88 set it, and 87 of those choose `.cg`.
- A line fetched once and never re-read only displaces lines that would have been reused; the reused operand is the one worth keeping resident.

#### Keep as alternatives

- The default cache policy everywhere — the control arm.

#### Same question in other implementations

- **triton** — `cache_modifier: tl.constexpr` on the a16w16 kernels, passed to `tl.load(b_ptrs, cache_modifier=...)` for B and never for A; the value is carried per bucket in the shipped configs. Evidence: `src_15cea189ebb33d4c` (`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16.py:81`), `src_92959715207084c1` (`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16.py:159`), `src_9667faa67584ba59` (`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16.py:168`)

#### Evidence

- `src_6a42df54a6f6b458` — `cache_policy` `cache` at `aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py:71`
- `src_e93e7dd39f7adedc` — `cache_policy` `b` at `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py:893`
- `src_106bcf16a01dfc38` — `cache_policy` `b` at `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py:3553`
- `src_e59fe6013e3decd4` — `cache_policy` `nontemporal=True` at `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py:2284`

#### Background (always-on reference docs, not measurements)

- [`hardware/shared/hbm_infinity_fabric.md:73`](../hardware/shared/hbm_infinity_fabric.md)
- [`hardware/cdna3_mi300/memory_hierarchy.md:48`](../hardware/cdna3_mi300/memory_hierarchy.md)

#### Limits

- The spelling differs per backend — a Triton modifier string, a FlyDSL integer `cache_modifier`, an LLVM `nontemporal` store flag — and the corpus does not map one onto another; read what the target API accepts.
- Whether streaming helps depends on the reuse the benchmark reproduces: a timing loop that re-reads the same weight from a warm cache measures a different reuse than serving, where each layer reads its own weight.

## 6. MFMA / dot compute

- `compute_instruction` — Which matrix-compute family and instruction shape does the implementation request?

Also bears on this step: `flydsl-gfx942-path-gates` (step 2), `flydsl-splitk-hot-loop-schedule-bundle` (step 7), `flydsl-a8-interleaved-schedule` (step 7), `flydsl-mfma-fragment-lane-layout` (step 4).

### Which scale granularity does the FlyDSL A8 kernel implement, and what changes for a 128x128 block-scale contract?

**Card:** `flydsl-a8-scale-granularity-contract` · **type:** `constraint` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `compute_instruction`, `epilogue`, `runtime_contract`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `scaled_quant_gemm`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`
- `dtypes`: `fp8_e4m3fnuz`, `fp8_e4m3fn`, `fp8_e4m3fnuz_blockscale`, `fp8_e4m3fn_blockscale`, `int8`

#### Use when

- target_language=flydsl
- fp8/int8 GEMM whose math contract names its scale granularity (per-tensor, per-token/per-channel, or [128,128] blocks)

#### Try

- Read the A8 family as per-token x per-channel: for fp8/int8 it loads one fp32 scale per output row of A and one per output column of B at the last tile and applies them in the epilogue. FP4 is its only block-scaled path (per-32 MX scales on gfx950).
- Do not reuse it unmodified for a [128,128] block-scale contract: those scales change every 128 elements of K, so they have to be applied per K block inside the loop (promote the partial to fp32 per block, multiply, accumulate), not once at the end — the loop body is gemm-blockscale-k-loop-scale-application.
- Do not fold arbitrary fp32 block scales into the gfx950 scaled MFMA: its scale operand is E8M0 (a power of two), so a general fp32 scale is rounded and parity fails (background).
- Settle the math contract against the task's oracle before timing anything: a per-token kernel measured against a block-scale oracle fails parity, it does not lose on speed.

#### Why this is a candidate

- The A8 compile entry point decides `_needs_per_token_scale` from the dtype and loads the row and column scales only for the last tile, which is the per-token contract written down.
- AITER's block-scale databases contain no FlyDSL row at all — every block-scale selection went to CK, CK-Tile or ASM — so there is no shipped FlyDSL block-scale kernel to copy.

#### Keep as alternatives

- For block scales, keep the A8 kernel's MFMA, staging and schedule as the precedent and change only where the scale is applied.

#### Same question in other implementations

- **triton** — one kernel per granularity, and the difference is where the multiply sits: the per-token kernel scales the finished accumulator once after the K loop (`accumulator *= a_scale[:, None] * b_scale[None, :]`), the block-scale kernel and its preshuffled variant scale every K block's `tl.dot` partial inside the loop. Evidence: `src_7a0ec7ada35d7647` (`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8w8_per_token_scale.py:177`), `src_426b43e7e6389512` (`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8w8_blockscale.py:190-194`), `src_359d07953d540490` (`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8w8_blockscale.py:407-411`)
- **gluon** — the fp8 block-scale kernel stages the A and B scales through LDS with their own shared layouts, next to the operands, and multiplies each MFMA partial by them inside the loop. Evidence: `src_1ba1837b9ca062ee` (`aiter/ops/triton/gluon/gemm_a8w8_blockscale.py:134-136`), `src_927f1e26846a58fd` (`aiter/ops/triton/gluon/gemm_a8w8_blockscale.py:137-139`), `src_db2ae65ef56818a7` (`aiter/ops/triton/gluon/gemm_a8w8_blockscale.py:301`)

#### Evidence

- `src_f854f40de1f28585` — `scale_operand` `_needs_per_token_scale` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:403`
- `src_1255666f8d08f366` — `mfma_intrinsic` `mfma_scale_f32_16x16x128_f8f6f4` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:934`
- `sel_b7d6386940696a48` — backend selected by AITER's tuner in `a8w8_blockscale_bpreshuffle_tuned_gemm` on `gfx950`/256 CUs, `M<=16`: ck 223 · cktile 12 · asm 6 (rows)
- `sel_8b543c4558078855` — backend selected by AITER's tuner in `a8w8_blockscale_tuned_gemm` on `gfx950`/256 CUs, `M>4096`: ck 178 · cktile 147 (rows)
- `sel_f1384cc1069c9823` — backend selected by AITER's tuner in `a8w8_blockscale_bpreshuffle_tuned_gemm` on `gfx942`/80 CUs, `M>4096`: asm 32 · ck 6 (rows)

#### Background (always-on reference docs, not measurements)

- [`operators/dense_gemm/numerics.md:36-52`](../operators/dense_gemm/numerics.md)
- [`quantization/block_scaling_mxfp.md:40-49`](../quantization/block_scaling_mxfp.md)
- [`operators/scaled_quant_gemm/backends/flydsl.md:96-103`](../operators/scaled_quant_gemm/backends/flydsl.md)

#### Limits

- The per-K-block promote is the arithmetic a block-scale contract requires; its cost on a given shape is a measurement, and measured recipes live behind the expert-skill switch.
- The E8M0 limitation is a hardware fact from the background docs, not from this source.

### Which MFMA does a FlyDSL quantized GEMM issue per dtype and architecture, and what does that fix about tile_k?

**Card:** `flydsl-a8-mfma-selection` · **type:** `semantic` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `compute_instruction`, `architecture_capability`, `workgroup_tile`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `scaled_quant_gemm`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`
- `dtypes`: `fp8_e4m3fnuz`, `fp8_e4m3fn`, `fp8_e4m3fnuz_blockscale`, `fp8_e4m3fn_blockscale`, `int8`, `int4`, `fp4_e2m1`, `mxfp4`

#### Use when

- target_language=flydsl
- quantized GEMM (fp8, int8/int4, fp4)

#### Try

- fp8: issue `mfma_f32_16x16x32_fp8_fp8` (K=32 per instruction); on gfx95x the A8 kernel switches fp8 to the block-scaled `mfma_scale_f32_16x16x128_f8f6f4` (K=128 per instruction).
- int8, and int4 widened to int8: issue `mfma_i32_16x16x32i8` and keep i32 accumulation until the scale is applied.
- f16 / bf16 inside the same kernel: `mfma_f32_16x16x16f16` / `mfma_f32_16x16x16bf16_1k`.
- fp4 (gfx950 only): the same x128 scaled MFMA with its operand-format codes set for FP4 and M/N/K packed by two.
- Derive tile_k from the instruction K: a tile_k that is not a multiple of 128 cannot take the gfx950 scaled path, which is why the gfx950 A8 catalog has no tile_k=64 tiles.
- When porting between architectures, keep the fp8 format per arch in view: gfx942 is FNUZ e4m3 (bias 8, max 240) and gfx950 is OCP e4m3 (bias 7, max 448), so FNUZ bytes bit-copied onto gfx950 are off by a factor of two (background).

#### Why this is a candidate

- The dtype branches and the gfx95 switch are explicit in one function, and the instruction K is what sets the catalog's tile_k granularity.
- The instruction choice changes the loop trip count, the scale plumbing and the legal tiles together, so it is the first decision rather than a late tuning knob.

#### Keep as alternatives

- In an authored kernel, keep the unscaled K=32 fp8 MFMA as a control arm on gfx950 when the math contract does not need the hardware scale operand.

#### Same question in other implementations

- **triton** — `tl.dot` over fp8 tiles inside the block-scale K loop; the compiler selects the fp8 MFMA. Evidence: `src_f4a3d14442cafc0c` (`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8w8_blockscale.py:191`)
- **gluon** — an explicit `instr_shape=[16, 16, 32]` MFMA layout in the fp8 block-scale kernel. Evidence: `src_cf70b9a5bae73954` (`aiter/ops/triton/gluon/gemm_a8w8_blockscale.py:123`)

#### Evidence

- `src_113ea8d0f42a693d` — `mfma_intrinsic` `mfma_f32_16x16x32_fp8_fp8` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:1009`
- `src_1255666f8d08f366` — `mfma_intrinsic` `mfma_scale_f32_16x16x128_f8f6f4` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:934`
- `src_84fa26c5447f28f7` — `arch_gate` `gfx95` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:848`
- `src_bc08df5300de01f3` — `config_validity` `tile_k must be divisible by 128 for mfma_scale_x128, got tile_k={tile_…` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:855-857`
- `src_f9f2544162138ea2` — `mfma_intrinsic` `mfma_i32_16x16x32i8` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:215-217`
- `src_a5d9e08624d3f251` — `mfma_intrinsic` `mfma_i32_16x16x32i8` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:220`
- `src_722ebc17457eec5c` — `mfma_intrinsic` `mfma_f32_16x16x16f16` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:1005`
- `src_d9a92f8642e73d56` — `mfma_intrinsic` `mfma_f32_16x16x16bf16_1k` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:1007`
- `src_011cf8d1222b7906` — `arch_gate` `gfx950` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:254`
- `src_273e37ba3a8cefa8` — `config_space` `_base_tiles_lds2_950_extra` at `aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py:227-229`

#### Background (always-on reference docs, not measurements)

- [`hardware/shared/dtype_numerics.md:40-49`](../hardware/shared/dtype_numerics.md)
- [`hardware/cdna4_mi350/matrix_core_blockscale.md`](../hardware/cdna4_mi350/matrix_core_blockscale.md)
- [`quantization/fnuz_vs_ocp.md`](../quantization/fnuz_vs_ocp.md)
- [`operators/scaled_quant_gemm/backends/flydsl.md:30-58`](../operators/scaled_quant_gemm/backends/flydsl.md)

#### Limits

- This is instruction selection in AITER's vendored kernel; it says nothing about which instruction is faster for a shape.
- The FNUZ/OCP statement comes from the background hardware docs, not from this source.

### How does a 128x128 block-scale GEMM apply its scales inside the K loop, and what does that fix about tile_k and the scale indexing?

**Card:** `gemm-blockscale-k-loop-scale-application` · **type:** `semantic` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `compute_instruction`, `workgroup_tile`, `runtime_contract`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `scaled_quant_gemm`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`
- `dtypes`: `fp8_e4m3fnuz_blockscale`, `fp8_e4m3fn_blockscale`

#### Use when

- target_language=flydsl
- the math contract is Y = (X*x_scale) @ (W*w_scale)^T with one fp32 scale per row of X per 128 elements of K and one per 128x128 block of W

#### Try

- Scale every K block's partial before it joins the accumulator: for each 128-wide K block, take the tile's fp32 MFMA partial, multiply it by the outer product of that block's per-row X scales and per-column-group W scales, and add it into a separate fp32 accumulator. Triton writes the loop body as `accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]`; Gluon as `acc += mfma_out * cur_a_scale[:, None] * cur_b_scale[None, :]`.
- Make one K tile exactly one scale block. The Triton kernel takes GROUP_K and BLOCK_SIZE_K as separate parameters, its docstring requires them equal, and every gfx942 configuration AITER ships sets BLOCK_SIZE_K = GROUP_K = GROUP_N = 128 — on gfx942 that is four 16x16x32 fp8 MFMA K-steps between scale multiplies.
- Index the scales per K block: X's scale is [M, ceil(K/128)] and advances one column per K tile; W's holds one value per 128x128 block, so each output column reads the scale of its 128-wide group for the current K block (Triton gathers it as `offs_bn // GROUP_N`).
- Keep the MFMA unscaled and the scale in software: on gfx942 issue the plain fp8 MFMA (flydsl-a8-mfma-selection); on gfx950 do not fold arbitrary fp32 block scales into the scaled MFMA, whose scale operand is E8M0 (flydsl-a8-scale-granularity-contract, background).
- If K is split across workgroups, start every split on a scale-block boundary: Triton offsets the scale column by the split start divided by GROUP_K, and every SPLITK_BLOCK_SIZE it ships for gfx942 (512 to 2048) is a multiple of 128.

#### Why this is a candidate

- Every block-scale kernel AITER ships multiplies per K block inside the loop, and its per-token kernel multiplies once after it: the placement is the math contract, not an optimisation.
- AITER has no FlyDSL block-scale kernel (flydsl-a8-scale-granularity-contract), so a FlyDSL author writes this loop body from these precedents.

#### Keep as alternatives

- Stage the scales through LDS with their own layouts, next to the operands, as the Gluon kernel does; the background pitfall is feeding the global scale layout straight to the MFMA.
- The same arithmetic sits inside CK's A/B-scale device operation and ASM's block-scale kernels, as template or binary configuration rather than a loop you write.

#### Same question in other implementations

- **gluon** — scales staged through LDS with their own shared layouts; each MFMA partial multiplied by the current block's scales inside the loop. Evidence: `src_db2ae65ef56818a7` (`aiter/ops/triton/gluon/gemm_a8w8_blockscale.py:301`), `src_1ba1837b9ca062ee` (`aiter/ops/triton/gluon/gemm_a8w8_blockscale.py:134-136`), `src_927f1e26846a58fd` (`aiter/ops/triton/gluon/gemm_a8w8_blockscale.py:137-139`)
- **ck** — an A/B-scale device operation (`DeviceGemmMultiD_ABScale_Xdl_CShuffle_V3`), plus a block-scale B-preshuffle variant. Evidence: `src_73c6170a2792b675` (`csrc/ck_gemm_a8w8_blockscale/include/gemm_a8w8_blockscale_common.cuh:98`), `src_6352032268991afb` (`csrc/ck_gemm_a8w8_blockscale_bpreshuffle/include/gemm_a8w8_blockscale_bpreshuffle_common.cuh:93`)
- **asm** — precompiled block-scale kernels with B preshuffled — the same six on gfx942 and gfx950, every one tile_n = 128 (one W scale group per tile) with tile_m 32 to 128 and split-capable; the launcher picks tile_m and a split per shape and multiplies the grid by the split. Evidence: `src_381f987b35a586b7` (`csrc/py_itfs_cu/asm_a8w8_blockscale_bpreshuffle.cu:116`), `src_b07a8dc43c913b9d` (`csrc/py_itfs_cu/asm_a8w8_blockscale_bpreshuffle.cu:353`), `src_3865c44e30f36de9` (`hsa/gfx942/fp8gemm_blockscale/fp8gemm_bf16_blockscale.csv:2`), `src_fcf5d73d12271d55` (`hsa/gfx942/fp8gemm_blockscale/fp8gemm_bf16_blockscale.csv:3`), `src_8866d3dbeb612f3a` (`hsa/gfx950/fp8gemm_blockscale/fp8gemm_bf16_blockscale.csv:2`)

#### Evidence

- `src_426b43e7e6389512` — `scale_in_k_loop` `accumulator, a_scale` at `aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8w8_blockscale.py:190-194`
- `src_359d07953d540490` — `scale_in_k_loop` `accumulator, a_scale` at `aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8w8_blockscale.py:407-411`
- `src_7a0ec7ada35d7647` — `scale_after_k_loop` `accumulator, a_scale, b_scale` at `aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8w8_per_token_scale.py:177`
- `src_4991e94450b3832a` — `tunable_param` `GROUP_K` at `aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8w8_blockscale.py:73`
- `src_4f4e8313f9775c89` — `tunable_param` `BLOCK_SIZE_K` at `aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8w8_blockscale.py:78`
- `cfg_d2642e1347a53ea1` — Triton seed `G E M M   A 8 W 8 _ B L O C K S C A L E` on `gfx942`, `M_LEQ_8` (see [`gemm_triton_seeds.md`](gemm_triton_seeds.md))
- `cfg_3eff28954f18b5fe` — Triton seed `G E M M   A 8 W 8 _ B L O C K S C A L E` on `gfx942`, `any` (see [`gemm_triton_seeds.md`](gemm_triton_seeds.md))
- `sel_f7e394faa23e37e6` — backend selected by AITER's tuner in `a8w8_blockscale_bpreshuffle_tuned_gemm` on `gfx942`/80 CUs, `M<=16`: ck 16 · asm 2 (rows)
- `sel_f1384cc1069c9823` — backend selected by AITER's tuner in `a8w8_blockscale_bpreshuffle_tuned_gemm` on `gfx942`/80 CUs, `M>4096`: asm 32 · ck 6 (rows)

#### Background (always-on reference docs, not measurements)

- [`operators/dense_gemm/numerics.md:36-52`](../operators/dense_gemm/numerics.md)
- [`operators/scaled_quant_gemm/tuning.md:50-57`](../operators/scaled_quant_gemm/tuning.md)
- [`operators/dense_gemm/numerics.md:59-63`](../operators/dense_gemm/numerics.md)

#### Limits

- The per-block multiply is VALU work on every K block; whether it hides behind the MFMAs is a measurement.
- These are Triton and Gluon spellings; FlyDSL expresses the same loop with its own MFMA, vector and conversion ops.
- Triton's scale strides are its wrapper's layout choice; take the x_scale / w_scale shapes from the task's math contract rather than from Triton.

### Which legal MFMA call forms can seed a FlyDSL f16/bf16 GEMM?

**Card:** `flydsl-half-mfma-call-forms` · **type:** `performance_candidate` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `compute_instruction`, `architecture_capability`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `grouped_gemm_moe`, `batched_gemm`, `splitk_streamk_gemm`, `skinny_gemv_decode`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`
- `dtypes`: `f16`, `bf16`

#### Use when

- operator_family=gemm
- target_language=flydsl
- input_dtype in {f16,bf16}

#### Try

- For an m16n16k16 candidate, pair bf16 with mfma_f32_16x16x16bf16_1k and f16 with mfma_f32_16x16x16f16.
- Check the target architecture before treating the K shape as a free axis: the shipped split-K kernel does not let you pick it. On gfx942 it selects m16n16k16 with 4-byte DMA and two MFMA steps per warp-K; on every other architecture it selects m16n16k32 with 16-byte DMA and one. Benchmarking the K=32 form on gfx942 means rewriting that branch, not passing a flag.
- Also benchmark the m16n16k32 bf16/f16 forms where the architecture branch does select them and the fragment shape and register budget allow.
- Keep 16x16 fragments as the default shape and reach for 32x32 only when the tile needs it: a 16x16 accumulator block costs 4 VGPRs per lane, a 32x32 block 16, at the same nominal rate (background).

#### Why this is a candidate

- AITER's FlyDSL split-K GEMM implements both K=16 and K=32 classes and branches explicitly on bf16 versus f16.
- The two classes are not offered side by side at runtime: one architecture test picks the WMMA implementation, the DMA width and the MFMA-per-warp-K count together, as one bundle.
- This establishes legal implementation precedents; it does not establish which form is faster for a new shape.

#### Keep as alternatives

- Use the m16n16k16 form when the larger-K fragment does not fit the surrounding tile or register budget.
- Let a higher-level backend select the instruction when direct MFMA control is not required.

#### Same question in other implementations

- **triton** — `tl.dot` over the tile; the compiler picks the MFMA, steered by `matrix_instr_nonkdim` in the shipped config. Evidence: `src_cb8f85060eec45f8` (`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16.py:170`)
- **gluon** — an explicit MFMA layout with `instr_shape` (e.g. [16, 16, 32] or [32, 32, 16]) chosen per kernel. Evidence: `src_cf70b9a5bae73954` (`aiter/ops/triton/gluon/gemm_a8w8_blockscale.py:123`), `src_422867b3bf9dfa13` (`aiter/ops/triton/_gluon_kernels/gfx942/moe/moe_op_gemm_int8_smoothquant.py:88`)
- **ck** — an XDL device-op template; the MFMA shape is a positional template argument the corpus does not decode. Evidence: `src_780fadebcc954d06` (`csrc/ck_batched_gemm_bf16/include/batched_gemm_bf16_common.cuh:86`)

#### Evidence

- `src_24517333d1cd6722` — `mfma_intrinsic` `mfma_f32_16x16x16bf16_1k` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:56-58`
- `src_0ed3b5780c589479` — `mfma_intrinsic` `mfma_f32_16x16x16f16` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:59-61`
- `src_387e95ce58e0449b` — `mfma_intrinsic` `mfma_f32_16x16x32_bf16` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:77-79`
- `src_79cfb57473ed817f` — `mfma_intrinsic` `mfma_f32_16x16x32_f16` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:80-82`
- `src_e689c0cbe1b4f8e8` — `arch_gate` `gfx942` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:128-137`

#### Background (always-on reference docs, not measurements)

- [`hardware/shared/matrix_core_mfma_smfmac.md`](../hardware/shared/matrix_core_mfma_smfmac.md)
- [`hardware/cdna4_mi350/matrix_core_blockscale.md`](../hardware/cdna4_mi350/matrix_core_blockscale.md)
- [`languages/flydsl/authoring_tile_programming.md:201-277`](../languages/flydsl/authoring_tile_programming.md)
- [`languages/flydsl/authoring_tile_programming.md:503-518`](../languages/flydsl/authoring_tile_programming.md)

#### Limits

- No latency or occupancy comparison is attached; treat both K shapes as candidates and re-measure.
- API spelling is version-sensitive; consult languages/flydsl/version_map.md before copying.

## 7. Pipeline and synchronization

- `pipeline_schedule` — How are loads, LDS buffering and matrix instructions pipelined and ordered?

Also bears on this step: `flydsl-hgemm-tiling-validity` (step 3), `flydsl-async-copy-by-family` (step 5), `flydsl-a8-preshuffle-config-space` (step 3), `flydsl-a8-data-movement-helpers` (step 5).

### What explicit instruction-order bundle can seed a hand-scheduled FlyDSL split-K GEMM hot loop?

**Card:** `flydsl-splitk-hot-loop-schedule-bundle` · **type:** `performance_candidate` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `pipeline_schedule`, `workgroup_tile`, `execution_geometry`, `memory_access`, `compute_instruction`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `scaled_quant_gemm`, `grouped_gemm_moe`, `splitk_streamk_gemm`, `skinny_gemv_decode`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`
- `requires`: `split_k_hot_loop`

#### Use when

- operator_family=gemm
- target_language=flydsl
- the hot loop is hand-scheduled with ROCm scheduling directives (a split-K or tiled K loop with both operands in LDS)

#### Try

- Copy the ORDER, not a set of magic numbers: A-fragment ds-reads, then B-fragment ds-reads, then the A and B global loads as vmem, then the MFMA issues, closed by a scheduling barrier.
- Derive each group's count from the loop geometry that produces it — warp K steps times warp M or N steps for the ds-reads, the per-block LDG register counts for the vmem groups, and the full MFMA count for the mfma group — rather than hard-coding the counts observed here.
- Move or retune the bundle as a unit against the exact load/MFMA loop, then compare with the unscheduled control.

#### Why this is a candidate

- The directives are adjacent inside one scheduler function and describe a single ordering pattern; copying one directive in isolation loses that context.
- An earlier version of this card recorded the literal group sizes vmem(2)/dsrd(2)/dsrd(4)/mfma(8). Upstream now emits unit-count directives inside geometry-derived loops, so those numbers described one build of one tile and expired quietly. The order survived the rewrite; the constants did not, which is the reason this card states a rule and not a bundle of integers.

#### Keep as alternatives

- Leave instruction ordering to the compiler; that is the control arm and it is often the right answer.
- Use the A8 family's interleaved scheduler instead, which spreads ds-reads and vmem across the MFMAs one at a time rather than emitting the groups back to back — see flydsl-a8-interleaved-schedule.

#### Same question in other implementations

- **ck** — a named loop scheduler (`BlockGemmPipelineScheduler::Intrawave` / `Interwave`) plus a pipeline version chosen per template instance. Evidence: `src_75b377d9dfc10083` (`csrc/ck_gemm_a4w4_blockscale/include/gemm_a4w4_blockscale_common.cuh:68`), `src_bb0d15e277ff0a86` (`csrc/ck_gemm_a4w4_blockscale/include/gemm_a4w4_blockscale_common.cuh:69`), `src_6a2107cadf863683` (`csrc/ck_gemm_a4w4_blockscale/include/gemm_a4w4_blockscale_common.cuh:85`)

#### Evidence

- `src_5da6811a509d9806` — `scheduling` `sched_dsrd` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:688`
- `src_b0da890c19bc671e` — `scheduling` `sched_dsrd` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:690`
- `src_ed1134f4e4cea08c` — `scheduling` `sched_vmem` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:694`
- `src_86d7e5af1117454e` — `scheduling` `sched_vmem` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:698`
- `src_0f18d7f50ce33cbb` — `scheduling` `sched_mfma` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:702`
- `src_dca1b81d9d2b2edf` — `scheduling` `sched_barrier` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:703`

#### Background (always-on reference docs, not measurements)

- [`optimization/mfma_scheduling.md`](../optimization/mfma_scheduling.md)
- [`languages/flydsl/authoring_gemm_levers.md:150-165`](../languages/flydsl/authoring_gemm_levers.md)

#### Limits

- Instruction scheduling is the most build-specific thing in this corpus; treat it as a candidate to try late, after tiling and split-K are settled.
- Changing tile geometry, LDS stages or access width changes the schedule that must be re-measured.

### How does the FlyDSL A8 kernel interleave LDS reads, global loads and MFMAs, and how does that differ between gfx942 and gfx950?

**Card:** `flydsl-a8-interleaved-schedule` · **type:** `performance_candidate` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `pipeline_schedule`, `memory_access`, `workgroup_tile`, `compute_instruction`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `scaled_quant_gemm`, `grouped_gemm_moe`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`
- `dtypes`: `fp8_e4m3fnuz`, `fp8_e4m3fn`, `int8`

#### Use when

- target_language=flydsl
- a K loop whose A tile goes through LDS and whose B is read straight from global memory
- instruction order is being pinned by hand

#### Try

- On gfx950 for fp8/int8, seed the preload depth from the family's per-tile table: `dsrd_preload` / `dvmem_preload` default to -1, which looks (tile_m, tile_n, tile_k) up in `_TILE_PRELOAD_TABLE` (values 2-8); every other architecture and dtype falls back to (0, 0).
- After the preload, issue one MFMA at a time and spread the remaining ds-reads and global loads evenly over the MFMA sequence (the kernel computes the per-MFMA counts with a ceiling-division distribution), then close with a sched barrier.
- Place the next tile's A ds-writes after the last ds-read of the current tile, and skip the ds-write group entirely when async copy writes A straight into LDS.
- On gfx942 the kernel uses a separate fixed pattern — vmem, a group of MFMAs, one ds-read, another group of MFMAs, with ds-writes near the tail — derived from num_acc_n and k_unroll; do not carry the gfx950 preload table over.
- Recompute the counts from your own geometry (LDS loads, global loads, MFMA total) whenever the tile, stage count or access width changes, instead of copying the table.

#### Why this is a candidate

- This is the scheduler of the FlyDSL family AITER selects most often in its gfx950 fp8 database, and both of its architecture branches are written out in one function.
- The preload table is keyed by tile, so it is a per-geometry fact rather than a constant that transfers.

#### Keep as alternatives

- The split-K HGEMM's grouped order — see flydsl-splitk-hot-loop-schedule-bundle.
- No hand schedule: leave ordering to the compiler as the control arm.

#### Evidence

- `src_62445650dbe2e65e` — `config_space` `_TILE_PRELOAD_TABLE` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:31-111`
- `src_d4d6210e665ad849` — `config_space` `_TILE_PRELOAD_DEFAULT` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:113`
- `src_d645d4f951f04c13` — `arch_gate` `gfx950` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:156-159`
- `src_43e09b045b1cc853` — `scheduling` `sched_vmem` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:1266`
- `src_a76d2aba228be108` — `scheduling` `sched_dsrd` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:1268`
- `src_69bc44a707a4ec5b` — `scheduling` `sched_mfma` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:1270`
- `src_5732b53b2a42edb6` — `scheduling` `sched_dsrd` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:1276`
- `src_47d88e0587130c78` — `scheduling` `sched_vmem` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:1284`
- `src_6dd333649bf42b12` — `scheduling` `sched_dswr` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:1291`
- `src_fc0ce347afaa09c3` — `scheduling` `sched_barrier` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:1299`
- `src_81f135c38346b8a5` — `scheduling` `sched_dsrd` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:1197`
- `src_be54e3472c40f00f` — `scheduling` `sched_mfma` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:1198`
- `src_5793d6410b67084b` — `scheduling` `sched_dswr` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:1225`
- `sel_7c8d8e410633ad1e` — backend selected by AITER's tuner in `a8w8_bpreshuffle_tuned_gemm` on `gfx950`/256 CUs, `M<=16`: flydsl 72 · ck 37 · cktile 2 (rows)
- `sel_1b5489e6d41507a3` — backend selected by AITER's tuner in `a8w8_bpreshuffle_tuned_gemm` on `gfx950`/256 CUs, `M>4096`: flydsl 77 · cktile 36 · ck 20 (rows)

#### Background (always-on reference docs, not measurements)

- [`optimization/mfma_scheduling.md`](../optimization/mfma_scheduling.md)
- [`optimization/memory_pipelining.md`](../optimization/memory_pipelining.md)
- [`languages/flydsl/authoring_gemm_levers.md:150-165`](../languages/flydsl/authoring_gemm_levers.md)

#### Limits

- The table values are AITER's per-tile choices for gfx950 fp8/int8; they are seeds for that geometry only.
- Scheduling is the most build-specific knowledge here; settle the tile and staging first.

### How many LDS stages should the K loop keep in flight, and what does each one cost?

**Card:** `flydsl-lds-pipeline-depth` · **type:** `performance_candidate` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `pipeline_schedule`, `configuration_constraints`, `workgroup_tile`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `scaled_quant_gemm`, `splitk_streamk_gemm`, `skinny_gemv_decode`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`
- `requires`: `lds_staging`

#### Use when

- target_language=flydsl
- the design stages operand tiles through LDS (trait lds_staging)

#### Try

- Treat the stage count as a knob that competes with the tile for LDS: the split-K HGEMM allocates `STAGES * BLOCK_M * BLOCK_K` elements for A and `STAGES * BLOCK_N * BLOCK_K` for B, so every extra stage costs one more A tile and one more B tile.
- Start from AITER's two stages — `STAGES = 2` in the split-K and small-M HGEMMs, `lds_stage=2` as the preshuffle GEMM's default — and try a deeper ring only where the tile is small enough for the LDS budget and the tiling validator to accept it.
- Keep the stage count in the kernel name, as AITER does (`_S{STAGES}`, `_lds{lds_stage}`), so two builds that differ only in depth never share a compiled-kernel cache entry.

#### Why this is a candidate

- A stage is how far the loader runs ahead of the MFMAs, and it is bought with LDS that the tile, occupancy and any LDS-staged epilogue also want.
- AITER fixes two stages in the HGEMMs and exposes the depth only on the preshuffle GEMM, where it also restricts it (FP4 with tile_k=128 supports only lds_stage=2): the depth is a design axis, not a constant.

#### Keep as alternatives

- More workgroups per CU instead of more stages: the same LDS spent on occupancy hides latency across workgroups rather than within one.

#### Evidence

- `src_dbdbdbf21af30e3c` — `lds_stage` `STAGES, 2` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:142`
- `src_8a036c537a13ff24` — `lds_stage` `STAGES, 2` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:72`
- `src_1a80adca750d9118` — `lds_stage` `lds_stage, 2` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:211`

#### Background (always-on reference docs, not measurements)

- [`optimization/lds_and_bank_conflicts.md:55-59`](../optimization/lds_and_bank_conflicts.md)
- [`optimization/lds_and_bank_conflicts.md:94-96`](../optimization/lds_and_bank_conflicts.md)
- [`operators/skinny_gemv_decode/backends/flydsl.md:59`](../operators/skinny_gemv_decode/backends/flydsl.md)

#### Limits

- The HGEMM families hard-code two stages; a different depth there means changing the kernel source, not passing a parameter.
- Deeper is not monotonically better: once global latency is hidden, further stages only take LDS away from occupancy.

## 8. Reduction, epilogue and write-back

- `epilogue` — How is the accumulator rearranged and written?

Also bears on this step: `flydsl-a8-scale-granularity-contract` (step 6), `gemm-split-k-when-grid-underfills` (step 3), `flydsl-a8-preshuffle-config-space` (step 3), `flydsl-mfma-fragment-lane-layout` (step 4), `flydsl-test-parity-gate` (step 10).

### How does the FlyDSL split-K HGEMM combine partial results, and what does that impose on the launcher?

**Card:** `flydsl-splitk-reduction-contract` · **type:** `constraint` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `epilogue`, `work_partition`, `runtime_contract`, `configuration_constraints`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `splitk_streamk_gemm`, `skinny_gemv_decode`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`

#### Use when

- target_language=flydsl
- split_k > 1 is a candidate, or a split-K kernel is being ported or written

#### Try

- Write the combine in the kernel as the HGEMM does, in three parts. First, only the split with k index 0 initialises the output tile — to zero, or to the bias when there is one, so the bias is added exactly once — then writes back L2 and raises the tile's counter to 1 (`zero_c`). Second, every split spins on that counter until it is non-zero before touching the output (`split_k_barrier`). Third, each split rounds its fp32 accumulator to the output dtype and adds it into C with `llvm.AtomicRMWOp(fadd)`, two elements per atomic, at agent scope (the `if const_expr(IS_SPLIT_K):` store block).
- The small-M family writes the same combine per N tile (a tile-indexed counter and the same packed fadd), so either kernel is a precedent for an authored split-K.
- Keep one global semaphore buffer per stream, sized SPLIT_K_SIGNAL_STATE_COUNT (3) x SPLIT_K_COUNTER_MAX_LEN (128) counters, and pass the current signal state into every launch; split 0 also clears the previous state's counters while the current ones are in use.
- Advance the per-stream signal state after a launch only when split_k > 1, cycling modulo 3, so consecutive split-K launches on one stream use different counter slices.
- Reject configurations whose output-tile count exceeds the counters: with split_k > 1, ceil(M/tile_m)*(N/tile_n) must be at most 128 — for large M this bounds how small tile_m can go while split-K is on.
- When porting, keep the combine bookkeeping in the host/launcher contract: a kernel copied without the semaphore allocation and the state rotation does not have the counters its reduction waits on.

#### Why this is a candidate

- The capacity check, the state constants and the rotation helpers are all module-level code in the entry point, so the contract is stated rather than implied.
- The kernel half — initialise, wait, atomically add — is three blocks in the kernel body, cited with their extents, so a port can copy the whole mechanism rather than its first line.
- The capacity bound interacts with the tile search: a split-K candidate that is legal by divisibility can still be rejected by the counter check.

#### Keep as alternatives

- split_k=1, which needs none of this.
- A two-pass combine — partial buffers plus a reduce kernel — as the Triton GEMM does.

#### Same question in other implementations

- **triton** — partial buffers per K split plus a reduce kernel (`NUM_KSPLIT`, `ACTUAL_KSPLIT`/`MAX_KSPLIT`), or `tl.atomic_add` into the output in the atomic variant. Evidence: `src_0f50a57ed622a4f4` (`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16.py:77`), `src_8f1fdafc50973349` (`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16.py:209`), `src_f5043ecafb832fa0` (`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16_atomic.py:55`), `src_9e9cdadae23d0f5f` (`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16_atomic.py:139`)
- **asm** — the split count is part of the selected object's config and becomes the launch grid's z dimension. Evidence: `src_05e16505bb238998` (`csrc/py_itfs_cu/asm_gemm_a16w16.cu:115`), `src_4044f249c79abf90` (`csrc/py_itfs_cu/asm_gemm_a16w16.cu:250`)

#### Evidence

- `src_1a4b9ad5c5b109de` — `split_k` `zero_c` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:296-391`
- `src_ad151ebc96795a5e` — `split_k` `split_k_barrier` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:393-432`
- `src_5ad1990942237536` — `split_k` `IS_SPLIT_K` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:615-616`
- `src_0294fc9392d3670a` — `split_k` `IS_SPLIT_K` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:830-905`
- `src_869d39483177e42c` — `atomic_combine` `fadd` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:875-882`
- `src_284881f3cc4c325a` — `split_k` `split_k_barrier` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:712-751`
- `src_39dac9712009eb08` — `atomic_combine` `fadd` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:966-973`
- `src_564badda94311abc` — `split_k` `SPLIT_K_SIGNAL_STATE_COUNT` at `aiter/ops/flydsl/gemm_kernels.py:38`
- `src_7c7ddd401ff4aaeb` — `config_limit` `SPLIT_K_COUNTER_MAX_LEN, 128` at `aiter/ops/flydsl/gemm_kernels.py:37`
- `src_194c17aa69828857` — `split_k` `SPLIT_K_GLOBAL_SEMAPHORE` at `aiter/ops/flydsl/gemm_kernels.py:66`
- `src_86aa6d1400d0f17f` — `split_k` `_advance_split_k_signal_state` at `aiter/ops/flydsl/gemm_kernels.py:688-692`
- `src_94058066019b30f2` — `split_k` `_get_split_k_signal_state` at `aiter/ops/flydsl/gemm_kernels.py:683-685`
- `src_5c7e74ddb8a071de` — `config_validity` `_check_split_k_counter_capacity` at `aiter/ops/flydsl/gemm_kernels.py:695-707`
- `src_679a944e53f65823` — `config_validity` `Split-K counter capacity exceeded:` at `aiter/ops/flydsl/gemm_kernels.py:704-707`
- `src_67a72909c71307d1` — `split_k` `SPLIT_K_SIGNAL_STATE_COUNT` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:24`
- `src_217562a4a9b879dc` — `parity_tolerance` `pass_pct, 99.0` at `aiter/ops/flydsl/test_flydsl_splitk_hgemm.py:85`
- `src_636fbc44252bb90e` — `parity_tolerance` `max_delta_limit, 32.0` at `aiter/ops/flydsl/test_flydsl_splitk_hgemm.py:86`

#### Background (always-on reference docs, not measurements)

- [`operators/splitk_streamk_gemm/numerics.md`](../operators/splitk_streamk_gemm/numerics.md)
- [`operators/splitk_streamk_gemm/backends/flydsl.md:77-85`](../operators/splitk_streamk_gemm/backends/flydsl.md)
- [`languages/flydsl/deep.md:105-114`](../languages/flydsl/deep.md)
- [`languages/flydsl/knobs.md:52-60`](../languages/flydsl/knobs.md)

#### Limits

- The atomic combine adds partials that were already rounded to the output dtype, in whatever order the splits arrive: error grows with split_k and results are not bit-reproducible run to run. Gate against the task's oracle tolerance, not against a split_k=1 output.
- AITER's own test of this kernel holds most split-K cases to its default gate but relaxes its two M=1 B-in-LDS cases, to 99.0% of elements close with a max |delta| of 8 at split_k 8 and 32 at split_k 16 (flydsl-test-parity-gate); a task oracle stricter than that is a reason to prefer a two-pass or fp32 combine.
- The counter capacity is a property of the current kernel; an authored split-K with a different combine has its own bound.
- Whether split-K pays for a shape is a measurement; this card only states what a legal split-K launch requires.

### When can a FlyDSL GEMM route its accumulators through an LDS CShuffle epilogue, and what does it cost?

**Card:** `flydsl-cshuffle-epilogue` · **type:** `performance_candidate` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `epilogue`, `configuration_constraints`, `workgroup_tile`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `scaled_quant_gemm`, `grouped_gemm_moe`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`

#### Use when

- target_language=flydsl
- the output store path (layout, vector width, atomics) is being chosen

#### Try

- Treat the epilogue as a choice between the row epilogue, which stores straight from the MFMA accumulator layout, and the LDS CShuffle epilogue, which writes the tile row-major into LDS, barriers, re-reads it with threads remapped to (MLane, NLane) = (8, 32) and emits vectorised half2 stores or atomics.
- Check legality first: block_size must divide by cshuffle_nlane (32), tile_m by CShuffleMLane (256/32 = 8 for the 256-thread A8 block), and tile_n by cshuffle_nlane*e_vec (64 by default).
- Price the LDS: the staging buffer is 2*tile_m*tile_n bytes and shares the A ping/pong space in the family's estimate; when it overflows, the helper can split the buffer between two wave groups.
- Use its `store_pair` hook for split-K or atomic accumulation: the same shuffled layout feeds either a store or an atomic.
- Keep both arms: in AITER's gfx950 fp8 selections CShuffle on and off both occur in every M bucket (22 of 72 rows on at M<=16, 32 of 50 at M 1025-4096).
- In the split-K HGEMM the equivalent (`c_to_lds`) is fixed off and rejected if requested.

#### Why this is a candidate

- The helper states its legality as raises and its data flow in its docstring, so the contract is readable rather than reverse-engineered.
- The epilogue is the only place the corpus has no other card for, and it is where store coalescing and split-K accumulation are decided.

#### Keep as alternatives

- Direct row store from the MFMA layout — the default and the control arm.

#### Same question in other implementations

- **ck** — the CShuffle epilogue is part of the device-op template (`CShuffleDataType`, `*_Xdl_CShuffle*` instances). Evidence: `src_5ebdfc64c47345bb` (`csrc/ck_gemm_a8w8_blockscale/include/gemm_a8w8_blockscale_common.cuh:52`), `src_7e81eb9b49b377d5` (`csrc/ck_gemm_a8w8_bpreshuffle/include/gemm_a8w8_bpreshuffle_common.cuh:50`), `src_780fadebcc954d06` (`csrc/ck_batched_gemm_bf16/include/batched_gemm_bf16_common.cuh:86`)

#### Evidence

- `src_1786f3708aa447c0` — `layout_epilogue` `c_shuffle` at `aiter/ops/flydsl/kernels/mfma_epilogues.py:15`
- `src_0a9c121156ff2c92` — `layout_epilogue` `c_shuffle` at `aiter/ops/flydsl/kernels/mfma_epilogues.py:85-416`
- `src_d0d34580300e6c02` — `config_validity` `block_size ({block_size}) must be divisible by cshuffle_nlane ({cshuff…` at `aiter/ops/flydsl/kernels/mfma_epilogues.py:132-134`
- `src_4c3052f8521fc401` — `config_validity` `tile_m must be divisible by CShuffleMLane ({cshuffle_mlane}), got tile…` at `aiter/ops/flydsl/kernels/mfma_epilogues.py:137-139`
- `src_949b583f0277e194` — `config_validity` `tile_n must be divisible by (CShuffleNLane*EVec) = {cshuffle_nlane*e_v…` at `aiter/ops/flydsl/kernels/mfma_epilogues.py:143-145`
- `src_756a8fa4df5e93d9` — `layout_epilogue` `CShuffle` at `aiter/ops/flydsl/kernels/mfma_epilogues.py:160`
- `src_c701f7377a04c988` — `config_validity` `preshuffle_gemm_estimated_lds_bytes` at `aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py:119-157`
- `src_f83d827dcfa59956` — `layout_epilogue` `use_cshuffle_epilog` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:134`
- `src_d0f75ad1bcdf6730` — `config_validity` `Current kernel only supports c_to_lds={FIXED_C_TO_LDS};` at `aiter/ops/flydsl/gemm_kernels.py:128-131`
- `cfg_909edfc582fce0bd` — FlyDSL `a8_preshuffle` `af8_wf8_b16` selections in `a8w8_bpreshuffle_tuned_gemm` on `gfx950`/256 CUs, `M<=16`: 72 shipped row(s)
- `cfg_496162f129968cb9` — FlyDSL `a8_preshuffle` `af8_wf8_b16` selections in `a8w8_bpreshuffle_tuned_gemm` on `gfx950`/256 CUs, `M1025-4096`: 50 shipped row(s)

#### Background (always-on reference docs, not measurements)

- [`operators/gemm_epilogue_fused/backends/flydsl.md:20-26`](../operators/gemm_epilogue_fused/backends/flydsl.md)
- [`languages/flydsl/authoring_gemm_levers.md:167-180`](../languages/flydsl/authoring_gemm_levers.md)

#### Limits

- Legality is the helper's; the LDS cost interacts with the A staging budget, so re-run the LDS estimate for the combined configuration.
- Selection shares are where AITER chose each arm, not how much either arm helped.

## 9. Host configuration, ABI and launch

- `runtime_contract` — Which host-side layout, argument block, workspace and launch contract must the kernel satisfy?

Also bears on this step: `flydsl-a8-scale-granularity-contract` (step 6), `gemm-blockscale-k-loop-scale-application` (step 6), `gemm-split-k-when-grid-underfills` (step 3), `flydsl-preshuffle-b-layout-contract` (step 4), `flydsl-splitk-reduction-contract` (step 8), `gemm-default-backend-routing` (step 2).

### What does the host side of a FlyDSL GEMM have to provide in AITER — compile cache, launch grid, per-stream state and kernel name?

**Card:** `flydsl-host-launch-contract` · **type:** `constraint` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `runtime_contract`, `work_partition`, `implementation_variant`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `scaled_quant_gemm`, `grouped_gemm_moe`, `splitk_streamk_gemm`, `skinny_gemv_decode`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`

#### Use when

- target_language=flydsl
- a FlyDSL kernel is being launched, cached or registered for dispatch

#### Try

- Compile once per configuration and cache it: AITER's HGEMM, small-M, A8 preshuffle and 2-stage MoE compilers are `@functools.lru_cache(maxsize=1024)` (the mixed-precision MoE ones unbounded), keyed on the full configuration, so every new tile or split is a new compile — warm up before timing.
- Make the launch grid match what the kernel reads: the HGEMM and small-M launch `grid=(bm, bn, SPLIT_K)` — block x is the M tile, y the N tile, z the K split — while the A8 GEMM launches `(gx, gy, 1)`. A kernel that reads `block_idx.z` as its split needs the host to launch that z.
- Supply split-K's per-stream state: the counter buffer and the current signal state go into every split-K launch and the state advances after it (flydsl-splitk-reduction-contract).
- Name the kernel in the registry's grammar: tuned rows encode every knob in the kernel name matched by `_HGEMM_KERNEL_RE`, and `tuned_gemm` drops a `flydsl` row whose name does not parse.
- Check every FlyDSL symbol against the installed version before copying code across versions (version_map); a module that moved fails at import, not at compile.

#### Why this is a candidate

- Each of these is a failure that surfaces only at run time — a recompile per call, a grid that leaves splits unlaunched, counters that were never allocated, a tuned row silently dropped.

#### Keep as alternatives

- Call the library entry points (`flydsl_hgemm`, `flydsl_preshuffle_gemm_a8`), which own this contract, instead of launching an authored kernel directly.

#### Evidence

- `src_4d28023c79d1fe52` — `jit_cache` `1024, compile_hgemm_kernel` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:102`
- `src_1e5ae08be5adf756` — `jit_cache` `1024, compile_small_m_hgemm_kernel` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:388`
- `src_ca495db46be070cd` — `jit_cache` `1024, compile_preshuffle_gemm_a8` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:123`
- `src_ada62c940d6fc161` — `jit_cache` `1024, compile_moe_gemm1` at `aiter/ops/flydsl/kernels/moe_gemm_2stage.py:87`
- `src_821a059af3542249` — `grid_mapping` `bm, bn, SPLIT_K` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:927-931`
- `src_569b9cec48bd2103` — `grid_mapping` `bm, bn, SPLIT_K` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:1399-1403`
- `src_328a63de09b9204a` — `grid_mapping` `gx, gy, 1` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:1674-1678`
- `src_1e7dc0218412b0cf` — `instance_name` `_HGEMM_KERNEL_RE` at `aiter/ops/flydsl/gemm_kernels.py:44-63`
- `src_c9b676e9661f7f1e` — `kernel_family` `flydsl` at `aiter/tuned_gemm.py:133-141`
- `src_86aa6d1400d0f17f` — `split_k` `_advance_split_k_signal_state` at `aiter/ops/flydsl/gemm_kernels.py:688-692`
- `src_194c17aa69828857` — `split_k` `SPLIT_K_GLOBAL_SEMAPHORE` at `aiter/ops/flydsl/gemm_kernels.py:66`

#### Background (always-on reference docs, not measurements)

- [`languages/flydsl/deep.md:116-122`](../languages/flydsl/deep.md)
- [`languages/flydsl/version_map.md:278-315`](../languages/flydsl/version_map.md)
- [`languages/flydsl/patterns.md:104-108`](../languages/flydsl/patterns.md)

#### Limits

- This is AITER's host contract at the pinned commit; an authored kernel launched outside AITER's wrappers must provide the same pieces itself.

## 10. Correctness validation and tuning

- `validation_contract` — Which reference and tolerance does the implementation's own test hold it to, and where does it relax them?
- `tunable_surface` — Which knobs exist, which values are legal seeds, and which fields are derived rather than tunable?

The oracle, parity gate and timing protocol belong to the workflow's verify step, not to this corpus; cards here state only what a kernel's own contract changes about validation.

Also bears on this step: `flydsl-hgemm-config-space` (step 3), `flydsl-a8-preshuffle-config-space` (step 3), `gemm-blockscale-shipped-bucket-structure` (step 3), `gemm-streamed-operand-cache-policy` (step 5).

### Which parity does AITER's own FlyDSL GEMM test demand, where does it relax it, and what does that say about validating an authored kernel?

**Card:** `flydsl-test-parity-gate` · **type:** `constraint` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `validation_contract`, `work_partition`, `epilogue`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `splitk_streamk_gemm`, `skinny_gemv_decode`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`

#### Use when

- target_language=flydsl
- a FlyDSL GEMM — especially a split-K one — is being validated against a reference

#### Try

- Gate against an fp32 reference first: AITER's split-K HGEMM test compares with `torch.isclose` at atol = rtol = 1e-2 and requires 99.9% of elements close.
- Relax only where the implementation had to: AITER's split-K precision cases at split_k 8 (M=104, K=7168) and split_k 4 (M=1, K=512) hold the default gate; only its two M=1 `b_to_lds` cases are relaxed, to 99.0% of elements with max |delta| <= 32 at split_k 16 (K=7168) and <= 8 at split_k 8 (K=1536). The drift comes from the atomic combine of partials already rounded to bf16 (flydsl-splitk-reduction-contract).
- Let the task's oracle and tolerance decide acceptance (the workflow's verify step); use AITER's gate to judge a failure — drift within it is the known cost of split-K, drift beyond it points at a bug.

#### Why this is a candidate

- The relaxed cases are the implementation stating where its numerics drift; without them a correct split-K kernel reads as broken, or a broken one as merely imprecise.

#### Keep as alternatives

- Split-K with an fp32 workspace and a separate reduce pass (the Triton pattern), which keeps the combine in fp32 at the cost of a second kernel.

#### Evidence

- `src_9d1d40ad1d82f015` — `parity_tolerance` `DEFAULT_ATOL, 1e-2` at `aiter/ops/flydsl/test_flydsl_splitk_hgemm.py:41`
- `src_f0baf9a97c14b111` — `parity_tolerance` `DEFAULT_RTOL, 1e-2` at `aiter/ops/flydsl/test_flydsl_splitk_hgemm.py:42`
- `src_1183d7ed5d869e8a` — `parity_tolerance` `DEFAULT_PASS_PCT, 99.9` at `aiter/ops/flydsl/test_flydsl_splitk_hgemm.py:43`
- `src_217562a4a9b879dc` — `parity_tolerance` `pass_pct, 99.0` at `aiter/ops/flydsl/test_flydsl_splitk_hgemm.py:85`
- `src_636fbc44252bb90e` — `parity_tolerance` `max_delta_limit, 32.0` at `aiter/ops/flydsl/test_flydsl_splitk_hgemm.py:86`
- `src_972b88aa5143a3f7` — `parity_tolerance` `pass_pct, 99.0` at `aiter/ops/flydsl/test_flydsl_splitk_hgemm.py:102`
- `src_f30b29895e6d16b2` — `parity_tolerance` `max_delta_limit, 8.0` at `aiter/ops/flydsl/test_flydsl_splitk_hgemm.py:103`
- `src_724636e4ecf6c5f6` — `config_validity` `_check_output` at `aiter/ops/flydsl/test_flydsl_splitk_hgemm.py:129-154`
- `src_dcea4e10d8996a4e` — `config_space` `SPLITK_PRECISION_CASES` at `aiter/ops/flydsl/test_flydsl_splitk_hgemm.py:46-105`

#### Background (always-on reference docs, not measurements)

- [`operators/splitk_streamk_gemm/backends/flydsl.md:77-85`](../operators/splitk_streamk_gemm/backends/flydsl.md)
- [`operators/dense_gemm/numerics.md:59-63`](../operators/dense_gemm/numerics.md)

#### Limits

- These are AITER's regression cases for its own kernel; an authored kernel is judged by the task's oracle, not by this test.

## Shipped configuration seeds: FlyDSL (AITER tuned databases)

Generated from the FlyDSL rows of AITER's tuned GEMM databases (`aiter/configs/**/*tuned*gemm*.csv`), decoded from AITER's own kernel-name serialisation. Match gfx, CU count, database, family, dtype and M bucket before using a row. **Seed candidate** means every selected row in the group carries the value; **vary next** lists knobs that moved with the shape, with the most frequent values first. The tuner's timings are not carried, so these are concrete candidates, not measured winners.

### `gfx950` · 256 CUs · `a8w8_bpreshuffle_tuned_gemm` · FlyDSL `a8_preshuffle` · `af8_wf8_b16`

| decision ref | M bucket | shipped rows | seed candidate | vary next |
|---|---|---|---|---|
| `cfg_909edfc582fce0bd` | `M<=16` | 72 | `scheduler=default` | `lds_stage`: 2: 36, 1: 36; `tile`: 16x64x512: 61, 16x64x256: 7, 16x256x512: 4; `use_async_copy`: 0: 55, 1: 17; `use_cshuffle_epilog`: 0: 50, 1: 22; `waves_per_eu`: 0: 18, 2: 15, 1: 15 (+2 more) |
| `cfg_2f03fe974e5d2b5c` | `M17-64` | 42 | `scheduler=default` | `lds_stage`: 2: 23, 1: 19; `tile`: 32x64x512: 14, 16x64x512: 8, 32x64x256: 6 (+9 more); `use_async_copy`: 0: 30, 1: 12; `use_cshuffle_epilog`: 0: 28, 1: 14; `waves_per_eu`: 1: 12, 4: 9, 3: 8 (+2 more) |
| `cfg_382465b289aad520` | `M65-256` | 34 | `scheduler=default` | `lds_stage`: 2: 19, 1: 15; `tile`: 64x64x256: 12, 128x64x256: 5, 128x128x256: 3 (+9 more); `use_async_copy`: 0: 19, 1: 15; `use_cshuffle_epilog`: 0: 22, 1: 12; `waves_per_eu`: 4: 8, 1: 8, 3: 6 (+2 more) |
| `cfg_8245d4268291ab55` | `M257-1024` | 28 | `scheduler=default` | `lds_stage`: 2: 23, 1: 5; `tile`: 128x128x256: 7, 128x128x128: 4, 256x128x128: 4 (+10 more); `use_async_copy`: 1: 18, 0: 10; `use_cshuffle_epilog`: 0: 20, 1: 8; `waves_per_eu`: 2: 12, 1: 7, 3: 4 (+2 more) |
| `cfg_496162f129968cb9` | `M1025-4096` | 50 | `scheduler=default` | `lds_stage`: 2: 35, 1: 15; `tile`: 128x256x128: 23, 256x128x128: 8, 128x128x256: 6 (+5 more); `use_async_copy`: 1: 39, 0: 11; `use_cshuffle_epilog`: 1: 32, 0: 18; `waves_per_eu`: 2: 29, 1: 9, 0: 7 (+1 more) |
| `cfg_9636311ea3a3ce0d` | `M>4096` | 77 | `scheduler=default` | `lds_stage`: 2: 59, 1: 18; `tile`: 128x256x128: 39, 256x128x128: 26, 128x128x256: 4 (+4 more); `use_async_copy`: 1: 65, 0: 12; `use_cshuffle_epilog`: 0: 41, 1: 36; `waves_per_eu`: 2: 52, 0: 12, 1: 10 (+2 more) |

### `gfx950` · 256 CUs · `bf16_tuned_gemm` · FlyDSL `hgemm` · `abf16_wbf16_bf16`

| decision ref | M bucket | shipped rows | seed candidate | vary next |
|---|---|---|---|---|
| `cfg_973fe63f897809df` | `M<=16` | 86 | `async_copy=true`, `b_preshuffle=false`, `c_to_lds=false`, `stages=2` | `b_to_lds`: true: 61, false: 25; `split_k`: 8: 34, 16: 29, 4: 19 (+2 more); `tile`: 32x64x128: 25, 32x64x64: 24, 32x128x128: 14 (+10 more); `warps`: 2x2: 61, 1x4: 25 |
| `cfg_ac0c03746ab634f6` | `M17-64` | 45 | `async_copy=true`, `b_preshuffle=false`, `c_to_lds=false`, `stages=2` | `b_to_lds`: true: 35, false: 10; `split_k`: 4: 14, 8: 14, 16: 7 (+2 more); `tile`: 32x64x128: 23, 64x64x128: 6, 32x64x64: 5 (+5 more); `warps`: 2x2: 35, 1x4: 10 |
| `cfg_147eed9b10acbc4e` | `M65-256` | 52 | `async_copy=true`, `b_preshuffle=false`, `c_to_lds=false`, `stages=2` | `b_to_lds`: true: 36, false: 16; `split_k`: 8: 25, 1: 17, 4: 6 (+2 more); `tile`: 32x64x128: 16, 128x128x128: 14, 64x64x128: 9 (+5 more); `warps`: 2x2: 36, 1x4: 16 |
| `cfg_ae017b25b6473300` | `M257-1024` | 29 | `async_copy=true`, `b_preshuffle=false`, `c_to_lds=false`, `split_k=1`, `stages=2` | `b_to_lds`: true: 21, false: 8; `tile`: 96x64x128: 9, 96x128x64: 9, 128x128x128: 8 (+2 more); `warps`: 2x2: 21, 1x4: 8 |
| `cfg_6660b64132bd9e80` | `M1025-4096` | 4 | `async_copy=true`, `b_preshuffle=false`, `b_to_lds=true`, `c_to_lds=false`, `split_k=1`, `stages=2`, `warps=2x2` | `tile`: 96x128x64: 2, 96x64x128: 1, 128x64x64: 1 |
| `cfg_94529fdef02c8a2f` | `M>4096` | 3 | `async_copy=true`, `b_preshuffle=false`, `c_to_lds=false`, `split_k=1`, `stages=2` | `b_to_lds`: true: 2, false: 1; `tile`: 64x64x128: 1, 128x128x128: 1, 128x64x64: 1; `warps`: 2x2: 2, 1x4: 1 |

## Backend AITER's tuner selected, by shape bucket

Each row counts the shipped tuned-database rows whose selected backend (`libtype`) was each implementation, per gfx, CU count, database and M bucket. It says where AITER chose a backend, not by how much: timings, the competing set and the tuner's box are not recorded, and a shape shipped in two model databases counts twice. Use it to see which implementation a FlyDSL kernel competes with in production for a case — and where FlyDSL has never been selected.

### `gfx950` · 256 CUs · `a8w8_blockscale_bpreshuffle_tuned_gemm`

| selection ref | M bucket | rows | selected backend (rows) |
|---|---|---|---|
| `sel_b7d6386940696a48` | `M<=16` | 241 | ck 223 · cktile 12 · asm 6 |
| `sel_bdf0a0ce3c910fc6` | `M17-64` | 139 | ck 120 · asm 15 · cktile 4 |
| `sel_bc5f94a60e7ac2a4` | `M65-256` | 243 | ck 185 · asm 54 · cktile 4 |
| `sel_9f87c053af503719` | `M257-1024` | 386 | ck 266 · asm 109 · cktile 11 |
| `sel_a91bc7afb96c3708` | `M1025-4096` | 306 | ck 264 · cktile 22 · asm 20 |
| `sel_b6b0e3ba26774e37` | `M>4096` | 183 | ck 140 · cktile 40 · asm 3 |

### `gfx950` · 256 CUs · `a8w8_blockscale_tuned_gemm`

| selection ref | M bucket | rows | selected backend (rows) |
|---|---|---|---|
| `sel_4b9bcc0f7ab9c1d0` | `M<=16` | 193 | ck 190 · cktile 3 |
| `sel_2f4edda53f68e173` | `M17-64` | 95 | ck 93 · cktile 2 |
| `sel_a0e632e90c21cdfc` | `M65-256` | 203 | ck 200 · cktile 3 |
| `sel_ee2dd068224e320e` | `M257-1024` | 323 | ck 296 · cktile 27 |
| `sel_b9a37194ba7c274a` | `M1025-4096` | 114 | ck 80 · cktile 34 |
| `sel_8b543c4558078855` | `M>4096` | 325 | ck 178 · cktile 147 |

### `gfx950` · 256 CUs · `a8w8_bpreshuffle_tuned_gemm`

| selection ref | M bucket | rows | selected backend (rows) |
|---|---|---|---|
| `sel_7c8d8e410633ad1e` | `M<=16` | 111 | flydsl 72 · ck 37 · cktile 2 |
| `sel_da4768bfb4a81119` | `M17-64` | 88 | flydsl 42 · ck 34 · cktile 12 |
| `sel_75a6d5abfd89238c` | `M65-256` | 98 | ck 44 · flydsl 34 · cktile 20 |
| `sel_a7859ab04377ab13` | `M257-1024` | 91 | ck 38 · flydsl 28 · cktile 25 |
| `sel_3b5495db028ae505` | `M1025-4096` | 91 | flydsl 50 · cktile 26 · ck 15 |
| `sel_1b5489e6d41507a3` | `M>4096` | 133 | flydsl 77 · cktile 36 · ck 20 |

### `gfx950` · 256 CUs · `bf16_tuned_gemm`

| selection ref | M bucket | rows | selected backend (rows) |
|---|---|---|---|
| `sel_81411b8982b03045` | `M<=16` | 165 | flydsl 86 · asm 45 · triton 34 |
| `sel_49310b1f04381a1e` | `M17-64` | 137 | asm 62 · flydsl 45 · triton 30 |
| `sel_4ff2330404509760` | `M65-256` | 174 | asm 88 · flydsl 52 · triton 34 |
| `sel_bb4f91d843dec8df` | `M257-1024` | 118 | asm 83 · flydsl 29 · triton 6 |
| `sel_0bf8dd0fd028cea8` | `M1025-4096` | 72 | asm 62 · triton 6 · flydsl 4 |
| `sel_c472ec443f7b30a0` | `M>4096` | 114 | asm 102 · triton 9 · flydsl 3 |

### `gfx942` · 80 CUs · `a8w8_blockscale_bpreshuffle_tuned_gemm`

| selection ref | M bucket | rows | selected backend (rows) |
|---|---|---|---|
| `sel_f7e394faa23e37e6` | `M<=16` | 18 | ck 16 · asm 2 |
| `sel_abae8b787c473637` | `M17-64` | 29 | asm 21 · ck 8 |
| `sel_989b9db8a18a0bb0` | `M65-256` | 110 | asm 79 · ck 31 |
| `sel_a216d21d97247805` | `M257-1024` | 218 | asm 146 · ck 72 |
| `sel_6d485fcb23ffd0fe` | `M1025-4096` | 155 | asm 84 · ck 71 |
| `sel_f1384cc1069c9823` | `M>4096` | 38 | asm 32 · ck 6 |

### `gfx942` · 80 CUs · `a8w8_bpreshuffle_tuned_gemm`

| selection ref | M bucket | rows | selected backend (rows) |
|---|---|---|---|
| `sel_5aecc3631bf688e9` | `M<=16` | 204 | ck 204 |
| `sel_5d90eccfeec27af1` | `M17-64` | 95 | ck 94 · asm 1 |
| `sel_def10f3ed4823c11` | `M65-256` | 279 | ck 260 · asm 18 · cktile 1 |
| `sel_3316f2390f8a8002` | `M257-1024` | 365 | ck 356 · asm 9 |
| `sel_bb0d68aae12eba45` | `M1025-4096` | 102 | ck 98 · asm 4 |
| `sel_438e88e4bc86460c` | `M>4096` | 395 | ck 391 · asm 4 |

### `gfx942` · 304 CUs · `a8w8_bpreshuffle_tuned_gemm`

| selection ref | M bucket | rows | selected backend (rows) |
|---|---|---|---|
| `sel_5cf769a02e468084` | `M17-64` | 2 | asm 2 |
| `sel_9609d454b46e0fab` | `M65-256` | 4 | asm 4 |
| `sel_20bfb36c5f1ee347` | `M257-1024` | 4 | asm 4 |
| `sel_6bb0baf26eda8568` | `M1025-4096` | 4 | asm 4 |
| `sel_4914d4c661ff1e2a` | `M>4096` | 18 | asm 18 |

Databases read but not counted as selections:

- `aiter/configs/a4w4_blockscale_tuned_gemm.csv` — no `libtype` column: the selected backend is not recorded, so these rows are inventoried but not counted as a selection
- `aiter/configs/a8w8_tuned_batched_gemm.csv` — no `libtype` column: the selected backend is not recorded, so these rows are inventoried but not counted as a selection
- `aiter/configs/a8w8_tuned_gemm.csv` — no `libtype` column: the selected backend is not recorded, so these rows are inventoried but not counted as a selection
- `aiter/configs/bf16_tuned_batched_gemm.csv` — no `libtype` column: the selected backend is not recorded, so these rows are inventoried but not counted as a selection
- `aiter/configs/model_configs/dsv3_a4w4_blockscale_tuned_gemm.csv` — no `libtype` column: the selected backend is not recorded, so these rows are inventoried but not counted as a selection

Databases inventoried: 24 (8317 rows).

## Other backends' shipped seeds

The 306 Triton configuration groups AITER ships under `aiter/ops/triton/configs/gemm` are rendered on [`gemm_triton_seeds.md`](gemm_triton_seeds.md). They are Triton knob names; open that page when the Triton baseline's own config is the question, not when choosing FlyDSL knobs.

## Sources

- Curated cards: [`decisions/gemm.yaml`](decisions/gemm.yaml).
- Source evidence: [`evidence/gemm_source.yaml`](evidence/gemm_source.yaml).
- Shipped tuned databases (FlyDSL seeds and backend selection): [`evidence/gemm_csv_tuned_configs.yaml`](evidence/gemm_csv_tuned_configs.yaml).
- Shipped Triton configs: [`evidence/gemm_tuned_configs.yaml`](evidence/gemm_tuned_configs.yaml).
- Performance axes: [`performance_axes/gemm.yaml`](performance_axes/gemm.yaml).
- Generated by [`_render_decisions.py`](_render_decisions.py); edit the cards or evidence, never this file.
