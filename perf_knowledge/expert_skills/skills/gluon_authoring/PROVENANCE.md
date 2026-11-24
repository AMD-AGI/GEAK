# gluon_authoring — provenance

Maintainer notes moved out of `skill.md` (2026-11-22); agents do not need this page.


## Sources

- **Provenance.** The upstream `tile-programming-gluon` pack —
  [`AMD-AGI/TileProgrammingAgentSkills@a649eaaf`](https://github.com/AMD-AGI/TileProgrammingAgentSkills/tree/a649eaaf0a989de949f38cd00916416c95d59b95)
  (`main`, fetched 2026-11-05), reorganized into GEAK's structure. The upstream entry (`SKILL.md`) and its
  method (`references/method-reference.md`) were folded into `references/method/*.md`; `gluon/`,
  `tile-programming/`, `hardware/` and `workloads/` keep upstream's layout, deduplicated and reordered
  gfx950-first. Upstream remains the SSOT for the method; a re-sync is a merge against the headings recorded
  in each method file's own `## Sources`, never an overwrite.
- **Priority rules applied while merging:** upstream Triton 3.8.0 (3.6/3.7 as downgrade notes); gfx950 the
  main line, gfx942 the downgrade; the pipeline hand-written first, re-injecting plain's auto-pipeliner a
  diagnostic of last resort; GEAK infrastructure wins where it owns a concern (GPU locking, profiler entry,
  frozen measurement, the commit gate), and the shared hardware data in `perf_knowledge/hardware/data/` is the
  single source for this skill and GEAK's e2e roofline.
- **Still GEAK-local — owed upstream, and must survive the next sync:**
  - `scripts/ttgir_to_gluon.py` emits `tilesPerWarp` / `elementBitWidth` on `AMDMFMALayout` when, and only
    when, the TTGIR prints them (dropping them transcribed chained-dot / scaled-MFMA kernels to a silently
    different layout; the witness is still synthetic — only a genuine `tt.dot_scaled` body reaches
    `deduceTilesPerWarpForScale`).
  - `scripts/recover_gluon.py` peels `@triton.autotune` / `@triton.heuristics` before translating, and
    `scripts/smoke_test_recover.sh` runs every offline `--selftest` of the toolchain.
  - `scripts/patch_reinject.py` exits 2 with a message where `triton` is absent and pins its splice point by
    selftest; `scripts/patch_async_reinject.py` (GEAK-only) covers a hand-written async body with no
    pipeliner; `scripts/pipeline_survey.py` keeps its dot-candidacy `--selftest`.
  - `scripts/ttgir_bridge.py`: an LDS/CU fallback when `amd_occupancy` cannot be imported, and the "a deeper
    depth may still be reachable via the annotation" advice on a `num_stages=1` verdict.
  - `kernel_tools/probe.py measure` prints both occupancy limiters per kernel and names the binding one;
    `kernel_tools/amd_occupancy.py` never walks the filesystem for its data and keeps the RDNA4
    `--compiler-sweep` tool GEAK's RDNA4 knowledge calls; `kernel_tools/asm_loop_audit.py` keeps a linear
    wait-counter regex; `kernel_tools/dump_ir.sh` pins with `--kernel-name` and refuses a bare name on
    `--kernel`, moves caches aside instead of deleting them, and refuses `--emit-gluon` without an arch.
  - `perf_knowledge/hardware/data/hw_constants.json` keeps GEAK's RDNA4 notes; GEAK's gfx1151
    `lds_per_wgp_kib` (128) is kept, because GEAK's default-path `amd_ryzen.md` points readers to it; its
    `_lds_basis` states that the per-WGP scope is family-derived (RDNA3 guide), not read from the RDNA3.5 guide. `sku.json` carries the
    corrected MI325X / MI350X / gfx1151 rows.
  - `references/pitfalls/platform-known-issues.md` keeps GEAK's RDNA4 profiling wording (asserted by
    `kernel_workflow/scripts/test_rdna4_policy.js`).
- **Not vendored, and what degrades without it:** the `tile-programming-triton` front end that emits
  `plain_champion.json` (GEAK's own tuned plain kernel serves as the comparison point); the
  upstream `gluon-direction` / `gluon-bench` / `amd-closure-skeptic` agents and the `kernel-opt-run` /
  `kernel-opt-fleet` captains — their roles are GEAK's existing roles (`## Roles in GEAK`); the
  `third_party/` submodules — `amd_matrix_instruction_calculator` (non-gfx950 operand layouts fall back to the
  distilled references), `rocprof-trace-decoder` (ATT timing; bundled with rocprofv3 on ROCm >= 7.13 images, `ROCPROF_ATT_LIBRARY_PATH` only on older ones), the pinned ISA
  XML (in-pack encoding databases cover gfx950 / cdna3 / rdna3 / rdna4).
- **How this file got short (2026-11-21).** The dedicated gluon expert workflow was removed on 2026-11-20;
  the method has since ridden `kernel_workflow` as advice. This file still carried that workflow's
  procedure (entry modes, a transcribe → recover → climb spine, gates G0–G4, launch-arg values, "one
  `deep_explore` direction"), which read as instructions to the injected roles, and `match.dtypes`
  listed only bf16 / fp16 / fp8.
  The procedure was removed, the method pages were marked optional background, and `match.dtypes` became
  `'*'` (`perf_knowledge/index/changelog.md`).
- **Dropped when the upstream KB pages were restored (deliberate, no new home).** (1) The CDNA2 / gfx90a
  granule-8 wording in `perf_knowledge/hardware/cdna2_mi200/{memory,occupancy}.md`: gfx90a is outside this
  skill's gfx950 / gfx942 scope and its occupancy was not checked here, so the skill carries no gfx90a rule
  (the granule-8 marker in `references/hardware/planning-constants.md` is scoped to gfx942 / gfx950).
  (2) The RDNA rows' rewording in `perf_knowledge/index/taxonomy.md`: product-policy wording for RDNA, which
  is out of scope; `sku.json` and `hardware_context.py` carry the product identity. (3) The gfx908 /
  gfx90a AGPR-file note in `perf_knowledge/hardware/shared/wavefront_simd_vgpr_agpr.md` (separate file on
  gfx908, shared on gfx90a): neither arch is in scope or was checked here; the gfx942 / gfx950 statement
  (one 512-entry file, combined budget) is in `references/hardware/planning-constants.md`.
- **The bar.** `expects.isolated_speedup_min` is a sanity reference; `validation.status` is `validated`
  (promoted 2026-11-20); `measured` stays empty until a `--record` run
  stamps numbers.
- Gluon language surface: ROCm Gluon GEMM tutorial
  (https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html), gfx950 Gluon
  tutorials (https://github.com/ROCm/gfx950-gluon-tutorials); onboarding and the measured GEMM ceilings stay in
  [`languages/gluon/`](../../../languages/gluon/).
