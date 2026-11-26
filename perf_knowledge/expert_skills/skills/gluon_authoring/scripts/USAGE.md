# scripts — usage

Run from the GEAK repo root with

```bash
SKILL=perf_knowledge/expert_skills/skills/gluon_authoring     # the skill: Gluon-specific tools + pack runtime
KT=kernel_workflow/scripts/kernel_tools                       # GEAK's shared kernel tools (ISA, profiling, roofline)
```

Three owners, and the skill defers to the first two:

- **GEAK infrastructure** — `kernel_workflow/scripts/gpu_lock.sh` (shared GPU ownership; lock dir
  `/tmp/team_gpu_locks`; a kernel tool started inside it detects the lock a parent holds and reuses it;
  with `GEAK_KERNEL_TOOLS=1` the lock also honours `GEAK_GPU_LOCK_DIR` (hermetic tests) and exports
  `GEAK_GPU_LOCK_HELD`), `kernel_workflow/scripts/profile_kernel.sh` +
  `profile_policy.sh` (the profiler entry), the frozen task runner (acceptance; `harness_lib.py` for extracted tasks), `scripts/gpu_identity.py` (`gfx`, `target`, `sku`).
- **`$KT`** — shared diagnostics and search helpers, including tools promoted from existing Gluon entry points. Their data is
  `perf_knowledge/hardware/data/` (resolved by `$KT/_hwdata.py`; override `GEAK_HW_DATA_DIR`).
- **`$SKILL/scripts`** — Gluon-specific tools and the pack runtime. Shared tools are called by their `$KT/` path.

**Arch is never defaulted.** A tool that needs one refuses without `--arch` (or a dump that names its target);
examples use `--arch gfx950`, the main line, with gfx942 as the downgrade.

## Tool map (by the flow in `skill.md ## Procedure`)

| step | tools | answers |
| --- | --- | --- |
| port: transcribe | `$SKILL/scripts/ttgir_bridge.py`, `recover_gluon.py`, `ttgir_to_gluon.py`, `emit_gluon.py`; `$KT/dump_ir.sh` | a Gluon skeleton with the compiler's layouts, numerics-checked against plain |
| port: recover the overlap | `pipeline_examples_cdna4.py` (gfx942: `_cdna3.py`), `smoke_layout_gpu.py`; diagnostics only: `pipeline_survey.py`, `gluon_swp.py`, `patch_reinject.py`, `patch_async_reinject.py` | a compile-checked pattern to author; how much of plain's speed was its pipeliner |
| optimize: evidence | `scripts/gluon_profile.py`; `kernel_workflow/scripts/profile_kernel.sh` (GEAK); `$KT/probe.py measure`, `amd_occupancy.py`, `asm_loop_audit.py`, `capture.sh`, `rocprofv3_safe.sh`, `parse_pmc.py`, `att_opclass.py`, `mfma_efficiency.py`, `amd_isa.py --arch <gfx>` | which playbook signal fires |
| optimize: what is live / bound | `probe_levers.py`; `$KT/hw_budget.py`, `mem_bw_probe.py` | which version-sensitive knobs work on this box; the roofline prior |
| measure | GEAK's frozen task runner (acceptance); `$KT/ab_bench.py` (screening only) | is the difference real |
| pack hygiene | `selftest_all.sh`, `smoke_test_recover.sh`, `check_pack_refs.py`, `check_term_index.py`, `gen_section_map.py` (`--check`; rewrite after heading edits) | the toolchain and the references are sound offline |

The upstream standalone workflow's record, gate and stage tools were removed; GEAK's rounds, records and
acceptance replace them.


## Budget

| command | gives you |
| --- | --- |
| **`$KT/hw_budget.py --hardware-context $EVAL_DIR/hardware_context.json --workload <archetype> --shapes k=v,… --dtype bf16 --brief`** (`--help`: 3-step card) | no GPU. Intensity vs the SKU ridge → bound-class prior with `bound_direction`; the MFMA-only floor and your multiple over it; the `references/workloads/index.md` row for the archetype (read it); `numerator_basis` / `denominator_basis`, `analysis_role: planning`, `may_gate: false` on every ratio. Calibrate with `--measured-dram-mb`, `--measured-hbm-tb-s`, `--measured-tflops lo,hi`; declare what the kernel moves with `--tensors`, `--flops`, `--flops-engine`; C0 with `--grid`, `--footprint-mb`, `--dispatches` |
| `$KT/mem_bw_probe.py --hardware-context $EVAL_DIR/hardware_context.json --runlen … --stride … --rw-mix …` (the file, once per run: `python3 $KT/hw_context.py --output $EVAL_DIR/hardware_context.json`) | the **in-shape** memory ceiling (a range, with its program count) — a separately requested diagnostic, not native TraceLens calibration or an acceptance gate |
| `$KT/extract_sku.py --facts <SKU> \| --print <SKU> \| --check \| --render … \| --check-docs \| --sync-docs` | reads, validates and renders `perf_knowledge/hardware/data/sku.json` (the single per-SKU source); every SKU table in the docs is generated from it |

## Transcription

Start with `ttgir_bridge.py`. The two older tools below it are still the only option where `import triton`
is unavailable, but they carry a hand-written mapping and a text-level equivalence check.

| tool | what it does |
| --- | --- |
| **`ttgir_bridge.py recover\|verify\|view --arch gfx950`** | **The recovery path.** Hands the `.ttgir` to the compiler's own parser and upstream's `layoutToGluon()`, so there is no per-kind mapping to fall behind Triton: an unsupported kind surfaces as a named `UNRECOVERABLE` row. Every layout carries a **round-trip proof** and a `num_warps`-vs-`warps_per_cta` cross-check. `verify` compares **LinearLayout normal forms** (two spellings of one layout compare equal; unroll skew is informational `MULTIPLICITY`). Needs `import triton` (≥ 3.7 for full capability; 3.6 compares shared layouts as text, no `view`). No GPU and no ROCm needed — `--arch` works anywhere, and is required. |
| `$KT/dump_ir.sh <compile cmd …> --variant <name> --out <ir_dir> --arch gfx950 [--knobs …] [--emit-gluon layouts\|anchor\|pipeline] [--kernel module.path:object] [--kernel-name <substring>]` | runs a compile with IR dumping and collects `.ttir` / `.ttgir` / `.amdgcn` per variant; accepts a single `bash -c` string as the command. **`--kernel-name` pins which compiled kernel's artifacts are taken** (multi-kernel ops); `--kernel` is `module.path:object` for the translator and a bare name there is refused. Old caches are moved aside, never deleted. `--emit-gluon` refuses without an arch |
| `recover_gluon.py` | the older driver: dump → recover → emit an anchor → `--verify`. Use it for **anchor assembly** (`--with-skeleton`) and `--record`; take the layout gate from `ttgir_bridge.py verify`. Its `--verify` compares canonical text as a set — sound in one direction only. Peels `@triton.autotune` / `@triton.heuristics` before translating |
| `ttgir_to_gluon.py` | the pure-text parser/emitter under `recover_gluon.py`; no GPU and no `triton` import — the one reason to reach for it. Emits `tilesPerWarp` / `elementBitWidth` on `AMDMFMALayout` when the TTGIR prints them. Its output is a starting point, not a proof; it does not place `convert_layout` and cannot name `amd_rotating_shared` |
| **`$KT/probe.py measure --dir <ir_dir> [--arch gfx950]` · `plan --arch gfx950 …`** | **compile-only occupancy — available as soon as the anchor builds.** `measure` parses the artifact for `shared` bytes/WG, ArchVGPR+AGPR and waves/SIMD in seconds, joined per kernel, and prints **both** limiters in workgroups/CU and which binds; `plan` adds up resident tensors on paper. Among these tools, the one that shows the LDS cost of transcribing a pass-through `ttg.local_alloc` as a user buffer |
| `$KT/amd_occupancy.py --vgpr N --arch <gfx>` | the occupancy model `probe.py` delegates to (CDNA: ArchVGPR+AGPR in one 512-entry file, granule 8, cap 8; LDS term per arch); `--compiler-sweep` derives an RDNA table from `llc` |
| `smoke_test_recover.sh [--selftest] [<gfx950-gluon-tutorials checkout>]` | offline end-to-end check of the port toolchain and every port tool's `--selftest`; run it before trusting these scripts on a new box. Its layer 5 launches `smoke_recover_gpu.py` only when a GPU is visible and `--selftest` is NOT given -- take the GPU lock for that run |
| `smoke_recover_gpu.py` | the on-GPU version; needs `torch` + `triton` + a visible GPU (exits 2 with a message otherwise) |

### Reading `ttgir_bridge.py recover`

- **`num_warps cross-check`** must be `PASS`; `FAIL` (exit 4) means the dump contradicts itself.
- **`UNRECOVERABLE: N`** must be 0 (exit 1 otherwise; seen in practice: `amd_rotating_shared`). Probe your build
  and re-dump at `ns=1` before concluding. No binding reads the encoding back on 3.8.0, but
  `amd_rotating_shared` is linear: an exact `SharedLinearLayout` built from upstream
  `sharedToLinearLayoutAMDRotating` and checked against that formula element by element substitutes for it
  (`../references/pitfalls/negative-patterns.md`).
- **`round-trip: EXACT=N`** must cover every layout.
- **Use source names, not role names** (roles rank by op kind; `FROM_SMEM` can be the layout going *into*
  shared). Each provenance line ends with the source variable and line; the emitted file has a `# DOTS` block.
- **`COMPILED FORM of the N buffer_load site(s)`** — the dump can differ from the source, and a faithful
  transcription follows the dump: `tl.load(..., other=0.0)` often compiles to a buffer load with neither mask
  nor other (buffer OOB returns 0 on CDNA), so `other=` in Gluon adds a `v_cndmask` per register that plain's
  compiled form did not have. `contiguity` and `stride` are not parameters of `gl.amd.cdna4.buffer_load`
  (Triton 3.8.0 signature: `ptr, offsets, mask, other, cache`).
- **`LDS: N allocation(s)`** is in ELEMENTS and sums declared allocations without liveness: an upper bound,
  not the `shared` bytes/WG (`probe.py measure` reads that from the artifact). The divisor is per arch:
  160 KiB gfx950, 64 KiB gfx942.
- **`constants-digest`** compares two recoveries (role names drift between `ns=2` and `ns=1` dumps).
- **unsupported-op table:** layouts recover 100% while an op (`amdg.in_thread_transpose`) has no Gluon builtin.
- `verify` states: `PASS` / `RECONCILED` (exit 0, every difference has a named structural cause) / `FAIL`
  (exit 1) / not comparable; none of them says a substitution was free — only the ISA can.
- **Multi-kernel ops:** `dump_ir.sh` takes the freshest artifact; an op that compiles two kernels (body +
  split-K reduce) hands you whichever finished last. Pass `--kernel-name <substring>`.

## Recover and re-injection (diagnostic only)

`pipeline_survey.py <root>` inventories which pipeline form each plain kernel can exercise (a screen, not a
verdict). Re-injecting plain's pipeliner (`gluon_swp.py`, `patch_reinject.py`, `patch_async_reinject.py`) is
a diagnostic / last resort; its tools, preconditions, failure modes and cache traps live in
[`../references/gluon/pipeline/reinjection.md`](../references/gluon/pipeline/reinjection.md).

## Evidence

Profiler entry is GEAK's, under its GPU lock (never pin `HIP_VISIBLE_DEVICES` inline):
`bash kernel_workflow/scripts/profile_kernel.sh <gpu_id> "<harness cmd> --profile" <out_dir>`; per-signal
tools: `kernel_workflow/scripts/kernel_tools/CARD.md`. `probe_levers.py --all --arch gfx950 --brief` says which
version-sensitive knobs are `live` / `dead-declaration` / `absent` on this build (out-of-range knobs are
accepted and change no IR). Screening A/B: `$KT/ab_bench.py --task <dir>` (`--cache cold|hot`); acceptance is
GEAK verify on the frozen task.

## Selftests — offline, no GPU

```bash
bash "$SKILL/scripts/selftest_all.sh"          # everything: pack scripts, $KT tools, core runtime, refs
bash "$SKILL/scripts/smoke_test_recover.sh" --selftest   # the port toolchain only (no launch)
python3 "$SKILL/scripts/check_pack_refs.py"    # every link / heading / runtime reference resolves
python3 "$SKILL/scripts/gen_section_map.py" --check   # reference.md ## E. Section map matches the headings
```

`ttgir_bridge.py --selftest` has two layers. The pure layers (type splitting, role ranking, operand
attribution, the rank guard, the config precheck) run with no `triton` at all and report the live layer
as skipped. Where `triton` *is* importable it additionally parses a synthetic TTGIR and asserts that
every layout round-trips EXACT, so a regression in upstream's converter is caught here rather than on a
kernel. Two of its guards are not defensive style: handing a memdesc to `get_gluon_layout_from_tensor`
**segfaults**, and calling `to_linear_layout` at a mismatched rank trips an LLVM assert. Both are
process death with no traceback, so both are checked before the call rather than caught after.

## Triton-version notes (things that bite on 3.8.0)

- **Availability is what compiles, not what imports.** `sched_barrier` / `sched_group_barrier` / `set_prio`
  are absent from `gl.amd.cdna3/.cdna4` (aiter's `pa_decode_gluon` imports them in `try/except` and runs
  no-op stubs); `gl.warp_specialize` exists in `gl` and fails to compile on gfx950 / gfx942;
  `gl.amd.warp_pipeline_stage` compiles on both CDNA targets, and the two-group offset needs `num_warps >= 8`. `pipeline_examples_cdna4.py` / `_cdna3.py` re-check this on your box.
- **Recovered layouts were stable across versions; performance ratios were not.** Recovery was byte-identical
  across 3.6.0–3.8.0 on the 16 kernels checked, while timings moved in both directions per kernel (a clear
  Gluon win collapsed to parity on a bump). A ratio measured on one Triton version is not evidence on another.
- **Barriers:** `gluon_to_ttgir` runs no membar pass, but membar runs inside the TritonGPU→LLVM conversion and
  covers Gluon (`../references/tile-programming/pipeline.md`, section "What the compiler inserts after
  wait_group"). An explicit `gl.barrier()` is not needed for the hazards that membar already covers.
- **The triton→gluon translator** (`--with-skeleton`, `--emit-gluon anchor|pipeline`) needs
  `triton.tools.triton_to_gluon_translator` (3.8+ spelling; 3.6/3.7 spell it `translater`). It imports on
  the Triton 3.8.0 ROCm wheel (checked in the image). Decide from the import, not `triton.__version__`.
- **`dump_ir.sh --knobs LLIR_SCHED|AMDGCN_AS|RA_HINTS` are fork-only** env vars; the stock Triton 3.8.0 wheel
  reads none of them (no match in its Python sources or `libtriton.so`).
- **A wrapped kernel translates to nothing:** under `@triton.heuristics` / `@triton.autotune` upstream's
  translator gets the wrapper and returns an empty translation; `recover_gluon.py` peels to the
  `JITFunction` and names the wrapper — point `--kernel` at the kernel.

## Other

Shared tool contracts (A/B schema, correctness rc): `kernel_workflow/scripts/kernel_tools/README.md`.
