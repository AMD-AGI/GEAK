---
id: gluon_authoring
title: "Gluon on AMD CDNA (gfx950 first, gfx942 as the downgrade) — advisory mechanism menu, Gluon API surface and tools for kernel_workflow roles"
kind: expert_skill
authors: [qiongz]
scope: kernel
revision: "2026-11-21"
match:
  operator: '*'
  arch_class: ['*']
  gens: [gfx950, gfx942]
  dtypes: ['*']
  regimes: ['*']
  from_backend: [triton, gluon]
  to_backend: gluon
expects:
  isolated_speedup_min: 1.10
  parity: required
validation:
  status: validated
  last_verified: ""
  gpu: ""
  model: ""
  measured: {isolated: "", e2e_pct: "", parity: ""}
  artifact: ""
playbook_file: playbook.md
role: advisory_prior
supersedes: []
---

# gluon_authoring — Gluon on AMD CDNA (gfx950 first, gfx942 as the downgrade)

**Lookup: [`reference.md`](reference.md) is the entry index** — symptom / error string / question →
page section or tool, at most two hops from here. **Mechanisms: [`playbook.md`](playbook.md)** — what paid
on gfx950, the mechanism menu, the checked pitfalls. This page: when the skill applies, the procedure, the tools.

## When to use

The box is **gfx950** (CDNA4, MI350X / MI355X — the main line) or **gfx942** (CDNA3, MI300X / MI325X — the
downgrade), Triton 3.8.0, and either the kernel is **already Gluon** (`@gluon.jit` with explicit `gl.*Layout`)
or it is **plain Triton** and moving it, or some shape buckets, to Gluon is an option. Gluon pays when the
residual is something plain Triton cannot express: an authored pipeline or wave schedule
(`warp_pipeline_stage`), an explicit LDS ring / swizzle / padding, per-operand staging, per-shape kernels. A
residual a config sweep can remove is cheaper to remove in Triton; so is one an `llvm_fn_attrs` change
removes after that sweep, when spills or `v_accvgpr_*` show in the loop (`llvm_fn_attrs` works on plain
`@triton.jit` too). Advisory for GEAK's `kernel_workflow` roles (`use_expert_skills: "true"`): your
plan, the frozen `COMMANDMENT.md` and GEAK's A/B decide. RDNA (gfx11*/gfx12*) and gfx1250 are out of scope.

**Precedence over the GEAK knowledge base.** Where a line here is marked **Overrides KB** it names the
`perf_knowledge/` page it supersedes; each such line names its own basis and scope (a compile-only ISA check,
a measured A/B, or a source / string check of the installed tree, on gfx950 / Triton 3.8 unless stated) and
wins for that scope while this skill is loaded. Unverified items are marked inline. Unmarked KB content still applies.

## Mechanism

Gluon onboarding: [`languages/gluon/overview.md`](../../../languages/gluon/overview.md),
[`gemm_cookbook.md`](../../../languages/gluon/gemm_cookbook.md). What this skill adds: a port can start from
the compiler's own layouts (`ttgir_bridge.py recover` on plain's `.ttgir`); a faithful port loses plain's
software pipeline (on Triton 3.8.0 `gluon_to_ttgir` runs no pipeliner and ignores `num_stages`), so the overlap
is authored (**Overrides KB** for Triton 3.8.0: `perf_knowledge/languages/gluon/overview.md` describes this skill as
"re-injecting plain's software pipeliner into `gluon_to_ttgir`"; re-injection is a below-parity diagnostic only;
basis: the compile check and the A/B runs in `references/gluon/pipeline/reinjection.md`, mostly gfx942, arch labelled per result); and what measurably paid on the gfx950 kernels tried is a short, signal-gated list —
[`playbook.md ## Skill levers on gfx950 (signal-gated)`](playbook.md#skill-levers-on-gfx950-signal-gated) — which adds to
the broad levers (tile / grid fill, config retune, algorithm) rather than replacing them:
[`playbook.md ## Skill levers add to the broad levers`](playbook.md#skill-levers-add-to-the-broad-levers).

## Procedure

GEAK owns rounds, budgets, GPU locking, timing and acceptance; nothing below is a gate.

### A. The kernel is already Gluon — optimize it

1. Read the evidence (GEAK's profile, the ISA) for the playbook's signals: spills / `v_accvgpr` shuttling,
   full-drain `lgkmcnt(0)` waits, LDS bank conflicts, a skinny-M grid, MFMA idle behind loads
   (`gluon_profile.py`, `probe.py measure`).
2. Match the signals against [`playbook.md`](playbook.md); each mechanism links one detail section.
3. Check before timing: numerics on every case, the Gluon kernel actually launched, the ISA shows the change
   (`dump_ir.sh`, `asm_loop_audit.py`).

### B. The kernel is plain Triton — port it to Gluon, then optimize

The numbered checklist with the command and check for every step, and the verbatim error strings met on
this route: [`transcribe.md ## Port checklist`](references/method/transcribe.md#port-checklist-verified-on-triton-38--gfx950),
[`## Port error strings and symptoms`](references/method/transcribe.md#port-error-strings-and-symptoms).
In short:

1. **Decide whether to port, and which shapes.** The plain kernel at its tuned config is the bar for every
   shape; Gluon for some buckets and plain for the rest is a legitimate result.
2. **Dump plain per config bucket, twice** — at the tuned config and at `num_stages=1` — with
   `dump_ir.sh --kernel-name`, and `ttgir_bridge.py recover` both. *Check:* `UNRECOVERABLE: 0`,
   `round-trip: EXACT`, and the `COMPILED FORM` of each buffer_load (it can differ from the source; a faithful
   anchor follows the compiled form).
3. **Faithful anchor** from the `num_stages=1` layouts: every `gl.arange` / load on a recovered layout,
   `gl.amd.cdna4.buffer_load` with element offsets, fp8 `tl.dot` → `mfma_scaled(..., None, ...)` on the
   recovered `[16, 16, 128]` MMA. *Check:* numerics on every case, the Gluon kernel's name in a launch
   trace, `buffer_load_dwordx4` (no `global_load_ubyte`) in `asm_loop_audit.py`, time ≈ `plain@num_stages=1`.
4. **Recover the overlap**: an `async_copy` ring at plain's depth with the tuned dump's layouts, in plain's
   compiled loop form ([`recover.md ### The async-ring loop form`](references/method/recover.md#the-async-ring-loop-form-that-reaches-plain-parity-gfx950)).
   *Check:* hot loop matches plain's (no extra `v_cndmask` / `s_barrier`), same LDS/WG, interleaved A/B
   vs plain per case. Re-injecting plain's pipeliner (`gluon_swp.py`) is a diagnostic; its precondition
   (plain loads and a dot in the loop) is not met by this anchor form ([`reinjection.md`](references/gluon/pipeline/reinjection.md#precondition-for-re-injection-plain-loads-and-a-dot-in-the-loop)).
5. **Optimize further** as in track A — parity is the starting line, not the result.
6. **Dispatch per shape**: Gluon where it wins beyond noise, plain elsewhere, and plain for any bucket
   without recovered layouts ([`references/method/dispatch-table.md`](references/method/dispatch-table.md));
   re-check numerics and the launch trace after wiring it.

How this maps onto GEAK: the tech lead decides whether a port is a direction, a `deep_explore`, or not worth
it; a port that only reaches parity in a round is a step, not a failure, if step 4 has a named next lever.

## Tools

Run from the GEAK root (`SKILL=perf_knowledge/expert_skills/skills/gluon_authoring`,
`KT=kernel_workflow/scripts/kernel_tools`). All optional; GPU commands go through GEAK's
`bash kernel_workflow/scripts/gpu_lock.sh <id>`, run from the workspace (kernel tools started inside it
detect the held lock); pass `--arch gfx950` (or `gfx942`) explicitly. Keep each tool's exit status
visible: no `| tail` / `| head` without `set -o pipefail`, no `>/dev/null` on a setup step. Full usage:
[`scripts/USAGE.md`](scripts/USAGE.md).

Shared evidence tools (counters, IR/ISA dump, registers/occupancy, hot-loop audit, roofline, MFMA
facts) are GEAK's `kernel_workflow/scripts/kernel_tools/`; their signal → tool → command table is
[`kernel_workflow/scripts/kernel_tools/CARD.md`](../../../../kernel_workflow/scripts/kernel_tools/CARD.md),
which the measuring roles get whenever `use_expert_skills` is on. The Gluon-specific tools:

| when | tool | command |
| --- | --- | --- |
| porting plain Triton: recover its TTGIR layouts as Gluon layouts / skeleton | `ttgir_bridge.py` | `$KT/dump_ir.sh --variant plain --out <ir> --kernel-name <k> -- <one-config cmd>`, then `python3 $SKILL/scripts/ttgir_bridge.py recover --ttgir <ir>/plain/plain.ttgir --arch gfx950 --out <dir>` |
| before authoring an async ring / warp pipeline / inline asm: a compile-checked skeleton | `pipeline_examples_cdna4.py` (gfx942: `_cdna3.py`) | `python3 $SKILL/scripts/pipeline_examples_cdna4.py --compile-only --arch gfx950` |
| before sweeping a knob (`num_stages`, `schedule_hint`, `loop_unroll_factor`, `llvm_fn_attrs` sched-strategy / agpr-alloc) or reaching for `warp_pipeline_stage` / scaled MFMA, and whenever a knob gives a flat result: does this Triton build read it (a `dead-declaration` knob is accepted and ignored) | `probe_levers.py` | `python3 $SKILL/scripts/probe_levers.py --all --arch gfx950 --brief` (no GPU, ~1 s) |
| a new LDS layout as an async-copy destination: does it copy back exactly | `smoke_layout_gpu.py` | `python3 $SKILL/scripts/smoke_layout_gpu.py --help` |
| all profile layers in one call (trace, counters, ATT, IR) and a `--compare` against the previous round's directory; for a single layer use the shared signal → tool table linked above | `gluon_profile.py` | `python3 $SKILL/scripts/gluon_profile.py <out> --kernel K --source S --gpu <id> --arch gfx950 -- <cmd>` |
| diagnostics only: re-inject plain's pipeliner, survey pipeline forms | `gluon_swp.py`, `patch_reinject.py`, `pipeline_survey.py` | [`references/gluon/pipeline/reinjection.md`](references/gluon/pipeline/reinjection.md) |

Shared tools live only under `$KT/…`; `scripts/probe.py` is the one forwarder left in `scripts/` (kept for the
pack's selftests).
The upstream standalone workflow's record, gate and stage tools were removed; GEAK's rounds, records and acceptance replace them.

## Arch dispatch: gfx950 (CDNA4) baseline, gfx942 (CDNA3) downgrade

Numbers live in [`hw_constants.json`](../../../hardware/data/hw_constants.json) and
[`sku.json`](../../../hardware/data/sku.json).

| what | gfx950 | gfx942 |
| --- | --- | --- |
| LDS per CU | 160 KiB | 64 KiB |
| LDS banking | 64 banks | 32 banks |
| async direct-to-LDS | `cdna4.async_copy`, exactly 128 or 32 bit per lane (64 bit fails) | 32-bit only, `order` = source fast dim (`[1, 0]` row-major) |
| read-with-transpose | `ds_read_b64_tr_*` | absent |
| MFMA | `version=4`, block-scaled MFMA (MXFP8/6/4) | `version=3`, no scaled MFMA |
| fp8 spelling | OCP `fp8_e4m3` (`gl.float8e4nv`) | FNUZ `fp8_e4m3_fnuz` (`gl.float8e4b8`); FNUZ reaching gfx950 silently upcasts to fp16; OCP fp8 on gfx942 fails or is misread as FNUZ (Gluon) or upcast (`tl.dot`) |
| regular fp8 / bf16 MFMA shapes | fp8 `16x16x32`, `32x32x16`; bf16 also `16x16x32`, `32x32x16`; fp8 `16x16x128` / `32x32x64` and all fp4 only via `mfma_scaled` | bf16 `16x16x16`, `32x32x8`; FNUZ fp8 `16x16x32` |
| compiler-derived padded LDS layout | `cdna4.compute_efficient_padded_shared_layout` (3.8.0) | none — `PaddedSharedLayout.with_identity_for` by hand |
| buffer ops / wave size | `gl.amd.cdna4.buffer_load/store` (offsets in elements); wave64 | `gl.amd.cdna3.buffer_load/store`; wave64 |

Look the rest up, do not recall it: `python3 kernel_workflow/scripts/kernel_tools/amd_isa.py --arch gfx950 search "<words>"`
(MFMA shapes, layouts, waitcnt, hazards, LDS banking), `extract_sku.py --facts --hardware-context $EVAL_DIR/hardware_context.json` (CUs, XCDs, clocks, peaks; the file is written once per run, no GPU work, by `python3 kernel_workflow/scripts/kernel_tools/hw_context.py --output $EVAL_DIR/hardware_context.json` from the GEAK root; never guess `--sku`: MI350X and MI355X differ),
`hw_budget.py --brief` (floor and headroom). Support level per target and Triton version:
[`references/hardware/capability-matrix.md`](references/hardware/capability-matrix.md).

## Do-no-harm notes

Advisory only; correctness on every case before timing; confirm the Gluon kernel actually ran; no serve case
may regress beyond noise — [`playbook.md ## Do-no-harm`](playbook.md#do-no-harm).

## Roles in GEAK

No new roles. `tech_lead` decides whether Gluon (or a port) is a direction and may put `playbook.md` in
`KK_REFS`, planning `deep_explore` or other directions as usual. `engineer` / `deep_engineer` use the
procedure, tools and docs; `profile_engineer` / `benchmark_engineer` the evidence readers and timing notes
([`references/method/profile.md`](references/method/profile.md),
[`references/method/benchmark-hygiene.md`](references/method/benchmark-hygiene.md)). `verify_engineer` and
`director` are unchanged: acceptance is the frozen `COMMANDMENT.md` re-benchmark and the `MIN_IMPROVE` gate.

## Sources

Upstream `tile-programming-gluon` pack
([`AMD-AGI/TileProgrammingAgentSkills@a649eaaf`](https://github.com/AMD-AGI/TileProgrammingAgentSkills/tree/a649eaaf0a989de949f38cd00916416c95d59b95)),
reorganized gfx950-first for GEAK; merge history in [`PROVENANCE.md`](PROVENANCE.md). Compiler-behaviour
statements carry their basis inline: most were checked by CPU AOT compile against the Triton 3.8.0 wheel for
gfx950 / gfx942; others are source reads, checks on 3.7.1 / 3.9.0, or marked as reasoning, not measured.
