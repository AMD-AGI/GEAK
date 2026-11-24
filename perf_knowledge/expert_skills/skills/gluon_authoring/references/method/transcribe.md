# Transcribe: the champion's TTGIR → a verified Gluon anchor

Advisory port path: plain-Triton champion `.ttgir` → recovered layouts → explicit `@gluon.jit` anchor
→ equivalence check. Transcription reproduces the compiler's inferred layouts; it is not optimization.
The anchor then goes to `recover.md`. `$SKILL` = `perf_knowledge/expert_skills/skills/gluon_authoring`;
`$KT` = `kernel_workflow/scripts/kernel_tools`. **Always pass `--arch` = the arch the dump was compiled
for**; a wrong arch surfaces as ``Dialect `amdg' not found`` (reads like a corrupt dump).

Port entry: [`## Port checklist`](#port-checklist-verified-on-triton-38--gfx950) (the steps with their checks), then
[`## Port error strings and symptoms`](#port-error-strings-and-symptoms).

## Port checklist (verified on Triton 3.8 / gfx950)

The shortest route from a plain `@triton.jit` kernel to a correct, launched, timed Gluon port. Each step
names its check; do not start the next step on a failed check. Worked on an fp8 GEMM (`tl.dot` on fp8,
per-M config buckets) to parity with the tuned plain kernel; the detail behind each step is further down
this page and in [`recover.md`](recover.md#the-async-ring-loop-form-that-reaches-plain-parity-gfx950).
`$KT=kernel_workflow/scripts/kernel_tools`, `$SKILL=perf_knowledge/expert_skills/skills/gluon_authoring`.

0. **A one-config driver.** A script that launches the op once at one shape (the task's own inputs and
   reference) and prints the max error. `dump_ir.sh` keeps one compilation per run, so one shape = one
   config bucket per dump. *Check:* the script prints a passing error for every bucket you will port.
1. **Tool sanity.** `python3 $SKILL/scripts/ttgir_bridge.py --selftest` → `SELFTEST PASS`.
2. **Dump plain twice per config bucket**: at its tuned config and at `num_stages=1` (pass a config dict
   with `num_stages` overridden).
   `bash $KT/dump_ir.sh --variant plain_<bucket> --out ir --kernel-name <kernel> -- python3 one.py <shape>`
   *Check:* the echoed `num_warps` equals the bucket's; the echo's `num_stages` comes from the compile
   metadata (the pipeliner consumes the TTGIR attribute) and `LDS ring depth` from the ring alloc.
   Cross-check on the dump: `grep -o 'memdesc<[0-9]*x' ir/plain_<bucket>/*.ttgir | sort -u` (leading dim =
   stages) and `grep -c buffer_load_to_local` (async staging on gfx950).
3. **Recover both dumps.** `python3 $SKILL/scripts/ttgir_bridge.py recover --ttgir <dump> --arch gfx950
   --out rec_<bucket>.py --json rec_<bucket>.json`. *Check:* `num_warps cross-check: PASS`,
   `UNRECOVERABLE: 0`, `round-trip: EXACT=<all>`; note the `COMPILED FORM of the N buffer_load site(s)`
   line (bare / mask-only / mask+`other`) — that, not the source, is what you transcribe. The
   `num_stages=1` dump gives the register-staged family (blocked loads, swizzled LDS) for the anchor; the
   tuned dump gives the async-ring family (linear load layouts, padded LDS) for step 6.
4. **Write the anchor from the `num_stages=1` layouts.** Build layouts per bucket on the host and pass
   them as `gl.constexpr` kernel arguments (one kernel, many buckets), or bind an in-kernel layout to a
   `gl.constexpr` name (an un-annotated local fails with `cannot convert <Layout>(...) to tensor`). Every `gl.arange` takes
   `gl.SliceLayout` of the recovered load layout; loads/stores are `gl.amd.cdna4.buffer_load/store` with
   **element** offsets (not bytes); the accumulator is `gl.zeros(..., layout=MMA)`; an fp8 `tl.dot` is
   `mfma_scaled(a, None, "e4m3", b, None, "e4m3", acc)`. Keep plain's `% M` / `% N` wrap and the
   bucket's `num_warps`. `waves_per_eu` is left out of the faithful anchor (it caps occupancy; A/B it
   separately).
5. **Check it is right and is the kernel that ran.** (a) numerics on every serve and edge case;
   (b) launch trace — `torch.profiler` CUDA event names (or `rocprofv3 --kernel-trace`) must show the
   Gluon kernel's name on every case you route to it; (c) ISA: `python3 $KT/asm_loop_audit.py <ir>/<v>.s
   --arch gfx950` shows `buffer_load_dwordx4` (16 B per lane) in the hot loop and no `global_load_ubyte` /
   `global_load_dword` — a scalar global load there means a tensor is still on the translator's scalar default layout;
   (d) time it against `plain@num_stages=1`: a faithful anchor lands on it.
6. **Recover the overlap**: an `async_copy` ring with plain's depth and the tuned dump's layouts, in
   plain's compiled loop form (bare steady-state copies, wait → issue next → LDS read → MFMA, peeled
   drain). Recipe and the traps that cost time:
   [`recover.md ## The async-ring loop form`](recover.md#the-async-ring-loop-form-that-reaches-plain-parity-gfx950).
   *Check:* hot loop has no `v_cndmask` the plain loop lacks; `probe.py measure --dir` shows the same LDS
   bytes and WGs/CU as plain; interleaved A/B vs plain on every serve case.
7. **Dispatch per shape**: Gluon only on buckets where it wins beyond noise, plain elsewhere, and any
   bucket you have no recovered layouts for goes to plain ([`dispatch-table.md`](dispatch-table.md)).
   Re-run step 5(a)(b) after wiring the dispatch.

**`ttgir_bridge.py verify` on an fp8 port.** `mfma_scaled(..., None, ...)` materialises unit E8M0 scale
constants (`ARITH_CONSTANT linear shape=[BM, BK/32]`) that plain's `tt.dot_scaled` does not have; verify
lists them as `reconciled` (reason `unit_scale_constants_of_mfma_scaled`), not as `EXTRA` (a build of the
tool without that rule reported `LAYOUT EQUIVALENCE: FAIL` with only those rows). Any `MISSING` row is real. The ISA mnemonic differs the same way
(`v_mfma_scale_f32_16x16x128_f8f6f4` vs plain's `v_mfma_f32_16x16x128_f8f6f4`); measured, no cost.

**If GEAK seeded a Gluon kernel before round 1** (a `triton2gluon` task can arrive with a
"simplest-correct" seed already routed): treat the seed as an anchor to check, not as a transcription —
dump it, `asm_loop_audit` it (step 5c), and compare its time to `plain@num_stages=1`; a seed well below
that line is a layout defect, not a missing pipeline.

## Port error strings and symptoms

Verbatim errors and symptoms hit on the port route (Triton 3.8 / gfx950), each with its cause and fix.

- **`no matching matrix core intrinsic for mfma version 4 with instruction shape [32, 32, 32] and element types A='f8E4M3FN', B='f8E4M3FN'`**
  — the translator skeleton's `helpers.tl_dot` picked a default MMA for fp8. Use the recovered MMA
  (`instr_shape=[16, 16, 128]` for fp8 on gfx950) and `gl.amd.cdna4.mfma_scaled` with `None` scales.
- **`LLVM Translation failed for operation: builtin.unrealized_conversion_cast`** (location points at a
  pointer argument) — a 16-byte `async_copy.buffer_load_to_shared` on rows that are not 16-byte aligned
  (fp8 `K % 16 != 0`). Plain's pipeliner silently falls back to register staging there (its TTGIR has no
  `buffer_load_to_local`). Route such shapes to the register-staged kernel or to plain.
- **`failed to legalize operation 'amdg.buffer_load' that was explicitly marked illegal`** — `gluon_swp`
  with `buffer_ops=True` on a body that already uses `buffer_load`
  ([`reinjection.md`](../gluon/pipeline/reinjection.md#precondition-for-re-injection-plain-loads-and-a-dot-in-the-loop)).
- **Port an order of magnitude slower than plain, bit-exact, `global_load_ubyte` in the hot loop, 256 VGPRs** — layouts
  declared (or applied only at the dot) while the loads stay on the translator's scalar default layout (§3 "Declaring a layout is
  not applying it"). Fix every `gl.arange` / load.
- **Wrong output only through 32-bit operands (scales, bf16 store), fp8 operands fine** — buffer offsets
  written in bytes; `buffer_load` / `buffer_store` / `buffer_load_to_shared` offsets are in **elements**
  (a 1-byte element hides the bug).
- **Recover NOTE `ttg.local_alloc have no initialiser and are written by N async copies`** — the
  allocations are written by plain's async copies; their layouts are the recovered `A_SMEM` / `B_SMEM`.
  Nothing to derive. (An older tool printed `NO initialiser and ZERO ttg.local_store ... DERIVE its layout
  ... by hand` here; that NOTE is real only when the dump has no `buffer_load_to_local`.)
- **No `A_LOAD` / `B_LOAD` rows, only `TO_SMEM` / `TO_SMEM_2`** — async-staged dump. The two names
  swap between buckets of the same kernel; tell A from B by the provenance line (`<- a_ptrs` /
  `<- b_ptrs`) or by the 16-byte vector dimension (`reg_bases[0]`: `[0, 1]` = K-contiguous A).
- **Recover: `PIPELINE ... INCONCLUSIVE: no tt.num_stages attribute`; dump_ir: `num_stages=?`** — the
  pipeliner consumed the attribute and the tool found neither an indexed LDS ring nor compile metadata
  beside the dump (current tools read the depth from `memdesc<Sx…>` + `memdesc_index`, or `meta_*.json`).
  Read the depth from `memdesc<Sx…>` and count `buffer_load_to_local` (step 2).
- **Port ring with `other=0.0` on the async copies is markedly slower** (TTGIR shows
  `buffer_load_to_local ... other = %cst`) — plain's dumps carried no `other` on these sites.

## Tools

| tool | role |
| --- | --- |
| `$SKILL/scripts/ttgir_bridge.py recover \| verify \| view` | **the recovery and layout-diff gate.** Parses `.ttgir` with the compiler's MLIR parser and converts via upstream `layoutToGluon()` (`GluonOpBuilder.get_gluon_layout_from_tensor / _from_memdesc`); every layout carries a round-trip proof; `verify` compares **LinearLayout normal forms**, never attribute text or counts (plain's pipeliner unrolls, so counts compare unroll factors) |
| `$SKILL/scripts/recover_gluon.py` | anchor assembly: `gl.constexpr` layout preamble, `--with-skeleton` translator skeleton. **Its `--verify` compares attribute text as a set — sound one way only; not the gate** |
| `$SKILL/scripts/ttgir_to_gluon.py` | pure-text fallback when `import triton` is unavailable; output is a starting point (a hand mapping once dropped `tilesPerWarp` / `elementBitWidth`) |
| `$KT/dump_ir.sh` | dump: per-variant `TRITON_CACHE_DIR`, `.ttgir/.llir/.amdgcn`, stripped `.s`, `meta_*.json` (LDS bytes/WG) |
| `$KT/probe.py measure --dir` | compile-only occupancy, both limiters, no GPU |

Hold the champion's config (`BLOCK_*`, `num_warps`) fixed for the recovered anchor: its layouts are literals
with `warps_per_cta` baked in, so another `num_warps` fails to compile (`Result has an invalid layout`;
compile-checked 3.8.0) and another tile needs a new recovery from plain's dump at that tile. To make tiles a
knob, rewrite the layouts as constexpr functions of `BLOCK_*` / `num_warps`
(`../tile-programming/layout-recipes.md ## Compute the layout instead of tabulating it (constexpr_function)`) and verify each swept
config against plain's recovered layouts (`ttgir_bridge.py verify`) before timing it. Record any deliberate deviation from the champion's construct and
measure the faithful variant too (`probe.py measure` costs seconds).

**Existing Gluon source can carry inline-asm sites** — audit them (match `inline_asm`, not only
`gl.inline_asm`): `../gluon/inline-asm/field-guide.md ## Reverse index: I see X in an unknown kernel`.

## 0. Preconditions

- The `.ttgir` is the **original** dump of the champion at its pinned config (check its
  `ttg.num-warps`). In GEAK there is no champion bundle: dump plain at the task's own tuned config per
  bucket (and at `num_stages=1` as the control), never at a config the task does not run; a hand-cleaned copy (`#loc` aliases stripped) does not parse.
- `import triton` works (CPU-only is fine; **upstream wheels ship the AMD backend** — `pip install
  triton==3.7.1` reports `['amd', 'nvidia']`, so `--arch gfx950` works without ROCm).
- Tool Triton ≥ 3.7 (3.8.0 reference) and at least as new as the dump's compiler.

### Triton version support (measured, not assumed)

| build | recover | verify basis | `view` |
| --- | --- | --- | --- |
| 3.8.0 / 3.7.0 / 3.7.1, vendor main | works | LinearLayout normal form | works |
| 3.6.0 (downgrade) | works | distributed = normal form, **shared = canonical text** (stricter) | absent |

- **3.6 cannot parse a dump containing `ttg.barrier`** (op added in 3.7); a dump without one recovers
  byte-identically. Failure is loud at parse time.
- **Recovered layouts were byte-identical across 3.6.0–3.8.0 for dumps without `ttg.barrier`** (16 such
  kernels on 3.6.0, 3.7.0, 3.7.1 and 3.8.0: identical recovered counts and layout constants for every
  kernel; 3.6.0 cannot parse a dump that has one, see above), so for those a
  layout preamble may be carried across versions; compare with `constants-digest`.
- **Perf ratios did not carry across a Triton bump in the cases recorded**: one attention plain regressed sharply on 3.8.0 vs
  3.7.1 while its anchor moved little, so the anchor/plain ratio jumped with no anchor gain; a GEMM's
  clear win collapsed to parity. Re-measure absolute times after a bump
  (`benchmark-hygiene.md ## Cross-version ratios: attribute the move before claiming it`).
- On 3.6, `to_linear_layout` of a shared layout aborts the interpreter; capabilities are probed in a
  child process (`TTGIR_BRIDGE_CAPS=...` skips the ~0.4 s probe).

## 1. Dump — and pin the body

```bash
python3 "$SKILL/scripts/ttgir_bridge.py" --selftest     # ~10 s, offline
bash "$KT/dump_ir.sh" <compile cmd> --variant plain --out ir/ --arch gfx950 [--kernel-name <substr>]
#   -> ir/plain/plain.{ttgir,llir,amdgcn,s} + meta_*.json
```

- **`dump_ir.sh` takes the freshest cache artifact: on a multi-kernel op (attention + split-K reduce)
  you get whichever compiled last.** It warns and lists candidates; pin with `--kernel-name <substring>`.
  `--kernel` is `module.path:object` for `--emit-gluon`; a bare name there is rejected (exit 1).
- **`dump_ir.sh` exit codes:** 0 dumped | 1 no compile command / bare `--kernel` | 2 unknown option
  before the command, invalid `--variant` / `--emit-gluon` | 3 `--emit-gluon` without
  `recover_gluon.py` | 4 `--emit-gluon` with no `--arch` given or detectable.
- `--knobs LLIR_SCHED|AMDGCN_AS|RA_HINTS` (`TRITON_ENABLE_*`) are fork-only — not reachable on the stock
  3.8.0 wheel (silent no-op); confirm with `probe_levers.py --all`.

## 2. Recover

```bash
python3 "$SKILL/scripts/ttgir_bridge.py" recover --ttgir champion.ttgir --arch gfx950 \
    --out anchor_layouts.py --json layouts.json          # [-v] every use site; [--force]
```

Read in order:

- **`num_warps cross-check` must be `PASS`; a FAIL exits 4** = dump internally inconsistent, discard it.
  (Exit 1 = an `UNRECOVERABLE` layout: part of the kernel is inexpressible, the rest is sound.)
- **`UNRECOVERABLE: N` must be 0. The one seen in practice is `amd_rotating_shared`** (plain's rotating
  operand staging, on some attention-shaped bodies, every minor 3.6–3.8): Gluon has no constructor. Not
  every attention body carries it (an fp8 MQA-logits kernel recovered with `UNRECOVERABLE: 0` on Triton 3.8 /
  gfx950) — read the count rather than expecting it. Order:
  (1) probe your build (`amd_wmma` once sat behind the same wording yet was constructible);
  (2) re-dump at `num_stages=1` and re-recover; (3) still unrecoverable → write `gl.convert_layout` and
  let the backend choose the staging (disclose it; `--allow-unrecoverable` only lets `recover` exit 0),
  or report it as a structural blocker. The tools cannot read the encoding back (on 3.7.1/3.8.0 no Python
  binding gets an encoding attribute out of a Value/Type; missing: `MemDescType::getEncoding()`), but the
  layout is linear and an exact `SharedLinearLayout` can be built from the upstream formula and checked
  against it element by element (`../pitfalls/negative-patterns.md ## Quick Reject Checklist (= gate skip-transcription conditions)`).
- **`round-trip: EXACT=N`** — every layout must be EXACT; `DIFFERS` = upstream converter lost a field.
- **Roles** (`A_LOAD`/`B_LOAD`/`A_SMEM`/`B_SMEM`/`*_DOT_OPERAND`/`MMA`/`INDEX`) are derived from
  `opIdx` backwards through `local_load` → `local_store` → load. **Trust the source names over role
  names**: on attention three loads feeding `local_alloc` are indistinguishable by rank, `FROM_SMEM` can
  be the layout going *into* shared. Each provenance line ends with source variable and line, and the
  emitted file has a **`# DOTS`** block per `tt.dot` (use it on multi-dot bodies sharing one `#mma`).

### Four more `recover` outputs worth acting on — layout equivalence is blind to all of them

- **`COMPILED FORM of the N buffer_load site(s)`: bare / mask-only / mask+`other`.** Transcribe the
  compiled bucket, not the source: **`tl.load(..., other=0.0)` often compiles to a `buffer_load` with
  neither operand because buffer OOB returns 0 on CDNA; passing `other=` in Gluon emits a `v_cndmask`
  per register (a measurable loss; on an async copy worse — the `other=` row of `## Port error strings and symptoms`)**,
  and an extra mask costs the same. `contiguity` and
  `stride` are not reachable from `gl.amd.cdna3.buffer_load`.
- **`LDS: N allocation(s), M element(s)`** — an upper bound (user allocations totalled by scope; plain
  allocates by liveness), divisor from `--arch` (160 KiB/CU gfx950, 64 KiB/CU gfx942). Once the anchor
  compiles, quote `probe.py measure` instead.
- **`op(s) ... have NO gluon.language equivalent`** — e.g. `amdg.in_thread_transpose`; arch-dependent:
  gfx950 emits `ds_read_tr` instead and the anchor inherits it.
- **`constants-digest`** — compare two recoveries of one body with it, not a file hash (role names
  drift: `A_LOAD` at ns=2 = `FROM_SMEM` at ns=1). Within one arch only (MFMA `version=4` gfx950 vs
  `version=3` gfx942).

Coverage: `#blocked`, `#amd_mfma`, `#swizzled_shared`, `#padded_shared`, `#linear`, `ttg.dot_op`,
`ttg.slice`. `convert_layout` **placement is manual**
(`../tile-programming/layout-recipes.md ## TTGIR -> Gluon recovery map (for transcription)`). Spill/RA is
not in TTGIR.

## 3. Apply — the step that carries the port

**Declaring a layout is not applying it: a translated body whose index tensors keep the translator's default
falls back to a scalar default `BlockedLayout` and every load becomes uncoalesced** (hand-written index math on
an unresolved `AutoLayout` fails to compile instead) — bit-exact, passes the oracle, several times
slower. On one kernel wiring the recovered layouts onto the index tensors was worth more than everything
else done in Gluon that run (largest single win). `verify` then reports exactly the unapplied layouts as
`missing`.

```bash
python3 "$SKILL/scripts/recover_gluon.py" --ttgir champion.ttgir --arch gfx950 --out plain.gluon.py \
    [--with-skeleton --kernel module.path:object]
# or: bash $KT/dump_ir.sh <cmd> --variant plain --out ir/ --arch gfx950 --emit-gluon layouts|anchor
```

- **`--with-skeleton` / `--emit-gluon anchor|pipeline` need `triton.tools.triton_to_gluon_translator`
  (3.8 spelling); 3.6.0–3.7.1 ship `triton_to_gluon_translater` with only `convert_triton_to_gluon(src)`.**
  Decide from the import, not `triton.__version__`; on failure it degrades to layouts-only. The Triton 3.8
  ROCm wheel ships it: `python3 -c "from triton.tools.triton_to_gluon_translator import translate_paths"`. The
  translator drops `num_stages` and re-infers default layouts — replace them. **Its skeleton does not
  compile for an fp8 `tl.dot` on gfx950** (`helpers.tl_dot` defaults to MMA `[32, 32, 32]`, which has no fp8
  intrinsic; error string in `## Port error strings and symptoms`): use
  `gl.amd.cdna4.mfma_scaled(a, None, "e4m3", b, None, "e4m3", acc)` on the recovered `[16, 16, 128]` MMA.
- **A `@triton.heuristics` / `@triton.autotune` wrapped kernel translates to an empty result without
  raising** (upstream `getattr` gets the wrapper). `recover_gluon.py` peels to the `JITFunction` and names
  the wrapper on stderr; point `--kernel` at the kernel.
- `--emit-gluon pipeline` scaffolds are a separate post-anchor experiment, never part of the anchor.

```python
offs_am = gl.arange(0, BM, layout=gl.SliceLayout(1, A_LOAD))     # index math, NOT AutoLayout
a_smem  = gl.allocate_shared_memory(dtype, [BM, BK], A_SMEM)
a_op    = a_smem.load(A_DOT_OPERAND)
acc     = gl.full([BM, BN], 0.0, gl.float32, layout=MMA)
```

Checklist: every `arange`/`expand_dims` chain uses a `SliceLayout` of its consumer's layout; every
`allocate_shared_memory` the recovered `*_SMEM`; dot inputs `*_DOT_OPERAND`; accumulator `MMA`; launch
with the emitted `NUM_WARPS` (another warp count contradicts the literal `warps_per_cta`; the cases checked
failed to compile with `Result has an invalid layout`); each `ttg.local_alloc` → its own `allocate_shared_memory`; each buffer_load uses the compiled
operand bucket; rebuilt store index tensors keep every offset term (dropping the block-row base aliases
all blocks).

- **`gl.convert_layout(x, L, assert_trivial=True)` fails at compile time if the conversion is not a
  no-op** — use after `reshape`/`permute`/`join` chains whose result equals a dot-operand layout without
  carrying the attribute (the MFMA verifier rejects that).
- **Barriers are inserted by the compiler on the AMD Gluon path** (membar runs in the shared
  `TritonGPUToLLVM` conversion, although `gluon_to_ttgir` has no membar pass). Probed: removing all four
  `gl.barrier()` from a working anchor stayed correct and still emitted 6 `s_barrier` (7 with them).
  A hand `gl.barrier()` then mostly repositions a sync; a hazard the membar analysis does not see
  (e.g. a direct-to-LDS refill) still needs an explicit fence.
- **Carrying the champion's `waves_per_eu` pins occupancy** (`waves_per_eu=N` emits `amdgpu-waves-per-eu`
  "N, N", compile-checked 3.8.0): it can drop the anchor below its natural occupancy (one recorded anchor
  lost a wave/SIMD at unchanged VGPRs and ran slower). A/B it — another record shows forcing a higher value
  losing to AGPR shuffles / spills.

### Not every `ttg.local_alloc` should become an `allocate_shared_memory`

Faithful form = one `allocate_shared_memory` per `local_alloc`. Collapsing a **pass-through** site
(`local_alloc` immediately followed by its only `local_load`, no cross-iteration reuse) into
`gl.convert_layout` lets the compiler share scratch across conversions (user allocations are live for
the whole function and charged separately) — an elective change to measure, not a rule:

1. **Classify against an `ns=1` dump**: at the shipped depth the staging is often the pipeliner's
   (two kernels read "staged" at ns=2 and "pass-through" at ns=1).
2. `verify` sees the choice when the shared layout is recoverable (hard FAIL naming the missing
   `swizzled_shared`); it never sees allocation size.
3. **On gfx950 / Triton 3.8.0 register-path GEMMs every pass-through variant tried was slower** (bit-exact,
   less `shared`). Tells: `s_barrier` per loop rises (scratch reuse = a full-drain barrier per reuse), an
   N-contiguous operand loses the hardware transpose (`ds_read_b64_tr` disappears from the loop), VGPRs can
   tip into spilling, and an LDS saving buys nothing when registers already bind occupancy. A single faster
   case from an earlier campaign was not reproduced. A win is plausible only where the explicit allocation
   is the binding occupancy limiter; A/B it there.

```bash
python3 "$KT/probe.py" measure --dir ir/anchor/
#   anchor.amdgcn [gfx950] vgpr= NNN waves/SIMD=N  spill=0 B
#   anchor.amdgcn lds/WG= NNNNN B  WGs/CU by LDS<=N  regs<=N -> N WGs/CU, REGISTERS bind
```

## 4. Compile the anchor and dump its TTGIR

```bash
bash "$KT/dump_ir.sh" <anchor compile cmd> --variant anchor --out ir/ --arch gfx950
python3 "$KT/probe.py" measure --dir ir/anchor/
```

Same config, emitted `NUM_WARPS`. Dump the champion as `--variant champion` for attribution
(`recover.md`).

## 5. Verify — the equivalence gate

Four checks: layout equivalence (`verify` PASS/RECONCILED), numerics at the champion's tolerance on
task **and boundary** shapes (Gluon writes every mask by hand; tile-multiple task cases hide a missing
mask), determinism (`benchmark-hygiene.md ## Determinism race-test (async / barrier / pipeline / layout
changes)`), asm parity (mix **and** sequence). A layout that emits beautiful asm can still be
numerically wrong.

### 5.1 Layout equivalence — `ttgir_bridge.py verify`

```bash
python3 "$SKILL/scripts/ttgir_bridge.py" verify --plain champion.ttgir --anchor ir/anchor/anchor.ttgir \
    --arch gfx950 --json verdict.json
```

| exit | state | meaning / next |
| --- | --- | --- |
| 0 | `PASS` | every plain layout reproduced, none extra (normal forms) |
| 0 | `RECONCILED` | every difference has a named structural cause (below); read them, continue |
| 1 | `FAIL` | unexplained MISSING / EXTRA. **MISSING blocked + EXTRA linear at the same shape = §3 not done** (the compiler chose because you did not) |
| 3 | `NOT_COMPARABLE` | num_warps / threads_per_warp / MMA family differ — re-dump at the same config; do not read the diff |

RECONCILED causes: (1) EXTRA at a shape where plain had an UNRECOVERABLE layout (probable, shape-matched
only); (2) MISSING but the same constructor exists at another shape (non-pipelined anchor vs pipelined
plain's extra K-shapes; `EXTRA` empty); (3) MISSING from an op Gluon cannot express
(`amdg.in_thread_transpose`).

**fp8 `tt.dot` can lower in plain to an unscaled `tt.dot_scaled`, which Gluon cannot spell** (plain
mfma builtin rejects the type; on Triton 3.8.0 the scaled one materializes a scale even with `None`,
compile-checked) → a correct `FAIL`, a real
expressibility wall. The `MULTIPLICITY` table never gates (`CVT_DST plain x1 anchor x8` = 8 relayouts
where plain paid one — a perf finding).

### 5.2 Numeric equivalence, at tolerance

Through the task harness, or `recover_gluon.py --verify --harness '<harness> --correctness'` (looks for
`CORRECTNESS PASS`). **`recover_gluon.py --verify` without `--harness` reports correctness `not-run` and
still exits 0.** Un-diverged anchor ⇒ expect bit-parity. Check `isfinite` first.

### 5.4 Asm parity — instruction MIX and class SEQUENCE

Mix: `$KT/asm_loop_audit.py <f>.s --opcodes N`. Sequence: normalized line diff of the op-class symbol
stream (strip `.loc`/`.Ltmp`, same build both sides;
`../tile-programming/compiler-contract.md ## Auditing the hot-loop schedule`). A histogram alone cannot
fail a reordering. Matched mix + clustered MFMA ⇒ the residual is the missing pipeliner overlap.

## 6. Inspecting a layout before choosing one

```bash
python3 "$SKILL/scripts/ttgir_bridge.py" view --ttgir f.ttgir --role A_SMEM --arch gfx950 [--hardware]
```

Per-lane table; pick among numerically identical swizzles by their `ds_read`/`ds_write` pattern before
timing. Not on 3.6.

## 7. Version-agnostic recovery: the rule, and how to check it when the tools drift

When a tool breaks on another build, read the layout definitions at the top of the layout-resolved IR
(tiling vectors, warp split, swizzle/pad, k-width, bases) and match each field to this build's Gluon
constructor **by meaning** (introspect `gluon.language`). A renamed construct is an API blocker to
record, not "Gluon can't express it".

**Small/odd async layouts: recover them when you can.** For a sub-tile the proven set does not cover (half
head-dim, odd contraction width), compile a standalone plain kernel at that exact sub-tile, dump it, and
reuse the inferred `DistributedLinearLayout` / `PaddedSharedLayout`. A hand-derived layout also lowers when it
follows the async-copy rules (128- or 32-bit per lane, `threads_per_warp * size_per_thread` == contiguous
extent, swizzle `vec >= size_per_thread`, padded producer derived from the destination's `offset_bases`;
compile-checked 3.8.0 gfx950). Violations fail loudly at LLVM translation (`failed to translate module to
LLVM IR`), except `vec < size_per_thread`, which compiles and is silently wrong, so copy-back-check any hand
layout. See also `../tile-programming/layout-recipes.md`.

## Offline check

```bash
python3 "$SKILL/scripts/ttgir_bridge.py" --selftest      # SELFTEST PASS, no GPU
python3 "$SKILL/scripts/ttgir_to_gluon.py" --selftest
python3 "$SKILL/scripts/recover_gluon.py" --selftest
bash "$SKILL/scripts/smoke_test_recover.sh"
```

Validate tools on clean upstream wheels, not a vendor tree (which carries a local pipeliner re-injection
into `gluon_to_ttgir` and `TRITON_GLUON_*` names absent from every upstream 3.6–3.8).
