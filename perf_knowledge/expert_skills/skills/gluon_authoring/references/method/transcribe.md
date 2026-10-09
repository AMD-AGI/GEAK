# Transcribe: the champion's TTGIR → a verified Gluon anchor

**What this stage decides.** Whether you hold an **anchor** — an explicit Gluon kernel whose layouts
are *proven* to be the champion's — and a recorded verdict on whether the transcription landed. It
does **not** produce a fast kernel. Transcription is not optimization: it reproduces the champion's
compiler-inferred layouts explicitly in Gluon to create an **equivalence anchor** under full author
control, and that anchor is the measurement baseline for everything after it.

**When you are here.** Entry mode **A** only (`entry.md`, "Which entry mode"), and only after
`champion_gate.py` has passed (`entry.md ## Stage-Entry: the champion assertion`). In mode **B** (the
source is already explicit Gluon) this stage is a defined no-op — except the inline-asm audit below,
which applies to any anchor that arrives as existing Gluon source. In mode **C** a resumed run has
already paid this, or carries it unpaid into `recover.md`.

**Where it goes next.** A PASS (or RECONCILED) anchor goes to `recover.md`: re-profile, attribute the
anchor→champion gap, repay it, clear the parity gate. Do not optimize here.

Paths: `$SKILL` = `perf_knowledge/expert_skills/skills/gluon_authoring`; `$KT` =
`kernel_workflow/scripts/kernel_tools` (`dump_ir.sh`, `probe.py`, `amd_occupancy.py`,
`asm_loop_audit.py`, `layout_facts.py` now live there; `$SKILL/scripts/<name>` shims keep the old paths
working). gfx950 is the main line in every command; gfx942 is a downgrade note. **Pass `--arch` for
the arch the dump was compiled for, every time** — a tool that needs an arch must not be left to
default one.

---

## Stage-Anchor: the transcription, and what it is allowed to be

The conversion is a **1:1 structural transcription of the champion's lowered IR into explicit Gluon
constructs** — register/thread tiling, the MFMA output tiling, the shared staging layout, each
dot-operand view, every layout conversion *at the same program point*. Nothing else. The opening move
is mechanical, not a design exercise: the layouts plain Triton's compiler inferred are recorded in the
tuned kernel's `.ttgir`, so there is nothing to search over — any two agents transcribing the same
pinned `.ttgir` land on the same layouts. That is why it is one fixed action and not a fan-out.
Deterministic does not mean single-shot: the layout-diff is something you converge on over several
passes (§5).

**The tools, and which one owns what.**

| tool | role here |
| --- | --- |
| `$SKILL/scripts/ttgir_bridge.py recover \| verify \| view` | **the recovery and the layout-diff gate — prefer this.** Hands the `.ttgir` to the compiler's own MLIR parser and then to upstream's `layoutToGluon()` (`python/src/gluon_ir.cc`, reached as `GluonOpBuilder.get_gluon_layout_from_tensor / _from_memdesc`), so there is no mapping table in this pack that can fall behind Triton. An unsupported kind surfaces as a named `UNRECOVERABLE` row, never a plausible wrong constructor; every layout carries a round-trip proof; `verify` compares **LinearLayout normal forms**, never attribute text and never attribute *counts* (plain's auto-pipeliner unrolls the loop, so counting occurrences compares unroll factors and can never pass against a pre-pipeline anchor) |
| `$SKILL/scripts/recover_gluon.py` | **anchor assembly only**: the recovered `gl.constexpr` layout factory, and under `--with-skeleton` the official translator's algorithm skeleton (control flow / masks / dtype). Its own `--verify` compares canonical attribute **text** as a set — sound in one direction only (equal text means equal layout, but two spellings of one layout read as different) — so do not use it as the gate |
| `$SKILL/scripts/ttgir_to_gluon.py` | the pure-text parser/emitter underneath `recover_gluon.py`; **the fallback when `import triton` is unavailable**. Its output is a starting point, not a proof: a hand-written mapping silently emits a layout missing a field whenever upstream adds or renames one (`tilesPerWarp` / `elementBitWidth` were once being dropped) |
| `$KT/dump_ir.sh` | the dump: per-variant `TRITON_CACHE_DIR`, copies `.ttgir/.llir/.amdgcn`, strips `.amdgcn` into a stable `.s`, writes `meta_*.json` (LDS bytes/WG) |
| `$KT/probe.py measure --dir` | compile-only occupancy — both limiters, seconds, no GPU |

Three properties make the anchor worth what it costs:

- **Hold the pinned config fixed through the whole run.** `BLOCK_*` is coupled to the recovered layout
  family (`warps_per_cta` ties mma / dot / shared / global), so a tile change means re-recovering the
  entire layout set and invalidates the anchor everything is measured against — a `resweep_request`,
  not a lever (`entry.md`, routing out of entry).
- **Every difference from the champion is NAMED, and the faithful version is still measured.** The
  constraint is attributability, not slowness — see the divergence ledger below. What destroys the
  anchor is an *unrecorded* improvement, because then a later delta cannot be split into what
  transcription lost and what the explicit tier bought.
- **All four equivalence checks run, not just numerics** (§5): layout-diff (`ttgir_bridge.py verify`
  — catches a silent mis-recovery, wrong `order` / `kWidth` / swizzle, that still passes the oracle),
  the numeric oracle at tolerance, determinism over ~40 launches (a racing async mis-recovery is
  invisible to a single-shot compare), and asm parity (instruction mix **and** sequence). `asm looks
  perfect` is not one of them: a recovered layout can emit beautiful asm and be numerically wrong.

**An anchor that arrives as existing Gluon source, rather than as a recovered TTGIR, can carry
inline-asm sites you did not write — and 39% of surveyed production files do.** Those sites are not
transcription output and no equivalence check covers them. Before the first round, audit each one:
`../gluon/inline-asm/field-guide.md ## Reverse index: I see X in an unknown kernel` reads a site from
its constraint string or its mnemonic, and `## Traps, ranked by how likely you are to hit one` names the
two that bite on a lift. **Neither a site that is one half of a split-phase pair (issue/wait,
save/restore) nor a site keyed on a tile constant analyses correctly alone** — the first is a
`PAIRED-WITH` obligation, the second is `## Shape-keying: inline asm is a per-M specialization`, and
re-deriving it is mandatory rather than optional when the tile changes. Match on `inline_asm`, not
`gl.inline_asm`: the helper is reachable through the Triton namespace too, and a callee-qualified
pattern drops those silently. This audit produces a ledger row, not a round: nothing here is a lever,
and deleting a site you have not classified is a correctness edit.

### The divergence ledger — and why a >100% anchor is allowed

A faithful anchor is the default because its difference set is empty, so nothing needs attributing.
It is not a requirement that the anchor be **slower**: `parity_gate.py` has always expected a ratio
above 1.0 and asked you to attribute it rather than pocket it. Where the champion's layout is what the
compiler could reach rather than what it would choose, transcribing it 1:1 copies plain's limitation
forward and then spends recovery rounds climbing back off it — which is a real cost, not a discipline.
So divergence is permitted in three named grades, and the grade travels with the site:

| grade | what it is | what it owes |
| --- | --- | --- |
| **faithful** | the recovered construct, same program point | nothing — the difference set is empty |
| **forced** | Gluon has no constructor for what the champion used (`amd_rotating_shared` on any arch); `ttgir_bridge` reports it as a named `UNRECOVERABLE` row rather than guessing. Only after the UNRECOVERABLE checks in §2 (probe your build, re-dump at ns=1) | the row, and which construct you wrote instead — a compiler-owned construct (`convert_layout`, let the backend choose the staging), never a hand-derived lookalike basis |
| **elective** | you judged the champion's choice sub-optimal *in Gluon* and wrote something else | the row, **plus the faithful variant compiled and measured** |

**An elective divergence owes two numbers, and that is the whole discipline.** Build the faithful
variant too — `$KT/probe.py measure --dir <ir-dir>` is compile-only and answers in seconds, so both
occupancy limiters for both variants cost no GPU time, and whether `shared` crossed a divisor is
usually the entire result. Without the pair you know neither what the divergence bought nor which
number every later round is measured against.

**Then bank it as a win, not as the anchor.** If an elective divergence is worth doing it is worth one
named round in Stage-Recover or Stage-Climb, attributed to the layer that owns it. Folding it into the
anchor gives away an attributable win *and* makes `vs_anchor` unreadable for the rest of the run. Same
rule the recovered pipeline scaffold follows (§3, `--emit-gluon pipeline`).

**Before electing anything, ask why the champion is weak** — the answer decides whether diverging is
even yours to do:

| the champion is weak because | do this |
| --- | --- |
| plain cannot express the better layout — the compiler chose under constraint | **elective divergence is exactly what this tier is for.** Record it and take it |
| its sweep did not cover the better config, or its tile was never re-swept | **return a `resweep_request` to the front end** (in the deep_engineer's result; tech_lead hands it to GEAK's plain rounds). Do not run the sweep in this role or claim its possible gain as a Gluon result. |

`champion_gate.py` cannot settle this for you: `[COMPARATOR]` only asserts the champion beats the
kernel's own default, and `[SAMPLING]` can only read the bundle's own claim about its sweep.

---

## 0. Preconditions — do not start without these

| condition | how | why |
| --- | --- | --- |
| the champion bundle is asserted | `python3 $SKILL/scripts/champion_gate.py --champion <work>/plain_champion.json` | a transcription measured against an unasserted champion is unfalsifiable, not merely inconvenient. The gate verifies the bundle's source still hashes to what was measured and that the `.ttgir` really was dumped at `config` (via the `<ttgir>.config.json` sidecar and the IR's own `ttg.num-warps`). Without it, "recovered from the tuned winner" is an assumption — one that has been wrong |
| the `.ttgir` is the ORIGINAL dump, from the champion at its pinned config | from `$KT/dump_ir.sh`, or the bundle's own copy | a hand-cleaned copy (`*.clean.ttgir`, `#loc` aliases stripped) will not parse, and the tool refuses rather than guessing |
| `import triton` works here | inside the container if the host has no torch | the whole design is to use the compiler's parser. No GPU, no launch, no compile — so a CPU-only container is fine. If it cannot work at all, `ttgir_to_gluon.py` is the fallback (output is a starting point, not a proof) |
| the Triton running the tool is **≥ 3.7** (3.8.0 is the reference), and at least as new as the compiler that wrote the dump | `python3 -c "import triton; print(triton.__version__)"` | see the matrix below |

**Read the champion bundle. Do not sweep, and do not dump your own baseline.** The layouts the compiler
infers depend on `num_warps` / block shape / `num_stages`, so recovering from a non-winning compile
yields an anchor that **cannot reach plain best** — and every later layer delta would then be measured
from the wrong floor. From the bundle you take: `config` (the pinned winner — `num_warps`, block sizes,
`num_stages`, any `matrix_instr_nonkdim`, `GROUP_SIZE_M`, split-K), `ttgir` (dumped from the champion
source **at** that config), `champion_ms` (the target line), and `served_range` (the shapes any win
must hold across). Schema: `entry.md`, "The champion bundle contract".

### Triton version support (measured, not assumed)

Established on clean upstream wheels (`pip install triton==<v>`, no ROCm, CPU-only container) and on a
vendor tree. Re-establish it on your own build with the recipe in `## Offline check` — every cell below
is one `--selftest` plus one `recover` of a dump you already have:

| build | recover | verify basis | `view` | parses a 3.7+ dump |
| --- | --- | --- | --- | --- |
| upstream **3.8.0** (reference) / **3.7.0 / 3.7.1** | works | both = LinearLayout normal form | works | yes |
| vendor tree (main-based) | works | both = LinearLayout normal form | works | yes |
| *downgrade:* upstream **3.6.0** | works | distributed = normal form, **shared = canonical text** | **absent** | **only if the dump has no `ttg.barrier`** |

**Use 3.7 or newer.** On 3.6 the tool still runs and says so honestly — every verdict prints its
`basis:` line, and a text basis is *sound but stricter* (two spellings of one layout read as different,
so a `FAIL` there may be a spelling difference rather than a layout difference). Since a kernel with
**no shared layouts** has nothing for that basis to weaken, `verify` prints `shared=n/a` there instead
of the caveat, and a 3.6 verdict on such a kernel is worth exactly as much as a 3.7 one.

The 3.6 parse limit is a property of the **DUMP, not of the version pair**: `ttg.barrier` does not exist
before 3.7, so a 3.6 build cannot parse a dump that *contains one*; a 3.7-produced dump without one
recovers on 3.6 byte-identically. Ask "does this dump contain `ttg.barrier`?", not "which versions am I
crossing?". Failure is loud, at parse time, and never a silent partial recovery.

**Recovery itself is version-invariant across 3.6.0–3.8.0, and you can check that on your own dump.**
Run `recover` on the same `.ttgir` under each minor and diff the emitted `layouts.json`: the recovered
site count and every layout constant should be byte-identical, because they are read out of the
attribute by upstream's own converter rather than derived per version. Measured: 16 kernels (8 aiter,
8 from a separate tuned-Triton set) × clean upstream 3.6.0 / 3.7.0 / 3.7.1 / 3.8.0 in per-version
containers — identical recovered counts and byte-identical layout constants, 32/32. A layout preamble
may be carried across versions.

**Do not extrapolate that to performance.** Time the plain kernel and its Gluon anchor under each minor
and read the two absolute numbers, not their ratio. The two arms move independently across a minor
bump and their deltas do not share a sign — the same 8 anchors on 3.8.0 moved in *both* directions
against 3.7.1: one attention forward's plain regressed 1.83× while its anchor lost only 11% (so the
anchor's ratio jumped from 1.005 to 1.655 without the anchor improving at all), while a GEMM's anchor
regressed 1.20× as its `plain@ns=1` improved 1.14×, collapsing a 1.36× win to parity. A ratio that
improves across versions is not evidence that the anchor improved (`benchmark-hygiene.md`, cross-version
ratios). **Layout constants are portable across versions; timings are not. Re-measure after a Triton
bump; never carry a ratio across one.**

The capability set is **probed, not version-gated**, in a child process — because the capability that
varies fails by *aborting*: on 3.6, `to_linear_layout` of a shared layout trips an MLIR assertion and
kills the interpreter. `TTGIR_BRIDGE_CAPS=...` skips the probe if its ~0.4 s matters in a loop.

Two facts worth knowing before you pick an environment:

- **upstream wheels ship the AMD backend.** `pip install triton==3.7.1` in a plain `python:3.10-slim`
  reports `backends installed: ['amd', 'nvidia']`, so `--arch gfx950` (or `gfx942`) works with no ROCm
  present. Nothing here launches a kernel.
- **the recovered constants are not a vendor-fork artefact.** The same dump recovered under clean
  3.6.0, clean 3.7.0 and the vendor tree yields byte-identical output — every site recovered, same
  layouts, same field values, all round-trips EXACT.

`--arch` must be the arch the dump was compiled for. Get it wrong and the AMD ops have no dialect,
which surfaces as ``Dialect `amdg' not found`` — a parse error that reads like a corrupt dump.

---

## 1. Dump — and pin the body

Use the bundle's `.ttgir` when it exists. If you re-dump yourself:

```bash
python3 "$SKILL/scripts/ttgir_bridge.py" --selftest       # ~10 s, offline: is the recovery itself sane?
bash "$KT/dump_ir.sh" <compile cmd> --variant plain --out ir/ --arch gfx950 \
     [--kernel-name <substring>]                          # PIN the body on a multi-kernel op
#   -> ir/plain/plain.{ttgir,llir,amdgcn,s} + meta_*.json (LDS bytes/WG)
```

Run the converter's `--selftest` first so that a later failure is attributable to your kernel rather
than to the tool. Confirm `dump_ir.sh`'s echoed config matches the bundle's `config` before trusting
the recovered layouts — that is the same check the gate already ran, and skipping it here re-opens the
hole the gate closed.

> **`dump_ir.sh` on a multi-kernel op.** It takes the freshest artifact in the cache, so an op that
> compiles two kernels (an attention body plus a split-K reduce, say) hands you whichever compiled
> *last* — and every layout recovered from it is confidently wrong for the body you meant, silently.
> It warns and lists the candidates when more than one exists; pass `--kernel-name <substring>` to pin
> it. `--kernel` is a different flag — `module.path:object` for `--emit-gluon anchor|pipeline` — and
> `dump_ir.sh` **rejects a bare name there** (exit 1) and tells you to use `--kernel-name`, because a
> bare name in `--kernel` would defeat the multi-kernel guard with no symptom.

`dump_ir.sh` exit codes: 0 dumped | 1 no compile command / bare `--kernel` | 2 unknown option before the
compile command | 3 `--emit-gluon` requested without `recover_gluon.py` present. **`--knobs
LLIR_SCHED|AMDGCN_AS|RA_HINTS` is fork-only**: `TRITON_ENABLE_LLIR_SCHED`, `TRITON_ENABLE_AMDGCN_AS`
and `TRITON_ENABLE_AMDGPU_RA_HINTS` appear in no upstream version (nor does `triton.tools.amdgcnas`);
on a stock build they export env vars nobody reads — a silent no-op. Confirm with `probe_levers.py
--all` before attributing a delta to them.

---

## 2. Recover

```bash
python3 "$SKILL/scripts/ttgir_bridge.py" recover \
    --ttgir <bundle>/ir/champion/champion.ttgir --arch gfx950 \
    --out anchor_layouts.py --json layouts.json      # [-v] every use site; [--force] overwrite an
                                                     # --out recovered from a different dump
```

Read four lines of the report, in this order. Each has one correct response.

**`num_warps cross-check`** — must be `PASS`. The recovered layouts hard-code `warps_per_cta`; the
module carries `ttg.num-warps`. A `FAIL` means the dump is internally inconsistent and nothing
downstream is trustworthy. **Exit code 4** — distinct from the exit 1 that an `UNRECOVERABLE` layout
gives, because the two call for opposite responses: exit 1 means "part of this kernel is not
expressible, the rest is sound", exit 4 means "throw this dump away".

**`UNRECOVERABLE: N`** — must be 0. Any other number means a layout in this champion has no
`gluon.language` constructor on this build, so **the kernel is not fully transcribable as written**.
The row names the TTGIR kind. The one seen in practice is `amd_rotating_shared` (the plain backend's
rotating operand staging), which appears in attention kernels and is a genuine language gap, not a
tool gap — upstream's own converter has no case for it. **Expect it on attention-shaped bodies, and
expect it on every minor** (3.6.0 through 3.8.0 alike): it is a missing constructor, not a version
gate. Handle it in this order, cheapest first:

1. **Probe your own build before concluding it is a language gap.** An `UNRECOVERABLE` row is a prompt
   to check, not proof — the Gluon surface moves, and `amd_wmma` sat behind identical wording while
   being constructible as `AMDWMMALayout` all along.
2. **Re-dump at `num_stages=1` and re-recover.** It is **not** necessarily a `num_stages` artefact — a
   body can still carry it at `ns=1` — but the ns=1 layout family is the one a Gluon anchor can express
   anyway, so this is worth doing before writing anything off. (On the ns=1 dump of the same body the
   blocked global-load layout reads as `FROM_SMEM` where the ns=2 dump calls it `A_LOAD`/`B_LOAD`,
   identical values — compare by `constants-digest`, below.)
3. **Still unrecoverable → either a forced divergence or `structure_suspect`.** A *forced* divergence
   is recorded in the ledger (above): write a compiler-owned construct in its place — typically express
   the round trip as `gl.convert_layout` and let the backend choose the staging layout, then read what
   it chose off the anchor's own TTGIR — and disclose it; `--allow-unrecoverable` lets `recover` exit 0
   so you can inspect the rest, it does **not** make the anchor faithful. `verify` then reports the
   anchor's replacement as `EXTRA at a shape where plain carried an UNRECOVERABLE layout` and grades
   the run `RECONCILED`, not `PASS` — it cannot prove the substitution is free, and only the ISA can
   tell you whether it cost anything. If no compiler-owned construct can stand in, or the cost cannot
   be priced, stop and record **`structure_suspect`** with the kind named. **Never hand-derive a
   similar-looking layout** (a hand-rolled basis for `amd_rotating_shared`).

Why a substitution cannot be *verified*, only disclosed — and why the fix is larger than one Python
binding (an earlier version of this page said otherwise and a transcription tried it).
`AMDRotatingSharedEncodingAttr` does carry the C++ traits for a `toLinearLayout`, and
`builder.to_linear_layout(attr, shape)` does exist (`verify` uses it). But it wants an `ir.attribute`,
and on 3.7.1 / 3.8.0 **no binding can get an encoding attribute out of a Value or a Type**: `ir.value`
exposes only `get_type/get_shape/get_loc`, `ir.type` only `is_fp16/is_integer`, and there is no
attribute parser (`ir.make_attr` takes `(Sequence[int], context)` and builds dense integer arrays). So
the LinearLayout normal form of `amd_rotating_shared` is not reachable from Python at all, and **a
substitution cannot be verified against the original even in principle.** Closing it needs a C++
binding, and the precise one is **`MemDescType::getEncoding()`** — the sink (`gl.SharedLinearLayout`)
and the converter (`builder.to_linear_layout(attr, shape)`) both already exist; what is missing is any
way to hand the memdesc's encoding to the converter. The earlier claim that one Python binding
(`to_linear_layout_from_memdesc`) would suffice was tried and is **retracted**.

**`round-trip: EXACT=N`** — every layout must be `EXACT`. This is the proof that the Python object
carries every field the attribute had: the object is re-printed as MLIR and compared against the
source text. A `DIFFERS` row means upstream's converter lost a field on this layout kind, which is an
upstream bug worth reporting and a blocker here.

**the role table** — `A_LOAD` / `B_LOAD` / `A_SMEM` / `B_SMEM` / `A_DOT_OPERAND` / `B_DOT_OPERAND` /
`MMA` / `INDEX`. A and B are resolved from the dot operand's own `opIdx` and then propagated backwards
through `ttg.local_load` → memdesc → `ttg.local_store` → the global load, so those labels are derived,
not guessed. If `A_LOAD`/`B_LOAD` are absent and you only see `GLOBAL_LOAD`, the kernel does not stage
through LDS (or stages through an op this pack does not know) — read the provenance lines before
assuming which is which.

**Read the source names, not the role names.** Roles rank by *op kind*, so on attention three global
loads that all feed a `local_alloc` are indistinguishable by rank and only one of them gets called
`GLOBAL_LOAD`; `FROM_SMEM` can be the layout going *into* shared; and `EPILOGUE_STORE` was, on one
kernel, the zero-write path rather than the real epilogue. Five separate transcriptions had to open the
raw TTGIR to resolve this. Each provenance line ends with the **source variable and line** taken from
the compiler's own location info, and the emitted file carries a **`# DOTS`** block listing every
`tt.dot` instance with its operands' source names — use it on a multi-dot body, where the MFMA-family
table collapses to one bucket whenever the compiler reuses a single `#mma` (on a 14-dot backward kernel
that made it useless):

```
ttg.local_load result[0]  shape=[64, 128] f16 (reg)  <- q @ fwd_decode.py:507
tt.dot operand[1]         shape=[128, 64] f16 (reg)  <- kT @ fwd_decode.py:652
#   dot #2: A=dv[128, 16]  B=do[16, 128]  -> dv[128, 128]
```

A name the kernel author wrote beats any taxonomy this tool could invent.

### Four more `recover` outputs worth acting on — layout equivalence is blind to all of them

- **`COMPILED FORM of the N buffer_load site(s)`** — transcribe the **dump**, not the source, and
  transcribe the *bucket*: each site is reported as **bare / mask-only / mask+`other`**, detected
  rather than inferred. Both directions of getting this wrong cost the same thing. `tl.load(...,
  other=0.0)` frequently compiles to a buffer_load carrying **neither** operand (buffer OOB returns zero
  on CDNA), so passing `other=` in Gluon emits it and costs a `v_cndmask` per register — measured at
  1.2–2% on two kernels; and adding a *mask* the compiled form never issued costs the same. Two operands
  are **not reachable from `gl.amd.cdna3.buffer_load` at all**: `contiguity`, and `stride` — the latter
  appears only on the pipeliner's peeled prologue loads, which a non-pipelined anchor does not have.
- **`LDS: N allocation(s), M element(s)`** — compare against the anchor's. One kernel's entire residual
  was its shared total crossing the LDS/CU divisor for that arch while every layout verified; `verify`
  cannot see allocation size by construction. The divisor is **arch-specific — 160 KiB/CU on gfx950
  (CDNA4); gfx942 downgrade: 64 KiB/CU** — so `recover` derives it from the `--arch` you passed and
  **declines to name one** for an arch this skill has no figure for, rather than applying another
  generation's number. It is an **upper bound**: it sums declared allocations without modelling liveness
  reuse, so once the anchor compiles quote `$KT/probe.py measure` off the artifact instead. (A user
  `allocate_shared_memory` is totalled **by scope** where plain's allocator peaks **by liveness**, so a
  faithful transcription can report a larger shared figure than the kernel it copies and land on the
  wrong side of the divisor while every layout still verifies.)
- **`op(s) ... have NO gluon.language equivalent`** — `recover` audits *layouts*, not *ops*, so 100%
  layout recovery does not mean transcribable. `amdg.in_thread_transpose` used to appear as a
  *successful* row and was only caught at authoring time. Whether it appears at all is
  **arch-dependent**: on gfx950, which has read-with-transpose LDS (`ds_read_tr`), the compiler emits
  that instead and a faithful anchor inherits the instruction for free (no Gluon source API either way);
  re-dump for the arch you target before treating it as your blocker.
- **`constants-digest`** (with `--out`) — a digest over the sorted constructor expressions with role
  names dropped. Use it, not a file hash, to compare two recoveries of the same body (two people, two
  versions): the emitted header carries the dump path *and* the recovering Triton's version, and role
  names legitimately drift with the dump (`A_LOAD`/`B_LOAD` at ns=2 vs `FROM_SMEM` at ns=1, identical
  values), so two correct recoveries do not compare byte-for-byte, and hand-rolled normalisations are
  not comparable to each other either. Assert digest consistency *within* one arch, never across gens —
  the MFMA layout family differs (`version=4` on gfx950, `version=3` on gfx942).

**Coverage.** The recovered constructors cover `#blocked`, `#amd_mfma`, `#swizzled_shared`,
`#padded_shared`, `#linear`, `ttg.dot_op` and `ttg.slice`. Two gaps are **not** tool bugs, so do not
spend verify cycles on them: `ttg.convert_layout` **placement is manual** — take it from the recovery
map in `../tile-programming/layout-recipes.md ## TTGIR -> Gluon recovery map (for transcription)`; and `amd_rotating_shared`
has no `gluon.language` constructor (above). Register spill / RA is NOT recovered — it is an LLVM stage
after `make_ttgir`, not in the TTGIR at all; it stays slicing + RA hints (`recover.md`, `lost_RA`).

---

## 3. Apply — the step that carries the port

Re-express the kernel as `@gluon.jit`, **preserving** logical tiles, masks, dtype, launch config and the
measured boundary, and wire in the recovered layouts.

**Declaring a layout is not applying it.** This is the single most expensive mistake in the phase: an
anchor that defines every constant and leaves the kernel body on `AutoLayout` compiles, is bit-exact,
passes a numeric oracle, and is several times slower than the champion, because the compiler resolves
`AutoLayout` on the index tensors to a scalar blocked layout and every load becomes uncoalesced. On one
measured kernel this single step was worth **more than everything else that run did in Gluon
combined**.

The recovered layouts arrive as a **preamble**; applying them is yours. `recover_gluon.py` (and
`ttgir_bridge.py recover --out`) emits a `gl.constexpr` block holding the layouts plain's compiler
inferred, and — under `recover_gluon.py --with-skeleton` — a kernel body beside it that still carries
the translator's own default MMA layout and `AutoLayout`. Nothing connects the two. A layout the body
never mentions never reaches the TTGIR, so `verify` will report exactly those layouts missing: read a
`missing` list that matches your preamble as "declared but not applied", not as "recovered wrong".

```bash
python3 "$SKILL/scripts/recover_gluon.py" --ttgir <bundle>/ir/champion/champion.ttgir \
    --arch gfx950 --record --out ir/plain/plain.gluon.py \
    [--with-skeleton --kernel module.path:object]          # algorithm skeleton (3.8+ translator)
# or, through the dump: bash $KT/dump_ir.sh <cmd> --variant plain --out ir/ --arch gfx950 \
#                         --emit-gluon layouts|anchor [--kernel module.path:object]
```

- **`--with-skeleton` (and `--emit-gluon anchor|pipeline`) needs the modern translator — 3.8+
  upstream.** It imports `translate_paths` and `TranslatorTarget` from
  `triton.tools.triton_to_gluon_translator`; at 3.6.0 / 3.7.0 / 3.7.1 the package is spelled
  `triton_to_gluon_translater` and exposes only `convert_triton_to_gluon(src)`. The import failure is
  caught and the run degrades to layouts-only with a note on stderr. **Decide this from the import,
  never from `triton.__version__`**: a main or vendor checkout can report 3.7.0 and carry the 3.8-era
  package. On an official pip build the translator has so far been found only in the gfx950 tutorial
  *fork*, so plan for **recover, then hand-author the anchor**. The translator drops `num_stages` and
  re-infers default layouts; replace those with the recovered ones.
- **A wrapped kernel translates to nothing.** Upstream resolves `module:object` with a bare `getattr`,
  so under `@triton.heuristics` / `@triton.autotune` it receives the wrapper and returns an *empty*
  translation without raising. `recover_gluon.py` peels to the `JITFunction` first (byte-identical to
  the stock path when there is nothing to peel) and names the wrapper on stderr; an empty result is
  reported as such, not as "translator unavailable" — point `--kernel` at the kernel.
- **`--emit-gluon pipeline` / `recover_gluon.py --with-pipeline`** may generate a recovered async
  double-buffer **candidate scaffold**, but it is never part of the anchor transaction: first persist
  and verify the faithful anchor, then evaluate the scaffold as a separate post-anchor direction (a
  hand-written-pipeline round in `recover.md` / `climb.md`). Mixing it into the anchor destroys
  attribution and does not count as pipeline evidence. **That scaffold rule is the general one, and it
  is what makes a divergence safe**: an elective deviation is handled exactly like the scaffold —
  persist and verify the faithful anchor FIRST, then carry the deviation as its own post-anchor
  direction with both numbers. What is never allowed is the deviation arriving *inside* the anchor
  with no faithful counterpart to measure it against.

Wire each constant onto the tensor that plays that role:

```python
offs_am = gl.arange(0, BM, layout=gl.SliceLayout(1, A_LOAD))     # index math, NOT AutoLayout
offs_k  = gl.arange(0, BK, layout=gl.SliceLayout(0, A_LOAD))
a_smem  = gl.allocate_shared_memory(dtype, [BM, BK], A_SMEM)     # staging
a_op    = a_smem.load(A_DOT_OPERAND)                             # dot operand
acc     = gl.full([BM, BN], 0.0, gl.float32, layout=MMA)         # accumulator
```

Checklist before moving on — every one of these is a site the report named:

- [ ] every `gl.arange` / `expand_dims` chain carries a `SliceLayout` of the layout its consumer wants,
      taken from the `INDEX` / `A_INDEX` / `B_INDEX` rows
- [ ] every `allocate_shared_memory` uses the recovered `*_SMEM` layout
- [ ] every `load`/`convert_layout` into a dot carries the recovered `*_DOT_OPERAND`
- [ ] the accumulator carries `MMA`
- [ ] the launch site uses `NUM_WARPS` from the emitted file, unchanged (never sweep `num_warps` on a
      transcribed kernel — any other warp count disagrees with the literal `warps_per_cta` in the
      layouts, a correctness bug, not a slow config)
- [ ] **every `ttg.local_alloc` became its own `allocate_shared_memory`** — that is the faithful form.
      Collapsing a pass-through one into `convert_layout` is a real lever and a real divergence, so it
      belongs to the recovery/climb round that measures it, not to this step (below)
- [ ] each buffer_load site uses the operand **bucket** the dump shows (bare / mask-only /
      mask+`other`), not the source's
- [ ] when rebuilding store/load index tensors in a recovered layout, **every offset term** is
      preserved (block-row base + intra-block arange) — dropping the block base makes all blocks alias
      the same rows (silent wrong output)

**Prove a relabel is free rather than assuming it:** `gl.convert_layout(x, L, assert_trivial=True)`
fails at compile time if the conversion is not a no-op. Use it wherever you believe a tensor already
has the target layout — most usefully after a `reshape`/`permute`/`join` chain, whose result can be
*equal to* a dot-operand layout without carrying that attribute, which the MFMA verifier then rejects.

**Barriers are inserted for you on the AMD path.** `gluon_to_ttgir` genuinely runs no membar pass,
which invites the conclusion that a hand-authored LDS loop needs explicit `gl.barrier()` — but membar
insertion happens lower, inside the shared `TritonGPUToLLVM` conversion, so it applies to Gluon too.
Probed: stripping all four `gl.barrier()` calls out of a working anchor left it numerically correct and
still emitted 6 `s_barrier` (vs 7 with them). Hand-written Gluon in the wild does
`shared.store()`/`shared.load()` in a loop with no `gl.barrier()` anywhere. Use `gl.barrier()` only to
**suppress or reposition**; adding one "to be safe" is a real instruction you pay for — one anchor paid
a redundant barrier for the opposite belief.

**Do not carry the champion's `waves_per_eu` across with the rest of its config.** It reaches LLVM as
`amdgpu-waves-per-eu` and **caps** occupancy outright. Measured: an anchor without it ran at 3
waves/SIMD and was the faster arm; adding the champion's `waves_per_eu=2` left VGPR count unchanged,
dropped it to 2 waves/SIMD and a slower clock. The throttle was tuned for plain's register-heavy
*pipelined* body, which is not what an un-pipelined anchor is. Carry the tile shape over; leave this one
off until it earns its way back in (`recover.md` / `climb.md`).

**Emit one anchor-state contract card per transcribed tile-op** (load A/B, the MFMA, LDS read/write,
epilogue store) using `../tile-programming/tile-op-contract.md`. Fill all four axes at faithful values:
the recovered **Layout** (the version-agnostic table below is the Layout-cell fill rule), the
**Dispatch** primitive, the issuing **Scope**, and a **Handoff** that is `none = compiler auto` at the
anchor — the anchor leaves the overlap unbuilt, and the pipeline layer is the step that fills that cell.
These cards are the unit the layer loop then mutates one cell at a time.

**Do not optimize while transcribing.** A mixed transcribe-and-improve step destroys the equivalence
anchor, and with it the ability to attribute the residual gap.

### Not every `ttg.local_alloc` should become an `allocate_shared_memory`

**This is an ELECTIVE DIVERGENCE, so it is not part of the faithful anchor.** The faithful transcription
gives every `ttg.local_alloc` its own `allocate_shared_memory`; choosing `convert_layout` for some of
them is a deliberate deviation from the champion's construct, so it is graded, recorded, and measured
against the faithful variant rather than folded in (the divergence ledger above). Its home is the layer
that owns it — the shared-layout half of `lost_layout`, which is what `parity_gate.py` names when the
gap shows up as grown `shared` bytes. Do the classification when you get there, or when the budget says
the faithful anchor is LDS-capped; not while transcribing.

The recovery report lists one `ttg.local_alloc` per staging site and the natural reading is one user
buffer each. That reading is right for a **staged** buffer and can be expensive for a **pass-through**
one, so classify each site:

| the site in the champion's TTGIR | transcribe it as |
| --- | --- |
| `local_alloc` whose `local_load`s are spread across the loop, or read on a later iteration than the write — a real staging buffer with reuse | `gl.allocate_shared_memory` with the recovered `*_SMEM` layout |
| `local_alloc` immediately followed by its only `local_load`, feeding one consumer — a pure layout round trip with no cross-iteration reuse | `gl.convert_layout` to the consumer's layout, and let the buffer stay **compiler-owned** |

Why the second row is not just a stylistic choice: a user `allocate_shared_memory` is live for the whole
function and the allocator charges every buffer separately, so the compiler's own conversion scratch is
allocated *on top of* it, while several `convert_layout`s can share one scratch. Two faithful
transcriptions of the same kernel can therefore differ substantially in `shared` bytes per workgroup and
in register count, for identical arithmetic — enough to move waves/SIMD, a step change rather than a
few percent. The `local_alloc` count and the layout diff are identical either way, so **neither the
numeric oracle nor (on a recoverable layout) anything but the compiled budget sees the size**. A
`local_alloc` carrying a layout with **no `gluon.language` constructor** is a common instance of the
second row: the missing constructor does not oblige you to hand-roll a buffer — express the round trip
as `convert_layout` and check what the backend chose in the anchor's own TTGIR (§4).

**Decide it by measurement, not by rule.** Neither form is always right. `$KT/probe.py` answers it
compile-only, in seconds, with no GPU time and no profiler:

```bash
# §4 already dumped the anchor's artifacts; point the probe at that directory
python3 "$KT/probe.py" measure --dir ir/anchor/
#   <kernel>.amdgcn   [gfx950] vgpr=NNN waves/SIMD=N [LLVM]  spill=0 B
#   <kernel>          lds/WG=NNNNN B   WGs/CU by LDS<=N
```

It reports **both** occupancy limiters, which is the point: a transcription can look safe on registers
and still be capped by LDS, and quoting the LDS side alone hands you generous headroom on a kernel that
is register-bound. Run it as soon as the anchor compiles — not after the first timing — and again after
any change to the staging shape. If `shared` crossed an LDS/CU divisor or the register count crossed a
wave threshold, that is the whole result, before any timing. (`probe.py` reads the per-arch LDS figures
through `_hwdata` from `perf_knowledge/hardware/data/hw_constants.json`; without it the LDS half reports
nothing — the half that catches an over-allocated staging buffer.)

**Three corrections to the classification above, all measured after it was written:**

1. **It is a property of the dump's `num_stages`, not of the kernel.** At the shipped depth the staging
   you are looking at is often the *pipeliner's*, which a faithful un-pipelined anchor has nothing to
   transcribe — so classify against a **`ns=1` dump**. Confirmed independently on two kernels, both of
   which read "staged" at the shipped depth and textbook "pass-through" at `ns=1`. That extra dump is a
   **diagnostic**, not a second anchor config: the run's pinned config does not move, and you already
   need `plain@ns=1` as the recovery-stage control (`recover.md`), so it costs nothing new.
2. **`verify` is blind to the choice only when the layout is `UNRECOVERABLE`.** That is the case the
   rule above was written for: a layout with no Gluon constructor is excluded from the comparison, so
   dropping its buffer is invisible. Where the shared layout *is* recoverable, `ttgir_bridge.py verify`
   sees it and returns a hard FAIL naming the missing `swizzled_shared`. What `verify` cannot see in
   either case is allocation **size**.
3. **The performance sign is kernel-dependent.** Three kernels authored both ways, all bit-exact:
   **two were slower as pass-through despite cutting `shared` by 3× and 8×, one was faster.** Cheaper
   LDS is not the same as faster. The mechanisms that decide it are all visible in the artifacts — read
   them rather than the rule:
   - **barriers.** Letting the backend reuse one small scratch several times per iteration buys a
     full-drain `s_barrier` on each reuse. On the two that lost, the loop's instruction *multiset* was
     identical and only `s_barrier` moved (**2 → 16** and **2 → 10**). Compare `s_barrier` and
     `lgkmcnt(0)`, not just `shared` bytes.
   - **whether the saving is on the binding limiter at all.** On one of them registers pinned occupancy
     at 1 WG/CU either way, so a **57 KiB** saving bought exactly nothing while the barriers cost real
     time.
   - **whether an arch instruction was lost** (gfx950): on one kernel the pass-through form lost the
     hardware transpose entirely (**`ds_read_b64_tr` 64 → 0**).

---

## 4. Compile the anchor and dump its TTGIR

```bash
bash "$KT/dump_ir.sh" <the command that compiles your anchor> --variant anchor --out ir/ --arch gfx950
#   -> ir/anchor/anchor.{ttgir,llir,amdgcn,s} + meta_*.json
python3 "$KT/probe.py" measure --dir ir/anchor/      # G3: both limiters, now
```

At the **same** config. The anchor must be launched with the emitted `NUM_WARPS`; a transcribed Gluon
kernel cannot follow plain to a different tile, because the recovered layouts pin the warp distribution.
Dump the champion under its own variant name too (`--variant champion`) — `parity_gate.py` reads the
literal `<out>/<variant>/<variant>.{ttgir,amdgcn}` pairs (`recover.md`).

---

## 5. Verify — the equivalence gate (the ONLY rejection in this stage)

```text
PASS  if layout-equivalence  (ttgir_bridge verify: PASS or RECONCILED vs plain)
      AND correctness == plain (same oracle, same tolerance)
      AND deterministic      (same-input maxdiff == 0 over N>=~40 launches -- race gate)
      AND asm parity         (unroll-normalized instruction MIX + class SEQUENCE vs plain)
record perf delta vs plain (may be neutral or slower)
```

**All four, not two.** An earlier revision listed only correctness and determinism, which reads as
permission to skip the two that catch a *silent* mis-recovery — layout-equivalence is the one that fires
when numerics pass anyway, and asm parity is what separates "the compiler's overlap is missing" from
"the transcription is wrong". `asm looks perfect` is not one of the four.

### 5.1 Layout equivalence — `ttgir_bridge.py verify`

```bash
python3 "$SKILL/scripts/ttgir_bridge.py" verify \
    --plain <bundle>/ir/champion/champion.ttgir \
    --anchor ir/anchor/anchor.ttgir --arch gfx950 --json verdict.json
```

Four states, three exit codes, four different next actions:

| exit | state | what it means | next |
| --- | --- | --- | --- |
| 0 | `PASS` | every layout in plain is reproduced in the anchor, and the anchor introduces none of its own, as LinearLayout normal forms | transcription landed. Run the other three checks |
| 0 | `RECONCILED` | there are differences, but **every one** has a named structural cause (below) | read the causes — each is a real fact about your anchor — then continue. Not a defect to fix |
| 1 | `FAIL` | at least one difference with **no** structural cause, listed as MISSING / EXTRA with the role, shape and constructor | fix the named layouts and re-verify. A `MISSING` blocked layout next to an `EXTRA` linear layout at the same shape is the signature of §3 not being done: the compiler chose a layout because you did not |
| 3 | `NOT_COMPARABLE` | the two dumps are not the same config (num_warps, threads_per_warp, or MMA family differ) | re-dump one of them at the other's config. **Do not read the diff** — every layout differs because the config is baked into all of them, and the report is then a pile of rows that look like a pile of bugs |

`RECONCILED` exists because a `FAIL` on a correct anchor sends you hunting for a mistake that is not
yours. Three causes qualify, and each stays visible and named — none is ever folded into a `PASS`:

1. **EXTRA at a shape where plain carried an UNRECOVERABLE layout** — your disclosed (forced)
   substitution. Matched on shape only, and labelled *probable*: the excluded side has no normal form
   to compare, so this is not a proof of equivalence.
2. **MISSING, but the anchor has the same constructor at another shape** — a faithful non-pipelined
   anchor against a pipelined plain, whose IR holds dot layouts at K-shapes your single decomposition
   never produces. `EXTRA` being empty is what separates this from "the compiler chose a layout because
   you did not".
3. **MISSING, produced by an op Gluon cannot express** — e.g. `amdg.in_thread_transpose`. No correct
   transcription of that body can ever produce the row. Two trial kernels were graded `FAIL` on anchors
   that were bit-exact *and faster than plain* on this row alone.

In all three cases `verify` cannot tell you the substitution was **free** — only the ISA can. And it is
blind to LDS allocation size by construction, which is the one gap that decided a whole kernel's
residual: use the `LDS:` line from `recover` and `probe.py measure` for that.

**A fourth structural cause exists and `verify` reports it as `FAIL`, correctly**, because it is a
genuine expressibility wall rather than a reconcilable difference: a `tt.dot` on fp8 can be lowered by
plain to an **unscaled `tt.dot_scaled`**, and Gluon has no spelling for that — the plain mfma builtin
rejects the element type, while the scaled one always materialises a default scale operand, so the
anchor is forced onto a *different instruction* than plain's. This is instruction-level, not
encoding-level: every layout multiplicity can match exactly and the port still cannot be made faithful.
If you see it, the finding is the wall itself (`structure_suspect`); report it upstream rather than
reconciling it away.

The `MULTIPLICITY` table is **informational and never gates**. A plain:anchor ratio above 1 is the
pipeliner's unroll factor, which is expected. It is still worth reading in one place: `CVT_DST plain x1
anchor x8` says the anchor pays eight relayouts where plain paid one, which is a performance finding
even when the layout gate passes.

**Budget several passes here, not one.** `verify` is a diff, and the expected shape of this step is
verify → read the missing/extra layout attributes it names → fix that one layout → recompile the anchor
→ verify again, until it reports PASS (or RECONCILED with every cause read). Do not proceed on a FAIL
and do not hand-wave a near-match: layout-diff is the only check that catches a transcription which
**passes the numeric oracle while having recovered the layout wrong** (wrong `order`, wrong `kWidth`),
and a numerically-correct wrong-layout anchor poisons every delta you measure afterwards. This
checkpoint is never the one allowed to slip.

### 5.2 Numeric equivalence, at tolerance

Same oracle, same tolerance as the champion; PASS = matches plain. Run it through GEAK's harness
(`e2e_workflow/scripts/harness_lib.py`) or pass it to `recover_gluon.py --verify --harness '<harness>
--correctness'` (it looks for `CORRECTNESS PASS` and folds the result into the transcribe record).
**`recover_gluon.py --verify` without `--harness` reports correctness as `not-run` and still exits 0**
— a green bare `--verify` is a (text-basis) layout check, not this gate. Transcription is layout-only,
so on an un-diverged anchor the expectation is bit-parity with plain; any numeric delta is a bug, not a
Gluon property. Check `isfinite` first — a tolerance comparison cannot fail on NaN.

**Trap — asm-match != correct.** A recovered layout can emit asm that looks *perfect* (e.g. an
in-register permlane epilogue vs an LDS round-trip) yet be **numerically wrong** if it does not match
this kernel's matrix-core element->thread mapping (warp arrangement / D-split). Never accept a layout on
asm shape alone; the numeric and determinism checks decide.

### 5.3 Determinism — the race gate

Same input, `N >= ~40` launches, `maxdiff == 0`. A numeric tolerance alone cannot catch a layout/async
mis-recovery that *races*; bit-identical proves no race **fired**. It does not prove the ordering is
sufficient: this check samples the interleavings that occurred, and an ordering constraint the recovery
dropped changes none of them — it only admits interleavings that did not happen. A *missing* barrier or
acquire is therefore the asm-parity check's job, where its absence is visible in the instruction stream.
Mandatory whenever the recovery touched async / shared ordering. No tool in this pack runs it: write the
launch loop and record the result.

### 5.4 Asm parity — instruction MIX and class SEQUENCE

Dump both sides (`dump_ir.sh --variant champion|anchor`), then compare the unroll-normalized
instruction **mix** (`$KT/asm_loop_audit.py --opcodes` histogram) **and** the op-class **sequence**.
Matched mix + clustered MFMA => the residual is the auto-pipeliner overlap (structural, `recover.md`),
not a recovery bug. No single tool in this pack runs both halves; record which you ran.

**The histogram is half of the asm-parity line, and it is the half that cannot fail a reordering.**
Counts are order-independent, so a program with the same instructions issued in a different order
passes it unchanged. The same holds for any other tally you might reach for — register counts, LDS
bytes, spill counts, code length. This generalises:

> **A check that a different program can pass is not an identity check, however many terms it has.**
> Adding terms to an order-independent check does not make it order-dependent. Identity needs a
> **normalised line-by-line diff** of the two dumps, which is usually about ten more lines of code
> than the tally you already wrote.

**The same question generalises past identity: ask what would make the check come out false.** The
wider form of that defect is a criterion that **cannot take the value false at all**:

> **A check's power is not in how strict it looks, it is in whether it can fail.** Having written a
> criterion down, ask: **on what input does this one report a problem?** If there is no answer, or the
> answer lies outside the quantity's own domain, it is not a check — it is a restatement, and it will
> be recorded as a gate that passed.

Three shapes it takes in this pack: the **order-independent tally** above; a **threshold set at the
architectural or type maximum** — a budget compared against the ceiling that already bounds it is
satisfied by construction, never reports "exceeded", and the pressure it was supposed to catch
re-appears as spill instead; and a **whitelist of known-good identifiers**, which passes everything it
was built from and is guaranteed to miss the next variant. Gate on the predicate, and report the
identity separately.

Practically: keep the histogram for the MIX half, and get the SEQUENCE half from the op-class **symbol
stream** the same tool emits (`../tile-programming/compiler-contract.md`, section *Auditing the hot-loop
schedule from the* `.s`), diffed side by side after the normalisation that page prescribes — strip the
`.loc` / `.Ltmp` noise and pin one build for both dumps, or the diff reports line-number churn instead
of program difference. A recorded "asm parity: PASS" that was a histogram alone is a mix check wearing
the gate's name.

---

## 6. Inspecting a layout before choosing one

```bash
python3 "$SKILL/scripts/ttgir_bridge.py" view --ttgir <f>.ttgir --role A_SMEM --arch gfx950 \
    [--hardware] [--max-rows 24] [--max-cols 200]
```

Prints the per-lane ASCII table (`get_layout_view`; `--hardware` for the hardware view instead of the
tensor view; 0 = no limit). Use it to choose between candidate shared layouts by their **access
pattern** rather than by the clock: several swizzles can be numerically identical and differ only in the
`ds_read`/`ds_write` mix they produce. Choosing on the view first and confirming on the clock second is
much cheaper than benchmarking six layouts, and it is how a run has taken an anchor from roughly half of
plain to parity in a single round (a `lost_layout` repayment in `recover.md`). Not available on 3.6.

---

## 7. Version-agnostic recovery: the rule, and how to check it when the tools drift

The procedure above and `../tile-programming/layout-recipes.md` are the **current-build instance**: the
scripts hardcode this Triton's IR-attribute spellings, Gluon class names, async API, and IR stage/pass
names — **all of which drift across Triton/ROCm versions**. When a script breaks (IR no longer parses, a
Gluon class/builtin renamed, the pipeliner emits different ops, the IR-dump env changed), do **not**
abandon the conversion or record a false ceiling — fall back to the version-stable **rule** and
**discover** this build's names.

**The rule (what the conversion IS, any version).** Plain lowers `tt.load`/`tt.dot`/softmax to a
layout-resolved IR in which the **compiler has already chosen every tile layout, shared layout, operand
layout, convert placement, and pipeline depth**. Gluon makes those *same* decisions explicit:

| compiler decision (read from the lowered IR) | explicit Gluon construct (this build's name) |
| --- | --- |
| register/thread tiling of a value tensor | the blocked / distributed tile-layout object |
| matrix-core output tiling | the MMA/MFMA layout object (version, instr shape, warp split) |
| shared staging layout (pad / swizzle params) | the padded / swizzled shared-layout object |
| dot-operand view of a tile | the dot-operand layout (operand idx, parent MMA, k-width) |
| each layout conversion + WHERE it sits | an explicit convert at the same program point |
| pipeline depth / async staging | the explicit multi-buffer + commit/wait structure (built in `recover.md` / `climb.md`, not in the anchor) |

Preserve logical tiles, masks, dtype, launch config, boundary; only the layout/memory/pipeline
**expression** becomes explicit. Spills/RA are NOT here. This table is the **Layout-cell fill rule**
for each tile-op's contract card.

**Discover this build's names (never assume a spelling).**
1. Dump the layout-resolved IR for the pinned best config (the dump env + IR-stage name are themselves
   version-specific — find them via the toolchain's dump/knobs doc or the `*_KERNEL_DUMP`-family +
   cache dir, not from memory).
2. Read the layout/encoding definitions at the top of that IR — those literal fields (tiling vectors,
   warp split, swizzle/pad params, k-width, basis vectors) are the **ground truth** to reproduce,
   whatever they are spelled.
3. Enumerate the Gluon layout + shared/async API actually present on the build (introspect the gluon
   language module; `triage.md`, missing-doc protocol) and match each IR field to a constructor arg
   **by meaning, not by remembered name**.
4. A renamed/removed construct is an **API blocker to record**, not "Gluon can't express it" —
   re-derive from the IR field's meaning.

**Rule — recover small/odd async layouts, never hand-derive.** When you need an async distributed/shared
layout for a tile the proven set does not cover (a sub-efficient-width head-dim half, an odd contraction
width), do **not** hand-build the `DistributedLinearLayout` / `PaddedSharedLayout` basis vectors — a
hand-intuited basis silently **fails LLVM translation** (the basis must match the compiler's exact
inferred mapping, not an intuited one). Instead **recover** it: compile a standalone plain kernel at that
exact sub-tile shape, dump its lowered IR, reuse the inferred layout verbatim. This is the general unlock
for small/odd head-dim async paths — and for the per-chunk layouts of a split dim
(`../tile-programming/layout-recipes.md`).

**How to check, names-independent:** re-lower the anchor to the *same* IR stage and run the same four
checks as §5 — layout attributes diffed field-by-field vs plain (compare the resolved tiling/basis, not
the string), the oracle, determinism, asm parity.

---

## 8. The anchor becomes the new baseline

- `gluon_anchor` = this transcription = the **new Gluon baseline** for the stages that follow. Its
  regression vs the champion is the starting gap, not a rejection.
- `champion_ms` (from the bundle) = the **target line** to reach and exceed. It is the tuned plain
  number, not the kernel's shipped default and not this anchor — beating the anchor only recovers what
  transcription gave away (`recover.md ## Two baselines`).
- **If the anchor carries an elective divergence, the FAITHFUL variant is the baseline** and the
  divergence is the first banked win. Quoting the divergent anchor as `gluon_anchor` folds a win into the
  denominator: every later `vs_anchor` then reads smaller than it is, and the divergence itself
  disappears from the record it should have been the first entry in.

**Performance regression vs plain is expected and is NOT a rejection here.** The faithful transcription
loses the compiler's automatic scheduling/pipelining: no upstream `gluon_to_ttgir` (3.6.0 / 3.7.0 /
3.7.1 / 3.8.0, all checked) runs plain's software pipeliner or automatic block ping-pong, so plain's
`num_stages` overlap does not survive even when every layout is recovered perfectly — and a naive anchor
can sit well below tuned plain, by a wide margin. That is the pipeline debt, not a verdict on the
transcription, and it is **a starting gap, not a ceiling**: it is repaid in `recover.md`, hand-written
first (register prefetch → authored LDS ring → `warp_pipeline_stage`), with re-injection of plain's
pipeliner only as a below-gate diagnostic or last resort. Do **not** "force" the pipeliner onto the
anchor by hand, and only declare a residual MFMA/VALU-overlap ceiling **after** an overlap mechanism has
actually been measured. The equivalence gate above remains the **only rejection** in this stage; the
claim-side gate (parity) belongs to `recover.md`.

(`recover_gluon.py --record` prints `perf_delta_vs_plain: <fill> # regression expected, NOT a reject`.
That line is upstream's transcribe-only expectation and is right *for this stage*; it does not mean the
run may stop below the parity criterion — `recover.md` holds the run to it.)

## Output

```text
anchor_layouts.py / layouts.json / verdict.json        (recover + verify)
ir/plain/  ir/anchor/  ir/champion/                    ({ttgir,llir,amdgcn,s} + meta_*.json)
gluon_anchor_metrics.json                              (incl. probe.py both limiters, the four checks)
divergence ledger rows                                  (faithful / forced / elective, per site)
tile-op contract cards                                  (anchor state, Handoff = none)
```

Then enter `recover.md`: the mandatory re-profile + recalibration, attribution, and the parity gate.

---

## Offline check

```bash
python3 "$SKILL/scripts/ttgir_bridge.py" --selftest    # no GPU; prints SELFTEST PASS
python3 "$SKILL/scripts/ttgir_to_gluon.py" --selftest  # parser / emitter
python3 "$SKILL/scripts/recover_gluon.py" --selftest   # text-basis equivalence checks
bash "$SKILL/scripts/smoke_test_recover.sh" [<gfx950-gluon-tutorials checkout>]  # all selftests
```

The pure layers (type splitting, role ranking, operand attribution, the rank guard, the config
precheck) run without Triton. The live layer parses a small synthetic TTGIR and asserts that every
layout round-trips EXACT, so a regression in upstream's converter is caught here rather than on a
kernel. It passes on upstream 3.6.0, 3.7.0, 3.7.1 and the vendor tree; on 3.6 it additionally proves the
capability probe survives the build whose capability probe would otherwise kill it.

To re-run the cross-version check after a change to the tool:

```bash
docker run -d --name triton-clean-test -v /apps:/apps:ro python:3.10-slim sleep infinity
docker exec triton-clean-test bash -lc '
  for v in 3.6.0 3.7.0 3.7.1 3.8.0; do python3 -m venv /v$v; /v$v/bin/pip -q install triton==$v; done'
for v in 3.6.0 3.7.0 3.7.1 3.8.0; do
  docker exec triton-clean-test /v$v/bin/python <this pack>/scripts/ttgir_bridge.py --selftest
done
```

Use a **clean** upstream wheel for this, never the vendor tree: that tree carries a locally-added
re-injection of plain's TTGIR pipeliner into `gluon_to_ttgir` (plus names of the form `TRITON_GLUON_*`),
none of which has a referent in any upstream tree — verified absent from `gluon_to_ttgir` on 3.6.0,
3.7.0, 3.7.1 and 3.8.0. A tool validated only there is validated against a fork.
