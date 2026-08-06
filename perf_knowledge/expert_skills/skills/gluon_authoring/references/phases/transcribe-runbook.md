# Transcribe runbook — champion TTGIR to a verified Gluon anchor

The executable form of `transcribe.md`, driven by `scripts/ttgir_bridge.py`. Follow it in
order; every step is one command and one decision.

**What this runbook produces:** a Gluon anchor whose layouts are *proven* to be the
champion's, and a recorded verdict on whether the transcription landed. **What it does
not produce:** a fast kernel. The anchor is the measurement baseline for the climb, not
the climb.

**Why the tool and not a text diff.** `ttgir_bridge.py` never parses a layout. It hands
the `.ttgir` to the compiler's own MLIR parser and then to `layoutToGluon()`, upstream's
own attribute→`gluon.language` converter, so there is no mapping table in this pack that
can fall behind Triton. Equivalence is decided on LinearLayout normal forms, not on
attribute text and never on attribute *counts* — plain's auto-pipeliner unrolls the loop,
so counting occurrences compares unroll factors and can never pass against a
pre-pipeline anchor.

`scripts/ttgir_to_gluon.py` still ships and still does a pure-text recovery with no
Triton import. Use it only when `import triton` is unavailable; its output is a starting
point, not a proof.

---

## 0. Preconditions — do not start without these

| condition | how | why |
| --- | --- | --- |
| the comparator is a **tuned** plain kernel, and its number reproduces on your harness | re-run the champion through the workflow's own bench and check it against the recorded figure | a transcription measured against an unasserted comparator is unfalsifiable, not merely inconvenient. Upstream asserts this with a `champion_gate.py` that is **not vendored here** — the workflow's own baseline capture and its independent re-benchmark serve the same purpose, so use those |
| the `.ttgir` is the ORIGINAL dump | from `scripts/dump_ir.sh`, at the champion's pinned config | a hand-cleaned copy (`*.clean.ttgir`, `#loc` aliases stripped) will not parse, and the tool refuses rather than guessing |
| `import triton` works here | inside the container if the host has no torch | the whole design is to use the compiler's parser. No GPU, no launch, no compile — so a CPU-only container is fine |
| the Triton running the tool is **≥ 3.7**, and at least as new as the compiler that wrote the dump | `python3 -c "import triton; print(triton.__version__)"` | see the matrix below |

### Triton version support (measured, not assumed)

Measured on clean upstream wheels (`pip install triton==<v>`, no ROCm, CPU-only
container) and on the vendor tree, over the same 44 real dumps:

| build | recover | verify basis | `view` | parses a 3.7+ dump |
| --- | --- | --- | --- | --- |
| upstream **3.6.0** | works | distributed = normal form, **shared = canonical text** | **absent** | **only if the dump has no `ttg.barrier`** |
| upstream **3.7.0 / 3.7.1 / 3.8.0** | works | both = LinearLayout normal form | works | yes |
| vendor tree (main-based) | works | both = LinearLayout normal form | works | yes |

**Use 3.7 or newer.** On 3.6 the tool still runs and says so honestly — every verdict
prints its `basis:` line, and a text basis is *sound but stricter* (two spellings of one
layout read as different, so a `FAIL` there may be a spelling difference rather than a
layout difference). Since a kernel with **no shared layouts** has nothing for that basis to
weaken, `verify` now prints `shared=n/a` there instead of the caveat, and a 3.6 verdict on
such a kernel is worth exactly as much as a 3.7 one.

The 3.6 parse limit is a property of the **DUMP, not of the version pair** — four separate
kernels established this. `ttg.barrier` does not exist before 3.7, so a 3.6 build cannot
parse a dump that *contains one*; a 3.7-produced dump without one recovers on 3.6
byte-identically. Ask "does this dump contain `ttg.barrier`?", not "which versions am I
crossing?". Failure is loud, at parse time, and never a silent partial recovery.

**Recovery itself is version-invariant.** Eight aiter kernels × {3.6.0, 3.7.0, 3.7.1,
3.8.0} produced identical recovered counts and byte-identical layout constants (32/32).
Do **not** extrapolate that to performance: the same eight anchors measured on 3.8.0 moved
in *both* directions versus 3.7.1, and by large factors — on one kernel plain regressed
sharply while its Gluon anchor held; on another the anchor regressed while plain improved.
These are not 2% effects and they do not share a sign. Layout
constants are portable across versions; timings are not.

The whole capability set is **probed, not version-gated**, in a child process — because
the capability that varies fails by *aborting*: on 3.6, `to_linear_layout` of a shared
layout trips an MLIR assertion and kills the interpreter. `TTGIR_BRIDGE_CAPS=...` skips
the probe if its ~0.4s matters in a loop.

Two facts worth knowing before you pick an environment:

- **upstream wheels ship the AMD backend.** `pip install triton==3.7.1` in a plain
  `python:3.10-slim` reports `backends installed: ['amd', 'nvidia']`, so `--arch gfx942`
  works with no ROCm present. Nothing here launches a kernel.
- **the recovered constants are not a vendor-fork artefact.** The same dump recovered
  under clean 3.6.0, clean 3.7.0 and the vendor tree yields byte-identical output:
  143/143 sites, 13 layouts, same field values, all round-trips EXACT. That is the
  cross-check that says a layout constant is the compiler's, not the fork's.

`--arch` must be the arch the dump was compiled for. Get it wrong and the AMD ops have
no dialect, which surfaces as ``Dialect `amdg' not found`` — a parse error that reads
like a corrupt dump.

---

## 1. Recover

```bash
python3 scripts/ttgir_bridge.py recover \
    --ttgir <bundle>/ir/champion/champion.ttgir --arch gfx942 \
    --out anchor_layouts.py --json layouts.json
```

Read four lines of the report, in this order. Each has one correct response.

**`num_warps cross-check`** — must be `PASS`. The recovered layouts hard-code
`warps_per_cta`; the module carries `ttg.num-warps`. A `FAIL` means the dump is
internally inconsistent and nothing downstream is trustworthy. **Exit code 4** — distinct
from the exit 1 that an `UNRECOVERABLE` layout gives, because the two call for opposite
responses: exit 1 means "part of this kernel is not expressible, the rest is sound", exit 4
means "throw this dump away".

**`UNRECOVERABLE: N`** — must be 0. Any other number means a layout in this champion has
no `gluon.language` constructor on this build, so **the kernel is not fully transcribable
as written**. The row names the TTGIR kind. The one seen in practice is
`amd_rotating_shared` (the plain backend's rotating operand staging), which appears in
attention kernels and is a genuine language gap, not a tool gap — upstream's own
converter has no case for it. Correct response: stop, and record `structure_suspect`
with the kind named. Do **not** substitute a similar layout. `--allow-unrecoverable`
exists only to let you inspect the rest; it does not make the anchor faithful.

It hit 5 of 8 aiter kernels on every version from 3.6.0 to 3.8.0, and it is **not** a
`num_stages` artefact — one kernel still shows it at `num_stages=1`, so re-dumping at ns=1
and re-recovering is worth doing before writing `structure_suspect` (the ns=1 layout family
is the one a Gluon anchor can express anyway). What blocks it is known, and **larger than one Python binding** — an earlier version of this
page said otherwise and a transcription tried it. `AMDRotatingSharedEncodingAttr` does carry
the C++ traits for a `toLinearLayout`, and `builder.to_linear_layout(attr, shape)` does exist
(`verify` uses it). But it wants an `ir.attribute`, and on 3.7.1 / 3.8.0 **no binding can get
an encoding attribute out of a Value or a Type**: `ir.value` exposes only
`get_type/get_shape/get_loc`, `ir.type` only `is_fp16/is_integer`, and there is no attribute
parser (`ir.make_attr` takes `(Sequence[int], context)` and builds dense integer arrays).
So the LinearLayout normal form of `amd_rotating_shared` is not reachable from Python at all,
and the consequence goes beyond the fallback: **a substitution cannot be verified against the
original even in principle.** This needs a C++ binding, and the precise one is **`MemDescType::getEncoding()`** — the sink
(`gl.SharedLinearLayout`) and the converter (`builder.to_linear_layout(attr, shape)`) both
already exist; what is missing is any way to hand the memdesc's encoding to the converter.

If you *do* substitute, disclose it: `verify` will report the anchor's replacement as
`EXTRA at a shape where plain carried an UNRECOVERABLE layout` and grade the run
`RECONCILED`, not `PASS` — it cannot prove the substitution is free, and only the ISA can
tell you whether it cost anything.

**`round-trip: EXACT=N`** — every layout must be `EXACT`. This is the proof that the
Python object carries every field the attribute had: the object is re-printed as MLIR
and compared against the source text. A `DIFFERS` row means upstream's converter lost a
field on this layout kind, which is an upstream bug worth reporting and a blocker here.

**the role table** — `A_LOAD` / `B_LOAD` / `A_SMEM` / `B_SMEM` / `A_DOT_OPERAND` /
`B_DOT_OPERAND` / `MMA` / `INDEX`. A and B are resolved from the dot operand's own
`opIdx` and then propagated backwards through `ttg.local_load` → memdesc →
`ttg.local_store` → the global load, so those labels are derived, not guessed. If
`A_LOAD`/`B_LOAD` are absent and you only see `GLOBAL_LOAD`, the kernel does not stage
through LDS (or stages through an op this pack does not know) — read the provenance
lines before assuming which is which.

**Read the source names, not the role names.** Roles rank by *op kind*, so on attention
three global loads that all feed a `local_alloc` are indistinguishable by rank and only one
of them gets called `GLOBAL_LOAD`; `FROM_SMEM` can be the layout going *into* shared; and
`EPILOGUE_STORE` was, on one kernel, the zero-write path rather than the real epilogue.
Five separate transcriptions had to open the raw TTGIR to resolve this. Each provenance line
now ends with the **source variable and line** taken from the compiler's own location info:

```
ttg.local_load result[0]  shape=[64, 128] f16 (reg)  <- q @ fwd_decode.py:507
tt.dot operand[1]         shape=[128, 64] f16 (reg)  <- kT @ fwd_decode.py:652
```

A name the kernel author wrote beats any taxonomy this tool could invent. The emitted file
also carries a **`# DOTS`** block listing every `tt.dot` instance with its operands' source
names, which is what to use on a multi-dot body — the MFMA-family table collapses to one
bucket whenever the compiler reuses a single `#mma`, and on a 14-dot backward kernel that
made it useless.

Three more `recover` outputs worth acting on:

- **`COMPILED FORM of the N buffer_load site(s)`** — transcribe the **dump**, not the
  source. `tl.load(..., other=0.0)` compiles to a buffer_load with a mask and *no* `other`
  operand (buffer OOB returns zero on CDNA), but passing `other=` in Gluon emits it and
  costs a `v_cndmask` per register. Measured at 1.2–2% on two kernels. `contiguity` is not
  reachable from `gl.amd.cdna3.buffer_load` at all.
- **`LDS: N allocation(s), M element(s)`** — compare against the anchor's. One kernel's
  entire residual was its shared total crossing the LDS/CU divisor for that arch while every
  layout verified; `verify` cannot see allocation size.
- **`op(s) ... have NO gluon.language equivalent`** — `recover` audits *layouts*, not *ops*,
  so 100% layout recovery does not mean transcribable. `amdg.in_thread_transpose` used to
  appear as a *successful* row and was only caught at authoring time.

> **`dump_ir.sh` on a multi-kernel op.** It takes the freshest artifact in the cache, so an
> op that compiles two kernels (an attention body plus a split-K reduce, say) hands you
> whichever compiled *last* — and every layout recovered from it is confidently wrong for
> the body you meant, silently. It now warns and lists the candidates when more than one
> exists; pass `--kernel <substring>` to pin it.

---

## 2. Apply — the step that carries the port

**Declaring a layout is not applying it.** This is the single most expensive mistake in
the phase: an anchor that defines every constant and leaves the kernel body on
`AutoLayout` compiles, is bit-exact, passes a numeric oracle, and is several times
slower than the champion, because the compiler resolves `AutoLayout` on the index
tensors to a scalar blocked layout and every load becomes uncoalesced. On one measured
kernel this single step was worth **more than everything else that run did in Gluon
combined**. It is the most expensive mistake available in this phase, not a tuning detail.

Wire each constant onto the tensor that plays that role:

```python
offs_am = gl.arange(0, BM, layout=gl.SliceLayout(1, A_LOAD))     # index math, NOT AutoLayout
offs_k  = gl.arange(0, BK, layout=gl.SliceLayout(0, A_LOAD))
a_smem  = gl.allocate_shared_memory(dtype, [BM, BK], A_SMEM)     # staging
a_op    = a_smem.load(A_DOT_OPERAND)                             # dot operand
acc     = gl.full([BM, BN], 0.0, gl.float32, layout=MMA)         # accumulator
```

Checklist before moving on — every one of these is a site the report named:

- [ ] every `gl.arange` / `expand_dims` chain carries a `SliceLayout` of the layout its
      consumer wants, taken from the `INDEX` / `A_INDEX` / `B_INDEX` rows
- [ ] every `allocate_shared_memory` uses the recovered `*_SMEM` layout
- [ ] every `load`/`convert_layout` into a dot carries the recovered `*_DOT_OPERAND`
- [ ] the accumulator carries `MMA`
- [ ] the launch site uses `NUM_WARPS` from the emitted file, unchanged
- [ ] **every `ttg.local_alloc` was classified before it was transcribed** — see below

### Not every `ttg.local_alloc` should become an `allocate_shared_memory`

The recovery report lists one `ttg.local_alloc` per staging site and the natural reading is
one user buffer each. That reading is right for a **staged** buffer and can be expensive for a
**pass-through** one, so classify each site before writing it:

| the site in the champion's TTGIR | transcribe it as |
| --- | --- |
| `local_alloc` whose `local_load`s are spread across the loop, or read on a later iteration than the write — a real staging buffer with reuse | `gl.allocate_shared_memory` with the recovered `*_SMEM` layout |
| `local_alloc` immediately followed by its only `local_load`, feeding one consumer — a pure layout round trip with no cross-iteration reuse | `gl.convert_layout` to the consumer's layout, and let the buffer stay **compiler-owned** |

Why the second row is not just a stylistic choice: a user `allocate_shared_memory` is live for
the whole function, so the compiler's own conversion scratch is allocated *on top of* it, while
several `convert_layout`s can share one scratch. Two faithful transcriptions of the same kernel
can therefore differ substantially in `shared` bytes per workgroup and in register count, for
identical arithmetic — enough to move waves/SIMD, which is a step change rather than a few
percent. The `local_alloc` count and the layout diff are identical either way, so **neither
`verify` nor the numeric oracle can see this**; only the compiled budget can.

A `local_alloc` carrying a layout with **no `gluon.language` constructor** is a common instance
of the second row: the constructor being missing does not oblige you to hand-roll a buffer.
Express the round trip as `convert_layout` and let the backend choose the staging layout — then
check what it chose in the anchor's own TTGIR (step 3).

**Decide it by measurement, not by rule.** Neither form is always right. `scripts/probe.py`
answers it compile-only, in seconds, with no GPU time and no profiler:

```bash
# step 3 already dumped the anchor's artifacts; point the probe at that directory
python3 scripts/probe.py measure --dir ir/anchor/
#   <kernel>.amdgcn   [gfx942] vgpr=NNN waves/SIMD=N [LLVM]  spill=0 B
#   <kernel>          lds/WG=NNNNN B   WGs/CU by LDS<=N
```

It reports **both** occupancy limiters, which is the point: a transcription can look safe on
registers and still be capped by LDS. Run it as soon as the anchor compiles, and again after any
change to the staging shape. If `shared` crossed an LDS/CU divisor or the register count crossed
a wave threshold, that is the whole result — before any timing.

Prove a relabel is free rather than assuming it: `gl.convert_layout(x, L,
assert_trivial=True)` fails at compile time if the conversion is not a no-op. Use it
wherever you believe a tensor already has the target layout — most usefully after a
`reshape`/`permute`/`join` chain, whose result can be *equal to* a dot-operand layout
without carrying that attribute, which the MFMA verifier then rejects.

**Do not optimize while transcribing.** A mixed transcribe-and-improve step destroys the
equivalence anchor, and with it the ability to attribute the residual gap.

---

## 3. Compile the anchor and dump its TTGIR

```bash
scripts/dump_ir.sh -- <the command that compiles your anchor>
```

At the **same** config. The anchor must be launched with the emitted `NUM_WARPS`; a
transcribed Gluon kernel cannot follow plain to a different tile, because the recovered
layouts pin the warp distribution.

---

## 4. Verify

```bash
python3 scripts/ttgir_bridge.py verify \
    --plain <bundle>/ir/champion/champion.ttgir \
    --anchor ir/anchor/anchor.ttgir --arch gfx942 --json verdict.json
```

Four states, three exit codes, four different next actions:

| exit | state | what it means | next |
| --- | --- | --- | --- |
| 0 | `PASS` | every layout in plain is reproduced in the anchor, and the anchor introduces none of its own, as LinearLayout normal forms | transcription landed. Go measure it (§5) |
| 0 | `RECONCILED` | there are differences, but **every one** has a named structural cause (below) | read the causes — each is a real fact about your anchor — then measure. Not a defect to fix |
| 1 | `FAIL` | at least one difference with **no** structural cause, listed as MISSING / EXTRA with the role, shape and constructor | fix the named layouts and re-verify. A `MISSING` blocked layout next to an `EXTRA` linear layout at the same shape is the signature of §2 not being done: the compiler chose a layout because you did not |
| 3 | `NOT_COMPARABLE` | the two dumps are not the same config (num_warps, threads_per_warp, or MMA family differ) | re-dump one of them at the other's config. **Do not read the diff** — every layout differs because the config is baked into all of them, and the report is then a pile of rows that look like a pile of bugs |

`RECONCILED` exists because a `FAIL` on a correct anchor sends you hunting for a mistake
that is not yours. Three causes qualify, and each stays visible and named — none is ever
folded into a `PASS`:

1. **EXTRA at a shape where plain carried an UNRECOVERABLE layout** — your disclosed
   substitution. Matched on shape only, and labelled *probable*: the excluded side has no
   normal form to compare, so this is not a proof of equivalence.
2. **MISSING, but the anchor has the same constructor at another shape** — a faithful
   non-pipelined anchor against a pipelined plain, whose IR holds dot layouts at K-shapes
   your single decomposition never produces. `EXTRA` being empty is what separates this
   from "the compiler chose a layout because you did not".
3. **MISSING, produced by an op Gluon cannot express** — e.g. `amdg.in_thread_transpose`.
   No correct transcription of that body can ever produce the row. Two trial kernels were
   graded `FAIL` on anchors that were bit-exact *and faster than plain* on this row alone.

In all three cases `verify` cannot tell you the substitution was **free** — only the ISA
can. And it is blind to LDS allocation size by construction, which is the one gap that
decided a whole kernel's residual: use the `LDS:` line from `recover` for that.

The `MULTIPLICITY` table is **informational and never gates**. A plain:anchor ratio above
1 is the pipeliner's unroll factor, which is expected. It is still worth reading in one
place: `CVT_DST plain x1 anchor x8` says the anchor pays eight relayouts where plain
paid one, which is a performance finding even when the layout gate passes.

---

## 5. Record, then attribute the residual

An anchor that passes §4 and is still slower than the champion is the **expected**
outcome, not a failure: no upstream `gluon_to_ttgir` calls the software pipeliner, so
plain's `num_stages` overlap does not survive a faithful transcription even when every
layout is recovered perfectly. That gap is a debt you knowingly took on — and, unlike the
other rows below, one that can be **paid off** rather than only attributed (see
`references/gluon/pipeline-reference.md`; measured to close fully on 3.6–3.8). Attribute it
first either way, because the recovery only helps if the pipeline is genuinely what is
missing:

| suspect | how it shows | how it is closed |
| --- | --- | --- |
| **lost pipeline** (usually dominant) | no overlap; MFMA stalls on a full `vmcnt`/`lgkmcnt` drain | **re-inject plain's pipeliner** — `scripts/patch_reinject.py`, measured to close the gap fully on all four upstream versions; or hand-built multi-buffering with `commit_group`/`wait_group` and **no `gl.barrier()` in the loop** |
| **lost vectorization** | anchor ≈ 0.5× or worse, and the asm load-width histogram shifted (`dwordx4` → `ushort`) | a `convert_layout` was folded backwards into the load; pin the staging with an explicit `allocate_shared_memory` |
| **lost layout** | §4 is `FAIL` | re-recover; never hand-derive a basis |
| **lost RA** | hot-loop `scratch_*`, or an AGPR/VGPR split the transcription did not preserve | register slicing — not in TTGIR, never visible to this tool |
| **lost schedule** | the instruction *multiset* matches plain exactly, but the waits do not (one kernel: 21 relaxed `lgkm` waits vs plain's 10) | reorder the body toward plain's program order; not a layout, pipeline, or selection problem |
| **LDS budget** | every layout verifies and the anchor is still slower, because `shared` crosses the LDS/CU divisor and costs a workgroup per CU. **That divisor is arch-specific** — 64 KiB on CDNA3, 160 KiB on CDNA4 — so pass `--arch` and never carry the verdict across generations | `recover` prints the LDS total and divides by the figure for your `--arch`; compare against plain's. `verify` is blind to allocation size by construction |

**Barriers are inserted for you on the AMD path.** `gluon_to_ttgir` genuinely runs no
membar pass, which invites the conclusion that a hand-authored LDS loop needs explicit
`gl.barrier()` — but membar insertion happens lower, inside the shared `TritonGPUToLLVM`
conversion, so it applies to Gluon too. Probed: stripping all four `gl.barrier()` calls out
of a working anchor left it numerically correct and still emitted 6 `s_barrier` (vs 7 with
them). Hand-written Gluon in the wild does `shared.store()`/`shared.load()` in a loop with
no `gl.barrier()` anywhere. Use `gl.barrier()` only to **suppress or reposition**; adding
one "to be safe" is a real instruction you pay for.

The expected anchor ratio depends on plain's `num_stages`. If the champion compiled at
`num_stages=1` there was no auto pipeline to lose, so a faithful anchor should land at
**≈1.00**, and anything much below that is a transcription defect, not a debt. Measure
plain at `num_stages=1` to settle it: if that number matches your anchor, the whole gap
is the pipeline and no amount of layout work will move it.

**Read plain's `num_stages` off the LOOP, not the launch.** aiter writes
`tl.range(..., num_stages=2)` on the row loop; a launch-level `num_stages=` is a different
knob and on a bare `range` loop it does nothing at all — measured, plain's TTGIR was
byte-identical at launch `num_stages` 1/2/3 while the loop-level annotation went 2 → 4 → 6
loads. So a champion whose launch passes nothing may still be fully pipelined, and the
`plain@ns=1` control has to flip the annotation the champion actually uses.

Once the gap is attributed to the pipeline, it is **recoverable, not a debt you have to
keep**: `scripts/patch_reinject.py` splices plain's `add_schedule_loops` + `add_pipeline`
into `gluon_to_ttgir` (Python-only, reversible, env-armed — verified byte-identical when
unarmed). It closed the whole gap on every upstream version, and the recipe has exactly two
requirements that the faithful-transcription rules happen to violate. Both, plus the dot
case where you have to hand the staging back to the pipeliner, are in
`references/gluon/pipeline-reference.md`.

Do not chase the pipeline by re-injecting plain's TTGIR pipeliner into Gluon. On a loop
that hand-authors shared memory it provably cannot fire: `isSafeToPipeline` bails on any
loop containing a `ttg.barrier`, and the pass finds pipelineable loads by walking SSA
operands backwards from the dot — a `local_store` is a side effect, not an SSA edge, so
`tt.load` is never reachable and the pass rewrites nothing. It runs and changes the IR
by zero bytes, which reads exactly like a tuning problem and is not one.

---

## 6. Inspecting a layout before choosing one

```bash
python3 scripts/ttgir_bridge.py view --ttgir <f>.ttgir --role A_SMEM --arch gfx942
```

Prints the per-lane ASCII table. Use it to choose between candidate shared layouts by
their **access pattern** rather than by the clock: several swizzles can be numerically
identical and differ only in the `ds_read`/`ds_write` mix they produce. Choosing on the
view first and confirming on the clock second is much cheaper than benchmarking six
layouts, and it is how a run has taken an anchor from roughly half of plain to parity in a
single round.

---

## Offline check

```bash
python3 scripts/ttgir_bridge.py --selftest    # no GPU; prints SELFTEST PASS
```

The pure layers (type splitting, role ranking, operand attribution, the rank guard, the
config precheck) run without Triton. The live layer parses a small synthetic TTGIR and
asserts that every layout round-trips EXACT, so a regression in upstream's converter is
caught here rather than on a kernel. It passes on upstream 3.6.0, 3.7.0, 3.7.1 and the
vendor tree; on 3.6 it additionally proves the capability probe survives the build whose
capability probe would otherwise kill it.

To re-run the cross-version check after a change to the tool:

```bash
docker run -d --name triton-clean-test -v /apps:/apps:ro python:3.10-slim sleep infinity
docker exec triton-clean-test bash -lc '
  for v in 3.6.0 3.7.0 3.7.1; do python3 -m venv /v$v; /v$v/bin/pip -q install triton==$v; done'
for v in 3.6.0 3.7.0 3.7.1; do
  docker exec triton-clean-test /v$v/bin/python <this pack>/scripts/ttgir_bridge.py --selftest
done
```

Use a **clean** upstream wheel for this, never the vendor tree: that tree carries a
locally-added re-injection of plain's TTGIR pipeliner into `gluon_to_ttgir` (plus
`TRITON_GLUON_SWP_PIPELINE` / `COOP_LDS` / `PINGPONG`), none of which exists upstream —
verified absent from `gluon_to_ttgir` on 3.6.0, 3.7.0 and 3.7.1. A tool validated only
there is validated against a fork.
