# AMD hardware primary sources — three machine-readable-first tiers

How to get an authoritative AMD hardware fact the DSL/compiler doesn't expose and that isn't
already distilled in `references/hardware/`. **Prefer machine-readable + generated over reading a
PDF.** Shared by every AMD DSL via overlay (no DSL fork). Nothing large is committed — sources are
fetched/generated on demand into a cache by `scripts/hw_sources.sh`; only the *distilled fact* (a
line/table with provenance) is written back into a hardware page.

**Where things live in GEAK.** The tools (`hw_sources.sh`, `gfx950_isa.py`, `layout_facts.py`) are
GEAK shared kernel tools in `kernel_workflow/scripts/kernel_tools/`; every `scripts/<name>` below is
this pack's shim to the same file and behaves identically. The committed ISA databases
(`gfx950-mfma-layout.json`, `gfx950-encoding.json.gz`, `gfx950-isa-facts.json.gz`,
`cdna3-/rdna3-/rdna4-encoding.json.gz`) live in `perf_knowledge/hardware/data/` and are located
through `kernel_tools/_hwdata.py` (override with `GEAK_HW_DATA_DIR`). gfx950 is the primary arch:
it is the one with all 13 ISA chapters offline; gfx942 (cdna3) answers encoding offline and layout
via the calculator. GEAK's human-readable ISA cards are
`perf_knowledge/hardware/cdna4_mi350/isa_notes.md` and `perf_knowledge/hardware/cdna3_mi300/isa_notes.md`
(distilled, no provenance-per-record — confirm against the databases before a fact gates a layout).

## Run the preflight first

```bash
scripts/hw_sources.sh preflight     # arch x question class -> who answers, here, right now
```

The tier that answers a question is **not the same for every architecture**, and two of the
sources can be missing from an environment without anything saying so until the moment you need
them. `preflight` prints the whole matrix and, for anything unavailable, the exact repair
command. It needs no cache, no network and no GPU, and the environment gate calls it — so a
launch that passed the gate has already told you what this box can answer. Reading a tool-gap
mid-climb and recording it as a hardware ceiling is the failure this exists to prevent.

## The three tiers (pick by what you need)

| need | tier | source (machine-readable?) | tool | `hw_sources.sh` |
| --- | --- | --- | --- | --- |
| operand **data layout** (which VGPR/lane/byte holds A[m][k], B, C/D; scale mapping) on **CDNA1-3 / RDNA3-4** | 2 | **AMD Matrix Instruction Calculator** — ✅ *generates* the exact table | the calculator (submodule / cache) | `layout <arch> <instr> [-A -B -C -D -k]` |
| operand **data layout** on **CDNA4 / gfx950** | 2 | **`gfx950-mfma-layout.json`** — ✅ committed, offline; CDNA4 ISA ch.7 extracted + verified | `scripts/gfx950_isa.py` | `layout gfx950 <instr>` |
| instruction **encoding**, opcode, operand fields, **pseudo-code** on **gfx950 / cdna3 / rdna3 / rdna4** | 2 | **`<arch>-encoding.json.gz`** — ✅ committed, offline; ISA ch.12-13, ingested from the XML + geometric pseudo-code | `gfx950_isa.py --arch <a>` (or `layout_facts.py <a> encoding …`) | — |
| instruction **encoding** on any **other** arch (cdna1, cdna2, rdna3.5) | 1 | GPUOpen ISA **XML** (`amdgpu_isa_<arch>.xml`) — ✅ machine-readable | `isa_spec_manager` (built on demand) | `xml <arch>` / `decode …` |
| registers, waitcnt, wait states, addressing, LDS banking — **gfx950 only** | 2 | **`gfx950-isa-facts.json.gz`** — ✅ committed, offline; ISA ch.1-6, 8-11 | `scripts/gfx950_isa.py` | — |
| the same, on **cdna3 / rdna3 / rdna4** | 3 | ISA **PDF** — not extracted for these arches yet | `pdftoppm` render | `pdf <arch>-isa <p0> <p1>` |
| microarch prose **not in any database**; re-verifying a record against the source | 3 | ISA / whitepaper **PDF** — ❌ visual only | `pdftoppm` render (pdftotext MANGLES tables) | `pdf <doc> <p0> <p1>` |

**The asymmetry is deliberate, and it is not the same gap in both directions.** gfx950 ships a
layout DB because the calculator has *no* CDNA4 support, so without one the only path to a CDNA4
layout was rendering a PDF. cdna3/rdna3/rdna4 ship *no* layout DB because the calculator already
generates theirs exactly, on demand — transcribing it would add a weaker copy of a derived
answer. What those arches were missing is the other half: an opcode, an operand field, a
pseudo-code body. That half is now committed for them too.

**A "not extracted" cell is missing KNOWLEDGE, not an absent hardware feature.** The tools say so
in those words when they cannot answer, because the two get recorded identically otherwise —
and one of them closes a direction that was still open.

**Tier order = preference.** Tier 2 is the machine-readable replacement for the PDF layout tables.
For CDNA1-3/RDNA the calculator *generates* the p52/p55-style table from the hardware rules. For
**CDNA4 the calculator still has no support** (this build lists only CDNA1/2/3 + RDNA3/4) — that
gap is now closed from the other side, and for the **whole document**, not just chapter 7: all 13
chapters of the CDNA4 ISA are committed as three offline databases and `scripts/gfx950_isa.py`
serves them with no PDF render and no network. **Tier 3 is no longer the default gfx950 path at
all**; reach for the PDF to re-verify a specific record, or when `search` comes back empty — which
is itself a coverage answer worth recording, not an error.

### cdna3 / rdna3 / rdna4: the instruction chapters, offline

Same tool, `--arch`. These three carry the ISA's two largest chapters (12-13 — instructions and
microcode formats, which are ~84% of the document's pages) as a committed database:

```bash
gfx950_isa.py --arch cdna3 encoding v_mfma_f32_16x16x16_f16   # opcode + operand fields + ch.12 code
gfx950_isa.py --arch rdna4 opcodes wmma                       # search every instruction
gfx950_isa.py --arch cdna3 errata                             # what this document gets wrong
layout_facts.py cdna3 encoding <instr>                        # same, via the routing entry point
```

Aliases: `cdna4`→gfx950, `gfx942`/`mi300`/`mi325`→cdna3, `gfx1100`→rdna3, `gfx1200`→rdna4.

**Both halves are machine-produced, so the trust question is different from gfx950's.** The
opcodes, encodings and operand slots are a lossless transform of the ISA XML — nobody read a
page. The pseudo-code is extracted geometrically: a line is code iff its vertical centre falls
inside one of the document's grey code boxes, read from the content stream, which is ground truth
the text layer does not carry. Each DB's `verification` block records what was checked, and each
arch's citation check confirms every record against the page it names.

**What licensed running that extractor on documents with no ground truth**: on CDNA4, where a
hand-built database already existed, it reproduces that database's attributed instruction set
exactly (1232/1232, with page, printed page and quote all exact) and independently re-derives its
hand-authored VOP3 opcode errata to the same 85-instruction set.

**A found defect, worth knowing before you hand-assemble.** The CDNA3 document has the same VOP3
opcode fault as CDNA4 — the printed opcode of 81 VOP1-promoted instructions is 64 too high in
section 12.11 — and this is the first place it is recorded. `--arch cdna3 errata` has it. The two
RDNA documents do **not** have the fault; `--arch rdna4 errata` says so as a checked result
rather than as an empty answer, and names the test behind it.

**Not covered for these three:** chapter 7 (matrix layout — ask the calculator, it is exact) and
chapters 1-6 / 8-11 (registers, waits, LDS, memory rules — not extracted; the tool routes you to
the PDF page and says it is missing knowledge, not missing hardware).

### gfx950: `scripts/gfx950_isa.py` — all 13 ISA chapters, offline

Three committed databases, split by what each source can faithfully express:

| database | ISA chapters | how it was built |
| --- | --- | --- |
| `gfx950-mfma-layout.json` | 7 (matrix layout) | extracted from the PDF, cross-verified against the Matrix Instruction Calculator |
| `gfx950-encoding.json.gz` | 12–13 (1240 instructions, 33 encodings) | opcodes/encodings/operands **mechanically ingested from the ISA XML** — lossless, nothing eye-read; per-instruction **pseudo-code, Notes and worked examples** added from ch.12 by geometric extraction (a line is code iff it falls inside one of the PDF's grey boxes) |
| `gfx950-isa-facts.json.gz` | 1–6, 8–11 (state, flow, SALU/VALU, memory, LDS) | extracted from the PDF; every record carries a page + verbatim quote + `_conf` |

```bash
scripts/gfx950_isa.py search "bank conflict"             # all three DBs — start here
scripts/gfx950_isa.py facts V_MFMA_F32_16X16X128_F8F6F4  # dims/dtypes/rules (+VGPR counts from the XML)
scripts/gfx950_isa.py layout <instr> -A --as fp8         # full lane x VGPR x bit table
scripts/gfx950_isa.py locate <instr> A 0 37 --as fp6     # one element -> lane, item, bits
scripts/gfx950_isa.py opcodes ds_read                    # ch.12: search all 1240 instructions
scripts/gfx950_isa.py encoding <instr>                   # opcode + operand fields + ch.12 pseudo-code
scripts/gfx950_isa.py format ENC_DS                      # ch.13: bit fields + VOP3 promotion / DPP limits
scripts/gfx950_isa.py limits vgpr                        # ch.3: capacities + allocation granularity
scripts/gfx950_isa.py waitcnt lgkm                       # ch.4: what LGKM_CNT counts, return ordering
scripts/gfx950_isa.py waits                              # ch.4 Table 11: required wait states
scripts/gfx950_isa.py lds transpose                      # ch.11: banking, conflicts, ds_read_*_tr_*
scripts/gfx950_isa.py mem buffer                         # ch.9: the buffer address formula, verbatim
scripts/gfx950_isa.py regs MODE                          # ch.3: bit fields of the state registers
scripts/gfx950_isa.py modifiers|sparse|hazard|errata     # CBSZ/BLGP, 4:2 tables, NOPs, source errata
scripts/gfx950_isa.py --selftest                         # re-derives the grids from the rules
```

**Do not assert a gfx950 hardware number from memory.** One `search` is cheaper than a wrong
kernel, and the answer arrives with the page number you are required to cite.

**How far to trust it.** The DB carries its own verification record and the tool prints the
applicable caveat — keep it when you quote an answer. The rules reproduce the Matrix Instruction
Calculator on 11776/11776 elements across 10 CDNA3 instructions (the one check with no PDF in its
chain) and 11264/11264 cells of the document's own grids. But that cross-check only covers what
CDNA4 **inherits** from CDNA3: every CDNA4-new surface (F8F6F4, E8M0 scaling, 6-bit packing, the
K=128 shapes, the new sparse shapes) rests on a single reading of one document. `sparse` and
`hazards` are marked *needs review*; two FP8 figures are flagged unusable (drawn in error); 19
`errata` records the places the source is wrong. The tool refuses rather than guesses — sparse
A/B operands and runtime-selected F8F6F4 widths exit 3 with the reason. **Validate a low-precision
layout numerically before shipping a kernel on it.**

The three databases do **not** have the same standing, and the tool prints which one answered:

- **`gfx950-encoding.json.gz` (ch.12–13) is the strongest.** Its opcodes, encodings and operand
  lists are a mechanical, lossless transform of the machine-readable ISA XML — no one read a page
  to produce them, so there is no transcription error to find. Where the XML and the PDF disagree,
  the XML is a superset (e.g. it carries a `GDS` field that Table 95 omits).
  The **`operation` pseudo-code is a second, weaker channel**: it comes from the PDF, but
  mechanically rather than by eye — a line is code iff its vertical centre falls inside one of the
  document's grey code boxes, read from the content stream, which is ground truth the text layer
  does not carry. `operation: null` is a real answer, not a failure: the ISA prints no code box for
  48 control/cache/LDS-DMA instructions, and `gfx950_isa.py encoding` says so explicitly rather
  than leaving a blank. Where the PDF and the XML disagree on an opcode, the XML wins and the
  disagreement is recorded — see `gfx950_isa.py errata vop3`, which settles the ch.12 §12.11
  VOP3 opcode column (off by exactly 64 for 85 instructions) three independent ways.
- **`gfx950-isa-facts.json.gz` (ch.1–6, 8–11) is single-channel.** It was eye-extracted once, by
  chapter, then merged and spot-checked; every record's quote is machine-verified to appear on the
  page it cites, which catches wrong pages and paraphrase-as-quote but *not* a misread value. The
  source document also genuinely contradicts itself in several places (`MODE.EXCP_EN` has three
  incompatible widths; LDS is "64 banks" and "32 concurrent accesses" on one page). Those are
  recorded as found, never silently resolved — a record marked `_conf: "uncertain"` says so on
  every print, and means *re-read the page before you act on it*.
- **`gfx950-mfma-layout.json` (ch.7)** carries the verification record described above.

## When to reach for a source

Only when the fact is NOT already in `references/hardware/` (probe-then-trust the distilled value
first) and the profiler/IR can't answer it. The #1 real trigger is **scaled-MFMA / low-precision
correctness**: a kernel that compiles but returns wrong numbers is almost always a data/scale
**layout** mismatch → Tier 2 (`layout …`) — the calculator for CDNA≤3/RDNA, `gfx950_isa.py` for
CDNA4. Others: a new
atom's operand/accumulator layout or cycle cost, exact `ds_read`/`waitcnt` encoding, an off-table
SKU peak (whitepaper).

## Which DSL / vendor

All **AMD** DSLs (gluon, flydsl, tilelang, HIP) — these are vendor-level facts, hence `vendor/amd`.
**NVIDIA (cutedsl) does NOT use any of this**; it needs PTX/SASS ISA + CUTLASS docs + Nsight
(a `vendor/nvidia` gap — see B4).

## How — `scripts/hw_sources.sh` (fetch/build/generate into a cache, never committed)

```bash
scripts/hw_sources.sh layout cdna3 v_mfma_f32_16x16x32_fp8_fp8 -A -B   # Tier 2: generate the table
scripts/hw_sources.sh layout gfx950 V_MFMA_F32_16X16X128_F8F6F4        # Tier 2: -> gfx950_isa.py (no fetch)
scripts/hw_sources.sh xml cdna4                                        # Tier 1: fetch+extract the ISA XML
scripts/hw_sources.sh decode cdna4 <encoded-bytes>                     # Tier 1: build isa_spec_manager, decode
scripts/hw_sources.sh pdf cdna4-isa 52 55                              # Tier 3: fetch PDF, pdftoppm-render p52-55
```

- `layout gfx950|cdna4` needs **no network and no cache** — it reads the committed layout DB. Only
  `gfx950_isa.py encoding` wants the ISA XML (seeded in the image, or `xml cdna4` to fetch it).

- Cache dir: `${TILE_HW_CACHE:-~/.cache/tile-hw}` (XML, cloned+built tools, PDFs). Tools
  (`isa_spec_manager`, the calculator) are cloned + built **on demand**, not committed and not
  pinned into a container.
- **A PDF fetch that "fails like a firewall" is usually the User-Agent.** amd.com's CDN drops
  curl's default UA *after* the TLS handshake, so it does not surface as a 403: over HTTP/2 you
  get `stream 0 was not closed cleanly: INTERNAL_ERROR` (exit 92), over HTTP/1.1 it hangs to
  timeout. `hw_sources.sh` sends a browser UA and that alone fixes it. Before recording a
  network-blocked tool-gap, check a second host on the same CDN — DNS, TCP and TLS all
  succeeding while HTTP returns nothing is the signature of a bot filter, not of egress.
- **Page numbers are offset from the PDF index.** The printed page number trails the PDF page by
  the front matter (PDF page 68 prints as "60 of 600"). Locate with `pdftotext` (prose greps
  fine), then pass the **PDF index** to the render — not the number printed on the page.
- **Pins** (reproducibility): the ISA XML zip is pinned to a dated release
  (`AMD_GPU_MR_ISA_XML_2026_03_05`); the PDFs to their AMD/GPUOpen URLs (below). `hw_sources.sh`
  records the resolved version in the cache.
- **Offline / fetch fails** → a scoped tool-gap: record it, fall back to the already-distilled
  `references/hardware/` values + probe-then-trust; never block.

## What to do with what you get — provenance rule

A verified fact goes into the matching hardware page **with a citation**, so the next agent trusts
it and never re-runs the tool:

- a generated layout table (Tier 2) → `isa-mechanisms.md` (or the DSL page that owns that op),
  cited `MI-calc <arch> <instr> [-A]`.
- an encoding (Tier 1) → cite `<arch> ISA XML <opcode>`.
- a gfx950 layout from `gfx950_isa.py` → cite the `# provenance:` line the tool prints; it names
  the governing rule and the printed page, so the citation stays `CDNA4 ISA p<NN>`. Carry any
  `!!` caveat with it.
- a CDNA4 fact read from the PDF by eye (Tier 3) → cite `CDNA4 ISA p<NN>`.

**Never** copy a source-derived fact into a per-DSL `lang/` file — vendor-level facts stay in
`vendor/amd`. This is the point of the single-sourced hardware layer.

## Source index (fetched by `hw_sources.sh`, not committed)

| what | where |
| --- | --- |
| ISA XML (all arches) | GPUOpen `https://gpuopen.com/download/machine-readable-isa/latest/` → `AMD_GPU_MR_ISA_XML_2026_03_05.zip` (`amdgpu_isa_cdna3.xml`, `amdgpu_isa_cdna4.xml`, `amdgpu_isa_rdna3.xml`, `amdgpu_isa_rdna4.xml`, …) |
| isa_spec_manager (XML decoder) | `https://github.com/GPUOpen-Tools/isa_spec_manager` (`build/prebuild_linux.sh` → `cd linux && make`) |
| Matrix Instruction Calculator | `https://github.com/ROCm/amd_matrix_instruction_calculator` (`pip install -r requirements.txt`; `matrix_calculator.py -a <arch> -i <instr> -A -B -C -D --markdown`) |
| CDNA4 ISA PDF | `https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf` |
| CDNA3/MI300 ISA PDF | `…/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf` |
| CDNA3 whitepaper | `…/white-papers/amd-cdna-3-white-paper.pdf` |
| CDNA4 whitepaper | `…/white-papers/amd-cdna-4-architecture-whitepaper.pdf` |
| RDNA3 / RDNA4 ISA PDFs | AMD radeon-tech-docs `…/rdna3-shader-instruction-set-architecture-feb-2023.pdf`, `…/rdna4-instruction-set-architecture.pdf` |

> There is **no public typeset/build source** for the whitepaper or ISA PDFs — the XML (Tier 1) is
> the closest thing to a machine-readable "source", and the calculator (Tier 2) *generates* the
> layout tables the PDF renders. Adjacent cross-checks (not primary): LLVM AMDGPU TableGen +
> ROCm `docs/conceptual/gpu-arch` encode some of the same layouts and can corroborate a fact.
