#!/usr/bin/env python3
"""gfx950_isa.py — the gfx950/CDNA4 ISA reference, offline. All 13 ISA chapters, three databases.

WHY THIS EXISTS. The AMD Matrix Instruction Calculator (Tier 2 of
perf_knowledge/expert_skills/skills/gluon_authoring/references/hardware/primary-sources.md) *generates* the operand layout table for cdna1/2/3 and
rdna3/4 — but it has NO cdna4/gfx950 support, so for the one chip this pack targets by default
the Tier-2 answer was a tool-gap and the only path was "render the PDF and read it by eye".
This script closes that gap for the whole document, not just for matrix layout.

Three committed databases, split by what each source can faithfully express:

  perf_knowledge/hardware/data/gfx950-mfma-layout.json     ISA ch.7      matrix operand layout
  perf_knowledge/hardware/data/gfx950-encoding.json.gz     ISA ch.12-13  1240 instructions, 33 encodings,
                                                                ingested from the ISA XML
  perf_knowledge/hardware/data/gfx950-isa-facts.json.gz    ISA ch.1-6,   registers, limits, waitcnt,
                                                  8-11          wait states, addressing, LDS

It is a LOOKUP, not a probe: deterministic, no GPU, no compile round, works offline.

  search         all three DBs at once -- start here when you do not know where the answer lives
  layout/locate  which lane + VGPR + bits hold A[m][k], B[k][n], C/D[m][n]   (ch.7)
  facts          dims, blocks, cycles, dtypes, governing rules + their trust verdict
  opcodes        search all 1240 instructions by name / functional group / encoding   (ch.12)
  encoding       opcode, encoding name, operand fields, for ANY instruction
  format         the bit-field layout of one encoding                        (ch.13)
  regs           MODE / STATUS / TRAPSTS / HW_ID / M0 bit fields              (ch.3)
  limits         VGPR / SGPR / LDS capacities, allocation granularity, alignment
  waitcnt        what VM_CNT / LGKM_CNT / EXP_CNT count, and their return ordering
  waits          required software-inserted wait states / NOPs                (ch.4)
  mem            addressing formulas, alignment, bounds checking              (ch.8-10)
  lds            LDS banks, conflict rules, MFMA transpose loads              (ch.11)
  tables         verbatim transcriptions of source tables
  concepts       actionable prose facts, each with its verbatim quote
  modifiers      CBSZ / ABID / BLGP / scale semantics and their value tables
  hazard         required independent instructions / NOPs around MFMA
  errata         the 19 places the source document is wrong

HOW MUCH TO TRUST IT. The DB carries its own verification record and this tool surfaces it
rather than hiding it. The layout rules matched the Matrix Instruction Calculator on
11776/11776 elements across 10 CDNA3 instructions — the one check with no PDF in its chain —
and 91/91 of the document's own worked examples. But that cross-check only covers what CDNA4
INHERITS from CDNA3. Every genuinely new surface (the F8F6F4 opcodes, E8M0 scaling, 6-bit
packing, the K=128 shapes, the new sparse shapes) rests on a single reading of one document,
and `sparse`/`hazards` are marked "needs review". Commands print the applicable caveat; do not
strip it when you quote the answer. Validate a low-precision layout numerically before you
ship a kernel on it.

Usage:
  gfx950_isa.py list [substr]                          # matrix instructions, optionally filtered
  gfx950_isa.py facts <instr>                          # dims/cycles/dtypes/rules + trust verdict
  gfx950_isa.py layout <instr> [-A|-B|-C|-D] [--as fp8|fp6|fp4|bf16|...]
                               [--lanes lo-hi] [--block N] [--markdown]
  gfx950_isa.py locate <instr> <A|B|C|D> <i> <k|j> [--block N] [--as FMT]
  gfx950_isa.py encoding <instr>                       # Tier 1: join against amdgpu_isa_cdna4.xml
  gfx950_isa.py search <substr>                        # all three DBs; e.g. `search bank conflict`
  gfx950_isa.py opcodes|format [substr]
  gfx950_isa.py regs|limits|waitcnt|waits|mem|lds|tables|concepts [substr]
  gfx950_isa.py modifiers [substr]
  gfx950_isa.py hazard [substr]
  gfx950_isa.py errata [substr]
  gfx950_isa.py --selftest                             # offline: re-derive the figures from the rules

Provenance: every record cites `CDNA4 ISA p<NN>`. Copy a distilled fact into
perf_knowledge/expert_skills/skills/gluon_authoring/references/hardware/isa-mechanisms.md WITH that citation so the next agent never re-derives it
(primary-sources.md provenance rule).
"""
import argparse
import json
import os
import re
import sys
from math import ceil
from pathlib import Path

# Database FILE NAMES. They live once, in perf_knowledge/hardware/data/, and resolve through the
# sibling `_hwdata` locator ($GEAK_HW_DATA_DIR first) -- see `_db_path`.
DB_REL = "gfx950-mfma-layout.json"
ENC_DB_REL = "gfx950-encoding.json.gz"
FACTS_DB_REL = "gfx950-isa-facts.json.gz"
DATA_HOME = "perf_knowledge/hardware/data"


def _db_path(name):
    """Path of a hardware database file, or None. `_hwdata.find` is the single locator; with it
    unreachable (this file copied out alone) only the spots beside the tool are tried."""
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    try:
        import _hwdata  # noqa: PLC0415
        return _hwdata.find(name)
    except ImportError:
        pass
    for p in (here / "data" / name, here.parent / "references" / "hardware" / name):
        if p.is_file():
            return p
    return None

# Which databases exist per architecture, and therefore which subcommands can answer.
#
# gfx950 needs all three because the Matrix Instruction Calculator has no CDNA4 support, so
# without them the ONLY path to any CDNA4 hardware fact was rendering the PDF. The other arches
# are not in that position for layout: the calculator generates their chapter-7 tables exactly,
# on demand, from the hardware rules -- so building a layout DB for them would be transcribing
# something a tool already derives. What the calculator does NOT answer is chapters 12-13, and
# THAT is what these arches were missing: an opcode, an operand field, a pseudo-code body.
#
# A `None` here is a deliberate, documented absence with a named alternative -- never a hint
# that the hardware lacks the feature. `_require` below prints the alternative.
_ARCH_DBS = {
    "gfx950": {"layout": DB_REL, "encoding": ENC_DB_REL, "facts": FACTS_DB_REL},
    "cdna3":  {"layout": None, "encoding": "cdna3-encoding.json.gz", "facts": None},
    "rdna3":  {"layout": None, "encoding": "rdna3-encoding.json.gz", "facts": None},
    "rdna4":  {"layout": None, "encoding": "rdna4-encoding.json.gz", "facts": None},
}
_ARCH_ALIAS = {"cdna4": "gfx950", "gfx942": "cdna3", "mi300": "cdna3", "mi325": "cdna3",
               "gfx1100": "rdna3", "gfx1200": "rdna4"}
ARCH = "gfx950"
# Printed beside every answer, so a record pasted into a report says which document it came from.
_ARCH_LABEL = {"gfx950": "gfx950 / CDNA4", "cdna3": "gfx942 / CDNA3",
               "rdna3": "RDNA3", "rdna4": "RDNA4"}


def arch_label():
    return _ARCH_LABEL.get(ARCH, ARCH)

# subcommand -> which database it reads. `search` and `errata` span whatever is present.
_CMD_DB = {
    "list": "layout", "facts": "layout", "layout": "layout", "locate": "layout",
    "modifiers": "layout", "sparse": "layout", "hazard": "layout",
    "encoding": "encoding", "format": "encoding", "opcodes": "encoding",
    "regs": "facts", "limits": "facts", "waitcnt": "facts", "waits": "facts",
    "mem": "facts", "lds": "facts", "tables": "facts", "concepts": "facts",
}
# What to do instead, per missing database. Stated as a command, because "not available" without
# a next step is the message that gets recorded as a hardware ceiling.
_ALTERNATIVE = {
    "layout": ("the AMD Matrix Instruction Calculator generates this arch's layout exactly -- "
               "run `kernel_workflow/scripts/kernel_tools/layout_facts.py {arch} <instr> --layout` (Tier 2, "
               "perf_knowledge/expert_skills/skills/gluon_authoring/references/hardware/primary-sources.md). No layout DB is shipped for {arch} "
               "because that tool already derives it."),
    "facts": ("ISA chapters 1-6 and 8-11 are not extracted for {arch} yet -- this is missing "
              "KNOWLEDGE, not an absent hardware feature. Read the page: "
              "`kernel_workflow/scripts/kernel_tools/hw_sources.sh pdf {arch}-isa <p0> <p1>`, and record what you find in "
              "perf_knowledge/expert_skills/skills/gluon_authoring/references/hardware/isa-mechanisms.md with its page."),
    "encoding": ("no encoding DB is shipped for {arch} -- fetch the machine-readable ISA XML "
                 "instead: `kernel_workflow/scripts/kernel_tools/hw_sources.sh xml {arch}`."),
}


def _select_arch(argv):
    """Consume a leading/anywhere `--arch <a>` and point the DB constants at it.

    Done before argparse because `main` loads a database to build the parser. Default stays
    gfx950 so every existing invocation, doc reference and catalog entry keeps working.
    """
    global ARCH, DB_REL, ENC_DB_REL, FACTS_DB_REL, XML_NAME
    out, i, want = [], 0, None
    while i < len(argv):
        a = argv[i]
        if a == "--arch" and i + 1 < len(argv):
            want, i = argv[i + 1], i + 2
            continue
        if a.startswith("--arch="):
            want, i = a.split("=", 1)[1], i + 1
            continue
        out.append(a)
        i += 1
    if want is not None:
        key = _ARCH_ALIAS.get(want.lower(), want.lower())
        if key not in _ARCH_DBS:
            gap("unknown --arch %r; known: %s (aliases: %s)"
                % (want, ", ".join(sorted(_ARCH_DBS)), ", ".join(sorted(_ARCH_ALIAS))))
        ARCH = key
        d = _ARCH_DBS[key]
        DB_REL, ENC_DB_REL, FACTS_DB_REL = d["layout"], d["encoding"], d["facts"]
        XML_NAME = _XML_NAME[key]
    return out


def _require(cmd):
    """Fail with the alternative, not with a bare absence, when this arch lacks cmd's database."""
    kind = _CMD_DB.get(cmd)
    if kind is None:
        return
    if _ARCH_DBS[ARCH][kind] is None:
        gap("`%s` reads the %s database, which is not shipped for %s.  -> %s"
            % (cmd, kind, ARCH, _ALTERNATIVE[kind].format(arch=ARCH)))
# The live XML is preferred over the committed encoding DB when it is on disk, so its filename
# has to follow --arch. The release ships one file per arch under these names.
_XML_NAME = {"gfx950": "amdgpu_isa_cdna4.xml", "cdna3": "amdgpu_isa_cdna3.xml",
             "rdna3": "amdgpu_isa_rdna3.xml", "rdna4": "amdgpu_isa_rdna4.xml"}
XML_NAME = _XML_NAME["gfx950"]
ARCHES = {"cdna4", "gfx950"}


def gap(msg):
    """Scoped tool-gap, per primary-sources.md: record it and fall back; never fabricate."""
    print(f"SCOPED TOOL-GAP: {msg}", file=sys.stderr)
    print("  -> fall back to distilled perf_knowledge/hardware/ + gluon_authoring/references/hardware/ values (probe-then-trust); "
          "never fabricate", file=sys.stderr)
    sys.exit(3)


def load_db(required=True):
    # DB_REL is None on an arch with no layout database. `main` still calls this to build the
    # parser, so return None rather than gapping -- the per-subcommand `_require` is what decides
    # whether the caller actually needed it, and only that path can name the alternative.
    if DB_REL is None:
        return None
    p = _db_path(DB_REL)
    if p is not None:
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except ValueError as e:
            gap(f"{p} is not valid JSON: {e}")
    if not required:
        return None
    gap(f"{DB_REL} not found under $GEAK_HW_DATA_DIR or {DATA_HOME}/ (layout error)")


_GZ_DB = {}


def _load_gz_db(rel, required=True):
    """Load one committed gzipped database, cached for the life of the process."""
    # rel is None for a database this arch does not ship. `search` spans all three, so it must
    # be able to ask for an absent one and get nothing rather than a type error.
    if rel is None:
        if required:
            gap(f"this database is not shipped for {ARCH}")
        return None
    if rel not in _GZ_DB:
        import gzip
        p = _db_path(rel)
        if p is not None:
            try:
                with gzip.open(p, "rt", encoding="utf-8") as fh:
                    _GZ_DB[rel] = json.load(fh)
            except (OSError, ValueError) as e:
                gap(f"{p} is unreadable: {e}")
        else:
            if not required:
                return None
            gap(f"{rel} not found under $GEAK_HW_DATA_DIR or {DATA_HOME}/ (layout error)")
    return _GZ_DB[rel]


def load_enc_db(required=True):
    """The committed encoding half: ISA chapters 12 + 13, ingested from the machine-readable XML.

    Kept gzipped (63 KB vs 2.8 MB) because it ships in every composed AMD pack. This is what
    makes `encoding` and `format` work with no XML seeded and no network -- the live XML is still
    preferred when present, since it is the authoritative and more current artifact.
    """
    return _load_gz_db(ENC_DB_REL, required)


def load_facts_db(required=True):
    """The committed prose half: ISA chapters 1-6 and 8-11, extracted from the PDF.

    This is the third of the three databases and the one with no machine-readable upstream at all:
    the XML cannot express a wait-state count, an addressing formula, a bank-conflict rule or an
    LGKM_CNT increment, and chapter 7 (matrix layout) is a separate DB. Every record carries an
    `_src` page + verbatim quote and a `_conf`, so a reader can always get back to the page.
    """
    return _load_gz_db(FACTS_DB_REL, required)


def _enc_instr(name):
    """The committed-DB counterpart of _xml_instr: look one instruction up by name."""
    edb = load_enc_db(required=False)
    if not edb:
        return None
    want, flat = name.upper(), name.upper().replace("_", "")
    for rec in edb["instructions"]:
        nm = (rec.get("name") or "").upper()
        if nm == want or nm.replace("_", "") == flat:
            return rec
    return None


# ---------------------------------------------------------------------------------------
# element widths. f8f6f4 has no fixed width: CBSZ/BLGP select it per operand at runtime.
# ---------------------------------------------------------------------------------------

ELEM_BITS = {"f64": 64, "f32": 32, "xf32": 32,
             "f16": 16, "bf16": 16,
             "i8": 8, "iu8": 8, "fp8": 8, "bf8": 8,
             "fp6": 6, "bf6": 6, "fp4": 4,
             "f8f6f4": None}


def fdiv(a, b):
    """PDF 7.1.1: 'the division operator rounds down (takes the floor)'."""
    return a // b


# ---------------------------------------------------------------------------------------
# layout_rules — transcribed from the DB's `expr` fields; see VERIFY T2/T3 for the evidence
# ---------------------------------------------------------------------------------------

def K_L(K, M, B):
    """'K_L = K / (64 / (M * B))' — consecutive k-values each lane holds.
    The PDF defines K_L with M even when it is used for the B operand."""
    return fdiv(K, fdiv(64, M * B))


def a_item_lane(M, B, K, i, k, b=0):
    """'A[b,i,k] is placed in the item k % K_L of lane i + M * (b + B * (k / K_L))'."""
    kl = K_L(K, M, B)
    return k % kl, i + M * (b + B * fdiv(k, kl))


def b_item_lane(M, N, B, K, j, k, b=0):
    """'...places B[b,k,j] in item k % K_L of lane j + N * (b + B * (k / K_L))'."""
    kl = K_L(K, M, B)
    return k % kl, j + N * (b + B * fdiv(k, kl))


def out_constants(M, N, H):
    """H = 1 for f64 instructions, 4 for every other MFMA.

    B_I is printed two ways on the same page. The bullet form 'ceil(64 / (N * M / H))' is the
    one that survives: 91/91 witness cells and 6656/6656 calculator D elements, where both
    readings of the worked box fail (55/91, and a division by zero). See VERIFY T2(b)/T2(c).
    """
    B_I = ceil(64 / fdiv(N * M, H))
    M_I = fdiv(fdiv(64, B_I), N)
    G = fdiv(M, H * M_I)
    return B_I, M_I, G


def d_item_lane(M, N, b, i, j, H):
    """'D[b,i,j] is located in item (i % H) + H * (i/(H * M_I) + G * (b / B_I))
        on lane j + N * ((i / H) % M_I + M_I * (b % B_I))'.  C uses the identical layout."""
    B_I, M_I, G = out_constants(M, N, H)
    item = (i % H) + H * (fdiv(i, H * M_I) + G * fdiv(b, B_I))
    lane = j + N * (fdiv(i, H) % M_I + M_I * (b % B_I))
    return item, lane


# ---------------------------------------------------------------------------------------
# packing_rules — item index -> (vgpr offset, bit_hi, bit_lo), low fragment first
# ---------------------------------------------------------------------------------------

def item_to_bits(item, elem_bits):
    if elem_bits == 64:
        # 'each item corresponds to a pair of registers, with the low bits ... in the first'
        return [(2 * item, 31, 0), (2 * item + 1, 31, 0)]
    if elem_bits == 32:
        return [(item, 31, 0)]
    if elem_bits == 16:
        # 'odd-numbered items taking up bits 31-16 and the even ones in bits 0-15'
        lo = 16 * (item % 2)
        return [(fdiv(item, 2), lo + 15, lo)]
    if elem_bits == 8:
        # DERIVED, NOT PRINTED. The PDF says only 'four items are packed into a register,
        # analogously to the 16-bit case'. The register assignment (item/4) is confirmed by
        # the IU8/BF16 grids; the byte position within the register is the analogy only,
        # corroborated by the calculator on CDNA3 fp8/i8 opcodes. See DB unknown #1.
        lo = 8 * (item % 4)
        return [(fdiv(item, 4), lo + 7, lo)]
    if elem_bits == 4:
        # 'item 0 lives in bits 3-0 ... item 1 is in bits 7-4 ... item 8 is placed in bits
        #  3-0 of the following register'
        lo = 4 * (item % 8)
        return [(fdiv(item, 8), lo + 3, lo)]
    if elem_bits == 6:
        # UNVERIFIED (DB unknown #4). Six registers treated as one 192-bit register; item i
        # at bits [6i+5 : 6i]. The PDF's 'item 4 in bits 35-30' is a typo for item 5 — stride
        # 6 puts item 4 at [29:24], and only item 5 crosses the 32-bit boundary the sentence
        # points at. Items 0 and 1 are unambiguous and fix the stride.
        g_lo, g_hi = 6 * item, 6 * item + 5
        v_lo, v_hi = fdiv(g_lo, 32), fdiv(g_hi, 32)
        if v_lo == v_hi:
            return [(v_lo, g_hi - 32 * v_lo, g_lo - 32 * v_lo)]
        return [(v_lo, 31, g_lo - 32 * v_lo), (v_hi, g_hi - 32 * v_hi, 0)]
    gap(f"no packing rule for {elem_bits}-bit elements")


def fmt_frags(frags):
    return " + ".join(f"v{v}[{hi}:{lo}]" for v, hi, lo in frags)


# ---------------------------------------------------------------------------------------
# instruction lookup
# ---------------------------------------------------------------------------------------

def find_instr(db, name, required=True):
    """Resolve a name against the 68 MATRIX opcodes in the layout DB.

    `required=False` returns None instead of gapping — used by the encoding commands, which
    cover all 1240 gfx950 instructions and only consult the layout DB to normalize a name.
    """
    # No layout DB on this arch (its layout comes from the calculator). The encoding commands
    # call this only to normalize a name, so "the layout DB does not know it" is the right
    # answer, not a failure. A caller that genuinely needs layout is stopped by `_require`.
    if db is None:
        if required:
            gap("no layout database for %s; `find_instr` has nothing to resolve against" % ARCH)
        return None
    want = name.upper().lstrip("_")
    by_name = {r["name"].upper(): r for r in db["instructions"]}
    if want in by_name:
        return by_name[want]
    # LLVM spells these without the underscore before the type suffix and without the block
    # token (v_mfma_f32_16x16x4f32 == V_MFMA_F32_16X16X4_F32). Normalise both sides.
    def norm(s):
        return s.upper().replace("_", "")
    cand = [r for r in db["instructions"] if norm(r["name"]) == norm(want)]
    if len(cand) == 1:
        return cand[0]
    near = sorted(n for n in by_name if want.replace("_", "") in n.replace("_", ""))
    if len(near) == 1:
        return by_name[near[0]]
    if not required:
        return None
    hint = f" (did you mean: {', '.join(near[:6])})" if near else ""
    gap(f"{name!r} is not a documented gfx950 matrix instruction{hint}; "
        f"list them with: gfx950_isa.py list   "
        f"(non-matrix instructions: gfx950_isa.py opcodes {name})")


def elem_bits_for(rec, as_fmt, operand):
    t = (as_fmt or rec.get("in_type_a" if operand != "B" else "in_type_b") or "").lower()
    if t not in ELEM_BITS:
        gap(f"unknown element type {t!r} for {rec['name']}")
    eb = ELEM_BITS[t]
    if eb is None:
        gap(f"{rec['name']} selects its operand width at RUNTIME via CBSZ (A) / BLGP (B); "
            f"re-run with --as fp8|fp6|fp4 to pick one "
            f"(see: gfx950_isa.py modifiers f8f6f4)")
    return eb


def out_bits_for(rec):
    return ELEM_BITS.get((rec.get("out_type") or "f32").lower(), 32)


def guard_dense(db, rec, operand):
    """The closed-form rules describe DENSE operands only.

    For an SMFMAC, A is 4:2-compressed and B is re-indexed through the sparsity index; the
    document gives those as tables (sparse[]), not formulas, and the instruction's own
    layout_rule_a/b point there. Computing them with the dense formula would produce a
    confident wrong answer, which is the one outcome worth engineering against. D is unaffected
    -- it still uses dense.out.
    """
    if not rec.get("sparse") or operand in ("C", "D"):
        return
    rule = rec.get("layout_rule_a") if operand == "A" else rec.get("layout_rule_b")
    gap(f"{rec['name']} is a 4:2 SPARSE instruction: operand {operand} does NOT follow the "
        f"dense closed form. Its layout is given as a table -> "
        f"gfx950_isa.py sparse {rule}   (verdict: "
        f"{db['verification']['array_verdicts']['sparse']})")


def caveats_for(db, rec):
    """The trust envelope that must travel with the answer."""
    v = db["verification"]
    out = []
    new_surface = (rec.get("sparse") or rec.get("scale")
                   or (rec.get("in_type_a") or "") in ("f8f6f4", "fp6", "bf6", "fp4")
                   or (rec.get("k") or 0) >= 64)
    if new_surface:
        out.append("CDNA4-NEW SURFACE: single-source evidence. " + v["standing_caveat"])
    if rec.get("sparse"):
        out.append(f"sparse[] verdict: {v['array_verdicts']['sparse']}")
    return out


# ---------------------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------------------

def cmd_list(db, args):
    sub = (args.substr or "").upper()
    rows = [r for r in db["instructions"] if sub in r["name"].upper()]
    if not rows:
        gap(f"no matrix instruction matches {args.substr!r}")
    for r in sorted(rows, key=lambda r: r["name"]):
        tags = []
        if r.get("sparse"):
            tags.append("sparse")
        if r.get("scale"):
            tags.append("scale")
        dims = f"{r['m']}x{r['n']}x{r['k']}" + (
            f"x{r['blocks']}b" if (r.get("blocks") or 1) != 1 else "")
        print(f"{r['name']:<44} {dims:<16} {r.get('in_type_a')}->{r.get('out_type')}"
              f"{'  [' + ','.join(tags) + ']' if tags else ''}")
    print(f"\n# {len(rows)} instruction(s). provenance: CDNA4 ISA ch.7 "
          f"(pp.{db['source']['printed_pages_covered'][0]}-"
          f"{db['source']['printed_pages_covered'][1]} printed)", file=sys.stderr)
    return 0


def cmd_facts(db, args):
    r = find_instr(db, args.instr)
    src = r.get("_src") or {}
    print(f"## {r['name']}  ({arch_label()})")
    print(f"  dims        M={r['m']} N={r['n']} K={r['k']} blocks={r.get('blocks')}"
          f"   cycles={r.get('cycles')}")
    print(f"  dtype       A={r.get('in_type_a')}  B={r.get('in_type_b')}  "
          f"D={r.get('out_type')}  integer={r.get('integer')}")
    vg = {"A": r.get("src_a_vgprs"), "B": r.get("src_b_vgprs"), "C/D": r.get("src_c_d_vgprs")}
    xml_vg, xp = {}, _xml_path(required=False)
    if any(v is None for v in vg.values()):
        # live XML first, then the committed ingest of it -- so the join survives offline.
        xml_vg = _xml_vgprs(_xml_instr(xp, r["name"])) if xp else {}
        if not any(xml_vg.values()):
            xml_vg = _enc_vgprs(_enc_instr(r["name"]))
    parts = []
    for k in ("A", "B", "C/D"):
        if vg[k] is not None:
            parts.append(f"{k}={vg[k]}")
        elif xml_vg.get(k) is not None:
            parts.append(f"{k}={xml_vg[k]}*")
        else:
            parts.append(f"{k}=?")
    print(f"  vgprs       {' '.join(parts)}"
          + ("    (* from the ISA XML OperandSize, not the PDF)" if xml_vg else ""))
    print(f"  flags       sparse={r.get('sparse')} scale={r.get('scale')} "
          f"cbsz_abid={r.get('cbsz_abid')} blgp={r.get('blgp')} opsel={r.get('opsel')}")
    print(f"  rules       A={r.get('layout_rule_a')} B={r.get('layout_rule_b')} "
          f"D={r.get('layout_rule_d')}")
    print(f"  assembles   llvm-mc -mcpu=gfx950: {r.get('llvm_mc_ok')}")
    if r.get("notes"):
        print(f"  notes       {r['notes']}")
    for c in caveats_for(db, r):
        print(f"  !! {c}")
    print(f"  # provenance: CDNA4 ISA p{src.get('printed_page')} "
          f"(pdf {src.get('pdf_page')}), sec.{src.get('section')}; conf={r.get('_conf')}")
    return 0


def _table(db, rec, operand, eb, block, lanes):
    """Build {(-vgpr,hi,lo): {lane: label}} for one operand of one instruction."""
    M, N, K = rec["m"], rec["n"], rec["k"]
    B = rec.get("blocks") or 1
    cells = {}
    if operand in ("A", "B"):
        outer = M if operand == "A" else N
        for idx in range(outer):
            for k in range(K):
                if operand == "A":
                    item, lane = a_item_lane(M, B, K, idx, k, block)
                    label = f"A[{idx}][{k}]"
                else:
                    item, lane = b_item_lane(M, N, B, K, idx, k, block)
                    label = f"B[{k}][{idx}]"
                if lane not in lanes:
                    continue
                frags = item_to_bits(item, eb)
                for n_f, f in enumerate(frags):
                    tag = label + ("" if len(frags) == 1 else (".lo" if n_f == 0 else ".hi"))
                    cells.setdefault(f, {})[lane] = tag
    else:
        H = 1 if out_bits_for(rec) == 64 else 4
        for i in range(M):
            for j in range(N):
                item, lane = d_item_lane(M, N, block, i, j, H)
                if lane not in lanes:
                    continue
                frags = item_to_bits(item, eb)
                for n_f, f in enumerate(frags):
                    tag = f"{operand}[{i}][{j}]" + ("" if len(frags) == 1
                                                   else (".lo" if n_f == 0 else ".hi"))
                    cells.setdefault(f, {})[lane] = tag
    return cells


def cmd_layout(db, args):
    r = find_instr(db, args.instr)
    operand = args.operand
    guard_dense(db, r, operand)
    eb = out_bits_for(r) if operand in ("C", "D") else elem_bits_for(r, args.as_fmt, operand)
    B = r.get("blocks") or 1
    if args.block >= B:
        gap(f"{r['name']} has {B} block(s); --block must be 0..{B - 1}")
    if args.lanes:
        try:
            lo, hi = (int(x) for x in args.lanes.split("-"))
        except ValueError:
            gap("--lanes expects lo-hi, e.g. --lanes 0-15")
        lanes = list(range(lo, hi + 1))
    else:
        lanes = list(range(64))

    cells = _table(db, r, operand, eb, args.block, set(lanes))
    if not cells:
        gap(f"no {operand} elements land on lanes {lanes[0]}-{lanes[-1]} for {r['name']}")
    rows = sorted(cells, key=lambda f: (f[0], -f[1]))
    hdr = [f"v{v}[{hi}:{lo}]" for v, hi, lo in rows]
    width = max(len(h) for h in hdr)

    print(f"# {r['name']}  operand {operand}  "
          f"({r['m']}x{r['n']}x{r['k']}, blocks={B}, block={args.block}, "
          f"{eb}-bit elements)")
    if args.markdown:
        print("| " + "register".ljust(width) + " | "
              + " | ".join(f"lane {l}" for l in lanes) + " |")
        print("|" + "-" * (width + 2) + "|"
              + "|".join("-" * (len(f"lane {l}") + 2) for l in lanes) + "|")
    for f, h in zip(rows, hdr):
        vals = [cells[f].get(l, "") for l in lanes]
        if args.markdown:
            print("| " + h.ljust(width) + " | "
                  + " | ".join(v.ljust(len(f"lane {l}")) for v, l in zip(vals, lanes)) + " |")
        else:
            print("  " + h.ljust(width) + "  " + "  ".join(vals))
    for c in caveats_for(db, r):
        print(f"!! {c}", file=sys.stderr)
    print(f"# provenance: gfx950-mfma-layout.json layout_rules + packing_rules "
          f"(CDNA4 ISA ch.7); generated, not transcribed — the DB's figures are the "
          f"transcription and --selftest checks this against them", file=sys.stderr)
    return 0


def cmd_locate(db, args):
    r = find_instr(db, args.instr)
    operand = args.operand.upper()
    guard_dense(db, r, operand)
    M, N, K = r["m"], r["n"], r["k"]
    B = r.get("blocks") or 1
    eb = out_bits_for(r) if operand in ("C", "D") else elem_bits_for(r, args.as_fmt, operand)
    i, second = args.i, args.second
    if operand == "A":
        if not (0 <= i < M and 0 <= second < K):
            gap(f"A index out of range: i must be 0..{M - 1}, k must be 0..{K - 1}")
        item, lane = a_item_lane(M, B, K, i, second, args.block)
        what = f"A[{i}][{second}]"
    elif operand == "B":
        if not (0 <= i < N and 0 <= second < K):
            gap(f"B index out of range: j must be 0..{N - 1}, k must be 0..{K - 1}")
        item, lane = b_item_lane(M, N, B, K, i, second, args.block)
        what = f"B[{second}][{i}]"
    elif operand in ("C", "D"):
        if not (0 <= i < M and 0 <= second < N):
            gap(f"{operand} index out of range: i must be 0..{M - 1}, j must be 0..{N - 1}")
        H = 1 if eb == 64 else 4
        item, lane = d_item_lane(M, N, args.block, i, second, H)
        what = f"{operand}[{i}][{second}]"
    else:
        gap("operand must be one of A B C D")
    frags = item_to_bits(item, eb)
    print(f"{r['name']}  {what}  (block {args.block})")
    print(f"  lane  {lane}")
    print(f"  item  {item}   ({eb}-bit element)")
    print(f"  bits  {fmt_frags(frags)}   (VGPR offsets are relative to the operand's first "
          f"register)")
    if len(frags) > 1 and eb == 6:
        print("  !! this 6-bit item CROSSES a 32-bit register boundary")
    for c in caveats_for(db, r):
        print(f"  !! {c}")
    print(f"  # provenance: gfx950-mfma-layout.json {r.get('layout_rule_a') if operand == 'A' else r.get('layout_rule_b') if operand == 'B' else r.get('layout_rule_d')}"
          f" (CDNA4 ISA ch.7)")
    return 0


def _xml_path(required=True):
    for env in ("TILE_HW_SEED", "TILE_HW_TOOLS", "TILE_HW_CACHE"):
        d = os.environ.get(env)
        if d and os.path.isfile(os.path.join(d, XML_NAME)):
            return os.path.join(d, XML_NAME)
    if not required:
        return None
    gap(f"{XML_NAME} absent from the image seed and runtime cache "
        f"(TILE_HW_SEED/TILE_HW_TOOLS/TILE_HW_CACHE); fetch it with: "
        f"kernel_workflow/scripts/kernel_tools/hw_sources.sh xml cdna4")


def _xml_instr(path, name):
    """Stream the 8.7 MB XML and return the <Instruction> element for `name`, or None."""
    import xml.etree.ElementTree as ET
    want, flat = name.upper(), name.upper().replace("_", "")
    for _event, el in ET.iterparse(path, events=("end",)):
        if not el.tag.endswith("Instruction"):
            continue
        nm = (el.findtext("InstructionName") or "").upper()
        if nm == want or nm.replace("_", "") == flat:
            return el
        el.clear()
    return None


def _xml_vgprs(el):
    """SrcA/SrcB/SrcC-D register counts from the encoding's OperandSize (bits / 32).

    The PDF's Table 28 has no register-count column, so these fields are null in the layout DB
    for most opcodes. The XML states operand sizes for every one of them -- this is exactly the
    join that makes the two halves worth keeping in one tool.
    """
    if el is None:
        return {}
    by_field = {}
    for enc in el.iter("InstructionEncoding"):
        for o in enc.iter("Operand"):
            sz = o.findtext("OperandSize")
            if sz and sz.isdigit():
                by_field[(o.findtext("FieldName") or "").upper()] = int(sz) // 32
    return {"A": by_field.get("SRC0"), "B": by_field.get("SRC1"),
            "C/D": by_field.get("SRC2") or by_field.get("VDST")}


def _enc_vgprs(rec):
    """_xml_vgprs against the committed encoding DB — same join, no XML on disk."""
    if not rec:
        return {}
    by_field = {}
    for enc in rec.get("encodings", []):
        for o in enc.get("operands", []):
            if o.get("size_bits"):
                by_field[(o.get("field") or "").upper()] = o["size_bits"] // 32
    return {"A": by_field.get("SRC0"), "B": by_field.get("SRC1"),
            "C/D": by_field.get("SRC2") or by_field.get("VDST")}


def cmd_encoding(db, args):
    """Tier 1: the encoding half of the gfx950 database, joined to the layout half by name.

    Works on ANY of the 1240 gfx950 instructions, not just the 68 matrix opcodes: the layout DB
    is consulted only to normalize the name when it happens to know it. Prefers the live ISA XML
    and falls back to the committed encoding DB, so this answers offline.
    """
    name = args.instr
    for rec in (find_instr(db, name, required=False),):
        if rec:
            name = rec["name"]

    path = _xml_path(required=False)
    found = _xml_instr(path, name) if path else None
    if found is not None:
        desc = (found.findtext("Description") or "").strip()
        encs = []
        for enc in found.iter("InstructionEncoding"):
            op = enc.find("Opcode")
            ops = [(o.findtext("FieldName"), o.findtext("DataFormatName"),
                    o.findtext("OperandType"), o.findtext("OperandSize"),
                    o.get("Input") == "true", o.get("Output") == "true")
                   for o in enc.iter("Operand")]
            encs.append((enc.findtext("EncodingName"), op.text if op is not None else "?",
                         enc.findtext("EncodingCondition"), ops))
        origin = f"cdna4 ISA XML ({os.path.basename(path)})"
        group = None
    else:
        erec = _enc_instr(name)
        if erec is None:
            edb = load_enc_db(required=False)
            where = "the ISA XML nor the committed encoding DB" if path else \
                    "the committed encoding DB (and no ISA XML is seeded)"
            gap(f"no gfx950 instruction named {args.instr!r} in {where}"
                + (f" — {len(edb['instructions'])} instructions are indexed; "
                   f"list them with: gfx950_isa.py opcodes <substr>" if edb else ""))
        name, desc, group = erec["name"], erec.get("description") or "", erec.get("functional_group")
        encs = [(e.get("encoding"), e.get("opcode"), e.get("condition"),
                 [(o.get("field"), o.get("format"), o.get("type"), o.get("size_bits"),
                   o.get("input"), o.get("output")) for o in e.get("operands", [])])
                for e in erec.get("encodings", [])]
        edb = load_enc_db()
        origin = f"committed encoding DB ({edb['source']['release']})"

    print(f"## {name}  ({arch_label()}) — encoding")
    if desc:
        print(f"  description {desc}")
    if group:
        print(f"  group       {group}")
    for ename, opcode, cond, ops in encs:
        print(f"  encoding    {ename}   opcode={opcode}   cond={cond}")
        for field, fmt, otype, size, is_in, is_out in ops:
            print(f"    {str(field):<8} {str(fmt):<20} {str(otype):<18} size={size} "
                  f"{'in' if is_in else ''}{'/out' if is_out else ''}")
        print(f"                -> bit fields of this encoding: gfx950_isa.py format {ename}")

    # Semantics come from ISA chapter 12 and live only in the committed DB, so look them up
    # regardless of which source supplied the encodings above.
    sem = _enc_instr(name) or {}
    src = sem.get("_src") or {}
    if sem.get("operation"):
        pp = src.get("operation_pages") or []
        where = (f" (printed p.{pp[0] - 8}" + (f"-{pp[-1] - 8}" if len(pp) > 1 else "") + ")") if pp else ""
        print(f"\n  operation   # exact pseudo-code as printed in ISA ch.12{where}")
        for line in sem["operation"].splitlines():
            print(f"    {line}")
    elif "operation" in sem:
        print("\n  operation   -- the ISA prints no pseudo-code box for this instruction.")
        print("              Control/cache/LDS-DMA instructions are documented in prose only;")
        print("              see the description above, and: gfx950_isa.py errata")
    if sem.get("pdf_notes"):
        print("\n  notes")
        for line in _wrap(sem["pdf_notes"], 74):
            print(f"    {line}")
    if sem.get("examples"):
        print("\n  examples    # worked examples printed alongside the instruction")
        for line in sem["examples"].splitlines():
            print(f"    {line}")
    if src.get("pdf_page"):
        also = src.get("also_stated_on")
        print(f"\n  # semantics: ISA ch.12 pdf p.{src['pdf_page']} (printed p.{src['printed_page']})"
              + (f", {src['section']}" if src.get("section") else "")
              + (f"; also stated on pdf p.{', p.'.join(map(str, also))}" if also else ""))
    print(f"  # provenance: {origin} <{name}>"
          + (f"; layout for the same opcode: gfx950_isa.py layout {name}"
             if find_instr(db, name, required=False) else ""))
    return 0


def cmd_format(db, args):
    """ISA chapter 13 (Microcode Formats): the bit-field layout of one encoding."""
    edb = load_enc_db()
    want = (args.encoding or "").upper().lstrip(".")
    hits = [e for e in edb["encodings"]
            if want in (e["name"] or "").upper() or want in e["id"].upper()]
    if not hits:
        gap(f"no encoding matches {args.encoding!r} — the {len(edb['encodings'])} gfx950 "
            f"encodings are: " + ", ".join(sorted(e["name"] for e in edb["encodings"])))
    for e in hits:
        print(f"## {e['name']}  ({e['bit_count']}-bit)")
        if e.get("description"):
            print(f"  {e['description']}")
        if e.get("conditions"):
            print(f"  conditions: {', '.join(e['conditions'])}")
        print()
        for f in e["fields"]:
            rng = " + ".join(f"[{r['bits'][0]}:{r['bits'][1]}]" for r in f["ranges"]) or "?"
            cond = " (conditional)" if f.get("is_conditional") else ""
            print(f"  {f['field']:<12} {rng:<18}{cond}")
            if f.get("description"):
                for line in _wrap(f["description"], 76):
                    print(f"               {line}")
        # Chapter-12 material that applies to this encoding as a whole rather than to any one
        # instruction: how a VOP1/VOP2/VOPC opcode maps into VOP3, and which instructions may
        # not carry a DPP or SDWA suffix. Printed pages apart from the bit fields, needed together.
        ename = (e["name"] or "").upper()
        base = ename.replace("ENC_", "").split("_")[0]      # ENC_VOP1 -> VOP1, VOP2_VOP_DPP -> VOP2
        for rule in edb.get("chapter12_rules") or []:
            frm = (rule.get("from_encoding") or "").upper().replace("ENC_", "")
            if (frm and frm == base) or (not frm and base in (rule.get("id") or "").upper()):
                to = (rule.get("to_encoding") or "").replace("ENC_", "")
                print(f"\n  rule: {rule.get('id')}"
                      + (f"   {frm} -> {to}  opcode offset +{rule.get('offset')}" if frm else ""))
                for line in _wrap(rule.get("statement") or "", 74):
                    print(f"    {line}")
                # Every promotion sentence in the ISA says "VOP2 opcode" regardless of the section
                # it appears in -- a copy-paste slip. The offset above is the authoritative number.
                if rule.get("notes"):
                    for line in _wrap(rule["notes"], 72):
                        print(f"    ! {line}")
                s = rule.get("_src") or {}
                if s.get("printed_page"):
                    print(f"    # ISA ch.12 printed p.{s['printed_page']}")
        for lim in edb.get("instruction_limitations") or []:
            mod = (lim.get("modifier") or "").upper()
            if mod and mod in ename:
                ents = lim.get("instructions") or []
                print(f"\n  limitation: {lim.get('id')}  ({len(ents)} instruction(s) excluded)")
                for line in _wrap(lim.get("statement") or "", 74):
                    print(f"    {line}")
                if lim.get("stale_mnemonics"):
                    print(f"    !! the printed list names {', '.join(lim['stale_mnemonics'])}, which "
                          f"do not exist on gfx950 -- carried over from an earlier ISA document")
                print(f"    excluded: {', '.join(map(str, ents[:10]))}"
                      + (f" ... ({len(ents)} total)" if len(ents) > 10 else ""))
        print(f"\n  # provenance: cdna4 ISA XML {edb['source']['release']} encoding <{e['name']}>"
              f" (= ISA ch.13 Microcode Formats, mechanically ingested)")
    return 0


def _wrap(text, width):
    out, line = [], ""
    for word in (text or "").split():
        if len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out


def cmd_opcodes(db, args):
    """ISA chapter 12 (Instructions): search all 1240 gfx950 instructions by name/group/encoding."""
    edb = load_enc_db()
    sub = (args.substr or "").lower()
    rows = []
    for r in edb["instructions"]:
        blob = " ".join([r["name"] or "", r.get("functional_group") or "",
                         " ".join(e.get("encoding") or "" for e in r.get("encodings", []))]).lower()
        if sub and sub not in blob:
            continue
        e0 = (r.get("encodings") or [{}])[0]
        rows.append((r["name"], e0.get("encoding") or "?", e0.get("opcode"),
                     r.get("functional_group") or "?", (r.get("description") or "")[:60]))
    if not rows:
        gap(f"no gfx950 instruction matches {args.substr!r} "
            f"(searched name, functional group and encoding across "
            f"{len(edb['instructions'])} instructions)")
    for name, enc, op, grp, desc in sorted(rows):
        print(f"  {name:<34} {enc:<18} op={str(op):<5} {grp:<8} {desc}")
    print(f"\n  {len(rows)} of {len(edb['instructions'])} instructions"
          f"   # provenance: cdna4 ISA XML {edb['source']['release']} (= ISA ch.12)")
    return 0


def _dump(db, key, substr, fields, soft=False):
    """`soft=True` returns 1 on no match instead of exiting, so a caller that searches a second
    database afterwards is not cut off by an empty result from the first."""
    # db is None on an arch with no layout database. `errata` and `search` span several
    # databases, so an absent one must yield "nothing from here", not an exception.
    if db is None:
        if soft:
            return 1
        gap(f"no {key} database for {ARCH}")
    sub = (substr or "").lower()
    rows = [r for r in db[key]
            if not sub or sub in json.dumps(r).lower()]
    if not rows:
        if soft:
            return 1
        gap(f"no {key} record matches {substr!r}")
    for r in rows:
        src = r.get("_src") or {}
        print(f"## {r.get('id') or r.get('name')}")
        for f in fields:
            if r.get(f) not in (None, "", [], {}):
                v = r[f]
                print(f"  {f:<34} {json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v}")
        if src:
            print(f"  # provenance: CDNA4 ISA p{src.get('printed_page')} "
                  f"sec.{src.get('section')}; conf={r.get('_conf')}")
        print()
    verdict = db["verification"]["array_verdicts"].get(key)
    print(f"# {len(rows)} record(s) from {key}"
          + (f"; verdict: {verdict}" if verdict else ""), file=sys.stderr)
    return 0


def cmd_modifiers(db, args):
    return _dump(db, "modifiers", args.substr,
                 ["field", "applies_to", "bits", "semantics", "value_table", "constraints",
                  "scope", "notes"])


def cmd_hazard(db, args):
    return _dump(db, "hazards", args.substr,
                 ["producer", "consumer", "condition", "required_independent_instructions",
                  "required_nops", "required_waits", "sub_row_labels", "pdf_comment", "notes"])


def cmd_sparse(db, args):
    return _dump(db, "sparse",
                 args.substr,
                 ["instructions", "compression", "index_bits_per_group", "index_operand",
                  "index_placement", "operand_layout_rule", "table", "notes"])


def cmd_errata(db, args):
    """Every place the source document is known to be wrong, or to document nothing at all.
    `applied_to_data: false` means the DB kept the document's value -- so a consumer reading that
    field gets the printed error unless they read this list too.

    Spans two databases: the chapter-7 layout errors, and the chapter-12/13 encoding errors and
    documentation gaps. A caller does not know in advance which chapter their problem lives in.
    """
    rc = _dump(db, "known_source_errors", args.substr,
               ["section", "printed_page", "where", "verbatim", "corrected_reading",
                "why_certain", "applied_to_data", "confidence", "note"], soft=True)
    edb = load_enc_db(required=False)
    sub = (args.substr or "").lower()
    erows = [r for r in (edb or {}).get("known_source_errors") or []
             if not sub or sub in json.dumps(r, ensure_ascii=False).lower()]
    for r in erows:
        src = r.get("_src") or {}
        print(f"## {r.get('id')}")
        for f in ("statement", "resolution", "affected_count"):
            if r.get(f) not in (None, "", [], {}):
                print(f"  {f:<34} {r[f]}")
        if r.get("affected"):
            aff = r["affected"]
            print(f"  {'affected':<34} {', '.join(aff[:8])}"
                  + (f" ... ({len(aff)} total)" if len(aff) > 8 else ""))
        if src.get("printed_page"):
            print(f"  # provenance: {arch_label()} ISA p{src['printed_page']} "
                  f"sec.{src.get('section')}; conf={r.get('_conf')}")
        print()
    print(f"# {len(erows)} record(s) from the encoding DB (ISA ch.12-13)", file=sys.stderr)
    if rc and not erows:
        # "This document has no known errors" is an ANSWER, and on an unfiltered query it is a
        # checked one: the encoding DB's opcode test compares every printed heading against the
        # XML, so an empty list means that test found nothing -- not that nobody looked. Gapping
        # here is how a positive result gets recorded as a tool-gap and the caller falls back.
        if args.substr:
            gap(f"no errata record matches {args.substr!r} in either database")
        edb = edb or {}
        t = ((edb.get("verification") or {}).get("tests") or {}).get("T3_opcode_check")
        print(f"## no known source errors for {arch_label()}")
        print("  This is a checked result, not an absence of checking: every printed instruction"
              "\n  heading in this document was compared against the machine-readable ISA XML.")
        if t:
            print(f"  {t}")
        print("  Scope: ISA chapters 12-13. Chapters 1-6 and 8-11 are not extracted for this"
              "\n  arch, so a defect there would not be listed here."
              if FACTS_DB_REL is None else "")
        return 0
    return 0


# ---------------------------------------------------------------------------------------
# ISA chapters 1-6 and 8-11 — the prose/table half, from gfx950-isa-facts.json.gz
# ---------------------------------------------------------------------------------------

def _dump2(key, substr, fields):
    """`_dump` against the facts DB. Kept separate because the two DBs carry independent
    per-array verdicts, and printing a chapter-7 verdict beside a chapter-11 record would
    misattribute the evidence behind it."""
    fdb = load_facts_db()
    sub = (substr or "").lower()
    rows = [r for r in fdb.get(key) or [] if not sub or sub in json.dumps(r, ensure_ascii=False).lower()]
    if not rows:
        gap(f"no {key} record matches {substr!r} "
            f"(searched {len(fdb.get(key) or [])} records; the facts DB covers ISA ch.1-6 and 8-11 "
            f"-- matrix layout is `layout`/`facts`, opcodes and bit fields are `opcodes`/`format`)")
    for r in rows:
        src = r.get("_src") or {}
        print(f"## {r.get('id') or r.get('name')}")
        for f in fields:
            if r.get(f) not in (None, "", [], {}):
                v = r[f]
                print(f"  {f:<32} {json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v}")
        if src:
            print(f"  # provenance: CDNA4 ISA p{src.get('printed_page')} (pdf p{src.get('pdf_page')})"
                  f" sec.{src.get('section')}; conf={r.get('_conf')}")
            if r.get("_conf") == "uncertain":
                print("  !! _conf=uncertain: the extractor read this but could not confirm it. "
                      "Re-read the page before you act on it.")
        print()
    verdict = ((fdb.get("verification") or {}).get("array_verdicts") or {}).get(key)
    print(f"# {len(rows)} record(s) from {key}" + (f"; verdict: {verdict}" if verdict else ""),
          file=sys.stderr)
    return 0


def cmd_regs(db, args):
    """ISA ch.3/5/6: MODE, STATUS, TRAPSTS, HW_ID, M0 and friends, with their bit fields."""
    return _dump2("registers", args.substr,
                  ["name", "width_bits", "scope", "access", "fields", "notes"])


def cmd_limits(db, args):
    """Capacities, allocation granularity and architectural limits -- the numbers an occupancy
    or register budget is built from. `granularity` is the one most often gotten wrong."""
    return _dump2("resources", args.substr,
                  ["name", "value", "unit", "granularity", "scope", "shared_with",
                   "constraints", "notes"])


def cmd_waitcnt(db, args):
    """The S_WAITCNT model: what VM_CNT / LGKM_CNT / EXP_CNT count, and in what order they return."""
    return _dump2("counters", args.substr,
                  ["name", "expands_to", "max_value", "increments", "decrements", "ordering",
                   "notes"])


def cmd_waits(db, args):
    """Required software-inserted wait states / NOPs (ISA ch.4 Table 11 and scattered prose).

    Chapter 7's MFMA-specific hazards are a separate array in the layout DB -- see `hazard`."""
    return _dump2("wait_states", args.substr,
                  ["producer", "consumer", "condition", "required_independent_instructions",
                   "required_nops", "pdf_comment", "notes"])


def cmd_mem(db, args):
    """Addressing formulas, alignment, bounds checking and resource descriptors (ch.8-10)."""
    return _dump2("memory_ops", args.substr,
                  ["applies_to", "topic", "formula", "vars", "alignment", "bounds_rule",
                   "fields", "notes"])


def cmd_lds(db, args):
    """LDS banking, conflict rules and the MFMA transpose loads (ch.11)."""
    return _dump2("lds_facts", args.substr,
                  ["topic", "statement", "banks", "bank_width_bytes", "conflict_rule", "cycles",
                   "applies_to", "notes"])


def cmd_tables(db, args):
    """Verbatim transcriptions of source tables that no typed array fully captures."""
    fdb = load_facts_db()
    edb = load_enc_db(required=False)
    sub = (args.substr or "").lower()
    # The compare-operation tables (59-62) live in the encoding DB because they are chapter-12
    # material; a caller asking for "Table 60" does not care which file it landed in.
    pool = (fdb.get("tables") or []) + ((edb or {}).get("tables") or [])
    rows = [r for r in pool
            if not sub or sub in json.dumps(r, ensure_ascii=False).lower()]
    if not rows:
        gap(f"no table matches {args.substr!r} among {len(pool)} transcribed tables")
    for r in rows:
        src = r.get("_src") or {}
        print(f"## {r.get('number')}  {r.get('title') or ''}   [{r.get('id')}]")
        cols = r.get("columns") or []
        rows_ = r.get("rows") or []
        # 47 of the 48 transcribed tables carry rows as positional lists; one carries them as
        # column-keyed dicts. Indexing a dict by position raised KeyError and took down the whole
        # unfiltered `tables` listing -- so the command worked for a search term and crashed for
        # the reader who asked what tables exist. Normalise on the column order instead of
        # requiring one shape from the extraction.
        if cols and any(isinstance(row, dict) for row in rows_):
            rows_ = [[row.get(c, "") for c in cols] if isinstance(row, dict) else row
                     for row in rows_]
        if cols:
            widths = [max([len(str(c))] + [len(str(row[i])) for row in rows_ if i < len(row)])
                      for i, c in enumerate(cols)]
            widths = [min(w, 38) for w in widths]
            print("  " + " | ".join(str(c)[:w].ljust(w) for c, w in zip(cols, widths)))
            print("  " + "-+-".join("-" * w for w in widths))
            for row in rows_:
                print("  " + " | ".join(
                    str(row[i])[:w].ljust(w) if i < len(row) else " " * w
                    for i, w in enumerate(widths)))
        if r.get("also_parsed_into"):
            print(f"  (also parsed into `{r['also_parsed_into']}`)")
        if r.get("notes"):
            print(f"  notes: {r['notes']}")
        print(f"  # provenance: CDNA4 ISA p{src.get('printed_page')} (pdf p{src.get('pdf_page')});"
              f" conf={r.get('_conf')}\n")
    return 0


def cmd_concepts(db, args):
    """One actionable prose fact per record: ordering guarantees, restrictions, 'X unless Y' rules."""
    return _dump2("concepts", args.substr,
                  ["topic", "statement", "applies_to", "actionable_for", "notes"])


def _search_matcher(q):
    """Agents name an instruction family the way the ISA prose -- and our own skill docs -- name it:
    `DS_READ_*_TR_*`, `ds_read_tr`, `ds read tr`. None of those is a substring of the real name
    `DS_READ_B64_TR_B16`, so a plain substring search answers "no match" and sends the agent back to
    guessing from memory, which is the exact failure this database exists to prevent. Try the literal
    substring first; if the query carries more than one token, also accept a record containing every
    token. Returns (predicate, loosened?)."""
    q = (q or "").lower()
    toks = [t for t in re.split(r"[^a-z0-9]+", q) if t]
    if len(toks) < 2:
        return (lambda blob: q in blob), False

    def loose(blob):
        if q in blob:
            return True
        # Match each query token against the record's own WORD SEGMENTS, not against the raw
        # string. A raw substring test makes short tokens useless -- `tr` is a substring of
        # "instruction" -- which floods a query like `ds_read_tr` with unrelated records. Matching
        # `tr` against the segment `TR` of DS_READ_B64_TR_B16 is what the caller meant.
        segs = set(re.split(r"[^a-z0-9]+", blob))
        return all(any(g == t or g.startswith(t) for g in segs) for t in toks)

    return loose, True


def cmd_search(db, args):
    """Search all three gfx950 databases at once, so a caller who does not yet know which one
    holds the answer still finds it. This is the entry point to reach for first."""
    sub = (args.substr or "").lower()
    if not sub:
        gap("search needs a substring")
    match, loosened = _search_matcher(sub)
    hits = 0
    # A loosened multi-token query can match 60+ records. Every one is a real hit, but a
    # 60-line wall is not a usable answer, so print the first few per section and report the
    # rest as a count -- the total stays honest and the caller knows to narrow the query.
    PER_SECTION = 12
    shown, extra = {}, {}
    def emit(key, line):
        if shown.get(key, 0) < PER_SECTION:
            shown[key] = shown.get(key, 0) + 1
            print(line)
        else:
            extra[key] = extra.get(key, 0) + 1

    fdb = load_facts_db(required=False)
    for key in ("registers", "resources", "counters", "wait_states", "memory_ops",
                "lds_facts", "tables", "concepts"):
        for r in (fdb or {}).get(key) or []:
            if match(json.dumps(r, ensure_ascii=False).lower()):
                src = r.get("_src") or {}
                label = r.get("statement") or r.get("title") or r.get("name") or r.get("topic") or ""
                emit(f"facts/{key}",
                     f"  facts/{key:<12} {r.get('id') or '?':<38} p{src.get('printed_page')}"
                     f"  {str(label)[:60]}")
                hits += 1

    ldb = db or {}
    for key in ("instructions", "layout_rules", "packing_rules", "modifiers", "sparse",
                "hazards", "formats"):
        for r in ldb.get(key) or []:
            if match(json.dumps(r, ensure_ascii=False).lower()):
                emit(f"layout/{key}",
                     f"  layout/{key:<11} {r.get('id') or r.get('name') or '?':<38}"
                     f"  (ISA ch.7 -- use `facts`/`layout`/`hazard`)")
                hits += 1

    edb = load_enc_db(required=False)
    for key in ("chapter12_rules", "tables", "instruction_limitations", "known_source_errors"):
        for r in (edb or {}).get(key) or []:
            if match(json.dumps(r, ensure_ascii=False).lower()):
                src = r.get("_src") or {}
                label = r.get("statement") or r.get("title") or ""
                emit(f"enc/{key}",
                     f"  enc/{key:<16} {r.get('id') or '?':<30} p{src.get('printed_page')}"
                     f"  {str(label)[:58]}")
                hits += 1

    enc_hits = [r for r in (edb or {}).get("instructions") or []
                if match((r.get("name") or "").lower())]
    for r in enc_hits[:20]:
        e0 = (r.get("encodings") or [{}])[0]
        print(f"  enc/instruction  {r['name']:<38} {e0.get('encoding') or '?'}"
              f" op={e0.get('opcode')}")
    hits += len(enc_hits)
    if len(enc_hits) > 20:
        print(f"  ... and {len(enc_hits) - 20} more instruction names "
              f"-- narrow with `opcodes {args.substr}`")

    if not hits:
        gap(f"nothing in the three gfx950 databases matches {args.substr!r}. Before concluding the "
            "document does not cover it, retry with ONE distinctive word (`transpose`, `xnack`, "
            "`scale`) -- the databases hold the ISA's own wording, which is often not the wording a "
            "kernel author uses. Only if a one-word query also comes back empty is this a real "
            "coverage answer: the databases carry ch.1-13, but only what each source can express.")
    for key in sorted(extra):
        print(f"  ... and {extra[key]} more in {key} -- narrow the query to see them")
    print(f"\n  {hits} hit(s) across the facts / layout / encoding databases"
          + ("  (matched on all tokens of the query, not the literal string)" if loosened else ""),
          file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------------------
# selftest — re-derive the transcribed grids from the rules. Offline, no GPU, no network.
# ---------------------------------------------------------------------------------------

def _check_figure(db, fig):
    """The figure was read by eye off a rendered image; the rules came from the text layer.
    Agreement is real evidence because the two arrived by independent paths."""
    sh = fig["shape"]
    M, N, K = sh["m"], sh["n"], sh["k"]
    eb = fig["elem_bits"]
    ok = bad = 0
    for cell in fig["cells"]:
        if cell.get("k") is None or cell.get("lanes") is None:
            continue
        l0, l1 = cell["lanes"]
        m0, m1 = cell["mn"]
        v0 = int(str(cell["vgprs"][0]).lstrip("v"))
        v1 = int(str(cell["vgprs"][-1]).lstrip("v"))
        for i in range(m0, m1 + 1):
            for k in range(cell["k"][0], cell["k"][1] + 1):
                item, lane = a_item_lane(M, 1, K, i, k, 0)
                vg = {f[0] for f in item_to_bits(item, eb)}
                if l0 <= lane <= l1 and all(v0 <= v <= v1 for v in vg):
                    ok += 1
                else:
                    bad += 1
    return ok, bad


def cmd_selftest(db, args):
    v = db["verification"]
    unusable = set(v["unusable_ids"])
    fail = 0

    # 1. referential integrity. A sparse instruction's operand rule lives in sparse[], not in
    # layout_rules[] -- the 4:2 layouts are tables, not closed forms -- so both namespaces count.
    rule_ids = ({r["id"] for r in db["layout_rules"]} | {r["id"] for r in db["sparse"]})
    pack_ids = {r["id"] for r in db["packing_rules"]}
    dangling = []
    for r in db["instructions"]:
        for f in ("layout_rule_a", "layout_rule_b", "layout_rule_d"):
            if r.get(f) and r[f] not in rule_ids:
                dangling.append(f"{r['name']}.{f}={r[f]}")
        for f in ("packing_rule_a", "packing_rule_b", "packing_rule_d"):
            if r.get(f) and r[f] not in pack_ids:
                dangling.append(f"{r['name']}.{f}={r[f]}")
    print(f"[{'ok ' if not dangling else 'FAIL'}] referential integrity: "
          f"{len(dangling)} dangling rule reference(s)")
    fail += bool(dangling)
    for d in dangling[:5]:
        print(f"       {d}")

    # 2. provenance present on every record
    missing = [r.get("id") or r.get("name")
               for key in ("instructions", "layout_rules", "packing_rules", "figure_layouts",
                           "modifiers", "formats", "sparse", "hazards")
               for r in db[key] if not r.get("_src") or not r.get("_conf")]
    print(f"[{'ok ' if not missing else 'FAIL'}] provenance: "
          f"{len(missing)} record(s) missing _src/_conf")
    fail += bool(missing)

    # 3. the real test: rules vs transcribed grids
    tot_ok = tot_bad = 0
    for fig in db["figure_layouts"]:
        ok, bad = _check_figure(db, fig)
        flag = "known-bad" if fig["id"] in unusable else ("ok " if bad == 0 else "FAIL")
        print(f"[{flag:>9}] {fig['id']:<24} {ok}/{ok + bad} cells reproduce from the rules")
        if fig["id"] in unusable:
            # The DB claims these two grids are drawn in error. If they suddenly agreed, the
            # claim would need re-examining -- so a clean pass here is itself a failure.
            if bad == 0:
                print(f"       FAIL: {fig['id']} is flagged unusable but now agrees with the "
                      f"rules; re-open the finding")
                fail += 1
            continue
        tot_ok += ok
        tot_bad += bad
        fail += bool(bad)
    print(f"[{'ok ' if tot_bad == 0 else 'FAIL'}] usable grids: {tot_ok}/{tot_ok + tot_bad} "
          f"cells reproduce from layout_rules + packing_rules")

    # 4. spot-check the packing rules the calculator confirmed at bit granularity
    checks = [(16, 0, [(0, 15, 0)]), (16, 3, [(1, 31, 16)]),
              (32, 5, [(5, 31, 0)]), (8, 6, [(1, 23, 16)]), (4, 9, [(1, 7, 4)]),
              (6, 5, [(0, 31, 30), (1, 3, 0)])]
    bad_pack = [(eb, it) for eb, it, want in checks if item_to_bits(it, eb) != want]
    print(f"[{'ok ' if not bad_pack else 'FAIL'}] packing rules: "
          f"{len(checks) - len(bad_pack)}/{len(checks)} known placements")
    fail += bool(bad_pack)

    # 5. the dense formulas must refuse sparse A/B rather than answering confidently
    leaked = []
    real_err, sys.stderr = sys.stderr, open(os.devnull, "w")   # the guard's message is the point
    try:                                                       # of the guard, not of this probe
        for r in db["instructions"]:
            if not r.get("sparse"):
                continue
            for op in ("A", "B"):
                try:
                    guard_dense(db, r, op)
                    leaked.append(f"{r['name']}.{op}")
                except SystemExit:
                    pass
    finally:
        sys.stderr.close()
        sys.stderr = real_err
    n_sparse = sum(1 for r in db["instructions"] if r.get("sparse"))
    print(f"[{'ok ' if not leaked else 'FAIL'}] sparse guard: {n_sparse} sparse instruction(s), "
          f"{len(leaked)} would have been answered with the dense formula")
    fail += bool(leaked)

    # --- the other two databases. Absent is a FAIL, not a skip: they ship committed in the pack,
    # so "not found" means a broken pack layout, which is exactly what a selftest should catch.
    edb = load_enc_db(required=False)
    if edb is None:
        print("[FAIL] encoding DB (ISA ch.12-13): not found next to the script")
        fail += 1
    else:
        n_op = sum(len(i.get("encodings") or []) for i in edb["instructions"])
        bad = [e["name"] for e in edb["encodings"]
               for f in e["fields"]
               for r in f["ranges"]
               if not (0 <= r["bits"][1] <= r["bits"][0] < (e["bit_count"] or 64))]
        print(f"[{'ok ' if not bad else 'FAIL'}] encoding DB: {len(edb['instructions'])} instructions, "
              f"{n_op} opcode entries, {len(edb['encodings'])} encodings; "
              f"{len(bad)} field range(s) outside their encoding's own bit width")
        fail += bool(bad)

        # The chapter-12 semantics layer. `operation: null` is a real answer (the ISA prints no
        # code box for 48 control/cache/LDS-DMA instructions), so count the three states apart:
        # a missing `operation` KEY means the instruction has no chapter-12 block at all.
        withkey = [i for i in edb["instructions"] if "operation" in i]
        withop = [i for i in withkey if i["operation"]]
        noblock = len(edb["instructions"]) - len(withkey)
        nopage = [i["name"] for i in withkey if not (i.get("_src") or {}).get("pdf_page")]
        print(f"[{'ok ' if not nopage else 'FAIL'}] ch.12 semantics: {len(withop)} instructions with "
              f"pseudo-code, {len(withkey) - len(withop)} documented but box-free, {noblock} with no "
              f"ch.12 block; {len(nopage)} without a page citation")
        fail += bool(nopage)
        missing_r = [k for k in ("chapter12_rules", "tables", "instruction_limitations")
                     if not edb.get(k)]
        print(f"[{'ok ' if not missing_r else 'FAIL'}] ch.12 non-instruction material: "
              f"{len(edb.get('chapter12_rules') or [])} promotion rules, "
              f"{len(edb.get('tables') or [])} compare tables, "
              f"{len(edb.get('instruction_limitations') or [])} DPP/SDWA exclusion lists"
              + (f"; MISSING {missing_r}" if missing_r else ""))
        fail += bool(missing_r)

    # Reachability: agents arrive from the skill docs knowing layout_facts.py, which forwards
    # gfx950 invocations here. If a subcommand exists here but not in that forwarding set it is
    # unreachable by the documented route, so hold the two lists equal.
    lf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "layout_facts.py")
    if not os.path.isfile(lf):
        print("[FAIL] reachability: layout_facts.py is missing from this pack")
        fail += 1
    else:
        src = open(lf, encoding="utf-8").read()
        m = re.search(r"GFX950_SUBCOMMANDS\s*=\s*\{(.*?)\}", src, re.S)
        fwd = set(re.findall(r'"([a-z0-9_]+)"', m.group(1))) if m else set()
        mine = set(SUBCOMMANDS)
        drift = (mine - fwd) | (fwd - mine)
        print(f"[{'ok ' if not drift else 'FAIL'}] reachability: layout_facts.py forwards "
              f"{len(fwd)}/{len(mine)} subcommand(s)"
              + (f"; DRIFT {sorted(drift)}" if drift else ""))
        fail += bool(drift)

    fdb = load_facts_db(required=False)
    if fdb is None:
        print("[FAIL] facts DB (ISA ch.1-6, 8-11): not found next to the script")
        fail += 1
    else:
        keys = ("registers", "resources", "counters", "wait_states", "memory_ops",
                "lds_facts", "tables", "concepts")
        recs = [(k, r) for k in keys for r in fdb.get(k) or []]
        # Every record must be traceable back to a page. This is the property the whole extraction
        # was built around -- if it does not hold, no answer from this DB can be cited.
        unsourced = [r.get("id") for k, r in recs
                     if not (r.get("_src") or {}).get("quote")
                     or not (r.get("_src") or {}).get("pdf_page") or not r.get("_conf")]
        # printed_page = pdf_page - 8 holds throughout this document; a violation means a page
        # number was typed rather than read.
        offset = [r.get("id") for k, r in recs
                  if (r.get("_src") or {}).get("printed_page") is not None
                  and r["_src"]["printed_page"] != r["_src"]["pdf_page"] - 8]
        ids = [r.get("id") for k, r in recs if r.get("id")]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        print(f"[{'ok ' if not unsourced else 'FAIL'}] facts DB: {len(recs)} records across "
              f"{len(keys)} arrays; {len(unsourced)} without a page+quote+conf")
        print(f"[{'ok ' if not offset else 'FAIL'}] facts DB page offset: "
              f"{len(offset)} record(s) where printed_page != pdf_page - 8")
        print(f"[{'ok ' if not dupes else 'FAIL'}] facts DB ids: {len(dupes)} duplicate id(s)")
        fail += bool(unsourced) + bool(offset) + bool(dupes)
        for d in (unsourced + offset + dupes)[:5]:
            print(f"       {d}")
        n_unc = sum(1 for k, r in recs if r.get("_conf") == "uncertain")
        if n_unc:
            print(f"       note: {n_unc} record(s) are _conf=uncertain by design -- the PDF did "
                  f"not settle them. Not a failure; re-read the page before acting on one.")

    print()
    print(f"verification record carried by the DB:")
    for k, t in v["tests"].items():
        print(f"  {k:<24} {t}")
    print()
    print("array verdicts:")
    for k, t in v["array_verdicts"].items():
        print(f"  {k:<16} {t}")
    print()
    print(f"!! {v['standing_caveat']}")
    return 1 if fail else 0


# Populated in main() from the parser itself, so the reachability check in --selftest compares
# against the real subparser list rather than a second hand-maintained copy.
SUBCOMMANDS = set()


def main(argv):
    global SUBCOMMANDS
    argv = [argv[0]] + _select_arch(argv[1:])
    db = load_db(required=False)
    have = [k for k, v in _ARCH_DBS[ARCH].items() if v]
    p = argparse.ArgumentParser(
        prog="gfx950_isa.py",
        description="AMD ISA reference, offline. Default arch gfx950/CDNA4 (all 13 chapters, "
                    "three databases). `--arch cdna3|rdna3|rdna4` answers the instruction "
                    "chapters (12-13) for those documents; their matrix layout comes from the "
                    "Matrix Instruction Calculator instead, which supports them and not gfx950.",
        epilog="arch=%s, databases present: %s.  Every arch's coverage and the named alternative "
               "for a missing one: `--arch <a> <cmd>` tells you in place. See "
               "perf_knowledge/expert_skills/skills/gluon_authoring/references/hardware/primary-sources.md." % (ARCH, ", ".join(have) or "none"))
    p.add_argument("--arch", default=ARCH, metavar="A",
                   help="gfx950 (default) | cdna3 | rdna3 | rdna4; aliases cdna4/gfx942/mi300/"
                        "mi325/gfx1100/gfx1200")
    p.add_argument("--selftest", action="store_true",
                   help="offline: re-derive the transcribed grids from the rules and exit")
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("list", help="list matrix instructions")
    s.add_argument("substr", nargs="?")
    s.set_defaults(fn=cmd_list)

    s = sub.add_parser("facts", help="dims/cycles/dtypes/rules + trust verdict")
    s.add_argument("instr")
    s.set_defaults(fn=cmd_facts)

    s = sub.add_parser("layout", help="full lane/VGPR/bit table for one operand")
    s.add_argument("instr")
    g = s.add_mutually_exclusive_group()
    for o in "ABCD":
        g.add_argument(f"-{o}", dest="operand", action="store_const", const=o)
    s.add_argument("--as", dest="as_fmt",
                   help="element format for F8F6F4 opcodes: fp8|fp6|fp4")
    s.add_argument("--lanes", help="restrict to a lane range, e.g. 0-15")
    s.add_argument("--block", type=int, default=0)
    s.add_argument("--markdown", action="store_true")
    s.set_defaults(fn=cmd_layout, operand=None)

    s = sub.add_parser("locate", help="where one element lives")
    s.add_argument("instr")
    s.add_argument("operand")
    s.add_argument("i", type=int)
    s.add_argument("second", type=int, metavar="k|j")
    s.add_argument("--as", dest="as_fmt")
    s.add_argument("--block", type=int, default=0)
    s.set_defaults(fn=cmd_locate)

    s = sub.add_parser("encoding", help="opcode + operand fields for ANY gfx950 instruction")
    s.add_argument("instr")
    s.set_defaults(fn=cmd_encoding)

    s = sub.add_parser("format", help="bit-field layout of an encoding (ISA ch.13)")
    s.add_argument("encoding", help="e.g. ENC_DS, VOP3P_MFMA, ENC_FLAT_GLBL")
    s.set_defaults(fn=cmd_format)

    s = sub.add_parser("opcodes", help="search all 1240 gfx950 instructions (ISA ch.12)")
    s.add_argument("substr", nargs="?", help="matches name, functional group or encoding")
    s.set_defaults(fn=cmd_opcodes)

    s = sub.add_parser("search", help="search all three gfx950 DBs at once (start here)")
    s.add_argument("substr")
    s.set_defaults(fn=cmd_search)

    for name, fn, helptext in (("modifiers", cmd_modifiers, "CBSZ/ABID/BLGP/scale semantics"),
                               ("sparse", cmd_sparse, "4:2 sparsity index + operand tables"),
                               ("hazard", cmd_hazard, "MFMA hazards: independent instructions / NOPs"),
                               ("errata", cmd_errata, "known errors in the source document"),
                               # ISA ch.1-6, 8-11 -- the prose/table half
                               ("regs", cmd_regs, "MODE/STATUS/TRAPSTS/HW_ID/M0 bit fields (ch.3)"),
                               ("limits", cmd_limits, "VGPR/SGPR/LDS capacities + granularity"),
                               ("waitcnt", cmd_waitcnt, "what VM_CNT/LGKM_CNT/EXP_CNT count"),
                               ("waits", cmd_waits, "required software-inserted wait states (ch.4)"),
                               ("mem", cmd_mem, "addressing formulas, alignment, bounds (ch.8-10)"),
                               ("lds", cmd_lds, "LDS banks, conflicts, transpose loads (ch.11)"),
                               ("tables", cmd_tables, "verbatim source tables"),
                               ("concepts", cmd_concepts, "actionable prose facts with a quote")):
        s = sub.add_parser(name, help=helptext)
        s.add_argument("substr", nargs="?")
        s.set_defaults(fn=fn)

    SUBCOMMANDS = set(sub.choices)

    args = p.parse_args(argv[1:])
    if args.selftest:
        if DB_REL is None:
            gap("--selftest re-derives the matrix grids from the layout rules, and %s ships no "
                "layout database. Run it without --arch (gfx950) to exercise that; the encoding "
                "database's own checks are recorded in its `verification` block." % ARCH)
        return cmd_selftest(db, args)
    if not getattr(args, "fn", None):
        p.print_help()
        return 2
    _require(args.cmd)
    if args.cmd == "layout" and args.operand is None:
        args.operand = "A"
    return args.fn(db, args)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
