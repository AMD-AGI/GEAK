#!/usr/bin/env python3
"""Offline AMD ISA database provider; use amd_isa.py --arch for public queries.

Reads committed encoding, matrix, and prose facts without SDK/DSL imports or
seed/cache XML overrides. Layouts are instruction-relative hardware facts,
not compiler layout objects or allocated registers. Source citations and
verification limits travel with answers; sparse A/B use source tables only.
Full source/grid regression and evidence live in scripts/tests; --selftest
checks runtime data integrity. See isa_tools.md for coverage and maintenance.
"""
import argparse
import json
import re
import sys
from math import ceil

# Database FILE NAMES. They live once, in perf_knowledge/hardware/data/, and resolve through the
# sibling `_hwdata` locator ($GEAK_HW_DATA_DIR first) -- see `_db_path`.
DB_REL = "gfx950-mfma-layout.json"
ENC_DB_REL = "gfx950-encoding.json.gz"
FACTS_DB_REL = "gfx950-isa-facts.json.gz"
DATA_HOME = "perf_knowledge/hardware/data"


# Which databases exist per architecture, and therefore which subcommands can answer.
#
# gfx950 needs all three because the Matrix Instruction Calculator has no CDNA4 support, so
# they provide a queryable source for CDNA4 hardware facts. The other arches
# are not in that position for layout: the calculator generates their chapter-7 tables exactly,
# on demand, from the hardware rules -- so building a layout DB for them would be transcribing
# something a tool already derives. What the calculator does NOT answer is chapters 12-13, and
# THAT is what these arches were missing: an opcode, an operand field, a pseudo-code body.
#
# A `None` here is a deliberate, documented absence with a named alternative -- never a hint
# that the hardware lacks the feature. `_require` below prints the alternative.
from amd_targets import ALIASES, TARGETS, database_path as _db_path

_ARCH_DBS = {key: {kind: row[kind] for kind in ('layout', 'encoding', 'facts')}
             for key, row in TARGETS.items() if row['encoding']}
_ARCH_ALIAS = {key: value for key, value in ALIASES.items() if value in _ARCH_DBS}
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
               "kernel_workflow/scripts/kernel_tools/isa_tools.md). No layout DB is shipped for {arch} "
               "because that tool already derives it."),
    "facts": ("ISA chapters 1-6 and 8-11 are not extracted for {arch} yet -- this is missing "
              "KNOWLEDGE, not an absent hardware feature. Read the page: "
              "`kernel_workflow/scripts/kernel_tools/hw_sources.sh pdf {arch}-isa <p0> <p1>`, and record what you find in "
              "your analysis record with its page."),
    "encoding": ("no encoding DB is shipped for {arch} -- fetch the machine-readable ISA XML "
                 "instead: `kernel_workflow/scripts/kernel_tools/hw_sources.sh xml {arch}`."),
}


def _select_arch(argv):
    """Consume a leading/anywhere `--arch <a>` and point the DB constants at it.

    Done before argparse because `main` loads a database to build the parser. Default stays
    gfx950 so every existing invocation, doc reference and catalog entry keeps working.
    """
    global ARCH, DB_REL, ENC_DB_REL, FACTS_DB_REL
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
            gap("unknown --arch {!r}; known: {} (aliases: {})".format(want, ", ".join(sorted(_ARCH_DBS)), ", ".join(sorted(_ARCH_ALIAS))))
        ARCH = key
        d = _ARCH_DBS[key]
        DB_REL, ENC_DB_REL, FACTS_DB_REL = d["layout"], d["encoding"], d["facts"]
    return out


def _require(cmd):
    """Fail with the alternative, not with a bare absence, when this arch lacks cmd's database."""
    kind = _CMD_DB.get(cmd)
    if kind is None:
        return
    if _ARCH_DBS[ARCH][kind] is None:
        gap(f"`{cmd}` reads the {kind} database, which is not shipped for {ARCH}.  -> {_ALTERNATIVE[kind].format(arch=ARCH)}")
ARCHES = {"cdna4", "gfx950"}


def gap(msg):
    """Scoped tool-gap, per isa_tools.md: record it and fall back; never fabricate."""
    print(f"SCOPED TOOL-GAP: {msg}", file=sys.stderr)
    print("  -> fall back to distilled perf_knowledge/hardware/ with explicit source and uncertainty; "
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
    """Fixed XML-derived encodings plus PDF semantics; no seed/cache XML override."""
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
    """Look up an instruction in the selected committed encoding database."""
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
            gap(f"no layout database for {ARCH}; `find_instr` has nothing to resolve against")
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
        f"dense closed form. No generated sparse operand mapping is available. Source evidence: "
        f"gfx950_isa.py sparse {rule} (A index tables describe metadata, not a complete A layout; verdict: "
        f"{db['verification']['array_verdicts']['sparse']})")


def caveats_for(db, rec):
    """The trust envelope that must travel with the answer."""
    v = db["verification"]
    out = []
    new_surface = (rec.get("sparse") or rec.get("scale")
                   or (rec.get("in_type_a") or "") in ("f8f6f4", "fp6", "bf6", "fp4")
                   or (rec.get("k") or 0) >= 64)
    if new_surface:
        out.append("CDNA4-NEW SURFACE: " + v["standing_caveat"])
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
    as_fmt = getattr(args, "as_fmt", None)
    variable_format = r.get("in_type_a") == "f8f6f4" or r.get("in_type_b") == "f8f6f4"
    if as_fmt and not variable_format:
        gap("facts --as applies only to format-selectable F8F6F4 instructions")
    # Reuse the actual operand table, rather than maintaining another packing formula.
    format_vg = {}
    if as_fmt:
        for operand in ("A", "B"):
            guard_dense(db, r, operand)
            eb = elem_bits_for(r, as_fmt, operand)
            cells = {}
            for block in range(r.get("blocks") or 1):
                cells.update(_table(db, r, operand, eb, block, set(range(64))))
            format_vg[operand] = max(fragment[0] for fragment in cells) + 1
    src = r.get("_src") or {}
    print(f"## {r['name']}  ({arch_label()})")
    print(f"  dims        M={r['m']} N={r['n']} K={r['k']} blocks={r.get('blocks')}"
          f"   cycles={r.get('cycles')}")
    print(f"  dtype       A={r.get('in_type_a')}  B={r.get('in_type_b')}  "
          f"D={r.get('out_type')}  integer={r.get('integer')}")
    vg = {"A": r.get("src_a_vgprs"), "B": r.get("src_b_vgprs"), "C/D": r.get("src_c_d_vgprs")}
    xml_vg = _enc_vgprs(_enc_instr(r["name"])) if any(v is None for v in vg.values()) else {}
    parts = []
    for k in ("A", "B", "C/D"):
        if k in format_vg:
            parts.append(f"{k}={format_vg[k]}")
        elif variable_format and k in ("A", "B"):
            parts.append(f"{k}=?")
        elif vg[k] is not None:
            parts.append(f"{k}={vg[k]}")
        elif xml_vg.get(k) is not None:
            parts.append(f"{k}={xml_vg[k]}*")
        else:
            parts.append(f"{k}=?")
    print(f"  vgprs       {' '.join(parts)}"
          + ("    (* XML OperandSize declaration, not compiler allocation)" if "*" in " ".join(parts) else ""))
    if variable_format:
        print("  xml_operand_vgprs " + " ".join(f"{k}={xml_vg.get(k, '?')}" for k in ("A", "B"))
              + " (encoding declaration; not format-specific usage)")
        if as_fmt:
            print(f"  format      A=B={as_fmt}; operand register span derived from layout/packing rules, not compiled allocation")
        else:
            print("  format      unspecified; use facts --as fp4|fp6|fp8 for equal A/B formats, or layout/locate separately for mixed formats")
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
    print("# provenance: gfx950-mfma-layout.json layout_rules + packing_rules "
          "(CDNA4 ISA ch.7); generated, not transcribed — the DB's figures are the "
          "transcription and --selftest checks this against them", file=sys.stderr)
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


def _enc_vgprs(rec):
    """Declared widths; SMFMAC SRC2 is metadata, not the C/D accumulator."""
    if not rec:
        return {}
    by_field = {}
    for enc in rec.get("encodings", []):
        for o in enc.get("operands", []):
            if o.get("size_bits"):
                by_field[(o.get("field") or "").upper()] = o["size_bits"] // 32
    return {"A": by_field.get("SRC0"), "B": by_field.get("SRC1"),
            "C/D": by_field.get("VDST") or (None if "SMFMAC" in rec["name"].upper() else by_field.get("SRC2"))}


def cmd_encoding(db, args):
    """Query the selected committed encoding/semantics record, independently of installed SDKs."""
    edb = load_enc_db()
    name = args.instr
    for rec in (find_instr(db, name, required=False),):
        if rec:
            name = rec["name"]

    erec = _enc_instr(name)
    if erec is None:
        gap(f"no {arch_label()} instruction named {args.instr!r} in the committed encoding DB")
    name, desc, group = erec["name"], erec.get("description") or "", erec.get("functional_group")
    encs = [(e.get("encoding"), e.get("opcode"), e.get("condition"),
             [(o.get("field"), o.get("format"), o.get("type"), o.get("size_bits"),
               o.get("input"), o.get("output")) for o in e.get("operands", [])])
            for e in erec.get("encodings", [])]
    origin = f"committed encoding DB ({edb['source']['release']})"

    print(f"## {name}  ({arch_label()}) — encoding")
    if desc:
        print(f"  description {desc}")
    if group:
        print(f"  group       {group}")
    for ename, opcode, cond, ops in encs:
        print(f"  encoding    {ename}   opcode={opcode}   cond={cond}")
        for field, fmt, otype, size, is_in, is_out in ops:
            print(f"    {field!s:<8} {fmt!s:<20} {otype!s:<18} size={size} "
                  f"{'in' if is_in else ''}{'/out' if is_out else ''}")
        print(f"                -> bit fields of this encoding: gfx950_isa.py format {ename}")

    # Encoding and PDF semantics share the same versioned record.
    sem = erec
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
        print(f"  {name:<34} {enc:<18} op={op!s:<5} {grp:<8} {desc}")
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
    pool = ((fdb.get("tables") or []) + ((edb or {}).get("tables") or [])
            + [r for r in (db or {}).get("layout_rules", []) if r.get("verbatim_table")])
    rows = [r for r in pool
            if not sub or sub in json.dumps(r, ensure_ascii=False).lower()]
    if not rows:
        gap(f"no table matches {args.substr!r} among {len(pool)} transcribed tables")
    for r in rows:
        src = r.get("_src") or {}
        if r.get("verbatim_table"):
            print(f"## {r['id']} (source table excerpt; not a generated Gluon layout)")
            for field in ("applies_to", "params", "placement", "verbatim_header", "verbatim_table", "notes"):
                if r.get(field) is not None:
                    print(f"  {field}: {json.dumps(r[field], ensure_ascii=False)}")
            print(f"  # provenance: CDNA4 ISA p{src.get('printed_page')} (pdf p{src.get('pdf_page')}); conf={r.get('_conf')}\n")
            continue
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
# Runtime integrity. Source/grid regressions and their evidence live in tests.
# ---------------------------------------------------------------------------------------


def cmd_selftest(db, args):
    """Check runtime data integrity; full source/grid regression lives in tests."""
    if db is None:
        gap("matrix database missing")
    rules = {r["id"] for k in ("layout_rules", "sparse", "packing_rules") for r in db[k]}
    bad = [r["name"] for r in db["instructions"]
           for key in ("layout_rule_a", "layout_rule_b", "layout_rule_d",
                       "packing_rule_a", "packing_rule_b", "packing_rule_d")
           if r.get(key) and r[key] not in rules]
    for key in ("instructions", "layout_rules", "packing_rules", "modifiers", "formats", "sparse", "hazards"):
        bad.extend(r.get("id") or r.get("name") for r in db[key] if not r.get("_src") or not r.get("_conf"))
    load_enc_db()
    load_facts_db()
    from amd_targets import COMMANDS
    bad.extend(sorted(SUBCOMMANDS ^ COMMANDS))
    print(f"[{'FAIL' if bad else 'ok '}] runtime data integrity: {len(bad)} error(s)")
    for error in bad[:5]:
        print(f"  {error}")
    print("Source/grid regression: kernel_workflow/scripts/tests/test_kernel_tools_isa.py")
    print(db["verification"]["standing_caveat"])
    return int(bool(bad))


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
        epilog="arch={}, databases present: {}.  Every arch's coverage and the named alternative "
               "for a missing one: `--arch <a> <cmd>` tells you in place. See "
               "kernel_workflow/scripts/kernel_tools/isa_tools.md.".format(ARCH, ", ".join(have) or "none"))
    p.add_argument("--arch", default=ARCH, metavar="A",
                   help="gfx950 (default) | cdna3 | rdna3 | rdna4; aliases cdna4/gfx942/mi300/"
                        "mi325/gfx1100/gfx1200")
    p.add_argument("--selftest", action="store_true",
                   help="offline runtime data integrity; full source/grid checks are in the test suite")
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("list", help="list matrix instructions")
    s.add_argument("substr", nargs="?")
    s.set_defaults(fn=cmd_list)

    s = sub.add_parser("facts", help="dims/cycles/dtypes/rules + trust verdict")
    s.add_argument("instr")
    s.add_argument("--as", dest="as_fmt", choices=["fp4", "fp6", "fp8"],
                   help="equal A/B formats for F8F6F4 operand register spans; mixed formats use layout/locate")
    s.set_defaults(fn=cmd_facts)

    s = sub.add_parser("layout", help="full lane/VGPR/bit table for one operand")
    s.add_argument("instr")
    g = s.add_mutually_exclusive_group()
    for o in "ABCD":
        g.add_argument(f"-{o}", dest="operand", action="store_const", const=o)
    g.add_argument("--operand", choices=list("ABCD"))
    s.add_argument("--wave-size", choices=["64"], help="gfx950 layouts use wave64")
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
    s.add_argument("--wave-size", choices=["64"], help="gfx950 layouts use wave64")
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
            gap(f"--selftest checks gfx950 runtime data; {ARCH} ships no matrix database")
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
