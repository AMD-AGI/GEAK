#!/usr/bin/env python3
"""layout_facts.py — one-key MFMA/WMMA layout pre-flight (Tier-2 generate, not trial-and-error).

The FIRST thing a subagent runs before touching any matrix-core / low-precision / transpose lever:
instead of "compile a tiny kernel and see", GENERATE the authoritative operand layout + dims +
cycle/GPR/dtype facts from the AMD Matrix Instruction Calculator (Tier 2 of
perf_knowledge/expert_skills/skills/gluon_authoring/references/hardware/primary-sources.md). Deterministic, no GPU, no compile round.

What it answers up front (the questions the old `probe` fields spent a compile round on):
  - matrix dims M/N/K, execution cycles, FLOPs/CU/cycle, VALU co-exec
  - GPRs for A/B/C/D + alignment  -> register-pressure budget
  - operand dtypes (Src0/Src1)    -> is this the fp8/bf8/fp4 path you think it is
  - **elements per lane** for A and B (derived) -> the ds_read_tr granularity fit-probe:
        a ds_read_trN delivers N elems/lane; if operand needs MORE, the transpose-read
        lever is DISQUALIFIED (it raises ds_read count, not lowers it).
  - A/B register layout table (which v{lane}.[bits] holds A[m][k]) via --markdown

Arch handling (matches hw_sources.sh / primary-sources.md tiers):
  - cdna1/2/3, rdna3/4  -> Tier 2, generated here.
  - cdna4 / gfx950      -> the calculator still has no CDNA4 support, but this is no longer a
                            gap: delegated to gfx950_isa.py, which answers the same questions
                            from the extracted+verified CDNA4 ISA chapter 7 layout database
                            (perf_knowledge/hardware/data/gfx950-mfma-layout.json). Run that tool
                            directly for locate / encoding / sparse / hazard / errata.
Never fabricates: on any tool failure it prints a SCOPED TOOL-GAP and exits 3.

Usage:
  layout_facts.py --help                            # this message
  layout_facts.py --selftest                        # offline dispatch check (no calculator)
  layout_facts.py <arch> <instr> [<instr> ...]      # facts + derived elems/lane for each instr
  layout_facts.py <arch> <instr> --layout           # also dump the A/B register-layout markdown table
  layout_facts.py <arch> --list [substr]            # list matrix instrs (optionally filtered)
  layout_facts.py <arch> <subcommand> [args]        # instruction-chapter questions, from the
                                                    #   in-pack DB: encoding | format | opcodes |
                                                    #   search | errata

  arch: cdna1|cdna2|cdna3|rdna3|rdna4 (layout via the Matrix Calculator)
      | gfx950/cdna4                   (layout + everything else from the in-pack databases)

  WHICH TOOL ANSWERS WHAT. Layout and the instruction chapters come from different sources, and
  this script routes between them so a caller does not have to know which:
    layout / facts / locate / --list  -> Matrix Calculator (it has no cdna4 support; gfx950 uses
                                         its in-pack layout DB instead)
    encoding / format / opcodes /     -> the in-pack encoding DB for gfx950, cdna3, rdna3, rdna4
      search / errata                    (offline; no calculator, no network, no cache)
  Run `hw_sources.sh preflight` to see that matrix for THIS environment before relying on it.
Provenance: each fact block ends with a `# provenance: MI-calc <arch> <instr>` line — copy the
distilled fact into perf_knowledge/expert_skills/skills/gluon_authoring/references/hardware/isa-mechanisms.md WITH that citation so the next agent
trusts it and never re-runs this (primary-sources.md provenance rule).
"""
import os, re, subprocess, sys

CACHE = os.environ.get("TILE_HW_CACHE", os.path.expanduser("~/.cache/tile-hw"))
TOOLS = os.environ.get("TILE_HW_TOOLS", "")
MI_CALC_URL = "https://github.com/ROCm/amd_matrix_instruction_calculator"
CDNA4 = {"cdna4", "gfx950"}
# Forwarded verbatim to gfx950_isa.py. gfx950_isa.py --selftest asserts this set equals its own
# subparser list, so adding a subcommand there without adding it here fails the gate.
GFX950_SUBCOMMANDS = {
    "list", "facts", "layout", "locate", "encoding", "format", "opcodes", "search", "modifiers",
    "sparse", "hazard", "errata", "regs", "limits", "waitcnt", "waits", "mem", "lds", "tables",
    "concepts",
}
# Arches that ship an ENCODING database (ISA ch.12-13) but no layout database, because the
# calculator already generates their layout. So the two halves of a question about these arches
# are answered by two different tools, and this script is the one place that knows which:
#   layout / facts / locate / modifiers / sparse   -> the calculator, below
#   encoding / format / opcodes / search / errata  -> the committed DB, via gfx950_isa.py --arch
# Without this split an agent that reached here for an opcode got a layout table and no opcode.
ENC_DB_ARCHES = {"cdna3": "cdna3", "gfx942": "cdna3", "mi300": "cdna3", "mi300x": "cdna3",
                 "mi325": "cdna3", "mi325x": "cdna3",
                 "rdna3": "rdna3", "gfx1100": "rdna3",
                 "rdna4": "rdna4", "gfx1200": "rdna4"}
ENC_SUBCOMMANDS = {"encoding", "format", "opcodes", "search", "errata"}
# byte width per calculator dtype token -> elems/lane = (GPRs*4 bytes) / bytes_per_elem
_DTYPE_BYTES = {"FP8": 1, "BF8": 1, "FP4": 0.5, "FP6": 0.75, "I8": 1, "INT8": 1,
                "FP16": 2, "BF16": 2, "F16": 2, "XF32": 4, "FP32": 4, "F32": 4, "FP64": 8, "F64": 8}


def _gap(msg):
    print(f"SCOPED TOOL-GAP: {msg}", file=sys.stderr)
    print("  -> fall back to distilled perf_knowledge/hardware/ + gluon_authoring/references/hardware/ values (probe-then-trust); never fabricate",
          file=sys.stderr)
    sys.exit(3)


def _calc_path():
    for d in ([os.path.join(TOOLS, "amd_matrix_instruction_calculator")] if TOOLS else []) + \
             [os.path.join(CACHE, "amd_matrix_instruction_calculator")]:
        if os.path.isfile(os.path.join(d, "matrix_calculator.py")):
            return os.path.join(d, "matrix_calculator.py")
    d = os.path.join(CACHE, "amd_matrix_instruction_calculator")
    os.makedirs(CACHE, exist_ok=True)
    if subprocess.call(["git", "clone", "--depth", "1", MI_CALC_URL, d],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
        _gap(f"Matrix Calculator absent (run 'make init', or set TILE_HW_TOOLS to third_party/)")
    return os.path.join(d, "matrix_calculator.py")


def _run(calc, *args):
    try:
        r = subprocess.run(["python3", calc, *args], capture_output=True, text=True, timeout=60)
    except Exception as e:
        _gap(f"calculator invocation failed: {e}")
    return r.returncode, r.stdout, r.stderr


def _detail(calc, arch, instr):
    rc, out, err = _run(calc, "-a", arch, "-i", instr, "-d")
    if rc != 0 or "Matrix Dimensions" not in out:
        _gap(f"calculator has no '{instr}' for {arch} (list: layout_facts.py {arch} --list)")
    return out


def _grab(text, label):
    m = re.search(rf"^\s*{re.escape(label)}:\s*(.+?)\s*$", text, re.M)
    return m.group(1).strip() if m else None


def _elems_per_lane(gprs, dtype_token):
    """GPRs*4 bytes/lane / bytes-per-elem. dtype_token like 'FP8 (AMD 4-bit...)'."""
    if gprs is None or dtype_token is None:
        return None
    key = dtype_token.split()[0].upper()
    bpe = _DTYPE_BYTES.get(key)
    if not bpe:
        return None
    return int(int(gprs) * 4 / bpe)


def facts(calc, arch, instr, want_layout):
    d = _detail(calc, arch, instr)
    M, N, K = _grab(d, "M"), _grab(d, "N"), _grab(d, "K")
    cyc = _grab(d, "Execution cycles")
    ga, gb = _grab(d, "GPRs required for A"), _grab(d, "GPRs required for B")
    gc, gd = _grab(d, "GPRs required for C"), _grab(d, "GPRs required for D")
    align = _grab(d, "GPR alignment requirement")
    coexec = _grab(d, "Can co-execute with VALU")
    src0, src1 = _grab(d, "Src0"), _grab(d, "Src1")
    epl_a = _elems_per_lane(ga, src0)
    epl_b = _elems_per_lane(gb, src1)

    print(f"## {instr.upper()}  ({arch})")
    print(f"  dims        M={M} N={N} K={K}   exec_cycles={cyc}   co-exec VALU={coexec}")
    print(f"  gpr/lane    A={ga} B={gb} C={gc} D={gd}   align={align}")
    print(f"  dtype       A(Src0)={src0}")
    print(f"              B(Src1)={src1}")
    if epl_a is not None:
        print(f"  elems/lane  A={epl_a}  B={epl_b}   "
              f"<- ds_read_tr fit: a ds_read_trN gives N/lane; need {epl_a} for A "
              f"(if trN < {epl_a}, transpose-read RAISES ds_read count -> disqualified)")
    if want_layout:
        rc, lay, err = _run(calc, "-a", arch, "-i", instr, "-R", "-A", "--markdown")
        if rc == 0:
            print("  --- A register layout (v{lane}.[bits] holding A[m][k]) ---")
            print("\n".join("  " + ln for ln in lay.splitlines() if "|" in ln or ln.startswith("Block")))
    print(f"  # provenance: MI-calc {arch} {instr} -> distill into "
          f"perf_knowledge/expert_skills/skills/gluon_authoring/references/hardware/isa-mechanisms.md with this citation")
    print()


def _selftest():
    """Offline: the dispatch that does not need the Matrix Calculator (no clone, no GPU)."""
    fails = []

    def chk(cond, msg):
        if not cond:
            fails.append(msg)

    chk(_elems_per_lane(4, "BF16 (brain float)") == 8, "4 GPRs of bf16 -> 8 elems/lane")
    chk(_elems_per_lane(4, "FP8 (AMD...)") == 16, "4 GPRs of fp8 -> 16 elems/lane")
    chk(_elems_per_lane(None, "FP8") is None and _elems_per_lane(4, "WAT") is None,
        "unknown gprs/dtype -> None, never a guess")
    g = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gfx950_isa.py")
    chk(os.path.isfile(g), "gfx950_isa.py must be a sibling (kernel_tools/)")
    with open(os.devnull, "w") as dn:
        rc = subprocess.call([sys.executable, os.path.abspath(__file__), "gfx950", "facts",
                              "v_mfma_f32_32x32x16_bf16"], stdout=dn, stderr=dn)
        chk(rc == 0, f"gfx950 facts must forward to gfx950_isa.py and answer offline (rc={rc})")
        rc = subprocess.call([sys.executable, os.path.abspath(__file__), "gfx942", "encoding",
                              "v_mfma_f32_32x32x8_f16"], stdout=dn, stderr=dn)
        chk(rc == 0, f"gfx942 encoding must route to the committed cdna3 DB (rc={rc})")
        # No arch is not a default: usage, non-zero.
        rc = subprocess.call([sys.executable, os.path.abspath(__file__)], stdout=dn, stderr=dn)
        chk(rc == 2, f"no arch must refuse with rc 2, got {rc}")
    for f in fails:
        print("FAIL", f)
    print("[layout_facts] SELFTEST " + ("FAIL" if fails else "PASS"))
    return 1 if fails else 0


def main(argv):
    if len(argv) > 1 and argv[1] == "--selftest":
        return _selftest()
    # `--help` used to fall through to the arch slot and come back as a tool-gap about a missing
    # instruction, which is why this tool's help contract read as unverified: an agent asking the
    # tool what it does was told it had used it wrong.
    if len(argv) < 2 or argv[1] in ("--help", "-h", "help"):
        print(__doc__)
        return 0 if len(argv) > 1 else 2
    arch = argv[1].lower()
    rest = argv[2:]

    if arch in CDNA4:
        # The Matrix Calculator still has no CDNA4 support, but this is no longer a gap: the
        # ISA chapter 7 layout tables are extracted and committed, and gfx950_isa.py serves the
        # same questions from them. Hand the whole invocation over.
        gfx950 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gfx950_isa.py")
        if not os.path.isfile(gfx950):
            _gap("Matrix Calculator has NO CDNA4/gfx950 layout support, and gfx950_isa.py is "
                 "missing from this pack -> Tier-3 PDF: kernel_workflow/scripts/kernel_tools/hw_sources.sh pdf cdna4-isa "
                 "<p0> <p1>  (render + read the layout table by eye)")
        if rest and rest[0] in ("--list", "-L"):
            sub = [rest[1]] if len(rest) > 1 else []
            return subprocess.call([sys.executable, gfx950, "list"] + sub)
        # The gfx950 databases answer far more than layout: encodings and chapter-12 pseudo-code,
        # errata, registers, wait states, LDS and memory rules. Agents arrive here from the skill
        # docs knowing only `layout_facts.py`, so forward any gfx950_isa.py subcommand straight
        # through rather than stranding them at facts/layout and leaving the rest undiscoverable.
        if rest and rest[0] in GFX950_SUBCOMMANDS:
            return subprocess.call([sys.executable, gfx950] + rest)
        instrs = [a for a in rest if not a.startswith("-")]
        if not instrs:
            _gap("no instruction given. Usage: layout_facts.py gfx950 <instr> [--layout], or pass "
                 "any gfx950_isa.py subcommand through: " + " ".join(sorted(GFX950_SUBCOMMANDS)) +
                 ".  Start with: layout_facts.py gfx950 search <term>")
        rc = 0
        for instr in instrs:
            rc |= subprocess.call([sys.executable, gfx950, "facts", instr])
            if "--layout" in rest:
                rc |= subprocess.call([sys.executable, gfx950, "layout", instr, "-A"])
        return rc

    # An instruction-chapter question about a calculator-supported arch: the calculator cannot
    # answer it (it models matrix layout, not encodings), but the committed DB can.
    if arch in ENC_DB_ARCHES and rest and rest[0] in ENC_SUBCOMMANDS:
        g = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gfx950_isa.py")
        if os.path.isfile(g):
            return subprocess.call([sys.executable, g, "--arch", ENC_DB_ARCHES[arch]] + rest)
        _gap(f"`{rest[0]}` needs the {arch} encoding database via gfx950_isa.py, which is missing "
             f"from this pack -> Tier-1 XML: kernel_workflow/scripts/kernel_tools/hw_sources.sh xml {arch}")

    calc = _calc_path()

    if rest and rest[0] in ("--list", "-L"):
        rc, out, err = _run(calc, "-a", arch, "-L")
        if rc != 0:
            _gap(f"cannot list instructions for {arch}")
        sub = rest[1].lower() if len(rest) > 1 else None
        for ln in out.splitlines():
            if sub is None or sub in ln.lower():
                print(ln)
        return 0

    want_layout = "--layout" in rest
    instrs = [a for a in rest if not a.startswith("-")]
    if not instrs:
        _gap("no instruction given (usage: layout_facts.py <arch> <instr> [--layout])")
    for instr in instrs:
        facts(calc, arch, instr, want_layout)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
