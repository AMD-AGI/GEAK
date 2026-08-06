#!/usr/bin/env python3
"""Triton-family hot-loop schedule auditor (gfx950 / gfx942 / RDNA3-4 AMDGCN).

Mechanical part of the asm-schedule audit
(references/tile-programming/compiler-contract.md ## Auditing the hot-loop schedule).
Consumes a stripped ``.s`` (from ``scripts/dump_ir.sh``; ``.amdgcn`` also works) and
reports SIGNALS ONLY -- the structural-vs-schedulable VERDICT stays with the agent:

  (1) the hot loop's op-class symbol stream + per-class histogram;
  (2) every memory wait classified relaxed(``cnt>0``)=pipelined vs full-drain(``cnt==0``)
      =serialized, with counts, in BOTH spellings: gfx9 ``s_waitcnt vmcnt(0)`` and
      gfx11/12 ``s_wait_dscnt 0x1`` (counter in the mnemonic, value as an immediate).
      ``s_wait_alu depctr_*`` / ``s_delay_alu`` are ALU-dependency waits, not memory
      drains -- they are counted on their own line and kept OUT of the drain ratio;
  (3) producer<->consumer sync barriers per iteration;
  (4) ``s_nop`` count + requested stall cycles = the exposed FIXED-latency hazard
      (e.g. MFMA/WMMA-write -> VALU-read), hidden by unroll/occupancy, never by reorder.

Matrix engines: CDNA MFMA (``v_mfma`` / ``v_smfmac``) and RDNA WMMA (``v_wmma``) are
both classified as "mfma" — the audit treats any matrix-multiply instruction the same
way. RDNA3/4 (gfx11*/gfx120*) use ``v_wmma_*`` exclusively; CDNA (gfx942/950) uses MFMA.

It does NOT decide if an exposed single-class run is structural (no slack) or a real
scheduling miss -- that judgment needs the dependency context the agent has. Pure
text parsing, no GPU, stdlib only.

Input forms (both accepted):
  (a) assembler ``.s`` / ``.amdgcn`` from ``scripts/dump_ir.sh`` — labels are ``name:``.
      This is the normal path when you control the kernel source (your own Triton/Gluon
      kernel): one tight hot loop, auto-detect works.
  (b) ``llvm-objdump -d`` of a prebuilt ``.hsaco`` / ``.co`` — labels are
      ``<hexaddr> <name>:``. Use this for a **prebuilt comparator** (e.g. a vendor asm
      kernel) where no ``.amdgcn`` source exists. Such a dump is the WHOLE function with
      nested back-edges, so auto-detect may pick an *outer* loop; pass ``--loop-label``
      with the inner MFMA loop's label to pin it.

Usage:
  python3 asm_loop_audit.py <file.s> [--loop-label .LBB0_5] [--max-stream 400]

If --loop-label is omitted the hottest back-edge loop (most MFMA, tie-break longest)
is auto-detected.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
from collections import Counter


def _load_amd_occupancy():
    """The occupancy model is a vendor/amd shared helper: a sibling in every composed pack,
    one layer up in the source tree. Returns None only when neither is reachable."""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    d = here
    for _ in range(7):
        try:
            return importlib.import_module("amd_occupancy")
        except ImportError:
            pass
        d = os.path.dirname(d)
        cand = os.path.join(d, "vendor", "amd", "scripts")
        if os.path.isfile(os.path.join(cand, "amd_occupancy.py")):
            sys.path.insert(0, cand)
    return None


_OCC = _load_amd_occupancy()

# Op-class -> single-char symbol. Order matters: first match wins.
# (regex on the mnemonic = first whitespace-delimited token of the instruction)
# Matrix engines: CDNA MFMA (v_mfma / v_smfmac) AND RDNA WMMA (v_wmma) are both
# classified as "mfma" — the audit's bound-class logic treats any matrix-multiply
# instruction the same way. RDNA3/4 (gfx11*/gfx120*) use v_wmma_* exclusively.
_CLASSES = [
    ("mfma",      "M", re.compile(r"^v_mfma|^v_smfmac|^v_wmma")),
    ("exp",       "e", re.compile(r"^v_(exp|log|rcp|rsq|sqrt|sin|cos)")),
    ("atomic",    "A", re.compile(r"^(global|buffer|flat|ds|scratch)_atomic")),
    ("lds_read",  "R", re.compile(r"^ds_read|^ds_load")),
    ("lds_write", "W", re.compile(r"^ds_write|^ds_store")),
    ("gld",       "g", re.compile(r"^(global|buffer|flat|scratch)_load")),
    ("gst",       "G", re.compile(r"^(global|buffer|flat|scratch)_store")),
    # ALU-dependency waits must be split OUT of the memory-wait class: s_wait_alu depctr_*
    # (gfx11/12) and s_delay_alu (gfx11) resolve a register hazard, not an outstanding
    # memory op, so folding them into the waitcnt bucket inflates the "full drain" ratio.
    ("aluwait",   "d", re.compile(r"^s_wait_alu|^s_delay_alu")),
    ("waitcnt",   "~", re.compile(r"^s_waitcnt|^s_wait_")),
    ("barrier",   "|", re.compile(r"^s_barrier|^buffer_wbinvl|^s_barrier_")),
    ("nop",       "n", re.compile(r"^s_nop")),
    ("valu",      "v", re.compile(r"^v_")),
    ("scalar",    "s", re.compile(r"^s_")),
]
_SYMBOL = {name: sym for name, sym, _ in _CLASSES}

# label line. Two emitters:
#   (a) assembler .s / .amdgcn:   "label:" (tolerate a trailing "; in Loop: ..." comment)
#   (b) llvm-objdump -d of a .hsaco/.co: "<hexaddr> <label>:"  (the prebuilt-comparator form)
_LABEL_RE = re.compile(
    r"^\s*(?:[0-9a-fA-F]+\s+)?<(?P<obj>[\w.$@]+)>:\s*$"      # (b) objdump  <name>:
    r"|^\s*(?P<asm>\.?\w[\w.$@]*):\s*(?:;.*)?$"               # (a) plain    name:
)
_BRANCH_RE = re.compile(r"^\s*s_(cbranch\w*|branch|cbranch)\s+(\.?\w[\w.$@]*)")
# TWO wait spellings. Matching only the first is why every RDNA kernel used to report
# "100% full drains": its relaxed waits fell through to the bare-waitcnt branch.
#   gfx9      s_waitcnt vmcnt(0) lgkmcnt(1)   -- counter NAMED, value in parens
#   gfx11/12  s_wait_dscnt 0x1                -- counter in the MNEMONIC, value an immediate
#             s_wait_loadcnt_dscnt 0x0        -- fused, one immediate for the pair
#             s_waitcnt_vscnt null, 0x0       -- gfx10 store-counter spelling
_CNT_RE = re.compile(r"(\w*cnt)\s*\(\s*(\d+)\s*\)")
_WAIT_IMM_RE = re.compile(
    r"^s_wait(?:cnt)?_([a-z_]*cnt(?:_[a-z_]*cnt)*)\s+(?:null\s*,\s*)?"
    r"(?:0[xX]([0-9a-fA-F]+)|(\d+))\s*$")


def classify_wait(text: str):
    """(kind, [(counter, value), ...]) for one wait instruction.

    kind: 'drain' (every named counter is 0 -- serialized), 'relaxed' (some counter > 0 --
    pipelined), 'alu' (register-dependency wait, not a memory drain), 'unknown' (a wait
    whose form this parser does not recognise; callers count it as a drain, conservatively,
    and it should be reported so the gap is visible instead of silently biasing the ratio).
    """
    s = re.split(r";|//", text, 1)[0].strip()
    if not s:
        return "unknown", []
    mn = s.split()[0]
    if mn.startswith(("s_wait_alu", "s_delay_alu")):
        return "alu", [(mn, -1)]
    cnts = [(k, int(v)) for k, v in _CNT_RE.findall(s)]           # gfx9 form
    if not cnts:
        m = _WAIT_IMM_RE.match(s)                                  # gfx11/12 form
        if m:
            val = int(m.group(2), 16) if m.group(2) else int(m.group(3))
            cnts = [(m.group(1), val)]
    if not cnts:
        return "unknown", []
    return ("drain" if all(v == 0 for _, v in cnts) else "relaxed"), cnts


def _mnemonic(line: str) -> str | None:
    """First token of a real instruction line, else None (label/directive/comment)."""
    s = line.strip()
    if not s or s.startswith((";", "//", "#", ".")) or s.endswith(":"):
        return None
    return s.split()[0]


# Kernel-descriptor / metadata register budget. Two emitters (both may appear in one .s):
#   (a) .amdhsa_kernel directive block: `.amdhsa_next_free_vgpr 244` / `.amdhsa_accum_offset 128`
#       / `.amdhsa_next_free_sgpr 96`  (LLVM AMDGPU, all gfx9/10/11/12 -- arch-agnostic).
#   (b) .amdgpu_metadata YAML tail:     `.vgpr_count: 244` / `.sgpr_count: 96` / `.agpr_count: 0`.
# next_free_vgpr = ArchVGPR + AGPR combined budget on CDNA (matches calc_perf.py KD occupancy).
_KD_RE = {
    "vgpr":       re.compile(r"^\s*\.amdhsa_next_free_vgpr\s+(\d+)"),
    "sgpr":       re.compile(r"^\s*\.amdhsa_next_free_sgpr\s+(\d+)"),
    "accum_off":  re.compile(r"^\s*\.amdhsa_accum_offset\s+(\d+)"),
}
_MD_RE = {
    "vgpr":  re.compile(r"^\s*\.vgpr_count:\s*(\d+)"),
    "sgpr":  re.compile(r"^\s*\.sgpr_count:\s*(\d+)"),
    "agpr":  re.compile(r"^\s*\.agpr_count:\s*(\d+)"),
}


def kd_regs(lines: list[str]) -> dict:
    """Scan the WHOLE file (not just the loop) for the kernel-descriptor register budget.
    Directive block wins over metadata YAML when both are present. Returns {} if neither seen
    (e.g. an llvm-objdump disasm with no KD) -- callers treat that as 'vgpr unavailable', never
    fabricate. On multiple kernels in one file, reports the MAX (worst-case occupancy driver)."""
    kd: dict = {}
    for ln in lines:
        for key, rx in _KD_RE.items():
            m = rx.match(ln)
            if m:
                kd[key] = max(kd.get(key, 0), int(m.group(1)))
        for key, rx in _MD_RE.items():
            m = rx.match(ln)
            if m:
                kd.setdefault("_md_" + key, 0)
                kd["_md_" + key] = max(kd["_md_" + key], int(m.group(1)))
    out: dict = {}
    vgpr = kd.get("vgpr", kd.get("_md_vgpr"))
    sgpr = kd.get("sgpr", kd.get("_md_sgpr"))
    if vgpr is not None:
        out["vgpr"] = vgpr           # next_free_vgpr = arch+accum combined (CDNA)
    if sgpr is not None:
        out["sgpr"] = sgpr
    if "_md_agpr" in kd:
        out["accum_vgpr"] = kd["_md_agpr"]
    elif "accum_off" in kd and vgpr is not None:
        out["accum_vgpr"] = max(0, vgpr - kd["accum_off"])
    return out


def print_kd_regs(lines: list[str]) -> None:
    """Register budget + waves/SIMD. The occupancy answer is ARCH-SPECIFIC (CDNA divides a
    512-entry combined file, RDNA a 1536-entry one with a 256/wave cap), and LLVM already
    computed it for this exact subtarget when the dump carries `; Occupancy:` -- prefer that
    over any model here."""
    kd = kd_regs(lines)
    text = "\n".join(lines)
    arch = _OCC.arch_from_asm(text) if _OCC else None
    print("kernel-descriptor register budget (occupancy driver):")
    if not kd:
        print("  (no .amdhsa_next_free_vgpr / .vgpr_count found -- KD absent in this dump; "
              "vgpr UNAVAILABLE, do not guess. Dump a full .s with the KD block, or read the "
              "profiler VGPR_Count.)")
        print()
        return
    v = kd.get("vgpr")
    is_cdna = (_OCC.family_for(arch) == "cdna") if _OCC else True
    if v is not None:
        combined = "  (ArchVGPR+AGPR combined, CDNA)" if is_cdna else \
                   "  (no AGPR file on this arch)"
        print(f"  next_free_vgpr = {v}{combined}")
        if _OCC is None:
            print("    waves/SIMD UNAVAILABLE -- amd_occupancy.py not importable; read LLVM's "
                  "`; Occupancy:` from the .s")
        else:
            llvm_occ = _OCC.llvm_occupancy(text)
            waves, model = _OCC.waves_by_vgpr(v, arch)
            if llvm_occ is not None:
                print(f"    -> waves/SIMD = {llvm_occ}   [LLVM `; Occupancy:` -- authoritative, "
                      f"computed by the backend for {arch or 'this subtarget'}]")
                if waves is not None and waves != llvm_occ:
                    print(f"       (model for {arch} says {waves}; trust LLVM and report the "
                          f"mismatch -- the model or the arch table is stale)")
            elif waves is not None:
                print(f"    -> waves/SIMD by VGPR <= {waves}   [model: {model}]")
            else:
                print(f"    -> waves/SIMD UNAVAILABLE: {model}")
    if kd.get("accum_vgpr") is not None:
        print(f"  accum_vgpr     = {kd['accum_vgpr']}")
    if kd.get("sgpr") is not None:
        print(f"  next_free_sgpr = {kd['sgpr']}")
    print(f"  (arch detected: {arch or 'UNKNOWN -- no .amdgcn_target in this dump'}; register "
          f"file geometry differs per arch -- use hotspot_analyzer.py for the full per-arch "
          f"limiter table.)")
    print()


def lds_from_meta(meta_path: str) -> list[tuple[str, int]]:
    """LDS bytes/workgroup from the Triton cache METADATA (`shared`) -- the only correct source.

    Why not the two obvious places: Triton sizes shared memory DYNAMICALLY at launch
    (`@global_smem = external ... addrspace(3) global [0 x i8]`), so the kernel descriptor's
    `.amdhsa_group_segment_fixed_size` is STRUCTURALLY 0 for every Triton kernel, and
    rocprof-compute's `7.1.8 LDS Allocation` inherits the same 0 -- while block 2.1.17 of the
    SAME report can show real LDS bandwidth. An LDS-staging kernel therefore reads as ZERO LDS
    everywhere except here.

    `meta_path` is a metadata .json or a directory to scan. `__grp__*.json` (launcher group
    metadata) carries no `shared` and is skipped. Returns [(kernel_name, bytes), ...].
    """
    cands: list[str] = []
    if os.path.isdir(meta_path):
        for root, _dirs, files in os.walk(meta_path):
            cands += [os.path.join(root, f) for f in files
                      if f.endswith(".json") and not os.path.basename(f).startswith("__grp__")]
    elif os.path.exists(meta_path):
        cands = [meta_path]
    out: list[tuple[str, int]] = []
    for p in sorted(cands):
        try:
            d = json.load(open(p))
        except (OSError, ValueError):
            continue
        if isinstance(d, dict) and isinstance(d.get("shared"), int):
            out.append((str(d.get("name") or os.path.basename(p))[:40], d["shared"]))
    return out


# LDS/CU comes from the shared vendor/amd occupancy model (per-arch there, because
# gfx94*/gfx95* share the register family but not the LDS: 64 KiB vs 160 KiB).
def lds_per_cu(arch, default=None):
    """LDS bytes per CU for `arch`, or None when the shared model has no figure.

    `default` is None for the same reason it is in `probe.py`: 65536 is the CDNA3 figure,
    and handing it to a gfx950 kernel overstates LDS pressure by 2.5x -- a confident wrong
    occupancy verdict, which is worse than declining. Pass `--lds-per-cu` to name it.
    """
    try:
        import amd_occupancy as _o
    except ImportError:
        return default
    return _o.lds_per_cu(arch) or default


    for k, v in LDS_PER_CU_BY_ARCH.items():
        if arch.startswith(k):
            return v
    return default


def print_lds_budget(meta_path: str | None, lds_per_cu: int | None) -> None:
    """LDS/WG -> WGs/CU. A SECOND occupancy limiter, independent of the register one: a kernel
    can be capped by both at once, and relieving only one buys nothing."""
    print("LDS budget per workgroup (the SECOND occupancy limiter):")
    kernels = lds_from_meta(meta_path) if meta_path else []
    if not kernels:
        print("  UNAVAILABLE -- pass --meta <ir-dir | kernel.json> (Triton cache metadata).")
        print("  Do NOT substitute the KD's group_segment_fixed_size or rocprof-compute 7.1.8: "
              "both are")
        print("  structurally 0 for Triton kernels and will read as 'this kernel uses no LDS'.")
        print()
        return
    # several kernels in one dump -> the MAX drives occupancy (same rule as kd_regs).
    name, shared = max(kernels, key=lambda kv: kv[1])
    if shared:
        if lds_per_cu:
            print(f"  lds_bytes_per_wg = {shared}  ({name})  -> WGs/CU by LDS <= "
                  f"{lds_per_cu // shared}  ({lds_per_cu}/{shared})")
        else:
            print(f"  lds_bytes_per_wg = {shared}  ({name})  -> WGs/CU by LDS UNKNOWN: no "
                  f"hw_constants.json figure for this arch. Pass --lds-per-cu rather than "
                  f"assuming 64 KiB, which is 2.5x wrong on CDNA4.")
    else:
        print(f"  lds_bytes_per_wg = 0  ({name})  -> no LDS staging in this kernel")
    for n, s in sorted(kernels, key=lambda kv: -kv[1])[1:]:
        print(f"    (also: {n} = {s} B)")
    print("  (source: Triton cache metadata `shared`; CDNA gfx942/gfx950 = 64 KiB LDS/CU.)")
    print()


def classify(mnemonic: str) -> tuple[str, str]:
    for name, sym, rx in _CLASSES:
        if rx.match(mnemonic):
            return name, sym
    return "other", "."


def find_loops(lines: list[str]) -> list[tuple[str, int, int]]:
    """Return (label, body_start_idx, branch_idx) for every back-edge loop.

    A back-edge = a branch whose target label appears at an EARLIER line.
    """
    label_line: dict[str, int] = {}
    for i, ln in enumerate(lines):
        m = _LABEL_RE.match(ln)
        if m:
            name = m.group("obj") or m.group("asm")
            label_line.setdefault(name, i)
    loops = []
    for i, ln in enumerate(lines):
        m = _BRANCH_RE.match(ln)
        if not m:
            continue
        target = m.group(2)
        ti = label_line.get(target)
        if ti is not None and ti < i:
            loops.append((target, ti + 1, i))
    return loops


def kernel_body_span(lines: list[str]) -> tuple[str, int, int] | None:
    """The instruction span of the LARGEST kernel body in the file: (name, start, end).

    Why this exists. The schedule audit keys on a back-edge, and a kernel with no back-edge used to
    end the static layer right there -- register budget printed, no op mix. That is fine for a GEMM
    (which always has a K loop) and wrong for the kernels raw C++ is most often used to write: an
    elementwise map, a fused epilogue, a fully-unrolled tile. Those have exactly one basic block,
    and their op mix is the whole point ("is the load 2 bytes or 16?"), so going blind there loses
    the evidence the memory lever is chosen on.

    The span runs from the kernel's own label to its `s_endpgm`. Directives before the label
    (`.amdhsa_*`, `.globl`) are excluded so the histogram counts instructions only."""
    ends = [i for i, ln in enumerate(lines) if _mnemonic(ln) == "s_endpgm"]
    if not ends:
        return None
    best = None
    for end in ends:
        # walk back to the nearest preceding label that is not a local branch target: the kernel
        # symbol. Fall back to the first instruction after the previous s_endpgm.
        start, name = None, None
        for i in range(end - 1, -1, -1):
            m = _LABEL_RE.match(lines[i])
            if not m:
                continue
            lab = m.group("obj") or m.group("asm")
            if lab.startswith(".LBB") or lab.startswith(".Lfunc"):
                continue          # a basic-block label, not the kernel entry
            start, name = i + 1, lab
            break
        if start is None:
            continue
        span = end - start
        if best is None or span > best[2] - best[1]:
            best = (name, start, end)
    return best


def _count_mfma(lines: list[str], start: int, end: int) -> int:
    """Count matrix-multiply instructions (MFMA on CDNA, WMMA on RDNA) in [start,end).
    Used by pick_loop to score the hottest back-edge loop."""
    n = 0
    for ln in lines[start:end]:
        mn = _mnemonic(ln)
        if mn and classify(mn)[0] == "mfma":
            n += 1
    return n


def _encloses(outer, inner) -> bool:
    """True if loop `outer` strictly contains loop `inner` (its body range is a superset and they are
    not the same loop). Ranges are [start, end)."""
    _, os_, oe = outer
    _, is_, ie = inner
    return os_ <= is_ and ie <= oe and (os_, oe) != (is_, ie)


def pick_loop(lines: list[str], loops, forced_label: str | None):
    if forced_label:
        for lab, s, e in loops:
            if lab == forced_label:
                return lab, s, e
        sys.exit(f"loop label {forced_label!r} not found among back-edges: "
                 f"{[l[0] for l in loops]}")
    if not loops:
        return None
    # Prefer the INNERMOST hot loop. An enclosing loop contains ALL of an inner loop's MFMA, so it
    # ties on the MFMA score and (with a longest-body tie-break) would win -- folding the kernel
    # EPILOGUE between the inner exit and the outer back-edge into the per-opcode histogram, so a
    # once-per-workgroup store/cast burst reads as a large fraction of the "hot loop" and misdirects
    # the lever choice. So:
    #   1) restrict to innermost loops (those enclosing no other back-edge) when any exist;
    #   2) among candidates, most MFMA;
    #   3) tie-break on SHORTEST body (the tight hot loop, not the enclosing scope).
    innermost = [lp for lp in loops if not any(_encloses(lp, other) for other in loops)]
    candidates = innermost or loops
    scored = [((_count_mfma(lines, s, e), -(e - s)), (lab, s, e)) for lab, s, e in candidates]
    scored.sort(reverse=True)
    return scored[0][1]


_OPC_ROW = re.compile(r"^\s{4}(\w+)\s+(\d+)\s+\(\s*([\d.]+)%\)")


def parse_opcodes(path: str) -> dict[str, int]:
    """Read the per-opcode section back out of a previous run's stdout, for --diff.
    Round-over-round movement is the point: a lever that was supposed to delete a mnemonic
    family and did not is a failed edit, visible here one round earlier than in the timing."""
    out: dict[str, int] = {}
    inside = False
    try:
        txt = open(path, errors="ignore").read()
    except OSError:
        return out
    for ln in txt.splitlines():
        if ln.startswith("per-opcode histogram"):
            inside = True
            continue
        if inside:
            m = _OPC_ROW.match(ln)
            if m:
                out[m.group(1)] = int(m.group(2))
            elif ln.strip() and not ln.startswith(" "):
                break
    return out


def print_opcodes(opc: Counter, total: int, top: int, prev: dict[str, int] | None) -> None:
    """Per-MNEMONIC histogram. The op-CLASS histogram above is one level too coarse to act on:
    a software dtype-conversion sequence, an address-arithmetic chain and a layout permute all
    land in 'valu' and have opposite fixes. This is the granularity a lever is chosen at."""
    if top <= 0 or not opc:
        return
    print(f"per-opcode histogram (hot loop, top {top} of {len(opc)} distinct mnemonics):")
    for mn, n in opc.most_common(top):
        pct = 100.0 * n / total if total else 0.0
        d = ""
        if prev is not None:
            was = prev.get(mn, 0)
            if n != was:
                d = f"   {n - was:+d} vs prev"
            elif mn in prev:
                d = "    = prev"
        print(f"    {mn:<26} {n:>5}  ({pct:4.1f}%){d}")
    if prev:
        gone = [m for m in prev if m not in opc]
        if gone:
            print(f"    (eliminated since prev: {', '.join(sorted(gone)[:8])}"
                  f"{' ...' if len(gone) > 8 else ''})")
    print("  (a family >10% of the loop that you cannot name the source construct for is the "
          "lever\n   to look for; op-class % cannot show you which one it is.)")
    print()


def audit(lines: list[str], start: int, end: int, max_stream: int,
          top_opcodes: int = 12, prev_opcodes: dict[str, int] | None = None) -> None:
    hist: Counter = Counter()
    opc: Counter = Counter()
    stream: list[str] = []
    relaxed = full_drain = unknown_wait = alu_waits = barriers = 0
    nops = nop_cycles = 0
    waitcnt_detail: Counter = Counter()
    alu_detail: Counter = Counter()
    for ln in lines[start:end]:
        mn = _mnemonic(ln)
        if mn is None:
            continue
        name, sym = classify(mn)
        hist[name] += 1
        opc[mn] += 1
        stream.append(sym)
        if name == "barrier":
            barriers += 1
        if name == "nop":
            nops += 1
            m = re.search(r"^s_nop\s+(?:0x([0-9a-fA-F]+)|(\d+))", ln.strip())
            if m:
                nop_cycles += int(m.group(1), 16) if m.group(1) else int(m.group(2))
        if name == "aluwait":
            alu_waits += 1
            alu_detail[re.split(r";|//", ln, 1)[0].strip()] += 1
        if name == "waitcnt":
            kind, cnts = classify_wait(ln)
            if kind == "drain":
                full_drain += 1
            elif kind == "relaxed":
                relaxed += 1
                for k, v in cnts:
                    waitcnt_detail[f"{k}({v})"] += 1
            else:
                # a wait whose form this parser does not know -> counted as a drain
                # (conservative) but reported, so a new spelling shows up as a parser gap
                # instead of quietly inflating the drain ratio.
                full_drain += 1
                unknown_wait += 1

    total = sum(hist.values())
    # Pipeline/interleave signals over the MFMA-vs-VALU subsequence (computed on the
    # correctly-scoped stream, not an external re-parse). "M" = matrix (mfma/wmma);
    # "V" = VALU-family (valu + exp/transcendental). Transitions ~ how finely the two
    # are interleaved; the longest V-only run is the largest exposed VALU stretch (no MFMA
    # to hide it). NOTE: more transitions is NOT automatically faster -- read alongside
    # per-unit busy (PMC) and the overhead classes (scalar/LDS-read/s_nop).
    mv = ["M" if s == "M" else "V" for s in stream if s in ("M", "v", "e")]
    mv_transitions = sum(1 for i in range(1, len(mv)) if mv[i] != mv[i - 1])
    longest_v_run = _cur = 0
    for c in mv:
        if c == "V":
            _cur += 1
            longest_v_run = max(longest_v_run, _cur)
        else:
            _cur = 0
    n_mfma_in_mv = mv.count("M")
    print(f"hot loop: lines [{start + 1}, {end + 1}]  ({total} instructions)\n")

    print("op-class symbol stream "
          "(M=mfma/wmma e=exp R=ldsRead W=ldsWrite g=gLoad G=gStore "
          "v=valu s=scalar ~=memWait d=aluDepWait |=barrier n=nop .=other):")
    s = "".join(stream)
    if len(s) > max_stream:
        s = s[:max_stream] + f" ... (+{len(stream) - max_stream} more)"
    print("  " + s + "\n")

    print("histogram:")
    for name, _sym, _rx in _CLASSES + [("other", ".", None)]:
        if hist.get(name):
            pct = 100.0 * hist[name] / total if total else 0.0
            print(f"  {_SYMBOL.get(name, '.'):>1} {name:<10} {hist[name]:>5}  ({pct:4.1f}%)")
    print()

    print_opcodes(opc, total, top_opcodes, prev_opcodes)

    wc_total = relaxed + full_drain
    print("memory-wait quality (s_waitcnt / s_wait_*cnt):")
    print(f"  relaxed (cnt>0, pipelined) : {relaxed}")
    print(f"  full-drain (cnt==0, serialized): {full_drain}")
    if wc_total:
        print(f"  -> {100.0 * full_drain / wc_total:.0f}% of memory waits are full drains")
    if unknown_wait:
        print(f"  ({unknown_wait} wait(s) in an unrecognised form counted as drains -- PARSER "
              f"GAP, the ratio above is an upper bound; report the spelling)")
    if waitcnt_detail:
        top = ", ".join(f"{k}={n}" for k, n in waitcnt_detail.most_common(6))
        print(f"  relaxed counts seen: {top}")
    print(f"ALU-dependency waits (s_wait_alu / s_delay_alu): {alu_waits}")
    if alu_detail:
        top = ", ".join(f"{k}={n}" for k, n in alu_detail.most_common(4))
        print(f"  {top}")
    print("  (register-hazard waits, NOT memory drains -- excluded from the ratio above. Many "
          "of them\n   means VALU/SALU dependency chains, which unroll or wider vectors hide, "
          "not waitcnt relaxation.)")
    print()

    print(f"producer<->consumer barriers in loop body: {barriers}")
    print()
    print("MFMA<->VALU interleave (pipeline schedule quality):")
    print(f"  MFMA<->VALU transitions: {mv_transitions}   (#MFMA in M/V stream: {n_mfma_in_mv})")
    print(f"  longest VALU-only run (no interleaved MFMA): {longest_v_run}")
    print("  (more transitions = finer interleave, but NOT automatically faster -- weigh")
    print("   against per-unit busy (PMC) + overhead classes; a long VALU-only run is an")
    print("   exposed stretch no MFMA hides. Use kernel_breakdown.py to merge with PMC.)")
    print()
    print("s_nop (exposed fixed-latency hazard, e.g. MFMA-write -> VALU-read):")
    print(f"  count: {nops}   requested stall cycles: {nop_cycles}")
    print("  (high s_nop => a fixed-latency hazard is NOT hidden; the fix is more")
    print("   unroll/occupancy to fill it, NOT reorder -- reorder cannot create slack)")
    print()
    print("VERDICT IS YOURS (signals only): scan the stream for an exposed single-class")
    print("run with no interleaved M; ask whether any independent ready op could fill it")
    print("(no => structural/no-slack; yes => scheduling miss). Many full drains =>")
    print("conservative-waitcnt; barriers scaling with depth => excess-sync; high s_nop")
    print("=> unhidden fixed-latency hazard (unroll/occupancy, not reorder). See")
    print("compiler-contract.md ## Auditing the hot-loop schedule.")


def _selftest() -> int:
    # Synthetic nested loop: an OUTER loop whose body contains an INNER loop plus an epilogue. The
    # inner loop holds the MFMA; the epilogue holds a store-cast burst. The outer loop contains ALL
    # the inner MFMA (ties on the MFMA score) and is LONGER -> the old (mfma, +len) tie-break picked
    # it, folding the epilogue into the hot-loop histogram. pick_loop must now pick the INNER loop.
    asm = [
        "outer_head:",                 # 0  outer back-edge target
        "    s_add_u32 s0, s0, 1",     # 1
        "inner_head:",                 # 2  inner back-edge target
        "    v_mfma_f32_16x16x16 a[0:3], v0, v1, a[0:3]",  # 3  MFMA (inner)
        "    v_mfma_f32_16x16x16 a[0:3], v2, v3, a[0:3]",  # 4  MFMA (inner)
        "    s_add_u32 s1, s1, 1",     # 5
        "    s_cbranch_scc0 inner_head",  # 6  inner back-edge
        "    v_cvt_f16_f32 v10, v10",  # 7  epilogue store-cast (once per outer iter, NOT hot)
        "    v_cvt_f16_f32 v11, v11",  # 8
        "    v_cvt_f16_f32 v12, v12",  # 9
        "    v_cvt_f16_f32 v13, v13",  # 10
        "    s_cbranch_scc0 outer_head",  # 11 outer back-edge
    ]
    loops = find_loops(asm)
    labs = {l[0] for l in loops}
    assert labs == {"outer_head", "inner_head"}, labs
    lab, s, e = pick_loop(asm, loops, None)
    assert lab == "inner_head", (f"innermost loop must win, got {lab}", loops)
    # the picked body must NOT include the epilogue cvt burst (that was the bug)
    assert not any("v_cvt" in asm[i] for i in range(s, e)), ("epilogue folded into hot loop", s, e)
    # --loop-label override still forces the outer loop
    lab2, _, _ = pick_loop(asm, loops, "outer_head")
    assert lab2 == "outer_head", lab2
    # a single flat loop still works
    flat = ["h:", "    v_mfma_f32_16x16x16 a[0:3], v0, v1, a[0:3]", "    s_cbranch_scc0 h"]
    fl = find_loops(flat)
    assert pick_loop(flat, fl, None)[0] == "h", fl

    # --- wait classification, both ISA spellings -------------------------------------
    # gfx9: counter named, value in parens
    assert classify_wait("s_waitcnt vmcnt(0) lgkmcnt(0)")[0] == "drain"
    assert classify_wait("s_waitcnt vmcnt(0) lgkmcnt(3)")[0] == "relaxed"
    assert classify_wait("s_waitcnt_vscnt null, 0x0")[0] == "drain"
    # gfx11/12: counter in the mnemonic, hex or decimal immediate
    assert classify_wait("s_wait_dscnt 0x0")[0] == "drain"
    assert classify_wait("s_wait_dscnt 0x1")[0] == "relaxed"
    assert classify_wait("s_wait_loadcnt 0x7")[0] == "relaxed"
    assert classify_wait("s_wait_loadcnt 7")[0] == "relaxed"
    assert classify_wait("s_wait_kmcnt 0x0")[0] == "drain"
    assert classify_wait("s_wait_loadcnt_dscnt 0x0")[0] == "drain"
    assert classify_wait("s_wait_dscnt 0xb")[1] == [("dscnt", 11)], classify_wait("s_wait_dscnt 0xb")
    # ALU-dependency waits are their OWN class, never a memory drain
    assert classify_wait("s_wait_alu depctr_va_vcc(0)")[0] == "alu"
    assert classify_wait("s_delay_alu instid0(VALU_DEP_1)")[0] == "alu"
    assert classify("s_wait_alu")[0] == "aluwait" and classify("s_wait_dscnt")[0] == "waitcnt"

    # --- hand-written gfx1201 fragment (synthetic: values chosen so the expected counts are
    # obvious by inspection; nothing here is excerpted from a real kernel) ---------------
    rdna = [
        '\t.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"',
        "\t\t.amdhsa_wavefront_size32 1",
        "\t\t.amdhsa_next_free_vgpr 168",
        "rdna_loop:",
        "    s_wait_dscnt 0x1",                     # relaxed
        "    ds_read_b128 v[0:3], v20",
        "    s_wait_dscnt 0x0",                     # drain
        "    v_wmma_f32_16x16x16_f16 v[8:11], v[0:3], v[4:7], v[8:11]",
        "    s_wait_alu depctr_va_vcc(0)",          # ALU dep, NOT a drain
        "    s_wait_loadcnt 0x7",                   # relaxed
        "    s_cbranch_scc0 rdna_loop",
    ]
    assert _OCC is not None, "amd_occupancy.py must be importable (sibling in a composed pack)"
    assert _OCC.arch_from_asm("\n".join(rdna)) == "gfx1201"
    # 168 VGPR on RDNA = 9 waves/SIMD; the old CDNA formula would have said 512/168 -> 3
    assert _OCC.waves_by_vgpr(168, "gfx1201")[0] == 9
    assert _OCC.waves_by_vgpr(168, "gfx942")[0] == 3
    lab, s, e = pick_loop(rdna, find_loops(rdna), None)
    assert lab == "rdna_loop", lab
    kinds = Counter()
    for ln in rdna[s:e]:
        mn = _mnemonic(ln)
        if mn and classify(mn)[0] in ("waitcnt", "aluwait"):
            kinds[classify_wait(ln)[0]] += 1
    assert kinds == Counter({"relaxed": 2, "drain": 1, "alu": 1}), kinds
    # WMMA counts as a matrix op, same as MFMA
    assert _count_mfma(rdna, s, e) == 1, _count_mfma(rdna, s, e)
    print("[asm_loop_audit] SELFTEST PASS")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("asm", help="stripped .s (or .amdgcn) file from dump_ir.sh")
    ap.add_argument("--loop-label", help="force a specific loop back-edge label")
    ap.add_argument("--max-stream", type=int, default=400,
                    help="truncate the printed symbol stream to N symbols (default 400)")
    # nargs="?" because the docs describe this as "LDS bytes/WG come from --meta", which
    # reads as a flag -- four separate runs invoked a bare `--meta`, got argparse's
    # "expected one argument", and lost the LDS read for that round. The bare form now
    # means what the help already promised: look next to <asm>.
    ap.add_argument("--meta", nargs="?", const="",
                    help="Triton cache metadata .json (or a dir to scan) for the LDS "
                         "bytes/WG. Bare `--meta`, or omitted, defaults to the dir holding "
                         "<asm>. The KD and rocprof-compute both report 0 LDS for Triton -- "
                         "this is the only correct source.")
    # The old help said "CDNA gfx942/gfx950 64 KiB", which is wrong and contradicts this
    # pack's own arch cards: gfx950 has 160 KiB/CU. Left as a flag, but derivable from --arch.
    ap.add_argument("--arch", default=None,
                    help="gfx942 / gfx950 / ... -- used to pick the LDS-per-CU divisor")
    ap.add_argument("--lds-per-cu", type=int, default=None,
                    help="LDS bytes per CU. Default is derived from --arch "
                         "(gfx942 65536, gfx950 163840); 65536 if neither is given.")
    ap.add_argument("--opcodes", type=int, default=12, metavar="N",
                    help="rows in the per-MNEMONIC hot-loop histogram (default 12, 0 = off). "
                         "The op-CLASS histogram is one level too coarse to pick a lever from")
    ap.add_argument("--diff", metavar="PREV.TXT",
                    help="a previous run's stdout -> per-opcode delta round-over-round")
    a = ap.parse_args()

    try:
        lines = open(a.asm).read().splitlines()
    except OSError as e:
        sys.exit(f"cannot read {a.asm!r}: {e}")

    # KD register budget (vgpr/sgpr/accum) is a whole-file fact -- print it FIRST, before the
    # loop check, so a loop-less kernel (e.g. a trivial reduction) still yields the vgpr occupancy
    # driver instead of exiting empty. The loop schedule audit is separate and may be absent.
    print(f"=== asm loop audit: {a.asm} ===\n")
    print_kd_regs(lines)
    # The register file is only ONE of the two occupancy limiters; print the LDS one next to it
    # so they are never reasoned about separately.
    _lds = a.lds_per_cu if a.lds_per_cu else lds_per_cu(a.arch)
    print_lds_budget(a.meta or os.path.dirname(os.path.abspath(a.asm)), _lds)

    loops = find_loops(lines)
    picked = pick_loop(lines, loops, a.loop_label)
    if picked is None:
        # No back-edge. Audit the WHOLE KERNEL BODY instead of going blind: for a single-block
        # kernel that body IS the hot code, and its op mix is the evidence a memory/index lever is
        # picked on. Labelled explicitly, because two things that must not be confused read the
        # same in a histogram: per-iteration cost (a loop) and per-thread total cost (this).
        span = kernel_body_span(lines)
        if span is None:
            print("no back-edge loop found AND no s_endpgm to bound a kernel body -- is this a "
                  "device .s? (dump_ir.sh writes <stem>.<arch>.s; a host-only dump has neither). "
                  "KD register budget above is still valid.")
            return
        name, start, end = span
        print(f"no back-edge loop found -> auditing the WHOLE KERNEL BODY of {name} "
              f"({end - start} lines).")
        print("  This is a PER-THREAD total, not a per-iteration cost: do not read the counts below "
              "as a steady-state schedule, and do not compare them against a looped kernel's "
              "per-iteration histogram.\n")
        print(f"--- kernel body @ {name} (no loop) ---\n")
        audit(lines, start, end, a.max_stream, a.opcodes,
              parse_opcodes(a.diff) if a.diff else None)
        return
    lab, start, end = picked
    print(f"--- hot loop @ {lab} ---\n")
    audit(lines, start, end, a.max_stream, a.opcodes,
          parse_opcodes(a.diff) if a.diff else None)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    main()
