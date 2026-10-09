#!/usr/bin/env python3
"""parity_gate.py - the anchor's transcription debt: is it paid, and if not, who owes it?

A faithful anchor -- transcribed into Gluon, or hand-ported into FlyDSL -- is a REGRESSION you
knowingly created, not a baseline to quietly climb
from. Climbing before the debt is paid caps the whole port: every later lever is measured against
a broken starting point, and the run closes below the champion while reporting a healthy-looking
gain "vs the anchor". That is the single most expensive procedural mistake available in a port,
and it is invisible without this gate, because the layout-equivalence checker says PASS and the
numeric oracle says PASS while the kernel runs 1.4x slower.

So: until `champion_ms / anchor_ms >= --threshold` (0.95 by default), a round's outcome is
`recovery` against the suspect it closed -- never a win. This tool decides that, and when the
gate is NOT cleared it attributes the gap across the three suspects, each from a signal in the
compiled artifacts rather than from a story:

  lost_pipeline  the champion's loop was software-pipelined and the anchor's is not.
                 Evidence: the champion's TTGIR carries `ttg.memdesc_index` / `ttg.local_store` /
                 `num_stages > 1` and the anchor's does not. Owned by the pipeline layer,
                 hand-written FIRST: register-level prefetch, then an authored LDS ring, then
                 warp_pipeline_stage + a scheduling-model choice (tile-programming/pipeline.md).
                 Re-injecting plain's pipeliner (gluon_swp / patch_reinject) is the LAST resort:
                 a diagnostic that measures this debt, or a fallback when the hand-written form
                 cannot reach parity -- its numbers are labelled `injected`, never a win, and it
                 never runs on an already-Gluon incumbent.
                 NOTE the inverse trap: `max iter_args >= 2` is NOT evidence of pipelining, any
                 accumulator loop satisfies it. Only memdesc_index / local_store / a peeled
                 prologue are.

  lost_layout    a conversion was folded backwards into a load, or a staging buffer was
                 materialized that the champion left to the compiler. Evidence: the load-width
                 or LDS-op histogram shifted toward NARROWER operations (dwordx4 -> ushort,
                 ds_read_b128 -> ds_read_u16), or `shared` bytes/WG crossed an LDS/CU divisor,
                 or occupancy fell AND the anchor's binding term is the LDS one.
                 Owned by the memory-path / shared-layout layers.

  lost_RA        the instruction multiset is essentially unchanged and the register allocator
                 serialized it anyway. Evidence: VGPR rose (especially across a wave
                 threshold), spill appeared, occupancy fell with the anchor REGISTER-bound, or
                 the number of DISTINCT address registers feeding the LDS read burst
                 collapsed. This is the one a layout-equivalence checker structurally cannot
                 see: equivalent layouts, equal counters, unequal address-register pressure.
                 Read the `ds_read` OPERANDS, not just the count.

Occupancy is compared as min(register term, LDS term) per arm, and the gate NAMES the term
that bound each arm -- because that name is what routes the owner. `; Occupancy: N` alone is
the register term, and an anchor that lost a wave to a materialized staging buffer moves no
register count, so a register-only comparison is silent on the most common lost_layout
mechanism there is. Every input the LDS term needs (arch, group-segment bytes, workgroup size)
is refused rather than defaulted when the artifacts do not state it: see `occupancy_terms`.

Usage:
  parity_gate.py --champion-ms 2.8966 --anchor-ms 4.0754 \
                 --champion-asm ir/champion/k.amdgcn --anchor-asm ir/anchor/k.amdgcn \
                 [--champion-ttgir ir/champion/k.ttgir] [--anchor-ttgir ir/anchor/k.ttgir] \
                 [--champion-lds 32768] [--anchor-lds 65536] \
                 [--champion-workgroup-size 256] [--anchor-workgroup-size 256] \
                 [--champion-arch gfx950] [--anchor-arch gfx950] \
                 [--threshold 0.95] [--json parity.json]
  parity_gate.py --selftest

Exit status: 0 when the gate is CLEARED, 2 when it is not (so a round script can branch on it).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

# --- what "narrower" means, per family. Ordered widest -> narrowest; the index IS the rank. ---
GLOBAL_WIDTHS = ["dwordx4", "dwordx3", "dwordx2", "dword", "ushort", "ubyte"]
LDS_WIDTHS = ["b128", "b96", "b64", "b32", "u16", "u8"]

# next_free_vgpr is the ONLY number the 256 cliff is measured against: ArchVGPR and AGPR share
# ONE 512-register file per SIMD, so `num_vgpr` / `.vgpr_count` (arch only) reads comfortably
# under 256 on a kernel that is actually past it. Keep both, decide on next_free.
_NEXTFREE_RE = re.compile(r"\.amdhsa_next_free_vgpr\s+(\d+)")
_ARCHVGPR_RE = re.compile(r"(?:\.set\s+\S*num_vgpr,\s*|\.vgpr_count:\s*)(\d+)")
_ACCUM_RE = re.compile(r"(?:\.amdhsa_accum_offset\s+|;\s*AccumOffset\s*:\s*)(\d+)")
_OCC_RE = re.compile(r"^\s*;\s*Occupancy\s*:\s*(\d+)", re.M)
_SPILL_RE = re.compile(r"(?:\.vgpr_spill_count:\s*|;\s*ScratchSize\s*:\s*)(\d+)")
# `LDSByteSize` in the .amdgcn is a structural 0 on Triton kernels -- shared memory is sized
# dynamically at launch, so the compile-time field says "0 bytes/workgroup (compile time only)".
# The real figure is the `shared` field of the Triton cache metadata; pass it in with --*-lds.
# Fallback only: when amd_occupancy is importable its reader is used instead, because it also
# reads `.amdhsa_group_segment_fixed_size` (the allocation) ahead of this comment (a report).
_LDS_RE = re.compile(r"^\s*;\s*LDSByteSize:\s*(\d+)", re.M)


def _first_int(rx, text, default=None):
    m = rx.search(text)
    return int(m.group(1)) if m else default


# --- the two occupancy terms, and which one binds ------------------------------------------
#
# `; Occupancy: N` is LLVM's own number and it is authoritative for the REGISTER term: it is
# emitted from the register allocation, and on every kernel whose shared memory is sized at
# launch the same dump prints `LDSByteSize: 0 bytes/workgroup (compile time only)` beside it.
# Two arms compared on that number alone therefore cannot see the mechanism THIS FILE names as
# the most common lost_layout failure in a port -- "a materialized staging buffer cost an
# occupancy step" -- because a staging buffer does not move a register count. The gate would
# report `occupancy_dropped: absent` on exactly the case it exists to catch.
#
# So each arm gets `min(register term, LDS term)` and RECORDS WHICH TERM BOUND IT. Whether a
# statically allocated group segment is already folded into `; Occupancy:` upstream is not
# verified here, and nothing below assumes either way -- it does not need to. If it IS folded
# in, the min is a no-op; if it is NOT, the min is the correction. The min is right under both
# hypotheses, which is why it can be taken without settling the question.
#
# Every input this needs is REFUSED when absent, never substituted:
#   * no arch named in the dump -> no LDS term at all. Never a default arch: gfx942 has 64 KiB
#     of LDS per CU and gfx950 has 160 KiB, so a guessed divisor is wrong by 2.5x -- and it
#     would be wrong INSIDE a min(), where too small an LDS term silently becomes the verdict.
#   * no group-segment field, or a field that states 0 -> LDS bytes are UNAVAILABLE, not 0.
#   * no workgroup size -> wg/CU cannot be converted to waves/SIMD, so no LDS term.
#   * amd_occupancy not importable -> no LDS term; this file carries no LDS/CU table of its own.
# In each case the arm reports the register term, flagged register-term-only WITH the reason.
_OCC_MOD = None
_OCC_MOD_WHY = None


def _occ_module():
    """`amd_occupancy`, or None. Lazy and non-fatal: this file still runs without it, at the
    cost of the LDS term -- which it then says it lost, rather than inventing a divisor."""
    global _OCC_MOD, _OCC_MOD_WHY
    if _OCC_MOD is None and _OCC_MOD_WHY is None:
        try:
            here = str(Path(__file__).resolve().parent)
            if here not in sys.path:
                sys.path.insert(0, here)
            import amd_occupancy as _o
            _OCC_MOD = _o
        except Exception as exc:                                   # pragma: no cover - env dep
            _OCC_MOD_WHY = (f"amd_occupancy is not importable beside this script "
                            f"({exc.__class__.__name__}), so no LDS term was computed")
    return _OCC_MOD


def lds_facts(text: str, lds_bytes: int | None = None) -> tuple[int | None, str]:
    """(bytes/workgroup, source) for one arm, or (None, "UNAVAILABLE -- ...").

    A STATED 0 is not zero LDS here. On any kernel whose shared memory is sized at launch the
    compile-time group-segment field reads 0 while the kernel really does stage through LDS,
    and that is this tool's whole population. So a 0 from the asm is reported as UNAVAILABLE --
    naming the field that said 0 -- and only --*-lds, or a NONZERO field, is used as a number.
    "Does not say" and "says zero" are both refusals; neither is allowed to become a divisor.
    """
    if lds_bytes is not None:
        return lds_bytes, "caller (--*-lds, from the Triton cache metadata's `shared`)"
    o = _occ_module()
    if o is not None:
        val, field = o.lds_bytes_per_wg_from_asm(text)
        if val:
            return val, f"asm {field}"
        if val == 0:
            return None, (f"UNAVAILABLE -- the asm's {field} states 0, which on a kernel whose "
                          f"shared memory is sized at launch means 'not stated at compile "
                          f"time', not 'no LDS'. Pass --champion-lds/--anchor-lds.")
        return None, f"UNAVAILABLE -- {field}; pass --champion-lds/--anchor-lds"
    lds_asm = _first_int(_LDS_RE, text)
    if lds_asm:
        return lds_asm, ("asm `; LDSByteSize` (fallback reader -- amd_occupancy was not "
                         "importable, so `.amdhsa_group_segment_fixed_size` was not consulted)")
    return None, ("UNAVAILABLE -- the asm field is a structural 0 on Triton kernels; pass "
                  "--champion-lds/--anchor-lds from the cache metadata's `shared`")


def occupancy_terms(text: str, lds_bytes: int | None = None,
                    workgroup_size: int | None = None, arch: str | None = None) -> dict:
    """Both occupancy terms for ONE arm, the binding value, and the name of the term that bound.

    `waves_per_simd` is min(register_term, lds_term) over whichever terms exist. When only one
    exists it is that one, labelled so the reader knows the min was not actually taken; when
    neither exists it is None. It is never a number with an unstated provenance.
    """
    out = {"arch": None, "arch_source": None,
           "register_term": None, "register_source": None,
           "lds_term": None, "lds_detail": None,
           "lds_bytes": None, "lds_source": None,
           "waves_per_simd": None, "bound_by": None, "why": None}
    o = _occ_module()
    # arch_from_asm already falls back from `.amdgcn_target` to any bare gfx token, so a None
    # here means the dump genuinely names no target -- not that the parse was too strict. An
    # operator may STATE the arch (--*-arch); what this will not do is pick one.
    arch_src = "caller (--*-arch)"
    if arch is None:
        arch = o.arch_from_asm(text) if o is not None else None
        arch_src = ".amdgcn_target / amdhsa.target" if arch else "NO TARGET NAMED in this dump"
    out["arch"], out["arch_source"] = arch, arch_src

    # --- register term: LLVM's own number first, the model only as a fallback ---
    occ = _first_int(_OCC_RE, text)
    if occ:
        out["register_term"], out["register_source"] = occ, "LLVM `; Occupancy:`"
    else:
        nfv = _first_int(_NEXTFREE_RE, text)
        if o is not None and arch and nfv:
            w, label = o.waves_by_vgpr(nfv, arch)
            if w:
                out["register_term"] = w
                out["register_source"] = f"amd_occupancy.waves_by_vgpr({nfv}, {arch}) [{label}]"
        if out["register_term"] is None:
            out["register_source"] = ("no `; Occupancy:` and no (arch, .amdhsa_next_free_vgpr) "
                                      "pair to derive one from")

    # --- LDS term: three inputs, each refused rather than defaulted when missing ---
    lds_b, lds_src = lds_facts(text, lds_bytes)
    out["lds_bytes"], out["lds_source"] = lds_b, lds_src
    if o is None:
        out["lds_detail"] = _OCC_MOD_WHY
    elif lds_b is None:
        out["lds_detail"] = f"no LDS bytes/workgroup: {lds_src}"
    elif not arch:
        out["lds_detail"] = (
            "this dump names no .amdgcn_target, so the LDS/CU divisor is unknown and NO LDS "
            "term is reported. It is not defaulted: gfx942 is 64 KiB/CU and gfx950 is 160 "
            "KiB/CU, so a guess is wrong by 2.5x inside a min() -- which would not look wrong, "
            "it would look like a verdict. Re-dump with the target, or pass --*-arch.")
    else:
        wgs = workgroup_size
        wgs_src = "caller (--*-workgroup-size)"
        if wgs is None:
            wgs, wgs_src = o.workgroup_size_from_asm(text)
        if not wgs:
            out["lds_detail"] = f"no workgroup size ({wgs_src}); wg/CU cannot become waves/SIMD"
        else:
            w, detail = o.waves_by_lds(lds_b, arch, wgs)
            out["lds_term"] = w
            out["lds_detail"] = f"{detail} [workgroup size from {wgs_src}]"

    # --- the min, and the name of whichever term produced it ---
    r, l = out["register_term"], out["lds_term"]
    if r is not None and l is not None:
        out["waves_per_simd"] = min(r, l)
        out["bound_by"] = ("LDS" if l < r else "register" if r < l
                           else "register and LDS (tie)")
    elif r is not None:
        out["waves_per_simd"], out["bound_by"] = r, "register (LDS term NOT computed)"
        out["why"] = out["lds_detail"]
    elif l is not None:
        out["waves_per_simd"], out["bound_by"] = l, "LDS (register term NOT computed)"
        out["why"] = out["register_source"]
    else:
        out["why"] = f"register: {out['register_source']}; LDS: {out['lds_detail']}"
    return out


def _hist(text: str, mnemonic_rx: str, widths: list[str]) -> dict[str, int]:
    """Count occurrences of each width suffix for the given mnemonic family."""
    out = {}
    for w in widths:
        n = len(re.findall(rf"\b{mnemonic_rx}[a-z0-9_]*{re.escape(w)}\b", text))
        if n:
            out[w] = n
    return out


def _weighted_rank(hist: dict[str, int], widths: list[str]) -> float | None:
    """Mean narrowness rank, 0 = all widest. A rise means the access got narrower."""
    total = sum(hist.values())
    if not total:
        return None
    return sum(widths.index(w) * n for w, n in hist.items()) / total


_DS_READ_RE = re.compile(r"^\s*(ds_read[a-z0-9_]*)\s+(?:v\[[\d:]+\]|v\d+)\s*,\s*(v\d+)")
_VALU_DEF_RE = re.compile(r"^\s*(v_[a-z0-9_]+)\s+(v\d+)\s*,")


def ds_addr_pressure(text: str) -> dict:
    """The address-register signature of the LDS read burst.

    This is the discriminator a layout-equivalence check structurally cannot make. A champion
    issuing its reads from N precomputed address registers has N mutually independent reads the
    hardware coalesces across banks. The same reads issued from a handful of registers, with the
    address rematerialized into one of them immediately before each read, are serialized by a
    WAR hazard on that register -- a dependency chain synthesised by the register allocator, not
    by the layout. Equivalent layouts, equal `ds_read` COUNTS, unequal address pressure.

    `remat` is the load-bearing number and it is measured locally (does the instruction
    immediately above this read define its address register?), so it does not need the hot loop
    to be delimited first. `distinct` and `max_per_reg` are whole-file and therefore diluted by
    prologue/epilogue code -- reported, but not decided on.
    """
    lines = [ln for ln in text.splitlines()
             if ln.strip() and not ln.lstrip().startswith((";", ".", "//"))]
    remat = 0
    per_reg: dict[str, int] = {}
    for i, ln in enumerate(lines):
        m = _DS_READ_RE.match(ln)
        if not m:
            continue
        addr = m.group(2)
        per_reg[addr] = per_reg.get(addr, 0) + 1
        for back in (1, 2):                     # the VALU write may sit 1-2 slots above
            if i - back < 0:
                break
            d = _VALU_DEF_RE.match(lines[i - back])
            if d and d.group(2) == addr:
                remat += 1
                break
    return {
        "n_reads": sum(per_reg.values()),
        "distinct": len(per_reg),
        "max_per_reg": max(per_reg.values()) if per_reg else 0,
        "remat": remat,
    }


def _count(text: str, pat: str) -> int:
    return len(re.findall(pat, text))


def asm_facts(text: str, lds_bytes: int | None = None,
              workgroup_size: int | None = None, arch: str | None = None) -> dict:
    occ = occupancy_terms(text, lds_bytes, workgroup_size, arch)
    return {
        # the occupancy-relevant register count, and the arch-only one it is often confused with
        "next_free_vgpr": _first_int(_NEXTFREE_RE, text),
        "arch_vgpr": _first_int(_ARCHVGPR_RE, text),
        "accum_offset": _first_int(_ACCUM_RE, text),
        "arch": occ["arch"],
        # THE BINDING occupancy: min(register term, LDS term), plus the terms it was taken over
        # and the name of the one that bound. Never a bare `; Occupancy:` -- see occupancy_terms.
        "occupancy_waves_per_simd": occ["waves_per_simd"],
        "occupancy_register_term": occ["register_term"],
        "occupancy_lds_term": occ["lds_term"],
        "occupancy_bound_by": occ["bound_by"],
        "occupancy_terms_detail": {k: occ[k] for k in
                                   ("arch_source", "register_source", "lds_detail", "why")},
        "spill_bytes": _first_int(_SPILL_RE, text, 0),
        # None, not 0: "does not say" and "says zero" are both refusals, never a divisor
        "lds_bytes": occ["lds_bytes"],
        "lds_source": occ["lds_source"],
        "global_load_hist": _hist(text, r"(?:buffer|global|flat)_load_", GLOBAL_WIDTHS),
        "ds_read_hist": _hist(text, r"ds_read", LDS_WIDTHS),
        "ds_write_hist": _hist(text, r"ds_write", LDS_WIDTHS),
        "ds_addr": ds_addr_pressure(text),
        "n_ds_read": _count(text, r"\bds_read"),
        "n_ds_write": _count(text, r"\bds_write"),
        "n_mfma": _count(text, r"\bv_mfma"),
        "n_barrier": _count(text, r"\bs_barrier\b"),
        "n_valu": _count(text, r"\bv_(?!mfma)[a-z]"),
    }


def ttgir_pipeline_facts(text: str) -> dict:
    """Only the signals that actually evidence a software pipeline.

    `iter_args` is deliberately absent: reading `max iter_args >= 2` as pipelining is a
    confident false positive on every accumulator loop, including any online-softmax kernel.
    """
    num_stages = None
    m = re.search(r"tt\.num_stages\s*=\s*(\d+)", text) or \
        re.search(r"num_stages\s*=\s*(\d+)\s*:", text)
    if m:
        num_stages = int(m.group(1))
    return {
        "num_stages": num_stages,
        "memdesc_index": _count(text, r"\bttg\.memdesc_index\b"),
        "local_store": _count(text, r"\bttg\.local_store\b"),
        "local_alloc": _count(text, r"\bttg\.local_alloc\b"),
        "async_copy": _count(text, r"async_copy|async_commit|AsyncCopy"),
        "barrier": _count(text, r"\bttg\.barrier\b|\bgpu\.barrier\b"),
    }


def _is_pipelined(f: dict) -> bool:
    return bool(f["memdesc_index"] or f["local_store"] or f["async_copy"]
                or (f["num_stages"] or 1) > 1)


def _occupancy_evidence(champ: dict, anch: dict) -> dict:
    """Per-arm occupancy terms: what bound each arm, and what the min was taken over.

    Recorded on BOTH suspects, because the term that binds is what routes the owner. An
    LDS-bound drop is a layout debt and a register-bound drop is an allocation debt; two
    `; Occupancy:` values cannot tell them apart, so handing the author "occupancy dropped"
    without the term sends them to the register layer for a shared-memory problem half the time.
    """
    oc, oa = champ["occupancy_waves_per_simd"], anch["occupancy_waves_per_simd"]
    ev = {
        "waves_per_simd": {"champion": oc, "anchor": oa},
        "bound_by": {"champion": champ["occupancy_bound_by"],
                     "anchor": anch["occupancy_bound_by"]},
        "register_term": {"champion": champ["occupancy_register_term"],
                          "anchor": anch["occupancy_register_term"]},
        "lds_term": {"champion": champ["occupancy_lds_term"],
                     "anchor": anch["occupancy_lds_term"]},
        "detail": {"champion": champ["occupancy_terms_detail"],
                   "anchor": anch["occupancy_terms_detail"]},
        "dropped": bool(oc and oa and oa < oc),
        "anchor_bound_by": anch["occupancy_bound_by"] or "",
    }
    if champ["occupancy_lds_term"] is None and anch["occupancy_lds_term"] is None:
        ev["warning"] = (
            "NEITHER arm has an LDS term, so `occupancy` is the REGISTER term on both sides and "
            "this comparison is register-vs-register. An anchor that lost a wave to a "
            "materialized staging buffer -- the most common lost_layout mechanism in a port -- "
            "moves no register count and is INVISIBLE in this state. Supply --champion-lds/"
            "--anchor-lds, and a dump that names its target, before reading an unchanged "
            "occupancy as evidence of anything.")
    return ev


def attribute(champ_asm: dict | None, anch_asm: dict | None,
              champ_ttgir: dict | None, anch_ttgir: dict | None) -> list[dict]:
    """One verdict per suspect: suspected / cleared / unknown, each with its numbers."""
    out = []

    # --- lost_pipeline ---
    if champ_ttgir and anch_ttgir:
        cp, ap = _is_pipelined(champ_ttgir), _is_pipelined(anch_ttgir)
        if cp and not ap:
            out.append({"suspect": "lost_pipeline", "verdict": "SUSPECTED",
                        "owned_by": "pipeline layer -- author the overlap by hand first "
                                    "(register prefetch -> LDS ring -> warp_pipeline_stage); "
                                    "re-injecting plain's pipeliner is the last resort "
                                    "(diagnostic / parity fallback, numbers labelled injected)",
                        "evidence": {"champion": champ_ttgir, "anchor": anch_ttgir}})
        else:
            why = ("the champion's own loop is not pipelined either (memdesc_index=%d, "
                   "local_store=%d, num_stages=%s), so there is no pipeline to lose"
                   % (champ_ttgir["memdesc_index"], champ_ttgir["local_store"],
                      champ_ttgir["num_stages"]))
            out.append({"suspect": "lost_pipeline", "verdict": "CLEARED",
                        "evidence": {"reason": why if not cp else "the anchor is pipelined too",
                                     "champion": champ_ttgir, "anchor": anch_ttgir}})
    else:
        out.append({"suspect": "lost_pipeline", "verdict": "UNKNOWN",
                    "evidence": {"reason": "pass --champion-ttgir and --anchor-ttgir; this "
                                           "suspect cannot be judged from the .amdgcn alone"}})

    if not (champ_asm and anch_asm):
        for s in ("lost_layout", "lost_RA"):
            out.append({"suspect": s, "verdict": "UNKNOWN",
                        "evidence": {"reason": "pass --champion-asm and --anchor-asm"}})
        return out

    # --- lost_layout: did any access family get NARROWER, or did shared cross a divisor? ---
    shifts, ev = [], {}
    for key, widths in (("global_load_hist", GLOBAL_WIDTHS),
                        ("ds_read_hist", LDS_WIDTHS),
                        ("ds_write_hist", LDS_WIDTHS)):
        rc, ra = _weighted_rank(champ_asm[key], widths), _weighted_rank(anch_asm[key], widths)
        ev[key] = {"champion": champ_asm[key], "anchor": anch_asm[key],
                   "narrowness_rank": {"champion": rc, "anchor": ra}}
        if rc is not None and ra is not None and ra > rc + 0.25:
            shifts.append(f"{key} narrowed (rank {rc:.2f} -> {ra:.2f})")
    lds_c, lds_a = champ_asm["lds_bytes"], anch_asm["lds_bytes"]
    if lds_c and lds_a and lds_a > lds_c:
        shifts.append(f"shared bytes/WG grew {lds_c} -> {lds_a}")
    ev["lds_bytes"] = {"champion": lds_c, "anchor": lds_a,
                       "source": {"champion": champ_asm["lds_source"],
                                  "anchor": anch_asm["lds_source"]}}
    if lds_c is None or lds_a is None:
        ev["lds_bytes"]["warning"] = (
            "LDS/WG unavailable, so the 'a materialized staging buffer cost an occupancy step' "
            "half of this suspect was NOT tested. That is the single most common lost_layout "
            "mechanism in a port -- pass --champion-lds/--anchor-lds before trusting a CLEARED.")
    # The occupancy terms belong to this suspect too: when the anchor's BINDING term is the LDS
    # one, the wave was lost to shared memory and the debt is owned here, not by the allocator.
    occ_ev = _occupancy_evidence(champ_asm, anch_asm)
    ev["occupancy"] = occ_ev
    if occ_ev["dropped"] and occ_ev["anchor_bound_by"].startswith("LDS"):
        shifts.append(
            f"occupancy fell {occ_ev['waves_per_simd']['champion']} -> "
            f"{occ_ev['waves_per_simd']['anchor']} waves/SIMD and the ANCHOR is LDS-BOUND "
            f"(LDS term {occ_ev['lds_term']['anchor']} vs register term "
            f"{occ_ev['register_term']['anchor']}): the wave went to shared memory, not to the "
            f"register allocator")
    out.append({"suspect": "lost_layout",
                "verdict": "SUSPECTED" if shifts else "CLEARED",
                "owned_by": "memory path / shared layout -- classify each ttg.local_alloc "
                            "(staged -> allocate_shared_memory; pass-through -> convert_layout, "
                            "buffer stays compiler-owned) before re-writing it",
                "evidence": {"shifts": shifts, **ev}})

    # --- lost_RA: same work, worse allocation ---
    reasons = {}
    inst_c = champ_asm["n_valu"] + champ_asm["n_mfma"] + champ_asm["n_ds_read"] + champ_asm["n_ds_write"]
    inst_a = anch_asm["n_valu"] + anch_asm["n_mfma"] + anch_asm["n_ds_read"] + anch_asm["n_ds_write"]
    multiset_same = inst_c and abs(inst_a - inst_c) / inst_c <= 0.05
    if anch_asm["spill_bytes"] and not champ_asm["spill_bytes"]:
        reasons["spill_appeared"] = anch_asm["spill_bytes"]
    vc, va = champ_asm["next_free_vgpr"], anch_asm["next_free_vgpr"]
    if vc and va and va > vc:
        reasons["next_free_vgpr_rose"] = {"champion": vc, "anchor": va}
        # the 256 cliff: ArchVGPR+AGPR share ONE 512/SIMD file, so crossing it costs a wave
        if vc <= 256 < va:
            reasons["crossed_256_wave_threshold"] = True
    # Occupancy here is min(register term, LDS term) on each arm -- see occupancy_terms. The
    # drop is only THIS suspect's when the anchor's binding term is the register one; an
    # LDS-bound drop was raised on lost_layout above, and firing it here as well would hand the
    # author a register-allocator story about a shared-memory regression.
    oc, oa = occ_ev["waves_per_simd"]["champion"], occ_ev["waves_per_simd"]["anchor"]
    if occ_ev["dropped"] and not occ_ev["anchor_bound_by"].startswith("LDS"):
        reasons["occupancy_dropped"] = {
            "champion": oc, "anchor": oa,
            "mechanism": (f"binding occupancy {oc} -> {oa} waves/SIMD, anchor bound by its "
                          f"{occ_ev['anchor_bound_by']} term (register "
                          f"{occ_ev['register_term']['anchor']}, LDS "
                          f"{occ_ev['lds_term']['anchor']}) -- an allocation debt, not a "
                          f"layout one")}
    ac, aa = champ_asm["ds_addr"], anch_asm["ds_addr"]
    # remat is the mechanism itself: an address recomputed into a register immediately before
    # the read that consumes it. A rise here IS the serial chain, whatever the counts say.
    if aa["remat"] > max(ac["remat"] * 1.5, ac["remat"] + 3):
        reasons["ds_address_rematerialization_rose"] = {
            "champion": ac["remat"], "anchor": aa["remat"],
            "mechanism": "each read is serialized behind a WAR hazard on an address register "
                         "recomputed immediately above it -- a dependency chain the allocator "
                         "synthesised. Read the ds_read OPERANDS, not just the count."}
    # `distinct` is a whole-file count, so it only means "the same reads lost their independent
    # addresses" when the reads ARE the same. On an anchor whose instruction stream genuinely
    # changed (u16 reads folded into b128, say) a drop here is arithmetic, not a regression --
    # firing on it would hand the author a serialization story about a kernel that got wider.
    if multiset_same and ac["distinct"] and aa["distinct"] < ac["distinct"] * 0.75:
        reasons["ds_address_registers_collapsed"] = {
            "champion": ac["distinct"], "anchor": aa["distinct"],
            "mechanism": "fewer distinct address registers feed the same reads, so they can no "
                         "longer issue independently. Whole-file count -- confirm in the hot loop."}
    ev_addr = {"champion": ac, "anchor": aa}
    out.append({"suspect": "lost_RA",
                "verdict": "SUSPECTED" if reasons else "CLEARED",
                "owned_by": "register/slicing layer -- and note a lower LDS instruction COUNT "
                            "can still be slower; the address-dependency chain is what binds",
                "evidence": {"instruction_multiset_unchanged": bool(multiset_same),
                             "instructions": {"champion": inst_c, "anchor": inst_a},
                             "ds_addr": ev_addr,
                             "occupancy": occ_ev,
                             **reasons}})
    return out


def evaluate(champion_ms: float, anchor_ms: float, threshold: float, suspects: list[dict]) -> dict:
    # A non-finite input is a broken measurement, not a fast one. `inf / 1.0` is `inf`, which
    # is `>= threshold`, so an unpopulated field or a divide-by-zero upstream used to CLEAR the
    # gate outright -- the one direction a gate must never fail in. NaN happens to fall the
    # safe way (`nan >= x` is False) but is refused here too, so neither depends on luck.
    bad = {k: v for k, v in (("champion_ms", champion_ms), ("anchor_ms", anchor_ms))
           if not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0}
    if bad:
        return {
            "champion_ms": champion_ms, "anchor_ms": anchor_ms,
            "ratio_champion_over_anchor": None, "threshold": threshold,
            "gate": "INVALID INPUT", "suspects": suspects, "suspected": [], "unattributed": [],
            "invalid": bad, "round_outcome_allowed": "none",
            "note": ("A timing that is not a finite positive number is a broken measurement, "
                     f"not a result: {bad}. This gate refuses rather than ranking it -- an "
                     "infinity clears any threshold, and a zero or a negative makes the ratio "
                     "meaningless. Re-measure before re-running."),
        }
    ratio = champion_ms / anchor_ms
    cleared = ratio >= threshold
    named = [s["suspect"] for s in suspects if s["verdict"] == "SUSPECTED"]
    unknown = [s["suspect"] for s in suspects if s["verdict"] == "UNKNOWN"]
    # Whether ANY compiled artifact backed this call. With none, the verdict rests on two
    # free-floating floats that nothing ties to a champion, a build or a box -- true as
    # arithmetic, unfalsifiable as evidence. It is reported rather than silently folded into
    # a PASS, because a CLEARED with no provenance is the shape a fabricated parity claim has.
    provenance = "artifact-backed" if len(unknown) < len(suspects) else "TIMINGS ONLY"
    res = {
        "champion_ms": champion_ms, "anchor_ms": anchor_ms,
        "ratio_champion_over_anchor": round(ratio, 4),
        "threshold": threshold,
        "gate": "CLEARED" if cleared else "NOT CLEARED",
        "provenance": provenance,
        "suspects": suspects,
        "suspected": named,
        "unattributed": unknown,
    }
    if cleared:
        res["round_outcome_allowed"] = "win"
        res["note"] = ("The debt is paid (or was never taken). Climbing is allowed, and a round "
                       "may be scored as a win. Reaching parity is NOT itself a win: it is "
                       "getting back to a number the front end already measured.")
        if provenance != "artifact-backed":
            res["round_outcome_allowed"] = "win (UNVERIFIED PROVENANCE)"
            res["note"] += (
                " PROVENANCE: this call passed no .amdgcn and no .ttgir, so the verdict is two "
                "numbers you typed. Nothing here checked that they came from the asserted "
                "champion, from this box, or from a build that still exists. Quote it with that "
                "caveat, or re-run with --champion-asm/--anchor-asm (and --*-ttgir) so the "
                "CLEARED is backed by the artifacts it claims to be about.")
        if ratio > 1.0:
            res["note"] += (" The anchor is FASTER than the champion -- attribute that too "
                            "rather than pocketing it; it is usually compiler-owned staging "
                            "buying an occupancy step, and knowing which one it is tells you "
                            "what the remaining levers can and cannot touch.")
    else:
        res["round_outcome_allowed"] = "recovery"
        res["note"] = (
            "DO NOT CLIMB YET. Until champion_ms/current_ms >= %.2f a round's outcome is "
            "`recovery` against the suspect it closed, never a win. Close the suspects above "
            "with the layer that OWNS each one, re-run this gate, and only then start the "
            "layer loop. Climbing from here caps the port: the best lever you find will be "
            "quoted against a broken anchor and the run will close below the champion while "
            "reporting a gain." % threshold)
        if not named:
            res["note"] += (" No suspect fired, which is itself a finding: pass the artifacts "
                            "(--champion-asm/--anchor-asm/--*-ttgir) if you have not, and if "
                            "they are all CLEARED then the gap is not in this taxonomy -- "
                            "profile it before authoring anything.")
    return res


def _fmt(res: dict) -> str:
    if res["gate"] == "INVALID INPUT":
        return (f"=== parity gate: INVALID INPUT {res['invalid']} ===\n"
                f"  round outcome allowed: {res['round_outcome_allowed']}\n\n  {res['note']}")
    L = [(f"=== parity gate: champion {res['champion_ms']:.4f} ms"
          f" / anchor {res['anchor_ms']:.4f} ms = {res['ratio_champion_over_anchor']}"
          f"  (threshold {res['threshold']}) ==="),
         (f"  {res['gate']}   round outcome allowed: {res['round_outcome_allowed']}"
          f"   provenance: {res['provenance']}")]
    for s in res["suspects"]:
        L.append(f"  [{s['verdict']:9s}] {s['suspect']}")
        ev = s.get("evidence", {})
        for k in ("shifts", "reason"):
            if ev.get(k):
                L.append(f"       {k}: {ev[k]}")
        for k, v in ev.items():
            if k in ("shifts", "reason", "champion", "anchor"):
                continue
            if isinstance(v, dict) and "mechanism" in v:
                L.append(f"       {k}: {v['champion']} -> {v['anchor']}")
                L.append(f"           {v['mechanism']}")
            elif not isinstance(v, dict):
                L.append(f"       {k}: {v}")
        if s["verdict"] == "SUSPECTED" and s.get("owned_by"):
            L.append(f"       -> owned by: {s['owned_by']}")
    L.append("")
    L.append(f"  {res['note']}")
    return "\n".join(L)


def _selftest() -> int:
    # Fixtures shaped like a real .amdgcn: the register number that matters is
    # .amdhsa_next_free_vgpr (ArchVGPR+AGPR), NOT the arch-only num_vgpr beside it.
    champ = """
		.amdhsa_next_free_vgpr 257
	.set k.num_vgpr, 248
; Occupancy: 1
	buffer_load_dwordx4 v[10:13], v4, s[0:3], 0 offen
	buffer_load_dwordx4 v[14:17], v5, s[0:3], 0 offen
	ds_read2_b64 v[34:37], v139 offset1:16
	ds_read2_b64 v[38:41], v141 offset1:16
	ds_read2_b64 v[42:45], v143 offset1:16
	ds_read2_b64 v[46:49], v146 offset1:16
	ds_write_b128 v200, v[10:13]
	v_mfma_f32_16x16x16_bf16 a[0:3], v[10:11], v[12:13], a[0:3]
	s_barrier
    """
    # the anchor rematerializes every address into ONE register right before its read
    anch = """
		.amdhsa_next_free_vgpr 202
	.set k.num_vgpr, 196
; Occupancy: 2
	buffer_load_dwordx4 v[10:13], v4, s[0:3], 0 offen
	buffer_load_dwordx4 v[14:17], v5, s[0:3], 0 offen
	v_add_u32_e32 v33, 0x2000, v139
	ds_read2_b64 v[34:37], v33 offset1:16
	v_add_u32_e32 v33, 0x2000, v141
	ds_read2_b64 v[38:41], v33 offset1:16
	v_add_u32_e32 v33, 0x2000, v143
	ds_read2_b64 v[42:45], v33 offset1:16
	v_add_u32_e32 v33, 0x2000, v146
	ds_read2_b64 v[46:49], v33 offset1:16
	ds_write_b128 v200, v[10:13]
	v_mfma_f32_16x16x16_bf16 a[0:3], v[10:11], v[12:13], a[0:3]
	s_barrier
    """
    cf, af = asm_facts(champ, 32768), asm_facts(anch, 16384)
    # next_free_vgpr is read, and is NOT confused with the arch-only count beside it
    assert cf["next_free_vgpr"] == 257 and cf["arch_vgpr"] == 248, cf
    assert af["next_free_vgpr"] == 202, af
    # the discriminator: 4 independent address regs, 0 remat -> 1 shared reg, 4 remats
    assert cf["ds_addr"]["distinct"] == 4 and cf["ds_addr"]["remat"] == 0, cf["ds_addr"]
    assert af["ds_addr"]["remat"] == 4, af["ds_addr"]
    # the asm LDS field is a structural 0 -> reported as UNAVAILABLE, never as 0
    assert asm_facts(anch)["lds_bytes"] is None, asm_facts(anch)
    assert "UNAVAILABLE" in asm_facts(anch)["lds_source"]

    # 1. an anchor FASTER than the champion clears, and is told to attribute the gain anyway
    res = evaluate(2.8966, 2.2255, 0.95, attribute(cf, af, None, None))
    assert res["gate"] == "CLEARED" and res["round_outcome_allowed"] == "win", res
    assert "FASTER than the champion" in res["note"], res["note"]

    # 2. the pa_decode-shaped failure: 0.71x, and lost_RA must fire on the address-reg collapse
    res = evaluate(2.8966, 4.0754, 0.95, attribute(cf, af, None, None))
    assert res["gate"] == "NOT CLEARED", res
    assert res["round_outcome_allowed"] == "recovery", res
    ra = next(s for s in res["suspects"] if s["suspect"] == "lost_RA")
    assert ra["verdict"] == "SUSPECTED", ra
    assert "ds_address_rematerialization_rose" in ra["evidence"], ra

    # 3. lost_layout fires when a load narrows dwordx4 -> ushort (a conversion folded into it)
    narrowed = anch.replace("buffer_load_dwordx4 v[10:13]", "buffer_load_ushort v10") \
                   .replace("buffer_load_dwordx4 v[14:17]", "buffer_load_ushort v14")
    res = evaluate(6.9035, 13.61, 0.95, attribute(cf, asm_facts(narrowed, 16384), None, None))
    lay = next(s for s in res["suspects"] if s["suspect"] == "lost_layout")
    assert lay["verdict"] == "SUSPECTED" and lay["evidence"]["shifts"], lay

    # 4. THE FALSE POSITIVE THIS TOOL MUST NOT MAKE: an accumulator loop with 3 iter_args and
    #    num_stages=1 is NOT pipelined, so lost_pipeline must be CLEARED, not suspected.
    champ_ttgir = """
      tt.num_stages = 1
      %0 = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %z, %p1 = %a, %p2 = %b) {
        %l = amdg.buffer_load %ptr[%off] : tensor<128x128xbf16>
      }
    """
    anch_ttgir = "%0 = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %z) { }"
    ct, at = ttgir_pipeline_facts(champ_ttgir), ttgir_pipeline_facts(anch_ttgir)
    assert ct["memdesc_index"] == 0 and ct["local_store"] == 0 and ct["num_stages"] == 1, ct
    res = evaluate(2.8966, 4.0754, 0.95, attribute(cf, af, ct, at))
    pipe = next(s for s in res["suspects"] if s["suspect"] == "lost_pipeline")
    assert pipe["verdict"] == "CLEARED", pipe
    assert "no pipeline to lose" in pipe["evidence"]["reason"], pipe

    # 5. a genuinely pipelined champion against a flat anchor DOES fire
    piped = "tt.num_stages = 2\n ttg.memdesc_index %x\n ttg.local_store %y\n"
    res = evaluate(2.0, 4.0, 0.95, attribute(cf, af, ttgir_pipeline_facts(piped), at))
    pipe = next(s for s in res["suspects"] if s["suspect"] == "lost_pipeline")
    assert pipe["verdict"] == "SUSPECTED", pipe

    # 6. missing artifacts are UNKNOWN, never CLEARED -- a dark signal is not a clean one
    res = evaluate(2.0, 4.0, 0.95, attribute(None, None, None, None))
    assert res["unattributed"] == ["lost_pipeline", "lost_layout", "lost_RA"], res
    assert "No suspect fired" in res["note"], res["note"]

    # 7. A NON-FINITE timing must not clear the gate. `inf / 1.0 >= 0.95` is True, so an
    #    unpopulated field used to CLEAR outright -- the one direction a gate cannot fail in.
    for bad_c, bad_a, why in ((float("inf"), 1.0, "inf champion"),
                              (float("nan"), 1.0, "nan champion"),
                              (1.0, 0.0, "zero anchor"),
                              (-1.0, 1.0, "negative champion")):
        r = evaluate(bad_c, bad_a, 0.95, attribute(None, None, None, None))
        assert r["gate"] == "INVALID INPUT", (why, r["gate"])
        assert r["round_outcome_allowed"] == "none", (why, r)
    assert _fmt(evaluate(float("inf"), 1.0, 0.95, [])).startswith("=== parity gate: INVALID")

    # 8. a CLEARED reached with no artifact at all says so. It stays exit-0 (the ratio really
    #    does meet the threshold) but must never read as an artifact-backed verdict.
    r = evaluate(2.0, 1.0, 0.95, attribute(None, None, None, None))
    assert r["gate"] == "CLEARED" and r["provenance"] == "TIMINGS ONLY", r
    assert "UNVERIFIED PROVENANCE" in r["round_outcome_allowed"], r
    assert "two numbers you typed" in r["note"], r["note"]
    r = evaluate(2.0, 1.0, 0.95, attribute(cf, af, None, None))
    assert r["provenance"] == "artifact-backed", r
    assert "UNVERIFIED" not in r["round_outcome_allowed"], r

    # --- 9-13: occupancy is min(register, LDS), and every missing input is REFUSED ---------
    #
    # 9. The fixtures above name no target, so there is NO LDS term and the gate says so. The
    #    occupancy it reports is the register term, explicitly labelled -- not silently passed
    #    off as the kernel's occupancy, and not computed against a guessed LDS/CU.
    assert cf["arch"] is None and cf["occupancy_lds_term"] is None, cf
    assert cf["occupancy_waves_per_simd"] == 1 and cf["occupancy_register_term"] == 1, cf
    assert cf["occupancy_bound_by"] == "register (LDS term NOT computed)", cf
    assert "no .amdgcn_target" in (cf["occupancy_terms_detail"]["why"] or ""), cf
    ra9 = next(s for s in attribute(cf, af, None, None) if s["suspect"] == "lost_RA")
    assert "NEITHER arm has an LDS term" in ra9["evidence"]["occupancy"]["warning"], ra9

    # 10. THE 2.5x HAZARD: an untargeted dump must not borrow gfx942's 64 KiB/CU. Adding LDS
    #     bytes alone changes nothing, because the divisor is still unknown -- refusal here is
    #     the whole point. A default would not read as an error, it would read as a verdict.
    t10 = occupancy_terms(champ, lds_bytes=32768, workgroup_size=256)
    assert t10["lds_term"] is None and t10["arch"] is None, t10
    assert "2.5x" in t10["lds_detail"] and "gfx950" in t10["lds_detail"], t10

    # 11. Name the arch and the LDS term appears -- and when it BINDS, the min takes it. 160
    #     KiB/CU on gfx950 // 65536 B/wg = 2 wg/CU, x 4 waves/wg / 4 SIMD = 2 waves/SIMD, under
    #     a `; Occupancy: 8`. The register-only reading would have reported 8.
    g950 = '\t.amdgcn_target "amdgcn-amd-amdhsa--gfx950"\n; Occupancy: 8\n'
    t11 = occupancy_terms(g950, lds_bytes=65536, workgroup_size=256)
    if t11["lds_term"] is not None:          # needs perf_knowledge/hardware/data/hw_constants.json
        assert t11["register_term"] == 8, t11
        assert t11["waves_per_simd"] == t11["lds_term"] < 8, t11
        assert t11["bound_by"] == "LDS", t11

        # 12. An LDS-bound drop is routed to lost_layout, NOT to the register allocator. Same
        #     `; Occupancy: 8` on both arms: register-vs-register sees nothing at all here.
        c12 = asm_facts(g950 + champ, 16384, 256)
        a12 = asm_facts(g950 + anch, 65536, 256)
        assert c12["occupancy_register_term"] == a12["occupancy_register_term"] == 8, (c12, a12)
        assert a12["occupancy_waves_per_simd"] < c12["occupancy_waves_per_simd"], (c12, a12)
        sus12 = attribute(c12, a12, None, None)
        lay12 = next(s for s in sus12 if s["suspect"] == "lost_layout")
        assert lay12["verdict"] == "SUSPECTED", lay12
        assert any("LDS-BOUND" in s for s in lay12["evidence"]["shifts"]), lay12["evidence"]
        ra12 = next(s for s in sus12 if s["suspect"] == "lost_RA")
        assert "occupancy_dropped" not in ra12["evidence"], ra12["evidence"]
        assert _fmt(evaluate(2.0, 4.0, 0.95, sus12))            # renders without raising

    # 13. A group-segment field that STATES 0 is a refusal, not a zero. Both spellings of
    #     "this dump does not give me the number" must land on None, or the 0 becomes a
    #     divisor and `waves_by_lds` is asked to divide an LDS budget by nothing.
    kd0 = '\t.amdgcn_target "amdgcn-amd-amdhsa--gfx950"\n\t.amdhsa_group_segment_fixed_size 0\n'
    b13, s13 = lds_facts(kd0)
    assert b13 is None and s13.startswith("UNAVAILABLE"), (b13, s13)
    assert "not 'no LDS'" in s13, s13
    assert occupancy_terms(kd0)["lds_term"] is None, occupancy_terms(kd0)
    b13b, s13b = lds_facts('\t.amdhsa_group_segment_fixed_size 38144\n')
    assert b13b == 38144 and "group_segment_fixed_size" in s13b, (b13b, s13b)

    print("[parity_gate] SELFTEST PASS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--champion-ms", type=float)
    ap.add_argument("--anchor-ms", type=float, help="the CURRENT Gluon number, not only round 1's")
    ap.add_argument("--champion-asm")
    ap.add_argument("--anchor-asm")
    ap.add_argument("--champion-ttgir")
    ap.add_argument("--anchor-ttgir")
    ap.add_argument("--champion-lds", type=int,
                    help="shared bytes/WG from the Triton cache metadata's `shared` field. The "
                         "asm's LDSByteSize is a structural 0 on Triton kernels, so without this "
                         "the LDS half of lost_layout is untested.")
    ap.add_argument("--anchor-lds", type=int)
    # The LDS term needs an arch (for LDS/CU and SIMDs/CU) and a workgroup size on top of the
    # bytes. Each is read from the dump when the dump states it, and REFUSED -- not guessed --
    # when it does not. These two flags exist so a refusal has a remedy: state the fact, or go
    # without the LDS term. Neither has a default, because a wrong LDS/CU divisor inside a
    # min() does not look wrong, it looks like a verdict.
    ap.add_argument("--champion-arch",
                    help="gfx target, when the dump names none (a stripped .hsaco objdump). "
                         "Without it there is no LDS term -- it is never guessed, because "
                         "gfx942 is 64 KiB LDS/CU and gfx950 is 160 KiB: a 2.5x wrong divisor.")
    ap.add_argument("--anchor-arch")
    ap.add_argument("--champion-workgroup-size", type=int,
                    help="threads/workgroup AS LAUNCHED. Falls back to the asm's "
                         ".max_flat_workgroup_size, which is the compile-time BOUND: a kernel "
                         "launched below it has more workgroups per CU than that implies.")
    ap.add_argument("--anchor-workgroup-size", type=int)
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if a.champion_ms is None or a.anchor_ms is None:
        ap.error("--champion-ms and --anchor-ms are required (or use --selftest)")

    def _read(p):
        return Path(p).read_text(errors="ignore") if p else None

    ca, aa = _read(a.champion_asm), _read(a.anchor_asm)
    ct, at = _read(a.champion_ttgir), _read(a.anchor_ttgir)
    suspects = attribute(asm_facts(ca, a.champion_lds, a.champion_workgroup_size,
                                   a.champion_arch) if ca else None,
                         asm_facts(aa, a.anchor_lds, a.anchor_workgroup_size,
                                   a.anchor_arch) if aa else None,
                         ttgir_pipeline_facts(ct) if ct else None,
                         ttgir_pipeline_facts(at) if at else None)
    res = evaluate(a.champion_ms, a.anchor_ms, a.threshold, suspects)
    print(_fmt(res))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2))
        print(f"\nwrote {a.json}")
    return 0 if res["gate"] == "CLEARED" else 2


if __name__ == "__main__":
    sys.exit(main())
