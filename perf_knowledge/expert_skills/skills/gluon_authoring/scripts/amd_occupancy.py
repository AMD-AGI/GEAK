#!/usr/bin/env python3
"""Per-arch VGPR->waves/SIMD occupancy for AMD targets. One model, three consumers.

Why a shared module: the CDNA formula (`512 / round_up(vgpr, 8)`, ArchVGPR+AGPR sharing one
file) is NOT the RDNA formula, and it was hard-coded in `calc_perf.py occ`, the loop audits,
`hw_budget.py` and `probe.py`. On an RDNA target every one of them under-reports occupancy by
2-3x -- 249 VGPRs on gfx1201 is 5 waves/SIMD, not 1 or 2 -- which reads as "register-capped"
and sends a round chasing pressure that is not there. Three register-file geometries exist:

    CDNA  (gfx90a/942/950)          wave64  512 VGPR/SIMD  granule  8  cap  8   arch+AGPR COMBINED
    CDNA5 (gfx1250)                 wave32 1024 VGPR/SIMD  granule 16  cap 16
    RDNA  (gfx11*/gfx1151/gfx120*)  wave32 1536 VGPR/SIMD  granule 24  cap 16   no AGPR file

`waves = min(cap, file // (granule * ceil(vgpr / granule)))`.

PREFER `llvm_occupancy()`: LLVM already emits `; Occupancy: N` in the resource-usage comment
block of a `.s`, computed by the backend for that exact subtarget. When the dump has it, it is
the authority and this module's table is only the fallback for KD-less disassembly or for
planning a tile that has not been compiled yet.

Provenance of the tables (`basis: compiler-derived`, reproducible in seconds, independent of
any kernel or measurement):

    for v in $(seq 1 254); do
      clob=$(python3 -c "print(','.join('~{v%d}'%i for i in range($v)))")
      printf 'target triple = "amdgcn-amd-amdhsa"\\n
      define amdgpu_kernel void @k(ptr addrspace(1) %%o) #0 {\\n
        call void asm sideeffect "", "%s"()\\n store float 1.0, ptr addrspace(1) %%o\\n ret void\\n}\\n
      attributes #0 = { "amdgpu-flat-work-group-size"="256,256" }\\n' "$clob" > t.ll
      llc -mtriple=amdgcn-amd-amdhsa -mcpu=<arch> t.ll -o - | grep -E "NumVgprs|Occupancy"
    done

Verified against ROCm 7.2.1 / LLVM 22 for gfx90a, gfx942, gfx950, gfx1100, gfx1151, gfx1200,
gfx1201, gfx1250 -- every step below is a measured breakpoint of that sweep, not an estimate.

Usage:
  python3 amd_occupancy.py --vgpr 249 --arch gfx1201     # model lookup
  python3 amd_occupancy.py --asm kernel.s                # read the KD + LLVM's own answer
  python3 amd_occupancy.py --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# Register-file geometry per ISA family. Keep in sync with references/hardware/hw_constants.json
# (`--selftest` cross-checks the two when the reference tree is reachable).
MODELS = {
    "cdna": {
        # vgpr_per_wave is the ADDRESSABLE cap for one wave; vgpr_file_per_simd is what
        # occupancy divides into. They coincide at 512 on CDNA -- which is exactly why code
        # written on CDNA tends to use one where it means the other, and then reads RDNA wrong.
        "wave_size": 64, "vgpr_file_per_simd": 512, "vgpr_per_wave": 512,
        "vgpr_alloc_granule": 8,
        "max_waves_per_simd": 8, "combined_arch_accum": True,
        "label": "CDNA wave64: 512 VGPR/SIMD (ArchVGPR+AGPR combined), granule 8, cap 8",
    },
    "cdna5": {
        # per-wave cap left unset: not verified for this family here, and a guessed ceiling
        # would silently prune legal tiles. Callers treat a missing cap as "no clamp".
        "wave_size": 32, "vgpr_file_per_simd": 1024, "vgpr_per_wave": None,
        "vgpr_alloc_granule": 16,
        "max_waves_per_simd": 16, "combined_arch_accum": False,
        "label": "CDNA5 wave32: 1024 VGPR/SIMD, granule 16, cap 16",
    },
    "rdna": {
        "wave_size": 32, "vgpr_file_per_simd": 1536, "vgpr_per_wave": 256,
        "vgpr_alloc_granule": 24,
        "max_waves_per_simd": 16, "combined_arch_accum": False,
        "label": "RDNA wave32: 1536 VGPR/SIMD, granule 24, cap 16 (no AGPR file)",
    },
}

# gfx -> family. Prefix-matched longest-first, so a new gfx94x/gfx120x lands correctly without
# an edit; an UNKNOWN arch returns None rather than defaulting to CDNA (a wrong model that
# prints a confident number is worse than no number).
_FAMILY_PREFIXES = [
    ("gfx90a", "cdna"), ("gfx908", "cdna"), ("gfx94", "cdna"), ("gfx95", "cdna"),
    ("gfx125", "cdna5"),
    ("gfx10", "rdna"), ("gfx11", "rdna"), ("gfx12", "rdna"),
]

_ARCH_RE = re.compile(r"\bgfx(?:9[0-9a-f]{2}|1[0-9]{3})\b")
_TARGET_RE = re.compile(r"^\s*(?:\.amdgcn_target|amdhsa\.target:)\s*\"?[^\"]*?"
                        r"(gfx\w+)", re.M)
_WAVE32_RE = re.compile(r"^\s*\.amdhsa_wavefront_size32\s+1\s*$", re.M)
_LLVM_OCC_RE = re.compile(r"^;\s*Occupancy:\s*(\d+)\s*$", re.M)


def family_for(arch):
    """ISA family key for a gfx name, or None when the arch is unknown to this table."""
    if not arch:
        return None
    a = arch.lower()
    for prefix, fam in sorted(_FAMILY_PREFIXES, key=lambda kv: -len(kv[0])):
        if a.startswith(prefix):
            return fam
    return None


def model_for(arch):
    """Register-file geometry dict for a gfx name, or None if the arch is unknown."""
    fam = family_for(arch)
    return dict(MODELS[fam], family=fam) if fam else None


def waves_by_vgpr(vgpr, arch=None, model=None):
    """(waves_per_simd, model_label). waves is None when the arch is unknown -- callers must
    print the label and NOT substitute a CDNA number."""
    m = model or model_for(arch)
    if m is None:
        return None, (f"unknown arch {arch!r}: no VGPR-file model on record -- read LLVM's "
                      f"`; Occupancy:` from the .s, or add the arch to amd_occupancy.MODELS")
    if not vgpr:
        return m["max_waves_per_simd"], m["label"]
    gran = m["vgpr_alloc_granule"]
    alloc = ((int(vgpr) + gran - 1) // gran) * gran
    return min(m["max_waves_per_simd"], m["vgpr_file_per_simd"] // alloc), m["label"]


def arch_from_asm(text):
    """gfx name from an AMDGCN dump: `.amdgcn_target` / `amdhsa.target` first, then any bare
    gfx token. Returns None on a dump that names no target (objdump of a stripped .hsaco)."""
    m = _TARGET_RE.search(text)
    if m:
        return m.group(1).split(":")[0]
    m = _ARCH_RE.search(text)
    return m.group(0) if m else None


def wave32_from_asm(text):
    """True when the kernel descriptor declares wave32. Independent of the gfx name, so it also
    catches a wave32 build of an arch that supports both."""
    return bool(_WAVE32_RE.search(text))


def llvm_occupancy(text):
    """waves/SIMD from LLVM's own `; Occupancy: N` resource comment, or None if absent."""
    m = _LLVM_OCC_RE.search(text)
    return int(m.group(1)) if m else None


def occupancy_from_asm(text, vgpr=None):
    """(waves, source, label). source is 'llvm-comment' (authoritative) | 'model' | 'unknown'."""
    arch = arch_from_asm(text)
    occ = llvm_occupancy(text)
    m = model_for(arch)
    label = m["label"] if m else f"unknown arch {arch!r}"
    if occ is not None:
        return occ, "llvm-comment", f"{label}; LLVM `; Occupancy:` in this dump"
    waves, label = waves_by_vgpr(vgpr, arch=arch, model=m)
    return waves, ("model" if waves is not None else "unknown"), label


# --------------------------------------------------------------------------------- reference SoT
def _find_hw_constants():
    """hw_constants.json sits under references/hardware/ one or two levels above scripts/."""
    import glob
    for c in (os.path.join(HERE, "..", "references", "hardware", "hw_constants.json"),
              os.path.join(HERE, "..", "..", "references", "hardware", "hw_constants.json")):
        if os.path.exists(c):
            return c
    hits = glob.glob(os.path.join(HERE, "..", "..", "**", "references", "hardware",
                                  "hw_constants.json"), recursive=True)
    return hits[0] if hits else None


def _selftest():
    # Step tables measured from LLVM (see module docstring). Each entry is (max_vgpr, waves):
    # the LAST VGPR count that still fits `waves` waves/SIMD.
    steps = {
        "gfx942":  [(64, 8), (72, 7), (80, 6), (96, 5), (128, 4), (168, 3), (255, 2)],
        "gfx950":  [(64, 8), (72, 7), (80, 6), (96, 5), (128, 4), (168, 3), (255, 2)],
        "gfx90a":  [(64, 8), (72, 7), (80, 6), (96, 5), (128, 4), (168, 3), (255, 2)],
        "gfx1250": [(64, 16), (80, 12), (96, 10), (112, 9), (128, 8), (144, 7), (160, 6),
                    (192, 5), (256, 4)],
        "gfx1100": [(96, 16), (120, 12), (144, 10), (168, 9), (192, 8), (216, 7), (240, 6),
                    (256, 5)],
        "gfx1151": [(96, 16), (120, 12), (144, 10), (168, 9), (192, 8), (216, 7), (240, 6),
                    (256, 5)],
        "gfx1200": [(96, 16), (120, 12), (144, 10), (168, 9), (192, 8), (216, 7), (240, 6),
                    (256, 5)],
        "gfx1201": [(96, 16), (120, 12), (144, 10), (168, 9), (192, 8), (216, 7), (240, 6),
                    (256, 5)],
    }
    for arch, table in steps.items():
        lo = 1
        for hi, want in table:
            for v in (lo, hi):                     # both ends of every step
                got, _ = waves_by_vgpr(v, arch)
                assert got == want, f"{arch} vgpr={v}: got {got} waves, expected {want}"
            lo = hi + 1
    # The bug this module exists to kill: the CDNA formula on an RDNA target.
    assert waves_by_vgpr(249, "gfx1201")[0] == 5, waves_by_vgpr(249, "gfx1201")
    assert waves_by_vgpr(249, "gfx942")[0] == 2, waves_by_vgpr(249, "gfx942")
    # An unknown arch must refuse rather than fall back to CDNA.
    waves, label = waves_by_vgpr(128, "gfx1399")
    assert waves is None and "unknown arch" in label, (waves, label)

    # Synthetic KD fragments -- hand-written, not excerpted from any kernel.
    rdna_asm = ('\t.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"\n'
                '\t\t.amdhsa_wavefront_size32 1\n'
                '\t\t.amdhsa_next_free_vgpr 249\n'
                '; NumVgprs: 249\n; Occupancy: 5\n')
    assert arch_from_asm(rdna_asm) == "gfx1201", arch_from_asm(rdna_asm)
    assert wave32_from_asm(rdna_asm) is True
    assert llvm_occupancy(rdna_asm) == 5
    assert occupancy_from_asm(rdna_asm, 249)[:2] == (5, "llvm-comment")
    # Same dump without LLVM's comment -> the model must reproduce the same answer.
    assert occupancy_from_asm(rdna_asm.replace("; Occupancy: 5\n", ""), 249)[:2] == (5, "model")
    cdna_asm = '\t.amdgcn_target "amdgcn-amd-amdhsa--gfx942"\n\t\t.amdhsa_next_free_vgpr 81\n'
    assert arch_from_asm(cdna_asm) == "gfx942" and wave32_from_asm(cdna_asm) is False
    assert occupancy_from_asm(cdna_asm, 81)[:2] == (5, "model"), occupancy_from_asm(cdna_asm, 81)
    # A dump with no target at all: refuse, do not guess.
    assert occupancy_from_asm("s_nop 0\n", 64)[1] == "unknown"

    # The JSON reference layer is the SoT for these numbers; when it is reachable (composed
    # pack), it must agree with the table above -- otherwise the two drift silently.
    p = _find_hw_constants()
    if p:
        archs = json.load(open(p)).get("arch", {})
        checked = 0
        for arch, facts in archs.items():
            table = facts.get("vgpr_wave_steps")
            if not table:
                continue
            m = model_for(arch)
            assert m, f"{arch} has vgpr_wave_steps in hw_constants.json but no model here"
            for hi, want in table:
                got, _ = waves_by_vgpr(hi, arch)
                assert got == want, (f"{arch} vgpr={hi}: hw_constants.json says {want} waves, "
                                     f"model says {got}")
            for key in ("vgpr_file_per_simd", "vgpr_alloc_granule", "max_waves_per_simd",
                        "vgpr_per_wave"):
                if key in facts:
                    assert facts[key] == m[key], f"{arch}.{key}: json {facts[key]} != {m[key]}"
            checked += 1
        assert checked >= 4, f"only {checked} archs cross-checked against {p}"
    print(f"[amd_occupancy] SELFTEST PASS ({'cross-checked hw_constants.json' if p else 'no reference tree; table-only'})")
    return 0


def main():
    if "--selftest" in sys.argv:
        return _selftest()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vgpr", type=int, default=None, help="next_free_vgpr (arch+AGPR on CDNA)")
    ap.add_argument("--arch", default=None, help="gfx target, e.g. gfx942 / gfx1201")
    ap.add_argument("--asm", default=None, help="AMDGCN .s: read target + LLVM's own occupancy")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.asm:
        text = open(a.asm, errors="ignore").read()
        vgpr = a.vgpr
        if vgpr is None:
            m = re.search(r"^\s*\.amdhsa_next_free_vgpr\s+(\d+)", text, re.M)
            vgpr = int(m.group(1)) if m else None
        waves, src, label = occupancy_from_asm(text, vgpr)
        print(f"arch    = {arch_from_asm(text)}  (wave32 KD flag: {wave32_from_asm(text)})")
        print(f"vgpr    = {vgpr}")
        print(f"waves/SIMD = {waves}  [{src}]")
        print(f"model   = {label}")
        return 0
    if a.arch is None:
        raise SystemExit("need --arch (with --vgpr) or --asm; see --help")
    waves, label = waves_by_vgpr(a.vgpr, a.arch)
    print(f"waves/SIMD by VGPR = {waves}")
    print(f"model = {label}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# --------------------------------------------------------------------------- #
# LDS capacity per CU -- the OCCUPANCY divisor for shared memory:
#     WGs/CU <= LDS_per_CU // lds_bytes_per_wg
#
# This is deliberately a PER-ARCH table and not a field on MODELS above, because the family
# model cannot carry it: `gfx94*` and `gfx95*` are both the "cdna" family and share the
# register geometry, but they do NOT share LDS -- CDNA3 has 64 KiB/CU, CDNA4 has 160 KiB.
# Folding LDS into the family model would hand gfx950 the CDNA3 number and overstate its LDS
# pressure by 2.5x, which is exactly the failure this table exists to prevent. (Banks and
# allocation granularity differ too -- 32/512 B vs 64/1280 B -- but nothing here divides by
# them, so they are documented in the arch cards rather than modelled.)
#
# Distinct from the per-WORKGROUP allocation cap, a separate and possibly lower limit (it is
# what RDNA's "<=64 KiB/WG against 128 KiB/WGP" refers to). Only the per-CU figure divides.
#
# An arch with no entry returns None, on the same principle as `waves_by_vgpr`: a confident
# wrong number is worse than no number.
# LDS capacity per CU -- the OCCUPANCY divisor for shared memory:
#     WGs/CU <= LDS_per_CU // lds_bytes_per_wg
#
# Read from references/hardware/hw_constants.json rather than duplicated here. That file is
# the pack's per-arch source of truth and it is more complete than any table worth hand-
# maintaining: gfx942 64 KiB, gfx950 160 KiB, gfx1250 320 KiB, and for RDNA it correctly
# distinguishes `lds_per_wgp_kib` (128) from `lds_per_wg_kib` (64), which a single "per CU"
# number cannot express.
#
# This is deliberately NOT a field on MODELS above, because the family model cannot carry it:
# gfx94* and gfx95* are both the "cdna" family and share the register geometry, but not the
# LDS -- 64 KiB vs 160 KiB. Folding it in would hand gfx950 the CDNA3 number and overstate
# its LDS pressure by 2.5x.
def lds_per_cu(arch):
    """LDS bytes per CU for `arch`, or None when the reference has no figure for it.

    None is a real answer, on the same principle as `waves_by_vgpr`: a divisor that is right
    for one generation and silently applied to another yields a confident wrong occupancy
    verdict. RDNA returns None on purpose -- its shared memory is per-WGP with a lower
    per-workgroup cap, so callers that need it must read both fields themselves.
    """
    if not arch:
        return None
    path = _find_hw_constants()
    if not path:
        return None
    try:
        import json
        with open(path) as f:
            table = json.load(f).get("arch", {})
    except (OSError, ValueError):
        return None
    for name in sorted(table, key=len, reverse=True):
        if str(arch).startswith(name):
            kib = table[name].get("lds_per_cu_kib")
            return int(kib) * 1024 if kib else None
    return None
    for k in sorted(LDS_PER_CU_BY_ARCH, key=len, reverse=True):
        if str(arch).startswith(k):
            return LDS_PER_CU_BY_ARCH[k]
    return None
