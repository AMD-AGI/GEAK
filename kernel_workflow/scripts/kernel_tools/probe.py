#!/usr/bin/env python3
"""Compile-only occupancy probe: register / LDS pressure in SECONDS, and whether a tile plan
can EVER reach a target occupancy -- answered BEFORE you compile it.

Two questions, one tool. Both were hand-rolled in every campaign that mattered:

  measure  "did that edit move the pressure?"  -- parse the compiled artifact. No kernel is
           launched, no GPU time is consumed, no profiler is involved, so it answers in seconds
           for a dozen variants. `capture.sh` gives the same VGPR/LDS facts but only as a
           by-product of a full profile (minutes).

  plan     "can this tile shape reach 2 waves/SIMD at all?" -- add up the resident tensors from
           the tile plan on paper. `calc_perf.py occ` computes occupancy FROM a VGPR count;
           nothing derived the VGPR count FROM a tile plan, and that is the half you need while
           you are still CHOOSING the tile. A plan that is already over budget is a tile to
           discard, not a tile to compile and measure.

Both occupancy limiters are reported together (registers AND LDS): a kernel can be capped by
both at once, and relieving only one buys nothing.

CDNA facts applied (gfx942/gfx950):
  * ArchVGPR + AGPR share ONE 512-entry register file per SIMD -- they are NOT separate budgets
    (this is the single biggest difference from NVIDIA's per-thread 255 and it is why an
    accumulator-heavy tile silently caps at 1 wave/SIMD).
  * VGPR allocation granularity 8; wave64; 4 SIMDs/CU.
  * LDS/CU is NOT a shared CDNA fact: 64 KiB on gfx94*, 160 KiB on gfx95*. The two generations
    share the register geometry and not this, which is why 65536 reads as a safe default and
    is not one -- being correct on gfx942 is exactly what stops a misapplication from looking
    wrong. Where the target is not established this script WITHHOLDS the LDS limiter (and says
    so) instead of dividing by a guess; `--lds-per-cu <bytes>` supplies it.
  * Occupancy is capped JOINTLY by LDS and by registers. Reporting only the LDS side invites the
    wrong decision: on a register-bound kernel the LDS line can advertise an order of magnitude
    of headroom that does not exist. `measure` prints both per metadata row, in WGs/CU, and
    names which one binds.

The register geometry is NOT the same on RDNA (wave32, a 1536-entry file per SIMD, 256 per
wave, no AGPRs) -- `measure` reads the target out of the artifact, and `plan` takes `--arch`.
Both print the model they used; applying the CDNA numbers to an RDNA tile under-reports
occupancy by 2-3x and invents a register wall that is not there.

No default arch. `plan` REQUIRES --arch; `measure` reads each dump's `.amdgcn_target` and takes
`--arch` only for dumps that name none -- with neither, that dump's register term is WITHHELD
rather than computed on a CDNA guess.

Usage (kernel_workflow/scripts/kernel_tools/probe.py):
  # measure -- from a compiled artifact directory (dump_ir.sh output, or a Triton cache dir)
  python3 probe.py measure --dir exp/round_3/capture/ir
  python3 probe.py measure --dir ~/.triton/cache --json probe.json --arch gfx950

  # plan -- from a tile plan, before compiling
  python3 probe.py plan --arch gfx950 --warps 4 --dtype-bits 16 --acc-bits 32 \\
      --acc  dk=128x128 dv=128x128 --operand k=128x128 v=128x128
  python3 probe.py plan --arch gfx942 --warps 8 --acc acc=128x128     # gfx942 downgrade
  python3 probe.py plan --arch gfx1201 --warps 8 --acc acc=128x128
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys


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

VGPR_PER_SIMD = 512        # CDNA: ArchVGPR + AGPR COMBINED
VGPR_GRANULE = 8
# LDS/CU lives in the shared vendor/amd occupancy model, which is where the arch dispatch
# already is. It is a PER-ARCH table there and not a family field, because gfx94*/gfx95* share
# the register geometry but not the LDS (64 KiB vs 160 KiB).
def lds_per_cu(arch) -> int | None:
    """LDS bytes per CU for `arch`, or None when no figure is available.

    None is a real answer, and there is deliberately NO numeric default. 65536 is the tempting
    one and it is the worst available: it is *correct* for gfx942, so a wrong application of it
    never looks wrong on inspection -- it emits a plausible number instead of an error, and on a
    160 KiB part it understates the LDS limiter by 2.5x. A divisor that is right for one
    generation and silently applied to another produces a confident wrong occupancy verdict,
    which is worse than declining to produce one.
    """
    if not arch:
        return None
    if _OCC is not None and hasattr(_OCC, "lds_per_cu"):
        return _OCC.lds_per_cu(arch)
    return None


def _sole_arch(kernels) -> str | None:
    """The one target EVERY dump names, or None when they disagree or any is unnamed.

    None means "refuse to price LDS here", not "pick one". Lifted out of `cmd_measure` so the
    rule can be asserted: while it was an inline expression the only thing that ever caught the
    borrowing version was an end-to-end run, and a rule no test can reach is a rule that will be
    re-broken. Taking the first kernel that happens to name a target and dividing every OTHER
    kernel's LDS by it applies a 64 KiB divisor to a 160 KiB part (or the reverse) for the rest
    of the run -- the same borrowing `_match_lds` refuses, and a directory holding both a gfx94*
    and a gfx95* dump has no single divisor at all.
    """
    archs = {k.get("arch") for k in kernels}
    return next(iter(archs)) if (len(archs) == 1 and None not in archs) else None


WAVE = 64
MAX_WAVES_PER_SIMD = 8
SIMDS_PER_CU = 4           # CDNA and RDNA alike; needed to convert wg/CU -> waves/SIMD


def arch_model(arch: str | None) -> dict:
    """Register-file geometry for `arch`, falling back to the CDNA constants above (with the
    fallback named in `label`) when the arch is unknown or the shared model is unreachable."""
    m = _OCC.model_for(arch) if (_OCC and arch) else None
    if m:
        return m
    return {"wave_size": WAVE, "vgpr_file_per_simd": VGPR_PER_SIMD,
            "vgpr_per_wave": VGPR_PER_SIMD,   # they coincide on CDNA; see amd_occupancy.MODELS
            "vgpr_alloc_granule": VGPR_GRANULE, "max_waves_per_simd": MAX_WAVES_PER_SIMD,
            "family": "cdna",
            "label": f"CDNA wave64 fallback (512/SIMD, granule 8, cap 8) -- no model for "
                     f"{arch!r}; pass --arch"}


def vgpr_budget_at(model: dict, target_waves: int) -> int:
    """VGPRs/lane a tile may spend to still reach `target_waves`. Two ceilings, tighter wins:
    the per-wave slice of the register FILE, and the architectural cap on what one wave can
    address at all (256 on RDNA, where file/target is 768 at 2 waves -- a budget no wave can
    ever spend). They coincide on CDNA, which is why the clamp was easy to miss."""
    budget = model["vgpr_file_per_simd"] // target_waves
    cap = model.get("vgpr_per_wave")
    return min(budget, cap) if cap else budget


def waves_by_vgpr(total_vgpr: int, arch: str | None = None) -> int:
    m = arch_model(arch)
    if not total_vgpr:
        return m["max_waves_per_simd"]
    gran = m["vgpr_alloc_granule"]
    alloc = ((total_vgpr + gran - 1) // gran) * gran
    return min(m["max_waves_per_simd"], m["vgpr_file_per_simd"] // alloc)


# --------------------------------------------------------------------------- measure
_RE = {
    "vgpr": re.compile(r"^\s*\.vgpr_count:\s*(\d+)", re.M),
    "agpr": re.compile(r"^\s*\.agpr_count:\s*(\d+)", re.M),
    "spill": re.compile(r"^\s*\.private_segment_fixed_size:\s*(\d+)", re.M),
    "kd_vgpr": re.compile(r"^\s*\.amdhsa_next_free_vgpr\s+(\d+)", re.M),
    "kd_spill": re.compile(r"^\s*\.amdhsa_private_segment_fixed_size\s+(\d+)", re.M),
    # Needed ONLY to convert the LDS limiter's wg/CU into waves/SIMD. Without it the two
    # quantities cannot be compared and we must not pretend otherwise -- see _waves_by_lds.
    "wg_size": re.compile(r"^\s*\.max_flat_workgroup_size:\s*(\d+)", re.M),
}


def _waves_by_lds(lds_bytes: int, wg_size: int | None, arch: str | None,
                  cap_override: int | None = None) -> tuple[int | None, str]:
    """LDS-limited occupancy in waves/SIMD, or (None, reason) when it cannot be derived.

    LDS is allocated per WORKGROUP and capped per CU, so it bounds wg/CU -- NOT waves/SIMD.
    Converting needs the workgroup's wave count, which is only knowable from
    `.max_flat_workgroup_size`. When the dump does not carry it we return None and say so,
    because reporting a wg/CU figure in a waves/SIMD field is the exact unit error this pack
    warns about; a missing number is recoverable, a wrong one in the right-looking slot is not.
    """
    if not lds_bytes:
        return None, "no LDS metadata"
    if not wg_size:
        return None, "no .max_flat_workgroup_size in the dump -- cannot convert wg/CU to waves/SIMD"
    # An explicitly supplied figure wins: the caller read it off the part, we only inferred it.
    cap = cap_override or lds_per_cu(arch)
    if not cap:
        # LDS/CU is 64 KiB on gfx94* and 160 KiB on gfx95*, so guessing it wrong moves the
        # limiter by 2.5x. Refuse rather than guess -- and name the fix.
        if not arch:
            return None, ("target not named in the dump -- LDS/CU is 64 vs 160 KiB across "
                          "CDNA3/4; re-dump with the target named, or pass --lds-per-cu <bytes>")
        return None, (f"no LDS/CU figure for {arch} in the shared occupancy model -- "
                      f"pass --lds-per-cu <bytes>")
    m = arch_model(arch)
    wave = m.get("wave_size") or WAVE
    waves_per_wg = max(1, -(-wg_size // wave))
    wgs_by_lds = cap // lds_bytes
    if wgs_by_lds < 1:
        return 0, "LDS/WG exceeds the whole CU budget"
    return (wgs_by_lds * waves_per_wg) // SIMDS_PER_CU, "ok"


def _match_lds(name: str, lds: dict[str, int]) -> int | None:
    """Join a dumped artifact to its cache-metadata LDS entry by kernel name.

    The `.s` is named for the file and the metadata for the kernel, so an exact hit is not
    guaranteed. A NON-unique containment match is treated as no match on purpose: borrowing
    another kernel's LDS produces a confident wrong number, which is worse than the warning.
    """
    if not lds:
        return None
    stem = os.path.splitext(name)[0]
    if stem in lds:
        return lds[stem]
    hits = [b for n, b in lds.items() if n and (n in stem or stem in n)]
    return hits[0] if len(hits) == 1 else None


def _from_asm(path: str, lds_bytes: int | None = None,
              cap_override: int | None = None, arch_hint: str | None = None) -> dict | None:
    try:
        txt = open(path, errors="ignore").read()
    except OSError:
        return None
    g = {k: (int(m.group(1)) if (m := rx.search(txt)) else None) for k, rx in _RE.items()}
    total = g["vgpr"] if g["vgpr"] is not None else g["kd_vgpr"]
    if total is None:
        return None
    # .vgpr_count is the ARCH count and .agpr_count is separate; .amdhsa_next_free_vgpr is
    # already the combined budget. Normalize to "combined" either way.
    combined = total + (g["agpr"] or 0) if g["vgpr"] is not None else total
    spill = g["spill"] if g["spill"] is not None else g["kd_spill"]
    # The artifact names its own target, so `measure` never has to be told the arch. When LLVM
    # left its `; Occupancy:` in the dump it beats any model here FOR THE REGISTER TERM -- the
    # backend computes it against the LDS it can see at compile time, which on a launch-sized
    # group segment is zero. So the register term is NOT the kernel's occupancy, and
    # `waves_per_simd` below is min(register term, LDS term) whenever the LDS term is derivable.
    # min() is right under BOTH hypotheses, which is why the open question "does the emitter fold
    # in statically-allocated LDS" does not have to be answered first: if it already folded LDS
    # in, its number is already <= the LDS term and the min is a no-op; if it did not, the min
    # corrects it. When the LDS term is NOT derivable we return the register term and say so in
    # `waves_warning` rather than silently shipping a figure that can be 3x high.
    # The dump's own target wins; `--arch` (arch_hint) only fills a dump that names none. With
    # neither there is NO default: the register model differs 2-3x between CDNA and RDNA, so
    # the register term is withheld instead of computed on a CDNA guess.
    arch = (_OCC.arch_from_asm(txt) if _OCC else None) or arch_hint
    llvm_occ = _OCC.llvm_occupancy(txt) if _OCC else None
    if llvm_occ is not None:
        reg_term, src = llvm_occ, "llvm-comment"
    elif arch:
        reg_term, src = waves_by_vgpr(combined, arch), "model"
    else:
        reg_term, src = None, "unknown"
    lds_waves, why = _waves_by_lds(lds_bytes or 0, g["wg_size"], arch, cap_override)
    if reg_term is None:
        waves, warn = None, ("target not named in the dump and no --arch: register term WITHHELD "
                             "(no default arch) -- pass --arch gfx950 (or the dump's real target)")
    elif lds_waves is None:
        waves, warn = reg_term, f"register term only ({why})"
    else:
        waves, src, warn = min(reg_term, lds_waves), src + "+lds", None
    return {"name": os.path.basename(path), "vgpr_arch": g["vgpr"], "agpr": g["agpr"],
            "vgpr_combined": combined, "spill_bytes": spill, "arch": arch,
            "waves_per_simd": waves,
            "waves_by_vgpr_term": reg_term, "waves_by_lds_term": lds_waves,
            "lds_bytes_per_wg": lds_bytes, "wg_size": g["wg_size"],
            "waves_source": src, "waves_warning": warn}


def _lds_from_meta(d: str) -> dict[str, int]:
    """LDS bytes/WG from the Triton cache metadata `shared`. The KD's group_segment_fixed_size
    and rocprof-compute 7.1.8 are structurally 0 for Triton (shared memory is sized dynamically
    at launch), so neither may be substituted -- a 0 from those is not evidence of 'no LDS'."""
    out = {}
    for root, _dirs, files in os.walk(d):
        for f in files:
            if not f.endswith(".json") or f.startswith("__grp__") or f.startswith("meta___grp__"):
                continue
            try:
                j = json.load(open(os.path.join(root, f)))
            except (OSError, ValueError):
                continue
            if isinstance(j, dict) and isinstance(j.get("shared"), int):
                out[str(j.get("name") or f)[:44]] = j["shared"]
    return out


def _num_warps_from_meta(d: str) -> dict[str, int | None]:
    """`num_warps` per kernel from the same Triton cache metadata `_lds_from_meta` reads, keyed
    the same way. The register limit cannot be turned into a WORKGROUP limit without it."""
    out = {}
    for root, _dirs, files in os.walk(d):
        for f in files:
            if not f.endswith(".json") or f.startswith("__grp__") or f.startswith("meta___grp__"):
                continue
            try:
                with open(os.path.join(root, f)) as fh:
                    j = json.load(fh)
            except (OSError, ValueError):
                continue
            if isinstance(j, dict) and isinstance(j.get("shared"), int):
                nw = j.get("num_warps")
                out[str(j.get("name") or f)[:44]] = nw if isinstance(nw, int) else None
    return out


def _simds_per_cu(arch: str | None) -> int:
    """SIMDs/CU from the shared per-arch reference when it states one, else SIMDS_PER_CU."""
    if arch and _OCC is not None and hasattr(_OCC, "simds_per_cu"):
        v = _OCC.simds_per_cu(arch)
        if v:
            return int(v)
    return SIMDS_PER_CU


def _waves_for(meta_name: str, kernels: list[dict]):
    """REGISTER-term waves/SIMD for the kernel a metadata row describes, joined BY NAME.

    The register term (`waves_by_vgpr_term`), not `waves_per_simd`: the latter already has the
    LDS term min'd in, and feeding it back against the LDS limit would report "both bind" on
    every LDS-bound kernel.

    Collapsing waves to one value across the directory drops the register limiter as soon as
    the dir holds two kernels with different occupancy -- and a multi-kernel dir is the NORMAL
    case, not the exception: a harness' pack / split-K reduce kernel compiles into the same
    cache dir as the body. Measured on one dump: the same `_mla_gluon` reported
    `regs<=1 ... both bind` alone and `regs<=?` beside a 8-waves/SIMD pack kernel, i.e. the tool
    silently withheld exactly the half it exists to make un-missable.

    Asm names carry a file extension and may carry a dump suffix, and the metadata name is
    truncated to 44 chars, so match the stem exactly first, then by prefix in either direction.
    Ambiguity returns None: a guess here is a wrong limiter.
    """
    stems = {}
    for k in kernels:
        stem = re.sub(r"\.(amdgcn|s)$", "", k["name"])
        stem = re.sub(r"_final_isa$", "", stem)
        w = k.get("waves_by_vgpr_term", k.get("waves_per_simd"))
        if w:
            stems[stem] = w
    if meta_name in stems:
        return stems[meta_name]
    cands = {w for s, w in stems.items() if s.startswith(meta_name) or meta_name.startswith(s)}
    return cands.pop() if len(cands) == 1 else None


def _limiter_row(name, shared_b, num_warps, waves, cap, simds, cap_src=None):
    """One `measure` LDS row: both occupancy limiters in WGs/CU, and which of them binds.

    Registers and LDS both cap WORKGROUPS per CU, and reporting only the LDS side invites the
    error this tool exists to prevent: on one measured gfx950 kernel the arm with the most LDS
    headroom (20 WGs/CU) was the SLOWEST, because 2 waves/SIMD x 4 SIMDs = 8 waves/CU against a
    num_warps=8 workgroup already pinned it to 1. So the register limit is turned into a
    workgroup limit the way the hardware does (`waves/SIMD x simds_per_cu / num_warps`).

    Pure and selftested, because three of its states are wrong-answer-shaped rather than
    crash-shaped and all three have been produced by this tool:

      cap is None    the LDS divisor is UNKNOWN. Naming a binder here asserts that LDS does
                     not bind, which is the confident-wrong-number failure the arch dispatch
                     exists to prevent -- so no binder is named and the LDS verdict is WITHHELD.
      shared_b == 0  the kernel allocates no LDS, so LDS genuinely does not bound it. That is
                     NOT the same as unknown; the cell reads `-` and REGISTERS bind.
      waves is None  no artifact matched this metadata row. Say that, rather than blaming
                     `num_warps` and sending someone to fix a file that is already correct.
    """
    if not shared_b:
        # A kernel that allocates no LDS is not bounded by LDS, and that holds whether or
        # not a divisor is reachable -- so this row stays decidable in the degraded case.
        by_lds, lds_unknown = None, False
    else:
        lds_unknown = not cap
        by_lds = None if lds_unknown else cap // shared_b
    by_reg = (waves * simds) // num_warps if (waves and num_warps) else None
    if lds_unknown:
        cell = " = WITHHELD"
    elif by_lds is None:
        cell = " = - (no LDS)"
    else:
        cell = f"<={by_lds}" + (f" [{cap_src}]" if cap_src else "")
    row = f"  {name:46s} lds/WG={shared_b:6d} B  WGs/CU by LDS{cell}"
    if by_reg is None:
        why = ("no num_warps in metadata" if not num_warps
               else f"no .amdgcn/.s matched the name {name!r} (or its target is unnamed -- "
                    f"pass --arch)")
        return row + f"  regs<=? -- {why}, register side not computable"
    regs = f"  regs<={by_reg} (waves/SIMD={waves} x{simds} / nw={num_warps})"
    if lds_unknown:
        return row + regs + f"  -> <={by_reg} WGs/CU from REGISTERS alone; LDS side unknown"
    if by_lds is None:
        return row + regs + f"  -> {by_reg} WGs/CU, REGISTERS bind (kernel allocates no LDS)"
    binder = ("REGISTERS" if by_reg < by_lds else "LDS" if by_lds < by_reg else "both")
    return row + regs + f"  -> {min(by_lds, by_reg)} WGs/CU, {binder} bind"


def cmd_measure(a):
    asms = []
    for root, _dirs, files in os.walk(a.dir):
        asms += [os.path.join(root, f) for f in files
                 if f.endswith((".amdgcn", ".s")) or f.endswith("_final_isa.s")]
    # LDS first: the occupancy a kernel row reports is min(register term, LDS term), so the
    # limiter has to be in hand BEFORE the rows are built, not printed in a separate block
    # underneath them. Keeping the two halves apart is what let the register term be read as
    # the occupancy for as long as it was.
    lds = _lds_from_meta(a.dir)
    kernels = [r for r in (_from_asm(p, _match_lds(os.path.basename(p), lds), a.lds_per_cu,
                                     a.arch)
                           for p in sorted(asms)) if r]

    print(f"=== probe measure: {a.dir} ===")
    if not kernels:
        print("  no .amdgcn/.s with a register budget found -- nothing to measure "
              "(dump one with dump_ir.sh). NOT reporting zeros.")
    for k in kernels:
        spill = "n/a" if k["spill_bytes"] is None else f"{k['spill_bytes']} B"
        warn = "  <- SPILL" if (k["spill_bytes"] or 0) > 0 else ""
        arch = k["arch"] or "UNNAMED target"
        print(f"  {k['name'][:44]:46s} [{arch}] vgpr={k['vgpr_combined']:4d} "
              f"waves/SIMD={k['waves_per_simd']} [{k['waves_source']}]  spill={spill}{warn}")
        if k["waves_per_simd"] is None:
            print(f"  {'':46s} WITHHELD: {k['waves_warning']}")
        elif k["waves_warning"]:
            print(f"  {'':46s} WARNING: {k['waves_warning']} -- this can be several times the "
                  f"real occupancy on an LDS-bound kernel; do not quote it as the occupancy.")
        elif (k["waves_by_lds_term"] is not None and k["waves_by_vgpr_term"] is not None
              and k["waves_by_lds_term"] < k["waves_by_vgpr_term"]):
            print(f"  {'':46s} LDS-BOUND: registers allow {k['waves_by_vgpr_term']}, "
                  f"LDS allows {k['waves_by_lds_term']} (lds/WG={k['lds_bytes_per_wg']} B, "
                  f"wg={k['wg_size']}) -- spending registers here buys nothing.")
    print()
    if lds:
        # Per-arch divisor or none at all. Labelling a guessed divisor does not make it safe --
        # it makes it quotable. With no target named, print the bytes (a measured fact) and
        # WITHHOLD the WGs/CU verdict (a derived one), naming the flag that would supply it.
        #
        # The arch is taken from the dumps only when EVERY dump names the SAME one. Picking the
        # first kernel that happens to name a target and dividing every other kernel's LDS by it
        # is the same borrowing error `_match_lds` refuses: one dump's fact is not another's, and
        # a dir holding both a gfx94* and a gfx95* dump has no single divisor at all.
        _archs = {k["arch"] for k in kernels}
        _arch = _sole_arch(kernels)
        _cap = a.lds_per_cu or lds_per_cu(_arch)
        _src = "--lds-per-cu" if a.lds_per_cu else _arch
        # Both limiters per row, in WGs/CU, joined to the kernel BY NAME (see `_limiter_row`):
        # the LDS figure alone advertises headroom a register-bound kernel does not have.
        _nw = _num_warps_from_meta(a.dir)
        _simds = _simds_per_cu(_arch)
        for n, b in sorted(lds.items(), key=lambda kv: -kv[1]):
            print(_limiter_row(n, b, _nw.get(n), _waves_for(n, kernels), _cap, _simds, _src))
        if not _cap:
            _why = ("the dumps here do not all name the same target"
                    if len(_archs) > 1 else
                    f"no LDS/CU divisor for {_arch or 'an unnamed target'}")
            print(f"  ^ {_why}: LDS/CU is 64 KiB on gfx94* and 160 KiB on gfx95*, a 2.5x "
                  f"spread. Re-dump with the target named, or pass --lds-per-cu <bytes>, rather "
                  f"than reading a WGs/CU number off a guess.")
    else:
        print("  lds/WG UNAVAILABLE -- no Triton cache metadata under this dir. Do NOT read LDS "
              "from the KD or rocprof-compute: both are structurally 0 for Triton kernels.")
    if a.json:
        json.dump({"kernels": kernels, "lds_bytes_per_wg": lds}, open(a.json, "w"), indent=2)
        print(f"\nwrote {a.json}")
    return 0


# ------------------------------------------------------------------------------ plan
def _tile(spec: str) -> tuple[str, int, int]:
    """'dk=128x128' -> ('dk', 128, 128)"""
    name, _, dims = spec.partition("=")
    m = re.fullmatch(r"(\d+)x(\d+)", dims.strip())
    if not m:
        sys.exit(f"bad tile spec {spec!r}; expected name=MxN (e.g. dk=128x128)")
    return name, int(m.group(1)), int(m.group(2))


def tensor_vgpr(m: int, n: int, warps: int, bits: int, wave: int = WAVE) -> int:
    """VGPRs/lane a resident [m,n]-element tensor occupies at `warps` warps, `bits`/element.
    elements/lane * bits / 32, rounding UP at each step (a tensor that does not divide evenly
    across the lane grid still occupies whole registers). `wave` is the wavefront width: 64 on
    CDNA, 32 on RDNA -- the SAME tile costs twice the VGPR/lane at wave32. The single source for
    the resident-VGPR rule -- imported by plain_autotune's budget pruner so there is ONE
    occupancy model, not three."""
    lanes = warps * wave
    per_lane = -(-(m * n) // lanes)
    return -(-(per_lane * bits) // 32)


def plan_vgpr(tiles, warps: int, acc_bits: int = 32, operand_bits: int = 16,
              arch: str | None = None) -> dict:
    """Resident VGPR/lane of a tile plan, before temporaries/addressing/staging (a LOWER bound).
    `tiles` = {"acc": [(name,m,n),...], "operand": [(name,m,n),...]}. Returns
    {total, waves_by_resident, rows:[(name,kind,m,n,vgpr)]}. Importable; `cmd_plan` renders it."""
    wave = arch_model(arch)["wave_size"]
    rows = []
    for name, m, n in tiles.get("acc", []):
        rows.append((name, "acc", m, n, tensor_vgpr(m, n, warps, acc_bits, wave)))
    for name, m, n in tiles.get("operand", []):
        rows.append((name, "operand", m, n, tensor_vgpr(m, n, warps, operand_bits, wave)))
    total = sum(r[4] for r in rows)
    return {"total": total, "waves_by_resident": waves_by_vgpr(total, arch), "rows": rows}


def cmd_plan(a):
    model = arch_model(a.arch)
    wave = model["wave_size"]
    file_size = model["vgpr_file_per_simd"]
    lanes = a.warps * wave
    tiles = {"acc": [_tile(s) for s in (a.acc or [])],
             "operand": [_tile(s) for s in (a.operand or [])]}
    p = plan_vgpr(tiles, a.warps, a.acc_bits, a.dtype_bits, a.arch)
    rows = [(name, kind, f"{m}x{n}", -(-(m * n) // lanes), vgpr)
            for name, kind, m, n, vgpr in p["rows"]]

    total = p["total"]
    waves = p["waves_by_resident"]
    print(f"=== probe plan: warps={a.warps} ({lanes} lanes/WG, wave{wave}) ===")
    print(f"    model: {model['label']}")
    print(f"{'tensor':12s} {'kind':9s} {'tile':11s} {'elem/lane':>10s} {'VGPR/lane':>10s}")
    for name, kind, tile, per_lane, vgpr in rows:
        print(f"{name:12s} {kind:9s} {tile:11s} {per_lane:10d} {vgpr:10d}")
    print(f"{'-'*56}\n{'RESIDENT TOTAL':12s} {'':9s} {'':11s} {'':>10s} {total:10d}")
    print(f"\n  resident set = {total} VGPR of the {file_size}/SIMD budget")
    print(f"  -> waves/SIMD <= {waves} from the RESIDENT SET ALONE, before any temporaries, "
          f"addressing, or pipeline staging")
    if a.target_waves:
        budget = vgpr_budget_at(model, a.target_waves)
        if total > budget:
            print(f"\n  VERDICT: {a.target_waves} waves/SIMD is UNREACHABLE for this tile plan. "
                  f"It needs <={budget} VGPR and the resident set alone is {total} "
                  f"(over by {total - budget}).")
            print("  Shrink a tile or move a tensor off-register; a compiler flag cannot fix "
                  "a plan that does not fit.")
        else:
            head = budget - total
            print(f"\n  VERDICT: {a.target_waves} waves/SIMD is reachable ONLY IF temporaries + "
                  f"addressing + pipeline staging fit in the remaining {head} VGPR/lane.")
            print("  This is a NECESSARY condition, not a sufficient one: the resident set is a "
                  "LOWER bound and\n  a real kernel's non-resident pressure is routinely larger "
                  "than the resident set itself.\n  Confirm with `probe.py measure` on the first "
                  "compile before betting a round on this tile.")
    # The divisor is per-arch, so it is NOT printed unless the arch supplied a real one.
    # A number here would be quotable, and the tempting number (65536) is correct on gfx94*
    # and 2.5x wrong on gfx95* -- i.e. it never looks wrong on inspection.
    _cap = a.lds_per_cu or lds_per_cu(a.arch)
    if _cap:
        _lds_note = f"{_cap} B/CU ({'--lds-per-cu' if a.lds_per_cu else a.arch})"
    else:
        _lds_note = ("LDS/CU is not established for "
                     f"{a.arch or 'an unnamed target'} -- it is 64 KiB on gfx94* and 160 KiB "
                     "on gfx95*, a 2.5x spread, so name the target or read it off the part "
                     "(`rocminfo` -> Segment: GROUP, Size)")
    print(f"\n  (LDS is a SECOND, independent limiter -- check it too: {_lds_note}.)")
    return 0


def _selftest():
    assert waves_by_vgpr(256) == 2 and waves_by_vgpr(257) == 1, "512-file boundary"
    assert waves_by_vgpr(128) == 4 and waves_by_vgpr(0) == 8
    # granularity: 130 rounds to 136 -> 512//136 = 3
    assert waves_by_vgpr(130) == 3, waves_by_vgpr(130)
    # RDNA is a different file, granule and cap -- the CDNA answer is wrong there by 2-3x
    if _OCC:
        assert waves_by_vgpr(130, "gfx1201") == 10, waves_by_vgpr(130, "gfx1201")
        assert waves_by_vgpr(249, "gfx1201") == 5
        assert arch_model("gfx1201")["wave_size"] == 32
        # the per-wave cap clamps the target budget: 1536//2 = 768 is not spendable by one wave
        assert vgpr_budget_at(arch_model("gfx1201"), 2) == 256
        assert vgpr_budget_at(arch_model("gfx1201"), 8) == 192      # file slice binds here
        assert vgpr_budget_at(arch_model("gfx942"), 2) == 256       # CDNA: the two agree
        # wave32 halves the lanes, so the same tile costs 2x the VGPR/lane
        assert tensor_vgpr(128, 128, 4, 32, 32) == 2 * tensor_vgpr(128, 128, 4, 32, 64)
    # an unknown arch falls back to CDNA but SAYS SO in the label it prints
    assert "no model for" in arch_model("gfx1399")["label"]
    # the CDNA rule that matters: arch and accum are ONE budget, not two
    r = _from_asm.__doc__  # noqa: F841  (documented behaviour exercised below via a temp file)
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".s", delete=False) as f:
        f.write("  .vgpr_count: 128\n  .agpr_count: 384\n  .private_segment_fixed_size: 456\n")
        p = f.name
    # no target in the dump and no --arch: the register term is WITHHELD, never a CDNA guess
    unnamed_k = _from_asm(p)
    assert unnamed_k["waves_per_simd"] is None and unnamed_k["waves_by_vgpr_term"] is None, unnamed_k
    assert "--arch" in unnamed_k["waves_warning"], unnamed_k
    got = _from_asm(p, arch_hint="gfx942")
    os.unlink(p)
    assert got["arch"] == "gfx942", got
    assert got["vgpr_combined"] == 512 and got["waves_per_simd"] == 1, got
    assert got["spill_bytes"] == 456, got
    # no LDS metadata -> register term, and it SAYS it is the register term
    assert got["waves_source"] == "model" and got["waves_warning"], got
    # the LDS limiter binds: 64 VGPR would allow 8 waves/SIMD, but on gfx942 (64 KiB/CU)
    # 32 KiB/WG allows 2 WGs/CU, and at 256 work-items (4 waves/WG) that is 2*4//4 = 2
    # waves/SIMD. min() must take the 2. The target has to be named or the LDS/CU divisor is
    # unknown (64 KiB on gfx94*, 160 KiB on gfx95*) and the term is refused by design.
    with tempfile.NamedTemporaryFile("w", suffix=".s", delete=False) as f:
        f.write("  .amdgcn_target: \"amdgcn-amd-amdhsa--gfx942\"\n"
                "  .vgpr_count: 64\n  .agpr_count: 0\n  .max_flat_workgroup_size: 256\n")
        p2 = f.name
    lds_bound = _from_asm(p2, 32 * 1024)
    assert lds_bound["waves_by_vgpr_term"] == 8, lds_bound
    assert lds_bound["waves_by_lds_term"] == 2, lds_bound
    assert lds_bound["waves_per_simd"] == 2, lds_bound
    assert lds_bound["waves_source"].endswith("+lds"), lds_bound
    # LDS bytes known but the workgroup size is not: the units cannot be converted, so the
    # register term is returned WITH a warning -- never a wg/CU number in a waves/SIMD field.
    with tempfile.NamedTemporaryFile("w", suffix=".s", delete=False) as f:
        f.write("  .amdgcn_target: \"amdgcn-amd-amdhsa--gfx942\"\n"
                "  .vgpr_count: 64\n  .agpr_count: 0\n")
        p3 = f.name
    no_wg = _from_asm(p3, 32 * 1024)
    os.unlink(p2)
    os.unlink(p3)
    assert no_wg["waves_per_simd"] == 8 and no_wg["waves_warning"], no_wg
    # --- the LDS/CU divisor. These are written so that a WRONG value has to FAIL them, which
    # is a stronger requirement than agreeing with a right one. In particular the constant
    # divisor this file used to carry (65536, correct on gfx942) satisfies any check that only
    # looks at gfx942; it is the INEQUALITY and the two None cases that reject it.
    assert lds_per_cu(None) is None, lds_per_cu(None)
    assert lds_per_cu("") is None
    if _OCC and hasattr(_OCC, "lds_per_cu"):
        assert lds_per_cu("gfx942") == 65536, lds_per_cu("gfx942")
        assert lds_per_cu("gfx950") == 163840, lds_per_cu("gfx950")
        assert lds_per_cu("gfx942") != lds_per_cu("gfx950"), "a constant divisor is the bug"
        assert lds_per_cu("gfx9999") is None, lds_per_cu("gfx9999")
        # ...and it has to CHANGE THE VERDICT, not just the constant: the same 32 KiB/WG,
        # 256-work-item kernel is 2 waves/SIMD on a 64 KiB part and 5 on a 160 KiB one.
        assert _waves_by_lds(32 * 1024, 256, "gfx942")[0] == 2
        assert _waves_by_lds(32 * 1024, 256, "gfx950")[0] == 5
    # The MISS path, exercised with the shared model absent. Every assertion above reaches
    # `lds_per_cu` only while `_OCC` is importable, so the function's own fallback return is
    # unreachable from them -- a literal default parked there (the shape the deleted
    # `LDS_PER_CU = 65536` had) satisfies all of them. That is not hypothetical: the packs that
    # ship without `amd_occupancy.py` run EXCLUSIVELY down this path, so it is the one place a
    # default would do its damage and the one place nothing was looking.
    _saved_occ = _OCC
    try:
        globals()["_OCC"] = None
        assert lds_per_cu("gfx942") is None, (
            f"with the occupancy model absent, lds_per_cu must return None, got "
            f"{lds_per_cu('gfx942')} -- a numeric fallback here is a guess wearing a fact's hat")
        assert lds_per_cu("gfx950") is None
        _no_model = _waves_by_lds(32 * 1024, 256, "gfx942")
        assert _no_model[0] is None and "--lds-per-cu" in _no_model[1], _no_model
    finally:
        globals()["_OCC"] = _saved_occ
    assert _OCC is _saved_occ

    # refusals: no target named -> None, and the message names the flag that fixes it
    unnamed = _waves_by_lds(32 * 1024, 256, None)
    assert unnamed[0] is None and "--lds-per-cu" in unnamed[1], unnamed
    # and the remedy has to actually work, or it is a consolation rather than a fix
    assert _waves_by_lds(32 * 1024, 256, None, 163840)[0] == 5
    assert _waves_by_lds(32 * 1024, 256, None, 65536)[0] == 2
    # ...including when the dump DOES name a target and the caller disagrees with the table.
    # That is the only case where the flag changes anything a user would care about, and while
    # it went untested the override could lose to the table without any check noticing: gfx942
    # tables at 65536 (2 waves), the caller says 163840 (5). An ignored remedy is worse than an
    # absent one -- it answers, so nobody goes looking.
    assert _waves_by_lds(32 * 1024, 256, "gfx942", 163840)[0] == 5, "--lds-per-cu must WIN"
    assert _waves_by_lds(32 * 1024, 256, "gfx950", 65536)[0] == 2, "--lds-per-cu must WIN"

    # the mixed-directory rule, asserted rather than left to an end-to-end run
    assert _sole_arch([{"arch": "gfx942"}, {"arch": "gfx942"}]) == "gfx942"
    assert _sole_arch([{"arch": "gfx942"}, {"arch": "gfx950"}]) is None, "mixed dir has no divisor"
    assert _sole_arch([{"arch": "gfx942"}, {"arch": None}]) is None, "one unnamed dump taints it"
    assert _sole_arch([{"arch": None}, {"arch": None}]) is None
    assert _sole_arch([]) is None
    # the override must not silently apply when it was not given: None, not a fallback number
    assert _waves_by_lds(32 * 1024, 256, None, None)[0] is None
    # LDS/WG larger than any plausible CU budget is 0 waves, not a negative or a crash
    assert _waves_by_lds(1 << 20, 256, None, 163840)[0] == 0
    # a non-unique name match must NOT borrow another kernel's LDS
    assert _match_lds("k.s", {"ak": 1, "kb": 2}) is None
    assert _match_lds("gemm_kernel.s", {"gemm_kernel": 4096}) == 4096
    # --- the two occupancy limiters, per metadata row. Both rules below have shipped wrong,
    # silently: the register side collapsed to one value per dir, and an unknown divisor named
    # a binder anyway.
    ks = [{"name": "body.amdgcn", "waves_by_vgpr_term": 2, "waves_per_simd": 1},
          {"name": "pack_final_isa.s", "waves_by_vgpr_term": 8, "waves_per_simd": 8}]
    assert _waves_for("body", ks) == 2 and _waves_for("pack", ks) == 8, "joined per kernel"
    assert _waves_for("absent", ks) is None
    # a metadata name truncated at 44 chars still matches its artifact by prefix
    assert _waves_for("body", [{"name": "body_0d1e2f.amdgcn", "waves_by_vgpr_term": 3}]) == 3
    # ...but an ambiguous prefix is not a guess: a wrong waves value is a wrong limiter
    assert _waves_for("k", [{"name": "k1.amdgcn", "waves_by_vgpr_term": 2},
                            {"name": "k2.amdgcn", "waves_by_vgpr_term": 4}]) is None
    _binders = ("REGISTERS bind", "LDS bind", "both bind")
    rb = _limiter_row("k", 65536, 4, 2, 163840, 4)
    assert "LDS<=2" in rb and "both bind" in rb, rb
    rb = _limiter_row("k", 8192, 8, 8, 163840, 4)
    assert "REGISTERS bind" in rb, rb
    rb = _limiter_row("k", 65536, 1, 8, 163840, 4)
    assert "LDS bind" in rb, rb
    # no LDS at all: LDS does not bound it, and comparing the limiters must not raise
    r0 = _limiter_row("k", 0, 4, 8, 163840, 4)
    assert "no LDS" in r0 and "REGISTERS bind" in r0 and "allocates no LDS" in r0, r0
    # no divisor: DISTINCT from the above, and no binder may be named
    ru = _limiter_row("k", 65536, 4, 2, None, 4)
    assert "WITHHELD" in ru and "unknown" in ru and not any(b in ru for b in _binders), ru
    # an unmatched artifact is named as such, not blamed on the metadata
    rw = _limiter_row("k", 65536, 4, None, 163840, 4)
    assert "no .amdgcn/.s matched" in rw, rw
    # a zero-LDS kernel stays decidable with no divisor: LDS cannot bound it either way
    r0u = _limiter_row("k", 0, 4, 8, None, 4)
    assert "no LDS" in r0u and "REGISTERS bind" in r0u, r0u

    # plan arithmetic: a [128,128] fp32 accumulator over 4 warps = 64 VGPR/lane
    assert -(-(128 * 128) // (4 * WAVE)) * 32 // 32 == 64
    print("probe selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    m = sub.add_parser("measure", help="parse compiled artifacts (seconds, no GPU)")
    m.add_argument("--dir", required=True, help="dir with .amdgcn/.s (+ Triton metadata json)")
    m.add_argument("--json")
    # The remedy for every LDS refusal in this file. There is NO default: a default here would
    # be a guess wearing a flag's clothes, and the plausible guess (65536) is right on gfx94*
    # and 2.5x wrong on gfx95*, so it would never look wrong. Read it off the part
    # (`rocminfo` -> Segment: GROUP, Size) and pass it, or accept the withheld verdict.
    _lds_help = ("LDS bytes per CU, when the target is not named in the dump or is unknown to "
                 "the shared model. No default -- without it the LDS limiter is WITHHELD, "
                 "never guessed.")
    m.add_argument("--lds-per-cu", type=int, default=None, help=_lds_help)
    m.add_argument("--arch", default=None,
                   help="gfx target for dumps that do NOT name one (a dump's .amdgcn_target always "
                        "wins). No default: an unnamed dump without it has its register term "
                        "withheld, e.g. --arch gfx950")
    p = sub.add_parser("plan", help="tile plan -> resident VGPR -> reachable occupancy")
    p.add_argument("--lds-per-cu", type=int, default=None, help=_lds_help)
    p.add_argument("--warps", type=int, required=True)
    p.add_argument("--acc", nargs="*", help="accumulator tiles, name=MxN (fp32 by default)")
    p.add_argument("--operand", nargs="*", help="resident operand tiles, name=MxN")
    p.add_argument("--acc-bits", type=int, default=32)
    p.add_argument("--dtype-bits", type=int, default=16)
    p.add_argument("--target-waves", type=int, default=2,
                   help="occupancy you are trying to reach (default 2)")
    p.add_argument("--arch", required=True,
                   help="gfx target for the wave width + register file, e.g. gfx950 (REQUIRED, "
                        "no default: CDNA and RDNA differ 2-3x; `measure` reads it from the "
                        "artifact instead)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if a.cmd == "measure":
        return cmd_measure(a)
    if a.cmd == "plan":
        return cmd_plan(a)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main() or 0)
