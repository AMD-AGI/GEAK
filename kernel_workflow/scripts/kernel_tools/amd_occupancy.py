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

PREFER `llvm_occupancy()` FOR THE REGISTER TERM: LLVM already emits `; Occupancy: N` in the
resource-usage comment block of a `.s`, computed by the backend for that exact subtarget. When
the dump has it, it is the authority **for the register term** and this module's table is only
the fallback for KD-less disassembly or for planning a tile that has not been compiled yet.

It is NOT, on its own, the occupancy of a kernel that uses LDS. The compiler prints its own
disclaimer two lines above it -- `; LDSByteSize: 0 bytes/workgroup (compile time only)` -- and
on a JIT'd kernel whose group segment is sized at launch that zero is what the occupancy number
was computed against. `occupancy_from_asm()` therefore folds in an LDS term whenever it can
OBSERVE one, and says so in `source`; when it cannot, it says that too rather than printing a
register-only number as if it were the answer.

Why folding it in is safe without settling an open question: whether a STATICALLY allocated
group segment is already reflected in the emitted `; Occupancy: N` is not verified here, and
this module does not assume either way -- it does not need to. `min(register_term, lds_term)`
is correct under BOTH hypotheses: if the emitter already folded the LDS term in, its number is
already <= the LDS term and the min is a no-op; if it did not, the min is the correction. The
one thing that is never safe is what this module used to do -- report the register term alone,
unlabelled, when the LDS term is the smaller of the two.

Provenance of the tables (`basis: compiler-derived`, reproducible in seconds, independent of
any kernel or measurement):

    for v in $(seq 1 256); do
      clob=$(python3 -c "print(','.join('~{v%d}'%i for i in range($v)))")
      printf 'target triple = "amdgcn-amd-amdhsa"\\n
      define amdgpu_kernel void @k(ptr addrspace(1) %%o) #0 {\\n
        call void asm sideeffect "", "%s"()\\n store float 1.0, ptr addrspace(1) %%o\\n ret void\\n}\\n
      attributes #0 = { "amdgpu-flat-work-group-size"="256,256" }\\n' "$clob" > t.ll
      llc -mtriple=amdgcn-amd-amdhsa -mcpu=<arch> t.ll -o - | grep -E "NumVgprs|Occupancy"
    done

Verified against ROCm 7.2.1 / LLVM 22 for gfx90a, gfx942, gfx950, gfx1100, gfx1151, gfx1200,
gfx1201, gfx1250. The gfx1201 table was regenerated unchanged with AMD clang/LLVM 23 from the
ROCm 10 R9700 image on 2026-09-21. Every step below is a measured compiler breakpoint.

Usage (kernel_workflow/scripts/kernel_tools/; --arch is required unless --asm names the target):
  python3 amd_occupancy.py --vgpr 168 --arch gfx950      # model lookup (register term only)
  python3 amd_occupancy.py --vgpr 249 --arch gfx1201     # the RDNA geometry this module exists for
  python3 amd_occupancy.py --asm kernel.s                # KD + LLVM's answer + observable LDS
  python3 amd_occupancy.py --asm kernel.s --lds-bytes-per-wg 38144 --workgroup-size 256
  python3 amd_occupancy.py --compiler-sweep --arch gfx1201 --format json
  python3 amd_occupancy.py --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# Register-file geometry per ISA family. Keep in sync with perf_knowledge/hardware/data/
# hw_constants.json, read through the sibling `_hwdata` locator (`--selftest` cross-checks the
# two when the data is reachable).
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
_LLVM_NUM_VGPR_RE = re.compile(r"^;\s*NumVgprs:\s*(\d+)\s*$", re.M)

# LDS per workgroup, in the three places a dump can carry it. Order matters: the KD directive
# and the metadata field are the ALLOCATION; the `; LDSByteSize:` comment is the compiler's own
# accounting and carries its own disclaimer suffix, which is why it is read last and its
# "(compile time only)" tail is preserved into the label rather than dropped.
_LDS_KD_RE = re.compile(r"^\s*\.amdhsa_group_segment_fixed_size\s+(\d+)", re.M)
_LDS_META_RE = re.compile(r"^\s*\.group_segment_fixed_size:\s*(\d+)", re.M)
_LDS_COMMENT_RE = re.compile(r"^;\s*LDSByteSize:\s*(\d+)\s*bytes/workgroup(.*)$", re.M)
# The compile-time BOUND on workgroup size, not the launch geometry. It is what the backend
# itself reasons with, so it is the right default and the wrong thing to state as fact.
_MAXFLAT_RE = re.compile(r"^\s*\.max_flat_workgroup_size:\s*(\d+)", re.M)


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


def lds_bytes_per_wg_from_asm(text):
    """(bytes, field) LDS per workgroup as the dump ACTUALLY states it, or (None, reason).

    Three fields, in descending order of being the allocation rather than a report. A dump that
    names none of them is not a dump of a zero-LDS kernel -- it is a dump that does not say, and
    the two must not be collapsed, which is why the miss returns None and not 0.
    """
    m = _LDS_KD_RE.search(text)
    if m:
        return int(m.group(1)), ".amdhsa_group_segment_fixed_size (KD)"
    m = _LDS_META_RE.search(text)
    if m:
        return int(m.group(1)), ".group_segment_fixed_size (metadata)"
    m = _LDS_COMMENT_RE.search(text)
    if m:
        return int(m.group(1)), f"; LDSByteSize{(m.group(2) or '').strip()}"
    return None, ("no group-segment field in this dump -- LDS per workgroup is unobservable "
                  "here; pass --lds-bytes-per-wg")


def workgroup_size_from_asm(text):
    """(threads, field) from `.max_flat_workgroup_size`, or (None, reason).

    This is the compile-time BOUND, not the launch geometry. It is returned labelled as such so
    a caller can override it; a kernel launched at half its declared maximum has twice the
    workgroups per CU that this implies.
    """
    m = _MAXFLAT_RE.search(text)
    if m:
        return int(m.group(1)), ".max_flat_workgroup_size (compile-time bound, not the launch)"
    return None, "no .max_flat_workgroup_size in this dump"


def waves_by_lds(lds_bytes_per_wg, arch, workgroup_size, model=None):
    """(waves_per_simd, detail) LDS-limited occupancy in waves/SIMD, or (None, why-not).

    Needs three things the register model does not: LDS/CU for the arch, SIMDs/CU for the arch,
    and the workgroup size. Any one missing returns None WITH the reason -- never a substituted
    default, on the same principle as `waves_by_vgpr` on an unknown arch.

    Unit note, because this is where the two terms are usually mixed: the LDS limit is naturally
    a workgroups/CU number and the register limit is a waves/SIMD number. They cannot be
    compared until one is converted. The conversion is
        waves/SIMD = wg/CU * (workgroup_size / wave_size) / simds_per_cu
    floored, i.e. the even-distribution figure; a workgroup is indivisible, so when the waves do
    not divide evenly across the SIMDs the real per-SIMD count is uneven and this floor is the
    conservative end of it. Conservative is the right direction for a veto criterion.
    """
    m = model or model_for(arch)
    if m is None:
        return None, f"unknown arch {arch!r}: no register/LDS geometry on record"
    if not lds_bytes_per_wg:
        return None, "LDS per workgroup is 0 or unknown: no LDS term"
    cap = lds_per_cu(arch)
    if not cap:
        return None, (f"no lds_per_cu_kib for {arch} in hw_constants.json -- on RDNA this is "
                      f"deliberate (shared memory is per-WGP with a separate per-WG cap); read "
                      f"lds_per_wgp_kib and lds_per_wg_kib yourself")
    simds = simds_per_cu(arch)
    if not simds:
        return None, (f"no simds_per_cu for {arch} in hw_constants.json: the wg/CU -> waves/SIMD "
                      f"conversion is not derivable, so no LDS term is reported for this arch")
    if not workgroup_size:
        return None, "workgroup size unknown: wg/CU cannot be converted to waves/SIMD"
    wg_per_cu = int(cap) // int(lds_bytes_per_wg)
    waves_per_wg = int(workgroup_size) // int(m["wave_size"])
    if wg_per_cu < 1:
        return 0, (f"{lds_bytes_per_wg} B/workgroup exceeds the {cap} B LDS of one CU on {arch}: "
                   f"this kernel does not launch")
    if waves_per_wg < 1:
        return None, (f"workgroup of {workgroup_size} threads is under one wave of "
                      f"{m['wave_size']} on {arch}")
    waves = (wg_per_cu * waves_per_wg) // int(simds)
    detail = (f"{cap} B LDS/CU // {lds_bytes_per_wg} B/wg = {wg_per_cu} wg/CU; "
              f"x {waves_per_wg} waves/wg / {simds} SIMD/CU = {waves} waves/SIMD")
    # A small LDS footprint divides out to more waves than the SIMD can hold. That figure is a
    # true statement about LDS ("not the limiter here") but a false one about occupancy, and it
    # is returned as occupancy -- so it is clamped at the hardware cap. Without this, an arm
    # whose register term is missing reports the raw quotient as its waves/SIMD, which is the
    # one direction this whole module is built to refuse: a number above what the part can do.
    hw_cap = m.get("max_waves_per_simd")
    if hw_cap and waves > hw_cap:
        return hw_cap, (f"{detail}, clamped to the {hw_cap}-wave/SIMD hardware maximum on "
                        f"{arch} -- at {lds_bytes_per_wg} B/wg, LDS is not the limiter")
    return waves, detail


def occupancy_from_asm(text, vgpr=None, lds_bytes_per_wg=None, workgroup_size=None):
    """(waves, source, label) -- the binding occupancy, register AND LDS terms.

    `source` names WHICH terms went into the number, because the old single word did not:
        'llvm-comment'      register term from LLVM's own `; Occupancy:`, NO LDS term folded in
        'model'             register term from this module's table, NO LDS term
        '<above>+lds'       the above, min'd against an observed LDS term
        'lds'               the LDS term binds outright
        'unknown'           no answer; read the label

    A bare 'llvm-comment' or 'model' is a REGISTER-TERM answer and nothing more. The label says
    so, including when the LDS term could not be computed and why. See the module docstring for
    why min'ing the two is safe without settling whether the emitter already folded LDS in.
    """
    arch = arch_from_asm(text)
    occ = llvm_occupancy(text)
    m = model_for(arch)
    label = m["label"] if m else f"unknown arch {arch!r}"

    if occ is not None:
        reg, reg_src = occ, "llvm-comment"
        reg_label = f"{label}; LLVM `; Occupancy:` in this dump (register term)"
    else:
        reg, reg_label = waves_by_vgpr(vgpr, arch=arch, model=m)
        reg_src = "model" if reg is not None else "unknown"

    if lds_bytes_per_wg is None:
        lds_bytes_per_wg, lds_field = lds_bytes_per_wg_from_asm(text)
    else:
        lds_field = "caller-supplied"
    if workgroup_size is None:
        workgroup_size, wg_field = workgroup_size_from_asm(text)
    else:
        wg_field = "caller-supplied"

    lds_waves, lds_detail = waves_by_lds(lds_bytes_per_wg, arch, workgroup_size, model=m)
    if lds_waves is None:
        note = f"NO LDS TERM ({lds_detail}) -- this is a register-term answer"
        if lds_bytes_per_wg:
            note += f"; observed {lds_bytes_per_wg} B/wg from {lds_field} was NOT applied"
        elif lds_bytes_per_wg == 0:
            note += (f"; {lds_field} reports 0 B/wg, which on a kernel that sizes its group "
                     f"segment at launch means the LDS term is missing, not absent")
        return reg, reg_src, f"{reg_label} | {note}"

    if reg is None:
        return lds_waves, "lds", f"{reg_label} | LDS term binds: {lds_detail} [{lds_field}]"
    waves = min(reg, lds_waves)
    binder = "LDS" if lds_waves < reg else ("register" if reg < lds_waves else "both (tied)")
    return waves, f"{reg_src}+lds", (f"{reg_label} | LDS term: {lds_detail} "
                                     f"[{lds_field}; wg size from {wg_field}] | "
                                     f"min -> {waves} waves/SIMD, {binder}-bound")


# ---------------------------------------------------------------- compiler-derived sweep
def _sweep_ir(clobbered_vgprs):
    clobbers = ",".join(f"~{{v{i}}}" for i in range(int(clobbered_vgprs)))
    return (
        'target triple = "amdgcn-amd-amdhsa"\n'
        'define amdgpu_kernel void @k(ptr addrspace(1) %o) #0 {\n'
        f'  call void asm sideeffect "", "{clobbers}"()\n'
        '  store float 1.0, ptr addrspace(1) %o\n'
        '  ret void\n'
        '}\n'
        'attributes #0 = { "amdgpu-flat-work-group-size"="256,256" }\n'
    )


def _compress_sweep(records):
    """Return [max NumVgprs, waves/SIMD] breakpoints from compiler records."""
    by_vgpr = {}
    for row in records:
        by_vgpr[int(row["num_vgprs"])] = int(row["occupancy"])
    ordered = sorted(by_vgpr.items())
    if not ordered:
        return []
    steps = []
    for index, (vgpr, waves) in enumerate(ordered):
        next_waves = ordered[index + 1][1] if index + 1 < len(ordered) else None
        if next_waves != waves:
            steps.append([vgpr, waves])
    return steps


def _default_rocm_compiler():
    """Prefer a ROCm-bundled LLVM; never silently use an unrelated system clang."""
    for path in (
        "/opt/rocm/llvm/bin/llc",
        "/opt/rocm/llvm/bin/clang",
    ):
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    for name in ("amdclang", "llc", "clang"):
        path = shutil.which(name)
        real = os.path.realpath(path) if path else ""
        if path and ("rocm" in real.lower() or "_rocm_sdk" in real.lower()):
            return path
    return None


def compiler_sweep(arch, compiler=None, max_clobbered_vgprs=256):
    """Run the documented LLVM clobber sweep and return raw rows + breakpoints."""
    binary = compiler or _default_rocm_compiler()
    if not binary:
        raise RuntimeError(
            "a ROCm-bundled llc/clang was not found (pass --compiler explicitly)"
        )
    version_proc = subprocess.run(
        [binary, "--version"], check=False, capture_output=True, text=True
    )
    if version_proc.returncode != 0:
        raise RuntimeError(f"{binary} --version exited {version_proc.returncode}")
    version = (version_proc.stdout or version_proc.stderr).strip()
    compiler_name = os.path.basename(os.path.realpath(binary))
    is_clang = "clang" in compiler_name
    records = []
    for count in range(1, int(max_clobbered_vgprs) + 1):
        command = (
            [binary, "-x", "ir", "--target=amdgcn-amd-amdhsa",
             f"-mcpu={arch}", "-S", "-o", "-", "-"]
            if is_clang
            else [binary, "-mtriple=amdgcn-amd-amdhsa", f"-mcpu={arch}", "-o", "-"]
        )
        proc = subprocess.run(
            command,
            input=_sweep_ir(count),
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip().splitlines()
            raise RuntimeError(
                f"llc sweep failed at {count} clobbered VGPRs: "
                f"{detail[-1] if detail else 'unknown error'}"
            )
        vgpr_match = _LLVM_NUM_VGPR_RE.search(proc.stdout)
        occ_match = _LLVM_OCC_RE.search(proc.stdout)
        if not vgpr_match or not occ_match:
            raise RuntimeError(
                f"llc output at {count} clobbered VGPRs omitted NumVgprs/Occupancy"
            )
        records.append(
            {
                "clobbered_vgprs": count,
                "num_vgprs": int(vgpr_match.group(1)),
                "occupancy": int(occ_match.group(1)),
            }
        )
    return {
        "arch": arch,
        "compiler": os.path.realpath(binary),
        "compiler_version": version,
        "max_clobbered_vgprs": int(max_clobbered_vgprs),
        "records": records,
        "vgpr_wave_steps": _compress_sweep(records),
    }


# --------------------------------------------------------------------------------- reference SoT
def _load_hwdata():
    """The sibling `_hwdata` locator (kernel_workflow/scripts/kernel_tools/_hwdata.py), or None."""
    try:
        import _hwdata  # noqa: PLC0415
        return _hwdata
    except ImportError:
        pass
    path = os.path.join(HERE, "_hwdata.py")
    if not os.path.isfile(path):
        return None
    import importlib.util
    spec = importlib.util.spec_from_file_location("_hwdata", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:  # noqa: BLE001 -- a broken locator is "no reference", not a crash
        return None
    sys.modules.setdefault("_hwdata", mod)
    return mod


def _find_hw_constants():
    """Path of the shared `hw_constants.json`, or None.

    Resolved ONLY through `_hwdata.find()` -- `$GEAK_HW_DATA_DIR`, then this checkout's
    `perf_knowledge/hardware/data/`, then a bounded (fixed-depth, no glob) ancestor check, then
    the copied-out-alone spots beside the tool. It never walks the filesystem, which is the
    property the earlier lookups each lost in their own direction:

      * a `glob(HERE/../../**/hw_constants.json, recursive=True)` HANGS once the tool is
        copied to a shallow directory (`HERE/../..` is `/`) -- silently, with no message;
      * an unbounded walk UP the ancestors accepts any stray tree it passes, so a leftover
        `/tmp/...` from an unrelated run becomes the source of truth.

    With `_hwdata` unreachable (this file copied out with nothing beside it) only the explicit
    beside-the-tool spots are tried. None is a real answer every caller already handles --
    `probe.py` reports the LDS side as unavailable rather than measuring against another
    generation's divisor.
    """
    hd = _load_hwdata()
    if hd is not None:
        p = hd.find("hw_constants.json")
        return str(p) if p else None
    for c in (os.path.join(HERE, "data", "hw_constants.json"),
              os.path.join(HERE, "references", "hardware", "hw_constants.json")):
        if os.path.isfile(c):
            return c
    return None


def _arch_facts(arch):
    """The `arch.<gfx>` block of hw_constants.json for `arch`, longest-prefix matched, or None.

    Every per-arch fact this module needs beyond the register geometry (LDS/CU, SIMDs/CU) comes
    from here rather than from a table duplicated in this file, so an arch that the reference
    does not describe yields None and the caller reports "not derivable" instead of guessing.
    """
    if not arch:
        return None
    path = _find_hw_constants()
    if not path:
        return None
    try:
        with open(path) as f:
            table = json.load(f).get("arch", {})
    except (OSError, ValueError):
        return None
    for name in sorted(table, key=len, reverse=True):
        if str(arch).startswith(name):
            return table[name]
    return None


def _assert_rung_is_maximal(arch, hi, want):
    """A ladder rung is a CLAIM OF MAXIMALITY: `hi` is the LAST vgpr count still fitting `want`
    waves. Checking only `waves_by_vgpr(hi) == want` cannot detect a rung set too LOW, because
    a too-low rung still yields the right wave count -- which is exactly how `[255, 2]` shipped
    in hw_constants.json for both CDNA arches while every selftest passed. The assertion that
    catches it is that `hi + 1` must yield FEWER waves.

    One legitimate exemption: a rung capped by the per-WAVE addressable limit rather than by
    the file division. On RDNA `vgpr_per_wave` is 256, so the 256 rung has no legal successor
    and `hi + 1` is not a counterexample -- it is not a reachable allocation at all. Applying
    the check without this exemption reports all four RDNA tables as broken; they are correct.
    """
    m = model_for(arch)
    cap = m.get("vgpr_per_wave") if m else None
    if cap and hi + 1 > cap:
        return
    nxt, _ = waves_by_vgpr(hi + 1, arch)
    assert nxt is not None and nxt < want, (
        f"{arch} rung [{hi}, {want}] is NOT MAXIMAL: vgpr={hi + 1} still yields {nxt} waves, so "
        f"the true rung is higher. A ladder rung must be the LAST count that fits. This is the "
        f"off-by-one that makes a consumer's `resident <= v` lookup miss and declare the "
        f"occupancy axis dead one step early.")


def _selftest():
    # Step tables measured from LLVM (see module docstring). Each entry is (max_vgpr, waves):
    # the LAST VGPR count that still fits `waves` waves/SIMD.
    steps = {
        "gfx942":  [(64, 8), (72, 7), (80, 6), (96, 5), (128, 4), (168, 3), (256, 2)],
        "gfx950":  [(64, 8), (72, 7), (80, 6), (96, 5), (128, 4), (168, 3), (256, 2)],
        "gfx90a":  [(64, 8), (72, 7), (80, 6), (96, 5), (128, 4), (168, 3), (256, 2)],
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
            _assert_rung_is_maximal(arch, hi, want)
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
    assert _compress_sweep([
        {"num_vgprs": 24, "occupancy": 16},
        {"num_vgprs": 48, "occupancy": 16},
        {"num_vgprs": 72, "occupancy": 12},
        {"num_vgprs": 96, "occupancy": 12},
    ]) == [[48, 16], [96, 12]]

    # The reference lookup goes through `_hwdata` only, and an explicit override wins.
    import tempfile
    _env_saved = os.environ.get("GEAK_HW_DATA_DIR")
    try:
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "hw_constants.json"), "w") as f:
                f.write("{}")
            os.environ["GEAK_HW_DATA_DIR"] = td
            got = _find_hw_constants()
            if _load_hwdata() is not None:
                assert got == os.path.join(td, "hw_constants.json"), \
                    f"$GEAK_HW_DATA_DIR must win, got {got}"
            os.environ["GEAK_HW_DATA_DIR"] = os.path.join(td, "absent")
            got2 = _find_hw_constants()
            assert got2 is None or not got2.startswith(td), \
                "an override to a missing dir must fall through, never invent a path"
    finally:
        if _env_saved is None:
            os.environ.pop("GEAK_HW_DATA_DIR", None)
        else:
            os.environ["GEAK_HW_DATA_DIR"] = _env_saved
    assert _find_hw_constants() == _find_hw_constants(), "lookup must be deterministic"

    # ---- the LDS term -------------------------------------------------------------------
    # Field extraction: all three spellings, in priority order, and a miss that is None (not 0).
    assert lds_bytes_per_wg_from_asm("\t.amdhsa_group_segment_fixed_size 38144\n")[0] == 38144
    assert lds_bytes_per_wg_from_asm("    .group_segment_fixed_size: 1024\n")[0] == 1024
    b, f = lds_bytes_per_wg_from_asm("; LDSByteSize: 0 bytes/workgroup (compile time only)\n")
    assert b == 0 and "compile time only" in f, (b, f)
    assert lds_bytes_per_wg_from_asm("s_nop 0\n")[0] is None
    assert workgroup_size_from_asm("    .max_flat_workgroup_size: 256\n")[0] == 256
    assert workgroup_size_from_asm("s_nop 0\n")[0] is None

    # The conversion itself, on the one family whose simds_per_cu the reference states.
    # 160 KiB/CU // 38144 B = 4 wg/CU; x 4 waves/wg (256 threads / wave64) / 4 SIMD = 4 waves/SIMD.
    if simds_per_cu("gfx950") == 4 and lds_per_cu("gfx950") == 160 * 1024:
        lw, detail = waves_by_lds(38144, "gfx950", 256)
        assert lw == 4, (lw, detail)
        # A workgroup that does not fit at all is 0 resident, not a positive number.
        assert waves_by_lds(200000, "gfx950", 256)[0] == 0
        # THE REGRESSION THIS BLOCK EXISTS TO PREVENT: a register term of 8 waves/SIMD on a
        # kernel whose LDS admits 4 must not be reported as 8. min() binds regardless of
        # whether the emitter already folded LDS in -- that question stays open by design.
        lds_asm = ('\t.amdgcn_target "amdgcn-amd-amdhsa--gfx950"\n'
                   '\t\t.amdhsa_next_free_vgpr 64\n'
                   '\t\t.amdhsa_group_segment_fixed_size 38144\n'
                   '; Occupancy: 8\n'
                   '    .max_flat_workgroup_size: 256\n')
        waves, src, label = occupancy_from_asm(lds_asm)
        assert (waves, src) == (4, "llvm-comment+lds"), (waves, src, label)
        assert "LDS-bound" in label, label
        # Idempotence of the min under the open hypothesis: an emitter that HAD folded LDS in
        # would have printed 4, and the same call must still return 4 with the same source.
        assert occupancy_from_asm(lds_asm.replace("; Occupancy: 8", "; Occupancy: 4"))[0] == 4
        # A register term that binds below the LDS term must survive unchanged.
        assert occupancy_from_asm(lds_asm.replace("; Occupancy: 8", "; Occupancy: 2"))[:2] \
            == (2, "llvm-comment+lds")
        # Caller-supplied LDS overrides a dump that says 0 (the JIT case).
        jit_asm = ('\t.amdgcn_target "amdgcn-amd-amdhsa--gfx950"\n'
                   '\t\t.amdhsa_group_segment_fixed_size 0\n; Occupancy: 8\n')
        w0, s0, l0 = occupancy_from_asm(jit_asm)
        assert (w0, s0) == (8, "llvm-comment") and "NO LDS TERM" in l0, (w0, s0, l0)
        assert "missing, not absent" in l0, l0
        assert occupancy_from_asm(jit_asm, lds_bytes_per_wg=38144, workgroup_size=256)[0] == 4

    # An arch the reference does not give simds_per_cu for must report NO LDS term rather than
    # assume a CU width. gfx1201 is the live case: RDNA shared memory is per-WGP.
    lw, why = waves_by_lds(16384, "gfx1201", 256)
    assert lw is None and ("lds_per_cu_kib" in why or "simds_per_cu" in why), (lw, why)

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
                # The SHIPPED facts get the maximality check too, not just the table above.
                # hw_constants.json states the contract itself in `_vgpr_occupancy_basis`
                # ("the LAST VGPR count that still fits that many waves"); this enforces it.
                _assert_rung_is_maximal(arch, hi, want)
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
    ap.add_argument("--compiler-sweep", action="store_true",
                    help="derive NumVgprs/Occupancy breakpoints by invoking llc")
    ap.add_argument("--compiler", "--llc", dest="compiler", default=None,
                    help="LLVM llc or clang binary for --compiler-sweep")
    ap.add_argument("--max-clobbered-vgprs", type=int, default=256,
                    help="highest v-register clobber count in the compiler sweep")
    ap.add_argument("--format", choices=("json", "tsv"), default="json",
                    help="compiler-sweep output format")
    ap.add_argument("--lds-bytes-per-wg", type=int, default=None,
                    help="LDS bytes per workgroup. REQUIRED for a JIT'd kernel that sizes its "
                         "group segment at launch -- the dump's 0 is not that kernel's LDS")
    ap.add_argument("--workgroup-size", type=int, default=None,
                    help="threads per workgroup at LAUNCH; overrides .max_flat_workgroup_size, "
                         "which is only the compile-time bound")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.compiler_sweep:
        if not a.arch:
            raise SystemExit("--compiler-sweep requires --arch")
        result = compiler_sweep(a.arch, a.compiler, a.max_clobbered_vgprs)
        if a.format == "json":
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print("clobbered_vgprs\tnum_vgprs\toccupancy")
            for row in result["records"]:
                print(f"{row['clobbered_vgprs']}\t{row['num_vgprs']}\t{row['occupancy']}")
            print("# vgpr_wave_steps=" + json.dumps(result["vgpr_wave_steps"]))
            print("# compiler_version=" + json.dumps(result["compiler_version"]))
        return 0
    if a.asm:
        text = open(a.asm, errors="ignore").read()
        vgpr = a.vgpr
        if vgpr is None:
            m = re.search(r"^\s*\.amdhsa_next_free_vgpr\s+(\d+)", text, re.M)
            vgpr = int(m.group(1)) if m else None
        arch = arch_from_asm(text)
        waves, src, label = occupancy_from_asm(text, vgpr, a.lds_bytes_per_wg, a.workgroup_size)
        obs_lds, lds_field = ((a.lds_bytes_per_wg, "caller-supplied")
                              if a.lds_bytes_per_wg is not None else lds_bytes_per_wg_from_asm(text))
        print(f"arch    = {arch}  (wave32 KD flag: {wave32_from_asm(text)})")
        print(f"vgpr    = {vgpr}  (printed; the hardware reads "
              f"ceil8(ceil4(vgpr)) = {((int(vgpr) + 3) // 4 * 4 + 7) // 8 * 8 if vgpr else None})")
        print(f"lds/wg  = {obs_lds}  [{lds_field}]")
        print(f"waves/SIMD = {waves}  [{src}]")
        print(f"model   = {label}")
        if src in ("llvm-comment", "model"):
            print("WARNING: register term only -- no LDS term was folded in. If this kernel "
                  "uses LDS, this is an UPPER BOUND, not its occupancy; re-run with "
                  "--lds-bytes-per-wg (and --workgroup-size).")
        return 0
    if a.arch is None:
        raise SystemExit("need --arch (with --vgpr) or --asm; see --help")
    waves, label = waves_by_vgpr(a.vgpr, a.arch)
    print(f"waves/SIMD by VGPR = {waves}   <- REGISTER TERM ONLY")
    print(f"model = {label}")
    if a.lds_bytes_per_wg:
        lw, detail = waves_by_lds(a.lds_bytes_per_wg, a.arch, a.workgroup_size)
        print(f"waves/SIMD by LDS  = {lw}   ({detail})")
        if lw is not None and waves is not None:
            binder = "LDS" if lw < waves else ("register" if waves < lw else "both (tied)")
            print(f"binding occupancy  = {min(waves, lw)} waves/SIMD ({binder}-bound)")
    else:
        print("no --lds-bytes-per-wg given: the LDS term is absent from this answer, not zero.")
    return 0


# --------------------------------------------------------------------------- #
# LDS capacity per CU -- the OCCUPANCY divisor for shared memory:
#     WGs/CU <= LDS_per_CU // lds_bytes_per_wg
#
# Read from perf_knowledge/hardware/data/hw_constants.json (via `_hwdata`) rather than duplicated here. That file is
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
    facts = _arch_facts(arch)
    if not facts:
        return None
    kib = facts.get("lds_per_cu_kib")
    return int(kib) * 1024 if kib else None


def simds_per_cu(arch):
    """SIMDs per CU for `arch`, or None when the reference does not state it.

    The wg/CU -> waves/SIMD conversion divides by this, so a guessed 4 on an arch whose CU is
    not 4-wide would misreport every LDS-limited occupancy on that target by the ratio. None
    propagates into `waves_by_lds` as "no LDS term for this arch", which is the honest output.
    """
    return (_arch_facts(arch) or {}).get("simds_per_cu")


if __name__ == "__main__":
    sys.exit(main())
