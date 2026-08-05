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
  * VGPR allocation granularity 8; LDS 64 KiB/CU; wave64.

The register geometry is NOT the same on RDNA (wave32, a 1536-entry file per SIMD, 256 per
wave, no AGPRs) -- `measure` reads the target out of the artifact, and `plan` takes `--arch`.
Both print the model they used; applying the CDNA numbers to an RDNA tile under-reports
occupancy by 2-3x and invents a register wall that is not there.

Usage:
  # measure -- from a compiled artifact directory (dump_ir.sh output, or a Triton cache dir)
  python3 probe.py measure --dir exp/round_3/capture/ir
  python3 probe.py measure --dir ~/.triton/cache --json probe.json

  # plan -- from a tile plan, before compiling
  python3 probe.py plan --warps 4 --dtype-bits 16 --acc-bits 32 \\
      --acc  dk=128x128 dv=128x128 --operand k=128x128 v=128x128
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

DEFAULT_ARCH = "gfx950"    # only used when nothing names a target; always printed
VGPR_PER_SIMD = 512        # CDNA: ArchVGPR + AGPR COMBINED
VGPR_GRANULE = 8
# LDS/CU lives in the shared vendor/amd occupancy model, which is where the arch dispatch
# already is. It is a PER-ARCH table there and not a family field, because gfx94*/gfx95* share
# the register geometry but not the LDS (64 KiB vs 160 KiB).
def lds_per_cu(arch, default=65536):
    """LDS bytes per CU for `arch`; `default` only when the shared model has no figure."""
    if _OCC is not None and hasattr(_OCC, "lds_per_cu"):
        v = _OCC.lds_per_cu(arch)
        if v:
            return v
    return default


def simds_per_cu(arch, default=4):
    """SIMDs per CU for `arch`, from the same per-arch reference the divisor comes from.

    Needed because waves/SIMD is not an answer on its own: the register limit becomes a
    WORKGROUP limit only after multiplying by the SIMDs and dividing by the waves a workgroup
    occupies, and on a wave64 kernel with num_warps=8 that arithmetic can pin occupancy at
    1 WG/CU while the LDS figure still reports twenty. 4 on every CDNA/RDNA part shipped so far.
    """
    if _OCC is not None and hasattr(_OCC, "_find_hw_constants"):
        path = _OCC._find_hw_constants()
        if path:
            try:
                with open(path) as f:
                    table = json.load(f).get("arch", {})
            except (OSError, ValueError):
                return default
            for name in sorted(table, key=len, reverse=True):
                if str(arch).startswith(name):
                    return table[name].get("simds_per_cu") or default
    return default


    for k, v in LDS_PER_CU_BY_ARCH.items():
        if arch.startswith(k):
            return v
    return default


LDS_PER_CU = 65536         # fallback only; prefer lds_per_cu(<arch>)
WAVE = 64
MAX_WAVES_PER_SIMD = 8


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
}


def _from_asm(path: str) -> dict | None:
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
    # left its `; Occupancy:` in the dump that is the answer -- it beats any model here.
    arch = _OCC.arch_from_asm(txt) if _OCC else None
    llvm_occ = _OCC.llvm_occupancy(txt) if _OCC else None
    return {"name": os.path.basename(path), "vgpr_arch": g["vgpr"], "agpr": g["agpr"],
            "vgpr_combined": combined, "spill_bytes": spill, "arch": arch,
            "waves_per_simd": llvm_occ if llvm_occ is not None else waves_by_vgpr(combined, arch),
            "waves_source": "llvm-comment" if llvm_occ is not None else "model"}


def _meta_from_dir(d: str) -> dict[str, dict]:
    """Per-kernel Triton cache metadata: `shared` bytes/WG plus the launch geometry.

    `shared` is the ONLY correct LDS source for a Triton kernel -- the KD's
    group_segment_fixed_size and rocprof-compute 7.1.8 are structurally 0 (shared memory is
    sized dynamically at launch), so neither may be substituted and a 0 from those is not
    evidence of 'no LDS'. `num_warps` comes along because the register limit cannot be turned
    into a workgroup limit without it.
    """
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
                tgt = j.get("target") or {}
                out[str(j.get("name") or f)[:44]] = {
                    "shared": j["shared"],
                    "num_warps": j.get("num_warps"),
                    "arch": tgt.get("arch") if isinstance(tgt, dict) else None,
                }
    return out


def cmd_measure(a):
    asms = []
    for root, _dirs, files in os.walk(a.dir):
        asms += [os.path.join(root, f) for f in files
                 if f.endswith((".amdgcn", ".s")) or f.endswith("_final_isa.s")]
    kernels = [r for r in (_from_asm(p) for p in sorted(asms)) if r]
    meta = _meta_from_dir(a.dir)
    lds = {n: m["shared"] for n, m in meta.items()}

    print(f"=== probe measure: {a.dir} ===")
    if not kernels:
        print("  no .amdgcn/.s with a register budget found -- nothing to measure "
              "(dump one with dump_ir.sh). NOT reporting zeros.")
    for k in kernels:
        spill = "n/a" if k["spill_bytes"] is None else f"{k['spill_bytes']} B"
        warn = "  <- SPILL" if (k["spill_bytes"] or 0) > 0 else ""
        src = "" if k["waves_source"] == "model" else " [LLVM]"
        arch = k["arch"] or f"{DEFAULT_ARCH}-assumed"
        print(f"  {k['name'][:44]:46s} [{arch}] vgpr={k['vgpr_combined']:4d} "
              f"waves/SIMD={k['waves_per_simd']}{src}  spill={spill}{warn}")
    print()
    if lds:
        # The divisor is per-arch (64 KiB on CDNA3, 160 KiB on CDNA4), so it has to come from
        # the arch the artifact was actually built for. Taking the module fallback here reads a
        # gfx950 kernel against the CDNA3 figure and overstates its LDS pressure by 2.5x --
        # which lands as "0 WGs/CU" on a kernel that runs, i.e. as a blocker that is not real.
        asm_arch = next((k["arch"] for k in kernels if k["arch"]), None)
        arch = asm_arch or next((m["arch"] for m in meta.values() if m["arch"]), None) or DEFAULT_ARCH
        cap = lds_per_cu(arch)
        simds = simds_per_cu(arch)
        basis = f"{arch}" if (asm_arch or arch != DEFAULT_ARCH) else f"{arch}-assumed"
        # Registers and LDS both cap WORKGROUPS per CU, and reporting only the LDS side invites the
        # error this tool exists to prevent: on one measured kernel the arm with the most LDS
        # headroom (20 WGs/CU) was the SLOWEST, because 2 waves/SIMD x 4 SIMDs = 8 waves/CU against
        # a num_warps=8 workgroup already pinned it to 1. So print both and name which one binds.
        wps = {k["waves_per_simd"] for k in kernels if k["waves_per_simd"]}
        waves = wps.pop() if len(wps) == 1 else None
        print(f"  LDS/CU basis: {cap} B [{basis}]  SIMDs/CU={simds}"
              + ("" if waves else "   (waves/SIMD differs across kernels -- register side per-kernel)"))
        for n, b in sorted(lds.items(), key=lambda kv: -kv[1]):
            nw = meta[n]["num_warps"]
            by_lds = cap // b if b else None
            by_reg = (waves * simds) // nw if (waves and nw) else None
            row = f"  {n:46s} lds/WG={b:6d} B  LDS<={by_lds if by_lds is not None else '-'}"
            if by_reg is not None:
                binder = ("REGISTERS" if by_reg < by_lds else
                          "LDS" if by_lds < by_reg else "both")
                row += (f"  regs<={by_reg} (waves/SIMD={waves} x{simds} / nw={nw})"
                        f"  -> {min(by_lds, by_reg)} WGs/CU, {binder} bind")
            else:
                row += "  regs<=? -- no num_warps in metadata, register side not computable"
            print(row)
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
    print(f"\n  (LDS is a SECOND, independent limiter -- check it too: "
          f"{lds_per_cu(a.arch)} B/CU on {a.arch}.)")
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
    got = _from_asm(p)
    os.unlink(p)
    assert got["vgpr_combined"] == 512 and got["waves_per_simd"] == 1, got
    assert got["spill_bytes"] == 456, got
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
    p = sub.add_parser("plan", help="tile plan -> resident VGPR -> reachable occupancy")
    p.add_argument("--warps", type=int, required=True)
    p.add_argument("--acc", nargs="*", help="accumulator tiles, name=MxN (fp32 by default)")
    p.add_argument("--operand", nargs="*", help="resident operand tiles, name=MxN")
    p.add_argument("--acc-bits", type=int, default=32)
    p.add_argument("--dtype-bits", type=int, default=16)
    p.add_argument("--target-waves", type=int, default=2,
                   help="occupancy you are trying to reach (default 2)")
    p.add_argument("--arch", default=DEFAULT_ARCH,
                   help=f"gfx target for the wave width + register file (default {DEFAULT_ARCH}; "
                        f"`measure` reads it from the artifact instead)")
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
