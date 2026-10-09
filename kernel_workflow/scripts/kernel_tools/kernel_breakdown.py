#!/usr/bin/env python3
"""Unified kernel breakdown + pipeline analyzer (the one-command entry).

Merges the STATIC ISA audit (asm_loop_audit.py: op-class % + waitcnt quality + s_nop +
MFMA<->VALU interleave) with the DYNAMIC PMC (parse_pmc.py: per-unit busy + bubble =
100-MfmaUtil) into ONE table, and emits a RANKED bubble-ownership stack (the 优先级 input
for the ranked census, gluon_authoring references/method/profile.md ## Bound classification -> primary metric).

No single vendor tool gives all of this: rocprofv3 has only the dynamic half and is blind
on JIT kernels (e.g. TileLang); rocprof-compute is the richest dynamic tool but same
blindness + no static schedule signals; nsight is NVIDIA-only. So this tool merges the two
halves and DEGRADES gracefully: when PMC is missing/blind it emits static-only columns and
an INFERRED bubble (from s_nop cycles + full-drain% + longest VALU-only run).

Covers MFMA (CDNA) AND WMMA (RDNA) -- asm_loop_audit classifies v_wmma as the matrix class,
so util/bubble apply on gfx1201 too.

Usage:
  python3 kernel_breakdown.py <kernel.s> [--pmc <rocprof_csv_dir>] [--kernel <substr>] [--loop-label L]
  python3 kernel_breakdown.py --compare a.s b.s c.s [--pmc-dirs da db dc] [--kernel <substr>]
  python3 kernel_breakdown.py --selftest

Stdlib only, no GPU. GEAK shared kernel tool (kernel_workflow/scripts/kernel_tools/; the Gluon pack's
scripts/ copy is a shim). Reuses asm_loop_audit.py + parse_pmc.py from this kernel_tools/ dir; the
--pmc dir is what `profile_kernel.sh <gpu> <cmd> <out> --pmc` writes under <out>/pmc/.
PMC reads follow GEAK's profiling_guide.md: busy counters (MfmaUtil / VALUBusy) are throughput;
VALUUtilization is a lane duty-cycle and is shown, never ranked as a bound.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import asm_loop_audit as ala  # noqa: E402
import parse_pmc as ppmc      # noqa: E402


def static_analyze(path: str, loop_label: str | None = None) -> dict:
    """Static half: op-class histogram %, waitcnt full-drain %, s_nop cycles, interleave."""
    lines = open(path).read().splitlines()
    loops = ala.find_loops(lines)
    picked = ala.pick_loop(lines, loops, loop_label)
    if picked is None:
        raise SystemExit(f"{path}: no back-edge loop found (pass --loop-label).")
    lab, start, end = picked
    hist: Counter = Counter()
    stream: list[str] = []
    relaxed = full_drain = barriers = nops = nop_cycles = 0
    for ln in lines[start:end]:
        mn = ala._mnemonic(ln)
        if mn is None:
            continue
        name, sym = ala.classify(mn)
        hist[name] += 1
        stream.append(sym)
        if name == "barrier":
            barriers += 1
        if name == "nop":
            nops += 1
            m = re.search(r"^s_nop\s+(?:0x([0-9a-fA-F]+)|(\d+))", ln.strip())
            if m:
                nop_cycles += int(m.group(1), 16) if m.group(1) else int(m.group(2))
        if name == "waitcnt":
            # both ISA spellings + ALU-dep waits excluded -- same classifier the standalone
            # audit uses, so the merged view cannot disagree with it
            kind, _cnts = ala.classify_wait(ln)
            if kind == "relaxed":
                relaxed += 1
            else:
                full_drain += 1
    total = sum(hist.values()) or 1
    mv = ["M" if s == "M" else "V" for s in stream if s in ("M", "v", "e")]
    transitions = sum(1 for i in range(1, len(mv)) if mv[i] != mv[i - 1])
    longest_v = cur = 0
    for c in mv:
        cur = cur + 1 if c == "V" else 0
        longest_v = max(longest_v, cur)
    wc = relaxed + full_drain
    pct = {name: 100.0 * n / total for name, n in hist.items()}
    return {
        "loop": lab, "instrs": total, "pct": pct,
        "full_drain_pct": (100.0 * full_drain / wc) if wc else 0.0,
        "nop_cycles": nop_cycles, "barriers": barriers,
        "mv_transitions": transitions, "longest_v_run": longest_v,
    }


def ranked_stack(stat: dict, dyn: dict | None) -> list[tuple[str, float, str]]:
    """Ranked bubble-ownership: (label, magnitude, source). Approximate, for prioritization.

    Dynamic magnitudes when PMC present; else static proxies. This feeds -- does not
    replace -- the ranked census; timing at the boundary remains the keep/revert arbiter.
    """
    p = stat["pct"]
    rows: list[tuple[str, float, str]] = []
    if dyn and dyn.get("MfmaUtil") is not None:
        rows.append(("compute-idle (bubble=100-MfmaUtil)", dyn["bubble_mfma_idle"], "PMC"))
        if dyn.get("LDSBankConflict") is not None:
            rows.append(("LDS-bank-conflict", dyn["LDSBankConflict"], "PMC"))
        if dyn.get("MemUnitStalled") is not None:
            rows.append(("mem-stall", dyn["MemUnitStalled"], "PMC"))
        # (No "VALUUtilization - VALUBusy" row: VALUUtilization is lane occupancy DURING VALU
        # cycles -- a duty-cycle, not throughput -- so the difference is not a stall measure.
        # Dependency stalls are read from the warp-state buckets / ATT, per profiling_guide.md.)
        src = "PMC+static"
    else:
        # degraded: bubble INFERRED from static schedule + overhead classes
        inferred = stat["full_drain_pct"] * 0.5 + min(100.0, stat["nop_cycles"] / 5.0)
        rows.append(("compute-idle (INFERRED: full-drain%+s_nop+long-VALU-run)", inferred, "static"))
        src = "static"
    # static overhead classes (always shown -- these own real cycles regardless of PMC)
    rows.append((f"address/scalar-arith (scalar% {p.get('scalar', 0):.0f})", p.get("scalar", 0.0), src))
    rows.append((f"LDS-read traffic (R% {p.get('lds_read', 0):.0f})", p.get("lds_read", 0.0), src))
    rows.append((f"s_nop hazard ({stat['nop_cycles']} cyc)", min(100.0, stat["nop_cycles"] / 5.0), src))
    rows.sort(key=lambda r: (r[1] if r[1] is not None else -1), reverse=True)
    return rows


def one(path: str, pmc_dir: str | None, kfilter: str, loop_label: str | None) -> dict:
    stat = static_analyze(path, loop_label)
    dyn = None
    if pmc_dir:
        parsed = ppmc.parse_pmc(pmc_dir, kfilter)
        if parsed["vals"]:
            dyn = ppmc.derived(parsed)
    return {"path": path, "static": stat, "dynamic": dyn}


def _fmt(v):
    return "n/a" if v is None else f"{v:.1f}"


def print_one(res: dict) -> None:
    s, d = res["static"], res["dynamic"]
    p = s["pct"]
    print(f"=== {res['path']}  (loop @ {s['loop']}, {s['instrs']} instr) ===")
    print("op-class %:  " + "  ".join(
        f"{k}={p.get(k, 0):.1f}" for k in ("mfma", "valu", "exp", "lds_read", "scalar",
                                           "waitcnt", "nop") if p.get(k)))
    print(f"pipeline:    full-drain={s['full_drain_pct']:.0f}%  s_nop_cyc={s['nop_cycles']}"
          f"  MFMA<->VALU transitions={s['mv_transitions']}  longest-VALU-run={s['longest_v_run']}")
    if d:
        print(f"PMC busy:    MfmaUtil={_fmt(d['MfmaUtil'])}  VALUBusy={_fmt(d['VALUBusy'])}"
              f"  VALUUtil={_fmt(d['VALUUtilization'])}  LDSbankconf={_fmt(d['LDSBankConflict'])}"
              f"  MemStall={_fmt(d['MemUnitStalled'])}  occ={_fmt(d['occupancy'])}")
        print(f"bubble:      {_fmt(d['bubble_mfma_idle'])}%  (MFMA idle)")
    else:
        print("PMC busy:    (blind / not supplied) -> degraded mode, bubble INFERRED")
    print("ranked bubble-ownership (attack #1 first, descend to #2/#3):")
    for i, (label, mag, src) in enumerate(ranked_stack(s, d), 1):
        print(f"  #{i} {label:52s} {_fmt(mag):>6}  [{src}]")
    print()


def print_compare(results: list[dict]) -> None:
    names = [os.path.basename(os.path.dirname(r["path"])) or r["path"] for r in results]
    def row(label, fn):
        print(f"  {label:28s} " + "  ".join(f"{fn(r):>10}" for r in results))
    print("=== compare (columns = kernels) ===")
    print("  " + " " * 28 + "  ".join(f"{n[:10]:>10}" for n in names))
    for cls in ("mfma", "valu", "lds_read", "scalar", "nop"):
        row(f"op {cls} %", lambda r, c=cls: f"{r['static']['pct'].get(c, 0):.1f}")
    row("full-drain %", lambda r: f"{r['static']['full_drain_pct']:.0f}")
    row("s_nop cyc", lambda r: str(r['static']['nop_cycles']))
    row("MFMA<->VALU trans", lambda r: str(r['static']['mv_transitions']))
    row("MfmaUtil (PMC)", lambda r: _fmt(r['dynamic']['MfmaUtil']) if r['dynamic'] else "n/a")
    row("VALUBusy (PMC)", lambda r: _fmt(r['dynamic']['VALUBusy']) if r['dynamic'] else "n/a")
    row("bubble MFMA-idle", lambda r: _fmt(r['dynamic']['bubble_mfma_idle']) if r['dynamic'] else "INFER")
    print()


def att_section(att_dir: str, ideal_cadence: int = 16) -> None:
    """ATT ground-truth fold-in (per-instruction): per-op active/stall + inter-MFMA
    cycle rollup = the ground-truth ranked bubble-ownership (supersedes the inferred
    bubble above). Requires a decoded ATT ui_output dir (rocprof-trace-decoder, not vendored:
    ROCPROF_ATT_LIBRARY_PATH; collected by `profile_kernel.sh ... --att` or capture.sh)."""
    import glob
    code = os.path.join(att_dir, "code.json")
    if not os.path.exists(code):
        print(f"[--att] no code.json in {att_dir} -- ATT layer DEGRADED (not decoded? the "
              f"rocprof-trace-decoder must be on ROCPROF_ATT_LIBRARY_PATH when collecting)")
        return
    import att_opclass
    import mfma_efficiency
    print("=== ATT ground-truth (per-instruction; supersedes the inferred bubble above) ===")
    att_opclass.report(att_opclass.analyze(code), "ATT per-op-class active/stall")
    waves = sorted(glob.glob(os.path.join(att_dir, "se*_wv0.json")))
    if waves:
        ev = mfma_efficiency.load(waves[0], code)
        mfma_efficiency.report(mfma_efficiency.analyze(ev),
                               f"ATT MFMA-efficiency [{os.path.basename(waves[0])}]", ideal_cadence)
    else:
        print("  (no se*_wv0.json wave file -> per-op table only)")
    print()


_SELFTEST_ASM = """\
_kern:
.LBB0_1:
\tbuffer_load_dwordx4 v[0:3], v4, s[0:3], 0 offen
\ts_waitcnt vmcnt(0)
\tds_read_b128 v[8:11], v5
\tv_mfma_f32_16x16x32_bf16 a[0:3], v[8:11], v[12:15], a[0:3]
\tv_add_f32_e32 v20, v21, v22
\ts_nop 3
\ts_add_u32 s4, s4, 1
\ts_cbranch_scc1 .LBB0_1
\ts_endpgm
"""


def _selftest() -> int:
    import contextlib
    import io
    import tempfile
    import _att_fixture
    fails = []
    with tempfile.TemporaryDirectory() as td:
        asm = os.path.join(td, "k.s")
        with open(asm, "w") as fh:
            fh.write(_SELFTEST_ASM)
        pmc = os.path.join(td, "pmc", "pmc_sol")
        os.makedirs(pmc)
        with open(os.path.join(pmc, "x_counter_collection.csv"), "w") as fh:
            fh.write("Dispatch_ID,Kernel_Name,Counter_Name,Counter_Value,Start_Timestamp,End_Timestamp\n")
            for cn, cv in (("MfmaUtil", 62.0), ("VALUBusy", 20.0), ("VALUUtilization", 95.0)):
                fh.write(f"1,my_kern,{cn},{cv},100,200\n")
        r = one(asm, os.path.join(td, "pmc"), "my_kern", None)
        st, dy = r["static"], r["dynamic"]
        if st["loop"] != ".LBB0_1" or not st["pct"].get("mfma"):
            fails.append(f"static half did not find the loop / mfma: {st}")
        if st["nop_cycles"] != 3:
            fails.append(f"s_nop 3 -> 3 cycles, got {st['nop_cycles']}")
        if not dy or dy["MfmaUtil"] != 62.0 or dy["bubble_mfma_idle"] != 38.0:
            fails.append(f"dynamic half: MfmaUtil 62 -> bubble 38, got {dy}")
        labels = [lab for lab, _m, _s in ranked_stack(st, dy)]
        if labels[0] != "compute-idle (bubble=100-MfmaUtil)":
            fails.append(f"bubble must rank first here: {labels}")
        if any("VALUUtil" in lab for lab in labels):
            fails.append("VALUUtilization (a duty-cycle) must not be ranked as a stall")
        if one(asm, None, "my_kern", None)["dynamic"] is not None:
            fails.append("no --pmc -> dynamic half must be None (degraded, inferred bubble)")
        wave, code = _att_fixture.write(td)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_one(r)
            att_section(td, 16)
        if "ATT ground-truth" not in buf.getvalue() or "waitcnt" not in buf.getvalue():
            fails.append("ATT fold-in did not render from a decoded fixture")
    for f in fails:
        print("FAIL", f)
    print("[kernel_breakdown] SELFTEST " + ("FAIL" if fails else "PASS"))
    return 1 if fails else 0


def main() -> None:
    if "--selftest" in sys.argv[1:]:
        sys.exit(_selftest())
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("asm", nargs="?", help="kernel .s / .amdgcn (single-kernel mode)")
    ap.add_argument("--pmc", help="rocprofv3/rocprof-compute CSV output dir (dynamic half)")
    ap.add_argument("--kernel", default="_attn_fwd", help="kernel-name substring filter for PMC")
    ap.add_argument("--loop-label", help="force a loop back-edge label")
    ap.add_argument("--compare", nargs="+", help="multiple .s files for a column comparison")
    ap.add_argument("--pmc-dirs", nargs="+", help="PMC dirs aligned with --compare files")
    ap.add_argument("--att", help="decoded ATT ui_output dir -> fold in per-instruction ground-truth")
    ap.add_argument("--ideal-cadence", type=int, default=16,
                    help="theoretical MFMA cadence for the ATT fold-in (fp16 ~16, fp8 ~32)")
    a = ap.parse_args()

    if a.compare:
        pmc_dirs = a.pmc_dirs or [None] * len(a.compare)
        if len(pmc_dirs) != len(a.compare):
            sys.exit("--pmc-dirs must align 1:1 with --compare files")
        results = [one(f, pmc_dirs[i], a.kernel, a.loop_label) for i, f in enumerate(a.compare)]
        print_compare(results)
        for r in results:
            print_one(r)
        return
    if not a.asm:
        sys.exit("provide <kernel.s> or --compare a.s b.s ...")
    print_one(one(a.asm, a.pmc, a.kernel, a.loop_label))
    if a.att:
        att_section(a.att, a.ideal_cadence)


if __name__ == "__main__":
    main()
