#!/usr/bin/env python3
"""Export a full rocprofv3 ATT wave into a Perfetto/Chrome-trace JSON.

Every decoded instruction becomes a complete slice (ph:"X") on its op-class lane,
positioned at its real GPU cycle (ts) with dur = cycles until the next instruction
(so stalls show as wide bars / gaps). Load the output at https://ui.perfetto.dev
(or chrome://tracing). ts/dur are GPU CYCLES (displayTimeUnit is cosmetic).

Requires the ATT decoder (rocprof-trace-decoder) output. Reuses asm_loop_audit.py's classifier +
tile_trace.py (same kernel_tools/ dir).
GEAK shared kernel tool (kernel_workflow/scripts/kernel_tools/); the Gluon pack's scripts/ copy is a shim.
The ATT decoder (rocprof-trace-decoder) is NOT vendored: point ROCPROF_ATT_LIBRARY_PATH at it when
collecting; without it there is no code.json and this tool has nothing to read (the layer is DEGRADED).

Usage:
  python3 att_to_perfetto.py <wave.json> <code.json> --out trace.json --label NAME
  python3 att_to_perfetto.py --selftest
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import asm_loop_audit as _A  # gluon / tilelang
except ModuleNotFoundError:
    import isa_loop_audit as _A  # flydsl
import tile_trace


def export(wave_json, code_json, out, label):
    w = json.load(open(wave_json))["wave"]
    code = json.load(open(code_json))["code"]
    begin = w["begin"]
    ins = w["instructions"]

    t = tile_trace.TileTrace(display_time_unit="ns", meta={
        "source": f"rocprofv3 ATT wave -> {label}",
        "units": "ts/dur are GPU cycles (not ns); relative to wave begin",
        "wave": {"cu": w.get("cu"), "simd": w.get("simd"),
                 "begin": begin, "end": w.get("end"), "num_insts": len(ins)},
    })
    pid = t.process(label)
    for lane in ["mfma", "valu", "exp", "lds_read", "lds_write", "gld", "gst",
                 "waitcnt", "barrier", "nop", "scalar", "other"]:
        t.thread(pid, lane)

    n = 0
    for i, e in enumerate(ins):
        idx = e[4]
        if idx >= len(code):
            continue
        row = code[idx]
        asm = (row[0] or "").strip()
        if not asm or asm.startswith(";"):
            continue
        cls, _sym = _A.classify(asm.split()[0])
        ts = e[0] - begin
        nxt = ins[i + 1][0] - begin if i + 1 < len(ins) else ts + max(e[2] or 1, 1)
        dur = max(nxt - ts, 1)
        src = row[3] if len(row) > 3 else ""
        t.slice(pid, cls, asm[:40], ts=ts, dur=dur, cat=cls,
                cost_cyc=e[2] or 0, src=src, full=asm)
        n += 1
    t.save(out)
    print(f"wrote {out}  ({n} instruction slices, {os.path.getsize(out)//1024} KB)")


def _selftest():
    import tempfile
    import _att_fixture
    with tempfile.TemporaryDirectory() as td:
        wave, code = _att_fixture.write(td)
        out = os.path.join(td, "t.json")
        export(wave, code, out, "fixture")
        ev = json.load(open(out))["traceEvents"]
        xs = [e for e in ev if e.get("ph") == "X"]
        ok = (len(xs) == len(_att_fixture.WAVE_INSTR) and xs[0]["ts"] == 0.0
              and {e["cat"] for e in xs} >= {"gld", "waitcnt", "mfma", "valu", "barrier"}
              and next(e for e in xs if e["cat"] == "waitcnt")["dur"] == 380.0)
    print("[att_to_perfetto] SELFTEST " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    if "--selftest" in sys.argv[1:]:
        raise SystemExit(_selftest())
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wave")
    ap.add_argument("code")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="wave")
    a = ap.parse_args()
    export(a.wave, a.code, a.out, a.label)


if __name__ == "__main__":
    main()
