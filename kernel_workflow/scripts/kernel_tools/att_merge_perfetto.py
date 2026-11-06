#!/usr/bin/env python3
"""Merge several rocprofv3 ATT waves into ONE Perfetto/Chrome-trace JSON.

One process row per wave (e.g. kernel A vs kernel B vs a vendor peer), each with its
op-class lanes, so you can compare interleave / stall structure side by side in
https://ui.perfetto.dev. Requires the ATT decoder; reuses tile_trace.py + the skill's
*_loop_audit.py classifier.

GEAK shared kernel tool (kernel_tools/); the Gluon pack's scripts/ copy is a shim. The decoder is not
vendored (ROCPROF_ATT_LIBRARY_PATH); without decoded waves there is nothing to merge.

Usage:
  python3 att_merge_perfetto.py --add wave,code,LABEL [wave,code,LABEL ...] --out merged.json
  python3 att_merge_perfetto.py --selftest
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

_LANES = ["mfma", "valu", "exp", "lds_read", "lds_write", "gld", "gst",
          "waitcnt", "barrier", "nop", "scalar", "other"]


def add_wave(t, wave_json, code_json, label):
    w = json.load(open(wave_json))["wave"]
    code = json.load(open(code_json))["code"]
    begin = w["begin"]
    ins = w["instructions"]
    pid = t.process(label)
    for lane in _LANES:
        t.thread(pid, lane)
    n = 0
    for i, e in enumerate(ins):
        idx = e[4]
        if idx >= len(code):
            continue
        asm = (code[idx][0] or "").strip()
        if not asm or asm.startswith(";"):
            continue
        cls, _ = _A.classify(asm.split()[0])
        ts = e[0] - begin
        nxt = ins[i + 1][0] - begin if i + 1 < len(ins) else ts + max(e[2] or 1, 1)
        t.slice(pid, cls, asm[:40], ts=ts, dur=max(nxt - ts, 1), cat=cls,
                cost_cyc=e[2] or 0, src=code[idx][3] if len(code[idx]) > 3 else "", full=asm)
        n += 1
    return n


def _selftest():
    import tempfile
    import _att_fixture
    with tempfile.TemporaryDirectory() as td:
        wave, code = _att_fixture.write(td)
        t = tile_trace.TileTrace(display_time_unit="ns")
        n1 = add_wave(t, wave, code, "A")
        n2 = add_wave(t, wave, code, "B")
        out = os.path.join(td, "m.json")
        t.save(out)
        ev = json.load(open(out))["traceEvents"]
        procs = {e["args"]["name"] for e in ev if e.get("name") == "process_name"}
        ok = n1 == n2 == len(_att_fixture.WAVE_INSTR) and procs == {"A", "B"}
    print("[att_merge_perfetto] SELFTEST " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    if "--selftest" in sys.argv[1:]:
        raise SystemExit(_selftest())
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--add", nargs="+", required=True, help="wave.json,code.json,LABEL")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t = tile_trace.TileTrace(display_time_unit="ns", meta={
        "source": "merged rocprofv3 ATT waves",
        "units": "ts/dur are GPU cycles (each process relative to its own wave begin)"})
    for e in a.add:
        w, c, l = e.split(",")
        print(f"added {l}: {add_wave(t, w, c, l)} slices")
    t.save(a.out)
    print(f"wrote {a.out} ({os.path.getsize(a.out)//1024} KB)")


if __name__ == "__main__":
    main()
