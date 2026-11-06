#!/usr/bin/env python3
"""MFMA-efficiency analysis from rocprofv3 ATT (measured cycles).

Solidifies the "what dilutes the MFMA stream" question by attributing REAL CYCLES
(not instruction counts) to what happens between consecutive MFMA instructions --
this IS the ground-truth ranked bubble-ownership that feeds the classify -> prioritize
-> verify spine (re-run each round; confirm the #1 bucket shrank).

For a decoded ATT wave (se*_wv0.json) + code.json it reports:
  * MFMA cadence: cycles between consecutive MFMA. Judged REFERENCE-FREE against
    (a) a p10-cadence self-reference floor (robust "achievable back-to-back" for THIS
    kernel's MFMA shape) and (b) the theoretical cadence (--ideal-cadence; fp16 16 cyc /
    fp8 32 cyc, planning-constants.md). median ~ ideal => genuine MFMA-issue-bound
    (attack tile/ILP); median >> ideal => bubble (attack the dominant inter-MFMA bucket).
  * inter-MFMA CYCLE attribution + rollup (softmax / lds / addressing / WAIT / SYNC):
    which class actually steals MFMA cycles. Each op is charged the gap to the NEXT
    issued instruction (real elapsed, includes its stall), so a cheap-but-stalling op
    shows its true cost.

A peer/vendor kernel is NOT required: --compare is an OPTIONAL diagnostic upper-bound.

Usage:
  python3 mfma_efficiency.py <wave.json> <code.json> [--ideal-cadence 16] [--label NAME]
  python3 mfma_efficiency.py --compare wave,code,LABEL wave,code,LABEL ...   # optional peer diag
"""
import argparse
import json
import os
import statistics
import sys
import collections
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import asm_loop_audit as _A  # noqa: F401  (kept for API parity / future use)
except ModuleNotFoundError:
    import isa_loop_audit as _A  # noqa: F401
import deep_mfma_analysis as _DM


# The theoretical MFMA cadence is a per-arch, per-dtype hardware fact, and it lives in
# perf_knowledge/hardware/data/hw_constants.json (`arch.<gfx>.mfma_cadence_cyc`). It used to arrive as
# `--ideal-cadence`, defaulting to 16 -- the fp16/bf16 figure -- for every kernel on every arch.
# That default is not merely imprecise on an fp8 kernel, it INVERTS the verdict below: fp8 issues
# one MFMA every 32 cyc, so a kernel sitting exactly at its hardware cadence reads as
# `median 32 >> ideal 16` and is reported "bubble-diluted -> attack the #1 inter-MFMA bucket",
# when the truth is "issue-bound -> attack tile/ILP". Opposite directions, from a default.
#
# The dtype does not need to be supplied either: the MFMA mnemonic carries it, and this tool has
# already read the mnemonic in order to classify the instruction.
_DTYPE_BY_MNEMONIC = (
    # ordered: the fp8/bf8 families must be tested before f16/bf16 so `..._bf8_bf8` is not read
    # as bf16 by a loose substring match.
    ("fp8", ("fp8",)), ("bf8", ("bf8",)),
    ("bf16", ("bf16",)), ("fp16", ("f16",)),
)


def mfma_dtype(mnemonics):
    """The MFMA input dtype, from the mnemonics. None when they disagree or are unrecognised --
    a mixed or unknown stream must not silently pick one arm's cadence."""
    seen = set()
    for m in mnemonics:
        low = m.lower()
        for dt, needles in _DTYPE_BY_MNEMONIC:
            if any(n in low for n in needles):
                seen.add(dt)
                break
    return seen.pop() if len(seen) == 1 else None


def resolve_ideal_cadence(override, arch, dtype):
    """-> (cyc, provenance). `override` wins, then the hw table, then None.

    Returning None rather than 16 is the point: an unknown cadence is reported as unknown, because
    the efficiency and the verdict computed from a guessed one are confident and wrong."""
    if override is not None:
        return override, "--ideal-cadence (explicit)"
    if not (arch and dtype):
        return None, f"UNRESOLVED (arch={arch or '?'}, dtype={dtype or '?'})"
    try:
        import amd_occupancy as _O
        path = _O._find_hw_constants()
        if not path:
            return None, "UNRESOLVED (hw_constants.json not reachable)"
        with open(path) as f:
            tbl = json.load(f)["arch"].get(arch, {}).get("mfma_cadence_cyc", {})
    except Exception as e:                                    # noqa: BLE001
        return None, f"UNRESOLVED ({type(e).__name__})"
    cyc = tbl.get(dtype)
    if cyc is None:
        return None, f"UNRESOLVED (no mfma_cadence_cyc.{dtype} for {arch})"
    return int(cyc), f"hw_constants.json {arch}.mfma_cadence_cyc.{dtype}"


def _cards_by_bucket():
    """bubble slug -> [card ids], read from the Gluon pack's lever-cards.json if present.

    The cards stay in the pack (they point at the pack's references), so they resolve through
    `_hwdata.pack_file('references/hardware/lever-cards.json')` -- `$GEAK_GLUON_PACK_DIR`, else
    this checkout's gluon_authoring. No glob: a miss degrades to no pointers rather than failing
    or walking the filesystem. Order preserved from each card's bubble_bucket list."""
    try:
        import _hwdata
        c = _hwdata.pack_file("references/hardware/lever-cards.json")
    except ImportError:
        c = None
    if c is None:
        return {}
    try:
        levers = json.load(open(c)).get("levers", {})
    except Exception:  # noqa: BLE001
        return {}
    out = collections.defaultdict(list)
    for cid, card in levers.items():
        for b in (card.get("bubble_bucket") or []):
            out[b].append(cid)
    return out


def _pct(xs, p):
    if not xs:
        return 0
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[k]


def load(wave_json, code_json):
    """Return issue-ordered [(cycle, subclass, mnemonic)] for the wave."""
    w = json.load(open(wave_json))["wave"]
    code = json.load(open(code_json))["code"]
    begin = w["begin"]
    ev = []
    for e in w["instructions"]:
        idx = e[4]
        if idx >= len(code):
            continue
        asm = (code[idx][0] or "").strip()
        if not asm or asm.startswith(";"):
            continue
        mnem = asm.split()[0]
        ev.append((e[0] - begin, _DM.sub_class(mnem), mnem))
    ev.sort(key=lambda x: x[0])
    return ev


def analyze(ev):
    gaps = [ev[i + 1][0] - ev[i][0] for i in range(len(ev) - 1)] + [1]
    mfma_cy = [c for c, sc, _m in ev if sc == "mfma"]
    cadence = [mfma_cy[i + 1] - mfma_cy[i] for i in range(len(mfma_cy) - 1)]
    span = (mfma_cy[-1] - mfma_cy[0]) if len(mfma_cy) > 1 else 1
    cyc_by_cls = Counter()
    cnt_by_cls = Counter()
    seen_mfma = False
    for i, (c, sc, _m) in enumerate(ev):
        if sc == "mfma":
            seen_mfma = True
            continue
        if not seen_mfma:
            continue
        if len(mfma_cy) > 1 and not (mfma_cy[0] <= c <= mfma_cy[-1]):
            continue
        cyc_by_cls[sc] += gaps[i]
        cnt_by_cls[sc] += 1
    return dict(
        cadence_mean=statistics.mean(cadence) if cadence else 0,
        cadence_median=statistics.median(cadence) if cadence else 0,
        cadence_p10=_pct(cadence, 10), n_mfma=len(mfma_cy), span=span,
        cyc_by_cls=cyc_by_cls, cnt_by_cls=cnt_by_cls,
        mfma_mnemonics=sorted({m for _c, sc, m in ev if sc == "mfma"}))


def report(r, label, ideal, ideal_src="", dtype=None):
    print(f"\n=== {label} ===")
    n, span = r["n_mfma"], r["span"]
    med, mean, p10 = r["cadence_median"], r["cadence_mean"], r["cadence_p10"]
    print(f"  MFMA: {n} in span {span} cyc"
          + (f"  dtype={dtype}" if dtype else "  dtype=? (mnemonics: "
             + ",".join(r.get("mfma_mnemonics") or ["none"])[:60] + ")"))
    print(f"  MFMA cadence (cyc between consecutive MFMA): "
          f"p10={p10} (self-ref floor)  median={med:.0f}  mean={mean:.1f}  "
          f"ideal={ideal if ideal is not None else 'UNKNOWN'}"
          + (f" [{ideal_src}]" if ideal_src else ""))
    # reference-free efficiency: median vs the two self/hardware references.
    if med and ideal:
        eff_self = 100.0 * p10 / med if med else 0        # how close to THIS kernel's own best
        eff_ideal = 100.0 * ideal / med if med else 0     # how close to the hardware cadence
        verdict = ("issue-bound (cadence ~ ideal -> attack tile/ILP)" if med <= 1.3 * ideal
                   else "bubble-diluted (cadence >> ideal -> attack the #1 inter-MFMA bucket)")
        print(f"  MFMA efficiency: vs-p10-self={eff_self:.0f}%  vs-ideal={eff_ideal:.0f}%  -> {verdict}")
    elif med:
        # The self-reference still stands without a hardware cadence; the verdict does not, and
        # printing one anyway is how a guessed constant becomes a direction.
        print(f"  MFMA efficiency: vs-p10-self={100.0 * p10 / med:.0f}%  vs-ideal=UNKNOWN "
              f"-> NO VERDICT ({ideal_src}). Pass --arch, or --ideal-cadence to assert one.")
    tot = sum(r["cyc_by_cls"].values()) or 1
    print(f"  inter-MFMA CYCLES stolen by ({tot} cyc total between MFMA):")
    for sc, cyc in r["cyc_by_cls"].most_common():
        cnt = r["cnt_by_cls"][sc]
        print(f"      {sc:16s} {cyc:8d} cyc ({100*cyc/tot:4.1f}%)   {cnt:5d} instr  "
              f"{cyc/cnt if cnt else 0:5.1f} cyc/instr")
    cb = r["cyc_by_cls"]
    # (human label, canonical bubble slug) -- the slug is the join key into lever-cards.json
    # `bubble_bucket`. cadence-diluted MFMA is reported separately (see cadence verdict above),
    # so the 'mfma-cadence' cards are surfaced there, not in this inter-MFMA rollup.
    roll = [
        ("softmax/fp-compute (fp_math+exp+pack)", "fp-compute", cb["valu_fp_math"] + cb["exp"] + cb["valu_fp_pack"]),
        ("lds feed (lgkmcnt-waited)",             "lds-feed",   cb["lds"]),
        ("gmem feed (vmcnt-waited)",              "gmem-feed",  cb["gmem"]),
        ("addressing (int)",                      "addressing", cb["valu_addr_int"] + cb["scalar"]),
        ("cvt/mov/other-valu",                    "cvt-mov",    cb["valu_cvt_mov"] + cb["valu_other"]),
        ("WAIT (waitcnt=unhidden latency)",       "wait",       cb["waitcnt"]),
        ("SYNC (barrier+nop)",                    "sync",       cb["barrier"] + cb["nop"]),
    ]
    cards = _cards_by_bucket()
    print("  rollup of inter-MFMA cycles (ranked bubble-ownership -> AMD lever cards for each bucket):")
    for label, slug, v in sorted(roll, key=lambda x: -x[2]):
        tac = cards.get(slug)
        ptr = f"   -> cards: {', '.join(tac)}" if tac else ""
        print(f"      {label:40s} {100*v/tot:5.1f}%{ptr}")
    if cards:
        print("  (a card is a candidate, not a verdict; one that does not fit is a named N/A. "
              "A ranked bucket with NO card is a catalogue gap -- report it.)")
    # 'lds/gmem feed' merges two DIFFERENT resources whose fixes are OPPOSITE: LDS traffic
    # (lgkmcnt-waited -> layout/swizzle/fewer-wider ds_read) and global traffic (vmcnt-waited
    # -> coalescing/async copy/prefetch). Published split, same denominator as the rollup, so a
    # large feed bucket is actionable instead of merely large.
    print("  feed-bucket split (resolves 'lds/gmem feed'; the two halves have opposite fixes):")
    print(f"      {'lds feed (lgkmcnt-waited)':40s} {100*cb['lds']/tot:5.1f}%")
    print(f"      {'gmem feed (vmcnt-waited)':40s} {100*cb['gmem']/tot:5.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wave", nargs="?")
    ap.add_argument("code", nargs="?")
    ap.add_argument("--label", default="wave")
    ap.add_argument("--arch", default=None,
                    help="gfx target (e.g. gfx950). Resolves the theoretical MFMA cadence from "
                         "perf_knowledge/hardware/data/hw_constants.json together with the dtype read off "
                         "the MFMA mnemonic -- pass this instead of hand-supplying a cadence.")
    ap.add_argument("--ideal-cadence", type=int, default=None,
                    help="OVERRIDE the resolved theoretical cadence (cyc). Only needed when --arch "
                         "cannot be given or the dtype is not inferable; a wrong value INVERTS the "
                         "issue-bound vs bubble-diluted verdict, so state why you set it.")
    ap.add_argument("--compare", nargs="+",
                    help="wave,code,LABEL entries (OPTIONAL peer/vendor diagnostic upper-bound)")
    a = ap.parse_args()
    specs = ([tuple(e.split(",")) for e in a.compare] if a.compare
             else [(a.wave, a.code, a.label)])
    for w, c, l in specs:
        r = analyze(load(w, c))
        # Resolve per stream: --compare can hold arms with different MFMA shapes, and reusing one
        # arm's cadence across them is the same inversion in a different disguise.
        dtype = mfma_dtype(r.get("mfma_mnemonics") or [])
        ideal, src = resolve_ideal_cadence(a.ideal_cadence, a.arch, dtype)
        report(r, l, ideal, src, dtype)


def _selftest():
    """No selftest shipped with this tool before, which is how a hardcoded fp16 cadence survived in
    a file whose verdict depends on it. These assert the resolution, not the formatting."""
    fail = 0

    def chk(cond, msg):
        nonlocal fail
        if not cond:
            print(f"FAIL: {msg}")
            fail = 1

    # dtype off the mnemonic, which is what removes the need to hand-supply a cadence at all.
    chk(mfma_dtype(["v_mfma_f32_32x32x16_fp8_fp8"]) == "fp8", "fp8 mnemonic -> fp8")
    chk(mfma_dtype(["v_mfma_f32_32x32x16_bf8_bf8"]) == "bf8", "bf8 must not read as bf16")
    chk(mfma_dtype(["v_mfma_f32_32x32x8_bf16_1k"]) == "bf16", "bf16_1k -> bf16")
    chk(mfma_dtype(["v_mfma_f32_16x16x16_f16"]) == "fp16", "f16 -> fp16")
    # A mixed stream has no single cadence, so it must refuse rather than pick one arm's.
    chk(mfma_dtype(["v_mfma_f32_16x16x16_f16", "v_mfma_f32_32x32x16_fp8_fp8"]) is None,
        "a mixed dtype stream must resolve to None")
    chk(mfma_dtype(["v_mfma_f32_16x16x32_f8f6f4"]) is None, "an unrecognised family -> None")

    # An explicit override always wins, and says so.
    cyc, src = resolve_ideal_cadence(24, "gfx942", "fp16")
    chk(cyc == 24 and "explicit" in src, f"override should win, got {cyc} {src}")
    # Unknown arch or dtype must yield None -- NOT a fallback constant.
    chk(resolve_ideal_cadence(None, None, "fp8")[0] is None, "no arch -> unresolved, not a default")
    chk(resolve_ideal_cadence(None, "gfx942", None)[0] is None, "no dtype -> unresolved")

    # The table lookup, when the composed reference tree is reachable (it is not in the upstream
    # source layout, so this half is conditional rather than skipped silently).
    import amd_occupancy as _O
    if _O._find_hw_constants():
        # gfx950 is the main line; gfx942 is the downgrade comparison (same fp16/fp8 cadence).
        g16, gs16 = resolve_ideal_cadence(None, "gfx950", "fp16")
        g8, gs8 = resolve_ideal_cadence(None, "gfx950", "fp8")
        chk(g16 == 16 and "hw_constants" in gs16, f"gfx950 fp16 -> 16, got {g16}")
        chk(g8 == 32 and "hw_constants" in gs8, f"gfx950 fp8 -> 32, got {g8}")
        f16, s16 = resolve_ideal_cadence(None, "gfx942", "fp16")
        f8, s8 = resolve_ideal_cadence(None, "gfx942", "fp8")
        chk(f16 == 16 and "hw_constants" in s16, f"gfx942 fp16 -> 16, got {f16}")
        chk(f8 == 32 and "hw_constants" in s8, f"gfx942 fp8 -> 32, got {f8}")
        # THE REGRESSION THIS FILE EXISTS TO PREVENT: at fp8's real cadence of 32 a kernel sitting
        # exactly at the hardware rate must read issue-bound. Under the old default of 16 the same
        # kernel read "bubble-diluted", which points the round at the opposite lever.
        chk(32 <= 1.3 * f8, "fp8 kernel at cadence 32 must be issue-bound against its own ideal")
        chk(not (32 <= 1.3 * 16), "the old default 16 would have inverted that verdict")
    else:
        print("  -- hw_constants.json not reachable in this layout; table lookup not asserted")

    # lever-cards.json resolves from the pack, never by globbing; when the pack is reachable
    # the bubble buckets must map to at least one card.
    try:
        import _hwdata
        if _hwdata.pack_file("references/hardware/lever-cards.json"):
            chk(bool(_cards_by_bucket()), "lever-cards.json reachable but no bubble_bucket cards")
    except ImportError:
        pass

    print("MFMA_EFFICIENCY SELFTEST " + ("PASS" if not fail else "FAIL"))
    return fail


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    main()
