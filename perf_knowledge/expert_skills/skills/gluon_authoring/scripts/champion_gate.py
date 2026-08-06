#!/usr/bin/env python3
"""champion_gate.py - the entry assertion for a DEEP-DIG skill (gluon / flydsl).

The deep-dig skills do not tune plain source themselves: they start from a `plain_champion` bundle
produced by the broad-search front end (tile-programming-triton) and climb from there. That handoff
is the single point where the whole run's honesty is decided, because every later speedup is quoted
against it. This gate refuses to start on a bundle that cannot support such a claim.

It generalizes the one surviving mechanical plain->escalated check ([PLAIN-UNTUNED] in gate.py), which
only fired inside the gated per-round engine and only compared 7 hardcoded knob names. Here the check
is a standalone tool, keys on whatever the config actually contains, and additionally pins the SOURCE
(a bundle whose champion source has been edited since it was measured is not the thing that was
measured) and the COMPARATOR (a champion slower than the kernel's own default is a strawman inverted).

Checks (HARD unless marked soft):
  [SCHEMA]     the bundle loads, `schema: plain_champion`, required fields present
  [SOURCE]     `source_ref` resolves and its sha256 matches `source_sha`
  [CONFIG]     the pinned `.ttgir` was dumped AT `config` -- cross-checked against the TTGIR's own
               `ttg.num-warps` and against a `<ttgir>.config.json` sidecar when either is available
               (soft when neither exists: unverifiable, not contradicted)
  [COMPARATOR] `champion_ms` <= `default_ms` and <= `sweep_winner_ms` (soft when `default_ms` is null,
               which is itself reported -- the "not a default strawman" claim is then unprovable)
  [GATED]      `trust_level` is not `ungated`, else FAIL unless --allow-ungated. An ungated sweep
               never checked its winner, so the bundle's timings may belong to a wrong kernel
               (soft `unknown` for a bundle written before trust_level existed)
  [SAMPLING]   `partially_sampled` is false, else FAIL unless --allow-provisional (then a declared
               soft degrade: the comparator is the best of a SUBSET, so downstream speedups are
               inflated by whatever the unmeasured configs would have gained)
  [RANGE]      `served_range` is non-empty (soft here; the captain's CLOSE requires it)
  [LOCUS]      the execution locus is recorded, so the deep-dig runs where the champion was measured
  [TOOLCHAIN]  the toolchain is pinned (soft) -- an unpinned bundle cannot be re-measured later

Usage:
  champion_gate.py --champion <work>/plain_champion.json [--allow-provisional] [--allow-ungated] [--json]
  champion_gate.py --selftest
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

# The keys whose mismatch means the TTGIR cannot reach the champion's performance. NOT a fixed list of
# knob names: this is only the fallback ORDER for reporting. The comparison itself runs over every key
# the two configs share, because a kernel that spells its tile `BLOCK_SIZE_M` (aiter) or `BLK_M` is
# just as config-pinned as one that spells it `BLOCK_M` -- keying on names is how the old sweep prune
# silently blocked 6 of 7 kernel families.
_REPORT_FIRST = ("BLOCK_M", "BLOCK_N", "BLOCK_K", "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K",
                 "num_warps", "num_stages", "matrix_instr_nonkdim", "GROUP_SIZE_M")

_REQUIRED = ("kernel", "source_ref", "source_sha", "config", "champion_ms", "ttgir")

_NUM_WARPS_RE = re.compile(r'ttg\.num-warps["\s]*[=:]\s*(\d+)')


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve(base: Path, ref) -> Path | None:
    """A bundle path is relative to the bundle dir (so the whole bundle is movable) or absolute."""
    if not ref:
        return None
    p = Path(str(ref))
    return p if p.is_absolute() else (base / p)


def _shared_key_mismatch(a: dict, b: dict) -> list:
    """Keys present in BOTH with different values, tile/warp knobs reported first."""
    mism = [k for k in set(a) & set(b) if a[k] != b[k]]
    return sorted(mism, key=lambda k: (_REPORT_FIRST.index(k) if k in _REPORT_FIRST else 99, k))


def gate(champ: dict, base: Path, allow_provisional: bool = False, allow_ungated: bool = False):
    """Returns (results, hard_fail). `base` is the dir the bundle's relative paths resolve against."""
    results, hard_fail = [], False

    def rec(check, ok, msg, hard=True):
        nonlocal hard_fail
        results.append({"check": check, "ok": bool(ok), "hard": bool(hard), "msg": msg})
        if not ok and hard:
            hard_fail = True

    # [SCHEMA]
    if not isinstance(champ, dict) or champ.get("schema") != "plain_champion":
        rec("SCHEMA", False, f"not a plain_champion bundle (schema={champ.get('schema') if isinstance(champ, dict) else type(champ).__name__!r}). "
                             "The broad-search front end (tile-programming-triton) writes it at close.")
        return results, hard_fail          # nothing below is meaningful without the schema
    missing = [k for k in _REQUIRED if champ.get(k) in (None, "")]
    if missing:
        rec("SCHEMA", False, f"missing required field(s): {missing}")
        return results, hard_fail
    rec("SCHEMA", True, f"plain_champion for {champ['kernel']}")

    # [SOURCE] -- the bundle must still describe the file that was measured.
    src = _resolve(base, champ.get("source_ref"))
    if src is None or not src.is_file():
        rec("SOURCE", False, f"source_ref does not resolve to a file: {champ.get('source_ref')!r} "
                             f"(resolved against {base})")
    else:
        got = _sha256(src)
        if got != champ["source_sha"]:
            rec("SOURCE", False, f"{src} has changed since it was measured "
                                 f"(sha256 {got[:12]} != recorded {str(champ['source_sha'])[:12]}). "
                                 "Re-measure the champion, or transcribe the recorded source -- a "
                                 "deep-dig anchored on an edited source is measured against nothing.")
        else:
            rec("SOURCE", True, f"{src.name} matches its recorded sha256")

    # [LIVE] -- the file the RUN loads must still be the file the gate just validated.
    #
    # [SOURCE] above hashes `source_ref`, which is by design a frozen copy -- so it
    # matches essentially forever and cannot detect the failure that actually happens.
    # Everything downstream loads `kernel`, and nothing was checking it. Found on a real
    # bundle: `kernel` -> task/kernel_jit.py had been overwritten by a later Gluon track
    # (its sha is that track's installed winner), while `source_ref` -> champion/
    # kernel_jit.py was pristine. The gate reported `[PASS] SOURCE ... matches its
    # recorded sha256` and cleared a run whose plain arm would have been GLUON. One agent
    # noticed on its own; the next one would not have, and would have reported a ratio
    # near 1.0 against the wrong denominator.
    live = _resolve(base, champ.get("kernel"))
    if live is None or not live.is_file():
        rec("LIVE", False, f"`kernel` does not resolve to a file: {champ.get('kernel')!r}. "
                           "That is the path the run loads, so nothing downstream is anchored.")
    else:
        got_live = _sha256(live)
        if got_live != champ["source_sha"]:
            same_name = (src is not None and src.is_file()
                         and _sha256(src) == champ["source_sha"])
            hint = (f"Restore it from source_ref ({champ.get('source_ref')}), which does still "
                    "match, or re-measure." if same_name else
                    "Both `kernel` and `source_ref` are off the recorded sha; re-measure.")
            rec("LIVE", False,
                f"`kernel` ({live}) is NOT the measured source: sha256 {got_live[:12]} != "
                f"recorded {str(champ['source_sha'])[:12]}. Whatever the run benchmarks as "
                f"'plain' is not what champion_ms describes. {hint}")
        else:
            rec("LIVE", True, f"`kernel` ({live.name}) is byte-identical to the measured source")

    # [CONFIG] -- the TTGIR the anchor is recovered from must be the tuned config's TTGIR.
    cfg = champ.get("config")
    ttgir = _resolve(base, champ.get("ttgir"))
    if not isinstance(cfg, dict):
        rec("CONFIG", False, f"config is not an object: {type(cfg).__name__}")
    elif ttgir is None or not ttgir.is_file():
        rec("CONFIG", False, f"ttgir does not resolve to a file: {champ.get('ttgir')!r}. The anchor is "
                             "recovered from the champion's own TTGIR; without it there is nothing to "
                             "recover from (dump_ir.sh at the pinned config).")
    else:
        text = ttgir.read_text(errors="ignore")
        evidence, contradiction = [], []
        m = _NUM_WARPS_RE.search(text)
        if m and "num_warps" in cfg:
            if int(m.group(1)) == int(cfg["num_warps"]):
                evidence.append(f"ttg.num-warps={m.group(1)}")
            else:
                contradiction.append(f"ttg.num-warps={m.group(1)} but config num_warps={cfg['num_warps']}")
        side = Path(str(ttgir) + ".config.json")
        if side.is_file():
            try:
                sc = json.loads(side.read_text())
            except (ValueError, OSError) as e:
                contradiction.append(f"{side.name} unreadable ({e})")
            else:
                mism = _shared_key_mismatch(sc if isinstance(sc, dict) else {}, cfg)
                if mism:
                    contradiction.append(f"{side.name} disagrees on {mism}")
                else:
                    evidence.append(f"{side.name} agrees on {len(set(sc) & set(cfg))} shared key(s)")
        if contradiction:
            rec("CONFIG", False, "the TTGIR was NOT dumped at the pinned config: "
                                 + "; ".join(contradiction) +
                                 ". Re-dump from the winning config, else the anchor starts below "
                                 "plain-best and every later delta is measured from the wrong floor.")
        elif evidence:
            rec("CONFIG", True, "TTGIR is consistent with the pinned config (" + ", ".join(evidence) + ")")
        else:
            rec("CONFIG", True, f"{ttgir.name} exists but carries no cross-checkable config signal "
                                f"(no ttg.num-warps, no {side.name}) -- UNVERIFIED, not contradicted. "
                                "Have dump_ir.sh write the sidecar to make this checkable.", hard=False)

    # [COMPARATOR] -- champion must beat the default and the config-only winner.
    ch = champ.get("champion_ms")
    dm, sw = champ.get("default_ms"), champ.get("sweep_winner_ms")
    if not isinstance(ch, (int, float)):
        rec("COMPARATOR", False, f"champion_ms is not a number: {ch!r}")
    elif isinstance(dm, (int, float)) and ch > dm:
        rec("COMPARATOR", False, f"champion_ms {ch:.4f} is SLOWER than the kernel's own default "
                                 f"{dm:.4f} ms. That is a strawman inverted: every downstream speedup "
                                 "quoted against it is measured from a floor the shipped kernel beats.")
    elif isinstance(sw, (int, float)) and ch > sw:
        rec("COMPARATOR", False, f"champion_ms {ch:.4f} is SLOWER than the config sweep's own winner "
                                 f"{sw:.4f} ms -- the source-level work regressed the tuned baseline.")
    elif not isinstance(dm, (int, float)):
        rec("COMPARATOR", True, f"champion_ms {ch:.4f} recorded, but default_ms is null -- the "
                                "'vs tuned plain, never the default strawman' claim is UNPROVABLE "
                                "until the front end publishes it.", hard=False)
    else:
        gain = f", {dm / ch:.3f}x vs default" if ch else ""
        rec("COMPARATOR", True, f"champion_ms {ch:.4f} <= default {dm:.4f}{gain}")

    # [GATED] -- an UNGATED sweep never checked its winner, so the bundle's timings describe a
    # program nobody proved computes the right thing. That outranks [SAMPLING]: a capped sweep is an
    # incomplete measurement, an ungated one may not be a measurement of the kernel at all.
    trust = champ.get("trust_level")
    if trust == "ungated":
        rec("GATED", bool(allow_ungated),
            "the champion's config sweep ran UNGATED (no oracle), so its winner was never checked "
            "for correctness and may be wrong-and-fast. Declare correctness.cmd and re-sweep, or "
            "pass --allow-ungated to proceed with it DECLARED in caveats[]."
            if not allow_ungated else
            "ungated sweep accepted via --allow-ungated -- record it in caveats[]: every timing in "
            "this bundle is unverified, and the deep dig inherits that.",
            hard=not allow_ungated)
    elif trust in ("pinned", "provisional"):
        rec("GATED", True, f"the config sweep was oracle-gated (trust_level={trust})")
    else:
        rec("GATED", True, "the sweep predates trust_level, so whether an oracle ran is UNKNOWN -- "
                           "re-emit the bundle from a current plain_autotune.py to make it checkable",
            hard=False)

    # [SAMPLING] -- a provisional comparator inflates everything downstream.
    if champ.get("partially_sampled"):
        rec("SAMPLING", bool(allow_provisional),
            "the champion's config sweep was PARTIALLY SAMPLED (best of a capped subset), so it is a "
            "provisional comparator, not a tuned one. Widen the grid / raise the sweep cap, or pass "
            "--allow-provisional to proceed with the inflation DECLARED in caveats[]."
            if not allow_provisional else
            "partially_sampled accepted via --allow-provisional -- record it in caveats[]: downstream "
            "speedups are inflated by whatever the unmeasured configs would have gained.",
            hard=not allow_provisional)
    else:
        # This branch can only read the bundle's OWN claim that the grid was covered, and a sweep
        # reporting that its winner survived says nothing about points it never tested. Observed: a
        # 6.1% plain win one grid step outside the swept range, on a kernel whose tier log recorded a
        # completed re-sweep -- inherited by the port as a fake escalation gain. So the PASS is
        # reported as unfalsified rather than verified, with the cheap check that would falsify it.
        rec("SAMPLING", True,
            "the bundle reports the feasible config grid as fully sampled. NOT VERIFIABLE HERE -- "
            "this check can only read the claim. Before the first round, spot-check the pinned "
            "config at +/-1 grid step on EACH swept axis: a sweep's own report that its pin survived "
            "is not evidence about points it did not test, and a port that starts one grid point "
            "short of the real champion inherits that gap as a fake escalation gain.")

    # [RANGE] -- soft here, hard at CLOSE.
    rng = champ.get("served_range")
    rec("RANGE", bool(rng) and isinstance(rng, list),
        f"served_range has {len(rng)} row(s)" if isinstance(rng, list) and rng else
        "served_range is empty -- a single-anchor win that regresses at other shapes is a bucketed "
        "dispatch, not a clean win. The captain's CLOSE requires this table; fill it before closing.",
        hard=False)

    # [LOCUS] -- run the deep dig where the champion was measured.
    loc = champ.get("locus") or {}
    if loc.get("docker"):
        rec("LOCUS", True, f"container {str(loc['docker'])[:12]}, gpu {loc.get('gpu', '?')}, "
                           f"pythonpath {'set' if loc.get('pythonpath') else 'UNSET'}",
            hard=False)
    else:
        rec("LOCUS", True, "host locus (no container recorded) -- confirm this box is the one the "
                           "champion was measured on, or the baseline is not comparable", hard=False)

    # [TOOLCHAIN]
    tc = champ.get("toolchain") or {}
    rec("TOOLCHAIN", bool(tc), f"pinned: {', '.join(f'{k}={v}' for k, v in sorted(tc.items()))}" if tc
        else "toolchain not pinned -- this bundle cannot be re-measured after a ROCm/Triton bump",
        hard=False)
    return results, hard_fail


def _selftest() -> int:
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        srcf = base / "champion" / "k.py"
        srcf.parent.mkdir(parents=True)
        srcf.write_text("# champion kernel\n")
        sha = _sha256(srcf)
        # The LIVE check needs `kernel` to be a real file. Same bytes as the frozen copy,
        # different path -- which is exactly the shape of a healthy bundle.
        livef = base / "task" / "k.py"
        livef.parent.mkdir(parents=True)
        livef.write_text("# champion kernel\n")
        irf = base / "ir" / "champion.ttgir"
        irf.parent.mkdir(parents=True)
        irf.write_text('module attributes {"ttg.num-warps" = 8 : i32} {\n}\n')
        good = {"schema": "plain_champion", "kernel": "task/k.py",
                "source_ref": "champion/k.py", "source_sha": sha,
                "config": {"num_warps": 8, "BLOCK_SIZE_M": 128},
                "default_ms": 11.16, "sweep_winner_ms": 10.0, "champion_ms": 9.31,
                "ttgir": "ir/champion.ttgir", "served_range": [{"shape": "m=1", "ms": 9.31}],
                "partially_sampled": False, "trust_level": "pinned",
                "locus": {"docker": "abc123", "gpu": "4"}, "toolchain": {"arch": "gfx942"}}

        def run(mut=None, **kw):
            c = json.loads(json.dumps(good))
            if mut:
                mut(c)
            return gate(c, base, **kw)

        res, hard = run()
        assert not hard, [r for r in res if not r["ok"]]
        by = {r["check"]: r for r in res}
        assert by["CONFIG"]["ok"] and "ttg.num-warps=8" in by["CONFIG"]["msg"]
        assert by["COMPARATOR"]["ok"] and by["SAMPLING"]["ok"] and by["SOURCE"]["ok"]
        assert by["LIVE"]["ok"], by["LIVE"]

        # [LIVE] the failure [SOURCE] structurally cannot see: the frozen copy is pristine
        # while the file the RUN loads has been overwritten. Found on a real bundle where
        # the overwrite was a later Gluon track's installed winner, i.e. the run's "plain"
        # arm would have been Gluon and the gate said PASS.
        livef.write_text("# overwritten by a later track\n")
        res_l, hard_l = run()
        by_l = {r["check"]: r for r in res_l}
        assert by_l["SOURCE"]["ok"], "the frozen copy is untouched, so SOURCE must still pass"
        assert not by_l["LIVE"]["ok"], "LIVE must catch an overwritten `kernel`"
        assert hard_l, "an unanchored run must be a hard fail"
        assert "source_ref" in by_l["LIVE"]["msg"], "say how to restore it"
        livef.write_text("# champion kernel\n")
        assert {r["check"]: r for r in run()[0]}["LIVE"]["ok"], "restoring must clear [LIVE]"

        # a `kernel` that does not resolve is also unanchored
        res_m, hard_m = run(lambda c: c.update(kernel="task/gone.py"))
        assert not {r["check"]: r for r in res_m}["LIVE"]["ok"] and hard_m

        # a wrong schema short-circuits and never claims the other checks passed
        res, hard = gate({"schema": "plain_best_config"}, base)
        assert hard and len(res) == 1 and res[0]["check"] == "SCHEMA"
        # a missing required field is a hard SCHEMA fail, not a later confusing failure
        res, hard = run(lambda c: c.pop("ttgir"))
        assert hard and res[-1]["check"] == "SCHEMA" and "ttgir" in res[-1]["msg"]

        # THE CORE PROPERTY: an edited champion source can never pass. This is the defect the whole
        # bundle exists to prevent -- transcribing something other than what was measured.
        srcf.write_text("# champion kernel, edited after measurement\n")
        res, hard = run()
        assert hard and not {r["check"]: r for r in res}["SOURCE"]["ok"]
        srcf.write_text("# champion kernel\n")
        assert not run()[1], "restoring the source must clear [SOURCE]"

        # config mismatch is caught from the TTGIR itself...
        res, hard = run(lambda c: c["config"].__setitem__("num_warps", 4))
        assert hard and "ttg.num-warps=8" in {r["check"]: r for r in res}["CONFIG"]["msg"]
        # ...and from a sidecar, keyed on whatever names the kernel actually uses (BLOCK_SIZE_M here,
        # not the GEMM-canonical BLOCK_M -- the naming assumption that blocked 6 of 7 families).
        side = Path(str(irf) + ".config.json")
        side.write_text(json.dumps({"num_warps": 8, "BLOCK_SIZE_M": 64}))
        res, hard = run()
        assert hard and "BLOCK_SIZE_M" in {r["check"]: r for r in res}["CONFIG"]["msg"]
        side.write_text(json.dumps({"num_warps": 8, "BLOCK_SIZE_M": 128}))
        assert not run()[1], "an agreeing sidecar must pass"
        side.unlink()
        # no cross-checkable signal at all -> soft UNVERIFIED, not a silent pass and not a hard fail
        irf.write_text("module {\n}\n")
        res, hard = run()
        cc = {r["check"]: r for r in res}["CONFIG"]
        assert not hard and cc["ok"] and not cc["hard"] and "UNVERIFIED" in cc["msg"]
        irf.write_text('module attributes {"ttg.num-warps" = 8 : i32} {\n}\n')

        # comparator inversions are hard: slower than the default, and slower than the sweep winner
        assert run(lambda c: c.update(champion_ms=12.0))[1]
        assert run(lambda c: c.update(champion_ms=10.5, sweep_winner_ms=10.0))[1]
        # a null default_ms is soft but reported as unprovable, never silently fine
        res, hard = run(lambda c: c.update(default_ms=None))
        cmp_r = {r["check"]: r for r in res}["COMPARATOR"]
        assert not hard and cmp_r["ok"] and not cmp_r["hard"] and "UNPROVABLE" in cmp_r["msg"]

        # partial sampling blocks by default and only degrades when the caller declares it
        assert run(lambda c: c.update(partially_sampled=True))[1]
        res, hard = run(lambda c: c.update(partially_sampled=True), allow_provisional=True)
        assert not hard and "caveats[]" in {r["check"]: r for r in res}["SAMPLING"]["msg"]

        # an ungated sweep blocks by default and only degrades when the caller declares it. This is
        # the stronger of the two caveats: partial sampling means "incomplete measurement", ungated
        # means "the winner was never checked to be correct at all".
        assert run(lambda c: c.update(trust_level="ungated"))[1]
        res, hard = run(lambda c: c.update(trust_level="ungated"), allow_ungated=True)
        assert not hard and "caveats[]" in {r["check"]: r for r in res}["GATED"]["msg"]
        # --allow-provisional must NOT wave through an ungated bundle: different caveat, different flag
        assert run(lambda c: c.update(trust_level="ungated"), allow_provisional=True)[1]
        # a bundle predating trust_level is UNKNOWN, not assumed gated
        res, hard = run(lambda c: c.pop("trust_level", None))
        g = {r["check"]: r for r in res}["GATED"]
        assert not hard and not g["hard"] and "UNKNOWN" in g["msg"], g

        # an empty served_range warns but does not block the deep dig from starting
        res, hard = run(lambda c: c.update(served_range=[]))
        assert not hard and not {r["check"]: r for r in res}["RANGE"]["ok"]
    print("[champion_gate] SELFTEST PASS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--champion", required=True, help="plain_champion.json from the broad-search skill")
    ap.add_argument("--allow-provisional", action="store_true",
                    help="proceed on a partially-sampled comparator, with the inflation declared")
    ap.add_argument("--allow-ungated", action="store_true",
                    help="proceed on a champion whose sweep had no oracle, with it declared")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    path = Path(a.champion).resolve()
    try:
        champ = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        print(f"[champion_gate] cannot read {path}: {e}", file=sys.stderr)
        return 2
    results, hard_fail = gate(champ, path.parent, allow_provisional=a.allow_provisional,
                              allow_ungated=a.allow_ungated)
    if a.json:
        print(json.dumps({"pass": not hard_fail, "checks": results}, indent=2))
    else:
        for r in results:
            tag = "PASS" if r["ok"] else ("FAIL" if r["hard"] else "WARN")
            print(f"  [{tag}] {r['check']:11s} {r['msg']}")
        print("\n" + ("CHAMPION GATE PASS -> may start the deep dig"
                      if not hard_fail else
                      "CHAMPION GATE FAIL -> do NOT start (fix the FAILs above)"))
    return 0 if not hard_fail else 2


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    sys.exit(main())
