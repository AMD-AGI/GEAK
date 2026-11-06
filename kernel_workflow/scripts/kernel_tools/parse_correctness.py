#!/usr/bin/env python3
"""parse_correctness.py - turn a bench `--check` oracle stdout into a machine correctness.json.

A GEAK shared kernel tool (kernel_workflow/scripts/kernel_tools/; the Gluon pack keeps a shim at
scripts/parse_correctness.py).

The per-experiment bench scripts already carry an fp32 correctness oracle (`--check`), but their
stdout is NOT uniform and none prints a clean PASS/FAIL:

  * gemm  ->  "correctness: max|diff|=0.0123 mean=0.00005 rel=4.6e-03"
  * fa-bwd -> "  <label>  dQ=0.0011 dK=0.0013 dV=0.0007"   (per row; "FAILED: <err>" on exception)

This adapter parses either shape against a tolerance and emits the artifact the round pipeline reads,
so a correctness verdict is MACHINE-made, not agent prose. It is the #1 anti-poison gate: a wrong-but-
faster config (v5: dQ rel 0.3952 accepted as a 1.32x "win") can never be timed as a win.

    python3 parse_correctness.py --stdin --parse rel --tol 5e-2 --out correctness.json
    python3 parse_correctness.py --log bench.out --parse dqdkdv --tol 5e-2 --out correctness.json

    # a GEAK unittest.py / harness_lib run (exit code + per-case JSON):
    python3 unittest.py > ut.out; echo $? > ut.rc
    python3 parse_correctness.py --log ut.out --parse geak --exit-code "$(cat ut.rc)" --out correctness.json

parse modes:
  rel     -> read the max `rel=<x>` value; pass iff <= tol
  maxdiff -> read the max `max|diff|=<x>` value; pass iff <= tol
  dqdkdv  -> read every dQ=/dK=/dV= value (fa-bwd); pass iff EVERY one <= tol; any "FAILED:" -> fail
  geak    -> a GEAK unittest / e2e_workflow/scripts/harness_lib.py run. The exit code is the verdict
             (--exit-code): 0 pass | 1 correctness FAIL | 2 environment error (unknown, NOT a kernel
             verdict) | 3 UT harness incomplete (`UT_HARNESS_INCOMPLETE` sentinel: regenerate the UT,
             never blame the candidate). Per-case JSON (`{"case", "correct", "max_rel_err"}` as
             emitted by harness_lib.check_correct_multi / run_correctness, one object per line or one
             JSON document) is parsed into `cases`; a case with correct=false fails even under rc 0,
             and --tol, when given, is an EXTRA stricter bound on every max_rel_err (it can only fail
             a run the UT passed, never pass one it failed).

Exit code of this tool: 0 pass | 1 fail | 2 unknown | 3 harness_incomplete (mirrors the GEAK UT).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys

_RE_REL = re.compile(r"rel\s*=\s*([0-9.eE+-]+)")
_RE_MAXDIFF = re.compile(r"max\|diff\|\s*=\s*([0-9.eE+-]+)")
_RE_DQDKDV = re.compile(r"\b(dQ|dK|dV|dq|dk|dv)\s*=\s*([0-9.eE+-]+)")
_RE_FAILED = re.compile(r"\bFAILED\b|\bCORRECTNESS FAIL\b|\btraceback\b", re.IGNORECASE)
# The GEAK unittest exit-code contract (e2e_workflow/roles/kernel_extractor.md, harness_lib
# HarnessIncompleteError): 0 pass, 1 correctness FAIL, 2 env error, 3 regenerate the UT.
UT_HARNESS_INCOMPLETE = "UT_HARNESS_INCOMPLETE"
_GEAK_RC = {0: "pass", 1: "fail", 2: "unknown", 3: "harness_incomplete"}
_EXIT_FOR_STATUS = {"pass": 0, "fail": 1, "unknown": 2, "harness_incomplete": 3}


def _json_docs(text):
    """Every JSON value in `text`: the whole text if it parses, else each line that does."""
    try:
        return [json.loads(text)]
    except (ValueError, TypeError):
        pass
    docs = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln[0] not in "[{":
            continue
        try:
            docs.append(json.loads(ln))
        except ValueError:
            continue
    return docs


def _cases_in(obj, out):
    """Collect per-case dicts (anything carrying `correct` or `max_rel_err`) from a JSON value."""
    if isinstance(obj, dict):
        if "correct" in obj or "max_rel_err" in obj:
            out.append(obj)
            return out
        for v in obj.values():
            _cases_in(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _cases_in(v, out)
    return out


def parse_geak(text: str, exit_code: int | None, tol: float | None = None) -> dict:
    """Verdict for a GEAK unittest / harness_lib run: exit code first, per-case JSON second."""
    text = text or ""
    res = {"schema": "correctness", "parse": "geak", "tol": tol, "exit_code": exit_code,
           "values": [], "worst": None, "cases": [], "status": "unknown", "reason": None}
    cases = []
    for d in _json_docs(text):
        _cases_in(d, cases)
    res["cases"] = cases
    errs = [c.get("max_rel_err") for c in cases]
    vals = [float(e) for e in errs if isinstance(e, (int, float)) and math.isfinite(float(e))]
    res["values"] = vals
    res["worst"] = max(vals) if vals else None
    failed = [c.get("case") for c in cases if c.get("correct") is False]
    if exit_code == 3 or UT_HARNESS_INCOMPLETE in text:
        res["status"] = "harness_incomplete"
        res["reason"] = ("UT_HARNESS_INCOMPLETE / exit 3: the generated UT is incomplete (e.g. no "
                         ">=2-shape replay bundle for a graph-deploy kernel). Regenerate the UT; this "
                         "is NOT a kernel-correctness verdict.")
        return res
    if exit_code is not None and exit_code not in _GEAK_RC:
        res["reason"] = f"exit code {exit_code} is outside the GEAK UT contract (0/1/2/3)"
        return res
    if exit_code == 2:
        res["reason"] = "exit 2: environment error -- no correctness verdict was reached"
        res["error_class"] = "env"
        return res
    if exit_code == 1 or failed:
        res["status"] = "fail"
        res["reason"] = (f"exit {exit_code}" if exit_code == 1 else "exit 0") + (
            f"; case(s) with correct=false: {failed}" if failed else "")
        if exit_code == 0 and failed:
            res["reason"] += " -- the UT exited 0 but a case FAILED; the case wins"
        return res
    if exit_code is None and not cases:
        res["reason"] = ("no exit code and no per-case JSON -> cannot certify correctness; pass "
                         "--exit-code from the UT run")
        return res
    if tol is not None and res["worst"] is not None and res["worst"] > tol:
        res["status"] = "fail"
        res["reason"] = (f"UT passed, but worst max_rel_err={res['worst']:.4g} > --tol {tol:.4g} "
                         f"(the extra, stricter bound)")
        return res
    res["status"] = "pass"
    res["reason"] = (f"exit {exit_code}" if exit_code is not None else "all cases correct") + (
        f"; {len(cases)} case(s), worst max_rel_err={res['worst']:.4g}" if res["worst"] is not None
        else f"; {len(cases)} case(s)")
    return res


def _floats(pat, text):
    out = []
    for m in pat.finditer(text):
        try:
            out.append(float(m.group(m.lastindex)))
        except (ValueError, IndexError):
            pass
    return out


def parse_correctness(text: str, parse: str, tol: float) -> dict:
    """Return {status, parse, tol, values, worst, oracle_lines, reason}. status in pass|fail|unknown."""
    text = text or ""
    res = {"schema": "correctness", "parse": parse, "tol": tol, "values": [], "worst": None,
           "status": "unknown", "reason": None}
    # an explicit failure marker in the oracle output is an immediate fail (an exception in the bench).
    if _RE_FAILED.search(text):
        res["status"] = "fail"
        res["reason"] = "oracle stdout carries an explicit FAILED / exception marker"
        return res
    if parse == "rel":
        vals = _floats(_RE_REL, text)
    elif parse == "maxdiff":
        vals = _floats(_RE_MAXDIFF, text)
    elif parse == "dqdkdv":
        vals = [v for _k, v in ((m.group(1), float(m.group(2))) for m in _RE_DQDKDV.finditer(text))]
    else:
        res["reason"] = f"unknown parse mode {parse!r}"
        return res
    res["values"] = vals
    if not vals:
        res["reason"] = (f"no correctness values matched for parse={parse!r} -> the oracle did not run "
                         f"or printed an unexpected format; cannot certify correctness")
        res["status"] = "unknown"
        return res
    worst = max(abs(v) for v in vals)
    res["worst"] = worst
    if tol is None:
        res["status"] = "unknown"
        res["reason"] = "no tolerance provided; values parsed but not gated"
        return res
    res["status"] = "pass" if worst <= tol else "fail"
    res["reason"] = (f"worst={worst:.4g} {'<=' if worst <= tol else '>'} tol={tol:.4g} "
                     f"over {len(vals)} value(s)")
    return res


def _selftest() -> int:
    gemm = "impl=plain M=4096 ...\ncorrectness: max|diff|=0.0123 mean=0.00005 rel=4.6e-03\n"
    r = parse_correctness(gemm, "rel", 5e-2)
    assert r["status"] == "pass" and abs(r["worst"] - 4.6e-3) < 1e-9, r
    r_md = parse_correctness(gemm, "maxdiff", 5e-2)
    assert r_md["status"] == "pass" and abs(r_md["worst"] - 0.0123) < 1e-9, r_md
    # a rel above tol -> fail (the v5 poison: rel 0.3952 must FAIL, not be a win)
    bad = "correctness: max|diff|=1.0 mean=0.2 rel=3.952e-01\n"
    rb = parse_correctness(bad, "rel", 5e-2)
    assert rb["status"] == "fail" and rb["worst"] > 5e-2, rb
    # fa-bwd dQ/dK/dV: all within tol -> pass
    fa = ("=== correctness (s=2048, max|diff| vs fp32 torch ref) ===\n"
          "  onekernel_noncausal  dQ=0.0011 dK=0.0013 dV=0.0007\n")
    rf = parse_correctness(fa, "dqdkdv", 5e-2)
    assert rf["status"] == "pass" and len(rf["values"]) == 3, rf
    # fa-bwd with one output out of tol -> fail
    fa_bad = "  onekernel  dQ=0.3952 dK=0.0013 dV=0.0007\n"
    rfb = parse_correctness(fa_bad, "dqdkdv", 5e-2)
    assert rfb["status"] == "fail" and abs(rfb["worst"] - 0.3952) < 1e-9, rfb
    # an explicit FAILED marker -> fail
    rfail = parse_correctness("  onekernel  FAILED: RuntimeError\n", "dqdkdv", 5e-2)
    assert rfail["status"] == "fail", rfail
    # no values matched -> unknown (cannot certify)
    ru = parse_correctness("no oracle here\n", "rel", 5e-2)
    assert ru["status"] == "unknown", ru
    # GEAK unittest contract: exit code 0/1/2/3, the sentinel, and per-case JSON
    ut_ok = ('{"case": "M=1", "correct": true, "max_rel_err": 0.0012}\n'
             '{"case": "M=256", "correct": true, "max_rel_err": 0.0031}\n')
    g0 = parse_geak(ut_ok, 0)
    assert g0["status"] == "pass" and abs(g0["worst"] - 0.0031) < 1e-12 and len(g0["cases"]) == 2, g0
    assert parse_geak(ut_ok, 0, tol=1e-3)["status"] == "fail", "--tol is an extra, stricter bound"
    assert parse_geak(ut_ok, 1)["status"] == "fail"
    g2 = parse_geak("ImportError: no module named aiter\n", 2)
    assert g2["status"] == "unknown" and g2.get("error_class") == "env", g2
    g3 = parse_geak("UT_HARNESS_INCOMPLETE: deploys under CUDA graph but no replay bundle\n", 3)
    assert g3["status"] == "harness_incomplete", g3
    # the sentinel alone (a main() that forgot the dedicated exit code) is still a regenerate
    assert parse_geak("UT_HARNESS_INCOMPLETE: x\n", 1)["status"] == "harness_incomplete"
    # a whole JSON report (run_correctness's dict of per-case lists) and a failing case under rc 0
    rep_doc = json.dumps({"eager": [{"case": "a", "correct": True, "max_rel_err": 1e-3},
                                    {"case": "output_independence", "correct": True,
                                     "max_rel_err": None}],
                          "random": [{"case": "r0", "correct": False, "max_rel_err": 0.4}]})
    gf = parse_geak(rep_doc, 0)
    assert gf["status"] == "fail" and "r0" in gf["reason"], gf
    assert parse_geak("no json, no rc\n", None)["status"] == "unknown"
    assert parse_geak("", 7)["status"] == "unknown"
    assert _EXIT_FOR_STATUS[g3["status"]] == 3 and _EXIT_FOR_STATUS[g0["status"]] == 0
    print(f"[selftest] geak rc0={g0['status']} rc1=fail rc2={g2['status']} rc3={g3['status']} "
          f"case-fail-under-rc0={gf['status']}")
    print(f"[selftest] gemm-rel={r['status']} gemm-maxdiff={r_md['status']} poison-rel={rb['status']} "
          f"fa-dqdkdv={rf['status']} fa-bad={rfb['status']} failed-marker={rfail['status']} "
          f"no-match={ru['status']}")
    print("PARSE_CORRECTNESS SELFTEST PASS")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", help="path to the bench --check stdout (else --stdin)")
    ap.add_argument("--stdin", action="store_true", help="read the oracle stdout from stdin")
    ap.add_argument("--parse", choices=["rel", "maxdiff", "dqdkdv", "geak"],
                    help="how to read the oracle output (geak = GEAK unittest / harness_lib run)")
    ap.add_argument("--tol", type=float, help="pass iff the worst |value| <= tol (geak: an EXTRA, "
                                              "stricter bound on max_rel_err)")
    ap.add_argument("--exit-code", type=int, default=None,
                    help="--parse geak: the UT's exit code (0 pass, 1 fail, 2 env, 3 regenerate UT)")
    ap.add_argument("--oracle", default=None, help="oracle identity (recorded in the artifact)")
    ap.add_argument("--config-id", default=None, help="config identity (recorded in the artifact)")
    ap.add_argument("--out", help="correctness.json output path (default stdout)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(_selftest())
    if not a.parse:
        ap.error("--parse is required")
    if a.log:
        with open(a.log) as f:
            text = f.read()
    else:
        text = sys.stdin.read()
    if a.parse == "geak":
        res = parse_geak(text, a.exit_code, a.tol)
    else:
        res = parse_correctness(text, a.parse, a.tol)
    res["oracle"] = a.oracle
    res["config_id"] = a.config_id
    res["_source"] = "run_round"
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"wrote {a.out}  status={res['status']} worst={res['worst']}")
    else:
        print(json.dumps(res, indent=2))
    # exit non-zero on a fail so a shell caller can gate on it (3 = regenerate the UT, as in GEAK)
    sys.exit(_EXIT_FOR_STATUS.get(res["status"], 2))


if __name__ == "__main__":
    main()
