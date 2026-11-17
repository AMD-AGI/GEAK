#!/usr/bin/env python3
"""Independent verification of a generated kernel UT (spec 3.2): can this test be trusted?

    bash <GEAK>/kernel_workflow/scripts/gpu_lock.sh $GPU_ID python3 $SHARED_TOOLS/verify_ut.py --task .
    python3 $SHARED_TOOLS/verify_ut.py --task <task> --static-only          # CPU part only

Result: run/verify_ut.json, verdict PASS | FAIL | INCONCLUSIVE, bound to the task's contract_sha256 and
snapshot_sha256 (the delivery gates require PASS with both hashes current).

Static (CPU; never imports cases.py):
  - the vendored runtime files are byte-identical to the shared ones next to this script, unittest.py is
    the generated one, run/ut_contract.json still matches the test files;
  - generate_ut.py check: required hooks for the declared capabilities, metadata, case-manifest coverage
    (ISSUE-006), GPU-count declaration, the reverse check of a single-rank declaration (cross-device
    symbols in kernel_src/ need capabilities.single_rank_justification), written reasons for disabled
    controls, empty declared coverage gaps (ISSUE-012).
GPU (run it under the task's GPU lease):
  - A/A (A16 C'): the candidate side runs the task's own overlay and binding with the baseline's own
    source, so a bias of the binding itself (e.g. a cheaper dispatch path) is measured. kernel_src/ holds
    copied source (meta.kernel_extraction.source_origins, origins frozen into run/ut_contract.json) and
    binding files written for the task (meta.kernel_extraction.binding_files); when every copied source
    is byte-identical to its origin, the UT's own result for the same contract and snapshot is the first
    A/A measurement (A08.2), else a fresh run. When the copied source was modified, both legs run the
    baseline (--aa) and verify_ut.json says the A/A did not go through the candidate binding. Every check
    must pass with errors far below tol -- the native self-noise native_noise.noise_rel at most half of tol
    (A37: correct() passes at t exactly when noise_rel <= t; a result without noise_rel is judged by its
    max_rel_err as before; with per-output tolerances, A42, every output's noise_rel at most half of its own
    tolerance), recorded per measurement with tol and 2 x noise_rel, and a note when tol is
    looser than the self-noise needs -- and the weighted speedup AND the call-weighted speedup of every
    phase with call mass must sit inside the A/A band (C3: at most +-3%). Timing always runs the full 3
    pairs (no early stop); outside the band it is measured once more, FAIL only when both are outside
    (A08), naming each out-of-band phase's worst case with both legs' ms, host_ms and dispatch-inclusive
    latency. A through-binding A/A out of band also runs --aa once as a diagnostic (harness or candidate
    side), recorded in verify_ut.json.
  - timing receipt: read-evict cache, the same timer on both legs, device-dominated readings on every
    positive-mass case of a graph deployment, the queued GPU spin on both legs of every eager-deployed
    single-rank case (A16), per-rank times + alignment for a multi-rank task.
  - controls, correctness only, each in its own worker (leg_runner Control): noop and within_tol must
    PASS; scale_out, stale_output (random draws), skip_state_write (stateful: sequence, and the graph
    replay of every group that carries state between consecutive replays, A18) and rank_skew
    (multi-rank: on the last rank) must be REJECTED; task controls (meta.controls.task) as declared.
    A case whose call raised compared no output and is never a catch; a NaN / inf output is (ISSUE-059,
    applied to every control here, stricter than the original which applied it to task controls only).
    Task controls: `equivalent_on` cases do not count, a control seen only there FAILS, a control seen
    only below 1.5 x tol is a warning (ISSUE-033); `expect: below_tol` is verified case by case (every
    error below tol AND the output changed) and is a coverage gap that blocks PASS (ISSUE-028).
    Recorded errors keep 5 decimals, and every record is checked with the static rule before it is written
    (ISSUE-060): a failing case reads above tol, as correct()'s err of a failing comparison always does (above
    min(tol, 1): failing_floor); one stored at tol only by the rounding is recorded rounded UP and named in
    rounded_up. A passing case keeps its stored error and is not held to tol: on a reference element near 0
    correct()'s err of a passing comparison reads up to 1 (about noise_rel / tol, A37).
  - a run stopped from outside its code -- out of GPU memory / resources, a time limit, the GPU lock not
    acquired, foreign work on the GPU, a GPU IPC fault (hipIpcOpenMemHandle invalid device pointer), a
    signal -- has no verdict: it is run again (at most 2 more times; multi-rank: with half as many pools
    alive each time), and stays INCONCLUSIVE if it never gets one (ISSUE-029, 024). An exception raised
    by the task's own code is a FAIL, never a reason to retry (ISSUE-014: every exception is named in the
    report, and the report is always written).
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import ut_runner  # noqa: E402
import generate_ut  # noqa: E402

VERSION = "verify_ut.py v1"
AA_BAND = 0.03                 # C3: the A/A band, at most +-3%
NOISE_FRACTION = 0.5           # "errors far below tol": native self-noise at most half the tolerance (A37: noise_rel)
TOL_SERIES = (1.0, 1.5, 2.0, 3.0, 5.0)    # tolerance steps {1, 1.5, 2, 3, 5} x 10^k (task-authoring.md step 0 (b))
LOOSE_NOTE = "tolerance looser than the measured self-noise needs"
RETRIES = 2                    # ISSUE-029: reruns of a run with no verdict
GENERIC = {
    # name: (must_pass, sections where a failing case must appear (any of), rank that must fail)
    "noop": (True, (), None),
    "within_tol": (True, (), None),
    "scale_out": (False, ("eager", "random", "graph_replay", "sequence"), None),
    "stale_output": (False, ("random",), None),
    "skip_state_write": (False, ("sequence",), None),     # + the carry_state replay group(s), see judge_control
    "rank_skew": (False, ("eager", "random", "graph_replay", "sequence"), "last"),
}
SECTIONS = ("eager", "random", "graph_replay", "sequence")


class Report:
    def __init__(self, task):
        self.task = task
        self.doc = {"contract": "geak_verify_ut/v1", "generated_by": VERSION, "task_dir": task,
                    "verdict": None, "static": {}, "aa": {}, "receipt": {}, "controls": {},
                    "gaps": [], "errors": [], "warnings": [], "notes": [], "attempts": []}

    def err(self, msg):
        self.doc["errors"].append(msg)

    def warn(self, msg):
        self.doc["warnings"].append(msg)


# --------------------------------------------------------------------------- static
def static_checks(task, rep):
    ok = True
    for name in ut_runner.RUNTIME_FILES:
        mine, theirs = os.path.join(task, name), os.path.join(_HERE, name)
        if not os.path.isfile(mine):
            rep.err(f"runtime file {name} missing from the task")
            ok = False
        elif ut_runner.sha256_file(mine) != ut_runner.sha256_file(theirs):
            rep.err(f"runtime file {name} differs from the shared {theirs} (re-vendor with generate_ut.py)")
            ok = False
    try:
        with open(os.path.join(task, "unittest.py"), encoding="utf-8") as fh:
            if fh.read() != generate_ut.UNITTEST_TEMPLATE:
                rep.err("unittest.py is not the generated entry (generate_ut.py vendor)")
                ok = False
    except OSError:
        rep.err("unittest.py missing")
        ok = False
    contract, problems = ut_runner.check_contract(task)
    rep.doc["contract_sha256"] = contract
    for p in problems:
        rep.err(f"contract: {p}")
        ok = False
    try:
        res = generate_ut.check(task)
    except Exception as exc:            # ISSUE-014
        res = {"ok": False, "problems": [f"check raised {type(exc).__name__}: {exc}"], "warnings": []}
    for p in res.get("problems") or []:
        rep.err(f"check: {p}")
        ok = False
    for w in res.get("warnings") or []:
        rep.warn(f"check: {w}")
    try:
        rep.doc["snapshot_sha256"] = ut_runner.source_snapshot(task)["snapshot_sha256"]
    except Exception as exc:
        rep.err(f"snapshot: {exc}")
        ok = False
    rep.doc["static"] = {"ok": ok, "world_size": res.get("world_size", 1)}
    return ok


# --------------------------------------------------------------------------- running the driver
def _driver(task, args, out, env_extra=None, timeout=None):
    """Run ut_runner in a fresh process (the task's vendored copy). Returns (exit_code, result dict)."""
    cmd = [sys.executable, os.path.join(task, "ut_runner.py"), "--task", task, "--out", out, *args]
    env = dict(os.environ)
    env.update(env_extra or {})
    try:
        os.remove(out)                     # a retry reuses --out: only this run's own result may be read back
    except FileNotFoundError:
        pass
    try:
        p = subprocess.run(cmd, cwd=task, env=env, capture_output=True, text=True, timeout=timeout)
        code, text = p.returncode, (p.stdout or "") + "\n--- stderr ---\n" + (p.stderr or "")[-20000:]
    except subprocess.TimeoutExpired as exc:
        code, text = ut_runner.EXIT_ENV, f"verify_ut: driver timed out after {timeout}s: {exc}"
    log = out[:-5] + ".log" if out.endswith(".json") else out + ".log"
    with open(log, "w", encoding="utf-8") as fh:
        fh.write(f"$ {' '.join(cmd)}\n[exit {code}]\n{text}")
    try:
        with open(out, encoding="utf-8") as fh:
            res = json.load(fh)
    except (OSError, ValueError):
        res = {"exit_code": code, "reason": text[-3000:]}
    if code < 0:                               # killed by a signal we did not send: no verdict
        res["killed_by_signal"] = -code
        code = ut_runner.EXIT_ENV
    res.setdefault("exit_code", code)
    res["log"] = log
    return code, res


def _no_verdict(code, res):
    """A run stopped from outside its code (ISSUE-029): no verdict, run again."""
    if code == ut_runner.EXIT_ENV:
        return True
    return bool(ut_runner.env_fault(str(res.get("reason") or "")))


def run_with_retries(rep, label, fn):
    """fn(attempt, parallel) -> (code, res). Reruns runs without a verdict (ISSUE-029); returns the last."""
    parallel = None
    for attempt in range(RETRIES + 1):
        code, res = fn(attempt, parallel)
        rep.doc["attempts"].append({"what": label, "attempt": attempt, "exit_code": code,
                                    "pool_parallel": parallel, "reason": str(res.get("reason") or "")[:600],
                                    "log": res.get("log")})
        if not _no_verdict(code, res):
            return code, res, False
        parallel = max(1, (parallel or res.get("pool_parallel") or 2) // 2)
    return code, res, True


# --------------------------------------------------------------------------- A/A
def binding_files(meta):
    """meta.kernel_extraction.binding_files: the kernel_src/ files written for the task (bindings, build
    glue), as opposed to copied source with a recorded origin (A16 C')."""
    b = (meta.get("kernel_extraction") or {}).get("binding_files") or []
    return [os.path.normpath(x) for x in b if isinstance(x, str)]


def aa_source(task, meta):
    """Can the A/A run through the candidate binding with the baseline's own source (A16 C')?

    ("binding", why) when every kernel_src/ source file is either copied source byte-identical to the
    origin recorded at freeze (meta.kernel_extraction.source_origins -> run/ut_contract.json) or a declared
    binding file; ("modified", why) otherwise -- the copied source was changed (or its origin moved), so
    the candidate side is not the baseline's code and the A/A falls back to --aa."""
    try:
        with open(os.path.join(task, "run", "ut_contract.json"), encoding="utf-8") as fh:
            frozen = json.load(fh)
    except (OSError, ValueError):
        return "modified", "no frozen contract"
    origins = frozen.get("kernel_src_origin") or {}
    bindings = set(binding_files(meta))
    snap = ut_runner.source_snapshot(task)
    srcs = [f for f in snap["files"] if f["path"].startswith("kernel_src" + os.sep)]
    copied = 0
    for f in srcs:
        if f["path"] in bindings:
            continue
        o = origins.get(f["path"])
        if not o:
            return "modified", (f"{f['path']} is neither copied source with a recorded origin "
                                "(meta.kernel_extraction.source_origins) nor a declared binding file")
        if o.get("sha256") != f["sha256"]:
            return "modified", f"{f['path']} differs from its origin {o.get('origin')}"
        try:
            if ut_runner.sha256_file(o["origin"]) != o["sha256"]:
                return "modified", f"origin {o['origin']} changed since freeze"
        except OSError:
            return "modified", f"origin {o['origin']} unreadable"
        copied += 1
    if not copied:
        return "modified", "kernel_src has no copied source with a recorded origin"
    return "binding", (f"{copied} copied source file(s) byte-identical to their origins, "
                       f"{len(bindings)} declared binding file(s)")


def kernel_src_unmodified(task, meta=None):
    """Back-compatible form of aa_source: (True, why) when the A/A can reuse / run through the binding."""
    if meta is None:
        with open(os.path.join(task, "meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
    kind, why = aa_source(task, meta)
    return kind == "binding", why


def phase_values(res, meta):
    """{phase: call-weighted speedup} of a UT result: timing.by_phase (ut_runner), else recomputed from
    per_case with meta.workload.cases effective_calls (a result written before by_phase existed)."""
    t = (res or {}).get("timing") or {}
    bp = t.get("by_phase")
    if isinstance(bp, dict):
        return {ph: v.get("speedup") for ph, v in bp.items() if isinstance(v, dict)}
    calls = {c.get("sig"): c.get("effective_calls") for c in ((meta.get("workload") or {}).get("cases") or [])
             if isinstance(c, dict)}
    acc = {}
    for c in t.get("per_case") or []:
        b, o, n = c.get("baseline_ms"), c.get("optimized_ms"), calls.get(c.get("sig"))
        if ut_runner._finite(b) and ut_runner._finite(o) and o > 0 and ut_runner._finite(n) and n > 0:
            a = acc.setdefault(c.get("regime") or "", [0.0, 0.0])
            a[0] += b * n
            a[1] += o * n
    return {ph: a[0] / a[1] for ph, a in acc.items() if a[1] > 0}


def _leg(c, leg):
    rec = ((c.get("legs") or {}).get(leg)) or {}
    return {"ms": c.get("baseline_ms" if leg == "baseline" else "optimized_ms"), "host_ms": rec.get("host_ms"),
            "latency_ms": rec.get("latency_ms")}


def worst_case(res, phase):
    """The case of `phase` whose A/A deviates most, with both legs' ms, host_ms and dispatch-inclusive
    latency (named when the phase is out of band)."""
    rows = [c for c in ((res or {}).get("timing") or {}).get("per_case") or []
            if (c.get("regime") or "") == phase and ut_runner._finite(c.get("speedup"))]
    if not rows:
        return None
    c = max(rows, key=lambda r: abs(r["speedup"] - 1.0))
    return {"sig": c.get("sig"), "speedup": c["speedup"], "baseline": _leg(c, "baseline"),
            "candidate": _leg(c, "current")}


def tol_step(x):
    """The smallest tolerance step {1, 1.5, 2, 3, 5} x 10^k that is not below x > 0."""
    k = math.floor(math.log10(x))
    for e in (k - 1, k, k + 1):
        for m in TOL_SERIES:
            v = m * 10.0 ** e
            if v >= x * (1 - 1e-9):
                return float(f"{v:.12g}")
    return float(f"{10.0 ** (k + 2):.12g}")


def self_noise(res, tol):
    """A37: the native self-noise record of one A/A measurement and the problem it raises, if any.
    With native_noise.noise_rel (correct()'s scale: it passes at t exactly when noise_rel <= t) the rule is
    noise_rel <= NOISE_FRACTION x tol; a result written before noise_rel existed is judged by its max_rel_err
    as before. `loose`: noise_rel > 0 and tol above the step holding 3 x noise_rel (a note, never a verdict).
    With per-output tolerances (A42, native_noise.per_output) the rule and the note apply output by output, each
    output's noise_rel against its own tolerance."""
    nn = res.get("native_noise") or {}
    if isinstance(nn.get("per_output"), dict) and nn["per_output"]:
        return self_noise_per_output(nn, tol)
    nr, mre = nn.get("noise_rel"), nn.get("max_rel_err")
    rec = {"noise_rel": nr, "max_rel_err": mre, "tol": tol,
           "twice_noise_rel": 2 * nr if ut_runner._finite(nr) else None,
           "rule": "noise_rel" if "noise_rel" in nn else "max_rel_err (a result without noise_rel)"}
    problem, loose = None, False
    if "noise_rel" in nn:
        if ut_runner._finite(nr):
            if tol and nr > NOISE_FRACTION * tol:
                problem = (f"native self-noise noise_rel {nr:.4g} is above {NOISE_FRACTION} x tol {tol} (correct() "
                           f"passes at a tolerance t exactly when noise_rel <= t; the self-noise may use at most "
                           f"half of the band): tol must be at least 2 x noise_rel = {2 * nr:.4g}")
            loose = bool(nr > 0 and tol and tol > tol_step(3 * nr) * (1 + 1e-9))
            if loose:
                rec["step_for_3x_noise_rel"] = tol_step(3 * nr)
        elif nn.get("noise_rel_nonfinite"):
            problem = (f"native self-noise has {nn['noise_rel_nonfinite']} comparison(s) with a non-finite noise_rel "
                       "(non-finite values that differ, or a shape mismatch)")
    elif ut_runner._finite(mre) and tol and mre > NOISE_FRACTION * tol:
        problem = f"native run-to-run error {mre} is not far below tol {tol} (> {NOISE_FRACTION} x tol)"
    return rec, problem, loose


def self_noise_per_output(nn, tol):
    """self_noise of a task with per-output tolerances (A42): every output's noise_rel at most NOISE_FRACTION x its own
    tolerance; `loose` when some output's tolerance is above the step holding 3 x its noise_rel."""
    nr, mre = nn.get("noise_rel"), nn.get("max_rel_err")
    rec = {"noise_rel": nr, "max_rel_err": mre, "tol": tol, "rule": "noise_rel per output", "per_output": {}}
    problems, loose = [], False
    for output, v in nn["per_output"].items():
        v = v if isinstance(v, dict) else {}
        n, t = v.get("noise_rel"), v.get("tol")
        t = float(t) if ut_runner._finite(t) else tol
        row = {"noise_rel": n, "tol": t, "twice_noise_rel": 2 * n if ut_runner._finite(n) else None}
        if ut_runner._finite(n):
            if t and n > NOISE_FRACTION * t:
                problems.append(f"output {output}: native self-noise noise_rel {n:.4g} is above {NOISE_FRACTION} x its "
                                f"tolerance {t} (correct() passes an output at a tolerance t exactly when its noise_rel "
                                f"<= t; the self-noise may use at most half of the band): its tolerance must be at "
                                f"least 2 x noise_rel = {2 * n:.4g}")
            if n > 0 and t and t > tol_step(3 * n) * (1 + 1e-9):
                row["step_for_3x_noise_rel"] = tol_step(3 * n)
                loose = True
        elif v.get("noise_rel_nonfinite"):
            problems.append(f"output {output}: native self-noise has {v['noise_rel_nonfinite']} comparison(s) with a "
                            "non-finite noise_rel (non-finite values that differ, or a shape mismatch)")
        rec["per_output"][output] = row
    return rec, "; ".join(problems) or None, loose


def _aa_ok(res, tol, rep, where, meta=None):
    """(timing within band, correctness ok, problems, detail). In band = the weighted A/A AND the
    call-weighted A/A of every phase with call mass within +-AA_BAND (A16 C'). The native self-noise must use
    at most half the tolerance (self_noise, A37); detail.self_noise records it."""
    problems = []
    t = res.get("timing") or {}
    # The UT's primary metric: the weighted speedup, or the unweighted geomean of a task without
    # workload weights (A19; a result written before `primary` existed has only `weighted`).
    w = t.get("primary", t.get("weighted"))
    phases = phase_values(res, meta or {})
    out = {ph: v for ph, v in phases.items() if not (ut_runner._finite(v) and abs(v - 1.0) <= AA_BAND)}
    in_band = ut_runner._finite(w) and abs(w - 1.0) <= AA_BAND and not out
    if not ut_runner._finite(w):
        problems.append(f"{where}: no primary (weighted / unweighted geomean) A/A speedup")
    corr = res.get("gates", {}).get("correctness") in ("pass", None)
    noise, problem, loose = self_noise(res, tol)
    if problem:
        problems.append(f"{where}: {problem}")
        corr = False
    if loose and rep is not None:
        if "per_output" in noise:
            note = (f"{LOOSE_NOTE}: " + ", ".join(
                f"output {o} tol {v['tol']}, native noise_rel {v['noise_rel']:.4g}, the step holding 3 x noise_rel "
                f"{v['step_for_3x_noise_rel']:g}" for o, v in noise["per_output"].items() if "step_for_3x_noise_rel" in v)
                + f" ({where}; task-authoring.md step 0 (b))")
        else:
            note = (f"{LOOSE_NOTE}: tol {tol}, native noise_rel {noise['noise_rel']:.4g} ({where}); the step holding "
                    f"3 x noise_rel is {noise['step_for_3x_noise_rel']:g} (task-authoring.md step 0 (b))")
        if note not in rep.doc["notes"]:
            rep.doc["notes"].append(note)
    detail = {"weighted": w, "by_phase": phases,
              "out_of_band": {ph: {"speedup": v, "worst_case": worst_case(res, ph)} for ph, v in out.items()},
              "self_noise": noise}
    return in_band, corr, problems, detail


def _reusable_ut(task, rep):
    """run/ut_result.json when it is a passing run of the same contract and snapshot (A08.2)."""
    ut_path = os.path.join(task, "run", "ut_result.json")
    try:
        with open(ut_path, encoding="utf-8") as fh:
            ut = json.load(fh)
    except (OSError, ValueError):
        return None
    if ut.get("exit_code") == 0 and ut.get("contract_sha256") == rep.doc.get("contract_sha256") \
            and ut.get("snapshot_sha256") == rep.doc.get("snapshot_sha256"):
        return ut
    return None


def _measure(task, rep, work, label, native_dir, through):
    """One full A/A timing run: through the candidate binding, or both legs on the baseline (--aa)."""
    out = os.path.join(work, f"{label}.json")

    def go(attempt, parallel):
        extra = ["--no-early-stop", "--reps", "3", "--native-dir", native_dir] + ([] if through else ["--aa"])
        env = {"GEAK_UT_POOL_PARALLEL": str(parallel)} if parallel else None
        return _driver(task, extra, out, env)
    code, res, nv = run_with_retries(rep, label, go)
    return out, code, res, nv


def run_aa(task, meta, rep, work, args):
    """The A/A (spec 3.2.1, A08, A16 C'): the candidate side runs the task's own overlay and binding with
    the baseline's own source, so a bias from the binding (e.g. a cheaper dispatch path) is measured too.

    - copied source byte-identical to its origins and every other kernel_src/ file a declared binding:
      the UT's own result (same contract + snapshot) is the first measurement (A08.2), else a fresh run;
    - copied source modified: both legs run the baseline (--aa), noted as not through the binding.
    In band = weighted AND every phase with call mass within +-AA_BAND. Outside: measured once more,
    FAIL only when both are outside (A08.1), naming the phase's worst case with both legs' ms, host_ms
    and dispatch-inclusive latency. --aa is also run once as a diagnostic when a through-binding A/A is out
    of band: it tells a harness bias from a candidate-side one."""
    tol = float((meta.get("tolerance") or {}).get("tol", meta.get("tol", 0)) or 0)
    kind, why = aa_source(task, meta)
    through = kind == "binding"
    rep.doc["aa"]["through_candidate_binding"] = {"value": through, "why": why}
    if not through:
        rep.doc["notes"].append(f"A/A did not go through the candidate binding (both legs ran the baseline, "
                                f"--aa): {why}")
    native_dir = os.path.join(work, "native")
    runs, last = [], None
    reused = _reusable_ut(task, rep) if through else None
    for i in range(2):
        if i == 0 and reused is not None:
            source, res, code, nv = "run/ut_result.json", reused, 0, False
        else:
            source, code, res, nv = _measure(task, rep, work, f"aa_{len(runs) + 1}", native_dir, through)
        if nv:
            runs.append({"source": source, "exit_code": code, "no_verdict": True,
                         "reason": str(res.get("reason"))[:800]})
            rep.doc["aa"].update(verdict="INCONCLUSIVE", runs=runs)
            return "INCONCLUSIVE", None
        if code != 0:
            runs.append({"source": source, "exit_code": code, "reason": str(res.get("reason"))[:2000]})
            rep.doc["aa"].update(verdict="FAIL", runs=runs)
            rep.err(f"A/A run failed (exit {code}): {str(res.get('reason'))[:600]}")
            return "FAIL", res
        in_band, corr, probs, detail = _aa_ok(res, tol, rep, source, meta)
        runs.append(dict(detail, source=source if source.startswith("run/") else os.path.relpath(source, task),
                         in_band=in_band, correctness_ok=corr, problems=probs))
        if not corr:
            rep.doc["aa"].update(verdict="FAIL", runs=runs)
            for p in probs:
                rep.err(f"A/A: {p}")
            return "FAIL", res
        if in_band:
            rep.doc["aa"].update(verdict="PASS", runs=runs,
                                 receipt_from=source if source.startswith("run/") else os.path.relpath(source, task))
            return "PASS", res
        last = detail
        if i == 0 and through:
            rep.doc["aa"]["diagnostic"] = _aa_diagnostic(task, meta, rep, work, native_dir, tol, detail)
    rep.doc["aa"].update(verdict="FAIL", runs=runs)
    named = []
    for ph, v in (last or {}).get("out_of_band", {}).items():
        wc = v.get("worst_case") or {}
        named.append(f"phase {ph or '(single)'} A/A {v.get('speedup')}: worst case {wc.get('sig')} "
                     f"{wc.get('speedup')} (baseline {wc.get('baseline')}, candidate {wc.get('candidate')})")
    rep.err(f"A/A outside +-{AA_BAND:.0%} in {len(runs)} measurements: weighted "
            f"{[r.get('weighted') for r in runs]}" + (("; " + "; ".join(named)) if named else ""))
    return "FAIL", None


def _aa_diagnostic(task, meta, rep, work, native_dir, tol, through_detail):
    """--aa once (both legs on the baseline) after an out-of-band A/A through the binding: in band means
    the bias sits on the candidate side (binding / build), out of band means the harness itself."""
    source, code, res, nv = _measure(task, rep, work, "aa_diagnostic", native_dir, False)
    if nv or code != 0:
        return {"source": os.path.relpath(source, task), "exit_code": code, "no_verdict": nv,
                "reason": str(res.get("reason"))[:600]}
    in_band, _corr, _probs, detail = _aa_ok(res, tol, rep, source, meta)
    return dict(detail, source=os.path.relpath(source, task), in_band=in_band,
                reading=("candidate side: the baseline against itself is in band, so the bias comes from the "
                         "candidate's binding / build") if in_band else
                        ("harness: the baseline against itself is out of band too"),
                through_binding_out_of_band=sorted(through_detail.get("out_of_band", {})))


def check_receipt(meta, res, rep):
    """The timing receipt of the A/A (or reused UT) run."""
    problems = []
    rc = (res or {}).get("receipt")
    if not isinstance(rc, dict):
        rep.err("timing receipt missing from the A/A result")
        rep.doc["receipt"] = {"ok": False}
        return False
    if rc.get("cache_mode") != "read-evict":
        problems.append(f"cache_mode {rc.get('cache_mode')!r} is not read-evict")
    world = int(rc.get("world_size") or 1)
    for sig, c in (rc.get("cases") or {}).items():
        b, o = c.get("baseline") or {}, c.get("current") or {}
        if b.get("timer") != o.get("timer"):
            problems.append(f"{sig}: legs used different timers")
        if world == 1 and c.get("deployed_graph") is False:
            # A16: an eager-deployed case is timed on a queued GPU spin on both legs, with the same method.
            for leg, x in (("baseline", b), ("current", o)):
                if x.get("timer") == "cuda_event" and not isinstance(x.get("spin"), dict):
                    problems.append(f"{sig}: {leg} eager timing lacks the queued GPU spin (A16)")
            sb, so = b.get("spin") or {}, o.get("spin") or {}
            if isinstance(b.get("spin"), dict) and isinstance(o.get("spin"), dict) \
                    and sb.get("queued") != so.get("queued"):
                problems.append(f"{sig}: the legs queued different spins ({sb.get('queued')} / {so.get('queued')})")
        if world > 1:
            for leg, x in (("baseline", b), ("current", o)):
                rk = x.get("rank_ms")
                if not isinstance(rk, list) or len(rk) != world or not all(ut_runner._finite(v) for v in rk):
                    problems.append(f"{sig}: {leg} lacks a time for each of the {world} ranks")
    if world > 1 and not isinstance(rc.get("alignment"), dict):
        problems.append("multi-rank receipt lacks the alignment method")
    for p in problems:
        rep.err(f"receipt: {p}")
    rep.doc["receipt"] = {"ok": not problems, "problems": problems, "world_size": world,
                          "cache_mode": rc.get("cache_mode"), "timer": rc.get("timer"),
                          "all_primed": rc.get("all_primed"), "alignment": rc.get("alignment"),
                          "flags": rc.get("flags") or []}
    return not problems


# --------------------------------------------------------------------------- controls
def _rank_reports(judged):
    """[(rank, report)] of a judged correctness result (single rank: [(0, report)])."""
    if isinstance(judged.get("ranks"), list):
        return [(r.get("rank", i), r.get("report") or {}) for i, r in enumerate(judged["ranks"])]
    return [(0, judged.get("report") or {})]


def classify_cases(report, tol, equivalent_on=()):
    """Every compared case of one rank's report: [{section, case, correct, err, kind}] with kind
    pass | fail (a seen bug) | nonfinite (NaN/inf output: a seen bug) | raised (no output compared)."""
    rows = []

    def section_rows(section, items):
        for c in items:
            if not isinstance(c, dict) or c.get("case") in ("output_independence",):
                continue
            note = str(c.get("note") or "")
            err = c.get("max_rel_err")
            if c.get("correct") is True:
                kind = "pass"
            elif "raised" in note or "graph_capture_failed" in note or "capture" in note and err is None:
                kind = "raised"
            elif err is None:
                kind = "nonfinite"
            else:
                kind = "fail"
            sig = ut_runner._sig_of(c.get("case") or "")
            row = {"section": section, "case": c.get("case"), "sig": sig, "correct": c.get("correct"),
                   "err": err, "kind": kind, "equivalent": sig in set(equivalent_on or ())}
            if ut_runner._finite(c.get("tol")):
                row.update(tol=float(c["tol"]), output=c.get("output"))   # A42: the row's output and its tolerance
            rows.append(row)

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in SECTIONS and isinstance(v, list):
                    section_rows(k, v)
                elif k == "error" or k == "incomplete":
                    rows.append({"section": "error", "case": None, "sig": None, "correct": False,
                                 "err": None, "kind": "raised", "equivalent": False, "note": str(v)[:300]})
                else:
                    walk(v)
    walk(report)
    return rows


def _above_tol_5dp(tol):
    """The smallest 5-decimal value strictly above tol (a failing error rounded UP, ISSUE-060)."""
    v = math.ceil(tol * 1e5 - 1e-6) / 1e5
    if v <= tol:
        v += 1e-5
    return round(v, 5)


def _row_tol(r, tol):
    """The tolerance a row was compared at: its own output's (per-output tolerances, A42), else the task's."""
    return r["tol"] if ut_runner._finite(r.get("tol")) else tol


def largest_tolerance(meta, tol):
    """The largest tolerance an output of the task is compared at: tol, or a larger meta.tolerance.per_output value
    (A42). failing_floor needs it only when some tolerance is above 1."""
    per = (meta.get("tolerance") or {}).get("per_output") if isinstance(meta.get("tolerance"), dict) else None
    per = per if isinstance(per, dict) else {}
    return max([tol] + [float(v) for v in per.values() if ut_runner._finite(v)])


def failing_floor(t, tol_max=None):
    """What correct()'s err of a comparison that FAILED at tolerance t always exceeds (the static rule, ISSUE-060).

    correct() fails an element when |d| > t (|ref| + R), R = RMS(ref), and reads it as |d| / (|ref| + t R): above t
    when t <= 1, above 1 when t > 1, so min(t, 1). A row compared output by output (A42) records the failing output
    furthest over its own tolerance t (the largest err / t); it reads above t / max(1, T), T = tol_max the largest
    tolerance of the task -- t when every tolerance is at most 1. A passing comparison has no bound of this kind
    below 1: on a reference element of 0 its err reads |d| / (t R), the share of the band used (A37) -- up to 1, and
    above t as soon as |d| > t^2 R -- so the static rule does not hold a passing case to tol."""
    return t / max(1.0, t, tol_max or 0.0)


def _rounded_onto(e, floor):
    """ISSUE-060: a failing case's stored error e (5 decimals) is at or below `floor` only because it is rounded: no
    less than what the smallest float above floor rounds to, and not above floor."""
    return round(math.nextafter(floor, math.inf), 5) <= e <= floor


def record_errors(rows, tol, tol_max=None):
    """ISSUE-060: keep 5 decimals. A case the harness FAILED reads above failing_floor (tol, up to 1); one whose
    stored error is at or below it only because it is rounded is recorded rounded UP and named in rounded_up. One
    stored lower is recorded as stored: it failed on something its error does not show, and the static rule refuses
    it. A row compared at its own output's tolerance (A42) is held to that tolerance."""
    rounded_up = []
    for r in rows:
        e = r.get("err")
        if not ut_runner._finite(e):
            continue
        floor = failing_floor(_row_tol(r, tol), tol_max)
        if r["kind"] == "fail" and _rounded_onto(e, floor):
            r["err_recorded"] = _above_tol_5dp(floor)
            rounded_up.append(r["case"])
        else:
            r["err_recorded"] = round(e, 5)
    return rounded_up


def static_record_problems(rows, tol, tol_max=None):
    """The static rule a record must satisfy before it is written (ISSUE-060): a failing case reads above
    failing_floor -- tol, or 1 when tol is above 1 (a row compared at its own output's tolerance, A42: that
    tolerance) -- as correct()'s err of a failing comparison always does. A passing case keeps its stored error and
    is not held to tol: correct()'s err of a passing comparison reads up to 1 on a reference element near 0."""
    bad = []
    for r in rows:
        e = r.get("err_recorded")
        if r["kind"] != "fail" or not ut_runner._finite(e):
            continue
        t = _row_tol(r, tol)
        floor = failing_floor(t, tol_max)
        if not e > floor:
            bad.append(f"{r['case']}: failing at {e} <= " + (f"tol {t}" if floor == t else f"{floor:.6g} (tol {t})"))
    return bad


def carry_groups(report):
    """{phase label: {group index}} of the replay groups that carry state across consecutive replays (A18.3),
    read from one rank's correctness report (`replay_groups` of each phase)."""
    out = {}
    for label, rep in (report or {}).items():
        if isinstance(rep, dict) and isinstance(rep.get("replay_groups"), list):
            idx = {g.get("group") for g in rep["replay_groups"] if isinstance(g, dict) and g.get("carry_state")}
            if idx:
                out[label] = idx
    return out


def carry_rows_failed(report, tol):
    """Failing (not raising) graph_replay rows of the carry_state groups in one rank's report."""
    rows = []
    for label, idx in carry_groups(report).items():
        for c in (report.get(label) or {}).get("graph_replay") or []:
            if isinstance(c, dict) and c.get("group") in idx and c.get("correct") is False:
                note = str(c.get("note") or "")
                if "raised" in note or ("capture" in note and c.get("max_rel_err") is None):
                    continue
                rows.append(f"{label}:group{c.get('group')}:{c.get('case')}")
    return rows


def judge_control(name, spec, code, res, tol, world, tol_max=None):
    """(verdict PASS|FAIL|INCONCLUSIVE, detail) of one control run. tol_max: largest_tolerance (failing_floor)."""
    generic = name in GENERIC
    must_pass, sections, rank_rule = GENERIC.get(name, (False, SECTIONS, None))
    expect = (spec or {}).get("expect", "pass" if must_pass else "reject")
    equivalent = (spec or {}).get("equivalent_on") or []
    judged = ((res.get("correctness") or {}).get("candidate")) or {}
    detail = {"exit_code": code, "expect": expect}
    if code == ut_runner.EXIT_ENV:
        return "INCONCLUSIVE", dict(detail, why=str(res.get("reason"))[:600])
    if code == ut_runner.EXIT_INCOMPLETE:
        return "FAIL", dict(detail, why="the control could not be evaluated: " + str(res.get("reason"))[:600])
    per_rank = {}
    rounded = []
    for rank, report in _rank_reports(judged):
        rows = classify_cases(report, tol, equivalent)
        rounded += record_errors(rows, tol, tol_max)
        per_rank[rank] = rows
    detail["rounded_up"] = rounded
    bad_records = [p for rows in per_rank.values() for p in static_record_problems(rows, tol, tol_max)]
    if bad_records:
        return "FAIL", dict(detail, why="record refused by the static rule: " + "; ".join(bad_records[:6]))
    seen = {rank: [r for r in rows if r["kind"] in ("fail", "nonfinite") and not r["equivalent"]
                   and (not sections or r["section"] in sections)] for rank, rows in per_rank.items()}
    raised = [r for rows in per_rank.values() for r in rows if r["kind"] == "raised"]
    only_equiv = [r for rows in per_rank.values() for r in rows if r["kind"] in ("fail", "nonfinite") and r["equivalent"]]
    detail.update(seen_cases={str(k): [f"{r['section']}:{r['case']}" for r in v][:20] for k, v in seen.items()},
                  raised_cases=[f"{r['section']}:{r['case']}" for r in raised][:20],
                  equivalent_only=[f"{r['section']}:{r['case']}" for r in only_equiv][:20])
    if expect == "pass":
        failing = [r for rows in per_rank.values() for r in rows if r["kind"] != "pass"]
        if code == 0 and not failing:
            return "PASS", detail
        return "FAIL", dict(detail, why=f"must PASS but the UT rejected it ({len(failing)} failing/raised case(s))")
    if expect == "below_tol":
        rows = [r for rs in per_rank.values() for r in rs]
        above = [r for r in rows if r["kind"] != "pass" and not r["equivalent"]]
        changed = any(ut_runner._finite(r.get("err")) and r["err"] > 0 for r in rows)
        if above:
            return "FAIL", dict(detail, why="declared below_tol but some case exceeds tol: "
                                            + ", ".join(str(r["case"]) for r in above[:6]))
        if not changed:
            return "FAIL", dict(detail, why="declared below_tol but no output changed (the bug never acted)")
        return "GAP", dict(detail, why="verified below tol on every case: a coverage gap the test cannot see")
    # expect reject
    if rank_rule == "last":
        last = max(per_rank) if per_rank else world - 1
        if not seen.get(last):
            return "FAIL", dict(detail, why=f"not rejected on the perturbed rank {last} "
                                            "(per-rank inputs must come from ctx / h.rank_context())")
        return "PASS", detail
    if not any(seen.values()):
        why = "never rejected"
        if raised:
            why += f" (only raising calls, which compare no output: {len(raised)})"
        if only_equiv:
            why += " (failed only on its declared equivalent_on cases)"
        if sections and generic:
            why += f"; must fail in one of {list(sections)}"
        return "FAIL", dict(detail, why=why)
    if name == "skip_state_write" and generic:
        # A18.3: a group replaying consecutive steps that carry state must see the dropped state writes too.
        reports = _rank_reports(judged)
        if any(carry_groups(r) for _, r in reports):
            carry_seen = [x for _, r in reports for x in carry_rows_failed(r, tol)]
            detail["carry_state_rows"] = carry_seen[:20]
            if not carry_seen:
                return "FAIL", dict(detail, why="rejected in sequence but not in the graph replay of the "
                                                "carry_state group(s): consecutive replays do not read the state "
                                                "the previous replay wrote")
    errs = [r["err"] for v in seen.values() for r in v if ut_runner._finite(r.get("err"))]
    near = [r["err"] < 1.5 * _row_tol(r, tol) for v in seen.values() for r in v if ut_runner._finite(r.get("err"))]
    if not generic and errs and all(near) and not any(r["kind"] == "nonfinite" for v in seen.values() for r in v):
        detail["warning"] = f"seen only below 1.5 x tol (max error {max(errs)}) (ISSUE-033)"
    return "PASS", detail


def control_plan(meta, world, rep):
    caps = meta.get("capabilities") or {}
    disabled = ((meta.get("controls") or {}).get("disabled") or {})
    names = ["noop", "within_tol", "scale_out", "stale_output"]
    if caps.get("stateful"):
        names.append("skip_state_write")
    if world > 1:
        names.append("rank_skew")
    plan = []
    for n in names:
        if n in disabled:
            rep.doc["controls"][n] = {"verdict": "DISABLED", "why": disabled[n]}
            continue
        plan.append((n, {}))
    for c in (meta.get("controls") or {}).get("task") or []:
        plan.append((c["name"], dict(c)))
    return plan


def run_controls(task, meta, rep, work, world, native_dir):
    """Every control, ONE AT A TIME: each is a correctness-only run with no timing (A08.2), and running
    them serially also keeps a peer-memory kernel's pools from sharing the GPUs (A04)."""
    tol = float((meta.get("tolerance") or {}).get("tol", meta.get("tol", 0)) or 0)
    tol_max = largest_tolerance(meta, tol)
    results = {}
    for name, spec in control_plan(meta, world, rep):
        out = os.path.join(work, f"control_{name}.json")

        def go(attempt, parallel, name=name, spec=spec, out=out):
            extra = ["--native-dir", native_dir, "--reuse-native", "--control", name]
            if spec:
                extra += ["--control-spec", json.dumps({k: v for k, v in spec.items() if k != "name"})]
            env = {"GEAK_UT_POOL_PARALLEL": str(parallel)} if parallel else None
            return _driver(task, extra, out, env)
        code, res, nv = run_with_retries(rep, f"control:{name}", go)
        verdict, detail = judge_control(name, spec, code, res, tol, world, tol_max)
        if nv:
            verdict = "INCONCLUSIVE"
        detail["result"] = os.path.relpath(out, task)
        results[name] = dict(detail, verdict=verdict)
        if verdict == "FAIL":
            rep.err(f"control {name}: {detail.get('why')}")
        elif verdict == "GAP":
            rep.doc["gaps"].append(f"control {name}: {detail.get('why')}")
        elif detail.get("warning"):
            rep.warn(f"control {name}: {detail['warning']}")
    rep.doc["controls"].update(results)
    return results


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--static-only", action="store_true")
    a = ap.parse_args(argv)
    task = os.path.realpath(a.task)
    out = a.out or os.path.join(task, "run", "verify_ut.json")
    rep = Report(task)
    t0 = time.monotonic()
    verdict = "FAIL"
    try:
        with open(os.path.join(task, "meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        static_ok = static_checks(task, rep)
        world = int(rep.doc["static"].get("world_size") or 1)
        rep.doc["world_size"] = world
        if not static_ok:
            verdict = "FAIL"
        elif a.static_only:
            verdict = "STATIC_PASS"
        else:
            work = os.path.join(task, "run", "verify_ut")
            shutil.rmtree(work, ignore_errors=True)
            os.makedirs(work, exist_ok=True)
            aa, aa_res = run_aa(task, meta, rep, work, a)
            receipt_ok = check_receipt(meta, aa_res, rep) if aa == "PASS" else False
            native_dir = os.path.join(work, "native")
            if aa == "PASS" and not os.path.isfile(os.path.join(native_dir, "native.json")):
                # The A/A was the UT's own result: its native references (same contract) serve the controls.
                ut_native = os.path.join(task, "run", "ut_native")
                try:
                    with open(os.path.join(ut_native, "native.json"), encoding="utf-8") as fh:
                        stamp = json.load(fh)
                except (OSError, ValueError):
                    stamp = {}
                if stamp.get("contract_sha256") == rep.doc.get("contract_sha256") \
                        and int(stamp.get("world") or 1) == world:
                    native_dir = ut_native
                else:
                    code, res, nv = run_with_retries(rep, "native_refs", lambda at, par: _driver(
                        task, ["--aa", "--correctness-only", "--native-dir", native_dir],
                        os.path.join(work, "native_refs.json"),
                        {"GEAK_UT_POOL_PARALLEL": str(par)} if par else None))
                    if nv:
                        aa = "INCONCLUSIVE"
                    elif code != 0:
                        rep.err(f"native reference run failed (exit {code}): {str(res.get('reason'))[:600]}")
                        aa = "FAIL"
            controls = run_controls(task, meta, rep, work, world, native_dir) if aa == "PASS" else {}
            verdicts = [c["verdict"] for c in controls.values()]
            failed = aa == "FAIL" or (aa == "PASS" and not receipt_ok) or "FAIL" in verdicts or rep.doc["errors"]
            if failed:
                verdict = "FAIL"
            elif rep.doc["gaps"]:
                verdict = "FAIL"
                rep.err("coverage gaps (ISSUE-028/012): " + "; ".join(rep.doc["gaps"]))
            elif aa == "INCONCLUSIVE" or "INCONCLUSIVE" in verdicts:
                verdict = "INCONCLUSIVE"
            else:
                verdict = "PASS"
    except Exception as exc:             # ISSUE-014: name it, still write the report
        rep.err(f"verify_ut raised {type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}")
        verdict = "FAIL"
    rep.doc["verdict"] = verdict
    rep.doc["wall_s"] = round(time.monotonic() - t0, 1)
    try:
        rep.doc.setdefault("snapshot_sha256", ut_runner.source_snapshot(task)["snapshot_sha256"])
    except Exception:
        pass
    ut_runner.write_result(out, rep.doc)
    print(f"FREEZE_VERIFY: {verdict}")
    print(json.dumps({"verdict": verdict, "out": out, "errors": rep.doc["errors"][:10],
                      "warnings": rep.doc["warnings"][:10]}, indent=1, ensure_ascii=False))
    return {"PASS": 0, "STATIC_PASS": 0, "INCONCLUSIVE": 2}.get(verdict, 1)


if __name__ == "__main__":
    sys.exit(main())
