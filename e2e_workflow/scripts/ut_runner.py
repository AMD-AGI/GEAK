#!/usr/bin/env python3
"""The ONE UT driver of a generated kernel task (`python3 unittest.py` runs this file).

A generated task (generate_ut.py) has a fixed unittest.py that only calls this driver. The driver runs
in the plain parent process -- it never imports the candidate. In order it does:

  contract      run/ut_contract.json: the frozen test files (unittest.py, cases.py, meta.json with its
                tolerance, the runtime files, baseline_overlay/, controls, reference data) still hash to
                the frozen contract_sha256; a changed test invalidates every earlier verdict (exit 3)
  snapshot      the source snapshot of kernel_source_delivery.mjs (meta/cases/unittest + kernel_src/
                + meta.kernel_extraction.source_delivery_files), recorded with the result
  legs          baseline overlay vs candidate overlay (kernel_src/ via meta.candidate_bind); the two
                legs must resolve the target to different code, the baseline outside the task
  oracle        baseline process: random-draw outputs for every random shape (same seeds both sides)
  refs          a SECOND baseline process: the full check sequence on the native code -- eager cases,
                random draws vs the oracle (= native run-to-run noise), graph replay, sequence --
                recording the native output of every comparison. A native that fails its own checks
                means cases.py or the tolerance is wrong: UT incomplete (exit 3). native_noise records the
                random draws' worst max_rel_err and noise_rel (A37, correct()'s own scale: it passes at t
                exactly when noise_rel <= t), also when the self-check stops the UT, whose reason then
                names noise_rel and 2 x noise_rel. With meta.tolerance.per_output (A42) every comparison
                is made output by output at each output's own tolerance, native_noise.per_output and the
                result's `tolerance` record every output's noise_rel and tolerance, and an output layout
                per_output does not fit stops the UT (exit 3)
  correctness   candidate process (leg_runner.py --mode correct): the same sequence, judged against
                the recorded native outputs; the worker reports the identity it actually resolved,
                which must be the candidate's (exit 1 on any failure)
  timing        world_size 1: harness_lib.measure_legs as before (fresh process per case per leg,
                interleaved B/C pairs, at most 3, early stop); a graph-deployed phase is a graph replay
                as before, an eager-deployed phase is timed on a queued GPU spin (A16) so the reading is
                GPU execution time, not the host's dispatch latency; world_size > 1: the rank pools (see
                the multi-rank section)
  inner         a graph-deployed case with call mass that reads host-dominated (primed false) on either leg
                is timed again on both legs with its inner doubled until both are device-dominated, up to
                meta.timing.graph_inner_max_factor (default 64) x the declared inner (A20); then a graph case
                with call mass whose window is under 0.1 ms is timed on both legs at 16 x its inner to measure
                the per-replay launch floor, and when that floor is over 10% of a leg's reading the case is
                raised: to the probe's inner and reading when the floor is at most 10% there, else to the
                smallest inner at which it is predicted to be at most 10%, timed once more (same cap; these
                timings run every pair, no early stop); the receipt records both
  weighting     harness_lib.serving_weighted_speedup over the observed call mass, plus the call-weighted
                speedup of every phase with call mass (timing.by_phase); without meta.workload the cases are
                unweighted and the primary metric is their geomean (A19, main's normal mode)
  output        CORRECTNESS line, one line per timed case, GEAK_WEIGHTED_SPEEDUP=<x> (GEOMEAN_SPEEDUP=<x>
                when unweighted), the GEAK_TIMING_RECEIPT: <json> line, run/ut_result.json with its
                `coverage` (A27); exit 0 pass / 1 correctness or timing-contract failure of the candidate /
                2 environment / 3 UT incomplete. Only a full run writes run/ut_result.json: a source mutation
                (--mutation LEAF) writes run/mutations/LEAF/ut_result.json, any other partial run
                run/ut_partial/<mode>/ut_result.json, and says whether the last full result is still current
                (A28)

Generated tasks are judged more strictly than hand-written ones (spec 4.2): a graph-deployed phase whose
capture fails FAILS (no "skipped"); both legs must use the same timer; under a graph deployment every
positive-mass case must read device-dominated (`primed`) on both legs, and not mostly the per-replay launch
floor, once inner has been raised as far as allowed, else exit 3 naming the cases, their host_ms (or floor)
and ms; every deployment graph of a timed case with
call mass is replayed by a replay group, a stateful graph task carries state across consecutive replays,
a stateful task's sequence covers its declared stateful tiers (A18); a stateful task must have
sequence_cases. An
eager-deployed single-rank case timed without the queued spin is incomplete (exit 3). Receipt flags,
never failures (C2/A16): an eager-deployed case whose dispatch is slower than its GPU time (scored on GPU
time, with its dispatch-inclusive latency), and a case whose spin ran out even doubled.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
if __name__ == "__main__":
    # Run as a script from a task dir, which holds a file named unittest.py: keep it from shadowing the
    # stdlib module. (Imported as a module by generate_ut.py / verify_ut.py, sys.path is left alone.)
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]

VERSION = "ut_runner.py v1"
EXIT_PASS, EXIT_FAIL, EXIT_ENV, EXIT_INCOMPLETE = 0, 1, 2, 3
GENERATED_MARKER = "# GEAK-GENERATED unittest.py"
RUNTIME_FILES = ("harness_lib.py", "leg_runner.py", "overlay_setup.py", "ut_runner.py")
CONTRACT_NAME = "geak_ut_contract/v1"
SOURCE_EXT = re.compile(r"\.(py|c|cc|cpp|cxx|h|hpp|cu|cuh|hip|s|mlir)$", re.IGNORECASE)
RECEIPT_VERSION = "geak_timing_receipt/v2"
DEFAULT_TIMEOUTS_SINGLE = {"setup_s": 900, "command_s": 3600, "total_s": 0}
DEFAULT_TIMEOUTS_MULTI = {"setup_s": 900, "command_s": 1800, "total_s": 3600}

# Failures that say nothing about the task: the run died of its environment (ISSUE-029, 024 and the
# GPU-IPC / lock rule). Matched on the failing process's own log.
RESOURCE_RE = re.compile(r"out of memory|outofmemoryerror|hiperroroutofmemory|hsa_status_error_out_of_resources"
                         r"|ncclsystemerror|rccl.*(unhandled system error|out of memory)"
                         r"|cannot allocate memory|resource exhausted", re.IGNORECASE)
IPC_FAULT_RE = re.compile(r"hipIpcOpenMemHandle\b.*invalid device pointer", re.IGNORECASE)
LOCK_FAULT_RE = re.compile(r"ERROR: (Failed to acquire GPU \S+ lock after|no free\+idle GPU in pool"
                           r"|GPU \S+ still shows foreign work after)|still locked after|shows foreign work")
DEVICE_FAULT_RE = re.compile(r"hipErrorNoDevice|no HIP-capable device|hipErrorInvalidDevice"
                             r"|CUDA error: no kernel image|HSA_STATUS_ERROR_OUT_OF_RESOURCES", re.IGNORECASE)


class UTStop(Exception):
    """Ends the run with an exit code and a reason (the reason goes to stdout and the result)."""

    def __init__(self, code, reason, gate=None):
        super().__init__(reason)
        self.code, self.reason, self.gate = code, reason, gate


def env_fault(text):
    """"resources" / "ipc" / "lock" / "device" when a failure is the environment's, else None."""
    if not text:
        return None
    if LOCK_FAULT_RE.search(text):
        return "lock"
    if IPC_FAULT_RE.search(text):
        return "ipc"
    if RESOURCE_RE.search(text):
        return "resources"
    if DEVICE_FAULT_RE.search(text):
        return "device"
    return None


# --------------------------------------------------------------------------- files and hashes
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _js_json(obj):
    """JSON.stringify of plain data (what the delivery gates hash)."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _inside(task, path):
    rel = os.path.relpath(path, task)
    return rel != "." and not rel.startswith(".." + os.sep) and rel != ".." and not os.path.isabs(rel)


def file_ref(task, name):
    """{path, sha256} of a task-local file, as kernel_source_delivery.mjs fileRef computes it."""
    task = os.path.realpath(task)
    real = os.path.realpath(os.path.join(task, name))
    if not _inside(task, real):
        raise ValueError(f"file outside task: {name}")
    return {"path": os.path.relpath(real, task), "sha256": sha256_file(real)}


def source_snapshot(task):
    """Python twin of kernel_source_delivery.mjs `sourceSnapshot` (same files, same digest)."""
    task = os.path.realpath(task)
    names = ["meta.json", "cases.py", "unittest.py"]

    def walk(rel):
        for e in os.scandir(os.path.join(task, rel)):
            name = os.path.join(rel, e.name)
            if e.is_dir(follow_symlinks=False) and e.name != "__pycache__":
                walk(name)
            elif SOURCE_EXT.search(e.name):
                names.append(name)
    if os.path.isdir(os.path.join(task, "kernel_src")):
        walk("kernel_src")
    with open(os.path.join(task, "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    names += list((meta.get("kernel_extraction") or {}).get("source_delivery_files") or [])
    files = [file_ref(task, n) for n in sorted(set(names))]
    if not any(f["path"].startswith("kernel_src" + os.sep) for f in files):
        raise ValueError("kernel_src contains no source files")
    return {"files": files, "snapshot_sha256": hashlib.sha256(_js_json(files).encode()).hexdigest()}


def _tree(task, rel):
    out = []
    base = os.path.join(task, rel)
    for root, dirs, files in os.walk(base):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for f in sorted(files):
            if f.endswith((".pyc", ".pyo")):
                continue
            out.append(os.path.relpath(os.path.join(root, f), task))
    return out


def contract_file_list(task, meta):
    """The frozen test files of a generated task (spec 3.1: unittest.py, cases.py, meta.json with the
    tolerance, reference data, runtime files, control definitions). Never kernel_src/, run/, the
    candidate overlay, cached reference outputs or build caches."""
    names = ["unittest.py", "cases.py", "meta.json", *RUNTIME_FILES]
    if os.path.isdir(os.path.join(task, "baseline_overlay")):
        names += _tree(task, "baseline_overlay")
    for c in ((meta.get("controls") or {}).get("task") or []):
        if isinstance(c, dict) and c.get("file"):
            names.append(c["file"])
    for rel in meta.get("ut_contract_files") or []:
        p = os.path.join(task, rel)
        names += _tree(task, rel) if os.path.isdir(p) else [rel]
    names = [os.path.normpath(n) for n in names]      # judged as the path it is: ./kernel_src/x is kernel_src/x
    bad = [n for n in names if n.split(os.sep)[0] in ("kernel_src", "run", "_cand_overlay")]
    if bad:
        raise ValueError(f"contract files may not live in kernel_src/, run/ or _cand_overlay/: {bad}")
    return sorted(set(names))


def compute_contract(task, names):
    files = [file_ref(task, n) for n in names]
    files.sort(key=lambda f: f["path"])
    return {"files": files, "contract_sha256": hashlib.sha256(_js_json(files).encode()).hexdigest()}


def check_contract(task):
    """(contract_sha256, problems). Problems name every file that no longer matches."""
    path = os.path.join(task, "run", "ut_contract.json")
    if not os.path.isfile(path):
        return None, ["run/ut_contract.json missing: freeze the task with generate_ut.py freeze"]
    with open(path, encoding="utf-8") as fh:
        frozen = json.load(fh)
    if frozen.get("contract") != CONTRACT_NAME:
        return None, [f"run/ut_contract.json is not {CONTRACT_NAME}"]
    problems = []
    names = [f["path"] for f in frozen.get("files") or []]
    for need in ("unittest.py", "cases.py", "meta.json", *RUNTIME_FILES):
        if need not in names:
            problems.append(f"{need} is not in the frozen contract")
    for f in frozen.get("files") or []:
        p = os.path.join(task, f["path"])
        if not os.path.isfile(p):
            problems.append(f"{f['path']}: missing")
        elif sha256_file(p) != f["sha256"]:
            problems.append(f"{f['path']}: changed after freeze")
    if not problems:
        got = hashlib.sha256(_js_json(frozen["files"]).encode()).hexdigest()
        if got != frozen.get("contract_sha256"):
            problems.append("contract_sha256 does not match its file list")
    return frozen.get("contract_sha256"), problems


# --------------------------------------------------------------------------- task facts
def world_size(meta):
    mr = ((meta.get("capabilities") or {}).get("multi_rank") or {})
    try:
        return max(1, int(mr.get("world_size") or 1))
    except (TypeError, ValueError):
        raise UTStop(EXIT_INCOMPLETE, f"capabilities.multi_rank.world_size is not an integer: {mr.get('world_size')!r}")


def timeouts(meta, world=None):
    """The run's time limits: meta.timeouts over the defaults. A setup_s / command_s that is not above 0 keeps its
    default (generate_ut.py check refuses one: a rank pool cannot start or wait with it); total_s 0 is no limit."""
    world = world_size(meta) if world is None else world
    base = dict(DEFAULT_TIMEOUTS_MULTI if world > 1 else DEFAULT_TIMEOUTS_SINGLE)
    for k, v in (meta.get("timeouts") or {}).items():
        if k in base and v is not None and (k == "total_s" or float(v) > 0):
            base[k] = float(v)
    return base


def graph_phases(h, meta):
    """{phase label: deploys under a CUDA graph}; label "" when there is one global regime."""
    pr = meta.get("phase_regimes")
    if isinstance(pr, dict) and pr:
        return {p: bool(h.deployment_graph_mode(r)) for p, r in pr.items() if isinstance(r, dict)}
    return {"": bool(h.deployment_graph_mode(meta.get("regime", {})))}


def positive_mass_sigs(h, meta):
    """Signatures whose observed call mass is positive (None when the task has no observed model)."""
    try:
        observed = h._observed_workload_cases(meta)
    except Exception:
        return None
    if observed is None:
        return None
    served = h.served_regimes(meta)
    return {s for s, c in observed.items()
            if c["effective_calls"] > 0 and (not served or c["regime"] in served)}


# --------------------------------------------------------------------------- legs
class Legs:
    """The processes of one UT run. Every subprocess runs leg_runner.py under one overlay."""

    def __init__(self, task, meta, h, *, command_s, deadline):
        self.task, self.meta, self.h = task, meta, h
        self.command_s, self.deadline = command_s, deadline

    def budget(self):
        left = self.command_s or 3600
        if self.deadline:
            rest = self.deadline - time.monotonic()
            if rest <= 5:
                raise UTStop(EXIT_ENV, "the whole-UT time limit (meta.timeouts.total_s) ran out", "timeout")
            left = min(left, rest)
        return left

    def run(self, overlay, mode, who, **kw):
        try:
            return self.h._run_leg(self.task, overlay, mode, timeout=self.budget(), **kw)
        except subprocess.TimeoutExpired:
            raise UTStop(EXIT_ENV, f"{who} leg ({mode}) exceeded its time limit "
                                   f"(meta.timeouts.command_s / total_s)", "timeout")
        except RuntimeError as exc:
            text = str(exc)
            fault = env_fault(text)
            if fault:
                raise UTStop(EXIT_ENV, f"{who} leg ({mode}) died of its environment ({fault}): {text[-600:]}",
                             fault)
            raise


def _worker_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def run_correctness(legs, base, cand, *, native_dir, draws, control=None, control_spec=None,
                    reuse_native=False, contract_sha=None, who="candidate", seen=None):
    """oracle + refs on the baseline, then the judged leg (`cand`, maybe under a control).
    Returns (native_result, judged_result); `seen["native"]` gets the native result as soon as it exists
    (also when the native self-check then stops the run)."""
    os.makedirs(native_dir, exist_ok=True)
    prefix = os.path.join(native_dir, "native")
    stamp = os.path.join(native_dir, "native.json")
    native = None
    if reuse_native and os.path.isfile(stamp):
        rec = _worker_json(stamp)
        if rec.get("contract_sha256") == contract_sha and os.path.isfile(prefix + ".pt") \
                and os.path.isfile(prefix + ".refs.pt"):
            native = rec.get("refs")
    if native is None:
        legs.run(base, "oracle", "baseline", out=prefix + ".pt", seed=0, draws=draws)
        refs_out = os.path.join(native_dir, "refs_result.json")
        try:
            legs.run(base, "refs", "baseline", extra=["--blob", prefix, "--out", refs_out, "--draws", str(draws)])
        except RuntimeError as exc:
            raise UTStop(EXIT_INCOMPLETE, f"the NATIVE refs leg crashed running this task's cases: {str(exc)[-1500:]}",
                         "native_self_check")
        native = _worker_json(refs_out)
        with open(stamp, "w", encoding="utf-8") as fh:
            json.dump({"contract_sha256": contract_sha, "refs": native}, fh, default=str)
    if seen is not None:
        seen["native"] = native
    if native.get("incomplete"):
        raise UTStop(EXIT_INCOMPLETE, "; ".join(native["incomplete"]), "native_self_check")
    if not native.get("ok"):
        raise UTStop(EXIT_INCOMPLETE,
                     "the NATIVE code fails this task's own checks (baseline refs leg): cases.py or the "
                     "tolerance is wrong, not the candidate" + noise_advice(native.get("native_noise")) + ": "
                     + json.dumps({"errors": native.get("errors"), "report": native.get("report")},
                                  default=str)[:3000], "native_self_check")
    out = os.path.join(native_dir, f"judged_{control or who}.json")
    extra = ["--blob", prefix, "--out", out, "--draws", str(draws)]
    if control:
        extra += ["--control", control]
        if control_spec:
            extra += ["--control-spec", json.dumps(control_spec)]
    try:
        legs.run(cand, "correct", who, extra=extra)
        judged = _worker_json(out)
    except RuntimeError as exc:
        judged = {"ok": False, "incomplete": [], "crashed": True,
                  "errors": [f"{who} correctness process crashed: {str(exc)[-2000:]}"], "report": {}}
    return native, judged


def correctness_by_rank(judged):
    """Per-rank rejection summary of a judged correctness result, in the shape of the A14 record
    (`ranks: [{rank, launch_succeeded, numerical_mismatches}]`): a source mutation's record is written
    from this. A case counts as a numerical mismatch only when it compared an output and failed (a call
    that raised compared nothing)."""
    rows = []
    ranks = judged.get("ranks") if isinstance(judged.get("ranks"), list) else [dict(judged, rank=0)]
    for i, r in enumerate(ranks):
        failing, raised = [], 0

        def walk(x):
            nonlocal raised
            if isinstance(x, dict):
                for k, v in x.items():
                    if k in ("eager", "random", "graph_replay", "sequence") and isinstance(v, list):
                        for c in v:
                            if not isinstance(c, dict) or c.get("correct") is not False:
                                continue
                            if "raised" in str(c.get("note") or ""):
                                raised += 1
                            else:
                                failing.append(c.get("case"))
                    else:
                        walk(v)
        walk(r.get("report") or {})
        rows.append({"rank": r.get("rank", i), "launch_succeeded": not (r.get("errors") or r.get("crashed")),
                     "numerical_mismatches": len(failing), "raised_cases": raised,
                     "failing_cases": failing[:50]})
    return rows


# --------------------------------------------------------------------------- receipt
def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _phase_graph(phases, row):
    """Does this row's phase deploy under a CUDA graph? (`phases` = graph_phases(h, meta))"""
    phase = row.get("regime", "") if len(phases) > 1 or "" not in phases else ""
    return phases.get(phase, phases.get("", False))


def build_receipt(per_case, world, alignment=None, phases=None):
    """The timing receipt (superset of oracle_freezer's GEAK_TIMING_RECEIPT). `phases` (graph_phases)
    adds each case's `deployed_graph`; a spun eager timing (A16) adds each leg's `spin` and `latency_ms`."""
    cases, primed_all, timers, modes = {}, [], set(), set()
    for row in per_case:
        legs = row.get("legs") or {}
        entry = {}
        for leg in ("baseline", "current"):
            rec = legs.get(leg) or {}
            item = {"ms": row.get("baseline_ms" if leg == "baseline" else "optimized_ms"),
                    "timer": rec.get("timer")}
            if "primed" in rec:
                item["primed"] = rec["primed"]
                item["host_ms"] = rec.get("host_ms")
            primed_all.append(rec.get("primed") if "primed" in rec else None)
            if rec.get("rank_ms") is not None:
                item["rank_ms"] = rec["rank_ms"]
            if isinstance(rec.get("spin"), dict):
                item["spin"] = rec["spin"]
                item["latency_ms"] = rec.get("latency_ms")
            timers.add(rec.get("timer"))
            modes.add(((rec.get("cache_condition") or {}).get("mode")))
            entry[leg] = item
        entry["cache_mode"] = ((legs.get("baseline") or {}).get("cache_condition") or {}).get("mode")
        if phases is not None:
            entry["deployed_graph"] = bool(_phase_graph(phases, row))
        if isinstance(row.get("inner_raised"), dict):
            entry["inner"] = row["inner_raised"]          # A20: declared, final, tried, reason
        if isinstance(row.get("inner_floor"), dict):
            entry["inner_floor"] = row["inner_floor"]     # A20: the floor was measured, the inner kept
        cases[row["sig"]] = entry
    return {
        "version": RECEIPT_VERSION,
        "all_primed": bool(primed_all) and all(p is True for p in primed_all),
        "timer_unprimed": any(p is None for p in primed_all),
        "cache_mode": (modes.pop() if len(modes) == 1 else "mixed") or "unknown_write_evict",
        "world_size": world,
        "timer": (timers.pop() if len(timers) == 1 else "mixed"),
        "alignment": alignment,
        "cases": cases,
    }


def receipt_problems(h, meta, per_case, receipt, strict=True):
    """Generated-task timing contract (spec 4.2.2-3, C2, A16). Returns (fail, incomplete, inconclusive)."""
    fail, incomplete, inconclusive = [], [], []
    phases = graph_phases(h, meta)
    positive = positive_mass_sigs(h, meta)
    single = int((receipt or {}).get("world_size") or 1) == 1
    unprimed = []
    for row in per_case:
        sig = row["sig"]
        legs = row.get("legs") or {}
        b, c = legs.get("baseline") or {}, legs.get("current") or {}
        if b.get("timer") != c.get("timer"):
            fail.append(f"{sig}: the legs used different timers (baseline {b.get('timer')}, candidate "
                        f"{c.get('timer')}) -- a candidate that cannot be timed like the baseline")
        graph = _phase_graph(phases, row)
        if not graph and single:
            # A16: an eager-deployed phase of a single-rank task is timed on a queued GPU spin; an eager GPU
            # reading without it starts on a drained queue and can be the host's dispatch latency.
            for leg, rec in (("baseline", b), ("candidate", c)):
                if rec.get("timer") == "cuda_event" and not isinstance(rec.get("spin"), dict):
                    incomplete.append(f"{sig}: the phase is not deployed under a CUDA graph, but the {leg} leg was "
                                      "timed without the queued GPU spin (A16): re-vendor the runtime files "
                                      "(generate_ut.py all)")
        if graph and b.get("timer") != "cuda_event_graph":
            if b.get("timer") in ("cuda_event", "mixed"):
                incomplete.append(f"{sig}: the phase deploys under a CUDA graph but the baseline leg was not "
                                  f"timed as a graph replay (timer {b.get('timer')}): its capture failed")
        weight = positive is None or sig in positive
        if graph and weight:
            for leg, rec in (("baseline", b), ("candidate", c)):
                if "primed" not in rec:
                    inconclusive.append(f"{sig}: {leg} leg could not tell whether its reading is device time "
                                        f"(no CUDA timeline)")
                elif rec.get("primed") is False:
                    unprimed.append((sig, leg, rec.get("host_ms"),
                                     row.get("baseline_ms" if leg == "baseline" else "optimized_ms")))
    if unprimed:
        raised = {row["sig"]: row.get("inner_raised") for row in per_case if isinstance(row.get("inner_raised"), dict)}

        def at(sig):
            r = raised.get(sig)
            return (f", inner {r.get('final')} = the cap ({r.get('declared')} x graph_inner_max_factor)"
                    if r else "")
        incomplete.append(
            "C2/A20: under the graph deployment these cases with call mass still read HOST-dominated with the "
            "largest inner allowed, so the reading is dispatch latency, not kernel time; raise "
            "meta.timing.graph_inner_max_factor or the case's `inner` in cases.timing_cases (one replay must run "
            "longer on the GPU than it takes to issue): "
            + "; ".join(f"{s} ({leg}: host_ms={hm}, ms={ms}{at(s)})" for s, leg, hm, ms in unprimed))
    floored = [row for row in per_case if isinstance(row.get("inner_raised"), dict)
               and ((row["inner_raised"].get("floor") or {}).get("dominated"))]
    if floored:
        incomplete.append(
            "C2/A20: under the graph deployment these cases with call mass still read mostly the per-replay graph "
            "launch floor with the largest inner allowed, so the reading is launch latency, not kernel time; "
            "raise meta.timing.graph_inner_max_factor or the case's `inner` in cases.timing_cases: "
            + "; ".join(f"{r['sig']} (inner {r['inner_raised'].get('final')}, floor "
                        + ", ".join(f"{leg} {us} us" for leg, us in (r['inner_raised']['floor'].get('floor_us')
                                                                    or {}).items())
                        + f" per replay, ms={r.get('baseline_ms')}/{r.get('optimized_ms')})" for r in floored))
    return fail, incomplete, inconclusive


# --------------------------------------------------------------------------- automatic inner (A20)
DEFAULT_GRAPH_INNER_MAX_FACTOR = 64

# The per-replay floor of a graph timing. Every timed sample starts on a drained queue (cache flush, sync), so
# the window of one replay also holds the latency of launching the graph -- from the host's submission to the
# GPU's first launch, ~15-25 us on a loaded ROCm host. `primed` cannot see it: host_ms is the cost of ISSUING
# replays back to back (a few us). For a small kernel that latency is most of the reading and its jitter is
# the A/A noise (CI round 2, gptoss add_rmsnorm: a ~2.9 us kernel read 16-24 us at inner=1 on every M, A/A
# band 10-12%). A per-call reading at inner i is K + E/i (E: the per-replay floor, K: the per-launch time in
# a packed graph, as in the deployment's graph), so readings at two inners give E and K.
FLOOR_SCREEN_MS = 0.1       # probe a case whose graph window (inner x per-call ms) is under this on either leg
FLOOR_PROBE_FACTOR = 16     # the probe times both legs at inner x 16 (at most the cap)
FLOOR_DOMINATED = 0.5       # still over half the reading at the final inner: not kernel time (exit 3)
FLOOR_TARGET = 0.1          # a floor over 10% of a reading raises the case to an inner where it is at most 10%
LEG_MS = (("baseline", "baseline_ms"), ("current", "optimized_ms"))


def graph_inner_max_factor(meta):
    """meta.timing.graph_inner_max_factor (a positive integer; default 64): how far the UT may raise a
    graph-deployed case's `inner` over the declared value."""
    v = (meta.get("timing") or {}).get("graph_inner_max_factor") if isinstance(meta.get("timing"), dict) else None
    if isinstance(v, int) and not isinstance(v, bool) and v >= 1:
        return v
    return DEFAULT_GRAPH_INNER_MAX_FACTOR


def _unprimed_legs(row):
    legs = row.get("legs") or {}
    return [leg for leg in ("baseline", "current") if (legs.get(leg) or {}).get("primed") is False]


def inner_targets(h, meta, per_case):
    """Signatures of the graph-deployed cases with call mass (every timed case when the task has no observed
    workload) whose reading is host-dominated on either leg (`primed` false)."""
    phases = graph_phases(h, meta)
    positive = positive_mass_sigs(h, meta)
    return [r["sig"] for r in per_case if _phase_graph(phases, r)
            and (positive is None or r["sig"] in positive) and _unprimed_legs(r)]


def floor_estimate(n, ms_n, p, ms_p):
    """(E, K) in ms from one leg's per-call readings at inners n < p: a reading at inner i is K + E/i, E the
    per-replay floor (never below 0) and K the per-launch time in a packed graph (<= 0: the reading is all
    floor). None when the two readings cannot say."""
    if not (_finite(ms_n) and _finite(ms_p) and ms_n > 0 and ms_p > 0) or p <= n:
        return None
    e = max(0.0, (ms_n - ms_p) * n * p / (p - n))
    return e, ms_p - e / p


def floor_share(e, inner, ms):
    """The floor's share of a per-call reading `ms` taken at `inner`: (E / inner) / ms."""
    return (e / inner) / ms if (_finite(ms) and ms > 0) else None


def floor_inner(e, k, cap):
    """The smallest integer inner whose predicted floor share E / (inner x K + E) is at most FLOOR_TARGET, i.e.
    ceil(E (1 - T) / (T K)) with T = FLOOR_TARGET (ceil(9 E / K) at 10%), at most `cap`; `cap` when K <= 0 (the
    reading is all floor). A tiny margin is taken off before rounding up, so a float error cannot add one to an
    inner that is exactly enough."""
    if not k > 0:
        return cap
    need = e * (1.0 - FLOOR_TARGET) / (FLOOR_TARGET * k)
    return max(1, min(cap, math.ceil(need - 1e-9 * max(1.0, need))))


def _inner_cap(row, factor):
    prev = row.get("inner_raised") if isinstance(row.get("inner_raised"), dict) else {}
    declared = int(prev.get("declared") or row.get("inner") or 1)
    return declared, declared * factor


def floor_targets(h, meta, per_case, factor):
    """Signatures of the graph-deployed cases with call mass (every timed case without an observed workload),
    timed as graph replays, whose window (inner x per-call ms) is shorter than FLOOR_SCREEN_MS on either leg and
    whose inner can still be doubled: the per-replay floor may be most of their reading."""
    phases = graph_phases(h, meta)
    positive = positive_mass_sigs(h, meta)
    out = []
    for r in per_case:
        if not _phase_graph(phases, r) or not (positive is None or r["sig"] in positive):
            continue
        if ((r.get("legs") or {}).get("baseline") or {}).get("timer") != "cuda_event_graph":
            continue                       # a failed capture is named by receipt_problems, not probed
        n = int(r.get("inner") or 1)
        windows = [r[k] * n for _, k in LEG_MS if _finite(r.get(k))]
        if windows and min(windows) < FLOOR_SCREEN_MS and n * 2 <= _inner_cap(r, factor)[1]:
            out.append(r["sig"])
    return out


def _us(ms):
    return round(ms * 1000.0, 3) if _finite(ms) else None


def _pct(shares):
    return ", ".join(f"{leg} {round(100 * s)}%" for leg, s in shares.items() if s is not None)


def raise_for_floor(row, measure, factor, log=print):
    """A20 for a reading that holds the per-replay floor (see FLOOR_*): time BOTH legs at the probe inner
    p = inner x FLOOR_PROBE_FACTOR (at most the cap) and estimate E and K per leg. When the floor is over
    FLOOR_TARGET of either leg's reading at `inner`, the case is raised: if the floor predicted at p is at most
    FLOOR_TARGET on both legs, it takes p and the probe's reading, with no further timing; otherwise its inner is
    the smallest integer whose predicted floor share is at most FLOOR_TARGET on both legs (floor_inner, at most
    the cap), both legs are timed once more with it and that reading is kept. A floor of at most FLOOR_TARGET
    keeps the case as it was and records the check in `inner_floor`. The probe can be above the smallest inner
    that suffices and is kept anyway: as long as 9 E/K <= p (a floor share up to 16/25 = 64% at `inner` when
    p = 16 x inner) every run takes p, so the final inner does not move with the host's load; beyond that the
    inner needed moves by a few launches with the floor, not by a factor of two. A raised case's `inner_raised`
    gets the floor record; `floor.dominated` (the floor still over half a reading at the cap) makes the UT
    incomplete (receipt_problems). A probe without a reading keeps the case as it was and records `inner_floor`
    {probe_inner, probe_failed: true}; a raise whose timing fails keeps the probe's reading and records
    `target_failed`."""
    sig = row["sig"]
    prev = row.get("inner_raised") if isinstance(row.get("inner_raised"), dict) else None
    declared, cap = _inner_cap(row, factor)
    n = int(row.get("inner") or 1)
    p = n
    while p * 2 <= min(cap, n * FLOOR_PROBE_FACTOR):
        p *= 2
    if p == n:
        return row
    window = min(row[k] * n for _, k in LEG_MS if _finite(row.get(k)))
    log(f"[ut] {sig}: a {_us(window)} us graph window at inner={n}; timing both legs with inner={p} to measure "
        f"the per-replay launch floor")
    got = measure(sig, p)
    if not got:
        # review-20 3.2.4: the receipt says the floor was not measured; the case keeps its inner and reading
        log(f"[ut] {sig}: the floor probe at inner={p} returned no reading; inner={n} kept")
        row["inner_floor"] = {"probe_inner": p, "probe_failed": True, "kept": n}
        return row
    probe = got[0]
    est = {leg: floor_estimate(n, row.get(k), p, probe.get(k)) for leg, k in LEG_MS}
    est = {leg: v for leg, v in est.items() if v is not None}
    share_n = {leg: floor_share(est[leg][0], n, row.get(k)) for leg, k in LEG_MS if leg in est}
    record = {"probe_inner": p, "floor_us": {leg: _us(v[0]) for leg, v in est.items()},
              "kernel_us": {leg: _us(v[1]) for leg, v in est.items()},
              "share": {str(n): {leg: round(s, 4) for leg, s in share_n.items() if s is not None}}}
    if max([s for s in share_n.values() if s is not None], default=0.0) <= FLOOR_TARGET:
        row["inner_floor"] = dict(record, dominated=False, kept=n)
        return row
    target = max(floor_inner(e, k, cap) for e, k in est.values())
    final, tried = probe, [p]
    if target > p:
        log(f"[ut] {sig}: the per-replay floor is "
            + ", ".join(f"{leg} {_us(est[leg][0])} us of {_us(row.get(k))} us" for leg, k in LEG_MS if leg in est)
            + f" at inner={n}; timing both legs again with inner={target}")
        more = measure(sig, target)
        if more:
            final = more[0]
            tried.append(target)
        else:
            record["target_failed"] = target          # the probe's reading (inner p) is kept instead
            target = p
    else:
        target = p
    share_f = {leg: floor_share(est[leg][0], target, final.get(k)) for leg, k in LEG_MS if leg in est}
    dominated = any(s is not None and s > FLOOR_DOMINATED for s in share_f.values())
    record["share"][str(target)] = {leg: round(s, 4) for leg, s in share_f.items() if s is not None}
    record["dominated"] = dominated
    at_cap = f" (the cap {cap})" if target == cap else ""
    reason = (f"the per-replay graph launch floor was {_pct(share_n)} of the reading at inner={n} (floor "
              + ", ".join(f"{leg} {_us(v[0])} us" for leg, v in est.items()) + f"; inner={n} vs {p}): "
              + (f"at inner={target}{at_cap} it is still {_pct(share_f)} of the reading" if dominated else
                 f"raised to inner={target}{at_cap}, floor {_pct(share_f)}"))
    final["inner"] = target
    final["inner_raised"] = {
        "declared": declared, "final": target, "tried": list((prev or {}).get("tried") or [n]) + tried,
        "cap": cap, "primed": not _unprimed_legs(final), "floor": record,
        "reason": ((prev["reason"] + "; then ") if prev and prev.get("reason") else "") + reason}
    return final


def raise_inner(h, meta, per_case, measure, log=print, measure_all=None):
    """A20: for every host-dominated graph case, double `inner` and time BOTH legs again with it (`measure(sig,
    inner)` -> [row]) until both legs are device-dominated or the cap (declared inner x
    graph_inner_max_factor) is reached; the smallest inner that primes both legs is kept. Then every graph
    case with call mass whose window is short is checked for the per-replay floor (raise_for_floor), timed
    with `measure_all` (as `measure`, but every pair is timed: an early stop after one outlying pair would set
    the floor estimate, and with it the inner, from one fresh process; default `measure`). Each reading stays
    per call. A raised case's row gets `inner_raised` {declared, final, tried, cap, primed, reason[, floor]}.
    Returns the rows in the original order."""
    factor = graph_inner_max_factor(meta)
    rows = {r["sig"]: r for r in per_case}
    for sig in inner_targets(h, meta, per_case):
        row = rows[sig]
        declared = int(row.get("inner") or 1)
        cap = declared * factor
        legs = _unprimed_legs(row)
        first = "; ".join(f"{leg}: host_ms={((row.get('legs') or {}).get(leg) or {}).get('host_ms')}, "
                          f"ms={row.get('baseline_ms' if leg == 'baseline' else 'optimized_ms')}" for leg in legs)
        inner, tried = declared, [declared]
        while _unprimed_legs(row) and inner * 2 <= cap:
            log(f"[ut] {sig}: host-dominated under the graph deployment ({', '.join(_unprimed_legs(row))}); "
                f"timing both legs again with inner={inner * 2}")
            new = measure(sig, inner * 2)
            if not new:
                break                      # the reading kept is the last one taken, at `inner`
            inner *= 2
            row = new[0]
            tried.append(inner)
        row["inner"] = inner
        row["inner_raised"] = {
            "declared": declared, "final": inner, "tried": tried, "cap": cap,
            "primed": not _unprimed_legs(row),
            "reason": (f"host-dominated at the declared inner={declared} ({first}): raised until both legs read "
                       "device-dominated" if not _unprimed_legs(row) else
                       f"host-dominated at the declared inner={declared} ({first}) and still at the cap {cap}")}
        rows[sig] = row
    for sig in floor_targets(h, meta, [rows[r["sig"]] for r in per_case], factor):
        rows[sig] = raise_for_floor(rows[sig], measure_all or measure, factor, log=log)
    return [rows[r["sig"]] for r in per_case]


SCORED_ON_GPU_TIME = ("scored on GPU time: this presumes the host runs ahead of the GPU in deployment, as a "
                      "server with a full queue does (C2/A16)")


def receipt_flags(h, meta, per_case):
    """Receipt notes that never fail a UT (C2 as amended by A16), for phases not deployed under a CUDA graph:
    a leg whose dispatch is slower than its GPU time (primed false), with its dispatch-inclusive latency;
    and a leg whose spin ran out even doubled (its reading includes dispatch)."""
    flags = []
    phases = graph_phases(h, meta)
    for row in per_case:
        if _phase_graph(phases, row):
            continue
        legs = row.get("legs") or {}
        for leg in ("baseline", "current"):
            rec = legs.get(leg) or {}
            ms = row.get("baseline_ms" if leg == "baseline" else "optimized_ms")
            if rec.get("primed") is False:
                flags.append({"sig": row["sig"], "leg": leg, "kind": "dispatch_slower_than_kernel",
                              "host_ms": rec.get("host_ms"), "ms": ms, "latency_ms": rec.get("latency_ms"),
                              "note": SCORED_ON_GPU_TIME})
            sp = rec.get("spin") if isinstance(rec.get("spin"), dict) else {}
            if sp.get("includes_dispatch"):
                flags.append({"sig": row["sig"], "leg": leg, "kind": "reading_includes_dispatch",
                              "late": sp.get("late"), "samples": sp.get("samples"), "spin_us": sp.get("us"),
                              "ms": ms, "note": "the GPU spin ran out before the host had issued the window, "
                                                "also when doubled: this case's reading includes dispatch"})
    return flags


def phase_speedups(rows):
    """{phase: {speedup, calls}} over serving_weighted_speedup's included rows: the call-weighted speedup
    of every phase with call mass (the A/A is judged per phase as well as overall, A16 C')."""
    acc = {}
    for r in rows or []:
        if not r.get("included"):
            continue
        b, o, n = r.get("baseline_ms"), r.get("optimized_ms"), r.get("calls")
        if not (_finite(b) and _finite(o) and o > 0 and _finite(n) and n > 0):
            continue
        a = acc.setdefault(r.get("regime") or "", [0.0, 0.0, 0.0])
        a[0] += b * n
        a[1] += o * n
        a[2] += n
    return {ph: {"speedup": a[0] / a[1], "calls": a[2]} for ph, a in acc.items() if a[1] > 0}


# --------------------------------------------------------------------------- weighting (A19)
def weighting_mode(h, meta):
    """"unweighted" without meta.workload (main's normal mode: golden / captured cases, unweighted geomean
    as the primary metric), "observed" for an observed workload model, else "analytic"."""
    if not meta.get("workload"):
        return "unweighted"
    try:
        return "observed" if h._observed_workload_cases(meta) is not None else "analytic"
    except Exception:
        return "observed"


def phase_geomeans(rows):
    """{phase: {speedup, calls, basis}}: the unweighted geomean of each phase's included cases (A19)."""
    acc = {}
    for r in rows or []:
        sp = r.get("speedup")
        if r.get("included") and _finite(sp) and sp > 0:
            acc.setdefault(r.get("regime") or "", []).append(math.log(sp))
    return {ph: {"speedup": math.exp(sum(v) / len(v)), "calls": len(v), "basis": "unweighted geomean"}
            for ph, v in acc.items() if v}


# --------------------------------------------------------------------------- coverage (ISSUE-006)
def manifest_problems(meta, timed_sigs, correctness_sigs, positive):
    """The case manifest's coverage buckets must each have a timing and a correctness case, and the cases
    the UT actually ran must be the declared ones. Empty list = covered."""
    man = meta.get("case_manifest")
    problems = []
    if not isinstance(man, dict) or not isinstance(man.get("cases"), list):
        return ["meta.case_manifest {dims, cases[]} is missing"]
    dims = list(man.get("dims") or ["regime"])
    declared_t = {c.get("sig") for c in man["cases"] if "timing" in (c.get("roles") or [])}
    declared_c = {c.get("sig") for c in man["cases"] if "correctness" in (c.get("roles") or [])}
    if timed_sigs is not None and set(timed_sigs) != declared_t:
        problems.append(f"timed cases {sorted(set(timed_sigs))} differ from the manifest's timing cases "
                        f"{sorted(declared_t)}")
    if correctness_sigs is not None:
        missing = declared_c - set(correctness_sigs)
        if missing:
            problems.append(f"manifest correctness cases never checked: {sorted(missing)}")
    if positive is not None:
        absent = set(positive) - declared_t
        if absent:
            problems.append(f"positive-mass observed cases without a timing case: {sorted(absent)}")
    return problems


def _sig_of(label):
    m = re.match(r"^(?:random\[\d+\]|seq\[\d+\]):(.*)$", label or "")
    return m.group(1) if m else label


def checked_sigs(report):
    out = set()

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ("eager", "random", "graph_replay") and isinstance(v, list):
                    for c in v:
                        if isinstance(c, dict) and c.get("case") not in (None, "", "output_independence",
                                                                          "graph_replay"):
                            out.add(_sig_of(c["case"]))
                else:
                    walk(v)
    walk(report)
    return out


def is_boundary_case(c):
    """A40: a case_manifest boundary case -- "boundary": true with the correctness role only."""
    return isinstance(c, dict) and c.get("boundary") is True and set(c.get("roles") or []) == {"correctness"}


def coverage_summary(h, meta, reports, timed_sigs, mode):
    """run/ut_result.json `coverage` (A27): the case manifest's buckets with their timing and correctness
    cases, the boundary cases (A40: sig, regime, dims and whether the native run compared them; never a bucket),
    the signatures with call mass against the timed ones and the calls declared uncovered
    (meta.workload.uncovered, A35), every graph phase's deployment graphs and the replay groups that recorded
    them, the groups carrying state and the stateful tiers the sequence covered. `reports` = the native
    reference run's report per rank (rank 0 first); None before it ran."""
    man = meta.get("case_manifest") if isinstance(meta.get("case_manifest"), dict) else {}
    dims = list(man.get("dims") or ["regime"])
    checked = set()
    for r in reports or []:
        checked |= checked_sigs(r)
    buckets, boundary = {}, []
    for c in man.get("cases") or []:
        if not isinstance(c, dict) or not c.get("sig"):
            continue
        if is_boundary_case(c):
            rec = {"sig": c["sig"], "regime": c.get("regime"), "dims": {d: c.get(d) for d in dims}}
            if reports is not None:
                rec["checked"] = c["sig"] in checked
            boundary.append(rec)            # A40: checked for correctness, never a bucket of its own
            continue
        key = json.dumps({d: c.get(d) for d in dims}, sort_keys=True, default=str)
        b = buckets.setdefault(key, {"bucket": {d: c.get(d) for d in dims}, "timing": [], "correctness": []})
        if "timing" in (c.get("roles") or []):
            b["timing"].append(c["sig"])
        if "correctness" in (c.get("roles") or []):
            b["correctness"].append(c["sig"])
    for b in buckets.values():
        if reports is not None:
            b["correctness_checked"] = [x for x in b["correctness"] if x in checked]
    declared_timing = sorted({c.get("sig") for c in man.get("cases") or [] if isinstance(c, dict)
                              and "timing" in (c.get("roles") or [])} - {None})
    positive = positive_mass_sigs(h, meta)
    with_calls = sorted(positive) if positive is not None else declared_timing
    timed = sorted(timed_sigs) if timed_sigs is not None else None
    wl = meta.get("workload") if isinstance(meta.get("workload"), dict) else {}
    out = {"weighting": mode, "buckets": list(buckets.values()), "boundary": boundary,
           "calls": {"with_calls": with_calls, "timed": timed,
                     "uncovered": sorted(set(with_calls) - set(timed)) if timed is not None else None,
                     # A35: the calls the task declares it does not time (meta.workload.uncovered, as written)
                     "declared_uncovered": wl.get("uncovered") if wl.get("uncovered") is not None else {}}}
    rep0 = (reports or [None])[0] or {}
    graph = {}
    try:
        needed = h.deployment_graphs(meta) if hasattr(h, "deployment_graphs") else {}
    except Exception as exc:
        needed, out["graph_problem"] = {}, f"{type(exc).__name__}: {exc}"
    for phase, info in needed.items():
        label = phase or "default"
        rep = rep0.get(label) if isinstance(rep0.get(label), dict) else {}
        cov = rep.get("graph_coverage") or {}
        groups = rep.get("replay_groups")
        graph[label] = {"capture": info.get("capture"), "dims": info.get("dims"),
                        "deployment_graphs": info.get("graphs"),
                        "groups": groups, "missing": cov.get("missing") if groups is not None else None,
                        "problems": info.get("problems") or []}
    out["graph"] = graph
    caps = meta.get("capabilities") or {}
    carry = [{"phase": label, "group": g.get("group"), "steps": g.get("steps")}
             for label, v in graph.items() for g in (v.get("groups") or []) if g.get("carry_state")]
    seq = rep0.get("sequence") if isinstance(rep0.get("sequence"), list) else None
    out["state"] = {"stateful": bool(caps.get("stateful")), "carry_groups": carry,
                    "tiers": rep0.get("sequence_coverage"),
                    "sequence_steps": len(seq) if seq is not None else None}
    if len(reports or []) > 1:
        out["ranks"] = len(reports)
    return out


def full_result_status(task, contract_sha, snapshot_sha):
    """A28: is the last full UT result (run/ut_result.json) current for this contract and snapshot?"""
    path = os.path.join(task, "run", "ut_result.json")
    try:
        with open(path, encoding="utf-8") as fh:
            full = json.load(fh)
    except (OSError, ValueError):
        return {"path": "run/ut_result.json", "current": False, "why": "no full UT result"}
    why = []
    if full.get("exit_code") != 0:
        why.append(f"its exit code is {full.get('exit_code')}")
    if full.get("contract_sha256") != contract_sha:
        why.append("another test contract")
    if full.get("snapshot_sha256") != snapshot_sha:
        why.append("another source snapshot")
    return {"path": "run/ut_result.json", "current": not why,
            "why": "exit 0, same contract and snapshot" if not why else "; ".join(why)}


# --------------------------------------------------------------------------- output
def _fmt(x, nd=6):
    if _finite(x):
        return f"{x:.{nd}g}"
    return "None"


def print_timing(per_case, weighted, unweighted=False):
    for c in per_case:
        legs = c.get("legs") or {}
        b, o = legs.get("baseline") or {}, legs.get("current") or {}
        line = (f"case {c['sig']}: baseline_ms={_fmt(c.get('baseline_ms'))} optimized_ms={_fmt(c.get('optimized_ms'))} "
                f"speedup={_fmt(c.get('speedup'), 5)} reps={c.get('reps')} timer={c.get('timer')} "
                f"primed={b.get('primed')}/{o.get('primed')} host_ms={_fmt(b.get('host_ms'), 4)}/{_fmt(o.get('host_ms'), 4)}")
        rank = c.get("rank_ms")
        if rank:
            line += f" rank_ms={json.dumps(rank)}"
        if isinstance(c.get("inner_raised"), dict):
            line += f" inner={c['inner_raised'].get('declared')}->{c['inner_raised'].get('final')}"
        print(line)
    if unweighted:
        # A19: no workload weights -- main's normal mode: the primary metric is the unweighted geomean, and
        # GEAK_WEIGHTED_SPEEDUP (which tells the kernel lane the UT is workload-weighted) is not printed.
        print(f"GEOMEAN_SPEEDUP={weighted}")
    else:
        print(f"GEAK_WEIGHTED_SPEEDUP={weighted}")


def write_result(path, result):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, default=str)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- multi-rank pools
# world_size > 1: every leg is a pool of `world` rank processes (leg_runner.py serve mode) that set up
# once and then run commands. A rank that fails ends its pool at once; a pool's ranks die with the
# driver; each pool has its own rendezvous port; at most two pools are alive (the timing's two), fewer
# when GEAK_UT_POOL_PARALLEL / meta.pool_parallel asks for it; a peer-memory kernel never has two pools
# issuing GPU work at the same time (A04: commands are serial and a pool only starts while the others
# are idle).
VISIBILITY_VARS = ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")
AITER_RAW_POOL = "AITER_CUSTOM_AR_RAW_INPUT_POOL"
PORT_ERRORS = ("failed to listen", "address already in use", "eaddrinuse", "errno 98")
# what a rank reports when it only saw a PEER fail
SECONDARY_RE = re.compile(r"Connection closed by peer|remote worker crash|Connection reset by peer"
                          r"|NCCL communicator was aborted|ProcessGroup\w* .*(aborted|was shut down)", re.IGNORECASE)
_PORTS_LOCK = threading.Lock()
_PORTS_GIVEN = set()
_PORT_RNG = random.SystemRandom()


def _port_floor():
    try:
        with open("/proc/sys/net/ipv4/ip_local_port_range") as fh:
            return int(fh.read().split()[0])
    except (OSError, ValueError, IndexError):
        return 32768


def free_port():
    """A rendezvous port below the ephemeral range (no concurrently starting socket can be handed it
    between this probe and rank 0's bind), never given out twice by this driver."""
    hi = _port_floor()
    lo = max(1024, hi - 12000)
    with _PORTS_LOCK:
        for _ in range(500):
            port = _PORT_RNG.randrange(lo, hi)
            if port in _PORTS_GIVEN:
                continue
            with socket.socket() as s:
                try:
                    s.bind(("", port))
                except OSError:
                    continue
            _PORTS_GIVEN.add(port)
            return port
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def leased_devices():
    """The GPU ids this run may use: the lease's HIP_VISIBLE_DEVICES (or CUDA_VISIBLE_DEVICES)."""
    raw = os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES") or ""
    return [d.strip() for d in raw.split(",") if d.strip() and d.strip() != "-1"]


def deployed_env(meta):
    """(env, dropped): the deployment's env for the rank processes (meta.regime.env) without the device
    selection variables -- the serving GPUs are not the leased ones (ISSUE-039 and its ROCR /
    GPU_DEVICE_ORDINAL follow-up)."""
    envs = []
    for reg in [meta.get("regime")] + list((meta.get("phase_regimes") or {}).values()):
        if isinstance(reg, dict):
            e = reg.get("env") or reg.get("deploy_env") or {}
            if isinstance(e, dict):
                envs.append(e)
    merged = {}
    for e in envs:
        merged.update({str(k): str(v) for k, v in e.items()})
    dropped = sorted(k for k in merged if k in VISIBILITY_VARS)
    return {k: v for k, v in merged.items() if k not in VISIBILITY_VARS}, dropped


class PoolError(Exception):
    """kind: env (stopped from outside the code: signal, resources, IPC fault, lock), task (the task's
    own code raised), timeout, port (rendezvous port taken). pool: the label of the pool it ended."""

    def __init__(self, kind, msg, rank=None, tail="", pool=None):
        super().__init__(msg)
        self.kind, self.rank, self.tail, self.pool = kind, rank, tail, pool


class RankPool:
    def __init__(self, task, overlay, *, label, world, devices, setup_s, command_s, deadline, env_extra=None):
        self.task, self.overlay, self.label, self.world = task, overlay, label, world
        self.devices, self.setup_s, self.command_s, self.deadline = devices, setup_s, command_s, deadline
        self.env_extra = dict(env_extra or {})
        self.seq, self.procs, self.port_retries, self.closed = 0, [], 0, False
        self.leg_runner = os.path.join(task, "leg_runner.py")
        self._spawn()

    def _env(self):
        meta = json.load(open(os.path.join(self.task, "meta.json")))
        env = dict(os.environ)      # the caller's environment, user site and PYTHONUSERBASE included (A05)
        dep, _ = deployed_env(meta)      # the deployment's own GPU selection never reaches a rank
        env.update(dep)
        env["PYTHONPATH"] = os.pathsep.join([self.overlay] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        if self.devices:
            ids = ",".join(self.devices[:self.world])
            env["HIP_VISIBLE_DEVICES"] = env["CUDA_VISIBLE_DEVICES"] = ids     # always together (ISSUE-039)
        env.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(self.port), WORLD_SIZE=str(self.world),
                   GEAK_UT_SETUP_S=str(max(1, math.ceil(self.setup_s))),
                   GEAK_UT_COMMAND_S=str(max(1, math.ceil(self.command_s))))
        if threading.current_thread() is threading.main_thread():
            env["GEAK_UT_PARENT_PID"] = str(os.getpid())
        env.update({k: str(v) for k, v in self.env_extra.items()})
        return env

    def _spawn(self):
        self.port = free_port()
        self.dir = tempfile.mkdtemp(prefix=f"geak_ut_pool_{self.label}_")
        env = self._env()
        self.procs = []
        for r in range(self.world):
            renv = dict(env, RANK=str(r), LOCAL_RANK=str(r))
            log = open(os.path.join(self.dir, f"rank{r}.log"), "w")
            # plain `python3 leg_runner.py`: no -I/-s/-E, so the user site stays on (A05)
            self.procs.append(subprocess.Popen([sys.executable, self.leg_runner, "--task", self.task,
                                                "--mode", "serve", "--ctl", self.dir],
                                               env=renv, stdin=subprocess.PIPE, stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True, text=True))
            log.close()

    def _log(self, r):
        try:
            with open(os.path.join(self.dir, f"rank{r}.log"), errors="replace") as fh:
                return fh.read()
        except OSError:
            return ""

    def _limit(self, own):
        left = own
        if self.deadline:
            left = min(left, self.deadline - time.monotonic())
        return left

    def _failure(self, what, among=None):
        """Classify a dead rank: its own exit code (a traceback) is the task's; a signal or an
        environment signature in any rank's log is the environment's. `among` (the ranks that did not
        raise) narrows the dead ranks to those: a rank that raised a peer's failure exits 1 for it."""
        exits = {r: p.poll() for r, p in enumerate(self.procs) if p.poll() is not None}
        if among is not None:
            exits = {r: c for r, c in exits.items() if r in among}
        logs = {r: self._log(r) for r in range(self.world)}
        own = [r for r, c in exits.items() if c is not None and c > 0]
        env = next((env_fault(logs[r]) for r in logs if env_fault(logs[r])), None)
        port = any(e in logs[r][-4000:].lower() for r in logs for e in PORT_ERRORS)
        r = own[0] if own else (min(exits) if exits else (min(among) if among else 0))
        tail = logs.get(r, "")[-2500:]
        if what == "setup" and port:
            return PoolError("port", f"pool {self.label}: rendezvous port {self.port} taken", r, tail,
                             pool=self.label)
        if env:
            return PoolError("env", f"pool {self.label}: {what}: rank {r} stopped by its environment ({env})", r, tail,
                             pool=self.label)
        if own:
            return PoolError("task", f"pool {self.label}: {what}: rank {r} exited {exits[r]}", r, tail,
                             pool=self.label)
        if among is not None and not exits:
            return PoolError("env", f"pool {self.label}: {what}: the ranks that raised only saw a peer fail, and no "
                                    "other rank died", r, tail, pool=self.label)
        return PoolError("env", f"pool {self.label}: {what}: rank {r} killed by signal {-(exits.get(r) or 0)}",
                         r, tail, pool=self.label)

    def _raised(self, results, what):
        """The PoolError of the ranks whose command raised: the root cause first -- a rank that only
        saw its peer die ("connection closed by peer") is a consequence, not the cause. When every
        error is such a consequence, the cause is a rank that did not raise (the OOM killer's SIGKILL
        leaves no traceback): it is judged like a dead rank."""
        bad = [r for r in results if r.get("error")]
        if not bad:
            return None
        root = [r for r in bad if not SECONDARY_RE.search(r["error"])]
        if not root:
            raised = {r.get("rank") for r in bad}
            err = self._failure(what, among=[i for i in range(self.world) if i not in raised])
            err.tail += "\n[ranks that saw it] " + "; ".join(
                f"rank {r.get('rank')}: {r['error'].strip().splitlines()[-1][:200]}" for r in bad)
            return err
        first = root[0]
        others = "; ".join(f"rank {r.get('rank')}: {r['error'].strip().splitlines()[-1][:200]}"
                           for r in bad if r is not first)
        tail = first["error"][-2500:] + (f"\n[other ranks] {others}" if others else "")
        return PoolError("env" if env_fault(first["error"]) else "task",
                         f"pool {self.label}: {what}: rank {first.get('rank')} raised", first.get("rank"), tail,
                         pool=self.label)

    def _wait(self, names, what, limit_s):
        t_end = time.monotonic() + max(1.0, self._limit(limit_s))
        paths = [os.path.join(self.dir, n) for n in names]
        while not all(os.path.exists(p) for p in paths):
            if any(p.poll() is not None for p in self.procs):
                time.sleep(0.5)       # let every failing rank's result file / log land
                got = []
                for p in paths:
                    try:
                        with open(p) as fh:
                            got.append(json.load(fh))
                    except (OSError, ValueError):
                        pass
                err = self._raised(got, what)
                if err is not None:
                    raise err
                raise self._failure(what)
            if time.monotonic() >= t_end:
                raise PoolError("timeout", f"pool {self.label}: {what} exceeded its time limit "
                                           f"({'setup_s' if what == 'setup' else 'command_s'} / total_s)",
                                pool=self.label)
            time.sleep(0.05)
        out = []
        for p in paths:
            with open(p) as fh:
                out.append(json.load(fh))
        err = self._raised(out, what)
        if err is not None:
            raise err
        return out

    def wait_ready(self):
        while True:
            try:
                self.ready = self._wait([f"rank{r}.ready.json" for r in range(self.world)], "setup",
                                        self.setup_s)
                return self.ready
            except PoolError as exc:
                self.close(kill=True)
                if exc.kind == "port" and self.port_retries < 3:
                    self.port_retries += 1
                    self.closed = False
                    self._spawn()
                    continue
                raise

    def run(self, mode, **params):
        self.seq += 1
        line = json.dumps({"seq": self.seq, "mode": mode, "params": params}) + "\n"
        for p in self.procs:
            try:
                p.stdin.write(line)
                p.stdin.flush()
            except (BrokenPipeError, OSError):
                raise self._failure(f"{mode}: rank stdin closed")
        try:
            return self._wait([f"rank{r}.{self.seq}.json" for r in range(self.world)], mode, self.command_s)
        except PoolError:
            self.close(kill=True)
            raise

    def close(self, kill=False, grace=20.0):
        if self.closed:
            return
        self.closed = True
        if not kill:
            for p in self.procs:
                if p.poll() is None:
                    try:
                        p.stdin.write(json.dumps({"mode": "exit"}) + "\n")
                        p.stdin.close()
                    except (BrokenPipeError, OSError, ValueError):
                        pass
            t_end = time.monotonic() + grace
            while any(p.poll() is None for p in self.procs) and time.monotonic() < t_end:
                time.sleep(0.05)
        for p in self.procs:
            if p.poll() is None:
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    try:
                        os.killpg(os.getpgid(p.pid), sig)
                    except (ProcessLookupError, OSError):
                        break
                    try:
                        p.wait(timeout=5)
                        break
                    except subprocess.TimeoutExpired:
                        continue
        shutil.rmtree(self.dir, ignore_errors=True)


def merge_time(per_rank):
    """One leg's `time` results over its ranks. Per case: a sample ends when its slowest rank does, so
    the case time is the median over the samples of the per-sample maximum; rank_ms keeps each rank's
    median; primed is the AND (absent when a rank could not tell). The ranks must agree on the cases,
    their order, the timer and the number of samples."""
    merged = []
    first = per_rank[0].get("cases") or []
    for i, row in enumerate(first):
        rows = [(pr.get("cases") or [])[i] if i < len(pr.get("cases") or []) else None for pr in per_rank]
        if any(r is None or r.get("sig") != row.get("sig") for r in rows):
            raise RuntimeError(f"the ranks disagree on the timed case order at {row.get('sig')}")
        timers = {r.get("timer") for r in rows}
        if len(timers) > 1:
            raise RuntimeError(f"case {row['sig']}: the ranks used different timers {sorted(map(str, timers))}")
        out = {k: row.get(k) for k in ("sig", "regime", "m", "timer", "cache_condition")}
        if row.get("inner") is not None:
            out["inner"] = row["inner"]
        ms = [r.get("ms") for r in rows]
        samples = [r.get("samples") for r in rows]
        if any(v is None for v in ms) or not all(samples) or len({len(s) for s in samples}) != 1:
            out["ms"] = None
        else:
            out["ms"] = _median([max(col) for col in zip(*samples)])
        out["rank_ms"] = ms
        for k in ("wall_ms", "host_ms"):
            vals = [r.get(k) for r in rows]
            out[k] = None if any(v is None for v in vals) else max(vals)
        primed = [r.get("primed") for r in rows]
        if all(p is not None for p in primed):
            out["primed"] = all(bool(p) for p in primed)
        else:
            out.pop("host_ms", None)
        al = [r.get("alignment") or {} for r in rows]
        out["alignment"] = {"samples": al[0].get("samples"),
                            "late_releases": sum(int(a.get("late_releases") or 0) for a in al),
                            "max_late_us": max([float(a.get("max_late_us") or 0) for a in al] or [0.0]),
                            "device_sync": al[0].get("device_sync")}
        merged.append(out)
    return merged


def merge_noise(per_rank):
    """The native self-noise over the ranks: the worst max_rel_err and (A37) the worst noise_rel of any rank,
    with the count of comparisons whose noise_rel was not finite; with per-output tolerances (A42) also per output."""
    noise = [r.get("native_noise") or {} for r in per_rank]
    out = {"max_rel_err": max([n for n in (x.get("max_rel_err") for x in noise) if _finite(n)] or [None],
                              key=lambda v: -1 if v is None else v)}
    if any("noise_rel" in x for x in noise):
        vals = [x["noise_rel"] for x in noise if _finite(x.get("noise_rel"))]
        out["noise_rel"] = max(vals) if vals else None
        nonfinite = sum(int(x.get("noise_rel_nonfinite") or 0) for x in noise)
        if nonfinite:
            out["noise_rel_nonfinite"] = nonfinite
    per = [x["per_output"] for x in noise if isinstance(x.get("per_output"), dict)]
    if per:
        out["per_output"] = merge_output_noise(per)
    return out


def merge_output_noise(per_rank):
    """native_noise.per_output over the ranks (A42): every output's worst finite noise_rel, its tolerance and the
    count of its comparisons whose noise_rel was not finite."""
    out = {}
    for rec in per_rank:
        for output, v in rec.items():
            if not isinstance(v, dict):
                continue
            m = out.setdefault(output, {"tol": v.get("tol"), "noise_rel": None})
            if _finite(v.get("noise_rel")) and (m["noise_rel"] is None or v["noise_rel"] > m["noise_rel"]):
                m["noise_rel"] = v["noise_rel"]
            if v.get("noise_rel_nonfinite"):
                m["noise_rel_nonfinite"] = int(m.get("noise_rel_nonfinite") or 0) + int(v["noise_rel_nonfinite"])
    return out


def noise_advice(native_noise):
    """A37: the native self-noise and the tolerance it needs, for the reason of a native self-check failure (A42:
    output by output when the task gives per-output tolerances)."""
    per = (native_noise or {}).get("per_output")
    if isinstance(per, dict) and per:
        parts = [f"{o}: noise_rel={v['noise_rel']:.4g}, tol {v.get('tol')}, needs at least {2 * v['noise_rel']:.4g}"
                 for o, v in per.items() if isinstance(v, dict) and _finite(v.get("noise_rel"))]
        if parts:
            return ("; its self-noise per output (the native refs leg against the native oracle on the random draws; "
                    "correct() passes an output at a tolerance t exactly when its noise_rel <= t), every output's "
                    "tolerance must be at least 2 x its noise_rel (task-authoring.md, step 0 (b)): " + "; ".join(parts))
    nr = (native_noise or {}).get("noise_rel")
    if not _finite(nr):
        return ""
    return (f"; its self-noise is noise_rel={nr:.4g} (the native refs leg against the native oracle on the random "
            f"draws; correct() passes at a tolerance t exactly when noise_rel <= t), so the tolerance must be at "
            f"least 2 x noise_rel = {2 * nr:.4g} (task-authoring.md, step 0 (b))")


def merge_correct(per_rank):
    """One leg's correctness results over its ranks: every rank must pass."""
    out = {"ok": all(r.get("ok") is True for r in per_rank),
           "incomplete": sorted({x for r in per_rank for x in (r.get("incomplete") or [])}),
           "errors": [f"rank{r.get('rank')}: {e}" for r in per_rank for e in (r.get("errors") or [])],
           "identity": per_rank[0].get("identity"),
           "native_noise": merge_noise(per_rank),
           "ranks": [{"rank": r.get("rank"), "ok": r.get("ok"), "report": r.get("report"),
                      "errors": r.get("errors"), "incomplete": r.get("incomplete")} for r in per_rank]}
    tol = next((r["tolerance"] for r in per_rank if isinstance(r.get("tolerance"), dict)), None)
    if tol is not None:
        out["tolerance"] = tol         # A42: the tolerance every output was compared at (the ranks compare alike)
    return out


def _rank_leg_summary(rows, h):
    """harness_lib's leg receipt of the pairs a bucket ran (the single-rank one), plus each rank's median."""
    out = h._leg_summary(rows)
    ranks = [r.get("rank_ms") for r in rows if r.get("rank_ms")]
    if ranks:
        out["rank_ms"] = [_median([rk[i] for rk in ranks if rk[i] is not None]) if any(rk[i] is not None for rk in ranks)
                          else None for i in range(len(ranks[0]))]
    return out


def _median(xs):
    s = sorted(xs)
    n = len(s)
    if not n:
        return None
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def measure_pools(pn, pc, *, h, early_stop=True, max_reps=3, undecided=(0.95, 1.10), sigs=None, inner=None):
    """measure_legs' semantics over two pools: interleaved B,C pairs per case, at most `max_reps`,
    stopping once decisive (unless `early_stop` is off), medians of the pairs; only one pool times at a
    time. `sigs` / `inner` ({sig: n}) as in harness_lib.measure_legs (A20); `h` the task's harness_lib (each
    leg's receipt). Returns (per_case, alignment rows)."""
    lo, hi = undecided
    per_case, aligned = [], []
    for sig in pn.run("list")[0]["sigs"]:
        if sigs is not None and sig not in sigs:
            continue
        kw = {"inner": int(inner[sig])} if (inner or {}).get(sig) else {}
        brows, orows, bms, oms = [], [], [], []
        for _ in range(max(1, int(max_reps))):
            b = merge_time(pn.run("time", bucket=sig, **kw))
            o = merge_time(pc.run("time", bucket=sig, **kw))
            if not b or not o:
                break
            brows.append(b[0])
            orows.append(o[0])
            aligned += [b[0].get("alignment") or {}, o[0].get("alignment") or {}]
            bms.append(b[0].get("ms"))
            oms.append(o[0].get("ms"))
            gb, go = [x for x in bms if x], [y for y in oms if y]
            if not gb or not go:
                break
            if early_stop and not (lo <= _median(gb) / _median(go) <= hi):
                break
        if not brows:
            continue
        gb, go = [x for x in bms if x], [y for y in oms if y]
        bm, om = (_median(gb) if gb else None), (_median(go) if go else None)
        pair = [x / y for x, y in zip(bms, oms) if x and y]
        legs = {"baseline": _rank_leg_summary(brows, h), "current": _rank_leg_summary(orows, h)}
        timers = {legs["baseline"]["timer"], legs["current"]["timer"]}
        per_case.append({"sig": sig, "regime": brows[0].get("regime", ""), "m": brows[0].get("m"),
                         "baseline_ms": bm, "optimized_ms": om, "speedup": (bm / om) if (bm and om) else None,
                         "reps": len(bms),
                         "speedup_spread": ((max(pair) - min(pair)) / _median(pair)) if len(pair) > 1 else 0.0,
                         "timer": timers.pop() if len(timers) == 1 else "mixed", "legs": legs,
                         "rank_ms": {"baseline": legs["baseline"].get("rank_ms"),
                                     "current": legs["current"].get("rank_ms")}})
        if brows[0].get("inner") is not None:
            per_case[-1]["inner"] = brows[0]["inner"]
    return per_case, aligned


def alignment_receipt(meta, aligned, harness_env):
    t = meta.get("timing") or {}
    spin = float(t.get("device_spin_us", 200.0))
    margin = float(t.get("release_margin_us", 500.0))
    sync = next((a.get("device_sync") for a in aligned if a.get("device_sync")), None)
    device = (f"GPU spin {spin:g} us on the rank's stream -> "
              + ("the task's cases.align (no tested data)" if sync == "cases.align" else
                 "one-element RCCL all_reduce on the pool's device group") + " -> ") \
        if sync and not str(sync).startswith("none") else "(no CUDA: no device step) -> "
    return {
        "per_sample": ("harness_lib.flush_cache -> torch.cuda.synchronize -> gloo all_reduce(MAX) barrier agreeing "
                       f"a release time (now + {margin:g} us) -> busy-wait to the release time -> " + device
                       + "start event -> kernel -> end event"),
        "merge": "per sample: max over the ranks; case time: median over the samples; rank_ms: each rank's median",
        "counts": "identical warmup / timed sample counts and case order on every rank (checked)",
        "release_margin_us": margin, "device_spin_us": spin if sync and not str(sync).startswith("none") else None,
        "device_sync": sync, "samples": sum(int(a.get("samples") or 0) for a in aligned),
        "late_releases": sum(int(a.get("late_releases") or 0) for a in aligned),
        "max_late_us": max([float(a.get("max_late_us") or 0) for a in aligned] or [0.0]),
        "harness_env": harness_env,
    }


def _pool_stop(exc, who):
    """A PoolError as the run's exit: the environment's (no verdict) or the code's."""
    if exc.kind in ("env", "timeout", "port"):
        return UTStop(EXIT_ENV, f"{exc} ({who}){': ' + exc.tail[-1500:] if exc.tail else ''}", "multi_rank")
    if who in ("baseline", "native", "aa", "control"):
        return UTStop(EXIT_INCOMPLETE, f"the {who} pool's own code failed on this task's cases: {exc}: "
                                       f"{exc.tail[-2000:]}", "multi_rank")
    return UTStop(EXIT_FAIL, f"the candidate failed: {exc}: {exc.tail[-2000:]}", "correctness")


def _multi_rank(task, meta, h, args, result, ctx):
    """world_size > 1: oracle, native references and the judged leg in rank pools, then the timing pools."""
    world = world_size(meta)
    legs = ctx["legs"]
    devices = leased_devices()
    cpu_ranks = not devices and os.environ.get("GEAK_UT_ALLOW_CPU_RANKS") == "1"
    if not cpu_ranks and len(devices) < world:
        raise UTStop(EXIT_ENV, f"{world} ranks need {world} visible GPUs but HIP_VISIBLE_DEVICES="
                               f"{os.environ.get('HIP_VISIBLE_DEVICES')!r}: run under one group lease of the "
                               f"whole set (gpu_lock.sh group:<ids>)", "multi_rank")
    tmo = timeouts(meta, world)
    caps = (meta.get("capabilities") or {}).get("multi_rank") or {}
    aa = args.aa or bool(args.control)
    parallel = max(1, int(os.environ.get("GEAK_UT_POOL_PARALLEL") or meta.get("pool_parallel") or 2))
    dep, dropped = deployed_env(meta)
    harness_env = {}
    shared = {}
    if not args.control and AITER_RAW_POOL not in dep:
        # Two pools stay alive on the same GPUs for the timing (A13): a test-harness setting, not part of
        # the deployment configuration.
        shared[AITER_RAW_POOL] = "1"
        harness_env[AITER_RAW_POOL] = "1 (test-harness setting for co-resident pools; not deployment configuration)"
    if dropped:
        harness_env["dropped_from_deployed_env"] = dropped
    result["multi_rank"] = {"world_size": world, "comm": caps.get("comm"), "devices": devices[:world],
                            "pool_parallel": parallel, "cpu_ranks": cpu_ranks, "harness_env": harness_env}

    base, cand = h.build_candidate_overlay(task, meta)
    if aa:
        cand = base
    else:
        try:
            ids = h.assert_legs_differ(task, base, cand, meta, timeout=legs.budget())
        except RuntimeError as exc:
            text = str(exc)
            if "candidate leg cannot import" in text:
                raise UTStop(EXIT_FAIL, f"the candidate does not import: {text}", "legs_differ")
            raise UTStop(EXIT_ENV if env_fault(text) else EXIT_INCOMPLETE, text, "legs_differ")
        except subprocess.TimeoutExpired:
            raise UTStop(EXIT_ENV, "identity resolution timed out", "legs_differ")
        result["identity"] = {"baseline": ids[0], "candidate": ids[1]}
        result["gates"]["legs_differ"] = "pass"

    pools = {}

    def open_pool(label, overlay, extra=None):
        env_extra = dict(shared)
        env_extra.update(extra or {})
        pool = RankPool(task, overlay, label=label, world=world, devices=devices, setup_s=tmo["setup_s"],
                        command_s=tmo["command_s"], deadline=legs.deadline, env_extra=env_extra)
        pools[label] = pool
        pool.wait_ready()      # sequential: no pool starts while another issues GPU work (A04)
        return pool

    def close_pool(label):
        p = pools.pop(label, None)
        if p is not None:
            p.close()

    draws = int(meta.get("random_draws", 3))
    native_dir = ctx["native_dir"]
    os.makedirs(native_dir, exist_ok=True)
    prefix = os.path.join(native_dir, "native")
    stamp = os.path.join(native_dir, "native.json")
    who = "control" if args.control else ("aa" if aa else "candidate")
    try:
        if not args.timing_only:
            native = None
            if args.reuse_native and os.path.isfile(stamp):
                rec = json.load(open(stamp))
                have = all(os.path.isfile(f"{prefix}.rank{r}.pt") and os.path.isfile(f"{prefix}.refs.rank{r}.pt")
                           for r in range(world))
                if rec.get("contract_sha256") == ctx["contract_sha"] and rec.get("world") == world and have:
                    native = rec.get("refs")
            if native is None:
                try:
                    pn = open_pool("native", base)
                    pn.run("oracle", blob=prefix, seed=0, draws=draws)
                    if parallel < 2:
                        close_pool("native")
                    pr = open_pool("native_refs", base)
                    native = merge_correct(pr.run("refs", blob=prefix, draws=draws))
                    close_pool("native_refs")
                except PoolError as exc:
                    raise _pool_stop(exc, "native")
                with open(stamp, "w") as fh:
                    json.dump({"contract_sha256": ctx["contract_sha"], "world": world, "refs": native}, fh, default=str)
            ctx["native_reports"] = [r.get("report") or {} for r in native.get("ranks") or []]
            result["native_noise"] = native.get("native_noise")      # A37: also when the self-check stops the UT
            if native.get("tolerance"):
                result["tolerance"] = native["tolerance"]             # A42: per-output tolerances
            if native.get("incomplete"):
                raise UTStop(EXIT_INCOMPLETE, "; ".join(native["incomplete"]), "native_self_check")
            if not native.get("ok"):
                raise UTStop(EXIT_INCOMPLETE, "the NATIVE code fails this task's own checks on some rank: cases.py "
                                              "or the tolerance is wrong, not the candidate"
                             + noise_advice(native.get("native_noise")) + ": "
                             + json.dumps({"errors": native.get("errors"), "ranks": native.get("ranks")},
                                          default=str)[:3000], "native_self_check")
            result["gates"]["native_self_check"] = "pass"
            extra = {"GEAK_UT_DATA_RANK_SHIFT": "1"} if args.control == "rank_skew" else {}
            params = {"blob": prefix, "draws": draws}
            if args.control:
                params["control"] = args.control
                if ctx.get("control_spec"):
                    params["control_spec"] = json.dumps(ctx["control_spec"])
            try:
                pc = open_pool("candidate", cand, extra)
                judged = merge_correct(pc.run("correct", **params))
            except PoolError as exc:
                if exc.kind == "task" and who == "candidate":
                    judged = {"ok": False, "crashed": True, "incomplete": [], "report": {},
                              "errors": [f"{exc}: {exc.tail[-2000:]}"],
                              "ranks": [{"rank": exc.rank, "ok": False, "errors": [exc.tail[-2000:]], "report": {}}]}
                elif exc.kind == "task" and who == "control":
                    # the control made the task's own code raise: a raising call is never a catch
                    judged = {"ok": False, "crashed": True, "incomplete": [], "report": {},
                              "errors": [f"{exc}: {exc.tail[-2000:]}"],
                              "ranks": [{"rank": exc.rank, "ok": False, "errors": [exc.tail[-2000:]],
                                         "report": {"raised": {"error": exc.tail[-500:]}}}]}
                else:
                    raise _pool_stop(exc, who)
            result["correctness"] = {"native": {"ok": native.get("ok"), "ranks": native.get("ranks")},
                                     "candidate": judged, "by_rank": correctness_by_rank(judged)}
            if not aa:
                want = (result.get("identity") or {}).get("candidate")
                seen = (judged.get("identity") or {}).get(meta["target_callable"])
                if judged.get("ok") and want and seen and seen != want:
                    raise UTStop(EXIT_INCOMPLETE, f"the correctness ranks resolved {seen}, not the candidate {want}",
                                 "correctness")
            if judged.get("incomplete"):
                raise UTStop(EXIT_INCOMPLETE, "; ".join(judged["incomplete"]), "correctness")
            if judged.get("crashed") and not args.control:
                print("CORRECTNESS: FAIL")
                result["gates"]["correctness"] = "fail"
                raise UTStop(EXIT_FAIL, "the candidate crashed on some rank: "
                             + json.dumps(judged.get("errors"), default=str)[:3000], "correctness")
            lister = pools.get("candidate") or pools.get("native")
            if not args.control and lister is not None:
                timed = lister.run("list")[0]["sigs"]
                ctx["timed_sigs"] = timed
                checked = set()
                for r in native.get("ranks") or []:
                    checked |= checked_sigs(r.get("report"))
                mp = manifest_problems(meta, timed, checked, positive_mass_sigs(h, meta))
                if mp:
                    raise UTStop(EXIT_INCOMPLETE, "coverage (ISSUE-006): " + "; ".join(mp), "coverage")
            ok = bool(judged.get("ok"))
            print(f"CORRECTNESS: {'PASS' if ok else 'FAIL'}")
            result["gates"]["correctness"] = "pass" if ok else "fail"
            if not ok:
                raise UTStop(EXIT_FAIL, "correctness failed on some rank: " + json.dumps(
                    {"errors": judged.get("errors"), "by_rank": result["correctness"]["by_rank"]},
                    default=str)[:4000], "correctness")
        if args.correctness_only or args.control:
            return None, None
        try:
            if parallel < 2:
                close_pool("candidate")
            pn = pools.get("native") or open_pool("native", base)
            pc = pools.get("candidate") or open_pool("candidate", cand)
            per_case, aligned = measure_pools(pn, pc, h=h, early_stop=not args.no_early_stop, max_reps=args.reps)

            def again(sig, inner, early_stop=not args.no_early_stop):
                rows, more = measure_pools(pn, pc, h=h, early_stop=early_stop, max_reps=args.reps,
                                           sigs=[sig], inner={sig: inner})
                aligned.extend(more)
                return rows
            per_case = raise_inner(h, meta, per_case, again,
                                   measure_all=lambda sig, inner: again(sig, inner, early_stop=False))
        except PoolError as exc:
            raise _pool_stop(exc, "aa" if aa else ("baseline" if exc.pool == "native" else "candidate"))
        except RuntimeError as exc:
            raise UTStop(EXIT_INCOMPLETE, f"multi-rank timing: {exc}", "timing")
        return per_case, alignment_receipt(meta, aligned, harness_env)
    finally:
        for label in list(pools):
            close_pool(label)


# --------------------------------------------------------------------------- main flow
def _single_rank(task, meta, h, args, result, ctx):
    """world_size == 1: correctness workers + measure_legs (unchanged numbers)."""
    legs = ctx["legs"]
    aa = args.aa or bool(args.control)
    base, cand = h.build_candidate_overlay(task, meta)
    if aa:
        cand = base                          # A/A or a control: the "candidate" is the baseline code
    else:
        try:
            ids = h.assert_legs_differ(task, base, cand, meta, timeout=legs.budget())
        except RuntimeError as exc:
            text = str(exc)
            if "candidate leg cannot import" in text:
                raise UTStop(EXIT_FAIL, f"the candidate does not import: {text}", "legs_differ")
            if "baseline leg cannot import" in text:
                raise UTStop(EXIT_ENV if env_fault(text) else EXIT_INCOMPLETE,
                             f"the baseline does not import: {text}", "legs_differ")
            raise UTStop(EXIT_INCOMPLETE, text, "legs_differ")
        except subprocess.TimeoutExpired:
            raise UTStop(EXIT_ENV, "identity resolution timed out", "legs_differ")
        result["identity"] = {"baseline": ids[0], "candidate": ids[1]}
        result["gates"]["legs_differ"] = "pass"
    draws = int(meta.get("random_draws", 3))
    if not args.timing_only:
        seen = {}
        try:
            native, judged = run_correctness(legs, base, cand, native_dir=ctx["native_dir"], draws=draws,
                                             control=args.control, control_spec=ctx.get("control_spec"),
                                             reuse_native=args.reuse_native, contract_sha=ctx["contract_sha"],
                                             who="control" if args.control else ("aa" if aa else "candidate"),
                                             seen=seen)
        finally:
            if seen.get("native") is not None:
                ctx["native_reports"] = [seen["native"].get("report") or {}]
                result["native_noise"] = seen["native"].get("native_noise")   # A37: also on exit 3
                if seen["native"].get("tolerance"):
                    result["tolerance"] = seen["native"]["tolerance"]          # A42: per-output tolerances
        result["gates"]["native_self_check"] = "pass"
        result["correctness"] = {"native": {"ok": native.get("ok"), "report": native.get("report")},
                                 "candidate": judged, "by_rank": correctness_by_rank(judged)}
        if not aa:
            want = (result.get("identity") or {}).get("candidate")
            seen = (judged.get("identity") or {}).get(meta["target_callable"])
            if judged.get("ok") and want and seen and seen != want:
                raise UTStop(EXIT_INCOMPLETE, f"the correctness worker resolved {seen}, not the candidate {want}",
                             "correctness")
        if judged.get("incomplete"):
            raise UTStop(EXIT_INCOMPLETE, "; ".join(judged["incomplete"]), "correctness")
        timed = None
        if not args.control:
            timed = h._run_leg(task, base, "list", timeout=legs.budget())["sigs"]
            ctx["timed_sigs"] = timed
        mp = manifest_problems(meta, timed, checked_sigs(native.get("report")), positive_mass_sigs(h, meta))
        if mp and not args.control:
            raise UTStop(EXIT_INCOMPLETE, "coverage (ISSUE-006): " + "; ".join(mp), "coverage")
        ok = bool(judged.get("ok"))
        print(f"CORRECTNESS: {'PASS' if ok else 'FAIL'}")
        result["gates"]["correctness"] = "pass" if ok else "fail"
        if not ok:
            env = env_fault("\n".join(judged.get("errors") or []))
            if env and judged.get("crashed"):
                raise UTStop(EXIT_ENV, f"the correctness process died of its environment ({env})", env)
            raise UTStop(EXIT_FAIL, "correctness failed: " + json.dumps(
                {"errors": judged.get("errors"), "report": judged.get("report")}, default=str)[:4000],
                "correctness")
    if args.correctness_only or args.control:
        return None
    try:
        per_case = h.measure_legs(task, meta, base=base, cand=cand, check_differ=False,
                                  early_stop=not args.no_early_stop, max_reps=args.reps,
                                  timeout=legs.budget())

        def again(sig, inner, early_stop=not args.no_early_stop):
            return h.measure_legs(task, meta, base=base, cand=cand, check_differ=False, early_stop=early_stop,
                                  max_reps=args.reps, timeout=legs.budget(), sigs=[sig], inner={sig: inner})
        per_case = raise_inner(h, meta, per_case, again,
                               measure_all=lambda sig, inner: again(sig, inner, early_stop=False))
    except subprocess.TimeoutExpired:
        raise UTStop(EXIT_ENV, "a timing leg exceeded its time limit", "timeout")
    except RuntimeError as exc:
        fault = env_fault(str(exc))
        raise UTStop(EXIT_ENV if fault else (EXIT_INCOMPLETE if aa else EXIT_FAIL),
                     f"timing leg failed{f' ({fault})' if fault else ''}: {str(exc)[-1500:]}", "timing")
    return per_case


def default_out(task, args):
    """Where a run writes its result without --out (A28): only a full run writes the delivered
    run/ut_result.json; a source-mutation run writes run/mutations/<leaf>/ut_result.json and every other
    partial run run/ut_partial/<mode>/ut_result.json."""
    if args.mutation:
        return os.path.join(task, "run", "mutations", args.mutation, "ut_result.json")
    mode = (f"control-{re.sub(r'[^A-Za-z0-9_.-]', '_', args.control)}" if args.control else
            "aa" if args.aa and not (args.correctness_only or args.timing_only) else
            ("aa-" if args.aa else "") + ("correctness-only" if args.correctness_only else
                                          "timing-only" if args.timing_only else ""))
    if mode:
        return os.path.join(task, "run", "ut_partial", mode, "ut_result.json")
    return os.path.join(task, "run", "ut_result.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description="generated kernel UT driver")
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", default="", help="result JSON (default run/ut_result.json)")
    ap.add_argument("--correctness-only", action="store_true")
    ap.add_argument("--timing-only", action="store_true")
    ap.add_argument("--aa", action="store_true", help="A/A: both legs run the baseline overlay (verify_ut)")
    ap.add_argument("--no-early-stop", action="store_true", help="always run --reps pairs (A/A)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--control", default="", help="correctness only, baseline code under this control")
    ap.add_argument("--control-spec", default="", help="JSON spec of a task control")
    ap.add_argument("--native-dir", default="", help="where the native blobs live (default run/ut_native)")
    ap.add_argument("--reuse-native", action="store_true", help="reuse native blobs of the same contract")
    ap.add_argument("--mutation", default="", metavar="LEAF",
                    help="a source-mutation run of LEAF (A28): correctness only, native references reused, "
                         "result in run/mutations/LEAF/ut_result.json")
    args = ap.parse_args(argv)
    t_start = time.monotonic()
    task = os.path.realpath(args.task)
    if args.mutation:
        if not re.match(r"^[A-Za-z0-9_.-]+$", args.mutation) or args.mutation in (".", ".."):
            ap.error("--mutation takes a leaf name of letters, digits, '_', '.', '-'")
        args.correctness_only = args.reuse_native = True
    out_path = args.out or default_out(task, args)
    partial = bool(args.correctness_only or args.timing_only or args.aa or args.control or args.mutation)
    result = {"ok": False, "smoke": "fail", "exit_code": None, "generated_by": VERSION,
              "task_dir": task, "gates": {}, "command": " ".join([os.path.basename(sys.argv[0])] + (argv or sys.argv[1:]))}
    code, reason, ctx, meta, h = EXIT_INCOMPLETE, "", None, None, None
    try:
        with open(os.path.join(task, "meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        result["target_callable"] = meta.get("target_callable")
        world = world_size(meta)
        result["world_size"] = result["required_gpu_count"] = world
        tmo = timeouts(meta, world)
        result["timeouts"] = tmo
        deadline = (t_start + tmo["total_s"] - 30) if tmo.get("total_s") else None
        contract_sha, problems = check_contract(task)
        result["contract_sha256"] = contract_sha
        print(f"[ut] contract {contract_sha}")
        if problems:
            raise UTStop(EXIT_INCOMPLETE, "test contract: " + "; ".join(problems), "contract")
        result["gates"]["contract"] = "pass"
        snap = source_snapshot(task)
        result["snapshot_sha256"] = snap["snapshot_sha256"]
        print(f"[ut] snapshot {snap['snapshot_sha256']}")
        h = _load("harness_lib", os.path.join(task, "harness_lib.py"))
        legs = Legs(task, meta, h, command_s=tmo["command_s"], deadline=deadline)
        ctx = {"legs": legs, "contract_sha": contract_sha,
               "native_dir": args.native_dir or os.path.join(task, "run", "ut_native"),
               "control_spec": json.loads(args.control_spec) if args.control_spec else None}
        if world > 1:
            per_case, alignment = _multi_rank(task, meta, h, args, result, ctx)
        else:
            per_case, alignment = _single_rank(task, meta, h, args, result, ctx), None
        if per_case is not None:
            res = h.serving_weighted_speedup(per_case, meta)
            mode = weighting_mode(h, meta)
            unweighted = mode == "unweighted"
            primary = res.get("geomean") if unweighted else res.get("weighted")
            receipt = build_receipt(per_case, world, alignment, phases=graph_phases(h, meta))
            receipt["flags"] = receipt_flags(h, meta, per_case)
            if unweighted:
                receipt["weighting"] = "unweighted"
            weighting = {k: res.get(k) for k in ("included", "dropped_unserved", "suspect_identity", "reason")}
            weighting["mode"] = mode
            result["timing"] = {"per_case": per_case, "weighted": None if unweighted else res.get("weighted"),
                                "geomean": res.get("geomean"), "primary": primary,
                                "by_phase": (phase_geomeans if unweighted else phase_speedups)(res.get("per_case")),
                                "weighting": weighting}
            if unweighted:
                result["weighting"] = "unweighted"
            result["receipt"] = receipt
            print_timing(per_case, primary, unweighted=unweighted)
            print("GEAK_TIMING_RECEIPT: " + json.dumps(receipt, separators=(",", ":"), default=str))
            fail, incomplete, inconclusive = receipt_problems(h, meta, per_case, receipt)
            if inconclusive:
                raise UTStop(EXIT_ENV, "timing receipt not measurable: " + "; ".join(inconclusive), "receipt")
            if incomplete:
                raise UTStop(EXIT_INCOMPLETE, "; ".join(incomplete), "receipt")
            if fail:
                raise UTStop(EXIT_FAIL, "; ".join(fail), "receipt")
            if primary is None:
                raise UTStop(EXIT_INCOMPLETE, f"{'geomean' if unweighted else 'weighted'} speedup unavailable: "
                                              f"{res.get('reason')}", "timing")
            result["gates"]["timing"] = result["gates"]["receipt"] = "pass"
        code = EXIT_PASS
    except UTStop as stop:
        code, reason = stop.code, stop.reason
        if stop.gate:
            result["gates"][stop.gate] = {EXIT_FAIL: "fail", EXIT_ENV: "env", EXIT_INCOMPLETE: "incomplete"}.get(code)
    except Exception as exc:
        h_inc = getattr(sys.modules.get("harness_lib"), "HarnessIncompleteError", None)
        if h_inc is not None and isinstance(exc, h_inc):
            code, reason = EXIT_INCOMPLETE, f"{h_inc.__name__}: {exc}"
        else:
            fault = env_fault(str(exc))
            code = EXIT_ENV if fault else EXIT_INCOMPLETE
            reason = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}"
    try:
        if meta is not None and h is not None and not args.control:
            result["coverage"] = coverage_summary(h, meta, (ctx or {}).get("native_reports"),
                                                  (ctx or {}).get("timed_sigs"), weighting_mode(h, meta))
    except Exception as exc:              # the record never costs the verdict
        result["coverage"] = {"unavailable": f"{type(exc).__name__}: {exc}"}
    if partial and result.get("contract_sha256") and result.get("snapshot_sha256"):
        fr = full_result_status(task, result["contract_sha256"], result["snapshot_sha256"])
        result["full_result"] = fr
        print(f"[ut] full UT result {fr['path']}: {'current' if fr['current'] else 'stale'} ({fr['why']})")
    result.update(exit_code=code, ok=code == EXIT_PASS, smoke="pass" if code == EXIT_PASS else "fail",
                  wall_s=round(time.monotonic() - t_start, 1))
    if reason:
        result["reason"] = reason
        tag = {EXIT_FAIL: "FAIL", EXIT_ENV: "ENV", EXIT_INCOMPLETE: "UT_HARNESS_INCOMPLETE"}.get(code, "")
        print(f"{tag}: {reason}")
    write_result(out_path, result)
    print("UT_RESULT: " + json.dumps({"ok": result["ok"], "exit_code": code, "result": out_path}))
    return code


if __name__ == "__main__":
    sys.exit(main())
