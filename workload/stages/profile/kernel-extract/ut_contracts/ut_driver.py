#!/usr/bin/env python3
"""Generic kernel UT driver: every gate a delivered task runs, driven by its frozen contract.

A task's `unittest.py` is three lines that hand its directory to `main()`. Everything a
hand-written UT used to re-implement (500+ lines per task) lives here once:

  contract           ut_contract.lint_task: contract, case manifest and meta agree (CPU only)
  inputs             each random case builds reproducibly, honours its layout facts and ranges
  binding            harness_lib.build_candidate_overlay + assert_legs_differ
  launches           (meta.expected_launches, stamped by build_task_plan) the baseline runs
                     every case once under torch.profiler; each phase must launch every
                     task kernel the server launched in that phase -- a UT that calls the
                     right callable can still miss a stage, a dispatch branch or the
                     other phase. The kernels it did launch are recorded per phase
  correctness        baseline and candidate oracle legs in fresh processes, compared per output
                     under the contract's bound, against a float64 reference where one exists
  server_calls       (server evidence) the entry calls the probe server recorded when it first
                     compiled each of the task's Triton variants -- warmup and startup included --
                     replayed in both legs; the candidate must run every call the baseline runs
                     and match it (meta.ut_contract.server_call_tol, default replay_tol)
  compile_variants   (server evidence) every Triton variant the server compiled for the task's
                     kernels must also be compiled by the baseline legs above, and the legs'
                     Triton/torch namespaces must match the server's (meta.server_preimports);
                     a missing variant names the fields that differ from the nearest UT variant
  isolation          (candidate process) call() leaves its inputs unchanged; outputs are fresh
  replay             (graph-deployed phases) baseline eager refs vs candidate graph replay over
                     the task's >= 2 boundary shapes; missing bundle = UT_HARNESS_INCOMPLETE
  negative_controls  each declared source mutation must FAIL the correctness compare by a wide
                     margin; launch/build failure is not a numerical control; source restored
  restore            kernel_src bytes are identical before and after the run
  timing             harness_lib.measure_legs + serving_weighted_speedup -> GEAK_WEIGHTED_SPEEDUP

With several GPUs (--gpus, else $UT_GPUS / $GPU_IDS) the oracle legs are sharded by case and
run concurrently, the two server_calls legs run side by side, and timing spreads its buckets
(each bucket's B,C pairs stay on one device). --quick runs every gate except timing on the
boundary cases of each phase, for fix iterations; --reuse keeps gates that already passed
on an unchanged task (reuse key over helpers, contract, cases, evidence and kernel_src).

Exit status: 0 every required gate passed; 1 a gate failed or could not run (incomplete);
3 the UT itself is incomplete (UT_HARNESS_INCOMPLETE printed) — regenerate the UT, do not
blame the candidate. The task's cases.py supplies only operand construction (via the contract),
`call`, and for graph phases `replay_bundle`; nothing here is operator specific.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# This file sits beside a file named unittest.py: drop the task dir before torch is imported.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != HERE]

GATES = ("contract", "inputs", "binding", "launches", "correctness", "server_calls",
         "compile_variants", "isolation", "replay", "negative_controls", "restore", "timing")
SENTINEL = "UT_HARNESS_INCOMPLETE"
DEFAULT_REPLAY_TOL = 0.02
DEFAULT_CONTROL_MARGIN = 10.0


def _load(name, path):
    if name in sys.modules and getattr(sys.modules[name], "__file__", None) == path:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class Task:
    """The task directory and the frozen pieces every gate reads."""

    def __init__(self, task):
        self.dir = os.path.abspath(task)
        with open(os.path.join(self.dir, "meta.json")) as f:
            self.meta = json.load(f)
        self.U = _load("ut_contract", os.path.join(self.dir, "ut_contract.py"))
        self.doc = self.U.load_task_contract(self.dir)
        with open(os.path.join(self.dir, "run", "ut_cases.json")) as f:
            self.manifest = json.load(f)
        self.mc = self.meta.get("ut_contract") or {}
        self._h = self._cases = self._R = None
        self.gpus = None          # logical GPU indices the gates may spread legs over
        self.sig_filter = None    # --quick: the random cases kept
        self.call_filter = None   # --quick: the server calls kept

    @property
    def h(self):
        if self._h is None:
            self._h = _load("harness_lib", os.path.join(self.dir, "harness_lib.py"))
        return self._h

    @property
    def cases(self):
        if self._cases is None:
            self._cases = _load("cases", os.path.join(self.dir, "cases.py"))
        return self._cases

    @property
    def R(self):
        if self._R is None:
            self._R = _load("reference_f64", os.path.join(self.dir, "reference_f64.py"))
        return self._R

    def spec(self, sig):
        for c in self.manifest["cases"]:
            if c["sig"] == sig:
                return c
        raise KeyError(sig)

    def random_specs(self):
        return [c for c in self.manifest["cases"] if c.get("random")
                and (self.sig_filter is None or c["sig"] in self.sig_filter)]

    def overlays(self):
        return (os.path.join(self.dir, "baseline_overlay"), os.path.join(self.dir, "_cand_overlay"))

    def leaves(self):
        return list(self.mc.get("editable_leaves") or [])

    def phase_regimes(self):
        pr = self.meta.get("phase_regimes")
        if isinstance(pr, dict):
            return pr
        phases = sorted({c.get("regime") for c in self.manifest["cases"] if c.get("regime")})
        return {p: self.meta.get("regime") or {} for p in phases}


def _device(torch):
    return "cuda" if torch.cuda.is_available() else "cpu"


def _gen(torch, device, seed):
    return torch.Generator(device=device).manual_seed(int(seed))


def _rows(T, n):
    """Evenly spaced row subset for the float64 reference, or None for all rows."""
    limit = int(T.mc.get("reference_rows") or T.doc.get("reference_rows") or 256)
    if n <= limit:
        return None
    torch = T.U._torch()
    return torch.linspace(0, n - 1, limit).round().long().unique()


def _cpu(obj):
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_cpu(v) for v in obj)
    return obj


def _references(T, args):
    """(refs, magnitudes, rows) of one built case, from its logical operands on the CPU."""
    name = T.doc.get("reference")
    if not name:
        return None, None, None
    logical = _cpu(args["_logical"])
    sym = args["_sym"]
    # The row axis is the leading axis of the first present tensor input (tokens / rows).
    lead = next((v.shape[0] for k, v in logical.items() if k in T.doc["inputs"]
                 and hasattr(v, "dim") and v.dim()), 0)
    rows = _rows(T, lead) if T.doc.get("reference_rows_axis", True) else None
    refs, mags = T.R.reference(name, logical, sym, rows)
    return refs, mags, rows


def _summarize(res):
    """Keep the measures that decide a compare, drop the bulk."""
    out = {"ok": res["ok"]}
    for name, r in res["outputs"].items():
        out[name] = {k: v for k, v in r.items() if k not in ("shape",)}
    return out


# ----------------------------------------------------------------------------- gates
def gate_contract(T, a):
    verdict = T.U.lint_task(T.dir)
    return {"status": "pass" if verdict["ok"] else "fail", **verdict}


def gate_inputs(T, a):
    torch = T.U._torch()
    device = _device(torch)
    rows, ok = [], True
    for spec in T.random_specs():
        seed = T.cases.case_seed(spec)
        same, digests = T.U.check_reproducible(lambda g: T.cases.make_args(spec, gen=g), seed, device)
        args = T.cases.make_args(spec, gen=_gen(torch, device, seed))
        lf_ok, lf_bad = T.U.check_layout_facts(T.doc, args, args["_sym"])
        iv_ok, iv_bad = T.U.check_input_values(T.doc, args, args["_sym"])
        rows.append({"sig": spec["sig"], "reproducible": same, "layout_facts": lf_bad,
                     "values": iv_bad, **({} if same else digests)})
        ok = ok and same and lf_ok and iv_ok
        del args
    return {"status": "pass" if ok else "fail", "cases": rows}


def gate_binding(T, a):
    base, cand = T.h.build_candidate_overlay(T.dir, T.meta)
    bi, ci = T.h.assert_legs_differ(T.dir, base, cand, T.meta)
    return {"status": "pass", "baseline": bi, "candidate": ci}


def _watch_env(T, state, name):
    """COMPILE_WATCH_OUT for one baseline leg (server evidence only); registered for
    compile_variants once the leg has written it."""
    if not name or not (T.meta.get("server_evidence") or {}).get("compile_variants"):
        return {}, None
    path = os.path.join(state["scratch"], "variants_%s.json" % name)
    return {"COMPILE_WATCH_OUT": path}, path


def _register_watch(state, path):
    if path and os.path.exists(path):
        state.setdefault("watch_files", []).append(path)


def _parallel(items, gpus, fn):
    """Run fn(item, gpu) for every item, at most one item per GPU at a time; re-raise the
    first failure only after every item has finished, so no leg is left running."""
    if len(gpus) <= 1 or len(items) <= 1:
        return [fn(it, gpus[0] if gpus else None) for it in items]
    import queue
    import threading
    free, out, errs = queue.Queue(), [None] * len(items), []
    for g in gpus:
        free.put(g)

    def run(i, it):
        g = free.get()
        try:
            out[i] = fn(it, g)
        except BaseException as e:  # noqa: B902 -- re-raised below
            errs.append(e)
        finally:
            free.put(g)
    threads = [threading.Thread(target=run, args=(i, it)) for i, it in enumerate(items)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errs:
        raise errs[0]
    return out


def _oracles(T, state, jobs):
    """Oracle legs -> one blob per job. A job is {overlay, path, seed, draws, watch}.

    With several GPUs (--gpus / GPU_IDS) every job is split into one shard of random cases
    per GPU and all shards of all jobs run concurrently. Draws are seeded per case, so a
    sharded blob is the unsharded blob."""
    torch = T.U._torch()
    sigs = [c["sig"] for c in T.random_specs()]
    gpus = T.gpus or [None]
    k = max(1, min(len(gpus), len(sigs)))
    shards = [sigs[i::k] for i in range(k)]
    items = [(j, i) for j in range(len(jobs)) for i in range(k)]

    def leg(item, gpu):
        j, i = item
        job = jobs[j]
        env, wpath = _watch_env(T, state, job.get("watch") and "%s.s%d" % (job["watch"], i))
        env.update(T.h.gpu_env(gpu))
        out = "%s.s%d" % (job["path"], i)
        T.h._run_leg(T.dir, job["overlay"], "oracle", out=out, seed=job["seed"],
                     draws=job["draws"], env_extra=env,
                     sigs=shards[i] if (k > 1 or T.sig_filter is not None) else None)
        _register_watch(state, wpath)
        return out
    paths = _parallel(items, gpus, leg)
    blobs = [{} for _ in jobs]
    for (j, _), path in zip(items, paths):
        blobs[j].update(torch.load(path, map_location="cpu"))
    return blobs


def _oracle(T, overlay, path, seed, draws, state=None, watch=None):
    state = state if state is not None else {"scratch": os.path.dirname(path)}
    return _oracles(T, state, [{"overlay": overlay, "path": path, "seed": seed, "draws": draws,
                                "watch": watch}])[0]


def _needs_noise(T):
    return any(o["bound"]["kind"] == "noise_relative" for o in T.doc["outputs"].values())


def compare_blobs(T, cand_blob, base_blob, seed, draws, noise_blob=None, only_draw=None):
    """Contract compare of every random case/draw; returns (ok, rows)."""
    torch = T.U._torch()
    device = _device(torch)
    rows, ok = [], True
    for spec in T.random_specs():
        for i in range(draws):
            if only_draw is not None and i != only_draw:
                continue
            key = "%s|%d" % (spec["sig"], i)
            if key not in cand_blob or key not in base_blob:
                rows.append({"case": key, "ok": False, "why": "missing from %s" % (
                    "candidate" if key not in cand_blob else "baseline")})
                ok = False
                continue
            args = T.cases.make_args(spec, gen=_gen(torch, device, seed + i), with_logical=True)
            refs, mags, sel = _references(T, args)
            inputs = {k: v for k, v in _cpu(args).items() if not str(k).startswith("_")}
            res = T.U.compare_outputs(T.doc, cand_blob[key], base_blob[key], sym=args["_sym"],
                                      refs=refs, magnitudes=mags, inputs=inputs, rows=sel,
                                      base_noise=(noise_blob or {}).get(key))
            rows.append({"case": key, **_summarize(res)})
            ok = ok and res["ok"]
            del args, refs, mags, inputs
    return ok, rows


def gate_correctness(T, a, state):
    base, cand = T.h.build_candidate_overlay(T.dir, T.meta)
    seed, draws = a.seed, a.draws
    sd = state["scratch"]
    jobs = [{"overlay": base, "path": os.path.join(sd, "base.pt"), "seed": seed, "draws": draws,
             "watch": "oracle"},
            {"overlay": cand, "path": os.path.join(sd, "cand.pt"), "seed": seed, "draws": draws}]
    if _needs_noise(T):
        jobs.append({"overlay": base, "path": os.path.join(sd, "base2.pt"), "seed": seed,
                     "draws": draws})
    blobs = _oracles(T, state, jobs)
    base_blob, cand_blob = blobs[0], blobs[1]
    noise_blob = blobs[2] if _needs_noise(T) else None
    state["base_blob"], state["noise_blob"] = base_blob, noise_blob
    out = {}
    if T.doc["compare"].get("determinism") == "bitwise":
        again = _oracle(T, cand, os.path.join(sd, "cand2.pt"), seed, draws)
        flaky = sorted(k for k in cand_blob
                       if T.U.tensor_digest(cand_blob[k]) != T.U.tensor_digest(again.get(k)))
        out["bitwise_repeat"] = {"ok": not flaky, "differing": flaky}
    ok, rows = compare_blobs(T, cand_blob, base_blob, seed, draws, noise_blob)
    out["cases"] = rows
    good = ok and out.get("bitwise_repeat", {}).get("ok", True)
    return {"status": "pass" if good else "fail", "seed": seed, "draws": draws, **out}


def _server_calls_leg(T, overlay, out, env=None):
    """One server_calls leg; a crash is attributed to the call it was running."""
    try:
        T.h._run_leg(T.dir, overlay, "server_calls", out=out, seed=0, timeout=7200, env_extra=env,
                 sigs=T.call_filter)
    except Exception as e:
        cid = None
        if os.path.exists(out + ".progress"):
            with open(out + ".progress") as f:
                cid = f.read().strip()
        return None, "%s crashed%s: %s" % (os.path.basename(overlay),
                                          " on call %s" % cid if cid else "", str(e)[-1500:])
    return T.U._torch().load(out, map_location="cpu"), None


def gate_server_calls(T, a, state):
    ev = T.meta.get("server_evidence") or {}
    if not ev.get("server_calls"):
        return {"status": "not_required", "why": "no server evidence (meta.server_evidence)"}
    with open(os.path.join(T.dir, ev["server_calls"])) as f:
        doc = json.load(f)
    if not doc.get("calls"):
        return {"status": "not_required", "why": "the server recorded no call of %s that compiled a "
                "variant of this task's kernels" % doc.get("target_callable"),
                "variants_without_entry_frame": doc.get("unmatched_variants")}
    base, cand = T.h.build_candidate_overlay(T.dir, T.meta)
    sd = state["scratch"]
    wenv, wpath = _watch_env(T, state, "server_calls")
    legs = [(base, os.path.join(sd, "calls_base.pt"), wenv),
            (cand, os.path.join(sd, "calls_cand.pt"), {})]
    (b, bwhy), (c, cwhy) = _parallel(legs, T.gpus or [None],
                                     lambda leg, g: _server_calls_leg(T, leg[0], leg[1],
                                                                      dict(leg[2], **T.h.gpu_env(g))))
    _register_watch(state, wpath)
    if b is None:
        return {"status": "fail", "why": "baseline " + bwhy}
    if c is None:
        return {"status": "fail", "why": "candidate " + cwhy}
    tol = float(T.mc.get("server_call_tol", T.mc.get("replay_tol", DEFAULT_REPLAY_TOL)))
    ignore = T.mc.get("server_call_ignore") or []
    cw = _load("compile_watch", os.path.join(T.dir, "compile_watch.py"))
    rows, ok = [], True
    for call in doc["calls"]:
        if T.call_filter is not None and call["id"] not in T.call_filter:
            continue
        rb, rc = b.get(call["id"], {}), c.get(call["id"], {})
        row = {"id": call["id"], "kernel": call.get("kernel"), "callable": call.get("callable"),
               "before_ready": call.get("before_ready"), "baseline": rb.get("status"),
               "candidate": rc.get("status")}
        if rb.get("status") != "ok":
            # Not the candidate's fault; the variant it would have compiled is then checked by
            # compile_variants, which fails unless another case builds it.
            row.update(ok=True, note="baseline did not replay it: %s" % rb.get("why"))
        elif rc.get("status") != "ok":
            row.update(ok=False, why="candidate: %s" % (rc.get("why") or rc.get("status")))
        else:
            good, outs = cw.compare_call_outputs(rb["outputs"], rc["outputs"], tol, ignore)
            row.update(ok=good, outputs=[o for o in outs if not o["ok"]] or len(outs))
        ok = ok and row["ok"]
        rows.append(row)
    replayed = sum(r["baseline"] == "ok" for r in rows)
    return {"status": "pass" if ok else "fail", "tol": tol, "calls": len(rows),
            "replayed_by_baseline": replayed, "rows": rows}


def gate_compile_variants(T, a, state):
    ev = T.meta.get("server_evidence") or {}
    if not ev.get("compile_variants"):
        return {"status": "not_required", "why": "no server evidence (meta.server_evidence)"}
    with open(os.path.join(T.dir, ev["compile_variants"])) as f:
        srv = json.load(f)
    if not srv.get("variants"):
        return {"status": "not_required",
                "why": "the server compiled no Triton variant on this task's launch paths"}
    if not state.get("watch_files"):
        _oracle(T, T.overlays()[0], os.path.join(state["scratch"], "base_cv.pt"), a.seed, 1,
                state, watch="oracle_only")
    cw = _load("compile_watch", os.path.join(T.dir, "compile_watch.py"))
    ut_events = []
    for p in state.get("watch_files") or []:
        with open(p) as f:
            ut_events += json.load(f).get("events") or []
    cov = cw.coverage(srv["variants"], ut_events, T.mc.get("compile_variant_waivers"))
    ut_env = next((e["env"] for e in ut_events if e.get("env")), {})
    env = cw.env_diff(srv.get("env"), ut_env)
    advisory = srv.get("server_mode") == "eager" and any(
        T.h.deployment_graph_mode(r or {}) for r in T.phase_regimes().values())
    bad = bool(cov["missing"] or env)
    out = {"server_variants": cov["server_variants"], "covered": cov["covered"],
           "missing": cov["missing"], "waived": cov["waived"], "env_differs": env,
           "ut_variants": len({e["id"] for e in ut_events})}
    if env:
        out["fix_env"] = ("the server's Triton/torch namespaces differ from the UT legs'; add the "
                          "module that rebinds them to meta.server_preimports")
    if cov["missing"]:
        out["fix_missing"] = ("add a case that builds each missing variant (the server call that "
                              "built it is in run/server_calls.json, if one was recorded) or waive "
                              "it in meta.ut_contract.compile_variant_waivers [{id, why}]")
    if bad and advisory:
        out["note"] = ("server evidence came from an eager fallback server, whose decode variants "
                       "a graph deployment does not build eagerly: advisory only")
        return {"status": "pass", **out}
    return {"status": "fail" if bad else "pass", **out}


def _inproc(T, overlay, mode, extra=(), timeout=3600):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([overlay] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    cmd = [sys.executable, os.path.join(T.dir, "ut_driver.py"), "--task", T.dir, "--inproc", mode,
           *extra]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    if SENTINEL in p.stdout:
        line = next(ln for ln in p.stdout.splitlines() if SENTINEL in ln)
        return {"status": "harness_incomplete", "why": line}
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
    if p.returncode != 0 or not lines:
        return {"status": "fail", "why": "inproc %s exited %d: %s" % (mode, p.returncode,
                                                                       (p.stderr or p.stdout)[-1500:])}
    return json.loads(lines[-1])


def gate_launches(T, a, state):
    want = T.meta.get("expected_launches") or {}
    if not want:
        return {"status": "not_required", "why": "meta.expected_launches is absent (no "
                "per-phase launch evidence for this task)"}
    base, _ = T.h.build_candidate_overlay(T.dir, T.meta)
    r = _inproc(T, base, "launches", ("--seed", str(a.seed)))
    if r.get("status") != "ok":
        return r
    seen = r["seen"]
    missing = {ph: [k for k in ks if k not in set(seen.get(ph) or [])]
               for ph, ks in want.items()}
    missing = {ph: ks for ph, ks in missing.items() if ks}
    out = {"expected": want, "seen": seen, "missing": missing,
           "other_kernels": {ph: sorted(set(ks) - {k for v in want.values() for k in v})
                             for ph, ks in seen.items()}}
    if missing and T.sig_filter is not None:
        out["note"] = "--quick runs a subset of cases: a missing branch is advisory here"
        return {"status": "pass", **out}
    if missing:
        out["fix"] = ("the UT never launches these server kernels in that phase: add the "
                      "cases or call path that reaches them (a stage of the op, a dispatch "
                      "branch, the other phase), or justify each in "
                      "meta.ut_contract.launch_waivers [{kernel, phase, why}]")
        waived = {(w.get("kernel"), w.get("phase")) for w in
                  (T.mc.get("launch_waivers") or []) if (w.get("why") or "").strip()}
        left = {ph: [k for k in ks if (k, ph) not in waived] for ph, ks in missing.items()}
        if not any(left.values()):
            return {"status": "pass", "waived": missing, **out}
        return {"status": "fail", **out}
    return {"status": "pass", **out}


def inproc_launches(T, a):
    """Baseline overlay: every random case once (after a warm-up call, so compilation and
    autotuning are outside the window), one profiler session per phase."""
    torch = T.U._torch()
    if not torch.cuda.is_available():
        return {"status": "skipped", "why": "no CUDA"}
    from torch.profiler import profile, ProfilerActivity
    device = _device(torch)
    by_phase = {}
    for spec in T.random_specs():
        by_phase.setdefault(spec.get("regime") or "-", []).append(spec)
    seen = {}
    for ph, specs in by_phase.items():
        names = set()
        for spec in specs:
            args = T.cases.make_args(spec, gen=_gen(torch, device, a.seed))
            T.cases.call(args)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                T.cases.call(args)
                torch.cuda.synchronize()
            for e in prof.events():
                if getattr(e, "device_type", None) is not None and "CUDA" in str(e.device_type):
                    names.add(e.name)
            del args
        seen[ph] = sorted(names)
    return {"status": "ok", "seen": seen}


def gate_isolation(T, a, state):
    _, cand = T.h.build_candidate_overlay(T.dir, T.meta)
    return _inproc(T, cand, "isolation", ("--seed", str(a.seed)))


def gate_replay(T, a, state):
    graph = [p for p, r in T.phase_regimes().items() if T.h.deployment_graph_mode(r or {})]
    if not graph:
        return {"status": "not_required", "why": "no phase deploys under a CUDA graph"}
    base, cand = T.h.build_candidate_overlay(T.dir, T.meta)
    out, status = {}, "pass"
    for phase in graph:
        refs = os.path.join(state["scratch"], "replay_%s.pt" % phase)
        r = _inproc(T, base, "replay-ref", ("--phase", phase, "--refs", refs))
        if r.get("status") == "pass":
            r = _inproc(T, cand, "replay", ("--phase", phase, "--refs", refs))
        out[phase] = r
        rank = {"pass": 0, "skipped": 1, "fail": 2, "harness_incomplete": 3}
        if rank.get(r.get("status"), 2) > rank[status]:
            status = r.get("status") if r.get("status") in rank else "fail"
    return {"status": status, "phases": out}


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def gate_negative_controls(T, a, state):
    controls = T.mc.get("negative_controls") or []
    if not controls:
        return {"status": "fail", "why": "meta.ut_contract.negative_controls is empty"}
    margin_needed = float(T.mc.get("min_control_margin", DEFAULT_CONTROL_MARGIN))
    base_blob, noise_blob = state.get("base_blob"), state.get("noise_blob")
    if base_blob is None:
        base = T.overlays()[0]
        base_blob = _oracle(T, base, os.path.join(state["scratch"], "base.pt"), a.seed, 1)
        if _needs_noise(T):
            noise_blob = _oracle(T, base, os.path.join(state["scratch"], "base2.pt"), a.seed, 1)
    rows, ok = [], True
    for i, c in enumerate(controls):
        path = os.path.join(T.dir, c["leaf"])
        with open(path, "rb") as f:
            pristine = f.read()
        text = pristine.decode("utf-8", errors="surrogateescape")
        row = {"leaf": c["leaf"], "expect": c.get("expect")}
        if text.count(c["find"]) != 1:
            row.update(ok=False, why="`find` occurs %d times" % text.count(c["find"]))
            rows.append(row)
            ok = False
            continue
        try:
            with open(path, "wb") as f:
                f.write(text.replace(c["find"], c["replace"]).encode("utf-8", errors="surrogateescape"))
            _, cand = T.h.build_candidate_overlay(T.dir, T.meta)
            try:
                blob = _oracle(T, cand, os.path.join(state["scratch"], "control_%d.pt" % i), a.seed, 1)
            except Exception as e:      # build/import/launch failure: not a numerical control
                row.update(ok=False, why="mutated source did not run to completion (%s); a "
                                          "control must execute and produce wrong numbers" % str(e)[-300:])
                rows.append(row)
                ok = False
                continue
            caught, results = False, []
            _, crow = compare_blobs(T, blob, base_blob, a.seed, 1, noise_blob, only_draw=0)
            margin = 0.0
            for r in crow:
                if not r["ok"]:
                    caught = True
                    m = T.U.control_margin({"outputs": {k: v for k, v in r.items()
                                                        if isinstance(v, dict)}})
                    margin = max(margin, m)
                results.append({"case": r["case"], "ok": r["ok"]})
            row.update(caught=caught, margin=margin, cases=results,
                       ok=caught and margin >= margin_needed)
            if caught and margin < margin_needed:
                row["why"] = "caught at %.3gx the bound, need >= %gx" % (margin, margin_needed)
            elif not caught:
                row["why"] = "the mutation passed every compare: the gate cannot see this defect"
        finally:
            with open(path, "wb") as f:
                f.write(pristine)
        row["restored"] = _sha(path) == hashlib.sha256(pristine).hexdigest()
        ok = ok and row["ok"] and row["restored"]
        rows.append(row)
    T.h.build_candidate_overlay(T.dir, T.meta)
    return {"status": "pass" if ok else "fail", "min_margin": margin_needed, "controls": rows}


def gate_restore(T, a, state):
    now = _tree_hashes(T)
    changed = sorted(k for k in set(now) | set(state["hashes"]) if now.get(k) != state["hashes"].get(k))
    return {"status": "pass" if not changed else "fail", "changed": changed}


def gate_timing(T, a, state):
    per = T.h.measure_legs(T.dir, T.meta, max_reps=a.max_reps, gpus=T.gpus)
    w = T.h.serving_weighted_speedup(per, T.meta)
    early = [c["sig"] for c in per if c.get("reps", 0) < a.max_reps]
    if w.get("primary") is not None:
        print("GEAK_WEIGHTED_SPEEDUP=%s" % w["primary"], flush=True)
    return {"status": "pass" if w.get("primary") is not None else "fail",
            "weighted_speedup": w.get("primary"), "geomean": w.get("geomean"),
            "aggregate": w, "per_case": per, "stopped_before_max_reps": early,
            "note": ("stopped_before_max_reps buckets were decided by measure_legs' early exit "
                     "(ratio outside its undecided band after fewer pairs); their spread is from "
                     "fewer samples"),
            "cache_condition": T.h.cache_policy(), "reason": w.get("reason", "")}


def _tree_hashes(T):
    out = {}
    root = os.path.join(T.dir, "kernel_src")
    for d, _, files in os.walk(root):
        for f in files:
            p = os.path.join(d, f)
            if "__pycache__" not in p:
                out[os.path.relpath(p, T.dir)] = _sha(p)
    return out


# ----------------------------------------------------------------------------- in-process modes
def inproc_isolation(T, a):
    """Runs under the candidate overlay: inputs unchanged by call(), outputs freshly allocated."""
    torch = T.U._torch()
    device = _device(torch)
    rows, ok = [], True
    for spec in T.random_specs():
        args = T.cases.make_args(spec, gen=_gen(torch, device, a.seed))
        same, changed, _ = T.U.inputs_unmodified(T.cases.call, args)
        other = T.cases.make_args(spec, gen=_gen(torch, device, a.seed + 1))
        ind_ok, why = T.h.assert_independent_outputs(T.cases.call, args, other)
        rows.append({"sig": spec["sig"], "inputs_unchanged": same, "changed": changed,
                     "fresh_outputs": ind_ok, "why": why})
        ok = ok and same and ind_ok
        del args, other
    return {"status": "pass" if ok else "fail", "cases": rows}


def _bundle(T, phase):
    fn = getattr(T.cases, "replay_bundle", None)
    b = fn(T.h, T.meta, phase) if fn else None
    if not b or len(b.get("cases") or []) < 2 or any(not callable(b.get(k)) for k in ("fill", "run", "read_out")):
        why = ("phase %s deploys under a CUDA graph but cases.replay_bundle returned %s; a graph "
               "phase needs fill/run/read_out and >= 2 boundary cases. This is a UT generation "
               "defect, not a kernel failure." % (phase, "nothing" if not b else
                                                  "%d case(s)" % len(b.get("cases") or [])))
        print("%s: %s" % (SENTINEL, why), flush=True)
        sys.exit(3)
    return b


def _read(b, case):
    out = b["read_out"]()
    if callable(b.get("valid")):
        out = b["valid"](case, out)
    return _cpu(out)


def inproc_replay_ref(T, a):
    """Baseline overlay, no graph: run every bundle case eagerly through the same static buffers."""
    torch = T.U._torch()
    if not torch.cuda.is_available():
        return {"status": "skipped", "why": "no CUDA: graph replay cannot run here"}
    b = _bundle(T, a.phase)
    refs = {}
    for c in b["cases"]:
        b["fill"](c)
        b["run"]()
        torch.cuda.synchronize()
        refs[c["sig"]] = _read(b, c)
    torch.save(refs, a.refs)
    return {"status": "pass", "cases": len(refs)}


def inproc_replay(T, a):
    """Candidate overlay: capture once on the capture case, replay every case through the same
    static buffers, compare with the baseline's eager outputs under the contract."""
    torch = T.U._torch()
    if not torch.cuda.is_available():
        return {"status": "skipped", "why": "no CUDA"}
    b = _bundle(T, a.phase)
    base = torch.load(a.refs, map_location="cpu")
    cases = b["cases"]
    try:
        b["fill"](cases[int(b.get("capture_idx", 0))])
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                b["run"]()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            b["run"]()
    except Exception as e:
        return {"status": "fail", "why": "graph capture failed under the candidate: %r" % e}
    tol = float(T.mc.get("replay_tol", DEFAULT_REPLAY_TOL))
    doc = _replay_doc(T, tol)
    rows, ok = [], True
    for c in cases:
        try:
            b["fill"](c)
            g.replay()
            torch.cuda.synchronize()
            got = _read(b, c)
            refs, mags, sel = (_references(T, c["args"]) if "_logical" in (c.get("args") or {})
                               else (None, None, None))
            sym = (c.get("args") or {}).get("_sym") or {}
            res = T.U.compare_outputs(doc, got, base[c["sig"]], sym=sym, refs=refs,
                                      magnitudes=mags, rows=sel,
                                      inputs={k: v for k, v in _cpu(c.get("args") or {}).items()
                                              if not str(k).startswith("_")})
            rows.append({"case": c["sig"], **_summarize(res)})
            ok = ok and res["ok"]
        except Exception as e:
            rows.append({"case": c["sig"], "ok": False, "why": "replay raised (OOB/stale under replay): %r" % e})
            ok = False
    return {"status": "pass" if ok else "fail", "replay_tol": tol, "cases": rows}


def _replay_doc(T, tol):
    """The contract for replay compares: bounds that need a second baseline run fall back to the
    harness tolerance; the rest apply unchanged (they get the float64 reference when the bundle's
    cases carry logical operands)."""
    doc = json.loads(json.dumps(T.doc))
    for o in doc["outputs"].values():
        if o["bound"]["kind"] == "noise_relative":
            o["bound"] = {"kind": "harness", "tol": tol}
    return doc


# ----------------------------------------------------------------------------- orchestration
def run_gates(T, a):
    state = {"scratch": tempfile.mkdtemp(prefix="ut_%s_" % os.path.basename(T.dir)),
             "hashes": _tree_hashes(T)}
    wanted = [g for g in GATES if g in a.gates]
    report = {"task": T.dir, "family": T.doc["family"],
              "contract_sha256": T.U.contract_sha256(T.doc), "started": time.time(), "gates": {}}
    report["reuse_key"] = state["reuse_key"] = _reuse_key(T)
    prev = _reusable(T, a, state["reuse_key"])
    stop = None
    incomplete = False
    try:
        for g in wanted:
            if stop and not a.keep_going and g not in ("restore",):
                report["gates"][g] = {"status": "skipped", "why": "after %s" % stop}
                continue
            if g in prev:
                report["gates"][g] = dict(prev[g], reused=True)
                print("UT_GATE %-18s %s (reused)" % (g, prev[g]["status"]), flush=True)
                continue
            t0 = time.time()
            fn = globals()["gate_" + g]
            try:
                r = fn(T, a) if g in ("contract", "inputs", "binding") else fn(T, a, state)
            except SystemExit:
                raise
            except Exception as e:
                r = {"status": "harness_incomplete" if type(e).__name__ == "HarnessIncompleteError"
                     else "fail", "why": "%s: %s" % (type(e).__name__, e)}
            r["seconds"] = round(time.time() - t0, 2)
            report["gates"][g] = r
            print("UT_GATE %-18s %s" % (g, r["status"]), flush=True)
            if r["status"] == "harness_incomplete":
                incomplete = True
            if r["status"] not in ("pass", "not_required"):
                stop = stop or g
    finally:
        shutil.rmtree(state["scratch"], ignore_errors=True)
    statuses = [r["status"] for r in report["gates"].values()]
    if incomplete:
        verdict, code = "harness_incomplete", 3
    elif a.quick:
        verdict, code = "quick", 1        # a subset of cases: an iteration aid, never a pass
    elif all(s in ("pass", "not_required") for s in statuses) and set(wanted) == set(GATES):
        verdict, code = "pass", 0
    elif all(s in ("pass", "not_required") for s in statuses):
        verdict, code = "partial", 1      # a subset of gates was asked for: never a full pass
    else:
        verdict, code = "fail", 1
    report.update(verdict=verdict, exit_code=code, seconds=round(time.time() - report["started"], 1))
    return report, code


def _reuse_key(T):
    """Everything a gate's verdict depends on inside the task: helpers, contract, cases,
    server evidence and kernel_src. A gate may be reused only under the same key."""
    h = hashlib.sha256()
    rels = sorted(f for f in os.listdir(T.dir) if f.endswith((".py", ".json")))
    run = os.path.join(T.dir, "run")
    rels += sorted("run/" + f for f in os.listdir(run) if f in ("ut_cases.json", "ut_contract.json")
                   or f.startswith("server_")) if os.path.isdir(run) else []
    calls = os.path.join(run, "server_calls")
    rels += sorted("run/server_calls/" + f for f in os.listdir(calls)) if os.path.isdir(calls) else []
    rels += sorted(_tree_hashes(T))
    for rel in rels:
        p = os.path.join(T.dir, rel)
        if os.path.isfile(p):
            h.update(rel.encode())
            h.update(_sha(p).encode())
    return h.hexdigest()


def _reusable(T, a, key):
    """--reuse: gates that passed in the last full report under the same reuse key."""
    if not a.reuse or a.quick:
        return {}
    try:
        with open(os.path.join(T.dir, "run", "ut_report.json")) as f:
            prev = json.load(f)
    except (OSError, ValueError):
        return {}
    if prev.get("reuse_key") != key or prev.get("quick"):
        return {}
    return {g: r for g, r in (prev.get("gates") or {}).items()
            if r.get("status") in ("pass", "not_required") and g != "restore"}


def _quick_subset(T, per_phase=3, calls=8):
    """The smallest, middle and largest random case of every phase, and a few server calls
    (startup ones first, one per kernel before repeats): what a fix iteration needs to see."""
    by_phase = {}
    for c in T.manifest["cases"]:
        if c.get("random"):
            by_phase.setdefault(c.get("regime"), []).append(c)
    keep = set()
    for rows in by_phase.values():
        rows.sort(key=lambda c: (c.get("m") if c.get("m") is not None else 0, c["sig"]))
        idx = sorted({0, len(rows) // 2, len(rows) - 1})[:per_phase]
        keep.update(rows[i]["sig"] for i in idx)
    T.sig_filter = keep
    ev = T.meta.get("server_evidence") or {}
    if ev.get("server_calls"):
        try:
            with open(os.path.join(T.dir, ev["server_calls"])) as f:
                doc = json.load(f)
        except (OSError, ValueError):
            return
        order = sorted(doc.get("calls") or [], key=lambda c: (not c.get("before_ready"),))
        picked, kernels = [], set()
        for c in order:
            if c.get("kernel") not in kernels:
                picked.append(c["id"])
                kernels.add(c.get("kernel"))
        picked += [c["id"] for c in order if c["id"] not in picked]
        T.call_filter = picked[:calls]


def _snapshot(T):
    h = hashlib.sha256()
    files = ["meta.json", "cases.py", "unittest.py", "run/ut_cases.json", "run/ut_contract.json"]
    files += sorted(_tree_hashes(T))
    for rel in files:
        p = os.path.join(T.dir, rel)
        if os.path.exists(p):
            h.update(rel.encode())
            h.update(_sha(p).encode())
    return h.hexdigest()


def main(task=None, argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default=task)
    ap.add_argument("--gates", default=",".join(GATES))
    ap.add_argument("--skip", default="", help="comma list of gates to leave out (the verdict is then partial)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--draws", type=int, default=None)
    ap.add_argument("--max-reps", type=int, default=3)
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--gpus", default=None,
                    help="logical GPU indices to spread legs over (default: $UT_GPUS, then "
                         "$GPU_IDS); correctness shards cases, timing spreads buckets")
    ap.add_argument("--quick", action="store_true",
                    help="boundary cases of every phase, one draw, a few server calls, no "
                         "timing: for fix iterations. Writes run/ut_report.quick.json; never a "
                         "pass and never a receipt")
    ap.add_argument("--reuse", action="store_true",
                    help="reuse gates that passed in run/ut_report.json when nothing they "
                         "depend on changed (e.g. rerun only timing)")
    ap.add_argument("--inproc", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--phase", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--refs", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if not a.task:
        ap.error("--task is required")
    T = Task(a.task)
    spec = a.gpus if a.gpus is not None else (os.environ.get("UT_GPUS") or os.environ.get("GPU_IDS"))
    T.gpus = [g.strip() for g in str(spec).split(",") if g.strip()] if spec else None
    if a.inproc:
        for mod in T.meta.get("server_preimports") or []:
            importlib.import_module(mod)
        fn = {"isolation": inproc_isolation, "replay-ref": inproc_replay_ref,
              "replay": inproc_replay, "launches": inproc_launches}[a.inproc]
        print(json.dumps(fn(T, a), default=str))
        return 0
    a.draws = a.draws or int(T.meta.get("random_draws", 3))
    skip = {g for g in a.skip.split(",") if g}
    if a.quick:
        a.draws = 1
        skip.add("timing")
        _quick_subset(T)
    a.gates = [g for g in a.gates.split(",") if g and g not in skip]
    report, code = run_gates(T, a)
    report["snapshot_sha256"] = _snapshot(T)
    report["target_callable"] = T.meta.get("target_callable")
    report["gpus"] = T.gpus
    if a.quick:
        report["quick"] = {"cases": sorted(T.sig_filter or []), "server_calls": T.call_filter}
    out = a.out or os.path.join(T.dir, "run", "ut_report%s.json" % (".quick" if a.quick else ""))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
        f.write("\n")
    if a.quick:
        print("UT_VERDICT quick (exit 1) -> %s" % out, flush=True)
        return code
    # run/ut_result.json is the receipt the optimize stage reads (scripts/run_task_entry.py,
    # scripts/worker.sh): `started`, `ok`, `exit_code`, a dict per gate with `ok`, and the
    # timing aggregate under gates.timing.aggregate.primary.
    summary = {k: report[k] for k in ("started", "verdict", "exit_code", "snapshot_sha256",
                                      "target_callable", "family", "contract_sha256")}
    summary["ok"] = code == 0
    summary["gates"] = {g: {"ok": r["status"] in ("pass", "not_required"), "status": r["status"]}
                        for g, r in report["gates"].items()}
    summary["reused_gates"] = sorted(g for g, r in report["gates"].items() if r.get("reused"))
    timing = report["gates"].get("timing") or {}
    if "aggregate" in timing:
        summary["gates"]["timing"].update(aggregate=timing["aggregate"], per_case=timing["per_case"])
    with open(os.path.join(T.dir, "run", "ut_result.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
        f.write("\n")
    print("UT_VERDICT %s (exit %d) -> %s" % (report["verdict"], code, out), flush=True)
    # The one-line verdict worker.sh accepts: exit 0 = every gate passed, weighted = the
    # serving-weighted speedup (null when timing did not run).
    print(json.dumps({"exit": code, "weighted": timing.get("weighted_speedup")}), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
