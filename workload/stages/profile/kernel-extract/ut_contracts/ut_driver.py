#!/usr/bin/env python3
"""Generic kernel UT driver: every gate a delivered task runs, driven by its frozen contract.

A task's `unittest.py` is three lines that hand its directory to `main()`. Everything a
hand-written UT used to re-implement (500+ lines per task) lives here once:

  contract           ut_contract.lint_task: contract, case manifest and meta agree (CPU only)
  inputs             each random case builds reproducibly, honours its layout facts and ranges
  binding            harness_lib.build_candidate_overlay + assert_legs_differ
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

GATES = ("contract", "inputs", "binding", "correctness", "server_calls", "compile_variants",
         "isolation", "replay", "negative_controls", "restore", "timing")
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
        return [c for c in self.manifest["cases"] if c.get("random")]

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


def _oracle(T, overlay, path, seed, draws):
    T.h._run_leg(T.dir, overlay, "oracle", out=path, seed=seed, draws=draws)
    return T.U._torch().load(path, map_location="cpu")


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


class _Watch:
    """COMPILE_WATCH_OUT for the legs started inside the block (baseline legs only)."""

    def __init__(self, T, state, name):
        self.path = (os.path.join(state["scratch"], "variants_%s.json" % name)
                     if (T.meta.get("server_evidence") or {}).get("compile_variants") else None)
        self.state = state

    def __enter__(self):
        if self.path:
            os.environ["COMPILE_WATCH_OUT"] = self.path
        return self

    def __exit__(self, *exc):
        os.environ.pop("COMPILE_WATCH_OUT", None)
        if self.path and os.path.exists(self.path):
            self.state.setdefault("watch_files", []).append(self.path)


def gate_correctness(T, a, state):
    base, cand = T.h.build_candidate_overlay(T.dir, T.meta)
    seed, draws = a.seed, a.draws
    sd = state["scratch"]
    with _Watch(T, state, "oracle"):
        base_blob = _oracle(T, base, os.path.join(sd, "base.pt"), seed, draws)
    noise_blob = _oracle(T, base, os.path.join(sd, "base2.pt"), seed, draws) if _needs_noise(T) else None
    cand_blob = _oracle(T, cand, os.path.join(sd, "cand.pt"), seed, draws)
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


def _server_calls_leg(T, overlay, out):
    """One server_calls leg; a crash is attributed to the call it was running."""
    try:
        T.h._run_leg(T.dir, overlay, "server_calls", out=out, seed=0, timeout=7200)
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
    with _Watch(T, state, "server_calls"):
        b, why = _server_calls_leg(T, base, os.path.join(sd, "calls_base.pt"))
    if b is None:
        return {"status": "fail", "why": "baseline " + why}
    c, why = _server_calls_leg(T, cand, os.path.join(sd, "calls_cand.pt"))
    if c is None:
        return {"status": "fail", "why": "candidate " + why}
    tol = float(T.mc.get("server_call_tol", T.mc.get("replay_tol", DEFAULT_REPLAY_TOL)))
    ignore = T.mc.get("server_call_ignore") or []
    cw = _load("compile_watch", os.path.join(T.dir, "compile_watch.py"))
    rows, ok = [], True
    for call in doc["calls"]:
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
        with _Watch(T, state, "oracle_only"):
            _oracle(T, T.overlays()[0], os.path.join(state["scratch"], "base_cv.pt"), a.seed, 1)
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
    per = T.h.measure_legs(T.dir, T.meta, max_reps=a.max_reps)
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
    stop = None
    incomplete = False
    try:
        for g in wanted:
            if stop and not a.keep_going and g not in ("restore",):
                report["gates"][g] = {"status": "skipped", "why": "after %s" % stop}
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
    elif all(s in ("pass", "not_required") for s in statuses) and set(wanted) == set(GATES):
        verdict, code = "pass", 0
    elif all(s in ("pass", "not_required") for s in statuses):
        verdict, code = "partial", 1      # a subset of gates was asked for: never a full pass
    else:
        verdict, code = "fail", 1
    report.update(verdict=verdict, exit_code=code, seconds=round(time.time() - report["started"], 1))
    return report, code


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
    ap.add_argument("--inproc", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--phase", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--refs", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if not a.task:
        ap.error("--task is required")
    T = Task(a.task)
    if a.inproc:
        for mod in T.meta.get("server_preimports") or []:
            importlib.import_module(mod)
        fn = {"isolation": inproc_isolation, "replay-ref": inproc_replay_ref,
              "replay": inproc_replay}[a.inproc]
        print(json.dumps(fn(T, a), default=str))
        return 0
    a.draws = a.draws or int(T.meta.get("random_draws", 3))
    skip = {g for g in a.skip.split(",") if g}
    a.gates = [g for g in a.gates.split(",") if g and g not in skip]
    report, code = run_gates(T, a)
    report["snapshot_sha256"] = _snapshot(T)
    report["target_callable"] = T.meta.get("target_callable")
    out = a.out or os.path.join(T.dir, "run", "ut_report.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
        f.write("\n")
    # run/ut_result.json is the receipt the optimize stage reads (scripts/run_task_entry.py,
    # scripts/worker.sh): `started`, `ok`, `exit_code`, a dict per gate with `ok`, and the
    # timing aggregate under gates.timing.aggregate.primary.
    summary = {k: report[k] for k in ("started", "verdict", "exit_code", "snapshot_sha256",
                                      "target_callable", "family", "contract_sha256")}
    summary["ok"] = code == 0
    summary["gates"] = {g: {"ok": r["status"] in ("pass", "not_required"), "status": r["status"]}
                        for g, r in report["gates"].items()}
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
