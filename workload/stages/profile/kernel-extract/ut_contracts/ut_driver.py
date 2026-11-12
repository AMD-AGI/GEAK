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
                     and match it (meta.ut_contract.server_call_tol, default replay_tol).
                     meta.ut_contract.server_call_postprocess lets the task reduce both
                     legs' return the same way first, for an entry whose output is only
                     partly defined (a routed expert kernel's unwritten rows)
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

Extraction: --validation-mode correctness requires all non-timing gates and emits an
explicit kernel_ut_validation/v1 receipt with performance_measured=false. Performance
mode remains the bare-command default for optimizer callers and also requires timing.

"""

import argparse
from contextlib import contextmanager
import math
import selectors
import signal
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
         "compile_variants", "isolation", "process_state", "replay", "negative_controls",
         "restore", "timing")
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
        same, digests, args = T.U.check_reproducible(
            lambda g: T.cases.make_args(spec, gen=g), seed, device, return_inputs=True)
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


class OracleWorker:
    """An isolated, immutable-overlay process reused across bounded output batches."""
    def __init__(self, T, state, job, gpu):
        serial = state["oracle_worker_starts"] = state.get("oracle_worker_starts", 0) + 1
        env, self.watch = _watch_env(T, state, job.get("watch") and "worker_%d" % serial)
        env.update(T.h.gpu_env(gpu))
        env = dict(os.environ, **env)
        env["PYTHONPATH"] = os.pathsep.join([job["overlay"]] +
                                           ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        self.log_path = os.path.join(state["scratch"], "oracle_worker_%d.log" % serial)
        self.log = open(self.log_path, "wb")
        try:
            self.proc = subprocess.Popen([sys.executable, os.path.join(T.dir, "leg_runner.py"),
                "--task", T.dir, "--mode", "oracle-worker"], env=env, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=self.log, start_new_session=True)
        except BaseException:
            self.log.close()
            raise
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.proc.stdout, selectors.EVENT_READ)
        self.buffer = b""
        self.timeout = float(T.mc.get("oracle_timeout", 3600))
        self.failed = False

    def run(self, job, sigs, output):
        try:
            return self._run(job, sigs, output)
        except BaseException:
            self.failed = True
            raise

    def _run(self, job, sigs, output):
        request = dict(seed=job["seed"], draws=job["draws"], sigs=sigs, out=output)
        self.proc.stdin.write((json.dumps(request) + "\n").encode())
        self.proc.stdin.flush()
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            while b"\n" in self.buffer:
                line, self.buffer = self.buffer.split(b"\n", 1)
                if line.startswith(b"GEAK_ORACLE_BATCH "):
                    result = json.loads(line[len(b"GEAK_ORACLE_BATCH "):])
                    if result.get("out") != output:
                        raise RuntimeError("oracle worker returned a different batch")
                    self.cache_stats = result.get("weight_cache")
                    return
            ready = self.selector.select(max(0, min(1, deadline - time.monotonic())))
            if ready:
                chunk = os.read(self.proc.stdout.fileno(), 65536)
                if not chunk:
                    break
                self.buffer = (self.buffer + chunk)[-262144:]
            elif self.proc.poll() is not None:
                break
        with open(self.log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 3000))
            why = f.read().decode(errors="replace")
        raise RuntimeError("oracle worker failed or timed out: " + why)

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            pass
        finally:
            # A multi-rank leg can leave children after its leader exits.
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.proc.wait()
            self.proc.stdout.close()
            self.selector.close()
            self.log.close()
        return self.proc.returncode


@contextmanager
def oracle_workers(T, state):
    workers = {}
    state["_oracle_workers"] = workers
    try:
        yield
    finally:
        errors = []
        for worker in workers.values():
            try:
                code = worker.close()
                if code != 0 and not worker.failed:
                    errors.append("oracle worker exited %s after its batch" % code)
                _register_watch(state, worker.watch)
                if getattr(worker, "cache_stats", None):
                    state.setdefault("weight_cache_workers", []).append(worker.cache_stats)
            except Exception as exc:
                errors.append(str(exc))
        state.pop("_oracle_workers", None)
        if errors and sys.exc_info()[0] is None:
            raise RuntimeError("; ".join(errors))


def oracle_output_bytes(T, spec, draws=1, copies=3):
    try:
        sym = T.U.symbols_for(T.doc, spec.get("axes") or {})
        size = sum(math.prod(T.U.shape_of(o, sym)) *
                   T.U._torch().empty((), dtype=T.U.torch_dtype(T.U.dtype_of(o, sym))).element_size()
                   for o in T.doc["outputs"].values() if T.U.present(o, sym))
        return max(1, size) * draws * copies
    except (KeyError, ValueError, TypeError):
        return int(T.mc.get("oracle_batch_bytes", 256 << 20))


def oracle_batches(T, draws=1, copies=3):
    """Bound output accumulation by case count and estimated serialized tensor bytes.

    A single oversized case remains whole (the numerical comparison is not sampled).
    Its size is reported explicitly; this is an output budget, not a total RAM limit.
    """
    budget = int(T.mc.get("oracle_batch_bytes", 256 << 20))
    limit = int(T.mc.get("oracle_batch_cases", 32))
    if budget < 1 or limit < 1:
        raise ValueError("oracle batch budgets must be positive")
    group, used = [], 0
    for spec in T.random_specs():
        size = oracle_output_bytes(T, spec, draws, copies)
        if group and (used + size > budget or len(group) >= limit):
            yield group, used
            group, used = [], 0
        group.append(spec["sig"])
        used += size
    if group:
        yield group, used


@contextmanager
def case_batch(T, sigs):
    previous = T.sig_filter
    T.sig_filter = set(sigs)
    try:
        yield
    finally:
        T.sig_filter = previous


def _oracles(T, state, jobs):
    """Oracle legs -> one blob per job. A job is {overlay, path, seed, draws, watch}.

    With several GPUs (--gpus / GPU_IDS) every job is split into one shard of random cases
    per GPU and all shards of all jobs run concurrently. Draws are seeded per case, so a
    sharded blob is the unsharded blob."""
    torch = T.U._torch()
    batch_id = state["oracle_batches"] = state.get("oracle_batches", 0) + 1
    sigs = [c["sig"] for c in T.random_specs()]
    gpus = T.gpus or [None]
    k = max(1, min(len(gpus), len(sigs)))
    shards = [sigs[i::k] for i in range(k)]
    items = [(j, i) for j in range(len(jobs)) for i in range(k)]

    def leg(item, gpu):
        j, i = item
        job = jobs[j]
        out = "%s.s%d" % (job["path"], i)
        workers = state.get("_oracle_workers")
        if workers is not None:
            key = (job["overlay"], job["path"], gpu)
            if key not in workers:
                workers[key] = OracleWorker(T, state, job, gpu)
            workers[key].run(job, shards[i], out)
        else:
            env, wpath = _watch_env(T, state, job.get("watch") and "%s.b%d.s%d" % (job["watch"], batch_id, i))
            env.update(T.h.gpu_env(gpu))
            T.h._run_leg(T.dir, job["overlay"], "oracle", out=out, seed=job["seed"],
                         draws=job["draws"], env_extra=env,
                         sigs=shards[i] if (k > 1 or T.sig_filter is not None) else None)
            _register_watch(state, wpath)
        return out
    paths = _parallel(items, gpus, leg)
    blobs = [{} for _ in jobs]
    total = sum(os.path.getsize(path) for path in paths)
    state["peak_oracle_files_bytes"] = max(state.get("peak_oracle_files_bytes", 0), total)
    try:
        for (j, i), path in zip(items, paths):
            blob = torch.load(path, map_location="cpu")
            expected = {"%s|%d" % (sig, n) for sig in shards[i] for n in range(jobs[j]["draws"])}
            if set(blob) != expected:
                raise RuntimeError("oracle batch is missing or added case/draw outputs")
            blobs[j].update(blob)
    finally:
        for path in paths:
            if os.path.exists(path):
                os.unlink(path)
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
            args = T.cases.make_args(spec, gen=_gen(torch, device, seed + i), with_logical=bool(T.doc.get("reference")))
            refs, mags, sel = _references(T, args)
            score_inputs = {o["bound"].get("scores") for o in T.doc["outputs"].values()
                            if o["bound"]["kind"] == "topk_equivalent"}
            inputs = {k: _cpu(args[k]) for k in score_inputs if k in args}
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
    rows, flaky, ok, batches, oversized = [], [], True, 0, []
    bitwise = T.doc["compare"].get("determinism") == "bitwise"
    copies = 2 + int(_needs_noise(T)) + int(bitwise)
    with oracle_workers(T, state):
        for sigs, estimated in oracle_batches(T, draws, copies):
            batches += 1
            if estimated > int(T.mc.get("oracle_batch_bytes", 256 << 20)):
                oversized.extend(sigs)
            with case_batch(T, sigs):
                jobs = [{"overlay": base, "path": os.path.join(sd, "base.pt"), "seed": seed,
                         "draws": draws, "watch": "oracle"},
                        {"overlay": cand, "path": os.path.join(sd, "cand.pt"), "seed": seed, "draws": draws}]
                if _needs_noise(T):
                    jobs.append({"overlay": base, "path": os.path.join(sd, "base2.pt"), "seed": seed,
                                 "draws": draws})
                blobs = _oracles(T, state, jobs)
                base_blob, cand_blob = blobs[:2]
                noise_blob = blobs[2] if _needs_noise(T) else None
                if bitwise:
                    again = _oracle(T, cand, os.path.join(sd, "cand2.pt"), seed, draws, state)
                    flaky.extend(k for k in cand_blob if
                                 T.U.tensor_digest(cand_blob[k]) != T.U.tensor_digest(again.get(k)))
                    del again
                good, batch_rows = compare_blobs(T, cand_blob, base_blob, seed, draws, noise_blob)
                rows.extend(batch_rows)
                ok = ok and good
                del blobs, base_blob, cand_blob, noise_blob
    out = {"cases": rows, "batches": batches, "oversized_single_cases": oversized,
           "output_budget_bytes": int(T.mc.get("oracle_batch_bytes", 256 << 20))}
    if bitwise:
        out["bitwise_repeat"] = {"ok": not flaky, "differing": flaky}
    return {"status": "pass" if ok and rows and not flaky else "fail", "seed": seed, "draws": draws, **out}


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
            # An unavailable baseline call is missing qualification, even if a generated
            # case happened to compile the same variant. Preserve that distinction instead
            # of calling an empty or partial recorded-call replay a pass.
            row.update(ok=False, why="baseline did not replay it: %s" %
                       (rb.get("why") or rb.get("status") or "missing result"))
        elif rc.get("status") != "ok":
            row.update(ok=False, why="candidate: %s" % (rc.get("why") or rc.get("status")))
        else:
            good, outs = cw.compare_call_outputs(rb["outputs"], rc["outputs"], tol, ignore)
            row.update(ok=good, outputs=[o for o in outs if not o["ok"]] or len(outs))
            if rb.get("postprocessed") and rc.get("postprocessed"):
                row["postprocessed"] = True
        ok = ok and row["ok"]
        rows.append(row)
    replayed = sum(r["baseline"] == "ok" for r in rows)
    res = {"status": "pass" if ok and rows else "fail", "tol": tol, "calls": len(rows),
            "replayed_by_baseline": replayed,
            "replayed_by_candidate": sum(r["candidate"] == "ok" for r in rows),
            "unreplayed_baseline_calls": len(rows) - replayed, "rows": rows,
            **({"why": "no recorded calls were selected for replay"} if not rows else {})}
    if T.mc.get("server_call_postprocess"):
        # Both legs reduced their return the same way before the compare. Name it here, so
        # a reader of the report sees what was compared without opening cases.py.
        res["postprocess"] = {"hook": "cases.server_call_postprocess",
                              "why": T.mc.get("server_call_postprocess_why"),
                              "applied_to": sum(1 for r in rows if r.get("postprocessed"))}
    return res


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
        for sigs, _ in oracle_batches(T, copies=1):
            with case_batch(T, sigs):
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


def state_changes(snap):
    """{key: (before, after)} for every process-wide value a leg changed while it ran."""
    before, after = snap.get("before") or {}, snap.get("after") or {}
    return {k: (before.get(k), after.get(k)) for k in set(before) | set(after)
            if before.get(k) != after.get(k)}


def candidate_only_changes(base_snap, cand_snap, allow=()):
    """What the candidate leg changed that the baseline leg did not change the same way.

    Both legs import the same serving stack, so whatever the stack itself does to process
    state cancels; what remains is the candidate's. Callable identities are compared by
    whether they were rebound, not by address, which differs between processes."""
    base, cand = state_changes(base_snap), state_changes(cand_snap)
    out, allowed = {}, {}
    for key, (b, c) in sorted(cand.items()):
        if key.startswith("callable:"):
            if key in base:
                continue
        elif base.get(key) == (b, c):
            continue
        why = next((r for p, r in allow if key.startswith(p)), None)
        (allowed if why else out)[key] = {"before": b, "after": c, **({"allowed": why} if why else {})}
    return out, allowed


def gate_process_state(T, a, state):
    """The candidate must not change process-wide state the baseline leaves alone.

    A UT runs the op by itself, so a kernel that flips a process-wide switch (TunableOp, a
    BLAS backend, a precision flag, an environment variable) or rebinds a shared torch
    callable passes every other gate, and in a server silently changes how every other op
    runs. Each leg runs each phase's smallest and largest case once with a snapshot before
    and after (leg_runner._process_state). A deliberate change is declared in
    ut_contract.process_state_allow: [{"prefix": "<key prefix>", "why": "..."}]."""
    base, cand = T.h.build_candidate_overlay(T.dir, T.meta)
    specs = T.random_specs()
    if not specs:
        return {"status": "not_required", "why": "no random cases to execute"}
    by_phase = {}
    for s in specs:
        by_phase.setdefault(s.get("regime", ""), []).append(s)
    sigs = []
    for rows in by_phase.values():
        rows = sorted(rows, key=lambda s: sorted((s.get("axes") or {}).items()))
        sigs += [r["sig"] for r in (rows[0], rows[-1])]
    sigs = list(dict.fromkeys(sigs))
    allow = [(d.get("prefix", ""), d.get("why") or "declared") for d in
             (T.mc.get("process_state_allow") or []) if isinstance(d, dict) and d.get("prefix")]
    snaps = {}
    gpu = (T.gpus or [None])[0]
    for name, overlay in (("baseline", base), ("candidate", cand)):
        path = os.path.join(state["scratch"], "process_state.%s.json" % name)
        env = dict(T.h.gpu_env(gpu), PROCESS_STATE_OUT=path)
        T.h._run_leg(T.dir, overlay, "oracle", out=os.path.join(state["scratch"], name + ".ps.pt"),
                     seed=a.seed, draws=1, env_extra=env, sigs=sigs)
        with open(path) as fh:
            snaps[name] = json.load(fh)
    changed, allowed = candidate_only_changes(snaps["baseline"], snaps["candidate"], allow)
    return {"status": "fail" if changed else "pass", "cases": sigs,
            "candidate_only_changes": changed, "allowed": allowed,
            "baseline_changes": sorted(state_changes(snaps["baseline"]))[:50],
            **({"why": "the candidate changes process-wide state the baseline does not: %s. "
                       "In a server every other op runs under it, which no UT case exercises."
                       % ", ".join(sorted(changed)[:8])} if changed else {})}


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


def negative_control_batches(T, control):
    """Probe likely witnesses first, then every remaining eligible case if needed.

    Author-provided probe_sigs name known leaf/branch cases. Without a hint, try
    the smallest estimated output first. Hints order the search; they never narrow
    its scope, and a quick run cannot expand beyond its own selected cases.
    """
    specs = T.random_specs()
    eligible = {s["sig"]: s for s in specs}
    known = {s["sig"] for s in T.manifest["cases"] if s.get("random")}
    probes = control.get("probe_sigs", [])
    if (not isinstance(probes, list) or any(not isinstance(s, str) for s in probes)
            or len(set(probes)) != len(probes) or set(probes) - known):
        raise ValueError("negative-control probe_sigs must be unique random-case signatures")
    probes = [s for s in probes if s in eligible]
    if not probes and specs:
        probes = [min(specs, key=lambda s: oracle_output_bytes(T, s))["sig"]]
    batches = [([sig], oracle_output_bytes(T, eligible[sig])) for sig in probes]
    with case_batch(T, set(eligible) - set(probes)):
        batches.extend(oracle_batches(T))
    return batches


def numerical_control_outputs(T, comparison):
    # Missing outputs and shape errors are not numerical rejections. In particular,
    # control_margin's generic exact-bound fallback must not turn them into infinity.
    metrics = {"peak_bound_multiple", "sqnr_db", "mismatches", "mismatched_bytes",
               "bad_rows", "max_rel_err", "nonfinite_mismatches"}
    return {name: result for name, result in comparison.items()
            if name in T.doc["outputs"] and isinstance(result, dict) and result.get("ok") is False
            and (bool(metrics.intersection(result)) or
                 result.get("kind") == "harness" and result.get("why") == "non-finite mismatch")}


def gate_negative_controls(T, a, state):
    controls = T.mc.get("negative_controls") or []
    if not controls:
        return {"status": "fail", "why": "meta.ut_contract.negative_controls is empty"}
    margin_needed = float(T.mc.get("min_control_margin", DEFAULT_CONTROL_MARGIN))
    rows = []
    for i, c in enumerate(controls):
        row = {"leaf": c["leaf"], "expect": c.get("expect"), "caught": False,
               "margin": 0.0, "cases": [], "restored": True,
               "available_cases": len(T.random_specs()), "checked_cases": 0,
               "early_stopped": False, "witness_case": None}
        rows.append(row)
        try:
            batches = negative_control_batches(T, c)
        except ValueError as exc:
            row.update(ok=False, why=str(exc))
            continue
        path = os.path.join(T.dir, c["leaf"])
        with open(path, "rb") as f:
            pristine = f.read()
        text = pristine.decode("utf-8", errors="surrogateescape")
        if text.count(c["find"]) != 1:
            row.update(ok=False, why="`find` occurs %d times" % text.count(c["find"]))
            continue
        try:
            with open(path, "wb") as f:
                f.write(text.replace(c["find"], c["replace"]).encode("utf-8", errors="surrogateescape"))
            base, cand = T.h.build_candidate_overlay(T.dir, T.meta)
            # A control's overlay stays unchanged until all its workers have stopped.
            # Baselines are regenerated in bounded batches, never retained suite-wide.
            with oracle_workers(T, state):
                for sigs, _ in batches:
                    with case_batch(T, sigs):
                        jobs = [{"overlay": base, "path": os.path.join(state["scratch"], "base.pt"),
                                 "seed": a.seed, "draws": 1},
                                {"overlay": cand, "path": os.path.join(state["scratch"], "control.pt"),
                                 "seed": a.seed, "draws": 1}]
                        if _needs_noise(T):
                            jobs.append({"overlay": base, "path": os.path.join(state["scratch"], "base2.pt"),
                                         "seed": a.seed, "draws": 1})
                        try:
                            blobs = _oracles(T, state, jobs)
                        except Exception as exc:
                            row["why"] = "control legs did not run to completion (%s); not a numerical control" % str(exc)[-300:]
                            break
                        _, comparisons = compare_blobs(T, blobs[1], blobs[0], a.seed, 1,
                                                       blobs[2] if _needs_noise(T) else None, only_draw=0)
                        for result in comparisons:
                            numerical = numerical_control_outputs(T, result)
                            if not result["ok"] and numerical:
                                row["caught"] = True
                                margin = T.U.control_margin({"outputs": numerical})
                                row["margin"] = max(row["margin"], margin)
                                if margin >= margin_needed and row["witness_case"] is None:
                                    row["witness_case"] = result["case"]
                            row["cases"].append({"case": result["case"], "ok": result["ok"]})
                        row["checked_cases"] = len(row["cases"])
                        del blobs
                        if row["witness_case"] is not None:
                            row["early_stopped"] = row["checked_cases"] < row["available_cases"]
                            break
        finally:
            with open(path, "wb") as f:
                f.write(pristine)
            row["restored"] = _sha(path) == hashlib.sha256(pristine).hexdigest()
            T.h.build_candidate_overlay(T.dir, T.meta)
        row["ok"] = (row["caught"] and row["margin"] >= margin_needed and row["restored"]
                     and "why" not in row)
        if not row["ok"] and "why" not in row:
            row["why"] = "mutation not detected at the required numerical margin"
    return {"status": "pass" if all(r["ok"] for r in rows) else "fail",
            "min_margin": margin_needed, "controls": rows}


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
        same, changed, first_output = T.U.inputs_unmodified(T.cases.call, args)
        other = T.cases.make_args(spec, gen=_gen(torch, device, a.seed + 1))
        ind_ok, why = T.h.assert_independent_outputs(T.cases.call, args, other, first_output=first_output)
        rows.append({"sig": spec["sig"], "inputs_unchanged": same, "changed": changed,
                     "fresh_outputs": ind_ok, "why": why})
        ok = ok and same and ind_ok
        del args, other, first_output
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
    mode = getattr(a, "validation_mode", "performance")
    required = [g for g in GATES if mode != "correctness" or g != "timing"]
    wanted = [g for g in required if g in a.gates]
    report = {"task": T.dir, "family": T.doc["family"],
              "contract_sha256": T.U.contract_sha256(T.doc), "started": time.time(), "gates": {},
              "validation_contract": "kernel_ut_validation/v1", "validation_mode": mode,
              "required_gates": required}
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
    elif all(s in ("pass", "not_required") for s in statuses) and set(wanted) == set(required):
        verdict, code = "pass", 0
    elif all(s in ("pass", "not_required") for s in statuses):
        verdict, code = "partial", 1      # a subset of gates was asked for: never a full pass
    else:
        verdict, code = "fail", 1
    report.update(verdict=verdict, exit_code=code, seconds=round(time.time() - report["started"], 1),
                  performance_measured=report["gates"].get("timing", {}).get("status") == "pass",
                  data_handling={"oracle_batches": state.get("oracle_batches", 0),
                                 "oracle_worker_starts": state.get("oracle_worker_starts", 0),
                                 "weight_cache_workers": state.get("weight_cache_workers", []),
                                 "peak_oracle_files_bytes": state.get("peak_oracle_files_bytes", 0)})
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
    ap.add_argument("--validation-mode", choices=("correctness", "performance"), default="performance",
                    help="correctness qualifies the complete UT without timing; performance also "
                         "requires measured timings (default, used by optimization)")
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
                                      "target_callable", "family", "contract_sha256",
                                      "validation_contract", "validation_mode", "required_gates",
                                      "performance_measured")}
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
