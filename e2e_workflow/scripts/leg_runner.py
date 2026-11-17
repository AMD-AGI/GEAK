#!/usr/bin/env python3
"""Run ONE measurement leg of a kernel task. WHICH leg is decided ONLY by the overlay on PYTHONPATH.

  baseline leg   PYTHONPATH=<task>/baseline_overlay    -> the REAL online serving stack (install +
                 every already-accepted kernel). Nothing in the task dir can change what it imports.
  candidate leg  PYTHONPATH=<task>/_cand_overlay       -> that SAME stack + exactly ONE entry built
                 from kernel_src/ (meta.candidate_bind).

Both legs execute THIS file and the task's OWN cases.py, so `speedup = baseline_ms / candidate_ms` is
always measured against the live path, whatever LANGUAGE the candidate is written in.

Modes: list (bucket sigs) | resolve (leg identity, for the direction assert) | time | oracle
       | refs (baseline correctness worker) | correct (candidate correctness worker)
       | serve (one rank of a multi-rank pool; see ut_runner.RankPool).
Driven by harness_lib.measure_legs and ut_runner.py — a unittest.py should not invoke it directly.

Timing (`time`) follows the deployment of each case's phase: a graph-deployed phase is timed as a graph
replay; in a generated task an eager-deployed phase is timed eagerly on a queued GPU spin (A16,
harness_lib.time_op `spin`), so its reading is GPU execution time. Old hand-written tasks are unchanged.

Correctness runs in THREE processes (ut_runner.py): `oracle` (baseline) records the random-draw
outputs; `refs` (baseline, another process) runs the full check sequence -- eager, random draws,
graph replay, sequence -- records the native output of every comparison and reports the native
run-to-run noise; `correct` (candidate) runs the same sequence and is judged against those recorded
native outputs. The `ref` a cases.py hook returns is therefore never what the candidate is judged
against (it is only checked on the native side), and the candidate never produces its own reference.
"""
import argparse
import importlib
import importlib.util
import json
import math
import os
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
# this file lives IN the task dir next to a file named unittest.py — drop it from sys.path before
# torch is imported, or that file shadows stdlib unittest.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]

GENERATED_MARKER = "# GEAK-GENERATED unittest.py"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _resolve(dotted_attr):
    mod_name, _, attr = dotted_attr.partition(":")
    obj = importlib.import_module(mod_name)
    for part in attr.split("."):
        if part:
            obj = getattr(obj, part)
    return obj


def _identity(target_callable):
    """Where this leg's callable ACTUALLY came from. The two legs must not report the same tuple."""
    try:
        fn = _resolve(target_callable)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    mod = getattr(fn, "__module__", "") or ""
    f = ""
    try:
        f = getattr(sys.modules.get(mod), "__file__", "") or ""
    except Exception:
        pass
    if not f:
        try:
            f = getattr(importlib.import_module(target_callable.split(":")[0]), "__file__", "") or ""
        except Exception:
            pass
    return {"module": mod, "file": os.path.realpath(f) if f else "",
            "qualname": getattr(fn, "__qualname__", repr(fn))}


def _snapshot(out):
    """Detach one op's output for the oracle blob.

    Attention entries commonly return `(out, lse)` and some return a dict, so a bare `out.detach()`
    raises AttributeError and — through `_run_leg` — surfaces as a generic non-zero-exit RuntimeError
    instead of naming the shape of the return value. Recurse over the container instead. Containers
    are matched BEFORE the tensor duck-check: a namedtuple carries both."""
    if isinstance(out, tuple) and hasattr(out, "_fields"):
        return type(out)(*(_snapshot(o) for o in out))
    if isinstance(out, (tuple, list)):
        return type(out)(_snapshot(o) for o in out)
    if isinstance(out, dict):
        return {k: _snapshot(v) for k, v in out.items()}
    if hasattr(out, "detach"):
        return out.detach().clone().cpu()
    if out is None or isinstance(out, (int, float, bool, str)):
        return out
    raise TypeError(
        f"oracle cannot record a {type(out).__name__} returned by the op — cases.call must return a "
        "tensor, or a tuple/list/dict of tensors")


def is_generated_task(task):
    """A task whose unittest.py was written by generate_ut.py (new contract; see ut_runner.py)."""
    try:
        with open(os.path.join(task, "unittest.py"), encoding="utf-8") as fh:
            return fh.readline().startswith(GENERATED_MARKER)
    except OSError:
        return False


def task_tol(meta):
    """meta.tolerance.tol (or meta.tol): the comparison tolerance, a finite positive number."""
    tol = (meta.get("tolerance") or {}).get("tol") if isinstance(meta.get("tolerance"), dict) else None
    if tol is None:
        tol = meta.get("tol")
    tol = float(tol)
    if not (tol == tol) or tol < 0 or tol == float("inf"):
        raise ValueError(f"meta tolerance is not a finite non-negative number: {tol!r}")
    return tol


# --------------------------------------------------------------------------- per-output tolerances (A42)
class TaskTol(float):
    """The tolerance of a generated UT's comparisons when meta.tolerance.per_output is given (A42).

    The float is meta.tolerance.tol: every check and control receives it as before, and an output per_output does
    not name is compared at it. per_output_correct, installed as h.correct while the correctness suite runs, makes a
    comparison at this tolerance output by output, each with correct() at its own tolerance. The outputs are those of
    the reference side, which is native on both legs (the case's own ref, the native oracle, the recorded native
    output); on the native refs leg (`native`) per_output must fit them -- a position within the outputs, a key the
    compared dict has, positions for a non-dict -- and every comparison must compare the outputs the first one did,
    also where an output is only recorded; anything else is a problem of the UT (`problems`: incomplete, exit 3). On
    the candidate leg the recorded native outputs fit, and a candidate output that does not pair up with them fails,
    as in correct()."""

    def __new__(cls, tol, per_output, *, h, native=False, problems=()):
        self = super().__new__(cls, tol)
        self.per_output = dict(per_output or {})
        self.h, self.native = h, native
        self.problems = list(problems)
        self.outputs = None         # [(output, tolerance)] of the first reference fitted
        self.noise = {}             # output -> noise_rel of each random-draw comparison (native refs leg, A37)
        return self

    def __reduce__(self):
        return (float, (float(self),))      # a copy or a pickle is the plain tolerance

    def layout(self, obj):
        """([(output, tolerance)] of every tensor flatten_outputs(obj) yields, in its order, and the problems of
        fitting per_output to obj). An output is its position ("0", "1", ...) or, when obj is a dict, its key
        ("<key>[<j>]" when the key holds several tensors)."""
        default, h = float(self), self.h
        positions = [k for k in self.per_output if h.is_output_position(k)]
        names = [k for k in self.per_output if not h.is_output_position(k)]
        problems, rows = [], []
        if isinstance(obj, dict):
            keys = sorted(obj)                       # flatten_outputs order
            have = [str(k) for k in keys]
            if positions:
                problems.append(f"the compared output is a dict (keys {have}): name its outputs by key, not by "
                                f"position {positions}")
            missing = [k for k in names if k not in have]
            if missing:
                problems.append(f"per_output names {missing}, which the compared dict (keys {have}) does not have")
            for k in keys:
                n = len(h.flatten_outputs(obj[k]))
                t = self.per_output.get(str(k), default)
                rows += [(str(k) if n == 1 else f"{k}[{j}]", t) for j in range(n)]
        else:
            n = len(h.flatten_outputs(obj))
            if names:
                problems.append(f"the compared output is not a dict but {n} tensor(s): name its outputs by "
                                f"position (\"0\" .. \"{max(n - 1, 0)}\"), not by key {names}")
            over = [k for k in positions if int(k) >= n]
            if over:
                problems.append(f"per_output names the positions {over}, but the compared output has {n} tensor(s)")
            rows = [(str(i), self.per_output.get(str(i), default)) for i in range(n)]
        return rows, problems

    def fit(self, obj):
        """[(output, tolerance)] of a reference output, or None when per_output does not fit it or it is not the
        layout of the first reference fitted; on the native refs leg that is a problem of the UT."""
        rows, problems = self.layout(obj)
        if not problems:
            if self.outputs is None:
                self.outputs = rows
            elif [r[0] for r in rows] != [r[0] for r in self.outputs]:
                problems = [f"one comparison compares the outputs {[r[0] for r in rows]}, an earlier one "
                            f"{[r[0] for r in self.outputs]}: per-output tolerances need every comparison of the "
                            "task to compare the same outputs"]
        if problems:
            if self.native:
                for msg in problems:
                    msg = f"meta.tolerance.per_output: {msg}"
                    if msg not in self.problems:
                        self.problems.append(msg)
            return None
        return rows

    def record_noise(self, out, ref, measure):
        """noise_rel of every output of one random-draw comparison of the native refs leg (A37, per output)."""
        rows = self.fit(ref)
        po, pr = self.h.flatten_outputs(out), self.h.flatten_outputs(ref)
        if rows is None or len(po) != len(pr):
            return
        for (output, _), o, r in zip(rows, po, pr):
            self.noise.setdefault(output, []).append(measure(o, r))

    def noise_record(self):
        """native_noise.per_output: {output: {tol, noise_rel (the worst finite one), noise_rel_nonfinite}}."""
        out = {}
        for output, t in self.outputs or []:
            vals = self.noise.get(output, [])
            finite = [float(v) for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]
            rec = {"tol": t, "noise_rel": max(finite) if finite else None}
            if len(finite) != len(vals):
                rec["noise_rel_nonfinite"] = len(vals) - len(finite)
            out[output] = rec
        return out

    def record(self):
        """`tolerance` of the worker result and of run/ut_result.json: the declared tolerances and the one every
        output was compared at."""
        return {"tol": float(self), "per_output": dict(self.per_output),
                "outputs": [{"output": o, "tol": t} for o, t in self.outputs or []]}


class OutputErr(float):
    """correct()'s err of one output of a per-output comparison (A42). harness_lib's checks copy `row_fields` -- the
    output and its tolerance -- into the comparison's row."""

    def __new__(cls, err, output, tol):
        self = super().__new__(cls, err)
        self.row_fields = {"output": output, "tol": tol}
        return self

    def __reduce__(self):
        return (float, (float(self),))


def _tol_share(err, tol):
    """err / tol of one output: how far over (> 1) or under its own tolerance it is; an err that is not a finite
    number is infinitely over (per_output_correct reads a passing output's as 0.0 first, so only a failing output's
    gets here)."""
    if not math.isfinite(err):
        return math.inf
    if tol > 0:
        return err / tol
    return math.inf if err > 0 else 0.0


def per_output_correct(h, correct):
    """h.correct while the correctness suite of a task with per-output tolerances runs (A42). A comparison at the
    task's TaskTol is made output by output, each with `correct` (harness_lib.correct) at that output's own
    tolerance: it passes when every output passes; nothing to compare, or a tensor count that differs, fails as in
    correct(). Its err, which the row records with that output and tolerance, is correct()'s err of the output with
    the largest err / tolerance: among the outputs that failed when the comparison fails -- a passing output reads up
    to 1 on a reference element near 0, far over a small tolerance, and is never named for a failure -- else among
    all. A passing output's err that is not a finite number reads 0.0: correct() reads 0/0 = NaN only at tolerance 0
    on a reference element of 0, and passing at tolerance 0 is being equal element for element. Any other call is
    `correct` itself."""
    def compare(out, ref, tol):
        if not isinstance(tol, TaskTol):
            return correct(out, ref, tol)
        if ref is None:
            return False, float("inf")
        rows = tol.fit(ref)
        po, pr = h.flatten_outputs(out), h.flatten_outputs(ref)
        if rows is None or not po or len(po) != len(pr):
            return False, float("inf")
        ok_all, worst, worst_failed = True, None, None
        for (output, t), o, r in zip(rows, po, pr):
            ok, err = correct(o, r, t)
            ok_all = ok_all and ok
            if ok and not math.isfinite(err):
                err = 0.0
            seen = (_tol_share(err, t), err, output, t)
            if worst is None or seen[0] > worst[0]:
                worst = seen
            if not ok and (worst_failed is None or seen[0] > worst_failed[0]):
                worst_failed = seen
        _share, err, output, t = worst if ok_all else worst_failed
        return ok_all, OutputErr(err, output, t)
    return compare


def task_tolerance(meta, h, native=False):
    """What the correctness suite compares at: task_tol(meta), a plain float -- correct() at one tolerance, exactly as
    before -- or, with meta.tolerance.per_output (A42), a TaskTol (a malformed per_output: its problems)."""
    tol = task_tol(meta)
    per, problems = h.per_output_tolerances(meta) if hasattr(h, "per_output_tolerances") else (None, [])
    if per is None and not problems:
        return tol
    return TaskTol(tol, per, h=h, native=native, problems=problems)


# --------------------------------------------------------------------------- native references
def _to_device(obj, device):
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        return type(obj)(*(_to_device(o, device) for o in obj))
    if isinstance(obj, (tuple, list)):
        return type(obj)(_to_device(o, device) for o in obj)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    return obj.to(device) if hasattr(obj, "to") else obj


class RefBook:
    """Native outputs of the eager / graph-replay / sequence comparisons, in check order.

    `wrap(kind, check)` returns `check` with `h.correct` swapped for the duration of the call.
    Recording (native `refs` leg): every compared output is kept, and checked against the case's own
    `ref` when it has one. Replaying (candidate `correct` leg): the n-th comparison of `kind` is made
    against the n-th recorded native output, whatever `ref` the case carries. Both legs run the same
    cases.py through the same sequence of checks, so the n-th comparisons line up; a count that
    differs is a failure."""

    KINDS = ("eager", "graph_replay", "sequence")

    def __init__(self, h, recorded=None):
        self.h = h
        self.recording = recorded is None
        self.recorded = {k: list((recorded or {}).get(k, [])) for k in self.KINDS}
        self.cursor = {k: 0 for k in self.KINDS}

    def wrap(self, kind, check):
        h, book = self.h, self

        def correct_against_native(out, ref, tol, _orig):
            i = book.cursor[kind]
            book.cursor[kind] += 1
            if book.recording:
                book.recorded[kind].append(_snapshot(out))
                if ref is not None:
                    return _orig(out, ref, tol)
                if isinstance(tol, TaskTol):
                    tol.fit(out)        # A42: per_output must fit the native output also where nothing is compared
                return True, 0.0
            native = book.recorded[kind]
            if i >= len(native):
                return False, float("inf")
            dev = next((t.device for t in h.flatten_outputs(out) if hasattr(t, "device")), "cpu")
            return _orig(out, _to_device(native[i], dev), tol)

        def wrapped(*args, **kwargs):
            orig = h.correct
            h.correct = lambda out, ref, tol: correct_against_native(out, ref, tol, orig)
            try:
                return check(*args, **kwargs)
            finally:
                h.correct = orig

        return wrapped

    def mismatches(self):
        if self.recording:
            return []
        return [f"{k}: candidate made {self.cursor[k]} comparisons, the native leg recorded "
                f"{len(self.recorded[k])}" for k in self.KINDS if self.cursor[k] != len(self.recorded[k])]

    def dump(self):
        return dict(self.recorded)


# --------------------------------------------------------------------------- controls (verify_ut)
def _map(out, f_float, f_other=None):
    """Apply `f_float` to every floating tensor of an output (and `f_other` to the rest, if given)."""
    if isinstance(out, tuple) and hasattr(out, "_fields"):
        return type(out)(*(_map(o, f_float, f_other) for o in out))
    if isinstance(out, (list, tuple)):
        return type(out)(_map(o, f_float, f_other) for o in out)
    if isinstance(out, dict):
        return {k: _map(v, f_float, f_other) for k, v in out.items()}
    if hasattr(out, "is_floating_point") and hasattr(out, "shape"):
        if out.is_floating_point():
            return f_float(out)
        return f_other(out) if f_other is not None else out
    return out


def _map_flat(out, f):
    """`f(tensor, i)` on every tensor of an output, i its position in harness_lib.flatten_outputs order (a dict by
    sorted key), keeping the output's structure (A42: per-output controls)."""
    count = [0]

    def walk(x):
        if isinstance(x, tuple) and hasattr(x, "_fields"):
            return type(x)(*(walk(o) for o in x))
        if isinstance(x, (list, tuple)):
            return type(x)([walk(o) for o in x])
        if isinstance(x, dict):
            done = {k: walk(x[k]) for k in sorted(x)}
            return {k: done[k] for k in x}
        if hasattr(x, "shape"):
            i = count[0]
            count[0] += 1
            return f(x, i) if hasattr(x, "is_floating_point") else x
        return x
    return walk(out)


def _off_by_much(t):
    """A wrong value for a non-float output: bool flips, integers move by more than their range."""
    torch = sys.modules.get("torch") or importlib.import_module("torch")
    if t.dtype == torch.bool:
        return ~t
    if t.numel() == 0:
        return t
    return t + (t.abs().max() + 1).to(t.dtype)


def _tensor_args(args, kwargs=None):
    found = []

    def walk(x):
        if hasattr(x, "is_floating_point") and hasattr(x, "data_ptr"):
            found.append(x)
        elif isinstance(x, (list, tuple)):
            for y in x:
                walk(y)
        elif isinstance(x, dict):
            for y in x.values():
                walk(y)
        elif hasattr(x, "__dict__") and not isinstance(x, type):
            for y in vars(x).values():
                if hasattr(y, "is_floating_point") and hasattr(y, "data_ptr"):
                    found.append(y)
    walk(args)
    if kwargs:
        walk(kwargs)
    seen, out = set(), []
    for t in found:
        if id(t) not in seen:
            seen.add(id(t))
            out.append(t)
    return out


class Control:
    """A wrapper around `cases.call` and the replay bundle's `read_out`, applied INSIDE the worker.

    The control acts on what the UT compares -- the entry's complete output -- so it works the same
    for module and rebind bindings, and nothing after it can cancel it (ISSUE-033: a scale applied
    inside forward was undone by a later norm). Generic controls (verify_ut must see each one):
      noop            wrapper only                              must PASS
      within_tol      float outputs * (1 + tol/10)              must PASS
      scale_out       float outputs * (1 + 4 tol); non-float outputs moved off their values   must FAIL
                      (with per-output tolerances, A42: each output by its own tolerance)
      stale_output    every call with an input shape seen before returns that shape's first output
                                                                must FAIL (random draws)
      skip_state_write  every tensor argument the call wrote in place is restored afterwards
                                                                must FAIL in the sequence check
      rank_skew       (multi-rank) the last rank builds the previous rank's data   must FAIL on that rank
    A task control (meta.controls.task[]) names `file` and `attr`: attr(call, h, meta, ctx) -> call."""

    GENERIC = ("noop", "within_tol", "scale_out", "stale_output", "skip_state_write", "rank_skew")

    def __init__(self, name, tol, *, task=None, spec=None, h=None, meta=None, ctx=None):
        self.name, self.tol, self.spec = name, float(tol), spec or {}
        self.outputs = tol if isinstance(tol, TaskTol) else None      # A42: per-output tolerances
        self.state = {}
        self.task_wrap = None
        if name not in self.GENERIC:
            spec = self.spec
            mod = _load(f"_geak_ut_control_{name}", os.path.join(task, spec["file"]))
            self.task_wrap = getattr(mod, spec.get("attr", "wrap"))
            self.h, self.meta, self.ctx = h, meta, ctx

    def _perturb(self, out, key):
        name = self.name
        if name == "within_tol":
            if self.outputs is not None:
                return self._per_output(out, lambda t, tol: t * (1.0 + tol / 10.0))
            return _map(out, lambda t: t * (1.0 + self.tol / 10.0))
        if name == "scale_out":
            if self.outputs is not None:
                return self._per_output(out, lambda t, tol: t * (1.0 + 4.0 * tol), _off_by_much)
            return _map(out, lambda t: t * (1.0 + 4.0 * self.tol), _off_by_much)
        if name == "stale_output":
            if key not in self.state:
                self.state[key] = _map(out, lambda t: t.detach().clone(), lambda t: t.detach().clone())
            return _map(self.state[key], lambda t: t.clone(), lambda t: t.clone())
        return out

    def _per_output(self, out, f_float, f_other=None):
        """within_tol / scale_out with per-output tolerances (A42; A15 point 9): every float output moves by its
        own tolerance (flatten_outputs order); an output per_output does not fit moves by meta.tolerance.tol."""
        rows, problems = self.outputs.layout(out)
        tols = [t for _, t in rows] if not problems else []
        default = float(self.outputs)

        def one(t, i):
            if t.is_floating_point():
                return f_float(t, tols[i] if i < len(tols) else default)
            return f_other(t) if f_other is not None else t
        return _map_flat(out, one)

    def wrap_call(self, call):
        if self.task_wrap is not None:
            return self.task_wrap(call, self.h, self.meta, self.ctx)
        if self.name in ("noop", "rank_skew"):
            return call
        if self.name == "skip_state_write":
            def skip_state(args):
                tensors = _tensor_args(args)
                saved = [t.detach().clone() for t in tensors]
                out = call(args)
                out = _map(out, lambda t: t.detach().clone(), lambda t: t.detach().clone())
                for t, s in zip(tensors, saved):
                    try:
                        if not bool((t == s).all()):
                            t.copy_(s)          # the call wrote this argument: drop the write
                    except Exception:
                        pass
                return out
            return skip_state

        def perturbed(args):
            key = tuple(tuple(t.shape) for t in _tensor_args(args))
            return self._perturb(call(args), ("call", key))
        return perturbed

    def wrap_replay(self, replay):
        if replay and self.task_wrap is None and self.name == "skip_state_write" and replay.get("carry_state"):
            return self._skip_replay_state(replay)
        if not replay or self.task_wrap is not None or self.name not in ("within_tol", "scale_out",
                                                                          "stale_output"):
            return replay
        read_out = replay["read_out"]
        out = dict(replay)
        out["read_out"] = lambda: self._perturb(read_out(), ("replay", None))
        return out

    @staticmethod
    def _skip_replay_state(replay):
        """skip_state_write on a carry_state replay group (A18.3): the declared state is restored after every
        replay to what it was when the step's inputs were filled, so the graph's state writes are dropped
        and the next step reads stale state. The output is read before the restore."""
        fill, read_out = replay["fill"], replay["read_out"]
        saved = {}

        def fill_and_save(case):
            fill(case)
            saved["state"] = [(t, t.detach().clone()) for t in state_tensors(replay)]

        def read_and_drop():
            out = _map(read_out(), lambda t: t.detach().clone(), lambda t: t.detach().clone())
            for t, s in saved.pop("state", []):
                t.copy_(s)
            return out
        return dict(replay, fill=fill_and_save, read_out=read_and_drop)


# --------------------------------------------------------------------------- correctness worker
def _phase_plan(cases_rows, meta):
    """[(phase, regime)] in a fixed order. Without meta.phase_regimes there is one phase ("")."""
    phase_regimes = meta.get("phase_regimes")
    if phase_regimes is None:
        return [("", meta.get("regime", {}))]
    if not isinstance(phase_regimes, dict):
        raise ValueError("meta.phase_regimes must map phase labels to regimes")
    seen = []
    for row in cases_rows:
        phase = row.get("regime", "")
        if phase not in seen:
            seen.append(phase)
    for phase in seen:
        if not isinstance(phase_regimes.get(phase), dict):
            raise ValueError(f"missing phase_regimes entry for {phase!r}")
    return [(p, phase_regimes[p]) for p in seen]


def _with_ref(rows):
    """Cases with an explicit `ref` key: a generated task's `ref` is optional (the native leg records
    the reference), and the shared checkers index it."""
    out = []
    for c in rows:
        if isinstance(c, dict) and "ref" not in c:
            c = dict(c, ref=None)
        out.append(c)
    return out


def _replay_bundle(cases, h, meta, phase):
    fn = getattr(cases, "replay", None)
    if fn is None:
        return None
    arity = h._positional_arity(fn)
    return fn(h, meta, phase) if (arity is None or arity >= 3) else fn(h, meta)


def _replay_groups(cases, h, meta, phase):
    """cases.replay as a list of groups, each recording ONE graph (A18.1): a dict is one group (the original
    form), a list holds several. None when cases.py has no replay for the phase."""
    b = _replay_bundle(cases, h, meta, phase)
    if b is None:
        return None
    if isinstance(b, dict):
        return [b]
    if isinstance(b, (list, tuple)):
        return list(b)
    raise TypeError(f"cases.replay must return a replay group (dict) or a list of groups, not {type(b).__name__}")


def _group_problem(g):
    """Why a replay group cannot be recorded (None = it can): its closures and >= 2 cases, main's rule for
    every group, so every replayed graph is checked with at least two inputs."""
    if not isinstance(g, dict):
        return f"is a {type(g).__name__}, not a replay group"
    missing = [k for k in ("fill", "run", "read_out") if not callable(g.get(k))]
    if missing:
        return f"lacks the closures {missing}"
    n = len(g.get("cases") or [])
    if n < 2:
        return f"has only {n} replay case(s); every group needs >=2"
    return None


def state_tensors(group):
    """A carry_state group's declared `state`: a tensor, a list/tuple of tensors, or a zero-arg callable
    returning either (the persistent buffers consecutive replays carry, e.g. a KV cache or a ring)."""
    s = group.get("state")
    if callable(s) and not hasattr(s, "data_ptr"):
        s = s()
    if s is None:
        return []
    if hasattr(s, "data_ptr"):
        return [s]
    return list(s)


def _same_values(a, b):
    torch = sys.modules.get("torch") or importlib.import_module("torch")
    if a.is_floating_point():
        return bool(((a == b) | (torch.isnan(a) & torch.isnan(b))).all())
    return bool(torch.equal(a, b))


def _carry_checked(group, steps_rewritten):
    """A carry_state group's fill, checked (A18.3): consecutive replays carry the declared state, so every
    fill after the first step's must leave it as it was. Fill call 0 prepares the capture and call 1 the
    first step (which may reset what the capture's warm-up launches wrote); a later step whose fill
    rewrote the state is appended to `steps_rewritten`."""
    fill = group["fill"]
    calls = [0]

    def checked(case):
        i = calls[0]
        calls[0] += 1
        if i < 2:
            return fill(case)
        ts = state_tensors(group)
        before = [t.detach().clone() for t in ts]
        out = fill(case)
        if any(not _same_values(t, b) for t, b in zip(ts, before)):
            steps_rewritten.append(i - 1)
        return out
    return dict(group, fill=checked)


def _group_summary(g, gi, rows, ok=None):
    cases_ = (g.get("cases") or []) if isinstance(g, dict) else []
    ci = int((g.get("capture_idx", 0) if isinstance(g, dict) else 0) or 0)
    cap = cases_[ci] if 0 <= ci < len(cases_) and isinstance(cases_[ci], dict) else {}
    if ok is None:
        ok = all(isinstance(r, dict) and r.get("correct") is True for r in rows or [])
    return {"group": gi, "graphs": g.get("graphs") if isinstance(g, dict) else None,
            "carry_state": bool(isinstance(g, dict) and g.get("carry_state")), "steps": len(cases_),
            "capture_sig": cap.get("sig"), "rows": len(rows or []), "ok": bool(ok)}


def _graph_coverage(h, label, info, groups):
    """A18.2: every deployment graph of this phase (h.deployment_graphs) is recorded by some group's
    declared `graphs`; a group capturing on a timing case declares that case's graph. Returns
    (record, problems)."""
    problems = list(info.get("problems") or [])
    spec = info.get("capture")
    declared = {}
    for gi, g in enumerate(groups):
        if not isinstance(g, dict):
            continue
        gp = h.declared_graph_problems(g.get("graphs"), spec, f"replay group {gi}", dims=info.get("dims"))
        problems += gp
        if gp:
            continue
        mine = {h.graph_key(x): x for x in g["graphs"]}
        declared.update(mine)
        cs = g.get("cases") or []
        ci = int(g.get("capture_idx", 0) or 0)
        cap = cs[ci] if 0 <= ci < len(cs) and isinstance(cs[ci], dict) else {}
        dep = (info.get("cases") or {}).get(cap.get("sig"))
        if dep is not None and h.graph_key(dep) not in mine:
            problems.append(f"replay group {gi} captures on timing case {cap.get('sig')}, whose deployment graph "
                            f"{dep} is not among its declared graphs {g['graphs']}")
    missing = [x for x in info.get("graphs") or [] if h.graph_key(x) not in declared] \
        if not info.get("problems") else []
    if missing:
        problems.append(f"deployment graphs no replay group records: {missing} -- every graph a timing case "
                        "with call mass replays in deployment needs a group (one graph per group, or one "
                        "capture issuing each size)")
    record = {"capture": spec, "dims": info.get("dims"), "deployment_graphs": info.get("graphs") or [],
              "declared": sorted(declared.values(), key=lambda x: (x.get("m", 0), h.graph_key(x))),
              "missing": missing}
    return record, [f"{label}: graph coverage: {p}" for p in problems]


def _tier_coverage(h, steps, declared):
    """A18.4: every declared stateful tier is covered by a sequence step naming it; every step names
    declared tiers. Returns (record, problems)."""
    covered = {t: [] for t in declared}
    untagged, undeclared = [], {}
    for i, st in enumerate(steps):
        sig = f"seq[{i}]:{st.get('sig', '')}" if isinstance(st, dict) else f"seq[{i}]"
        names = h.step_tiers(st)
        if not names:
            untagged.append(sig)
        for n in names:
            if n in covered:
                covered[n].append(sig)
            else:
                undeclared.setdefault(n, []).append(sig)
    uncovered = [t for t, v in covered.items() if not v]
    problems = []
    if not declared:
        problems.append("capabilities.stateful without capabilities.stateful_tiers")
    if uncovered:
        problems.append(f"declared stateful tiers no sequence step covers: {uncovered}")
    if untagged:
        problems.append(f"{len(untagged)} sequence step(s) name no tier (`tier` or `tiers`): {untagged[:6]}")
    if undeclared:
        problems.append(f"sequence steps name undeclared tiers: {sorted(undeclared)}")
    record = {"declared": list(declared), "steps": len(steps),
              "covered": {t: {"steps": len(v), "first": v[:5]} for t, v in covered.items()},
              "uncovered": uncovered, "untagged": len(untagged), "undeclared": sorted(undeclared)}
    return record, [f"stateful tiers: {p}" for p in problems]


def correctness_suite(h, cases, meta, *, blob, draws, book, ctx, strict, control=None, fail_fast=False, tol=None):
    """The full check sequence one correctness worker runs (identically on the native refs leg and on
    the candidate leg). Returns {ok, incomplete, report, errors}. `fail_fast` (a rank of a multi-rank
    pool): an exception is raised instead of recorded -- the rank must stop at once, its peers may be
    waiting for it in a collective.

    Graph replay (A18): cases.replay may return several groups, each recording one graph; the first runs
    inside h.run_correctness as before, every other one through h.check_graph_replay, all strict. In a
    generated task (`strict`) every deployment graph of a timing case with call mass must be recorded by
    a group, a stateful task with a graph phase needs a group carrying state across consecutive replays,
    and a stateful task's sequence steps must cover its declared stateful tiers; any gap is `incomplete`.

    `tol` (default task_tol(meta)): a TaskTol (meta.tolerance.per_output, A42) has every comparison made output by
    output at each output's own tolerance (per_output_correct is h.correct meanwhile); its problems are `incomplete`.
    A generated task that declares a comparison plugin is incomplete: the plugin is not supported (A42)."""
    tol = task_tol(meta) if tol is None else tol
    call = cases.call
    if control is not None:
        call = control.wrap_call(call)
    capture_context = getattr(cases, "capture_context", None) if strict else None
    eager_all = _with_ref(cases.eager_cases(h, meta)) if hasattr(cases, "eager_cases") else []
    shapes_all = list(cases.random_shapes(h, meta))
    caps = meta.get("capabilities") or {}
    stateful = bool(caps.get("stateful"))
    report, errors, incomplete, ok = {}, [], [], True
    graphs_needed = h.deployment_graphs(meta) if strict and hasattr(h, "deployment_graphs") else {}
    carried, graph_phase_seen = [], False
    if strict and hasattr(h, "compare_plugin_problem"):
        plugin = h.compare_plugin_problem(meta)
        if plugin:
            incomplete.append(plugin)
    saved = h.check_correct_multi, h.check_graph_replay, h.check_correct_sequence
    saved_correct = h.correct
    try:
        if isinstance(tol, TaskTol):
            h.correct = per_output_correct(h, saved_correct)
        h.check_correct_multi = book.wrap("eager", saved[0])
        h.check_graph_replay = book.wrap("graph_replay", saved[1])
        h.check_correct_sequence = book.wrap("sequence", saved[2])
        for phase, regime in _phase_plan(shapes_all + eager_all, meta):
            eager = [c for c in eager_all if not phase or c.get("regime", "") == phase]
            shapes = [s for s in shapes_all if not phase or s.get("regime", "") == phase]
            label = phase or "default"
            graph = h.deployment_graph_mode(regime)
            graph_phase_seen = graph_phase_seen or graph
            try:
                groups = _replay_groups(cases, h, meta, phase) if graph else None
            except Exception:
                if fail_fast:
                    raise
                errors.append(f"{label}: replay: {traceback.format_exc(limit=8)}")
                report[label] = {"error": traceback.format_exc(limit=8)}
                ok = False
                continue
            groups = [dict(g, cases=_with_ref(g["cases"]))
                      if isinstance(g, dict) and isinstance(g.get("cases"), (list, tuple)) else g
                      for g in (groups or [])]
            rewritten = {}
            if strict:
                for gi, g in enumerate(groups):
                    if not isinstance(g, dict) or not g.get("carry_state") or _group_problem(g):
                        continue
                    try:
                        has_state = bool(state_tensors(g))
                    except Exception as exc:
                        has_state = False
                        incomplete.append(f"{label}: replay group {gi}: its `state` could not be read: {exc!r}")
                    if not has_state:
                        incomplete.append(f"{label}: replay group {gi} declares carry_state but no `state` "
                                          "(the tensors consecutive replays carry)")
                        continue
                    groups[gi] = _carry_checked(g, rewritten.setdefault(gi, []))
            if control is not None:
                groups = [control.wrap_replay(g) if isinstance(g, dict) else g for g in groups]
            first = groups[0] if groups else None
            try:
                p_ok, rep = h.run_correctness(regime, eager_cases=eager, current_call=call,
                                              random_shapes=shapes, tol=tol, baseline_outputs=blob,
                                              replay=first, draws=draws, strict_graph=strict,
                                              capture_context=capture_context, rank_ctx=ctx)
            except h.HarnessIncompleteError as exc:
                incomplete.append(f"{label}: {exc}")
                report[label] = {"incomplete": str(exc)}
                ok = False
                continue
            except Exception:
                if fail_fast:
                    raise
                errors.append(f"{label}: {traceback.format_exc(limit=8)}")
                report[label] = {"error": traceback.format_exc(limit=8)}
                ok = False
                continue
            if graph and groups:
                rows0 = rep.get("graph_replay") if isinstance(rep.get("graph_replay"), list) else []
                for r in rows0:
                    if isinstance(r, dict):
                        r.setdefault("group", 0)
                summaries = [_group_summary(groups[0], 0, rows0)]
                for gi, g in enumerate(groups[1:], 1):
                    why = _group_problem(g)
                    if why:
                        msg = f"{label}: replay group {gi} {why}"
                        print(f"{h.UT_HARNESS_INCOMPLETE_SENTINEL}: {msg}")
                        incomplete.append(msg)
                        summaries.append({"group": gi, "ok": False, "problem": why})
                        continue
                    kw = {"strict": True} if strict else {}
                    if capture_context is not None:
                        kw["capture_context"] = capture_context
                    try:
                        g_ok, rows = h.check_graph_replay(g["fill"], g["run"], g["read_out"], g["cases"], tol,
                                                          capture_idx=int(g.get("capture_idx", 0) or 0), **kw)
                    except Exception:
                        if fail_fast:
                            raise
                        errors.append(f"{label}: replay group {gi}: {traceback.format_exc(limit=8)}")
                        g_ok, rows = False, []
                    rows = [dict(r, group=gi) if isinstance(r, dict) else r for r in rows or []]
                    rep.setdefault("graph_replay", []).extend(rows)
                    p_ok = p_ok and bool(g_ok)
                    summaries.append(_group_summary(g, gi, rows, g_ok))
                rep["replay_groups"] = summaries
                for gi, steps in sorted(rewritten.items()):
                    if steps:
                        incomplete.append(f"{label}: replay group {gi} (carry_state): fill rewrote the declared "
                                          f"state at step(s) {steps}; a carry_state group's fill refreshes only "
                                          "the per-step inputs")
                carried += [dict(s, phase=label) for s in summaries
                            if s.get("carry_state") and s.get("steps", 0) >= 2]
            if strict and graph and phase in graphs_needed:
                rec, probs = _graph_coverage(h, label, graphs_needed[phase], groups)
                rep["graph_coverage"] = rec
                incomplete += probs
            report[label] = rep
            ok = ok and bool(p_ok)
        seq_fn = getattr(cases, "sequence_cases", None)
        if seq_fn is not None:
            try:
                steps = _with_ref(seq_fn(h, meta))
                if strict and stateful and hasattr(h, "step_tiers"):
                    rec, probs = _tier_coverage(h, steps, h.stateful_tiers(meta))
                    report["sequence_coverage"] = rec
                    incomplete += probs
                s_ok, s_rep = h.check_correct_sequence(call, steps, tol)
                report["sequence"] = s_rep
                ok = ok and bool(s_ok)
            except Exception:
                if fail_fast:
                    raise
                errors.append("sequence: " + traceback.format_exc(limit=8))
                report["sequence"] = {"error": traceback.format_exc(limit=8)}
                ok = False
        elif stateful and strict:
            msg = ("capabilities.stateful is declared but cases.py has no sequence_cases: a stateful "
                   "kernel needs an ordered multi-step check (UT incomplete)")
            incomplete.append(msg)
            ok = False
        if strict and stateful and graph_phase_seen and not carried:
            incomplete.append("capabilities.stateful with a graph-deployed phase: no replay group carries state "
                              "between consecutive replays (carry_state: true, a declared `state`, >= 2 cases)")
    finally:
        h.check_correct_multi, h.check_graph_replay, h.check_correct_sequence = saved
        h.correct = saved_correct
    if isinstance(tol, TaskTol):
        incomplete += [p for p in tol.problems if p not in incomplete]
    mismatched = book.mismatches()
    if mismatched:
        report["native_refs"] = mismatched
        ok = False
    return {"ok": bool(ok) and not errors and not incomplete, "incomplete": incomplete,
            "errors": errors, "report": report}


def _recording_noise(h, noise):
    """`h.check_random_vs_baseline` with every comparison it makes also appending h.noise_rel(out, ref) to
    `noise` (A37): the native refs leg against the native oracle on the random draws. h.correct and the rows
    are unchanged."""
    check = h.check_random_vs_baseline
    measure = getattr(h, "noise_rel", None)

    def recording(*args, **kwargs):
        if measure is None:
            return check(*args, **kwargs)
        orig = h.correct

        def correct_and_record(out, ref, tol):
            noise.append(measure(out, ref))
            if isinstance(tol, TaskTol):
                tol.record_noise(out, ref, measure)       # A42: also output by output
            return orig(out, ref, tol)
        h.correct = correct_and_record
        try:
            return check(*args, **kwargs)
        finally:
            h.correct = orig
    return recording


def _native_noise(report, noise=None):
    """Worst random-draw error of the native refs leg against the native oracle leg (two processes,
    same seeds, same code: the floor any tolerance must clear). `max_rel_err` is correct()'s err at the task's
    tolerance; `noise_rel` (A37) is the same comparisons on correct()'s pass/fail scale, which does not depend
    on the tolerance: correct() passes at t exactly when noise_rel <= t. Comparisons whose noise_rel is not
    finite (non-finite positions that differ, a shape mismatch) are counted in `noise_rel_nonfinite`."""
    errs = []

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "random" and isinstance(v, list):
                    for c in v:
                        e = c.get("max_rel_err") if isinstance(c, dict) else None
                        if isinstance(e, (int, float)):
                            errs.append(float(e))
                else:
                    walk(v)
    walk(report)
    out = {"max_rel_err": max(errs) if errs else None, "cases": len(errs)}
    if noise is not None:
        finite = [float(v) for v in noise if isinstance(v, (int, float)) and math.isfinite(v)]
        out["noise_rel"] = max(finite) if finite else None
        if len(finite) != len(noise):
            out["noise_rel_nonfinite"] = len(noise) - len(finite)
    return out


def _identities(meta):
    targets = list(dict.fromkeys([meta["target_callable"], *(meta.get("phase_target_callables") or {}).values()]))
    return {t: _identity(t) for t in targets}


def _correct_mode(a, task, meta, h, cases, fail_fast=False):
    """`refs` (native, records) or `correct` (candidate, judged against the native records)."""
    torch = h._torch()
    ctx = h.rank_context()
    rank = ctx.rank
    blob_path = f"{a.blob}.rank{rank}.pt" if ctx.world > 1 else f"{a.blob}.pt"
    refs_path = f"{a.blob}.refs.rank{rank}.pt" if ctx.world > 1 else f"{a.blob}.refs.pt"
    result = {"rank": rank, "mode": a.mode, "identity": _identities(meta)}
    blob = torch.load(blob_path, map_location="cpu")
    if a.mode == "refs":
        book = RefBook(h)
    else:
        if not os.path.isfile(refs_path):
            result.update(ok=False, incomplete=[f"native refs missing: {refs_path}"], errors=[], report={})
            return result
        book = RefBook(h, torch.load(refs_path, map_location="cpu"))
    tol = task_tolerance(meta, h, native=a.mode == "refs")
    control = None
    if a.control:
        spec = json.loads(a.control_spec) if a.control_spec else {}
        control = Control(a.control, tol, task=task, spec=spec, h=h, meta=meta, ctx=ctx)
    strict = is_generated_task(task)
    draws = a.draws or int(meta.get("random_draws", 3))
    noise = [] if a.mode == "refs" else None
    saved = h.check_random_vs_baseline
    if noise is not None:
        h.check_random_vs_baseline = _recording_noise(h, noise)
    try:
        result.update(correctness_suite(h, cases, meta, blob=blob, draws=draws, book=book, ctx=ctx,
                                        strict=strict, control=control, fail_fast=fail_fast, tol=tol))
    finally:
        h.check_random_vs_baseline = saved
    if a.mode == "refs":
        torch.save(book.dump(), refs_path)
        result["native_noise"] = _native_noise(result["report"], noise)
        if isinstance(tol, TaskTol):
            result["native_noise"]["per_output"] = tol.noise_record()
    if isinstance(tol, TaskTol):
        result["tolerance"] = tol.record()
    return result


def leg_ctx_from_env(h):
    """Install this process's RankContext from the pool environment (rank 0 of 1 otherwise)."""
    world = int(os.environ.get("WORLD_SIZE", "1") or 1)
    rank = int(os.environ.get("RANK", "0") or 0) if world > 1 else 0
    shift = int(os.environ.get("GEAK_UT_DATA_RANK_SHIFT", "0") or 0)
    data_rank = rank
    if shift and world > 1 and rank == world - 1:
        data_rank = (rank - shift) % world     # rank_skew: the last rank builds another rank's data
    local = int(os.environ.get("LOCAL_RANK", str(rank)) or rank) if world > 1 else 0
    return h.set_rank_context(h.RankContext(rank, world, data_rank, local))


# --------------------------------------------------------------------------- multi-rank serve mode
# One process per rank of a leg (ut_runner.RankPool). The process sets up once -- the task's
# `setup(h, meta, ctx)` or a bounded torch.distributed init -- then runs one command per stdin line
# (list / time / oracle / refs / correct), every rank the same commands in the same order, and answers
# each with a JSON file in the pool's control directory. A command that raises writes its traceback
# and the rank exits at once: its peers would otherwise wait in a collective until the timeout.
DEFAULT_SPIN_US = 200.0       # A03: GPU idle before the device all-reduce (meta.timing.device_spin_us)
DEFAULT_MARGIN_US = 500.0     # A02: release margin after the gloo barrier (meta.timing.release_margin_us)
DEFAULT_SETTLE_S = 0.3        # pause after the warm-up collectives before a graph capture


def _die_with_parent():
    """Kill this rank when the driver dies (PR_SET_PDEATHSIG): a rank still in setup or blocked in a
    collective never reads stdin again and would keep its GPU memory."""
    ppid = os.environ.get("GEAK_UT_PARENT_PID")
    if not ppid or not sys.platform.startswith("linux"):
        return
    import ctypes
    import signal
    ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL)
    if os.getppid() != int(ppid):
        os._exit(1)


def _timing_settings(meta):
    t = meta.get("timing") or {}
    return (float(t.get("device_spin_us", DEFAULT_SPIN_US)), float(t.get("release_margin_us", DEFAULT_MARGIN_US)),
            float(t.get("capture_settle_s", DEFAULT_SETTLE_S)))


# meta.timing keys of the single-rank eager spin (A16) -> harness_lib.time_op(spin=...) keys
EAGER_SPIN_KEYS = {"eager_spin_us": "us", "eager_spin_min_us": "min_us", "eager_spin_factor": "factor",
                   "eager_spin_late_limit": "late_limit"}


def eager_spin(meta):
    """time_op(spin=...) for an eager-deployed phase of a generated single-rank task (A16): the defaults,
    overridden by numeric meta.timing.eager_spin_* values (generate_ut.py check rejects other values)."""
    t = meta.get("timing") if isinstance(meta.get("timing"), dict) else {}
    out = {}
    for key, arg in EAGER_SPIN_KEYS.items():
        v = t.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
            out[arg] = float(v)
    return out


class RankControl:
    """The pool's host-side control plane: a gloo group of its own (never the device group), so the
    harness's barriers and agreements add no communicator on the GPUs. Only int64 tensors travel."""

    def __init__(self, torch, timeout_s):
        import datetime
        self.torch, self.dist = torch, torch.distributed
        self.group = self.dist.new_group(backend="gloo", timeout=datetime.timedelta(seconds=timeout_s))

    def agree(self, ok):
        t = self.torch.tensor([1 if ok else 0], dtype=self.torch.int64)
        self.dist.all_reduce(t, op=self.dist.ReduceOp.MIN, group=self.group)
        return bool(t.item())

    def barrier(self):
        # An all-reduce of a CPU tensor every rank must join: dist.barrier() may probe the accelerator
        # for its tensor's device, which a CPU control group has no business doing.
        self.dist.all_reduce(self.torch.zeros(1, dtype=self.torch.int64), group=self.group)

    def max_int(self, value):
        t = self.torch.tensor([int(value)], dtype=self.torch.int64)
        self.dist.all_reduce(t, op=self.dist.ReduceOp.MAX, group=self.group)
        return int(t.item())


class SampleAligner:
    """Aligns the ranks before every timed sample; harness_lib.time_op calls it as `pre_sample`, after
    the cache flush (`harness_lib.flush_cache`, looked up by its module name) and the device sync:

      gloo barrier agreeing a release time (all_reduce MAX of now + margin: no rank gets it before the
      last rank arrived) -> busy-wait to that time (A02: the barrier's own staggered release stays out
      of the window; never a sleeping wait) -> a GPU spin on this rank's stream -> a one-element
      all-reduce on the pool's device group, or the task's `cases.align` (A03) -> [start event].

    The spin outlasts the time a rank needs to queue the all-reduce, the start event and the kernel,
    so every GPU leaves the all-reduce together and the start events fire together whatever skew the
    hosts still had. Without CUDA only the barrier and the release run."""

    def __init__(self, torch, ctl, *, device, dev_group, spin_us, margin_us, align_call=None):
        self.torch, self.ctl, self.device = torch, ctl, device
        self.dev_group, self.align_call = dev_group, align_call
        self.spin_us, self.margin_ns = float(spin_us), int(float(margin_us) * 1000)
        self.samples = self.late = 0
        self.max_late_ns = 0
        self.spin_cycles = 0
        if device:
            self.one = torch.ones(1, device="cuda")
            if align_call is None:
                torch.distributed.all_reduce(self.one, group=dev_group)      # warm the device group up
            self.spin_cycles = self._calibrate()
            torch.cuda.synchronize()

    def _calibrate(self):
        torch = self.torch
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        cycles, per_us = 1_000_000, None
        for _ in range(3):
            torch.cuda.synchronize()
            start.record()
            torch.cuda._sleep(cycles)
            end.record()
            end.synchronize()
            ms = start.elapsed_time(end)
            if ms > 0:
                per_us = cycles / (ms * 1000.0)
        return max(1, int(self.spin_us * per_us)) if per_us else 0

    def release(self):
        t = self.torch.tensor([time.monotonic_ns() + self.margin_ns], dtype=self.torch.int64)
        self.torch.distributed.all_reduce(t, op=self.torch.distributed.ReduceOp.MAX, group=self.ctl.group)
        target = int(t.item())
        now = time.monotonic_ns()
        if now > target:
            self.late += 1
            self.max_late_ns = max(self.max_late_ns, now - target)
        while time.monotonic_ns() < target:
            pass

    def device_align(self):
        if not self.device:
            return
        if self.spin_cycles:
            self.torch.cuda._sleep(self.spin_cycles)
        if self.align_call is not None:
            self.align_call()
        else:
            self.torch.distributed.all_reduce(self.one, group=self.dev_group)

    def __call__(self):
        self.samples += 1
        self.release()
        self.device_align()

    def receipt(self):
        return {"samples": self.samples, "late_releases": self.late,
                "max_late_us": round(self.max_late_ns / 1000.0, 1), "spin_cycles": self.spin_cycles,
                "device_sync": ("cases.align" if self.align_call is not None else "rccl_all_reduce_1")
                if self.device else "none (no CUDA)"}


def _capture_scope(cases):
    import contextlib
    make = getattr(cases, "capture_context", None)
    return make() if make else contextlib.nullcontext()


def _agreed_capture(torch, ctl, cases, settle_s, tries=3):
    """A capture function for harness_lib.time_op: a graph EVERY rank captured, or None on every rank
    (a rank that silently fell back to eager would issue its collectives outside its peers' graph)."""
    def capture(_torch, call, inner):
        for _ in range(max(1, tries)):
            g = None
            try:
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        call()
                torch.cuda.current_stream().wait_stream(s)
                torch.cuda.synchronize()
                time.sleep(settle_s)        # let the communicator's watchdog retire the warm-up work
                g = torch.cuda.CUDAGraph()
                with _capture_scope(cases):
                    with torch.cuda.graph(g, capture_error_mode="thread_local"):
                        for _ in range(inner):
                            call()
                torch.cuda.synchronize()
            except Exception as exc:
                sys.stderr.write(f"[leg_runner] rank capture failed: {exc!r}\n")
                g = None
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            if ctl.agree(g is not None):
                return (g, inner)
            ctl.barrier()
        return None
    return capture


def _agreed_graph_replay(h, torch, ctl, cases, settle_s):
    """harness_lib.check_graph_replay with an agreed capture; a capture that fails anywhere FAILS."""
    capture = _agreed_capture(torch, ctl, cases, settle_s)

    def check(fill, run, read_out, rows, tol, capture_idx=0, warmup=3, **_):
        if not rows:
            return True, [{"case": "graph_replay", "correct": True, "max_rel_err": None, "note": "skipped: no cases"}]
        fill(rows[capture_idx])
        g = capture(torch, run, 1)
        if g is None:
            return False, [{"case": "graph_replay", "correct": False, "max_rel_err": None,
                            "note": "graph_capture_failed on at least one rank"}]
        per, all_ok = [], True
        for c in rows:
            try:
                fill(c)
                g[0].replay()
                torch.cuda.synchronize()
                ok, err = h.correct(read_out(), c["ref"], tol)
                per.append({"case": c.get("sig", ""), "correct": ok,
                            "max_rel_err": round(err, 5) if err == err and err != float("inf") else None,
                            "note": "graph_replay", **(getattr(err, "row_fields", None) or {})})
            except Exception as exc:
                ok = False
                per.append({"case": c.get("sig", ""), "correct": False, "max_rel_err": None,
                            "note": f"graph_replay_raised: {exc!r}"})
            all_ok = all_ok and ok
        return all_ok, per
    return check


class RankWorker:
    """One rank of a pool: the framework set up once, then any number of commands."""

    def __init__(self, task):
        import datetime
        self.task = task
        with open(os.path.join(task, "meta.json")) as fh:
            self.meta = json.load(fh)
        self.h = _load("harness_lib", os.path.join(task, "harness_lib.py"))
        self.ctx = leg_ctx_from_env(self.h)
        self.cases = _load("cases", os.path.join(task, "cases.py"))
        torch = self.torch = self.h._torch()
        self.cuda = bool(torch.cuda.is_available())
        if self.cuda:
            torch.cuda.set_device(self.ctx.local_rank)
        setup_s = float(os.environ.get("GEAK_UT_SETUP_S", "900") or 900)
        command_s = float(os.environ.get("GEAK_UT_COMMAND_S", "1800") or 1800)
        dist = torch.distributed
        t0 = time.monotonic()
        if hasattr(self.cases, "setup"):
            self.cases.setup(self.h, self.meta, self.ctx)
        if not dist.is_initialized():
            # Default setup: the pool's process group (NCCL on GPUs) and this rank's device -- an
            # ordinary single-GPU op needs nothing else.
            dist.init_process_group(backend="nccl" if self.cuda else "gloo", init_method="env://",
                                    world_size=self.ctx.world, rank=self.ctx.rank,
                                    timeout=datetime.timedelta(seconds=setup_s))
        self.ctl = RankControl(torch, command_s)
        spin_us, margin_us, self.settle_s = _timing_settings(self.meta)
        dev_group = None
        if self.cuda and not hasattr(self.cases, "align"):
            # A03: the device all-reduce runs on this pool's device group; a task whose own init built
            # only a gloo group gets one from here.
            dev_group = None if dist.get_backend() == "nccl" else dist.new_group(backend="nccl")
        align_call = None
        if hasattr(self.cases, "align"):
            align_call = lambda: self.cases.align(self.h, self.meta, self.ctx)   # noqa: E731
        self.aligner = SampleAligner(torch, self.ctl, device=self.cuda, dev_group=dev_group,
                                     spin_us=spin_us, margin_us=margin_us, align_call=align_call)
        if self.cuda:
            self.h.check_graph_replay = _agreed_graph_replay(self.h, torch, self.ctl, self.cases, self.settle_s)
        self.ctl.barrier()
        self.profile = {"rank": self.ctx.rank, "setup_s": round(time.monotonic() - t0, 2),
                        "data_rank": self.ctx.data_rank, "backend": dist.get_backend(), "cuda": self.cuda}

    def _case_execution(self, case):
        h, meta, cases = self.h, self.meta, self.cases
        pr = meta.get("phase_regimes")
        regime = pr.get(case.get("regime", "")) if isinstance(pr, dict) else meta.get("regime", {})
        if not isinstance(regime, dict):
            raise ValueError(f"missing phase_regimes entry for {case.get('regime')!r}")
        call = cases.call
        if h.deployment_compile_mode(regime):
            call = h.compiled_op(call, regime)
        return call, h.deployment_graph_mode(regime)

    def do(self, mode, params):
        h, cases, meta, torch = self.h, self.cases, self.meta, self.torch
        out = {"rank": self.ctx.rank}
        if mode == "list":
            out["sigs"] = [c["sig"] for c in cases.timing_cases(h, meta)]
        elif mode == "time":
            rows = []
            capture = _agreed_capture(torch, self.ctl, cases, self.settle_s) if self.cuda else None
            for c in cases.timing_cases(h, meta):
                if params.get("bucket") and c["sig"] != params["bucket"]:
                    continue
                call, graph = self._case_execution(c)
                kw = {"detail": True, "samples": True, "pre_sample": self.aligner,
                      "inner": max(1, int(params.get("inner") or c.get("inner") or 1))}
                if capture is not None:
                    kw["capture"] = capture
                before = self.aligner.receipt()
                self.ctl.barrier()
                r = h.time_op(lambda c=c, call=call: call(c["args"]), graph=graph, **kw)
                self.ctl.barrier()
                row = {"sig": c["sig"], "regime": c.get("regime", ""), "m": c.get("m"),
                       "ms": (r or {}).get("ms"), "wall_ms": (r or {}).get("wall_ms"),
                       "timer": (r or {}).get("timer"), "cache_condition": (r or {}).get("cache_condition"),
                       "samples": (r or {}).get("samples"), "inner": kw["inner"]}
                if isinstance(r, dict) and "primed" in r:
                    row.update(primed=r["primed"], host_ms=r.get("host_ms"))
                after = self.aligner.receipt()
                row["alignment"] = dict(after, samples=after["samples"] - before["samples"],
                                        late_releases=after["late_releases"] - before["late_releases"])
                rows.append(row)
            out["cases"] = rows
            out["identity"] = _identity(meta["target_callable"])
        elif mode == "oracle":
            device = "cuda" if self.cuda else "cpu"
            draws = int(params.get("draws") or meta.get("random_draws", 3))
            blob = {}
            for shape in cases.random_shapes(h, meta):
                call, _ = self._case_execution(shape)
                for i in range(max(1, draws)):
                    rng = torch.Generator(device=device).manual_seed(int(params.get("seed") or 0) + i)
                    args = h.call_make_inputs(shape["make_inputs"], rng,
                                              self.ctx.for_draw(int(params.get("seed") or 0) + i, device))
                    blob[f"{shape['sig']}|{i}"] = _snapshot(call(args))
            path = f"{params['blob']}.rank{self.ctx.rank}.pt"
            torch.save(blob, path)
            out.update(out=path, n=len(blob))
        elif mode in ("refs", "correct"):
            a = argparse.Namespace(mode=mode, blob=params["blob"], draws=int(params.get("draws") or 0),
                                   control=params.get("control") or "",
                                   control_spec=params.get("control_spec") or "")
            out.update(_correct_mode(a, self.task, meta, h, cases, fail_fast=True))
        else:
            raise ValueError(f"unknown pool command {mode!r}")
        return out

    def close(self):
        try:
            if hasattr(self.cases, "teardown"):
                self.cases.teardown()
        finally:
            try:
                self.torch.distributed.destroy_process_group()
            except Exception:
                pass


def _write_json(path, obj):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, default=str)
    os.replace(tmp, path)


def serve_main(task, ctl_dir):
    """Pool worker: set up once, then run one command per stdin line (see ut_runner.RankPool)."""
    _die_with_parent()
    rank = int(os.environ.get("RANK", "0"))
    try:
        worker = RankWorker(task)
    except Exception:
        _write_json(os.path.join(ctl_dir, f"rank{rank}.setup_error.json"),
                    {"rank": rank, "error": traceback.format_exc(limit=12)})
        sys.stderr.write(traceback.format_exc())
        sys.stderr.flush()
        os._exit(1)
    _write_json(os.path.join(ctl_dir, f"rank{rank}.ready.json"), dict(worker.profile, pid=os.getpid()))
    code = 0
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            cmd = json.loads(line)
            if cmd.get("mode") == "exit":
                break
            t0 = time.monotonic()
            try:
                result = worker.do(cmd["mode"], cmd.get("params") or {})
            except Exception:
                tb = traceback.format_exc(limit=12)
                sys.stderr.write(tb)
                sys.stderr.flush()
                _write_json(os.path.join(ctl_dir, f"rank{rank}.{int(cmd['seq'])}.json"),
                            {"rank": rank, "error": tb})
                os._exit(1)          # the peers may sit in a collective: end the pool now
            result["seconds"] = round(time.monotonic() - t0, 3)
            _write_json(os.path.join(ctl_dir, f"rank{rank}.{int(cmd['seq'])}.json"), result)
    finally:
        worker.close()
    return code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--mode", required=True,
                    choices=("list", "resolve", "time", "oracle", "refs", "correct", "serve"))
    ap.add_argument("--bucket", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--draws", type=int, default=0)
    ap.add_argument("--target-callable", default="", help="identity probe only; defaults to meta target")
    ap.add_argument("--raw-samples", action="store_true", help="time: also return every sample's ms")
    ap.add_argument("--inner", type=int, default=0,
                    help="time (generated tasks): launches per sample, overriding the case's inner (A20)")
    ap.add_argument("--blob", default="", help="refs/correct: path prefix of the native blobs")
    ap.add_argument("--control", default="", help="correct: verify_ut control name")
    ap.add_argument("--control-spec", default="", help="correct: JSON spec of a task control")
    ap.add_argument("--ctl", default="", help="serve: the pool's control directory")
    a = ap.parse_args()

    task = os.path.abspath(a.task)
    if a.mode == "serve":
        if not a.ctl:
            ap.error("--mode serve needs --ctl")
        sys.exit(serve_main(task, a.ctl))
    with open(os.path.join(task, "meta.json")) as fh:
        meta = json.load(fh)

    if a.mode == "resolve":
        print(json.dumps(_identity(a.target_callable or meta["target_callable"])))
        return
    if a.target_callable:
        ap.error("--target-callable is only valid with --mode resolve")

    h = _load("harness_lib", os.path.join(task, "harness_lib.py"))
    if hasattr(h, "set_rank_context"):
        leg_ctx_from_env(h)
    cases = _load("cases", os.path.join(task, "cases.py"))
    regime = meta.get("regime", {})

    if a.mode == "list":
        print(json.dumps({"sigs": [c["sig"] for c in cases.timing_cases(h, meta)]}))
        return

    if a.mode in ("refs", "correct"):
        if not a.blob or not a.out:
            ap.error(f"--mode {a.mode} needs --blob and --out")
        result = _correct_mode(a, task, meta, h, cases)
        with open(a.out, "w") as fh:
            json.dump(result, fh, default=str)
        print(json.dumps({"out": a.out, "ok": result.get("ok")}))
        return

    call = cases.call
    if meta.get("phase_regimes") is None and h.deployment_compile_mode(regime):
        call = h.compiled_op(call, regime)
    graph = h.deployment_graph_mode(regime)
    generated = is_generated_task(task)

    def case_execution(case):
        phase_regimes = meta.get("phase_regimes")
        if phase_regimes is None:
            return call, graph
        phase = case.get("regime", "")
        if not isinstance(phase_regimes, dict) or not isinstance(phase_regimes.get(phase), dict):
            raise ValueError(f"missing phase_regimes entry for {phase!r}")
        phase_regime = phase_regimes[phase]
        phase_call = cases.call
        if h.deployment_compile_mode(phase_regime):
            phase_call = h.compiled_op(phase_call, phase_regime)
        return phase_call, h.deployment_graph_mode(phase_regime)

    if a.mode == "time":
        sel = [c for c in cases.timing_cases(h, meta) if not a.bucket or c["sig"] == a.bucket]
        out = []
        for c in sel:
            case_call, case_graph = case_execution(c)
            kw = {"detail": True}
            if a.raw_samples:
                kw["samples"] = True
            if generated:
                # Generated tasks only: a case may raise `inner` (C2: a graph-replayed kernel whose one
                # replay is shorter than its dispatch), and graph capture enters cases.capture_context.
                kw["inner"] = a.inner if a.inner > 0 else max(1, int(c.get("inner") or 1))
                if getattr(cases, "capture_context", None) is not None:
                    kw["capture_context"] = cases.capture_context
                if not case_graph:
                    # A16: a phase the server runs without a CUDA graph is timed on a queued GPU spin, so
                    # the reading is GPU execution time and not the host's dispatch latency, which differs
                    # between the two legs' bindings. Graph timing and old tasks are unchanged.
                    kw["spin"] = eager_spin(meta)
            r = h.time_op(lambda c=c: case_call(c["args"]), graph=case_graph, **kw)
            row = {"sig": c["sig"], "regime": c.get("regime", ""), "m": c.get("m"),
                   "ms": (r or {}).get("ms"), "wall_ms": (r or {}).get("wall_ms"),
                   "timer": (r or {}).get("timer"),
                   # Which cache condition produced `ms`. A decode A/B read under 'write-evict' is
                   # not comparable to one read under 'read-evict' (1.40 vs 1.12 on MI355X --
                   # see harness_lib.cache_policy), so the two must never be folded together.
                   "cache_condition": (r or {}).get("cache_condition")}
            if isinstance(r, dict) and "primed" in r:
                # The timing receipt travels with the time instead of being dropped here.
                row["primed"] = r["primed"]
                row["host_ms"] = r.get("host_ms")
            if isinstance(r, dict) and isinstance(r.get("spin"), dict):
                row["spin"] = r["spin"]
                row["latency_ms"] = r.get("latency_ms")
            if isinstance(r, dict) and "samples" in r:
                row["samples"] = r["samples"]
            if generated and (c.get("inner") or a.inner > 0):
                row["inner"] = kw["inner"]
            out.append(row)
        print(json.dumps({"cases": out, "identity": _identity(meta["target_callable"])}))
        return

    torch = h._torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    draws = a.draws or int(meta.get("random_draws", 3))
    ctx = h.rank_context() if hasattr(h, "rank_context") else None
    blob = {}
    for shape in cases.random_shapes(h, meta):
        case_call, _ = case_execution(shape)
        for i in range(max(1, draws)):
            rng = torch.Generator(device=device).manual_seed(int(a.seed) + i)
            if ctx is not None:
                args = h.call_make_inputs(shape["make_inputs"], rng, ctx.for_draw(int(a.seed) + i, device))
            else:
                args = shape["make_inputs"](rng)
            out = case_call(args)
            blob[f"{shape['sig']}|{i}"] = _snapshot(out)
    torch.save(blob, a.out)
    print(json.dumps({"out": a.out, "n": len(blob)}))


if __name__ == "__main__":
    main()
