"""Optional opaque-oracle retention used by ``run_task_entry.py``.

Activation is an absolute ``WORKLOAD_ORACLE_ARCHIVE_DIR`` outside the task, or
the worker's existing ``WORK_ROOT/runs/<task>/.evidence_pending`` handoff. The
explicit directory takes precedence, must not be a symlink, and is created if
absent. Each entry creates a fresh ``generated-oracles-*`` session. With neither
handoff the entry does not load this helper or install an observer.

The observer copies ``_baseline_random.pt`` as opaque bytes at return events of
the exact task ``harness_lib.py:baseline_random_outputs``. It imports no tensor
module and leaves the harness function and its return object unchanged. Python
also reports return events during exception unwinding: a retained file then may
be partial or left over from an earlier call. A copy and its hash prove only the
bytes present at observation, not freshness, oracle use, or a native UT verdict.
Zero matching calls are recorded as ``not_observed``.

Direct calls are observed in the entry process. The supported subprocess bridge
is the inspected untimed ``subprocess.run([sys.executable, script], ...)`` form,
where script is an absolute regular ``TASK/run/correctness_*.py`` file containing
a call to ``baseline_random_outputs``. It requires an inspectable overlay
``sitecustomize.py`` whose only argv-sensitive branch, if any, checks for the
``rocm_agent_enumerator`` basename. Unsupported matching launches or startup
conditions fail before execution. Other launch APIs and indirect oracle calls
are outside this bridge; fresh timing and baseline legs are never instrumented.

The bridge adds a real bootstrap in the original script directory, retaining the
original cwd, PYTHONPATH and startup selection. It restores the requested script
argv before running its unchanged bytes. Evidence separately records requested
argv, bootstrap argv and real OS argv; the real OS command necessarily names the
bootstrap. CompletedProcess.args and subprocess exception cmd retain the
caller's requested argv. Bootstrap files and helper snapshots remain as evidence.

An existing profiler is refused and preserved; installed hooks are restored on
exit. Archival failure blocks further oracle generation and makes an otherwise
successful entry exit 78. Native failures remain native failures, including when
archival also fails; a caller can still catch its original exception. Native
verdicts remain the frozen entry's own and are never derived from copied bytes.
"""
import ast
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import re
import runpy
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time


class OracleArchiveFailure(BaseException):
    """Bypass ordinary UT `except Exception` gates after an archival failure."""


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def outside(task, path):
    if path.is_symlink() or path.resolve().is_relative_to(task.resolve()):
        raise OracleArchiveFailure("oracle archive must be a regular directory outside the task")


def archive_destination(task):
    explicit = os.environ.get("WORKLOAD_ORACLE_ARCHIVE_DIR")
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            raise OracleArchiveFailure("explicit oracle archive must be absolute")
        outside(task, path)
        return path
    work = os.environ.get("WORK_ROOT")
    if not work:
        return None
    run = Path(work) / "runs" / task.name
    pending = run / ".evidence_pending"
    if pending.is_symlink():
        raise OracleArchiveFailure("pending evidence pointer must not be a symlink")
    if not pending.exists():
        return None
    path = Path(pending.read_text().strip())
    outside(task, path)
    if not path.is_absolute() or not path.resolve().is_relative_to(run.resolve()):
        raise OracleArchiveFailure("pending oracle archive is outside this worker's run")
    record = json.loads((path / "retention.json").read_text())
    if Path(record.get("task_dir", "")).resolve() != task or record.get("complete") is not False:
        raise OracleArchiveFailure("pending evidence does not identify this active task measurement")
    return path


def native_failure(error):
    if error is None or isinstance(error, OracleArchiveFailure):
        return False
    return not isinstance(error, SystemExit) or error.code not in (None, 0)


def startup_contract(script, env, cwd):
    """Support the inspected overlay's only argv branch; refuse other argv uses."""
    locations = [script.parent] + [Path(p) if p else cwd for p in env.get("PYTHONPATH", "").split(os.pathsep)]
    site = next((p / "sitecustomize.py" for p in locations if (p / "sitecustomize.py").is_file()), None)
    if site is None or site.is_symlink():
        raise OracleArchiveFailure("correctness bootstrap requires an inspectable overlay sitecustomize")
    tree = ast.parse(site.read_text())
    expected = ast.parse('sys.argv and os.path.basename(sys.argv[0]) == "rocm_agent_enumerator"', mode="eval").body
    allowed = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and ast.dump(node.value) == ast.dump(expected):
            allowed.update(id(child) for child in ast.walk(node.value))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and node.attr in ("argv", "orig_argv")
                and id(node) not in allowed):
            raise OracleArchiveFailure("unsupported argv-sensitive overlay startup")
        if isinstance(node, ast.ImportFrom) and node.module == "sys" and any(n.name in ("argv", "orig_argv") for n in node.names):
            raise OracleArchiveFailure("unsupported imported argv in overlay startup")
    return {"file": str(site.resolve()), "sha256": sha(site),
            "checked_argv_branch": "original correctness and bootstrap are both non-rocm_agent_enumerator"}


class Observation:
    def __init__(self, task, destination=None, *, session=None, process_dir=None, wrap_diagnostics=True, provenance=None):
        self.task = Path(task).resolve()
        self.harness = str(self.task / "harness_lib.py")
        self.destination = destination
        self.session = session
        self.process_dir = process_dir
        self.wrap_diagnostics = wrap_diagnostics
        self.provenance = provenance or {}
        self.pid, self.thread = os.getpid(), threading.get_ident()
        self.rows, self.errors, self.children, self.active = [], [], [], {}
        self.sequence = 0
        self.callback = self.profile

    def __enter__(self):
        if self.session is None:
            outside(self.task, self.destination)
            self.destination.mkdir(parents=True, exist_ok=True)
            self.session = Path(tempfile.mkdtemp(prefix="generated-oracles-", dir=self.destination))
            shutil.copy2(__file__, self.session / "observer.py")
            write(self.session / "session.json", {"task_dir": str(self.task), "helper_sha256": sha(self.session / "observer.py"),
                  "scope": "opaque file snapshots at exact harness function return events; no native verdict/use inference"})
        else:
            self.session = Path(self.session)
            outside(self.task, self.session)
        if self.process_dir is None:
            self.process_dir = self.session / "entry"
            self.process_dir.mkdir()
        else:
            self.process_dir = Path(self.process_dir)
        self.previous_profile = sys.getprofile()
        self.previous_run = subprocess.run
        self.record(False, None)
        if self.previous_profile is not None:
            self.errors.append("existing sys profiler prevents safe oracle observation")
            self.record(True, None)
            raise OracleArchiveFailure(self.errors[-1])
        sys.setprofile(self.callback)
        if self.wrap_diagnostics:
            subprocess.run = self.run_diagnostic
        return self

    def record(self, finished, error):
        count = sum(row.get("copied") is True for row in self.rows) + sum(row.get("captured_calls", 0) for row in self.children)
        status = "running" if not finished else "archive_failed" if self.errors else "captured" if count else "not_observed"
        write(self.process_dir / "observation.json", {
            "pid": self.pid, "task_dir": str(self.task), "observation_finished": finished,
            "status": status, "captured_calls": count, "calls": self.rows, "children": self.children,
            "errors": self.errors, "native_failure": native_failure(error),
            "native_exception": type(error).__name__ if native_failure(error) else None,
            "native_exit": error.code if isinstance(error, SystemExit) and isinstance(error.code, (int, str, type(None))) else None,
            "provenance": self.provenance,
            "scope": "return events also occur during exception unwinding; copied bytes are not a UT verdict"})

    def __exit__(self, kind, error, traceback):
        current = sys.getprofile()
        displaced = current is not self.callback and not (current is None and isinstance(error, OracleArchiveFailure))
        sys.setprofile(self.previous_profile)
        if self.wrap_diagnostics:
            subprocess.run = self.previous_run
        if displaced:
            self.errors.append("sys profiler was replaced during oracle observation")
        if isinstance(error, OracleArchiveFailure) and not self.errors:
            self.errors.append(str(error))
        try:
            self.record(True, error)
        except (OSError, TypeError, ValueError) as exc:
            self.errors.append("cannot publish oracle observation: " + str(exc))
        if self.errors:
            print("[oracle-archive] " + "; ".join(self.errors), file=sys.stderr)
            if not native_failure(error) and not isinstance(error, OracleArchiveFailure):
                raise OracleArchiveFailure(self.errors[0])
        return False

    def profile(self, frame, event, argument):
        if (os.getpid() != self.pid or event not in ("call", "return")
                or frame.f_code.co_name != "baseline_random_outputs"
                or os.path.realpath(frame.f_code.co_filename) != self.harness):
            return
        if event == "call":
            if self.errors:
                raise OracleArchiveFailure("refusing another oracle generation after archival failure")
            try:
                source = self.task / "_baseline_random.pt"
                if source.is_symlink():
                    raise ValueError("generated oracle path must not be a symlink")
                if Path(frame.f_locals.get("task_dir", self.task)).resolve() != self.task:
                    raise ValueError("oracle function was called for a different task")
            except (OSError, TypeError, ValueError) as exc:
                self.errors.append(str(exc))
                raise OracleArchiveFailure(self.errors[-1]) from exc
            self.sequence += 1
            self.active[id(frame)] = self.sequence
            return
        sequence = self.active.pop(id(frame), None)
        row = {"call": sequence, "seed": frame.f_locals.get("seed"), "draws": frame.f_locals.get("draws"),
               "pid": self.pid, "returned_none": argument is None, "copied": False,
               "caller_file": frame.f_back.f_code.co_filename if frame.f_back else None}
        self.rows.append(row)
        invalid_scalars = False
        for key in ("seed", "draws"):
            if ((row[key] is not None and type(row[key]) not in (int, float, str, bool))
                    or (type(row[key]) is float and not math.isfinite(row[key]))):
                row[key] = {"unsupported_type": type(row[key]).__name__}
                invalid_scalars = True
        try:
            if sequence is None:
                raise ValueError("oracle return was not observed from its call boundary")
            if invalid_scalars:
                raise ValueError("oracle seed/draw metadata is not scalar")
            directory = self.process_dir / ("call-%04d" % sequence)
            directory.mkdir()
            source = self.task / "_baseline_random.pt"
            digest = hashlib.sha256()
            with os.fdopen(os.open(source, os.O_RDONLY | os.O_NOFOLLOW), "rb") as incoming:
                before = os.fstat(incoming.fileno())
                if not stat.S_ISREG(before.st_mode):
                    raise ValueError("generated oracle is not a regular file")
                with (directory / "_baseline_random.pt").open("xb") as outgoing:
                    for chunk in iter(lambda: incoming.read(1 << 20), b""):
                        outgoing.write(chunk)
                        digest.update(chunk)
                after = os.fstat(incoming.fileno())
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("generated oracle changed during synchronous copy")
            row.update(copied=True, path=str((directory / "_baseline_random.pt").relative_to(self.session)),
                       size=before.st_size, mtime_ns=before.st_mtime_ns, copy_sha256=digest.hexdigest())
            write(directory / "copy.json", row)
        except (OSError, ValueError) as exc:
            row["copied"] = False
            row["error"] = str(exc)
            self.errors.append("oracle copy failed: " + str(exc))
            # A non-None return proves normal function return. During exception
            # unwinding preserve the original failure; the next CALL is refused.
            if argument is not None:
                raise OracleArchiveFailure(self.errors[-1])

    def run_diagnostic(self, *popenargs, **kwargs):
        requested = popenargs[0] if popenargs else kwargs.get("args")
        if os.getpid() != self.pid or threading.get_ident() != self.thread or not isinstance(requested, (list, tuple)):
            return self.previous_run(*popenargs, **kwargs)
        scripts = [Path(arg) for arg in requested[1:] if isinstance(arg, str) and arg.endswith(".py")]
        matching = []
        for path in scripts:
            if path.is_absolute() and path.parent == self.task / "run" and path.is_file():
                try:
                    tree = ast.parse(path.read_text())
                except SyntaxError:
                    # A syntactically invalid script cannot generate an oracle;
                    # let its original Python invocation report the native error.
                    continue
                if any(isinstance(node, ast.Call) and
                       (getattr(node.func, "attr", None) or getattr(node.func, "id", None)) == "baseline_random_outputs"
                       for node in ast.walk(tree)):
                    matching.append(path)
        if not matching:
            return self.previous_run(*popenargs, **kwargs)
        if self.errors:
            raise OracleArchiveFailure("refusing another correctness launch after archival failure")
        script = matching[0]
        if (len(requested) != 2 or requested[0] != sys.executable or len(matching) != 1
                or script.is_symlink() or not re.fullmatch(r"correctness_[A-Za-z0-9_]+\.py", script.name)):
            self.errors.append("unsupported oracle-producing subprocess launch")
            raise OracleArchiveFailure(self.errors[-1])
        env = kwargs.get("env") or os.environ
        cwd = Path(kwargs.get("cwd") or os.getcwd()).resolve()
        startup = startup_contract(script, env, cwd)
        child = Path(tempfile.mkdtemp(prefix="correctness-", dir=self.session))
        fd, filename = tempfile.mkstemp(prefix="correctness_oracle_archive_", suffix=".py", dir=script.parent)
        os.close(fd)
        bootstrap = Path(filename)
        body = ("import importlib.util, sys\n"
                f"_spec=importlib.util.spec_from_file_location('_oracle_archive_bootstrap', {str(self.session / 'observer.py')!r})\n"
                "_module=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_module)\n"
                "_module.bootstrap_main(sys.argv[1:])\n")
        bootstrap.write_text(body)
        launch = {"task_dir": str(self.task), "session": str(self.session), "process_dir": str(child),
                  "original_argv": list(requested[1:]), "original_script_sha256": sha(script),
                  "startup": startup, "bootstrap_sha256": sha(bootstrap),
                  "helper_sha256": sha(self.session / "observer.py")}
        write(child / "launch.json", launch)
        actual = [requested[0], str(bootstrap), str(child / "launch.json")]
        write(child / "os_launch.json", {"requested": list(requested), "actual": actual, "cwd": str(cwd),
                                        "pythonpath": env.get("PYTHONPATH")})
        positional, named = list(popenargs), dict(kwargs)
        if positional:
            positional[0] = actual
        else:
            named["args"] = actual
        result, original_error = None, None
        try:
            result = self.previous_run(*positional, **named)
            result.args = requested
        except BaseException as exc:
            original_error = exc
            if isinstance(exc, (subprocess.CalledProcessError, subprocess.TimeoutExpired)):
                exc.cmd = requested
            raise
        finally:
            try:
                observation = json.loads((child / "observation.json").read_text())
                self.children.append({"record": str((child / "observation.json").relative_to(self.session)),
                                      "captured_calls": observation.get("captured_calls", 0)})
                failed = not observation.get("observation_finished") or bool(observation.get("errors"))
                native_failed = observation.get("native_failure") is True
            except (OSError, ValueError):
                failed, native_failed = True, bool(result and result.returncode)
            if failed:
                self.errors.append("correctness subprocess oracle archival failed: " + str(child))
                if original_error is None and not native_failed:
                    raise OracleArchiveFailure(self.errors[-1])
        return result


def entry_context(task):
    task = Path(task).resolve()
    try:
        destination = archive_destination(task)
    except (OSError, TypeError, ValueError) as exc:
        raise OracleArchiveFailure("invalid oracle archive handoff: " + str(exc)) from exc
    return nullcontext() if destination is None else Observation(task, destination)


def bootstrap_main(argv):
    launch = json.loads(Path(argv[0]).read_text())
    script = Path(launch["original_argv"][0])
    site = sys.modules.get("sitecustomize")
    actual_site = Path(site.__file__).resolve() if site is not None else None
    if (actual_site != Path(launch["startup"]["file"]) or sha(actual_site) != launch["startup"]["sha256"]
            or sha(script) != launch["original_script_sha256"] or sha(__file__) != launch["helper_sha256"]):
        raise OracleArchiveFailure("bootstrap startup/source provenance changed")
    provenance = {"bootstrap_sys_argv": list(sys.argv), "original_argv": launch["original_argv"],
                  "os_argv": Path("/proc/self/cmdline").read_bytes().rstrip(b"\0").decode(errors="replace").split("\0"),
                  "cwd": os.getcwd(), "pythonpath": os.environ.get("PYTHONPATH"), **launch["startup"]}
    sys.argv = launch["original_argv"]
    with Observation(launch["task_dir"], session=launch["session"], process_dir=launch["process_dir"],
                     wrap_diagnostics=False, provenance=provenance):
        runpy.run_path(str(script), run_name="__main__")
