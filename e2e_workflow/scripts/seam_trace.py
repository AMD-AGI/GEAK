"""Install safe profiler markers on candidate live call seams."""

import atexit
import functools
import importlib
import inspect
import os
import sys
import threading
import types


MARKER_PREFIX = "GEAK_TARGET::"
INSTALL_PREFIX = "GEAK_INSTALLED::"
_INSTALLED = {}
_TLS = threading.local()
_PROFILE = {
    "lock": threading.Lock(),
    "active": False,
    "done": False,
    "owner": None,
    "profiler": None,
    "active_calls": 0,
    "root_calls": 0,
    "out": "",
    "trace_index": 0,
    "atexit_registered": False,
}


def _rank():
    for key in ("RANK", "LOCAL_RANK", "TP_RANK", "SLURM_PROCID"):
        value = os.environ.get(key)
        if value is not None:
            return str(value)
    return "unknown"


def _trace_path(trace_index):
    template = os.environ.get("GEAK_SELECTION_TRACE", "").strip()
    if not template:
        return ""
    pid = os.getpid()
    if "{pid}" in template or "{rank}" in template:
        path = template.format(pid=pid, rank=_rank())
    else:
        root, ext = os.path.splitext(template)
        path = f"{root}.pid-{pid}.rank-{_rank()}{ext or '.json'}"
    if os.environ.get("GEAK_SELECTION_TRACE_UNIQUE", "1") == "0":
        return template
    root, ext = os.path.splitext(path)
    return f"{root}.call-{trace_index}{ext or '.json'}"


def _record_install_markers():
    """Put proof of successful installation in every process-local trace."""
    import torch
    for target in sorted(_INSTALLED):
        with torch.profiler.record_function(INSTALL_PREFIX + target):
            pass


def _start_profile():
    """Start the next bounded process-local root-call profile."""
    with _PROFILE["lock"]:
        if _PROFILE["active"] or _PROFILE["done"]:
            return False
        budget = max(1, int(os.environ.get("GEAK_SELECTION_PROFILE_CALLS", "32")))
        if _PROFILE["trace_index"] >= budget:
            _PROFILE["done"] = True
            return False
        out = _trace_path(_PROFILE["trace_index"] + 1)
        if not out:
            return False
        ident = threading.get_ident()
        try:
            import torch
            activities = [torch.profiler.ProfilerActivity.CPU]
            if hasattr(torch.profiler.ProfilerActivity, "CUDA"):
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            profiler = torch.profiler.profile(activities=activities)
            profiler.__enter__()
            _PROFILE.update(active=True, owner=ident, profiler=profiler, out=out,
                            active_calls=0, root_calls=0)
            _record_install_markers()
            if not _PROFILE["atexit_registered"]:
                atexit.register(_finish_profile)
                _PROFILE["atexit_registered"] = True
            return True
        except Exception as exc:
            _PROFILE["done"] = True
            sys.stderr.write(f"[seam_trace] profiler start failed: {exc!r}\n")
            return False


def _finish_profile():
    """Stop and atomically export this process's trace once."""
    with _PROFILE["lock"]:
        if not _PROFILE["active"] or _PROFILE["done"]:
            return
        profiler = _PROFILE.get("profiler")
        out = _PROFILE.get("out")
        _PROFILE["trace_index"] += 1
        budget = max(1, int(os.environ.get("GEAK_SELECTION_PROFILE_CALLS", "32")))
        _PROFILE.update(active=False, done=_PROFILE["trace_index"] >= budget,
                        owner=None, profiler=None)
        try:
            profiler.__exit__(None, None, None)
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            root, ext = os.path.splitext(out)
            tmp = f"{root}.tmp-{os.getpid()}{ext}"
            profiler.export_chrome_trace(tmp)
            os.replace(tmp, out)
            sys.stderr.write(f"[seam_trace] selection trace -> {out}\n")
        except Exception as exc:
            sys.stderr.write(f"[seam_trace] profiler export failed: {exc!r}\n")


def _wrappable(value):
    """Only replace callables whose identity/protocol a Python wrapper preserves."""
    if isinstance(value, (types.FunctionType, types.MethodType, functools.partial)):
        return True
    if isinstance(value, (types.BuiltinFunctionType, types.BuiltinMethodType)):
        return False
    if any(hasattr(value, attr)
           for attr in ("fn", "cache", "warmup", "run", "__torch_dispatch__")):
        return False
    return callable(value) and type(value).__module__ != "builtins"


def _enter_call():
    _start_profile()
    depth = getattr(_TLS, "depth", 0)
    _TLS.depth = depth + 1
    with _PROFILE["lock"]:
        if _PROFILE["active"]:
            _PROFILE["active_calls"] += 1
    return depth == 0


def _leave_call(root_call):
    _TLS.depth = max(0, getattr(_TLS, "depth", 1) - 1)
    finish = False
    with _PROFILE["lock"]:
        if _PROFILE["active"]:
            _PROFILE["active_calls"] = max(0, _PROFILE["active_calls"] - 1)
            if root_call:
                _PROFILE["root_calls"] += 1
            finish = root_call and _PROFILE["active_calls"] == 0
    if finish:
        _finish_profile()


def install(target):
    """Wrap one module:attr with a record_function marker; idempotent per target."""
    if target in _INSTALLED:
        return
    module_name, attr = target.split(":", 1)
    module = importlib.import_module(module_name)
    original = getattr(module, attr)
    if not _wrappable(original):
        raise RuntimeError(f"cannot safely mark non-Python callable {target}")

    @functools.wraps(original)
    def marked(*args, **kwargs):
        root_call = _enter_call()
        try:
            import torch
            record_function = torch.profiler.record_function
        except Exception:
            record_function = None
        try:
            if record_function is None:
                return original(*args, **kwargs)
            with record_function(MARKER_PREFIX + target):
                return original(*args, **kwargs)
        finally:
            _leave_call(root_call)

    try:
        marked.__signature__ = inspect.signature(original)
    except (TypeError, ValueError):
        pass
    _INSTALLED[target] = original
    setattr(module, attr, marked)
    sys.stderr.write(f"[seam_trace] marked {target}\n")
