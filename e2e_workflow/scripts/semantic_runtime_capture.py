#!/usr/bin/env python3
"""GEAK Semantics 1.2 runtime metadata and profiler-marker injection.

This module is copied into the serving runtime and installed after ModelRunner
loads the model.  It records metadata only and wraps each selected module call
in a profiler record_function marker so launched kernels can be proven to be
contained by the wrapper that supplied the Shape.
"""
import json
import functools
import importlib
import os
import re
import sys
import threading


_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_TRUE = ("1", "true", "True", "TRUE", "yes", "on")
_LOGGER = None
_INSTALLED_CLASSES = set()
_PATCHED_CALLABLES = set()


def _flag(name, default="0"):
    return os.environ.get(name, default) in _TRUE


def _csv_int(name):
    value = os.environ.get(name, "")
    return set(int(item) for item in value.split(",") if item.strip())


def _csv_upper(name):
    return set(
        item.strip().upper()
        for item in os.environ.get(name, "").split(",")
        if item.strip())


def _canonical_phase(value):
    value = str(value or "").upper()
    if value in ("EXTEND", "PREFILL", "PROMPT"):
        return "PREFILL"
    if value in ("DECODE", "GENERATION"):
        return "DECODE"
    return value


def _profiler_active():
    """Return whether record_function ranges can reach the active trace."""
    try:
        import torch
        probe = getattr(torch.autograd, "_profiler_enabled", None)
        return bool(probe and probe())
    except Exception:
        return False


def _phase_of(forward_batch):
    try:
        mode = forward_batch.forward_mode
    except Exception:
        return "UNKNOWN", -1, -1
    phase = "UNKNOWN"
    for method, candidate in (
            ("is_decode", "DECODE"),
            ("is_target_verify", "TARGET_VERIFY"),
            ("is_extend", "EXTEND"),
            ("is_prefill", "PREFILL"),
            ("is_draft_extend", "DRAFT_EXTEND")):
        try:
            if getattr(mode, method, lambda: False)():
                phase = candidate
                break
        except Exception:
            pass
    if phase == "UNKNOWN":
        phase = str(getattr(mode, "name", mode)).upper()
    batch_size = -1
    for name in ("batch_size", "bs"):
        value = getattr(forward_batch, name, None)
        if isinstance(value, int):
            batch_size = value
            break
    input_tokens = -1
    try:
        input_ids = getattr(forward_batch, "input_ids", None)
        if input_ids is not None:
            input_tokens = int(input_ids.numel())
    except Exception:
        pass
    if batch_size == -1:
        try:
            batch_size = int(forward_batch.seq_lens.shape[0])
        except Exception:
            pass
    return phase, batch_size, input_tokens


def _layer_id(path):
    match = _LAYER_RE.search(path or "")
    return int(match.group(1)) if match else -1


def _op_path(path):
    if path.startswith("layers."):
        return "model." + path
    return path


def _metadata(value, aliases=None):
    import torch
    aliases = aliases if aliases is not None else {}
    if isinstance(value, torch.Tensor):
        identity = id(value)
        if identity not in aliases:
            aliases[identity] = "tensor-%d" % len(aliases)
        try:
            stride = list(value.stride())
        except Exception:
            stride = None
        return {
            "kind": "tensor",
            "alias_id": aliases[identity],
            "shape": list(value.shape),
            "dtype": str(value.dtype).replace("torch.", ""),
            "device": str(value.device),
            "stride": stride,
            "contiguous": bool(value.is_contiguous()),
            "requires_grad": bool(value.requires_grad),
        }
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [
            _metadata(item, aliases) for item in value]}
    if isinstance(value, list):
        return {"kind": "list", "items": [
            _metadata(item, aliases) for item in value]}
    if isinstance(value, dict):
        return {"kind": "dict", "items": {
            str(key): _metadata(item, aliases)
            for key, item in value.items()}}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return {
            "kind": "scalar", "type": type(value).__name__,
            "value": value,
        }
    return {
        "kind": "object",
        "type": "%s.%s" % (
            type(value).__module__, type(value).__name__),
    }


class SemanticRuntimeLogger(object):
    def __init__(self):
        self.enabled = _flag("GEAK_SEMANTICS_CAPTURE")
        self.path = os.environ.get("GEAK_SEMANTICS_SHAPE_LOG", "")
        self.rank = int(os.environ.get("GEAK_SEMANTICS_RANK", "0"))
        self.layers = _csv_int("GEAK_SEMANTICS_LAYERS")
        self.phases = _csv_upper("GEAK_SEMANTICS_PHASES")
        self.require_profiler = _flag(
            "GEAK_SEMANTICS_REQUIRE_PROFILER")
        self._profile_seen = False
        self.max_forwards = int(os.environ.get(
            "GEAK_SEMANTICS_FORWARDS_PER_BUCKET", "1"))
        self._context = {
            "phase": "UNKNOWN", "batch_size": -1, "input_tokens": -1}
        self._bucket_forwards = {}
        self._next_id = 0
        self._rank = None
        self._lock = threading.Lock()
        self._stacks = threading.local()
        self._fh = None
        if self.enabled and self.path:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)),
                        exist_ok=True)
            self._fh = open(self.path, "a", buffering=1)

    def active(self):
        return self._fh is not None and self._rank_ok()

    def _rank_ok(self):
        if self._rank is None:
            try:
                import torch.distributed as dist
                self._rank = (
                    dist.get_rank()
                    if dist.is_available() and dist.is_initialized() else 0)
            except Exception:
                self._rank = 0
        return self._rank == self.rank

    def set_context(self, phase, batch_size, input_tokens):
        self._context = {
            "phase": phase,
            "batch_size": batch_size,
            "input_tokens": input_tokens,
        }

    def mark_forward(self):
        key = (
            self._context["phase"], self._context["batch_size"],
            self._context["input_tokens"])
        self._bucket_forwards[key] = self._bucket_forwards.get(key, 0) + 1

    def _allowed(self, layer_id):
        if not self.active():
            return False
        # Benchmark warmups run before torch.profiler starts.  Consuming the
        # one-forward bucket there records shape JSON but leaves no matching
        # trace marker, which used to remove every steady decode bucket.
        #
        # Profiler-enabled state can be thread-local: prefill may observe it
        # while the subsequent decode worker does not.  Latch the first active
        # observation for the rest of this benchmark capture.
        if self.require_profiler and not self._profile_seen:
            if not _profiler_active():
                return False
            self._profile_seen = True
        if layer_id < 0 or (self.layers and layer_id not in self.layers):
            return False
        if (self.phases
                and _canonical_phase(self._context["phase"])
                not in {_canonical_phase(item) for item in self.phases}):
            return False
        key = (
            self._context["phase"], self._context["batch_size"],
            self._context["input_tokens"])
        return self._bucket_forwards.get(key, 0) < self.max_forwards

    def _stack(self):
        if not hasattr(self._stacks, "value"):
            self._stacks.value = {}
        return self._stacks.value

    def _active_modules(self):
        if not hasattr(self._stacks, "active_modules"):
            self._stacks.active_modules = []
        return self._stacks.active_modules

    def begin(self, module, layer_id, op_path):
        if not self._allowed(layer_id):
            return
        with self._lock:
            self._next_id += 1
            op_id = "geak-op-%08d" % self._next_id
        context = dict(self._context)
        marker = (
            "GEAK_SEMANTICS|op=%s|phase=%s|bs=%s|toks=%s|layer=%s|path=%s"
            % (op_id, context["phase"], context["batch_size"],
               context["input_tokens"], layer_id, op_path))
        record = None
        try:
            import torch
            factory = getattr(
                getattr(torch, "profiler", None), "record_function", None)
            if factory is None:
                factory = torch.autograd.profiler.record_function
            record = factory(marker)
            record.__enter__()
        except Exception:
            record = None
        entry = {
            "op_id": op_id,
            "marker": marker,
            "record": record,
            "context": context,
            "layer_id": layer_id,
            "op_path": op_path,
        }
        self._stack().setdefault(id(module), []).append(entry)
        self._active_modules().append(entry)

    def end(self, module, layer_id, op_name, op_type, op_path,
            args, kwargs, output):
        entries = self._stack().get(id(module), [])
        if not entries:
            return output
        entry = entries.pop()
        active = self._active_modules()
        if active and active[-1] is entry:
            active.pop()
        elif entry in active:
            active.remove(entry)
        if entry["record"] is not None:
            try:
                entry["record"].__exit__(None, None, None)
            except Exception:
                pass
        aliases = {}
        parameters = {}
        try:
            for name, value in module.named_parameters(recurse=False):
                parameters[name] = value
            for name, value in module.named_buffers(recurse=False):
                parameters[name] = value
        except Exception:
            pass
        payload = {
            "schema": "geak.semantics_runtime.v2",
            "op_instance_id": entry["op_id"],
            "marker": entry["marker"],
            "rank": self.rank,
            "layer_id": layer_id,
            "phase": entry["context"]["phase"].lower(),
            "batch_size": entry["context"]["batch_size"],
            "input_tokens": entry["context"]["input_tokens"],
            "op_name": op_name,
            "op_type": op_type,
            "op_path": op_path,
            "mapping_cardinality": "1:N",
            "evidence_level": "runtime_marker_containment",
            "inputs": _metadata(args, aliases),
            "kwargs": _metadata(kwargs, aliases),
            "parameters": _metadata(parameters, aliases),
            "output": _metadata(output, aliases),
        }
        with self._lock:
            self._fh.write(json.dumps(
                payload, sort_keys=True, separators=(",", ":")) + "\n")
        return output

    def begin_callable(self, target):
        active = self._active_modules()
        if not active:
            return None
        parent = active[-1]
        layer_id = parent["layer_id"]
        if not self._allowed(layer_id):
            return None
        with self._lock:
            self._next_id += 1
            op_id = "geak-call-%08d" % self._next_id
        context = dict(self._context)
        op_path = "%s::launcher:%s" % (parent["op_path"], target)
        marker = (
            "GEAK_SEMANTICS|op=%s|phase=%s|bs=%s|toks=%s|layer=%s|path=%s"
            % (op_id, context["phase"], context["batch_size"],
               context["input_tokens"], layer_id, op_path))
        record = None
        try:
            import torch
            factory = getattr(
                getattr(torch, "profiler", None), "record_function", None)
            if factory is None:
                factory = torch.autograd.profiler.record_function
            record = factory(marker)
            record.__enter__()
        except Exception:
            record = None
        return {
            "op_id": op_id,
            "marker": marker,
            "record": record,
            "context": context,
            "layer_id": layer_id,
            "op_path": op_path,
            "target": target,
        }

    def end_callable(self, entry, args, kwargs, output):
        if entry is None:
            return output
        if entry["record"] is not None:
            try:
                entry["record"].__exit__(None, None, None)
            except Exception:
                pass
        aliases = {}
        payload = {
            "schema": "geak.semantics_runtime.v2",
            "op_instance_id": entry["op_id"],
            "marker": entry["marker"],
            "rank": self.rank,
            "layer_id": entry["layer_id"],
            "phase": entry["context"]["phase"].lower(),
            "batch_size": entry["context"]["batch_size"],
            "input_tokens": entry["context"]["input_tokens"],
            "op_name": entry["target"].split(":")[-1],
            "op_type": "targeted_python_launcher",
            "op_path": entry["op_path"],
            "mapping_cardinality": "probe_required",
            "evidence_level": "targeted_launcher_probe",
            "inputs": _metadata(args, aliases),
            "kwargs": _metadata(kwargs, aliases),
            "parameters": _metadata({}, aliases),
            "output": _metadata(output, aliases),
        }
        with self._lock:
            self._fh.write(json.dumps(
                payload, sort_keys=True, separators=(",", ":")) + "\n")
        return output


def get_logger():
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = SemanticRuntimeLogger()
    return _LOGGER


def _callable_targets():
    return [
        item.strip()
        for item in os.environ.get(
            "GEAK_SEMANTICS_CALLABLE_TARGETS", "").split(",")
        if item.strip()]


def _install_callable_probes():
    logger = get_logger()
    for target in _callable_targets():
        if target in _PATCHED_CALLABLES:
            continue
        module_name, separator, attr_name = target.partition(":")
        if not separator:
            raise RuntimeError(
                "invalid GEAK callable target %r; expected module:attr" %
                target)
        module = importlib.import_module(module_name)
        original = getattr(module, attr_name)
        if not callable(original):
            raise RuntimeError(
                "GEAK callable target is not callable: %s" % target)

        @functools.wraps(original)
        def wrapped(*args, __original=original, __target=target, **kwargs):
            entry = logger.begin_callable(__target)
            output = None
            try:
                output = __original(*args, **kwargs)
                return output
            finally:
                logger.end_callable(entry, args, kwargs, output)

        setattr(module, attr_name, wrapped)
        _PATCHED_CALLABLES.add(target)
        sys.stderr.write(
            "[GEAK_SEMANTICS] targeted launcher wrapped %s\n" % target)


def _register_hooks(model):
    logger = get_logger()
    if not logger.active():
        return
    count = 0
    for raw_path, module in model.named_modules():
        if not raw_path:
            continue
        path = _op_path(raw_path)
        layer = _layer_id(path)
        if logger.layers and layer not in logger.layers:
            continue
        children = list(module.children())
        class_name = module.__class__.__name__
        is_key = any(token in class_name for token in (
            "Attention", "MoE", "Moe", "MLP", "DecoderLayer",
            "RMSNorm", "LayerNorm", "Linear", "Expert"))
        if children and not is_key:
            continue

        def make_pre(layer_id, op_path):
            def pre_hook(mod, args, kwargs=None):
                logger.begin(mod, layer_id, op_path)
            return pre_hook

        def make_post(layer_id, op_name, op_type, op_path):
            def post_hook(mod, args, kwargs, output):
                return logger.end(
                    mod, layer_id, op_name, op_type, op_path,
                    args, kwargs, output)
            return post_hook

        try:
            module.register_forward_pre_hook(
                make_pre(layer, path), with_kwargs=True)
            module.register_forward_hook(
                make_post(layer, path.split(".")[-1], class_name, path),
                with_kwargs=True)
            count += 1
        except TypeError:
            # GEAK Semantics 1.2 requires kwargs-capable hooks for reliable
            # tensor roles; fail explicitly instead of silently mislabelling.
            raise RuntimeError(
                "runtime torch lacks kwargs-capable forward hooks")
    sys.stderr.write(
        "[GEAK_SEMANTICS] registered %d marker+metadata hooks\n" % count)


def install_on_model(model):
    logger = get_logger()
    if not logger.active():
        return model
    model_class = model.__class__
    if model_class not in _INSTALLED_CLASSES:
        original_forward = model_class.forward

        def wrapped_forward(self, *args, **kwargs):
            forward_batch = args[2] if len(args) >= 3 else None
            forward_batch = kwargs.get("forward_batch", forward_batch)
            if forward_batch is not None:
                logger.set_context(*_phase_of(forward_batch))
            if not getattr(self, "_geak_semantics_hooked", False):
                _register_hooks(self)
                self._geak_semantics_hooked = True
            result = original_forward(self, *args, **kwargs)
            if forward_batch is not None:
                logger.mark_forward()
            return result

        model_class.forward = wrapped_forward
        _INSTALLED_CLASSES.add(model_class)
    elif not getattr(model, "_geak_semantics_hooked", False):
        _register_hooks(model)
        model._geak_semantics_hooked = True
    _install_callable_probes()
    sys.stderr.write(
        "[GEAK_SEMANTICS] wrapped %s.forward\n" % model_class.__name__)
    return model
